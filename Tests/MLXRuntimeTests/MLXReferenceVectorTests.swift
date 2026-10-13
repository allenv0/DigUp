import DigUpKit
import Foundation
import LlamaRuntime
import Testing
@testable import MLXRuntime

/// The vectors MLX makes for the reference inputs, against the reference llama.cpp made (one file, one bar:
/// cosine >= 0.999 for every input, as `ReferenceVectorTests` holds llama.cpp to). This is the evidence
/// behind `VectorCompat.embeddingGemma2` — the two engines claiming the same namespace.
///
/// The texts are the reference's (`ReferenceVectorTests`), prompts and all; the images and audio join when
/// the MLX engine's media encoders are wired (the inputs sit in the same folder). Needs the model files;
/// skipped without them.
@Suite struct MLXReferenceVectorTests {
    static let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("../LlamaRuntimeTests/Reference").standardizedFileURL
    static let modelsThere = (try? MLXModels.locate())?.upToDate == true

    /// Same bar, same meaning as llama.cpp's: less would be a change people would see in their results.
    static let agreement: Float = 0.999

    static let texts: [(name: String, text: String)] = [
        ("query", Prompts.query("payment declined error")),
        ("query-bn", Prompts.query("ইলিশ মাছ রান্না")),
        ("query-ar", Prompts.query("متى تغلق المكتبة")),
        ("document", Prompts.document(title: "Tokyo itinerary", text: "Flight to Tokyo Haneda on November 3. Then the "
                                    + "train to Kamakura to see the Great Buddha, and back for dinner in Shibuya.")),
        // About 1,800 tokens: past the sliding attention window, so the full-attention layers count as well.
        ("document-long", Prompts.document(title: nil, text: (1...60).map {
            "Note \($0): the review moved to room \($0 % 7 + 2) because the projector in the main hall failed again, "
                + "and the team agreed to send the slides the night before."
        }.joined(separator: " "))),
    ]
    /// The reference's images: a screenshot at 280 tokens, a photo at 140 (and 70, as a video frame).
    static let images: [(name: String, file: String, budget: Int)] =
        [("screen-280", "screen.png", 280), ("photo-140", "photo.jpg", 140), ("frame-70", "photo.jpg", 70)]

    struct Reference: Codable {
        let vectorVersion: Int
        let madeWith: String
        let vectors: [String: [Float]]

        enum CodingKeys: String, CodingKey {
            case vectorVersion = "vector_version", madeWith = "made_with", vectors
        }
    }

    @Test(.enabled(if: modelsThere)) func textsMatchTheReference() throws {
        let embedder = try MLXEmbedder(models: try MLXModels.locate(), textOnly: true, log: nil)
        defer { embedder.close() }
        // All five at once, as the indexer batches: the padded batch exercises the attention mask too.
        let made = try embedder.embedTexts(Self.texts.map(\.text))
        let reference = try #require(try? JSONDecoder().decode(Reference.self, from:
            Data(contentsOf: Self.folder.appendingPathComponent("vectors.json"))))
        for (input, vector) in zip(Self.texts, made) {
            guard let expected = reference.vectors[input.name] else {
                Issue.record("no reference for \(input.name)")
                continue
            }
            let cosine = zip(vector, expected).reduce(0) { $0 + $1.0 * $1.1 }
            #expect(cosine >= Self.agreement, "\(input.name): cosine \(cosine) with the reference")
        }
    }

    /// The image inputs of the reference set, at their budgets, through the vision tower.
    @Test(.enabled(if: modelsThere && MLXModels.complete)) func imagesMatchTheReference() throws {
        let embedder = try MLXEmbedder(models: try MLXModels.locate(), textOnly: false, log: nil)
        defer { embedder.close() }
        let reference = try #require(try? JSONDecoder().decode(Reference.self, from:
            Data(contentsOf: Self.folder.appendingPathComponent("vectors.json"))))
        for (name, file, budget) in Self.images {
            let embedding = try embedder.embedImages(
                [Self.folder.appendingPathComponent(file).path], budget: budget)[0]
            let vector = try #require(embedding.vector, "\(file): \(embedding.error ?? "no vector")")
            guard let expected = reference.vectors[name] else {
                Issue.record("no reference for \(name)")
                continue
            }
            let cosine = zip(vector, expected).reduce(0) { $0 + $1.0 * $1.1 }
            #expect(cosine >= Self.agreement, "\(name): cosine \(cosine) with the reference")
        }
    }

    /// The reference's audio: one 16 kHz WAV window through the audio tower.
    @Test(.enabled(if: modelsThere && MLXModels.complete)) func audioMatchesTheReference() throws {
        let embedder = try MLXEmbedder(models: try MLXModels.locate(), textOnly: false, log: nil)
        defer { embedder.close() }
        let embedding = try embedder.embedAudio([Self.folder.appendingPathComponent("speech.wav").path])[0]
        let vector = try #require(embedding.vector, "speech.wav: \(embedding.error ?? "no vector")")
        let reference = try #require(try? JSONDecoder().decode(Reference.self, from:
            Data(contentsOf: Self.folder.appendingPathComponent("vectors.json"))))
        let expected = try #require(reference.vectors["speech"])
        let cosine = zip(vector, expected).reduce(0) { $0 + $1.0 * $1.1 }
        #expect(cosine >= Self.agreement, "speech: cosine \(cosine) with the reference")
    }

    /// The media framing against mtmd itself: marker ids around one `-1` per soft-token row.
    @Test(.enabled(if: LlamaModels.complete)) func mediaFramingMatchesMtmd() throws {
        let llama = try LlamaEmbedder(models: try LlamaModels.locate(), textOnly: false, log: nil)
        defer { llama.close() }
        let ids = try llama.mediaTokenIDs(path: Self.folder.appendingPathComponent("photo.jpg").path, image: true,
                                          budget: 70)
        let expected = Framing.mediaIds(begin: Framing.beginImage, end: Framing.endImage,
                                        count: ids.filter { $0 == -1 }.count)
        #expect(ids.filter { $0 != -1 } == expected.filter { $0 != Framing.image })
        #expect(ids.filter { $0 == -1 }.count == expected.filter { $0 == Framing.image }.count)
        let audioIDs = try llama.mediaTokenIDs(path: Self.folder.appendingPathComponent("speech.wav").path,
                                               image: false, budget: 0)
        let expectedAudio = Framing.mediaIds(begin: Framing.beginAudio, end: Framing.endAudio,
                                             count: audioIDs.filter { $0 == -1 }.count)
        #expect(audioIDs.filter { $0 != -1 } == expectedAudio.filter { $0 != Framing.audio })
        #expect(audioIDs.filter { $0 == -1 }.count == expectedAudio.filter { $0 == Framing.audio }.count)
    }

    /// The ids both engines frame text with come from the same GGUF tokenizer (`GGUFVocab`), so they agree by
    /// construction — this pins the framing itself: `<bos> … <eos>`, cut keeping the end.
    @Test(.enabled(if: LlamaModels.complete)) func framingMatchesLlamaCpp() throws {
        let vocab = try GGUFVocab(path: try LlamaModels.locate().text.path, log: nil)
        let llama = try LlamaEmbedder(models: try LlamaModels.locate(), textOnly: true, log: nil)
        defer { llama.close() }
        for text in Self.texts.map(\.text) {
            let framed = vocab.tokenize(text, maxLength: Framing.maxLength)
            let llamaIds = llama.tokenize(text, maxLength: Framing.maxLength)
            #expect(framed == llamaIds, "framing differs for “\(text.prefix(40))”")
        }
        #expect(vocab.tokenize("a", maxLength: Framing.maxLength).count == 3)   // bos, a, eos
        #expect(Framing.frame((0..<Framing.maxLength + 10).map { _ in 77 }).count == Framing.maxLength)
    }
}
