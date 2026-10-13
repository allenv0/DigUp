import CStbMLX
import DigUpKit
import Foundation
import LlamaRuntime
import MLX

/// EmbeddingGemma 2 on MLX, in this process: the same model files as llama.cpp (`ModelFiles`), converted once
/// into bf16 weights (`ModelConvert`), run by `EmbeddingGemma2Text` (and the image and audio encoders beside
/// it). Mean-pooled, L2-normalized out, like `LlamaEmbedder`.
///
/// One thread at a time (MLX streams are per-thread); `close()` frees the model and the GPU's cache. The
/// parity gates this engine must pass are the same ones llama.cpp passes: `MLXReferenceVectorTests` against
/// the one reference, and the evals.
///
/// Execution is deliberately uncompiled: `mx.compile` with shape bucketing was evaluated for Phase 4 and
/// deferred — the encoder workload is batch-16 text plus single media inputs, where graph compilation
/// saves little and risks parity drift at the 0.999 gate. Batching matches the reference's instead
/// (16 sequences, 2048 real tokens), and media batches clear the GPU cache on return.
public final class MLXEmbedder: Embedder {
    public static let budgets = [70, 140, 280, 560, 1120]

    /// Which vectors this pipeline makes. The namespace is shared with llama.cpp (`VectorCompat`), which the
    /// parity tests earn; bump only with a new reference, as llama.cpp's version is bumped.
    public static let vectorVersion = 1

    /// The mlx-swift version this was built against (the exact pin in Package.swift), for logs and status.
    public static let mlxVersion = "0.31.6"

    /// What every MLXEmbedder reports, known without loading it.
    public static let modelInfo = WorkerInfo(model: "google/embeddinggemma-2", revision: ModelFiles.revision,
                                             runtime: "mlx-vectors\(vectorVersion)",
                                             dtype: "text-bf16+mmproj-bf16", dim: 768, textOnly: false,
                                             loadSeconds: nil, vectorCompat: VectorCompat.embeddingGemma2)

    public private(set) var info: WorkerInfo
    private let models: MLXModels
    private let textOnly: Bool
    private let config: EmbeddingGemma2Text.Config
    private let model: EmbeddingGemma2Text
    private let vision: VisionTower?
    private let audio: AudioTower?
    private let vocab: GGUFVocab
    /// The text GGUF, kept for its metadata (the config).
    private let textGGUF: GGUFFile

    /// Batching matches the reference's: 16 sequences, 2048 real tokens, and a padded-size guard so the
    /// attention mask of a batch with one long text and many short ones stays sane.
    static let maxSequences = 16
    static let maxRealTokens = 2048
    static let maxPaddedTokens = 4096

    /// `textOnly`: just the text tower (queries). `log`: where conversion and load notes go.
    public init(models: MLXModels, textOnly: Bool, log: URL? = nil) throws {
        let started = Date()
        Self.note(log, "mlx \(Self.mlxVersion): loading (\(textOnly ? "text-only" : "full"))")
        self.models = models
        self.textOnly = textOnly
        if !models.upToDate {
            try ModelConvert.convert(models: models, log: { Self.note(log, $0) })
        }
        guard let textGGUF = try? GGUFFile(path: models.folder.appendingPathComponent(ModelFiles.text.name).path) else {
            throw WorkerError.failed("can't read the text model's metadata")
        }
        self.textGGUF = textGGUF
        config = try EmbeddingGemma2Text.Config(file: textGGUF)
        model = EmbeddingGemma2Text(config)
        let weights = try MLX.loadArrays(url: models.text)
        do {
            try model.load(weights)
        } catch {
            throw WorkerError.failed("the MLX weights don't fit the model: \(error)")
        }
        vocab = try GGUFVocab(path: models.folder.appendingPathComponent(ModelFiles.text.name).path, log: log)
        if textOnly {
            vision = nil
            audio = nil
        } else {
            guard models.media != nil else {
                throw WorkerError.notFound("the media encoder isn't converted yet (no mmproj.safetensors)")
            }
            let mediaFile = try GGUFFile(path: models.folder.appendingPathComponent(ModelFiles.media.name).path)
            let tower = try VisionTower(file: mediaFile)
            let audioTower = try AudioTower(file: mediaFile)
            let mediaWeights = try MLX.loadArrays(url: models.media!)
            do {
                try tower.load(mediaWeights)
                try audioTower.load(mediaWeights)
            } catch {
                throw WorkerError.failed("the MLX media weights don't fit the tower: \(error)")
            }
            vision = tower
            audio = audioTower
        }
        // The query encoder keeps a smaller GPU cache than indexing, mirroring the Python worker's policy.
        MLX.GPU.set(cacheLimit: (textOnly ? 256 : 512) << 20)

        let known = Self.modelInfo
        info = WorkerInfo(model: known.model, revision: known.revision, runtime: known.runtime, dtype: known.dtype,
                          dim: config.embeddingOut, textOnly: textOnly, loadSeconds: nil,
                          vectorCompat: known.vectorCompat)
        do {
            for _ in 0..<2 { _ = try embedTexts([Prompts.query("warm up")]) }   // pay the pipeline build now
        } catch {
            close()
            throw error
        }
        let seconds = (Date().timeIntervalSince(started) * 100).rounded() / 100
        info = WorkerInfo(model: info.model, revision: info.revision, runtime: info.runtime, dtype: info.dtype,
                          dim: info.dim, textOnly: textOnly, loadSeconds: seconds, vectorCompat: known.vectorCompat)
        Self.note(log, "mlx: ready in \(seconds) s (\(textOnly ? "text-only" : "full"), vectors \(Self.vectorVersion))")
    }

    deinit { close() }

    public func close() {
        MLX.GPU.clearCache()
    }

    // MARK: Embedder

    public func embedTexts(_ texts: [String]) throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        // The GGUF's tokenizer frames by itself (add_special, exactly what LlamaEmbedder's ids are); no
        // second framing on top.
        let encoded = texts.map { vocab.tokenize($0, maxLength: Framing.maxLength) }
        var vectors: [[Float]] = []
        var batch: [[Int]] = [], real = 0
        func run() throws {
            guard !batch.isEmpty else { return }
            vectors += try embed(batch: batch)
            batch = []
            real = 0
        }
        for tokens in encoded {
            if batch.count == Self.maxSequences || real + tokens.count > Self.maxRealTokens
                || max(tokens.count, batch.map(\.count).max() ?? 0) * (batch.count + 1) > Self.maxPaddedTokens {
                try run()
            }
            batch.append(tokens)
            real += tokens.count
        }
        try run()
        return vectors
    }

    public func embedImages(_ paths: [String], budget: Int) throws -> [Embedding] {
        guard let vision else { throw WorkerError.failed("this embedder was started text-only") }
        guard Self.budgets.contains(budget) else { throw WorkerError.failed("budget must be one of \(Self.budgets)") }
        defer { MLX.GPU.clearCache() }   // media batches allocate transient attention maps per input
        return try paths.map { path in
            do {
                return Embedding(vector: try embedImage(path: path, budget: budget), error: nil)
            } catch {
                return Embedding(vector: nil, error: String("\(error)".prefix(300)))
            }
        }
    }

    public func embedAudio(_ paths: [String]) throws -> [Embedding] {
        guard audio != nil else { throw WorkerError.failed("this embedder was started text-only") }
        defer { MLX.GPU.clearCache() }   // conformer attention maps are the tower's peak transient
        return try paths.map { path in
            do {
                return Embedding(vector: try embedAudio(path: path), error: nil)
            } catch {
                return Embedding(vector: nil, error: String("\(error)".prefix(300)))
            }
        }
    }

    // MARK: Images

    /// One image: read, resize for `budget` (the shared `ImageSizing`, the same pixels mtmd gets), through
    /// the vision tower, then through the text tower framed `<bos> <|image> soft tokens <image|> <eos>`.
    private func embedImage(path: String, budget: Int) throws -> [Float] {
        let pixels = try Self.readRGB(path: path)
        let resized = try Self.resized(pixels, budget: budget)
        let rows = vision!(MLXArray(resized.values, [resized.height, resized.width, 3]))[0]   // [tokens, out]
        let count = rows.dim(0)

        // The framed sequence: ids for the markers, the tower's rows at the soft-token positions (which sit
        // contiguously after <bos> and the begin marker).
        let ids = Framing.mediaIds(begin: Framing.beginImage, end: Framing.endImage, count: count)
        let tokenRows = model.tokEmbd[MLXArray(ids.map { Int32($0) })]
            * Float(Foundation.sqrt(Double(config.embedding)))
        let embeddings = MLX.concatenated([tokenRows[0..<2], rows, tokenRows[(2 + count)...]], axis: 0)
        let result = model(embeddings.expandedDimensions(axis: 0), lengths: [ids.count])
        return Self.pooled(result[0])
    }

    /// Decodes an image file into packed RGB, 0-255 — through the same stb_image decoder mtmd uses, so
    /// the vision tower sees the pixels llama.cpp sees (JPEG decoders differ at +-1-2 LSB; the reference
    /// inputs are JPEG).
    static func readRGB(path: String) throws -> (values: [UInt8], width: Int, height: Int) {
        var width: Int32 = 0, height: Int32 = 0
        guard let base = stb_image_load_rgb(path, &width, &height) else {
            throw WorkerError.failed("can't read \(path)")
        }
        defer { free(base) }
        let count = Int(width) * Int(height) * 3
        let bytes = UnsafeBufferPointer(start: base, count: count)
        return ([UInt8](bytes), Int(width), Int(height))
    }

    /// The budget's target size (`ImageSizing`), and the resized pixels when they differ.
    static func resized(_ pixels: (values: [UInt8], width: Int, height: Int), budget: Int)
        -> (values: [UInt8], width: Int, height: Int) {
        guard let target = ImageSizing.targetSize(height: pixels.height, width: pixels.width, budget: budget),
              target != (pixels.height, pixels.width) else { return pixels }
        let values = pixels.values.withUnsafeBufferPointer {
            ImageSizing.resizeBicubic($0.baseAddress!, width: pixels.width, height: pixels.height,
                                      toWidth: target.width, toHeight: target.height)
        }
        return (values, target.width, target.height)
    }

    // MARK: Audio

    /// One WAV window: decode, log-mel, conformer, then the text tower framed
    /// `<bos> <|audio> soft tokens <audio|> <eos>`. The indexer hands windows of at most 30 s over.
    private func embedAudio(path: String) throws -> [Float] {
        let samples = try AudioFile.readMono16K(path: path)
        let window = Array(samples.prefix(30 * MelFrontend.sampleRate))
        let rows = audio!(try MelFrontend()(window))[0]   // [tokens, out]
        let count = rows.dim(0)
        let ids = Framing.mediaIds(begin: Framing.beginAudio, end: Framing.endAudio, count: count)
        let tokenRows = model.tokEmbd[MLXArray(ids.map { Int32($0) })]
            * Float(Foundation.sqrt(Double(config.embedding)))
        let embeddings = MLX.concatenated([tokenRows[0..<2], rows, tokenRows[(2 + count)...]], axis: 0)
        let result = model(embeddings.expandedDimensions(axis: 0), lengths: [ids.count])
        return Self.pooled(result[0])
    }

    /// Mean-pools one sequence and L2-normalizes it, like the text batches above.
    static func pooled(_ sequence: MLXArray) -> [Float] {
        let pooled = sequence.mean(axis: 0)
        let norm = MLX.sqrt((pooled * pooled).sum()).item(Float.self)
        return (norm > 0 ? pooled / norm : pooled).asArray(Float.self)
    }

    /// One batch through the tower: token rows scaled like the graph's input, mean-pooled and L2-normalized.
    private func embed(batch: [[Int]]) throws -> [[Float]] {
        let tokens = batch.map(\.count).max() ?? 0
        let lengths = batch.map(\.count)
        var padded = batch
        for index in 0..<padded.count { padded[index] += Array(repeating: 0, count: tokens - padded[index].count) }
        let flat = padded.flatMap { $0 }
        let ids = MLXArray(flat, [batch.count, tokens])   // [Int] makes an int32 array, as ids go in
        let rows = model.tokEmbd[ids] * Float(Foundation.sqrt(Double(config.embedding)))
        let result = model(rows, lengths: lengths).asType(.float32)   // [batch, tokens, dim]

        var pooled: [MLXArray] = []
        for (index, length) in lengths.enumerated() {
            pooled.append(result[index, 0..<length].mean(axis: 0))
        }
        let matrix = MLX.stacked(pooled, axis: 0)                      // [batch, dim]
        let squares = (matrix * matrix).sum(axis: 1)                    // [batch]
        let scale = MLX.where(squares .> 0, 1 / MLX.sqrt(squares), MLXArray(0))
        let normalized = matrix * scale.reshaped([batch.count, 1])
        return normalized.asArray(Float.self).chunked(into: config.embeddingOut)
    }

    /// Notes into the helper's log, or stderr.
    static func note(_ log: URL?, _ line: String) {
        guard let log else {
            FileHandle.standardError.write(Data("[mlx worker] \(line)\n".utf8))
            return
        }
        FileManager.default.createFile(atPath: log.path, contents: nil)
        if let handle = try? FileHandle(forWritingTo: log) {
            try? handle.seekToEnd()
            try? handle.write(contentsOf: Data("[mlx worker] \(line)\n".utf8))
            try? handle.close()
        }
    }
}

extension Array {
    /// Splits a flat vector into rows.
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map { Array(self[$0..<Swift.min($0 + size, count)]) }
    }
}
