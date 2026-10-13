import DigUpKit
import Foundation

/// Where EmbeddingGemma 2's MLX weights are.
///
/// The one verified download (`ModelFiles`: two Q8_0 GGUFs plus the tokenizer.json) serves both runtimes:
/// the first MLX run converts the GGUFs into bf16 safetensors beside them (`ModelConvert`), keyed by a
/// manifest that says which sources made it. Loading is then `loadArrays` on the cache, same as llama.cpp
/// memory-maps its GGUF. `LlamaModels` is the llama.cpp counterpart of this type.
public struct MLXModels: Sendable {
    /// What the cache holds and how it was made: the recipe names the weight format (`ModelConvert`), the
    /// version bumps when the conversion output changes meaning. Both are part of the cache folder's name,
    /// so an old cache is simply left behind (and `ModelConvert` cleans it up). Single-sourced in
    /// `DigUpKit.MLXCache` so the app can check conversion state without linking MLX.
    public static let recipe = MLXCache.recipe
    public static let cacheVersion = MLXCache.version
    public static var cacheName: String { "eg2-v\(cacheVersion)-\(recipe)" }

    /// The models folder these were found in (the GGUFs live beside the cache; `DIGUP_MODELS` wins, as for
    /// llama.cpp).
    public let folder: URL
    /// `<folder>/mlx/<cache>`, holding `text.safetensors`, `mmproj.safetensors` and `manifest.json`.
    public let cache: URL
    public let text: URL
    /// Nil while only the text part is converted.
    public let media: URL?
    /// The manifest matches what this build converts (source revision, recipe, converter version).
    public let upToDate: Bool

    /// The folders to look in, first match wins (`ModelFiles.modelFolders`).
    public static var folders: [URL] {
        ModelFiles.modelFolders
    }

    /// The first folder with the model files in it, whether converted already or still to convert.
    public static func locate() throws -> MLXModels {
        for folder in folders {
            let text = folder.appendingPathComponent(ModelFiles.text.name)
            guard FileManager.default.fileExists(atPath: text.path) else { continue }
            let cache = folder.appendingPathComponent("mlx").appendingPathComponent(cacheName)
            let convertedText = cache.appendingPathComponent("text.safetensors")
            let convertedMedia = cache.appendingPathComponent("mmproj.safetensors")
            let mediaSource = folder.appendingPathComponent(ModelFiles.media.name)
            let media = FileManager.default.fileExists(atPath: mediaSource.path) ? mediaSource : nil
            return MLXModels(folder: folder, cache: cache, text: convertedText,
                             media: FileManager.default.fileExists(atPath: convertedMedia.path) ? convertedMedia : nil,
                             upToDate: Self.manifestMatches(cache: cache, media: media))
        }
        throw WorkerError.notFound("no \(ModelFiles.text.name) in " + folders.map(\.path).joined(separator: ", ")
            + "; get it with: digup download-model")
    }

    /// A usable cache exists (or the sources to make one are there), so MLX can run.
    public static var available: Bool {
        (try? locate()) != nil
    }

    /// The full model is ready for indexing (text and media): converted, or there to convert.
    public static var complete: Bool {
        guard let models = try? locate() else { return false }
        return models.media != nil
    }

    /// The manifest at `cache`, when it is there and names this build's sources.
    static func manifestMatches(cache: URL, media: URL?) -> Bool {
        guard let data = try? Data(contentsOf: cache.appendingPathComponent("manifest.json")),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data) else { return false }
        guard manifest.version == cacheVersion, manifest.recipe == recipe,
              manifest.revision == ModelFiles.revision else { return false }
        if media != nil, manifest.mediaSHA256 == nil { return false }
        return true
    }

    /// What the manifest records: enough to know the cache was made from these sources, this way.
    public struct Manifest: Codable, Sendable {
        public var version: Int
        public var recipe: String
        public var revision: String
        public var textSHA256: String
        public var mediaSHA256: String?
        /// Seconds the conversion took, for the log.
        public var seconds: Double
    }
}
