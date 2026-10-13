import Foundation

/// The MLX weight cache's identity, shared by the converter (`MLXRuntime`) and the app (which checks
/// conversion state without linking MLX). Single-sourced here so Settings and `MLXModels` can't drift.
///
/// Cache folder: `<models>/mlx/eg2-v<version>-<recipe>/` with `text.safetensors`, `mmproj.safetensors`,
/// `manifest.json`. Version bumps when conversion output changes meaning; recipe names the weight format.
public enum MLXCache {
    public static let recipe = "bf16"
    public static let version = 9
    public static var folderName: String { "eg2-v\(version)-\(recipe)" }

    /// `<models>/mlx/<cache>` for a models folder.
    public static func cache(in models: URL) -> URL {
        models.appendingPathComponent("mlx").appendingPathComponent(folderName)
    }

    /// Conversion state for Settings and `status`: ready (manifest matches this build), convertible
    /// (GGUFs present, cache missing or stale), or missing (no GGUFs at all).
    public enum State: Sendable, Equatable {
        case ready(cache: URL)
        case convertible(models: URL)
        case missing
    }

    public static func state(modelsDirectory: URL) -> State {
        let models = modelsDirectory
        let text = models.appendingPathComponent(ModelFiles.text.name)
        guard FileManager.default.fileExists(atPath: text.path) else { return .missing }
        let cache = cache(in: models)
        let manifest = cache.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifest),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (json["version"] as? Int) == version,
              (json["recipe"] as? String) == recipe,
              (json["revision"] as? String) == ModelFiles.revision else {
            return .convertible(models: models)
        }
        // Full readiness needs the media file converted too when the GGUF is present.
        let mediaSource = models.appendingPathComponent(ModelFiles.media.name)
        let mediaConverted = cache.appendingPathComponent("mmproj.safetensors")
        if FileManager.default.fileExists(atPath: mediaSource.path),
           !FileManager.default.fileExists(atPath: mediaConverted.path) {
            return .convertible(models: models)
        }
        return .ready(cache: cache)
    }

    /// Bytes the converted cache occupies (0 when absent), for Settings and `status`.
    public static func bytes(modelsDirectory: URL) -> Int64 {
        let cache = cache(in: modelsDirectory)
        var total: Int64 = 0
        for name in ["text.safetensors", "mmproj.safetensors"] {
            let url = cache.appendingPathComponent(name)
            total += (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? 0
        }
        return total
    }
}

public extension EmbeddingBackend {
    /// What Settings shows: "Automatic (llama.cpp)", "llama.cpp", "MLX".
    var displayName: String {
        switch self {
        case .automatic: "Automatic (llama.cpp)"
        case .llama: "llama.cpp"
        case .mlx: "MLX"
        }
    }

    /// One line for Settings' footer: what automatic means today and what switching costs (nothing, by D3).
    static var automaticExplainer: String {
        "Automatic uses llama.cpp today. Switching engines never re-indexes: both make the same vectors (\(VectorCompat.embeddingGemma2)). If MLX can't start, DigUp quietly uses llama.cpp."
    }
}
