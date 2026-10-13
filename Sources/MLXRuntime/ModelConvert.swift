import DigUpKit
import Foundation
import MLX

/// Converts the verified Q8_0 GGUFs (`ModelFiles`) into the MLX engine's weights, once, into
/// `<models>/mlx/<cache>` beside them: `text.safetensors`, `mmproj.safetensors`, `manifest.json`.
///
/// Tensors keep their ggml names — the Swift model's modules are declared under those names, so nothing gets
/// renamed and nothing can be mismapped. Matrices become bf16 (the recipe the validated Python reference
/// runs; bf16's exponent range avoids the fp16 activation overflow `ModelFiles` warns about) — the
/// reference-vector gates hold the result to llama.cpp's, which computes the same Q8_0 values in f32. The
/// vision tower upcasts its weights to f32 in memory (`VisionTower.load`): its pooled rows have less
/// mean-pooling dilution than the text tower's. Weight names: linear weights keep ggml's, raw parameters
/// drop the `.weight` ggml appends.
public enum ModelConvert {
    public static let recipe = MLXModels.recipe

    /// Converts (or remakes with `reconvert`) the cache `models` names. `log` gets a line per step.
    public static func convert(models: MLXModels, reconvert: Bool = false, log: (String) -> Void) throws {
        let folder = models.folder
        let textGGUF = folder.appendingPathComponent(ModelFiles.text.name)
        let mediaGGUF = folder.appendingPathComponent(ModelFiles.media.name)
        guard FileManager.default.fileExists(atPath: textGGUF.path) else {
            throw WorkerError.notFound("\(ModelFiles.text.name) isn't in \(folder.path); digup download-model first")
        }
        if models.upToDate, !reconvert {
            log("the cache is up to date already (\(models.cache.lastPathComponent))")
            return
        }
        // One partial folder, renamed into place when everything is written: a half-converted cache never
        // looks done (its manifest would be missing), but it doesn't get mistaken for the real one either.
        let root = models.cache.deletingLastPathComponent()
        let partial = root.appendingPathComponent(models.cache.lastPathComponent + ".partial")
        try? FileManager.default.removeItem(at: partial)
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)

        let started = Date()
        log("reading \(ModelFiles.text.name)")
        let text = try GGUFFile(path: textGGUF.path)
        let textSHA = try ModelFiles.sha256(of: textGGUF)
        log("converting \(text.tensors.count) text tensors (Q8_0 -> bf16)")
        try convert(file: text, to: partial.appendingPathComponent("text.safetensors"), log: log)

        var mediaSHA: String?
        if FileManager.default.fileExists(atPath: mediaGGUF.path) {
            log("reading \(ModelFiles.media.name)")
            let media = try GGUFFile(path: mediaGGUF.path)
            mediaSHA = try ModelFiles.sha256(of: mediaGGUF)
            log("converting \(media.tensors.count) media tensors")
            try convert(file: media, to: partial.appendingPathComponent("mmproj.safetensors"), log: log)
        }

        let manifest = MLXModels.Manifest(version: MLXModels.cacheVersion, recipe: recipe,
                                          revision: ModelFiles.revision, textSHA256: textSHA,
                                          mediaSHA256: mediaSHA, seconds: Date().timeIntervalSince(started))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(manifest).write(to: partial.appendingPathComponent("manifest.json"))

        if FileManager.default.fileExists(atPath: models.cache.path) {
            try FileManager.default.removeItem(at: models.cache)
        }
        try FileManager.default.moveItem(at: partial, to: models.cache)
        // Caches of other recipes or versions are superseded; they'd only be disk.
        for name in (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? [] {
            guard name != models.cache.lastPathComponent, name.hasPrefix("eg2-") else { continue }
            try? FileManager.default.removeItem(at: root.appendingPathComponent(name))
        }
        MLX.GPU.clearCache()
        log("done in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
    }

    /// Dequantizes every tensor into bf16 and writes one safetensors file, under the names the Swift model's
    /// modules use: ggml's names for the linear weights (their paths end in `.weight` either way), and the
    /// same names without ggml's `.weight` suffix for the raw ones (norms, scales, the token embedding),
    /// which MLXNN registers under the module's own key. Nothing else is renamed.
    static func convert(file: GGUFFile, to url: URL, log: (String) -> Void) throws {
        var arrays: [String: MLXArray] = [:]
        arrays.reserveCapacity(file.tensors.count)
        for (index, tensor) in file.tensors.enumerated() {
            let values = try tensor.floats()
            let shape = tensor.shape.reversed()                    // ggml's fastest axis last
            arrays[weightName(tensor.name)] = MLXArray(values, shape).asType(.bfloat16)
            if (index + 1) % 100 == 0 { log("  \(index + 1)/\(file.tensors.count) tensors") }
        }
        try MLX.save(arrays: arrays, url: url)
        arrays.removeAll()
        MLX.GPU.clearCache()
    }

    /// The linear layers a ggml name can belong to (`ModuleInfo` keys under `blk.N.`), and the tower-level
    /// ones.
    static let linearLayers = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
                               "inp_gate", "proj", "conv_pw1", "conv_pw2",
                               "attn_out"]
    static let linearNames = ["per_layer_model_proj", "output", "mm.input_projection", "mm.a.input_projection",
                              "a.input_projection", "a.pre_encode.out"]

    /// `blk.0.attn_q.weight` keeps its name (a Linear's weight sits at that path); `blk.0.attn_norm.weight`
    /// becomes `blk.0.attn_norm` (a raw parameter's path has no `.weight`); the same for the tower-level
    /// raw parameters (`token_embd`, `output_norm`, the per-layer projection's norm).
    static func weightName(_ ggmlName: String) -> String {
        guard ggmlName.hasSuffix(".weight") else { return ggmlName }
        let stripped = String(ggmlName.dropLast(".weight".count))
        if linearNames.contains(stripped) { return ggmlName }
        // A layer's linear: `blk.N.<linear>`, with the media towers' `v.` / `a.` prefix in front of it.
        var parts = stripped.split(separator: ".").map(String.init)
        if parts.first == "v" || parts.first == "a" { parts.removeFirst() }
        if parts.count == 3, parts[0] == "blk", Int(parts[1]) != nil, linearLayers.contains(parts[2]) {
            return ggmlName
        }
        return stripped
    }
}
