import DigUpKit
import Foundation
import MLX
import MLXFast
import MLXNN

/// EmbeddingGemma 2's vision tower on MLX: a port of llama.cpp's `clip_graph_gemma4v` (tools/mtmd/models/
/// gemma4v.cpp) and the shared ViT it builds on (tools/mtmd/clip.cpp's build_vit).
///
/// SigLIP-style: pixels scaled to [-1, 1], one 16x16 patch convolution (no bias), learned position tables
/// looked up per patch (x and y), 16 pre-norm blocks with per-head q/k RMS norms, axial RoPE (theta 100: the
/// first half of each head's 64 dims rotates by the patch's column, the second half by its row — rotate-half
/// within each half), a weightless RMS norm on V, attention at scale 1, GeGLU FFNs. Then the pooler: 3x3
/// average over the patch grid, x sqrt(768), a weightless RMS norm, and the projection into the text
/// tower's input space (768 -> 512). Those rows go into `EmbeddingGemma2Text` unscaled, like llama.cpp's
/// raw-embedding rows.
final class VisionTower: Module {
    struct Config {
        let layers: Int, hidden: Int, heads: Int, headDim: Int, feedForward: Int, out: Int
        let eps: Float, ropeTheta: Float
    }

    let config: Config
    let patchSize = 16
    let merge = 3

    @ModuleInfo(key: "v") var v: VPart
    @ModuleInfo(key: "mm") var mm: MM

    /// The compute dtype, which follows the loaded weights (f32 for this tower — see `load`).
    var dtype: MLX.DType {
        v.blocks.first?.ln1.dtype ?? .bfloat16
    }

    /// The patch kernel as a matmul matrix [out, ky*kx*c], folded by `load` from the stored [o, c, ky, kx].
    /// Held in a box: a bare MLXArray property would register itself as a module parameter.
    private final class MatrixBox {
        var value: MLXArray?
    }
    private let patchMatrix = MatrixBox()

    init(_ config: Config) {
        self.config = config
        _v.wrappedValue = VPart(config: config)
        _mm.wrappedValue = MM(hidden: config.hidden, out: config.out)
        super.init()
    }

    final class VPart: Module {
        // The patch convolution lives in `VisionTower.patchMatrix` (a matmul, not a module parameter).
        @ParameterInfo(key: "position_embd") var positionEmbd: MLXArray    // [2, 10240, 768]: x table, y table
        @ModuleInfo(key: "blk") var blocks: [Block]

        init(config: Config) {
            _positionEmbd.wrappedValue = MLXArray.zeros([2, 10240, config.hidden], dtype: .bfloat16)
            _blocks.wrappedValue = (0..<config.layers).map { _ in
                Block(hidden: config.hidden, heads: config.heads, headDim: config.headDim,
                      feedForward: config.feedForward)
            }
            super.init()
        }
    }

    final class Block: Module {
        @ParameterInfo(key: "ln1") var ln1: MLXArray
        @ModuleInfo(key: "attn_q") var attnQ: Linear
        @ParameterInfo(key: "attn_q_norm") var attnQNorm: MLXArray         // [64], per head
        @ModuleInfo(key: "attn_k") var attnK: Linear
        @ParameterInfo(key: "attn_k_norm") var attnKNorm: MLXArray
        @ModuleInfo(key: "attn_v") var attnV: Linear
        @ModuleInfo(key: "attn_out") var attnOut: Linear
        @ParameterInfo(key: "attn_post_norm") var attnPostNorm: MLXArray
        @ParameterInfo(key: "ln2") var ln2: MLXArray
        @ModuleInfo(key: "ffn_gate") var ffnGate: Linear
        @ModuleInfo(key: "ffn_up") var ffnUp: Linear
        @ModuleInfo(key: "ffn_down") var ffnDown: Linear
        @ParameterInfo(key: "ffn_post_norm") var ffnPostNorm: MLXArray

        init(hidden: Int, heads: Int, headDim: Int, feedForward: Int) {
            _ln1.wrappedValue = MLXArray.zeros([hidden], dtype: .bfloat16)
            _attnQ.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            _attnQNorm.wrappedValue = MLXArray.zeros([headDim], dtype: .bfloat16)
            _attnK.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            _attnKNorm.wrappedValue = MLXArray.zeros([headDim], dtype: .bfloat16)
            _attnV.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            _attnOut.wrappedValue = Linear(heads * headDim, hidden, bias: false)
            _attnPostNorm.wrappedValue = MLXArray.zeros([hidden], dtype: .bfloat16)
            _ln2.wrappedValue = MLXArray.zeros([hidden], dtype: .bfloat16)
            _ffnGate.wrappedValue = Linear(hidden, feedForward, bias: false)
            _ffnUp.wrappedValue = Linear(hidden, feedForward, bias: false)
            _ffnDown.wrappedValue = Linear(feedForward, hidden, bias: false)
            _ffnPostNorm.wrappedValue = MLXArray.zeros([hidden], dtype: .bfloat16)
            super.init()
        }
    }

    final class MM: Module {
        @ModuleInfo(key: "input_projection") var inputProjection: Linear

        init(hidden: Int, out: Int) {
            _inputProjection.wrappedValue = Linear(hidden, out, bias: false)
            super.init()
        }
    }

    /// Replaces the placeholder weights. The patch convolution's kernel stays as stored — its flat order
    /// (c, ky, kx per output) is exactly the im2col patch order ggml's conv_2d produces, so the conv is a
    /// plain matmul against these rows. The audio tower's tensors (a.*, mm.a.*) live in the same file and
    /// are that tower's to load. Everything computes in f32: the vision tower's pooled rows have less
    /// mean-pooling dilution than the text tower's, and bf16's rounding showed up right at the parity gate.
    func load(_ weights: [String: MLXArray]) throws {
        var weights = weights.filter { !$0.key.hasPrefix("a.") && !$0.key.hasPrefix("mm.a.") }
            .mapValues { $0.asType(.float32) }
        patchMatrix.value = weights.removeValue(forKey: "v.patch_embd")?.reshaped(config.hidden, 3 * 16 * 16)
        try update(parameters: .unflattened(weights), verify: [.all])
    }

    /// Runs the tower over one image and returns one row per soft token, [tokens, out] — ready for the text
    /// tower. `pixels` is the resized image's RGB, [height, width, 3], 0-255 as float; height and width are
    /// multiples of 48 (16 px patches, 3x3 pooling — `ImageSizing` rounds them there). One image at a time,
    /// as llama.cpp's mtmd encodes them.
    func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let patch = patchSize
        var x = ((pixels.asType(dtype) / 255) * 2 - 1).transposed(2, 0, 1)   // mtmd's /255, then [-1, 1]
        let height = x.dim(1), width = x.dim(2)
        let B = 1
        let n_py = height / patch, n_px = width / patch

        // The patch convolution as a matmul: [3, H, W] -> patches [P, c*ky*kx] in ggml's im2col order, then
        // the kernel matrix. No bias.
        x = x.reshaped(B, 3, n_py, patch, n_px, patch)
        x = x.transposed(0, 2, 4, 1, 3, 5)                                // [B, py, px, c, ky, kx]
        x = x.reshaped(B, n_py * n_px, 3 * patch * patch)
        x = MLX.matmul(x, patchMatrix.value!.T)

        // Learned position tables, one per axis: patch p of the grid sits at (p % n_px, p / n_px).
        let positions = (0..<(n_py * n_px)).map { Int32($0) }
        let posX = MLXArray(positions.map { $0 % Int32(n_px) })
        let posY = MLXArray(positions.map { $0 / Int32(n_px) })
        x = x + v.positionEmbd[0][posX].expandedDimensions(axis: 0)
        x = x + v.positionEmbd[1][posY].expandedDimensions(axis: 0)

        for block in v.blocks { x = step(block, x: x, posX: posX, posY: posY) }

        // The pooler: 3x3 average over the grid, then sqrt(768), a weightless RMS norm, and the projection.
        x = x.reshaped(B, n_py, n_px, -1)
        x = x[0..., 0..<(n_py / merge * merge), 0..<(n_px / merge * merge), 0...]
        x = x.reshaped(B, n_py / merge, merge, n_px / merge, merge, -1).mean(axes: [2, 4])
        let tokens = (n_py / merge) * (n_px / merge)
        x = x.reshaped(B, tokens, -1) * Float(Foundation.sqrt(Double(config.hidden)))
        x = MLXFast.rmsNorm(x, weight: MLXArray.ones([config.hidden], dtype: x.dtype), eps: config.eps)
        return mm.inputProjection(x)
    }

    private func step(_ block: Block, x: MLXArray, posX: MLXArray, posY: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let heads = config.heads, headDim = config.headDim

        var cur = MLXFast.rmsNorm(x, weight: block.ln1, eps: config.eps)
        var q = block.attnQ(cur).reshaped(B, T, heads, headDim)
        var k = block.attnK(cur).reshaped(B, T, heads, headDim)
        var v = block.attnV(cur).reshaped(B, T, heads, headDim)
        q = MLXFast.rmsNorm(q, weight: block.attnQNorm, eps: config.eps)
        k = MLXFast.rmsNorm(k, weight: block.attnKNorm, eps: config.eps)
        v = MLXFast.rmsNorm(v, weight: MLXArray.ones([headDim], dtype: x.dtype), eps: config.eps)

        // Axial RoPE: within each head, the first half rotates by the patch's column, the second half by
        // its row — rotate-half inside each half (theta 100), like ggml's two NEOX calls with an offset.
        q = Self.axialRope(q, posX: posX, posY: posY, base: config.ropeTheta)
        k = Self.axialRope(k, posX: posX, posY: posY, base: config.ropeTheta)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        // llama.cpp's flash attention runs the vision K/V in f16 (its cast before the kernel); matching it.
        cur = MLXFast.scaledDotProductAttention(queries: q.asType(.float16), keys: k.asType(.float16),
                                                values: v.asType(.float16), scale: 1, mask: nil)
        cur = cur.transposed(0, 2, 1, 3).reshaped(B, T, heads * headDim)
        cur = block.attnOut(cur)
        cur = MLXFast.rmsNorm(cur, weight: block.attnPostNorm, eps: config.eps)
        var out = cur + x

        cur = MLXFast.rmsNorm(out, weight: block.ln2, eps: config.eps)
        cur = block.ffnDown(geluFastApproximate(block.ffnGate(cur)) * block.ffnUp(cur))
        cur = MLXFast.rmsNorm(cur, weight: block.ffnPostNorm, eps: config.eps)
        out = cur + out
        return out
    }

    /// One NEOX rotate-half over `half` dims with the given positions, applied to both halves of each head.
    /// ggml does this as two rope calls, the second offset by half the head's dims. `x` is [B, T, H, D].
    static func axialRope(_ x: MLXArray, posX: MLXArray, posY: MLXArray, base: Float) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = rope(x[0..., 0..., 0..., 0..<half], positions: posX, dims: half, base: base)
        let second = rope(x[0..., 0..., 0..., half...], positions: posY, dims: half, base: base)
        return MLX.concatenated([first, second], axis: -1)
    }

    /// NeoX rotate-half with explicit per-token positions (the patch grid's columns or rows): pairs
    /// (i, i + half) rotate by position * base^(-2i/dims), like ggml's rotate_pairs. The angles broadcast
    /// over batch and heads.
    static func rope(_ x: MLXArray, positions: MLXArray, dims: Int, base: Float) -> MLXArray {
        let half = dims / 2
        let exponent = MLXArray(stride(from: 0, to: half, by: 1).map { Float($0) }) * (-2 / Float(dims))
        let inverse = MLX.pow(MLXArray(base), exponent)                                  // [half]
        let angles = positions.asType(.float32).expandedDimensions(axis: -1) * inverse   // [T, half]
        let cos = MLX.cos(angles).expandedDimensions(axis: 0).expandedDimensions(axis: 2) // [1, T, 1, half]
        let sin = MLX.sin(angles).expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let x0 = x[0..., 0..., 0..., 0..<half], x1 = x[0..., 0..., 0..., half...]
        return MLX.concatenated([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis: -1)
    }
}

extension VisionTower {
    /// Reads the tower's shape from the mmproj GGUF's metadata.
    convenience init(file: GGUFFile) throws {
        func value(_ key: String) throws -> GGUFFile.Value {
            guard let value = file.meta("clip.vision.\(key)") else {
                throw WorkerError.failed("the media model's metadata has no vision.\(key)")
            }
            return value
        }
        func int(_ key: String) throws -> Int { try value(key).int ?? 0 }
        let hidden = try int("embedding_length")
        let heads = try int("attention.head_count")
        // rope theta isn't in the metadata: clip.cpp pins 100 for the gemma4 vision tower.
        self.init(Config(layers: try int("block_count"), hidden: hidden, heads: heads,
                         headDim: hidden / heads, feedForward: try int("feed_forward_length"),
                         out: try int("projection_dim"),
                         eps: try Float(value("attention.layer_norm_epsilon").double ?? 1e-6),
                         ropeTheta: 100))
    }
}
