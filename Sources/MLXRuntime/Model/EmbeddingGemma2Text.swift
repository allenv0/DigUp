import DigUpKit
import Foundation
import MLX
import MLXFast
import MLXNN

/// EmbeddingGemma 2's text tower on MLX: a faithful port of llama.cpp's `gemma-embedding2` graph
/// (Vendor/llama.cpp/src/models/gemma-embedding2.cpp), which the repo's parity tests hold against the
/// mlx-vlm and transformers references.
///
/// A bidirectional encoder (no causal mask, no KV cache): hybrid attention — every 6th layer full, the rest
/// sliding with a symmetric window (|i - j| <= n_swa / 2) — with per-head q/k norms and no 1/sqrt(d)
/// attention scale (the norms make it unnecessary), a V norm without a weight, per-layer embeddings (the
/// PLE-style side path fed by one shared projection of the input), per-layer output scales, and a final
/// projection to the 768-d space applied per token. Mean pooling happens above this (`MLXEmbedder`), as
/// llama.cpp's pooling does.
///
/// Weights and activations run in bf16, as the validated Python reference does. Norm weights are used as
/// stored: the GGUF carries the reference's `1 + w` already.
final class EmbeddingGemma2Text: Module {
    let config: Config

    @ParameterInfo(key: "token_embd") var tokEmbd: MLXArray
    @ModuleInfo(key: "per_layer_model_proj") var perLayerModelProj: Linear
    @ParameterInfo(key: "per_layer_proj_norm") var perLayerProjNorm: MLXArray
    @ParameterInfo(key: "output_norm") var outputNorm: MLXArray
    @ModuleInfo(key: "output") var output: Linear
    @ModuleInfo(key: "blk") var blocks: [Block]

    /// What the GGUF's metadata says about the tower (`Config(file:)` reads it).
    struct Config {
        let layers: Int
        let embedding: Int
        let embeddingOut: Int
        let perLayer: Int
        let feedForward: Int
        let heads: Int
        /// Key/value head count per layer (the full-attention layers share one KV head).
        let kvHeads: [Int]
        let headDim: (sliding: Int, full: Int)
        let ropeDims: (sliding: Int, full: Int)
        let ropeBase: (sliding: Float, full: Float)
        /// True where a layer's attention is windowed (`attention.sliding_window_pattern`, true = sliding).
        let sliding: [Bool]
        let window: Int
        let eps: Float
        let vocab: Int
    }

    init(_ config: Config) {
        self.config = config
        _tokEmbd.wrappedValue = MLXArray.zeros([config.vocab, config.embedding], dtype: .bfloat16)
        _perLayerModelProj.wrappedValue = Linear(config.embedding, config.perLayer * config.layers, bias: false)
        _perLayerProjNorm.wrappedValue = MLXArray.zeros([config.perLayer], dtype: .bfloat16)
        _outputNorm.wrappedValue = MLXArray.zeros([config.embedding], dtype: .bfloat16)
        _output.wrappedValue = Linear(config.embedding, config.embeddingOut, bias: false)
        _blocks.wrappedValue = (0..<config.layers).map { Block(config: config, index: $0) }
        super.init()
    }

    /// Replaces the placeholder weights with the converted GGUF tensors, names matched exactly (and
    /// completely: a missing or extra tensor is an error, not a silent zero).
    func load(_ weights: [String: MLXArray]) throws {
        try update(parameters: .unflattened(weights), verify: [.all])
    }

    /// Runs the tower over a batch of independent sequences (right-padded) and returns the projected
    /// per-token vectors, [batch, tokens, embeddingOut] in bf16. `embeddings` rows are already what the
    /// graph's input is: token rows scaled by sqrt(n_embd), media rows as their encoder projected them
    /// (unscaled, like llama.cpp's raw-embedding rows). Padding positions come out as zeros, so pooling
    /// above can just skip them.
    func callAsFunction(_ embeddings: MLXArray, lengths: [Int]) -> MLXArray {
        let batch = embeddings.dim(0)
        let tokens = embeddings.dim(1)
        let scale = Float(Foundation.sqrt(Double(config.embedding)))

        // The per-layer inputs, one shared projection of the input (the graph's build_inp_per_layer). The
        // input is the scaled one, so the projection comes back down by the same scale, as the graph does:
        // [batch, tokens, layers * perLayer] -> reshaped -> normed per layer -> [batch, layers, tokens, perLayer].
        var perLayer = perLayerModelProj(embeddings) * (1 / scale)
        perLayer = perLayer.reshaped(batch, tokens, config.layers, config.perLayer)
        perLayer = MLXFast.rmsNorm(perLayer, weight: perLayerProjNorm, eps: config.eps)
        perLayer = perLayer.transposed(0, 2, 1, 3)

        // Two masks serve the whole batch: the sliding layers' window and the full layers' padding-only.
        let windowMask = Self.attentionMask(lengths: lengths, batch: batch, tokens: tokens, window: config.window / 2)
        let paddingMask = Self.attentionMask(lengths: lengths, batch: batch, tokens: tokens, window: nil)

        var hidden = embeddings
        for (index, block) in blocks.enumerated() {
            hidden = step(block, index: index, hidden: hidden, perLayer: perLayer,
                          mask: block.halfWindow == nil ? paddingMask : windowMask, batch: batch, tokens: tokens)
        }

        // The output projection is per token, which pools the same as pooling first (the graph's note).
        var result = MLXFast.rmsNorm(hidden, weight: outputNorm, eps: config.eps)
        result = output(result)
        return result
    }

    /// One transformer block, in the graph's order: attention, both residuals with their post-norms, the
    /// per-layer-embedding side path, then the output scale.
    private func step(_ block: Block, index: Int, hidden: MLXArray, perLayer: MLXArray, mask: MLXArray,
                      batch: Int, tokens: Int) -> MLXArray {
        let heads = config.heads
        let kvHeads = config.kvHeads[index]
        let headDim = block.headDim

        var cur = MLXFast.rmsNorm(hidden, weight: block.attnNorm, eps: config.eps)
        var q = block.attnQ(cur).reshaped(batch, tokens, heads, headDim)
        var k = block.attnK(cur).reshaped(batch, tokens, kvHeads, headDim)
        var v = block.attnV(cur).reshaped(batch, tokens, kvHeads, headDim)
        q = MLXFast.rmsNorm(q, weight: block.attnQNorm, eps: config.eps)
        k = MLXFast.rmsNorm(k, weight: block.attnKNorm, eps: config.eps)
        v = MLXFast.rmsNorm(v, weight: block.unit, eps: config.eps)

        // Positions run along axis -2 for RoPE and attention: [batch, heads, tokens, headDim].
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: block.ropeDims, traditional: false, base: block.ropeBase, scale: 1,
                         offset: 0)
        k = MLXFast.RoPE(k, dimensions: block.ropeDims, traditional: false, base: block.ropeBase, scale: 1,
                         offset: 0)
        cur = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1, mask: mask)
        cur = cur.transposed(0, 2, 1, 3).reshaped(batch, tokens, heads * headDim)
        cur = block.attnOutput(cur)
        cur = MLXFast.rmsNorm(cur, weight: block.postAttentionNorm, eps: config.eps)
        let attended = cur + hidden

        cur = MLXFast.rmsNorm(attended, weight: block.ffnNorm, eps: config.eps)
        cur = block.ffnDown(geluApproximate(block.ffnGate(cur)) * block.ffnUp(cur))
        cur = MLXFast.rmsNorm(cur, weight: block.postFFWNorm, eps: config.eps)
        var out = cur + attended

        // The per-layer embedding: a gated copy of this block's shared per-layer input, projected back and
        // added in (the graph's "per-layer embedding" step).
        let gate = geluApproximate(block.inpGate(out)) * perLayer[0..., index, 0..., 0...]
        let side = MLXFast.rmsNorm(block.proj(gate), weight: block.postNorm, eps: config.eps)
        out = out + side
        return out * block.layerOutputScale
    }

    /// What each position may attend to, additive (0 to attend, -inf blocked): nothing past its batch row's
    /// length, the symmetric window where there is one, and — so no row is left with nothing, which softmax
    /// would make NaN — a padding position may look at position 0. Real rows always see themselves, and
    /// padding rows' outputs are dropped before anything reads them.
    static func attentionMask(lengths: [Int], batch: Int, tokens: Int, window: Int?) -> MLXArray {
        var mask = [Float](repeating: -Float.infinity, count: batch * tokens * tokens)
        for b in 0..<batch {
            let length = lengths[b]
            for i in 0..<tokens {
                for j in 0..<tokens {
                    guard j < length else { continue }
                    let inWindow = window.map { abs(i - j) <= $0 } ?? true
                    if inWindow || (i >= length && j == 0) {
                        mask[(b * tokens + i) * tokens + j] = 0
                    }
                }
            }
        }
        return MLXArray(mask, [batch, 1, tokens, tokens]).asType(.bfloat16)
    }

    final class Block: Module {
        let headDim: Int
        let ropeDims: Int
        let ropeBase: Float
        /// Half the symmetric window; nil on a full-attention layer, which masks nothing but padding.
        let halfWindow: Int?
        /// rms_norm needs a weight and V's graph has none: ones stand in (the product is the norm itself).
        lazy var unit: MLXArray = MLXArray.ones([headDim], dtype: .bfloat16)

        @ParameterInfo(key: "attn_norm") var attnNorm: MLXArray
        @ModuleInfo(key: "attn_q") var attnQ: Linear
        @ParameterInfo(key: "attn_q_norm") var attnQNorm: MLXArray
        @ModuleInfo(key: "attn_k") var attnK: Linear
        @ParameterInfo(key: "attn_k_norm") var attnKNorm: MLXArray
        @ModuleInfo(key: "attn_v") var attnV: Linear
        @ModuleInfo(key: "attn_output") var attnOutput: Linear
        @ParameterInfo(key: "post_attention_norm") var postAttentionNorm: MLXArray
        @ParameterInfo(key: "ffn_norm") var ffnNorm: MLXArray
        @ModuleInfo(key: "ffn_gate") var ffnGate: Linear
        @ModuleInfo(key: "ffn_up") var ffnUp: Linear
        @ModuleInfo(key: "ffn_down") var ffnDown: Linear
        @ParameterInfo(key: "post_ffw_norm") var postFFWNorm: MLXArray
        @ModuleInfo(key: "inp_gate") var inpGate: Linear
        @ModuleInfo(key: "proj") var proj: Linear
        @ParameterInfo(key: "post_norm") var postNorm: MLXArray
        @ParameterInfo(key: "layer_output_scale") var layerOutputScale: MLXArray

        init(config: Config, index: Int) {
            let sliding = config.sliding[index]
            headDim = sliding ? config.headDim.sliding : config.headDim.full
            ropeDims = sliding ? config.ropeDims.sliding : config.ropeDims.full
            ropeBase = sliding ? config.ropeBase.sliding : config.ropeBase.full
            halfWindow = sliding ? config.window / 2 : nil
            let kvHeads = config.kvHeads[index]
            _attnNorm.wrappedValue = MLXArray.zeros([config.embedding], dtype: .bfloat16)
            _attnQ.wrappedValue = Linear(config.embedding, config.heads * headDim, bias: false)
            _attnQNorm.wrappedValue = MLXArray.zeros([headDim], dtype: .bfloat16)
            _attnK.wrappedValue = Linear(config.embedding, kvHeads * headDim, bias: false)
            _attnKNorm.wrappedValue = MLXArray.zeros([headDim], dtype: .bfloat16)
            _attnV.wrappedValue = Linear(config.embedding, kvHeads * headDim, bias: false)
            _attnOutput.wrappedValue = Linear(config.heads * headDim, config.embedding, bias: false)
            _postAttentionNorm.wrappedValue = MLXArray.zeros([config.embedding], dtype: .bfloat16)
            _ffnNorm.wrappedValue = MLXArray.zeros([config.embedding], dtype: .bfloat16)
            _ffnGate.wrappedValue = Linear(config.embedding, config.feedForward, bias: false)
            _ffnUp.wrappedValue = Linear(config.embedding, config.feedForward, bias: false)
            _ffnDown.wrappedValue = Linear(config.feedForward, config.embedding, bias: false)
            _postFFWNorm.wrappedValue = MLXArray.zeros([config.embedding], dtype: .bfloat16)
            _inpGate.wrappedValue = Linear(config.embedding, config.perLayer, bias: false)
            _proj.wrappedValue = Linear(config.perLayer, config.embedding, bias: false)
            _postNorm.wrappedValue = MLXArray.zeros([config.embedding], dtype: .bfloat16)
            _layerOutputScale.wrappedValue = MLXArray.zeros([1], dtype: .bfloat16)
            super.init()
        }
    }
}

extension EmbeddingGemma2Text.Config {
    /// Reads the tower's shape from the text GGUF's metadata (the keys llama.cpp's `gemma-embedding2` reads).
    init(file: GGUFFile) throws {
        func value(_ key: String) throws -> GGUFFile.Value {
            guard let value = file.meta("gemma-embedding2.\(key)") else {
                throw WorkerError.failed("the text model's metadata has no \(key)")
            }
            return value
        }
        func int(_ key: String) throws -> Int { try value(key).int ?? 0 }
        self.init(
            layers: try int("block_count"),
            embedding: try int("embedding_length"),
            embeddingOut: try int("embedding_length_out"),
            perLayer: try int("embedding_length_per_layer_input"),
            feedForward: try int("feed_forward_length"),
            heads: try int("attention.head_count"),
            kvHeads: try value("attention.head_count_kv").ints ?? [],
            headDim: (try int("attention.key_length_swa"), try int("attention.key_length")),
            ropeDims: (try int("rope.dimension_count_swa"), try int("rope.dimension_count")),
            ropeBase: (try Float(value("rope.freq_base_swa").double ?? 10_000),
                       try Float(value("rope.freq_base").double ?? 1_000_000)),
            sliding: try value("attention.sliding_window_pattern").array?.compactMap {
                if case .bool(let value) = $0 { return value }
                return nil
            } ?? [],
            window: try int("attention.sliding_window"),
            eps: try Float(value("attention.layer_norm_rms_epsilon").double ?? 1e-6),
            vocab: file.meta("tokenizer.ggml.tokens")?.array?.count ?? 0)
    }
}
