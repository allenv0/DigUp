import DigUpKit
import Foundation
import MLX
import MLXFast
import MLXNN

/// One clipped matmul's four scalar bounds. Checkpoints initialize absent bounds to infinities, so clipping
/// is a no-op for those ends.
struct ClipBounds: Sendable {
    var inputMin: Float = -.infinity
    var inputMax: Float = .infinity
    var outputMin: Float = -.infinity
    var outputMax: Float = .infinity
}

/// Checkpoint-loaded clip bounds by the GGUF linear name they belong to (for example,
/// `a.blk.0.attn_q`). Plain storage is ignored by `Module.update`, which only matches MLXArrays and modules.
final class ClipTable: @unchecked Sendable {
    private let lock = NSLock()
    private var bounds: [String: ClipBounds] = [:]

    subscript(_ linear: String) -> ClipBounds {
        lock.withLock { bounds[linear] ?? ClipBounds() }
    }

    func set(_ linear: String, _ update: (inout ClipBounds) -> Void) {
        lock.withLock {
            var current = bounds[linear] ?? ClipBounds()
            update(&current)
            bounds[linear] = current
        }
    }
}

/// A linear layer with the checkpoint's input/output clamps around its one matmul, matching
/// `ClippableLinear` in the validated Python reference and `build_mm` in llama.cpp.
final class ClippedLinear: Linear {
    private let clips: ClipTable
    private let clipKey: String

    init(input: Int, output: Int, bias: Bool = false, clips: ClipTable, clipKey: String) {
        self.clips = clips
        self.clipKey = clipKey
        super.init(input, output, bias: bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let bounds = clips[clipKey]
        let current = super.callAsFunction(MLX.clip(x, min: bounds.inputMin, max: bounds.inputMax))
        return MLX.clip(current, min: bounds.outputMin, max: bounds.outputMax)
    }
}

/// EmbeddingGemma 2's USM conformer audio tower on MLX: a port of `clip_graph_gemma4a`
/// (`tools/mtmd/models/gemma4a.cpp`) using the validated `mlx-vlm` Python implementation as the second
/// reference. Two stride-two convolutions turn 128 mel bins into 1024 channels, the projection reaches the
/// conformer's 1024 width, twelve blocks apply chunked attention with sinusoidal relative positions, and a
/// 1536-wide output projection plus the audio multimodal projection makes text-tower rows.
///
/// One audio window at a time, as the indexing helper hands one WAV over.
final class AudioTower: Module {
    struct Config {
        let layers: Int
        let hidden: Int
        let heads: Int
        let headDim: Int
        let feedForward: Int
        let mels: Int
        let out: Int
        let eps: Float
        let chunkSize: Int
        let maxPast: Int
        let contextSize: Int
        let queryScale: Float
        let keyScale: Float
        let softcap: Float
        let invalidLogit: Float
        let clip: Float
        let residual: Float
    }

    let config: Config
    private let clips = ClipTable()

    @ModuleInfo(key: "a") var encoder: Encoder
    @ModuleInfo(key: "mm") var embed: Embed

    /// Activation constants derived from loaded weights: per-dimension query scales and the projected
    /// relative positions. Plain properties are ignored by `Module.update`.
    private var queryScales: [MLXArray] = []
    private var relativeKeys: [MLXArray] = []

    init(_ config: Config) {
        self.config = config
        _encoder.wrappedValue = Encoder(config: config, clips: clips)
        _embed.wrappedValue = Embed(out: config.out)
        super.init()
    }

    /// Reads the tower's shape from the mmproj GGUF's metadata.
    convenience init(file: GGUFFile) throws {
        try self.init(Config(file: file))
    }

    final class Encoder: Module {
        @ModuleInfo(key: "conv0") var stage0: ConvStage
        @ModuleInfo(key: "conv1") var stage1: ConvStage
        @ModuleInfo(key: "input_projection") var inputProjection: Linear
        @ModuleInfo(key: "pre_encode") var output: OutputProjection
        @ModuleInfo(key: "blk") var blocks: [Block]

        init(config: Config, clips: ClipTable) {
            _stage0.wrappedValue = ConvStage(inChannels: 1, outChannels: 128)
            _stage1.wrappedValue = ConvStage(inChannels: 128, outChannels: 32)
            _inputProjection.wrappedValue = Linear(32 * 32, config.hidden, bias: false)
            _output.wrappedValue = OutputProjection(inChannels: config.hidden, outChannels: 1536)
            _blocks.wrappedValue = (0..<config.layers).map { index in
                Block(config: config, clips: clips, prefix: "a.blk.\(index)")
            }
            super.init()
        }

        func front(_ mel: MLXArray) -> MLXArray {
            // [T, 128] -> [1, T, 128, 1], the channel-last shape `MLX.conv2d` expects.
            var x = mel.expandedDimensions(axis: 0).expandedDimensions(axis: -1)
            x = stage0(x)
            x = stage1(x)
            let channels = x.dim(-1)
            x = x.reshaped(1, x.dim(1), x.dim(2) * channels)
            return inputProjection(x)
        }
    }

    final class ConvStage: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        @ParameterInfo(key: "norm") var norm: MLXArray
        let eps: Float

        init(inChannels: Int, outChannels: Int) {
            _weight.wrappedValue = MLXArray.zeros([outChannels, 3, 3, inChannels], dtype: .bfloat16)
            _norm.wrappedValue = MLXArray.ones([outChannels], dtype: .bfloat16)
            eps = 1e-6
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // Symmetric padding of one, then stride two in time and frequency.
            var current = MLX.conv2d(x, weight, stride: 2, padding: 1)
            current = MLXFast.layerNorm(current, weight: norm, bias: nil, eps: eps)
            return MLXNN.relu(current)
        }
    }

    final class OutputProjection: Module {
        @ModuleInfo(key: "out") var linear: Linear

        init(inChannels: Int, outChannels: Int) {
            _linear.wrappedValue = Linear(inChannels, outChannels, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear(x)
        }
    }

    final class Block: Module {
        @ParameterInfo(key: "ffn_norm") var firstNorm: MLXArray
        @ModuleInfo(key: "ffn_up") var firstUp: ClippedLinear
        @ModuleInfo(key: "ffn_down") var firstDown: ClippedLinear
        @ParameterInfo(key: "ffn_post_norm") var firstPostNorm: MLXArray
        @ParameterInfo(key: "attn_pre_norm") var attentionPreNorm: MLXArray
        @ModuleInfo(key: "attn_q") var query: ClippedLinear
        @ModuleInfo(key: "attn_k") var key: ClippedLinear
        @ModuleInfo(key: "attn_v") var value: ClippedLinear
        @ModuleInfo(key: "attn_rel") var relativeKey: Linear
        @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray
        @ModuleInfo(key: "attn_out") var output: ClippedLinear
        @ParameterInfo(key: "attn_post_norm") var attentionPostNorm: MLXArray
        @ParameterInfo(key: "norm_conv") var convNorm: MLXArray
        @ModuleInfo(key: "conv_pw1") var convStart: ClippedLinear
        @ParameterInfo(key: "conv_dw") var convDepth: MLXArray
        @ParameterInfo(key: "conv_norm") var convWeightNorm: MLXArray
        @ModuleInfo(key: "conv_pw2") var convEnd: ClippedLinear
        @ParameterInfo(key: "ln2") var outputNorm: MLXArray
        @ParameterInfo(key: "ffn_norm_1") var secondNorm: MLXArray
        @ModuleInfo(key: "ffn_up_1") var secondUp: ClippedLinear
        @ModuleInfo(key: "ffn_down_1") var secondDown: ClippedLinear
        @ParameterInfo(key: "ffn_post_norm_1") var secondPostNorm: MLXArray

        init(config: Config, clips: ClipTable, prefix: String) {
            func clipped(_ input: Int, _ output: Int, _ name: String) -> ClippedLinear {
                ClippedLinear(input: input, output: output, clips: clips, clipKey: "\(prefix).\(name)")
            }
            _firstNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _firstUp.wrappedValue = clipped(config.hidden, config.feedForward, "ffn_up")
            _firstDown.wrappedValue = clipped(config.feedForward, config.hidden, "ffn_down")
            _firstPostNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _attentionPreNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _query.wrappedValue = clipped(config.hidden, config.hidden, "attn_q")
            _key.wrappedValue = clipped(config.hidden, config.hidden, "attn_k")
            _value.wrappedValue = clipped(config.hidden, config.hidden, "attn_v")
            _relativeKey.wrappedValue = Linear(config.hidden, config.hidden, bias: false)
            _perDimScale.wrappedValue = MLXArray.zeros([config.headDim], dtype: .bfloat16)
            _output.wrappedValue = clipped(config.hidden, config.hidden, "attn_out")
            _attentionPostNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _convNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _convStart.wrappedValue = clipped(config.hidden, 2 * config.hidden, "conv_pw1")
            _convDepth.wrappedValue = MLXArray.zeros([config.hidden, 5], dtype: .bfloat16)
            _convWeightNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _convEnd.wrappedValue = clipped(config.hidden, config.hidden, "conv_pw2")
            _outputNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _secondNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            _secondUp.wrappedValue = clipped(config.hidden, config.feedForward, "ffn_up_1")
            _secondDown.wrappedValue = clipped(config.feedForward, config.hidden, "ffn_down_1")
            _secondPostNorm.wrappedValue = MLXArray.zeros([config.hidden], dtype: .bfloat16)
            super.init()
        }
    }

    final class Embed: Module {
        @ModuleInfo(key: "a") var audio: Projector

        init(out: Int) {
            _audio.wrappedValue = Projector(inChannels: 1536, outChannels: out)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            audio(x)
        }
    }

    final class Projector: Module {
        @ModuleInfo(key: "input_projection") var projection: Linear

        init(inChannels: Int, outChannels: Int) {
            _projection.wrappedValue = Linear(inChannels, outChannels, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            projection(x)
        }
    }

    /// Loads this tower's weights from the shared media file. Clip scalars are checkpoint parameters for
    /// the surrounding linear layers, not module weights: they are parsed into `clips` and excluded from the
    /// module update. Anything else outside this tower is an error.
    func load(_ weights: [String: MLXArray]) throws {
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(weights.count)
        for (source, array) in weights {
            if let linear = Self.clipLinearName(source) {
                let value = array.asType(.float32).item(Float.self)
                clips.set(linear) { bounds in
                    if source.hasSuffix(".input_min") { bounds.inputMin = value }
                    else if source.hasSuffix(".input_max") { bounds.inputMax = value }
                    else if source.hasSuffix(".output_min") { bounds.outputMin = value }
                    else { bounds.outputMax = value }
                }
                continue
            }
            guard Self.isAudioWeight(source) else { continue }
            let internalName = try Self.internalKey(source)
            var value = array
            // The shared cache stores convolution kernels as `[out, in, ky, kx]`; MLX's channel-last
            // convolution expects `[out, ky, kx, in]`.
            if internalName == "a.conv0.weight" || internalName == "a.conv1.weight" {
                value = value.transposed(0, 2, 3, 1)
            }
            mapped[internalName] = value
        }
        try update(parameters: .unflattened(mapped), verify: [.all])
        deriveConstants()
    }

    /// Runs the tower over Float32 log-mel frames `[frames, 128]` and returns one row per soft token,
    /// `[tokens, 512]`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        precondition(mel.dim(1) == config.mels, "the audio frontend makes \(config.mels) mel bins")
        var hidden = encoder.front(mel)
        for (index, block) in encoder.blocks.enumerated() {
            hidden = forward(block, index: index, hidden)
        }
        hidden = encoder.output(hidden)
        let norm = MLXFast.rmsNorm(hidden, weight: MLXArray.ones([1536], dtype: hidden.dtype), eps: config.eps)
        return embed.audio.projection(norm)
    }

    private func forward(_ block: Block, index: Int, _ input: MLXArray) -> MLXArray {
        var x = feedForward(block: block, first: true, input)
        x = attention(block: block, index: index, x)
        x = lightConv(block: block, x)
        x = feedForward(block: block, first: false, x)
        return MLXFast.rmsNorm(MLX.clip(x, min: -config.clip, max: config.clip),
                               weight: block.outputNorm, eps: config.eps)
    }

    private func feedForward(block: Block, first: Bool, _ input: MLXArray) -> MLXArray {
        let norm = first ? block.firstNorm : block.secondNorm
        let up = first ? block.firstUp : block.secondUp
        let down = first ? block.firstDown : block.secondDown
        let postNorm = first ? block.firstPostNorm : block.secondPostNorm
        var x = MLX.clip(input, min: -config.clip, max: config.clip)
        x = MLXFast.rmsNorm(x, weight: norm, eps: config.eps)
        x = up(x)
        x = MLXNN.silu(x)
        x = down(x)
        x = MLX.clip(x, min: -config.clip, max: config.clip)
        x = MLXFast.rmsNorm(x, weight: postNorm, eps: config.eps)
        return input + x * config.residual
    }

    private func lightConv(block: Block, _ input: MLXArray) -> MLXArray {
        let residual = input
        var x = MLXFast.rmsNorm(input, weight: block.convNorm, eps: config.eps)
        x = block.convStart(x)
        let width = x.dim(-1) / 2
        let first = x[0..., 0..., 0..<width]
        x = first * MLX.sigmoid(x[0..., 0..., width...])
        let padding: [IntOrPair] = [.init((0, 0)), .init((4, 0)), .init((0, 0))]
        x = MLX.padded(x, widths: padding)
        // The checkpoint stores one length-five kernel per channel: [channels, 5] becomes grouped Conv1d's
        // [channels, 5, 1].
        let depth = block.convDepth.reshaped([config.hidden, 5, 1])
        x = MLX.conv1d(x, depth, stride: 1, padding: 0, groups: config.hidden)
        x = MLX.clip(x, min: -config.clip, max: config.clip)
        x = MLXFast.rmsNorm(x, weight: block.convWeightNorm, eps: config.eps)
        x = MLXNN.silu(x)
        return residual + block.convEnd(x)
    }

    private func attention(block: Block, index: Int, _ input: MLXArray) -> MLXArray {
        let time = input.dim(1)
        let heads = config.heads
        let headDim = config.headDim
        let chunk = config.chunkSize

        let residual = input
        var x = MLX.clip(input, min: -config.clip, max: config.clip)
        x = MLXFast.rmsNorm(x, weight: block.attentionPreNorm, eps: config.eps)
        let query = block.query(x).asType(.float32).reshaped([1, time, heads, headDim])
        let key = block.key(x).asType(.float32).reshaped([1, time, heads, headDim])
        let value = block.value(x).asType(.float32).reshaped([1, time, heads, headDim])
        let scaledQuery = query * config.queryScale * queryScales[index].expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let scaledKey = key * config.keyScale

        let blocks = (time + chunk - 1) / chunk
        let paddedTime = blocks * chunk
        let padding = paddedTime - time
        let queryPadded = padding == 0 ? scaledQuery
            : MLX.concatenated([scaledQuery, MLXArray.zeros([1, padding, heads, headDim], dtype: .float32)], axis: 1)
        let queries = queryPadded.reshaped([1, blocks, chunk, heads, headDim])
        let keys = blockedContext(scaledKey, blocks: blocks)
        let values = blockedContext(value, blocks: blocks)

        // Content scores: [B, N, U, W, H] @ [B, N, U, H, C] -> [B, N, U, W, C].
        let queryHeads = queries.transposed(0, 3, 1, 2, 4)
        let keyHeads = keys.transposed(0, 3, 1, 4, 2)
        var scores = MLX.matmul(queryHeads, keyHeads)
        scores = scores + relativeScores(queries: queryHeads, index: index)
        scores = MLX.tanh(scores / config.softcap) * config.softcap
        scores = MLX.where(attentionMask(blocks: blocks, time: time), scores, config.invalidLogit)
        let probabilities = MLX.softmax(scores, axis: -1)
        let valueHeads = values.transposed(0, 3, 1, 2, 4)
        var attended = MLX.matmul(probabilities, valueHeads)
        attended = attended.transposed(0, 2, 3, 1, 4).reshaped([1, blocks * chunk, heads * headDim])
        attended = attended[0..., 0..<time, 0...]
        attended = block.output(attended)
        attended = MLX.clip(attended, min: -config.clip, max: config.clip)
        return residual + MLXFast.rmsNorm(attended, weight: block.attentionPostNorm, eps: config.eps)
    }

    /// Overlapping key/value windows for chunked attention: `[1, T, N, H]` becomes `[1, U, C, N, H]` with
    /// twelve older positions and eleven future slots of padding around every twelve-query chunk.
    private func blockedContext(_ x: MLXArray, blocks: Int) -> MLXArray {
        let heads = x.dim(2)
        let headDim = x.dim(3)
        let padded = MLX.concatenated([
            MLXArray.zeros([1, config.maxPast, heads, headDim], dtype: x.dtype),
            x,
            MLXArray.zeros([1, config.maxPast + config.chunkSize - 1, heads, headDim], dtype: x.dtype),
        ], axis: 1)
        var windows: [MLXArray] = []
        windows.reserveCapacity(blocks)
        for block in 0..<blocks {
            windows.append(padded[0..., block * config.chunkSize..<(block * config.chunkSize + config.contextSize),
                                  0..., 0...])
        }
        return MLX.stacked(windows, axis: 1)
    }

    /// Blocked relative-position scores, matching the Python reference's padded reshape/truncate shift.
    private func relativeScores(queries: MLXArray, index: Int) -> MLXArray {
        let blocks = queries.dim(2)
        let chunk = queries.dim(3)
        let flat = queries.reshaped([1, config.heads, blocks * chunk, config.headDim])
        let keys = relativeKeys[index].transposed(1, 2, 0)   // [N, H, R]
        var scores = MLX.matmul(flat, keys)                  // [1, N, U*W, R]
        scores = scores.reshaped([1, config.heads, blocks, chunk, relativeKeys[index].dim(0)])
        let pad = config.contextSize + 1 - relativeKeys[index].dim(0)
        let padded = MLX.padded(scores, widths: [0, 0, 0, 0, [0, pad]])
        let flattened = padded.reshaped([1, config.heads, blocks, chunk * (config.contextSize + 1)])
        let truncated = MLX.split(flattened, indices: [chunk * config.contextSize], axis: 3)[0]
        return truncated.reshaped([1, config.heads, blocks, chunk, config.contextSize])
    }

    /// Valid query/key positions for chunked attention. Queries have no padding here (they are trimmed after
    /// attention); keys exclude both padded edge slots and any frame beyond this input.
    private func attentionMask(blocks: Int, time: Int) -> MLXArray {
        // The blocked-context layout is [B, U, W, C] after broadcasting, but construct [U, W, C] first.
        var mask = [Bool](repeating: false, count: blocks * config.chunkSize * config.contextSize)
        for block in 0..<blocks {
            for query in 0..<config.chunkSize {
                for key in 0..<config.contextSize {
                    // Query q at global time b*C+q sees context key k at global time b*C-maxPast+k.
                    let keyTime = block * config.chunkSize - config.maxPast + key
                    let distance = query + config.maxPast - key
                    let causal = distance >= 0 && distance < config.maxPast
                    mask[(block * config.chunkSize + query) * config.contextSize + key]
                        = causal && keyTime >= 0 && keyTime < time
                }
            }
        }
        return MLXArray(mask, [1, 1, blocks, config.chunkSize, config.contextSize])
    }

    /// The GGUF linear that a clip-scalar tensor belongs to (`nil` for a module weight).
    static func clipLinearName(_ source: String) -> String? {
        for suffix in [".input_min", ".input_max", ".output_min", ".output_max"] where source.hasSuffix(suffix) {
            return String(source.dropLast(suffix.count))
        }
        return nil
    }

    /// Whether a converted safetensors key belongs to this tower. Vision's `mm.input_projection` is not
    /// audio's, despite sharing the `mm.` prefix.
    static func isAudioWeight(_ source: String) -> Bool {
        source.hasPrefix("a.") || (source.hasPrefix("mm.a.")
            || (source.hasPrefix("mm.") && !source.hasPrefix("mm.input_projection.")))
    }

    /// Maps this checkpoint's converted key to the module tree's nested key. The safetensors cache already
    /// dropped `.weight` from raw tensors whose module paths have no `.weight`; the leaves that need one
    /// (`ffn_up_1`, the depthwise kernel, the relative-key projection, and the frontend kernels) regain it.
    static func internalKey(_ source: String) throws -> String {
        switch source {
        case "a.conv1d.0": return "a.conv0.weight"
        case "a.conv1d.0.norm": return "a.conv0.norm"
        case "a.conv1d.1": return "a.conv1.weight"
        case "a.conv1d.1.norm": return "a.conv1.norm"
        case "mm.a.input_projection.weight": return "mm.a.input_projection.weight"
        case "mm.input_projection.weight": throw WorkerError.failed("vision weights do not belong to audio")
        default: break
        }
        if source.hasPrefix("a.blk.") {
            let components = source.split(separator: ".").map(String.init)
            if components.count == 4, components.last == "attn_k_rel" {
                return "\(components[0]).\(components[1]).\(components[2]).attn_rel.weight"
            }
            // llama.cpp loads these two convolution norms in reverse order because upstream swapped
            // their names when writing the GGUF.
            if components.count == 4, components[0] == "a", components[1] == "blk",
               components[3] == "norm_conv" {
                return "\(components[0]).\(components[1]).\(components[2]).conv_norm"
            }
            if components.count == 4, components[0] == "a", components[1] == "blk",
               components[3] == "conv_norm" {
                return "\(components[0]).\(components[1]).\(components[2]).norm_conv"
            }
            if source.hasSuffix(".weight") { return source }
            let leaf = source.split(separator: ".").last.map(String.init) ?? ""
            if ["ffn_up_1", "ffn_down_1"].contains(leaf) {
                return source + ".weight"
            }
            if ["conv_dw", "ffn_norm", "ffn_post_norm", "ffn_norm_1", "ffn_post_norm_1", "attn_pre_norm",
                "attn_post_norm", "norm_conv", "conv_norm", "ln2", "per_dim_scale"].contains(leaf) {
                return source
            }
            throw WorkerError.failed("unexpected audio tensor \(source)")
        }
        if ["a.input_projection.weight", "a.pre_encode.out.weight", "a.pre_encode.out.bias"]
            .contains(source) { return source }
        throw WorkerError.failed("unexpected audio tensor \(source)")
    }

    /// Precomputes input-independent attention constants once weights are loaded.
    private func deriveConstants() {
        queryScales = encoder.blocks.map { block in
            // The GGUF already stores the softplus-transformed per-dimension scales (llama.cpp multiplies
            // them directly). Keep them as loaded; attention applies the global query scale exactly once.
            block.perDimScale.asType(.bfloat16)
        }
        let positions = config.maxPast + 1
        let frequencies = config.hidden / 2
        let positionIndex = (0..<positions).map { Float(config.maxPast - $0) }
        let inverse = (0..<frequencies).map { index in
            Float(Foundation.exp(-Double(index) * Foundation.log(10_000.0) / Double(frequencies - 1)))
        }
        relativeKeys = encoder.blocks.map { block in
            var projectedInput = [Float](repeating: 0, count: positions * config.hidden)
            for position in 0..<positions {
                for frequency in 0..<frequencies {
                    let angle = positionIndex[position] * inverse[frequency]
                    projectedInput[position * config.hidden + frequency] = Foundation.sin(angle)
                    projectedInput[position * config.hidden + frequencies + frequency] = Foundation.cos(angle)
                }
            }
            let projected = MLX.matmul(MLXArray(projectedInput, [positions, config.hidden]).asType(.float32),
                                       block.relativeKey.weight.asType(.float32).T)
            return projected.reshaped([positions, config.heads, config.headDim]).asType(.float32)
        }
    }
}

extension AudioTower.Config {
    /// Reads the tower's shape from the mmproj GGUF's metadata. Several audio constants are architecture
    /// constants rather than metadata: the C++ graph pins the 320-sample window, 160-sample hop, chunked
    /// 12/12/0 attention geometry, 1e10 gradient clips, and 50 logit softcap; the Python reference confirms them.
    init(file: GGUFFile) throws {
        func value(_ key: String) throws -> GGUFFile.Value {
            guard let value = file.meta("clip.audio.\(key)") else {
                throw WorkerError.failed("the media model's metadata has no audio.\(key)")
            }
            return value
        }
        func int(_ key: String) throws -> Int { try value(key).int ?? 0 }
        let hidden = try int("embedding_length")
        let heads = try int("attention.head_count")
        let headDim = hidden / heads
        self.init(
            layers: try int("block_count"),
            hidden: hidden,
            heads: heads,
            headDim: headDim,
            feedForward: try int("feed_forward_length"),
            mels: try int("num_mel_bins"),
            out: try int("projection_dim"),
            eps: try Float(value("attention.layer_norm_epsilon").double ?? 1e-6),
            chunkSize: 12,
            maxPast: 12,
            contextSize: 24,
            queryScale: Float(1 / Foundation.sqrt(Double(headDim))) / Float(Foundation.log(2.0)),
            keyScale: Float(Foundation.log(1 + Foundation.exp(1.0))) / Float(Foundation.log(2.0)),
            softcap: 50,
            invalidLogit: -1e9,
            clip: 10_000_000_000,
            residual: 0.5)
    }
}
