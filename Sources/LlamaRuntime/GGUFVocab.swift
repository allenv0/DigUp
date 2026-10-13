import CLlama
import DigUpKit
import Foundation

/// The model's tokenizer, read from inside the text GGUF through llama.cpp — the same tokenizer llama.cpp's
/// engine runs, so both engines make identical ids by construction (the reference vectors' ids too).
///
/// Only the model file is opened (memory-mapped; the weights are never touched), which costs a fraction of a
/// second. `MLXEmbedder` runs one of these beside its MLX weights.
public final class GGUFVocab: @unchecked Sendable {
    private let model: OpaquePointer
    private let vocab: OpaquePointer?
    public let eos: Int

    /// `path` is the text GGUF (`ModelFiles.text`).
    public init(path: String, log: URL? = nil) throws {
        LlamaLog.shared.send(to: log)
        _ = LlamaEmbedder.backend
        var params = llama_model_default_params()
        params.n_gpu_layers = 0          // ids only: nothing goes to the GPU, the weights stay mapped and cold
        guard let model = llama_model_load_from_file(path, params) else {
            throw WorkerError.failed("can't read the tokenizer out of \(path)")
        }
        self.model = model
        vocab = llama_model_get_vocab(model)
        eos = Int(llama_vocab_eos(vocab))
    }

    deinit { llama_model_free(model) }

    /// `<bos> text <eos>`, cut at `maxLength` keeping the end, exactly as `LlamaEmbedder` frames text.
    public func tokenize(_ text: String, maxLength: Int) -> [Int] {
        let utf8 = Array(text.utf8CString)
        var tokens = [llama_token](repeating: 0, count: utf8.count + 8)
        var count = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &tokens, Int32(tokens.count), true, true)
        if count < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-count))
            count = llama_tokenize(vocab, utf8, Int32(utf8.count - 1), &tokens, Int32(tokens.count), true, true)
        }
        tokens.removeLast(tokens.count - Int(max(count, 0)))
        if tokens.count > maxLength {
            tokens = Array(tokens[0..<(maxLength - 1)]) + [llama_vocab_eos(vocab)]
        }
        return tokens.map(Int.init)
    }
}
