import DigUpKit
import Foundation

/// The tokens the model's inputs are framed with, and the framing itself. The ids come out of the GGUF's
/// tokenizer (`GGUFVocab`); these constants frame media and cut text, and the framing test holds them to
/// what llama.cpp's mtmd makes of the same inputs, so a wrong id cannot hide.
enum Framing {
    static let bos = 2
    static let eos = 1
    static let beginImage = 255_999
    static let image = 258_880
    static let endImage = 258_882
    static let beginAudio = 256_000
    static let audio = 258_881
    static let endAudio = 258_883

    /// The reference's cut (transformers' max_length, llama.cpp's context): nothing longer goes in.
    static let maxLength = 2048

    /// `<bos> ids <eos>`, cut the way the reference cuts when over long: the end kept, the middle dropped.
    static func frame(_ ids: [Int]) -> [Int] {
        var tokens = ids
        if tokens.count > maxLength - 2 { tokens = Array(tokens[0..<(maxLength - 2)]) }
        return [bos] + tokens + [eos]
    }

    /// The token ids of a media input's framing, around `count` soft tokens.
    static func mediaIds(begin: Int, end: Int, count: Int) -> [Int] {
        [bos, begin] + Array(repeating: begin == beginImage ? image : audio, count: count) + [end, eos]
    }
}
