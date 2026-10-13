import DigUpKit
import Foundation
import Testing
@testable import MLXRuntime

/// The audio frontend's pinned constants and filterbank, held to the Python reference's values.
/// The full mel spectrogram for `speech.wav` is already pinned end-to-end by `audioMatchesTheReference`
/// (cosine ≥ 0.999); these lock the filterbank itself so a regression points at the right layer.
@Suite struct MelFrontendTests {
    /// First mel filter's first five taps, from the validated Python reference
    /// (`Gemma4AudioFeatureExtractor`, HTK, 128 bins, 512-point FFT). Computed in Double exactly as
    /// transformers does; Float rounding must stay within 1e-6.
    @Test func filterBankMatchesReference() {
        let bank = MelFrontend.filterBank()
        #expect(bank.count == 257 * 128)
        // Bin 0 is below the first filter's lower edge: all zeros (this is the warning transformers prints).
        #expect(bank[0..<128].allSatisfy { $0 == 0 })
        // Bin 1, filters 0–4: from transformers' `mel_filter_bank` (HTK, triangular in Hz, no Slaney norm).
        let expected: [Float] = [0, 0.766, 0.234, 0, 0]
        for (index, value) in expected.enumerated() {
            #expect(abs(bank[1 * 128 + index] - value) < 1e-3, "bin 1 filter \(index): \(bank[1 * 128 + index])")
        }
        // Triangular filters sampled at discrete FFT bins peak just under 1.0 (mel spacing isn't bin-aligned),
        // and the top filters run past Nyquist — so values stay in [0, 1], most filters peak above 0.9,
        // and filter 0 is entirely zero.
        #expect(bank.allSatisfy { $0 >= 0 && $0 <= 1 })
        let peaks = (0..<128).map { mel in (0..<257).map { bank[$0 * 128 + mel] }.max() ?? 0 }
        #expect(peaks.filter { $0 > 0.9 }.count >= 40, "only \(peaks.filter { $0 > 0.9 }.count) filters peak above 0.9")
        #expect(abs((peaks.max() ?? 0) - 0.99) < 0.02, "max peak \(peaks.max() ?? 0)")
    }

    @Test func constantsMatchPinnedConfig() {
        #expect(MelFrontend.sampleRate == 16_000)
        #expect(MelFrontend.frameLength == 320)
        #expect(MelFrontend.hopLength == 160)
        #expect(MelFrontend.fftLength == 512)
        #expect(MelFrontend.melCount == 128)
        #expect(MelFrontend.floor == 0.001)
    }

    /// 1 s of silence frames to 99 log-mel rows (160 leading zeros, 321-sample unfold every 160).
    @Test func framesSilenceToFloor() throws {
        let silence = [Float](repeating: 0, count: 16_000)
        let mel = try MelFrontend()(silence)
        #expect(mel.dim(0) == 99)
        #expect(mel.dim(1) == 128)
        // Silence through the floor: log(0 + 0.001) everywhere.
        let first = mel[0].asArray(Float.self)
        #expect(first.allSatisfy { abs($0 - log(0.001)) < 1e-4 })
    }
}

/// The GGUF→safetensors name mapping has no rename table by design (module trees are keyed to ggml
/// names), except the two documented cases: linear weights keep `.weight`, raw parameters drop it.
@Suite struct ModelConvertTests {
    @Test func weightNamesFollowModulePaths() {
        #expect(ModelConvert.weightName("blk.0.attn_q.weight") == "blk.0.attn_q.weight")
        #expect(ModelConvert.weightName("blk.0.attn_norm.weight") == "blk.0.attn_norm")
        #expect(ModelConvert.weightName("token_embd.weight") == "token_embd")
        #expect(ModelConvert.weightName("output_norm.weight") == "output_norm")
        #expect(ModelConvert.weightName("mm.input_projection.weight") == "mm.input_projection.weight")
        #expect(ModelConvert.weightName("a.pre_encode.out.weight") == "a.pre_encode.out.weight")
    }

    @Test func cacheIdentityMatchesApp() {
        #expect(MLXCache.recipe == ModelConvert.recipe)
        #expect(MLXCache.folderName == MLXModels.cacheName)
    }
}
