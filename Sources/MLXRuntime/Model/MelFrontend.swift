import DigUpKit
import Foundation
import MLX

/// Reads the 16 kHz mono WAV pieces the indexer writes, without resampling them.
enum AudioFile {
    /// The Float samples in a PCM WAV (`AudioExtractor.writeWAV`): mono 16 kHz, 16-bit or Float32.
    static func readMono16K(path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard data.count >= 44, data[0..<4] == Data("RIFF".utf8), data[8..<12] == Data("WAVE".utf8) else {
            throw WorkerError.failed("not a WAV file: \(path)")
        }

        var format: UInt16?
        var channels: UInt16?
        var rate: UInt32?
        var bits: UInt16?
        var samples: [Float]?
        var offset = 12
        while offset + 8 <= data.count {
            let chunk = data[offset..<offset + 4]
            let size = Int(data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self) })
            let body = offset + 8
            guard body + size <= data.count else { throw WorkerError.failed("a truncated WAV chunk: \(path)") }
            if chunk == Data("fmt ".utf8) {
                guard size >= 16 else { throw WorkerError.failed("a bad WAV format chunk: \(path)") }
                format = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: body, as: UInt16.self) }
                channels = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: body + 2, as: UInt16.self) }
                rate = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: body + 4, as: UInt32.self) }
                bits = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: body + 14, as: UInt16.self) }
            } else if chunk == Data("data".utf8) {
                guard let format, let channels, let rate, let bits else {
                    throw WorkerError.failed("a WAV data chunk before its format: \(path)")
                }
                guard format == 1 || format == 3, channels == 1, rate == 16_000, bits == 16 || bits == 32 else {
                    throw WorkerError.failed("audio must be mono 16 kHz, 16- or 32-bit: \(path)")
                }
                let bodyData = data[body..<(body + size)]
                if format == 1 {
                    guard bits == 16 else { throw WorkerError.failed("unsupported WAV integer depth: \(path)") }
                    samples = bodyData.withUnsafeBytes { raw in
                        let words = raw.bindMemory(to: Int16.self)
                        return words.map { Float($0) / 32768 }
                    }
                } else {
                    guard bits == 32 else { throw WorkerError.failed("unsupported WAV float depth: \(path)") }
                    samples = bodyData.withUnsafeBytes { raw in
                        Array(raw.bindMemory(to: Float.self))
                    }
                }
            }
            offset = body + size + (size % 2)
        }
        guard let samples else { throw WorkerError.failed("a WAV file with no samples: \(path)") }
        return samples
    }
}

/// EmbeddingGemma 2's log-mel frontend: the Hugging Face feature extractor's USM pipeline, with the model's
/// pinned parameters (20 ms frames, 10 ms hops, 512-point FFTs, 128 HTK mel bins, a 0.001 floor, natural logs).
///
/// The first frame is centered at sample zero (160 leading zeros), frames are 321 samples unfolded every 160,
/// only the first 320 enter each Hann-windowed FFT, and nothing is normalized per bin. This exactly matches
/// `Gemma4AudioFeatureExtractor` and therefore the values llama.cpp's audio path encodes.
final class MelFrontend {
    static let sampleRate = 16_000
    static let frameLength = 320
    static let hopLength = 160
    static let fftLength = 512
    static let melCount = 128
    static let padLeft = 160
    static let unfoldLength = 321
    static let floor: Float = 0.001

    private let hann: MLXArray
    private let filters: MLXArray   // [257, 128]

    init() {
        hann = MLXArray((0..<Self.frameLength).map { index in
            0.5 - 0.5 * Foundation.cos(2 * Float.pi * Float(index) / Float(Self.frameLength))
        })
        filters = MLXArray(Self.filterBank(), [Self.fftLength / 2 + 1, Self.melCount]).asType(.float32)
    }

    /// Log-mel frames for up to 30 s of samples, in Float32: `[frames, 128]`.
    func callAsFunction(_ samples: [Float]) throws -> MLXArray {
        guard samples.count <= 30 * Self.sampleRate else {
            throw WorkerError.failed("audio past 30 s goes in as another window, not one input")
        }
        let padded = [Float](repeating: 0, count: Self.padLeft) + samples
        guard padded.count >= Self.unfoldLength else { throw WorkerError.failed("audio too short to frame") }
        let frames = (padded.count - Self.unfoldLength) / Self.hopLength + 1
        var windowed = [Float](repeating: 0, count: frames * Self.frameLength)
        for frame in 0..<frames {
            let start = frame * Self.hopLength
            for index in 0..<Self.frameLength { windowed[frame * Self.frameLength + index] = padded[start + index] }
        }
        let matrix = MLXArray(windowed, [frames, Self.frameLength]).asType(.float32) * hann
        let spectrum = MLX.rfft(matrix, n: Self.fftLength, axis: -1)
        let magnitude = MLX.abs(spectrum)
        let mel = MLX.matmul(magnitude, filters)
        return MLX.log(mel + Self.floor)
    }

    /// Transformers' `mel_filter_bank` for this model: HTK mel points, triangular filters in Hz, no Slaney
    /// normalization. Computed in Double, exactly as its NumPy code does.
    static func filterBank() -> [Float] {
        let bins = fftLength / 2 + 1
        let hz = (0..<bins).map { 8000.0 * Double($0) / Double(bins - 1) }
        let melMax = 2595 * Foundation.log10(1 + 8000.0 / 700.0)
        let melEdges = (0...(melCount + 1)).map {
            700 * (Foundation.pow(10, Double($0) * melMax / Double(melCount + 1) / 2595) - 1)
        }
        var filters = [Float](repeating: 0, count: bins * melCount)
        for bin in 0..<bins {
            for mel in 0..<melCount {
                let lower = melEdges[mel], center = melEdges[mel + 1], upper = melEdges[mel + 2]
                let rising = (hz[bin] - lower) / max(center - lower, 1e-10)
                let falling = (upper - hz[bin]) / max(upper - center, 1e-10)
                filters[bin * melCount + mel] = Float(max(0, min(rising, falling)))
            }
        }
        return filters
    }
}
