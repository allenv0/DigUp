import Foundation

/// Reads a GGUF file: ggml's container for model weights (one header, key/value metadata, a tensor directory,
/// then the tensor data at a fixed alignment).
///
/// DigUp's models are Q8_0 (`ModelFiles`), so that is what gets dequantized; the plain types a file may keep
/// alongside them are read as they are, and anything else is an error that names itself. Metadata is read
/// eagerly (a few hundred keys and tensor entries); tensor bytes are read per tensor.
public struct GGUFFile {
    /// One ggml tensor type this package needs, with its block layout and how it becomes Float32.
    public enum ElementType: UInt32, Sendable {
        case f32 = 0
        case f16 = 1
        case q8_0 = 8
        case bf16 = 30

        /// Bytes per element (Q8_0: a block of 32 values, an f16 scale then 32 int8s).
        var typeSize: Int {
            switch self {
            case .f32: 4
            case .f16: 2
            case .q8_0: 32 * 1 + 2
            case .bf16: 2
            }
        }

        /// Elements per chunk of `typeSize` bytes (1 except Q8_0's blocks of 32).
        var blockSize: Int {
            self == .q8_0 ? 32 : 1
        }

        /// `bytes` (a whole tensor, or any whole number of blocks) as Float32, in ggml's element order.
        func dequantize(_ bytes: Data) -> [Float] {
            switch self {
            case .f32:
                var out = [Float](repeating: 0, count: bytes.count / 4)
                _ = out.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
                return out
            case .f16, .bf16:
                var out = [Float](repeating: 0, count: bytes.count / 2)
                out.withUnsafeMutableBytes { target in
                    let floats = target.bindMemory(to: Float.self)
                    bytes.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                        if self == .f16 {
                            for i in 0..<floats.count {
                                let bits = source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                                floats[i] = Float(Float16(bitPattern: bits))
                            }
                        } else {
                            for i in 0..<floats.count {
                                let bits = source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                                floats[i] = Float(bitPattern: UInt32(bits) << 16)
                            }
                        }
                    }
                }
                return out
            case .q8_0:
                // ggml packs Q8_0 unaligned: an f16 scale, then 32 int8 values, 34 bytes a block.
                let typeSize = Self.q8_0.typeSize
                let blocks = bytes.count / typeSize
                var out = [Float](repeating: 0, count: blocks * 32)
                out.withUnsafeMutableBytes { target in
                    let floats = target.bindMemory(to: Float.self)
                    bytes.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
                        for b in 0..<blocks {
                            let base = b * typeSize
                            let bits = source.loadUnaligned(fromByteOffset: base, as: UInt16.self)
                            let scale = Float(Float16(bitPattern: bits))
                            for i in 0..<32 {
                                let value = source.loadUnaligned(fromByteOffset: base + 2 + i, as: Int8.self)
                                floats[b * 32 + i] = scale * Float(value)
                            }
                        }
                    }
                }
                return out
            }
        }
    }

    public struct Tensor: Sendable {
        public let name: String
        public let type: ElementType
        /// The shape as ggml writes it: element 0 is the fastest-varying axis.
        public let shape: [Int]
        /// Where the data sits in the file.
        public let dataOffset: UInt64

        public var count: Int { shape.reduce(1, *) }
        public var byteCount: Int { count / type.blockSize * type.typeSize }

        public func read() throws -> Data {
            let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
            defer { try? handle.close() }
            try handle.seek(toOffset: dataOffset)
            guard let data = try handle.read(upToCount: byteCount), data.count == byteCount else {
                throw GGUFError.truncated(name)
            }
            return data
        }

        public func floats() throws -> [Float] {
            try type.dequantize(try read())
        }

        /// A row-major Float32 copy in the shape transformers/MLX think in: slowest axis first (ggml's shape
        /// reversed). Only makes sense for 2-D tensors; `gguf-info` prints shapes to check against.
        public func matrix() throws -> (rows: Int, columns: Int, values: [Float]) {
            guard shape.count == 2 else { throw GGUFError.notAMatrix(name) }
            // ggml's [ne0, ne1] is ne0 wide and ne1 tall with ne0 fastest: [columns, rows] to a row-major reader.
            return (shape[1], shape[0], try floats())
        }

        let path: String
    }

    // MARK: Metadata

    /// One metadata value, as the file wrote it.
    public enum Value: Sendable {
        case bool(Bool), uint8(UInt8), int8(Int8), uint16(UInt16), int16(Int16), uint32(UInt32), int32(Int32)
        case float32(Float), uint64(UInt64), int64(Int64), float64(Double), string(String), array([Value])

        public var string: String? {
            if case .string(let string) = self { return string }
            return nil
        }

        public var int: Int? {
            switch self {
            case .bool(let value): return value ? 1 : 0
            case .uint8(let value): return Int(value)
            case .int8(let value): return Int(value)
            case .uint16(let value): return Int(value)
            case .int16(let value): return Int(value)
            case .uint32(let value): return Int(value)
            case .int32(let value): return Int(value)
            case .uint64(let value): return value <= UInt64(Int.max) ? Int(value) : nil
            case .int64(let value): return Int(exactly: value)
            case .float32(let value): return value.rounded() == value ? Int(value) : nil
            case .float64(let value): return value.rounded() == value ? Int(value) : nil
            case .array, .string: return nil
            }
        }

        public var double: Double? {
            switch self {
            case .float32(let value): return Double(value)
            case .float64(let value): return value
            case .int64(let value): return Double(value)
            default: return int.map(Double.init)
            }
        }

        public var array: [Value]? {
            if case .array(let values) = self { return values }
            return nil
        }

        public var strings: [String]? { array?.compactMap(\.string) }

        public var ints: [Int]? { array?.compactMap(\.int) }
    }

    public let metadata: [String: Value]
    public let tensors: [Tensor]

    public func meta(_ key: String) -> Value? { metadata[key] }

    /// The tensor named as ggml's tensor-info table has it (no mapping, no guessing).
    public func tensor(_ name: String) -> Tensor? {
        tensors.first { $0.name == name }
    }

    public var architecture: String? { meta("general.architecture")?.string }

    // MARK: Reading

    public init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.starts(with: Self.magic) else { throw GGUFError.notGGUF(url.lastPathComponent) }
        var reader = Reader(data: data)
        let version = try reader.u32()
        guard version == 2 || version == 3 else { throw GGUFError.version(version) }
        let tensorCount = try reader.u64()
        let metadataCount = try reader.u64()

        var metadata: [String: Value] = [:]
        metadata.reserveCapacity(Int(metadataCount))
        for _ in 0..<metadataCount {
            let key = try reader.string()
            metadata[key] = try reader.value()
        }

        var alignment: UInt64 = 32
        if case .uint32(let value)? = metadata["general.alignment"] { alignment = UInt64(value) }

        struct Info {
            let name: String
            let type: ElementType
            let shape: [Int]
            let offset: UInt64
        }
        var infos: [Info] = []
        infos.reserveCapacity(Int(tensorCount))
        for _ in 0..<tensorCount {
            let name = try reader.string()
            let dimensions = try reader.u32()
            var shape = [Int](repeating: 0, count: Int(dimensions))
            for i in 0..<Int(dimensions) { shape[i] = Int(try reader.u64()) }
            let rawType = try reader.u32()
            guard let type = ElementType(rawValue: rawType) else { throw GGUFError.type(name, rawType) }
            let offset = try reader.u64()
            infos.append(Info(name: name, type: type, shape: shape, offset: offset))
        }
        // The data section starts after header, metadata and tensor infos, rounded up to the alignment.
        let dataStart = Self.aligned(UInt64(reader.offset), to: alignment)
        tensors = infos.map {
            Tensor(name: $0.name, type: $0.type, shape: $0.shape, dataOffset: dataStart + $0.offset, path: path)
        }
        self.metadata = metadata
    }

    static let magic = Data([0x47, 0x47, 0x55, 0x46])   // "GGUF"

    static func aligned(_ offset: UInt64, to alignment: UInt64) -> UInt64 {
        alignment < 2 ? offset : (offset + alignment - 1) / alignment * alignment
    }

    public enum GGUFError: Error, CustomStringConvertible {
        case notGGUF(String), version(UInt32), type(String, UInt32), truncated(String), notAMatrix(String)

        public var description: String {
            switch self {
            case .notGGUF(let name): "\(name) is not a GGUF file"
            case .version(let version): "GGUF version \(version), and this reads versions 2 and 3"
            case .type(let name, let type): "\(name) is of ggml type \(type), which nothing here reads"
            case .truncated(let name): "\(name) ends before its data does"
            case .notAMatrix(let name): "\(name) isn't a 2-D tensor"
            }
        }
    }

    private struct Reader {
        let data: Data
        var offset: Int

        init(data: Data) {
            self.data = data
            offset = GGUFFile.magic.count
        }

        mutating func u32() throws -> UInt32 {
            guard offset + 4 <= data.count else { throw GGUFError.truncated("(header)") }
            defer { offset += 4 }
            return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        }

        mutating func u64() throws -> UInt64 {
            guard offset + 8 <= data.count else { throw GGUFError.truncated("(header)") }
            defer { offset += 8 }
            return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self) }
        }

        mutating func string() throws -> String {
            let length = try u64()
            guard offset + Int(length) <= data.count else { throw GGUFError.truncated("(name)") }
            defer { offset += Int(length) }
            return String(decoding: data[offset..<(offset + Int(length))], as: UTF8.self)
        }

        mutating func value() throws -> Value {
            let raw = try u32()
            guard let kind = Kind(rawValue: raw) else { throw GGUFError.truncated("(value type \(raw))") }
            return try value(kind)
        }

        mutating func value(_ kind: Kind) throws -> Value {
            func bytes(_ count: Int) throws -> Data {
                guard offset + count <= data.count else { throw GGUFError.truncated("(metadata)") }
                defer { offset += count }
                return data.subdata(in: offset..<(offset + count))
            }
            func scalar<Element>(_ type: Element.Type) throws -> Element {
                try bytes(MemoryLayout<Element>.size).withUnsafeBytes { $0.loadUnaligned(as: type) }
            }
            switch kind {
            case .uint8: return .uint8(try scalar(UInt8.self))
            case .int8: return .int8(try scalar(Int8.self))
            case .uint16: return .uint16(try scalar(UInt16.self))
            case .int16: return .int16(try scalar(Int16.self))
            case .uint32: return .uint32(try scalar(UInt32.self))
            case .int32: return .int32(try scalar(Int32.self))
            case .float32: return .float32(try scalar(Float.self))
            case .bool: return .bool(try scalar(UInt8.self) != 0)
            case .string: return .string(try string())
            case .uint64: return .uint64(try scalar(UInt64.self))
            case .int64: return .int64(try scalar(Int64.self))
            case .float64: return .float64(try scalar(Double.self))
            case .array:
                let raw = try u32()
                guard let elementKind = Kind(rawValue: raw) else { throw GGUFError.truncated("(array type \(raw))") }
                let count = try u64()
                var values = [Value]()
                values.reserveCapacity(Int(count))
                for _ in 0..<count { values.append(try value(elementKind)) }
                return .array(values)
            }
        }

        fileprivate enum Kind: UInt32 {
            case uint8 = 0, int8, uint16, int16, uint32, int32, float32, bool, string, array, uint64, int64, float64
        }
    }
}
