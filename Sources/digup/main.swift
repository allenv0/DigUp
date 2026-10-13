import DigUpKit
import Foundation
import LlamaRuntime
#if canImport(MLXRuntime)
import MLXRuntime
#endif

// DigUp's developer CLI. Every command works on an explicit index folder, so tests never touch the real one.

#if !os(Windows)
signal(SIGPIPE, SIG_IGN)   // a dead worker must surface as an error, not kill us (Windows has no SIGPIPE)
#endif

let usage = """
    digup: search your files by meaning (DigUp developer CLI)

      digup estimate <folder>... [--json]     what would be indexed, and how long the first pass takes
      digup index <folder>... [--limit N]     sync the index with these folders, embed what's new, then the rest
                                              of long PDFs and documents
      digup search <words>... [--kind K,K] [-n 10] [--json]
                                              "code: <words>" searches the code index
      digup status                            what's in the index
      digup speed [--backend B] [--json]     this machine's speed next to the M4 Pro behind the estimates, what
                                              indexing has measured here, and the engine's embedding throughput
      digup eval <queries.json> [-n 10]       recall of known answers (see evals/; "code: …" queries search code)
      digup index-service <folder>... [--code-root DIR]... [--code-exclude DIR]... [--backfill] [--prune]
                          [--memory-limit MB] [--time-limit S]
                                              the app's indexing helper (JSON lines; see IndexService); it stops
                                              on a file whose reading passes the limits (see Watchdog)
      digup worker [--text-only]              the model as a process (JSON lines; the app's query encoder)
      digup search-service <folder>... [--code-root DIR]... [--encoder-idle S]
                                              the searcher as a process, index open and encoder warm (JSON lines;
                                              see SearchService): for shells that can't link DigUpKit
      digup download-model [--to DIR]         the model files, from the Hugging Face cache or downloaded with
                                              resume, checked (JSON lines: see DownloadModel)
      digup convert-mlx [--reconvert]         the MLX engine's weights, from the GGUFs (done by itself on the
                                              first MLX run; --reconvert remakes the cache)
      digup gguf-info <file>... [--tensors]   what's in a GGUF: its metadata (and tensors), as the runtimes
                                              read it (for checking a model update)

    Options:
      --backend B          which engine embeds: automatic (default), llama or mlx (macOS only)
      --index DIR          index folder (default: DigUp's Application Support folder; see `digup status`)
      --code               estimate, index or show the code index of these folders (code.sqlite in the index
                           folder): source files, their repos' docs and notebooks, minus what git ignores
      --kind K,K           screenshot, image, pdf, doc, audio, video
      --include-code       also index loose source code and data files as documents (outside code projects)
      --include-datasets   don't skip folders that look like datasets
      --exclude DIR        don't index this folder (repeatable)
      --exclude-ext EXT    don't index files with this extension (repeatable, or comma-separated: "heic,txt")
      --reindex            drop vectors and embed everything again (after changing settings or model)
      --zgate Z            how far a meaning match must stand out (default 2.0)
      --word-weight W      bonus for containing all of the query's words (default 2.0, and for code see Searcher)
      --shot-tokens N      vision tokens per screenshot (default 280; 560 was 2× slower, no better on real shots)
      --code-stretch N     characters a stretch of code holds at most (default 1800; 1200 and 3600 lost on the evals)

    The model: EmbeddingGemma 2, from $DIGUP_MODELS (only there, when set), else
    ~/Library/Application Support/DigUp/Models or the Hugging Face cache. The engine is llama.cpp unless
    --backend says otherwise ($DIGUP_BACKEND too): mlx runs EmbeddingGemma 2 on mlx-swift from weights
    converted once out of the same GGUFs. DIGUP_WORKER_DIR=worker uses the mlx-vlm reference worker instead
    (evals).
    """

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct Arguments {
    var positional: [String] = []
    var values: [String: String] = [:]
    var lists: [String: [String]] = [:]
    var flags: Set<String> = []
    static let valued: Set<String> = ["--index", "--limit", "--kind", "-n", "--zgate", "--word-weight", "--new-limit",
                                      "--linger", "--shot-tokens", "--code-stretch", "--encoder-idle", "--to",
                                      "--backend", "--memory-limit", "--time-limit"]
    static let repeated: Set<String> = ["--exclude", "--exclude-ext", "--code-root", "--code-exclude"]

    init(_ raw: ArraySlice<String>) throws {
        var index = raw.startIndex
        while index < raw.endIndex {
            let argument = raw[index]
            if Self.valued.contains(argument) || Self.repeated.contains(argument) {
                guard index + 1 < raw.endIndex else { throw CLIError("\(argument) needs a value") }
                if Self.repeated.contains(argument) {
                    lists[argument, default: []].append(raw[index + 1])
                } else {
                    values[argument] = raw[index + 1]
                }
                index += 2
            } else if argument.hasPrefix("--") || argument == "-h" {
                flags.insert(argument)
                index += 1
            } else {
                positional.append(argument)
                index += 1
            }
        }
    }

    var indexFolder: URL {
        guard let chosen = values["--index"] else { return AppPaths.defaultIndex.standardizedFileURL }
        return URL(fileURLWithPath: (chosen as NSString).expandingTildeInPath).standardizedFileURL
    }

    /// `--code`: the code index, not the main one.
    var code: Bool { flags.contains("--code") }

    func store() throws -> IndexStore {
        try IndexStore(directory: indexFolder, file: code ? IndexStore.codeFile : IndexStore.mainFile)
    }

    var options: IndexOptions {
        var options = IndexOptions()
        options.code = code
        options.includeCode = flags.contains("--include-code")
        options.includeDatasets = flags.contains("--include-datasets")
        options.excludedFolders = (lists["--exclude"] ?? []).map { Self.folderURL($0).path }
        options.excludedExtensions = IndexOptions.extensions((lists["--exclude-ext"] ?? []).joined(separator: ","))
        if let tokens = values["--shot-tokens"].flatMap(Int.init) { options.screenshotBudget = tokens }
        if let characters = values["--code-stretch"].flatMap(Int.init) { options.codeStretch = characters }
        return options
    }

    func folders() throws -> [URL] {
        guard !positional.isEmpty else { throw CLIError("which folders? e.g. digup index ~/DigUpTestbed") }
        return try positional.map {
            let url = Self.folderURL($0)
            guard Self.isFolder(url) else { throw CLIError("not a folder: \($0)") }
            guard Self.isReadable(url) else { throw CLIError("can't read \($0) (macOS privacy settings?)") }
            return url
        }
    }

    static func folderURL(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
    }

    static func isFolder(_ url: URL) -> Bool {
        var isFolder: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder) && isFolder.boolValue
    }

    /// The folder can be listed. One that macOS won't let us read (Files and Folders privacy) looks empty to a crawl,
    /// which must never be mistaken for "everything in it was deleted". (The first look may bring up macOS's prompt.)
    static func isReadable(_ url: URL) -> Bool {
        (try? FileManager.default.contentsOfDirectory(atPath: url.path)) != nil
    }

    /// Dev knobs for A/B runs in evals/.
    var zGate: Double { values["--zgate"].flatMap(Double.init) ?? 2.0 }
    var wordWeight: Double? { values["--word-weight"].flatMap(Double.init) }   // nil: the index's own (`Searcher`)

    func kinds() throws -> Set<FileKind>? {
        guard let list = values["--kind"] else { return nil }
        return Set(try list.split(separator: ",").map {
            guard let kind = FileKind(rawValue: $0.trimmingCharacters(in: .whitespaces)) else {
                throw CLIError("unknown kind \($0); use screenshot, image, pdf, doc, audio, video")
            }
            return kind
        })
    }

    /// `--backend mlx`: which engine embeds (default: $DIGUP_BACKEND, else automatic).
    func backend() throws -> EmbeddingBackend {
        let chosen = values["--backend"] ?? ProcessInfo.processInfo.environment["DIGUP_BACKEND"]
        guard let chosen, !chosen.isEmpty else { return .automatic }
        guard let backend = EmbeddingBackend(rawValue: chosen.lowercased()) else {
            throw CLIError("unknown backend \(chosen); use automatic, llama or mlx")
        }
        return backend
    }
}

// MARK: Formatting

let home = FileManager.default.homeDirectoryForCurrentUser.path
func tilde(_ path: String) -> String { path.hasPrefix(home) ? "~" + path.dropFirst(home.count) : path }
func pad(_ text: String, _ width: Int) -> String { text.count >= width ? text : text + String(repeating: " ", count: width - text.count) }
func lpad(_ text: String, _ width: Int) -> String { text.count >= width ? text : String(repeating: " ", count: width - text.count) + text }
func number(_ value: Int) -> String { value.formatted(.number) }

func duration(_ seconds: Double) -> String {
    switch seconds {
    case ..<90: "\(Int(seconds.rounded())) s"
    case ..<5400: "\(Int((seconds / 60).rounded())) min"
    default: String(format: "%.1f h", seconds / 3600)
    }
}

func clock(_ seconds: Double) -> String {
    let total = Int(seconds)
    return total >= 3600 ? String(format: "%d:%02d:%02d", total / 3600, total / 60 % 60, total % 60)
                         : String(format: "%d:%02d", total / 60, total % 60)
}

func location(_ hit: SearchHit) -> String {
    switch hit.segment {
    case .page, .pageImage: hit.loc.map { "p. \(Int($0))" } ?? ""
    case .audio, .frame: hit.loc.map { "at \(clock($0))" + (hit.segment == .audio ? " (sound)" : "") } ?? ""
    case .chunk: hit.loc.map { "part \(Int($0))" } ?? ""
    case .lines: hit.loc.map { "lines \(Int($0))–\(Int(hit.locEnd ?? $0))" + (hit.focus.map { ", opens at \(Int($0))" } ?? "") } ?? ""
    case .cells: hit.loc.map { $0 == hit.locEnd ? "cell \(Int($0))" : "cells \(Int($0))–\(Int(hit.locEnd ?? $0))" } ?? ""
    case .ocr: "text in image"
    default: ""
    }
}

func skippedLine(_ skipped: [String: Int]) -> String {
    skipped.sorted { $0.value > $1.value }.map { "\(number($0.value)) \($0.key)" }.joined(separator: " · ")
}

// MARK: Commands

/// One line into a helper's log, or stderr when there is none (the fallback notices go where the engine's logs
/// are, which the app's engine reads).
func note(_ log: URL?, _ line: String) {
    guard let log else {
        FileHandle.standardError.write(Data("[digup] \(line)\n".utf8))
        return
    }
    FileManager.default.createFile(atPath: log.path, contents: nil)
    if let handle = try? FileHandle(forWritingTo: log) {
        try? handle.seekToEnd()
        try? handle.write(contentsOf: Data("[digup] \(line)\n".utf8))
        try? handle.close()
    }
}

/// The embedding runtime: llama.cpp in this process, MLX beside it (`--backend mlx`, weights converted once
/// out of the same GGUFs), or the worker in $DIGUP_WORKER_DIR (the mlx-vlm reference). An engine that can't
/// start falls back to llama.cpp, logged — unless it was asked for by name.
func makeEmbedder(textOnly: Bool, backend: EmbeddingBackend = .automatic, log: URL?,
                  qos: QualityOfService = .default) throws -> any Embedder {
    if var config = try WorkerConfig.fromEnvironment() {
        config.log = log
        config.qos = qos
        return try EmbeddingWorker(config, textOnly: textOnly)
    }
    switch backend.resolved {
    case .mlx:
        do {
            #if canImport(MLXRuntime)
            return try MLXEmbedder(models: try MLXModels.locate(), textOnly: textOnly, log: log)
            #else
            throw WorkerError.failed("MLX needs macOS")
            #endif
        } catch {
            guard backend == .automatic else { throw error }
            note(log, "mlx can't start (\(String("\(error)".prefix(300)))); llama.cpp it is")
            return try LlamaEmbedder(models: try LlamaModels.locate(), textOnly: textOnly, log: log)
        }
    case .llama, .automatic:
        return try LlamaEmbedder(models: try LlamaModels.locate(), textOnly: textOnly, log: log)
    }
}

/// `makeEmbedder`'s vector namespace when it's known without starting anything (`WorkerInfo.compat`): llama.cpp's
/// and MLX's are. The Python worker's isn't — its `vector_source` arrives with its ready line.
func embedderCompat(backend: EmbeddingBackend) -> String? {
    if ProcessInfo.processInfo.environment["DIGUP_WORKER_DIR"] != nil { return nil }
    switch backend.resolved {
    case .mlx:
        #if canImport(MLXRuntime)
        return MLXEmbedder.modelInfo.compat
        #else
        return nil
        #endif
    case .llama, .automatic: return LlamaEmbedder.modelInfo.compat
    }
}

/// Whether the chosen engine's whole model (text and media) is ready for indexing, checked without loading it.
func modelComplete(backend: EmbeddingBackend) -> Bool {
    switch backend.resolved {
    case .mlx:
        #if canImport(MLXRuntime)
        return MLXModels.complete
        #else
        return false
        #endif
    case .llama, .automatic: return LlamaModels.complete
    }
}

func estimate(_ args: Arguments) throws {
    guard !args.positional.isEmpty else { throw CLIError("which folders? e.g. digup estimate ~/Documents") }
    let options = args.options
    // What indexing costs on this machine: as measured by indexing here, kind by kind, once there's enough of it;
    // until then the M4 Pro's fixed costs, scaled by a second's speed check (done once and kept beside the index).
    var measured = MeasuredCosts.load(from: args.indexFolder)
    if measured.speed == nil, let speed = MeasuredCosts.Speed.measure() {
        measured.speed = speed
        MeasuredCosts.setSpeed(speed, in: args.indexFolder)
    }
    let costs = measured.costs
    var total = 0.0
    for folder in args.positional.map(Arguments.folderURL) {
        // One folder that isn't there or can't be read (a denied privacy prompt) mustn't sink the others.
        guard Arguments.isFolder(folder), Arguments.isReadable(folder) else {
            if args.flags.contains("--json") {
                emit(["folder": folder.path, "readable": false, "seconds": 0, "files": [String: Int](),
                      "pdf_pages": 0, "audio_seconds": 0, "video_seconds": 0])
            } else {
                print("\(tilde(folder.path))    can't read it (not there, or not allowed in macOS privacy settings)")
            }
            continue
        }
        let started = Date()
        let crawl = Crawler.crawl([folder], options: options)
        let parts = Estimator.estimates(bySubfolder: crawl.candidates, of: folder.path, options: options, costs: costs)
        let estimate = parts.values.reduce(Estimate(), +)
        // Biggest first; the app lists them under the folder, so that a long first pass can be cut down.
        let subfolders = parts.filter { $0.key != folder.path }.sorted {
            $0.value.seconds != $1.value.seconds ? $0.value.seconds > $1.value.seconds : $0.key < $1.key
        }
        total += estimate.seconds
        if args.flags.contains("--json") {
            func files(_ estimate: Estimate) -> [String: Int] {
                Dictionary(uniqueKeysWithValues: estimate.files.map { ($0.key.rawValue, $0.value) })
            }
            // One line per folder, as soon as it's done (the app's onboarding shows them as they come).
            emit(["folder": folder.path, "seconds": estimate.seconds, "readable": true, "files": files(estimate),
                  "pdf_pages": estimate.pdfPages, "pdf_pages_total": estimate.pdfPagesTotal,
                  "audio_seconds": estimate.audioSeconds, "video_seconds": estimate.videoSeconds,
                  "long_files": estimate.longFiles, "later_seconds": estimate.laterSeconds,
                  "code_stretches": estimate.codeStretches, "bytes": estimate.bytes,
                  "subfolders": subfolders.prefix(500).map {
                      ["folder": $0.key, "seconds": $0.value.seconds, "files": files($0.value),
                       "long_files": $0.value.longFiles, "later_seconds": $0.value.laterSeconds,
                       "bytes": $0.value.bytes] as [String: Any]
                  },
                  "skipped": crawl.skipped.values.reduce(0, +), "datasets": crawl.datasets.count,
                  "ms": Int(Date().timeIntervalSince(started) * 1000)])
            continue
        }
        var kinds: [String] = []
        for kind in FileKind.allCases {
            guard let count = estimate.files[kind], count > 0 else { continue }
            switch kind {
            case .pdf: kinds.append("\(number(count)) PDFs (\(number(estimate.pdfPagesTotal)) pages; \(number(estimate.pdfPages)) in the first pass)")
            case .audio: kinds.append("\(number(count)) audio (\(duration(estimate.audioSeconds)))")
            case .video: kinds.append("\(number(count)) video (\(duration(estimate.videoSeconds)))")
            case .code: kinds.append("\(number(count)) code \(count == 1 ? "file" : "files") "
                                     + "(\(number(estimate.codeStretches)) stretches)")
            default: kinds.append("\(number(count)) \(kind.rawValue)\(count == 1 ? "" : "s")")
            }
        }
        print("\(tilde(folder.path))    first index ≈ \(duration(estimate.seconds))"
              + (args.code ? ", index ≈ \(ByteCountFormatter.string(fromByteCount: Int64(estimate.bytes), countStyle: .file))" : ""))
        print("  " + (kinds.isEmpty ? "nothing to index" : kinds.joined(separator: " · ")))
        if estimate.longFiles > 0 {
            print("  then the rest of \(number(estimate.longFiles)) long \(estimate.longFiles == 1 ? "file" : "files") "
                  + "≈ \(duration(estimate.laterSeconds))")
        }
        if !subfolders.isEmpty {
            print("  biggest subfolders: " + subfolders.prefix(5).map {
                "\(($0.key as NSString).lastPathComponent) ≈ \(duration($0.value.seconds))"
            }.joined(separator: " · ") + (subfolders.count > 5 ? " · \(subfolders.count - 5) more" : ""))
        }
        if !crawl.skipped.isEmpty { print("  skipped: " + skippedLine(crawl.skipped)) }
        if !crawl.skippedFolders.isEmpty { print("  folders not walked: " + skippedLine(crawl.skippedFolders)) }
        for (folder, count) in crawl.datasets.sorted(by: { $0.value > $1.value }) {
            print("  looks like a dataset (skipped): \(tilde(folder)) (\(number(count)) files)")
        }
        print(String(format: "  (looked at it in %.2f s)", Date().timeIntervalSince(started)))
    }
    if args.positional.count > 1, !args.flags.contains("--json") { print("all together ≈ \(duration(total))") }
}

func index(_ args: Arguments) throws {
    let backend = try args.backend()
    let store = try args.store()
    if args.flags.contains("--reindex") { try store.resetAll() }
    let indexer = try Indexer(store: store, options: args.options, makeEmbedder: {
        try makeEmbedder(textOnly: false, backend: backend, log: store.directory.appendingPathComponent("worker.log"))
    })
    indexer.embedderCompat = embedderCompat(backend: backend)
    indexer.costsFolder = store.directory
    try store.readAgainOnce()   // what older indexes couldn't read: scanned pages' words, SVGs
    try indexer.checkModel()
    let folders = try args.folders()
    let sync = try indexer.sync(folders)
    print("index: \(tilde(store.file.path))")
    print("sync: \(sync.added) new · \(sync.changed) changed · \(sync.moved) moved · \(sync.removed) removed · \(sync.unchanged) unchanged")
    if !sync.crawl.skipped.isEmpty { print("skipped: " + skippedLine(sync.crawl.skipped)) }
    for (folder, count) in sync.crawl.datasets { print("looks like a dataset (skipped): \(tilde(folder)) (\(number(count)) files)") }

    let limit = args.values["--limit"].flatMap(Int.init)
    var started = Date()
    let states = try indexer.run(limit: limit, onEvent: { event in
        let name = (event.path as NSString).lastPathComponent
        let note = event.note.map { "  (\($0))" } ?? ""
        print("[\(lpad("\(event.done)", 4))/\(event.total)] \(pad(event.kind.rawValue, 10)) \(pad(event.state, 7)) "
              + String(format: "%5.2fs  ", event.seconds) + name + note)
    })
    if states.isEmpty {
        print("nothing pending")
    } else {
        print("done in \(duration(Date().timeIntervalSince(started))): " + statesLine(states))
    }
    // The rest of long files, as the app does once nothing is pending (not after a --limit run: things still are).
    guard limit == nil, try store.unfinishedCount().files > 0 else { return }
    started = Date()
    let finished = try indexer.finishLongFiles(onStep: { step in
        let name = (step.path as NSString).lastPathComponent
        let note = step.note.map { "  (\($0))" } ?? ""
        print("[rest \(lpad("\(step.done)", 5))/\(step.total)] \(pad(step.kind.rawValue, 4)) \(pad(step.state, 7)) "
              + String(format: "%5.2fs  ", step.seconds) + name + note)
    })
    print("the rest of long files in \(duration(Date().timeIntervalSince(started))): "
          + (finished.isEmpty ? "none finished" : statesLine(finished)))
}

func statesLine(_ states: [String: Int]) -> String {
    states.sorted { $0.key < $1.key }.map { "\($0.value) \($0.key)" }.joined(separator: " · ")
}

/// Embeds queries with the text model alone (0.3B params, ~0.25 GB): the query encoder. `prompt`: the search's
/// (`Prompts.query`, or `Prompts.codeQuery` for the code index).
func queryVectors(_ queries: [String], store: IndexStore, searcher: Searcher, backend: EmbeddingBackend = .automatic,
                  prompt: (String) -> String = Prompts.query) throws -> [[Float]]? {
    guard searcher.vectorCount > 0, !queries.isEmpty else { return nil }
    let worker = try makeEmbedder(textOnly: true, backend: backend,
                                  log: store.directory.appendingPathComponent("worker-query.log"))
    defer { worker.close() }
    if let indexed = searcher.vectorCompat, indexed != worker.info.compat {
        throw CLIError("index vectors live in \(indexed) but the query encoder makes \(worker.info.compat)")
    }
    return try worker.embedTexts(queries.map(prompt))
}

func search(_ args: Arguments) throws {
    var query = args.positional.joined(separator: " ")
    guard !query.isEmpty else { throw CLIError("search for what?") }
    let code = CodeQuery.text(of: query)
    if let code { query = code }
    guard !query.isEmpty else { throw CLIError("search the code for what?") }
    let store = try IndexStore(directory: args.indexFolder, file: code == nil ? IndexStore.mainFile : IndexStore.codeFile)
    let searcher = try Searcher(store: store)
    let started = Date()
    let vector = try queryVectors([query], store: store, searcher: searcher, backend: try args.backend(),
                                  prompt: code == nil ? Prompts.query : Prompts.codeQuery)?.first
    let encoded = Date()
    let hits = try searcher.search(query, queryVector: vector, kinds: try args.kinds(),
                                   limit: args.values["-n"].flatMap(Int.init) ?? 10, zGate: args.zGate,
                                   wordWeight: args.wordWeight)
    let searched = Date()
    if args.flags.contains("--json") {
        let rows = hits.map(hitFields)
        let data = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
        print(String(data: data, encoding: .utf8) ?? "[]")
        return
    }
    if hits.isEmpty { print("no good matches") }
    for (rank, hit) in hits.enumerated() {
        let why = [hit.z.map { String(format: "z %.1f", $0) }, hit.keywordRank.map { "words #\($0)" }]
            .compactMap { $0 }.joined(separator: ", ")
        let place = location(hit)
        print("\(lpad("\(rank + 1)", 2)). \(pad(hit.kind.rawValue, 10)) \((hit.path as NSString).lastPathComponent)"
              + (place.isEmpty ? "" : "  [\(place)]") + "   (\(why))")
        print("      \(tilde((hit.path as NSString).deletingLastPathComponent))")
        if let excerpt = hit.excerpt, !excerpt.isEmpty { print("      “\(excerpt.prefix(160))”") }
    }
    print(String(format: "(%d vectors · query encoder %.2f s incl. start-up · search %.0f ms)", searcher.vectorCount,
                 encoded.timeIntervalSince(started), searched.timeIntervalSince(encoded) * 1000))
}

func status(_ args: Arguments) throws {
    let store = try args.store()
    let counts = try store.counts()
    print("index: \(tilde(store.file.path))")
    print("vectors: \(number(counts.vectors)) in \(number(counts.segments)) segments")
    for kind in FileKind.allCases {
        guard let states = counts.byKindAndState[kind.rawValue] else { continue }
        let total = states.values.reduce(0, +)
        print("  \(pad(kind.rawValue, 11)) \(lpad(number(total), 6))   " + statesLine(states))
    }
    if counts.unfinished > 0 {
        print("long files still being read: \(number(counts.unfinished)) (\(number(counts.left)) pages and chunks left)")
    }
    if let fingerprint = try store.meta("fingerprint") { print("built with: \(fingerprint)") }
    if let source = try store.meta("vector_source") {
        print("vectors made by: \(source)" + ((try store.meta("vector_compat")).map { " (namespace \($0))" } ?? ""))
    }
    let backend = try args.backend()
    print("engine: \(engineLine(backend))")
    print(mlxCacheLine())
    print("size: \(ByteCountFormatter.string(fromByteCount: IndexStore.size(of: store.file), countStyle: .file))")
}

/// What `status` says about the MLX weight cache: ready (with its size), convertible, or missing.
/// Uses the shared `MLXCache` identity so the CLI and Settings can't disagree about the folder name.
func mlxCacheLine() -> String {
    #if canImport(MLXRuntime)
    let folders = ModelFiles.modelFolders
    guard let folder = folders.first(where: {
        FileManager.default.fileExists(atPath: $0.appendingPathComponent(ModelFiles.text.name).path)
    }) else { return "mlx weights: no model files (digup download-model first)" }
    switch MLXCache.state(modelsDirectory: folder) {
    case .ready(let cache):
        let size = MLXCache.bytes(modelsDirectory: folder)
        return "mlx weights: ready (\(ByteCountFormatter.string(fromByteCount: size, countStyle: .file)) in \(cache.lastPathComponent))"
    case .convertible:
        return "mlx weights: to convert on first MLX use (digup convert-mlx does it now)"
    case .missing:
        return "mlx weights: no model files (digup download-model first)"
    }
    #else
    return "mlx weights: unavailable off macOS"
    #endif
}

/// What `status` says about the engines: which is picked, and what each needs.
func engineLine(_ backend: EmbeddingBackend) -> String {
    var line = "\(backend.resolved == .mlx ? "MLX" : "llama.cpp") (\(backend.rawValue == "automatic" ? "automatic" : "chosen"))"
    if backend.resolved == .mlx {
        #if canImport(MLXRuntime)
        line += ", mlx-swift \(MLXEmbedder.mlxVersion)"
        if let models = try? MLXModels.locate() {
            line += models.upToDate ? ", weights ready" : ", weights to convert"
        } else {
            line += ", no model files"
        }
        #else
        line += ", unavailable off macOS"
        #endif
    } else {
        line += ", llama.cpp \(LlamaEmbedder.buildTag), vectors \(LlamaEmbedder.vectorVersion)"
    }
    line += " · llama.cpp \(LlamaEmbedder.buildTag) also here"
    return line
}

/// The app's indexing helper: sync + index on request, JSON events out (IndexService.swift).
func indexService(_ args: Arguments) throws -> Int32 {
    let backend = try args.backend()
    // One model for both indexes, started when the first file needs it (`SharedEmbedder`).
    let model = SharedEmbedder {
        try makeEmbedder(textOnly: false, backend: backend, log: args.indexFolder.appendingPathComponent("worker.log"),
                         qos: .utility)
    }
    func part(_ file: String, options: IndexOptions, roots: [URL]) throws -> IndexService.Part {
        let store = try IndexStore(directory: args.indexFolder, file: file)
        let indexer = try Indexer(store: store, options: options, makeEmbedder: { try model.embedder() })
        indexer.costsFolder = args.indexFolder   // what indexing costs here, for later estimates
        // A new app version may bring a new model: the index starts over by itself (keyword search on names goes on).
        indexer.embedderCompat = embedderCompat(backend: backend)
        indexer.resetOnModelChange = { indexed, now in
            emit(["event": "reindex", "from": indexed, "to": now, "code": options.code])
        }
        // A chosen folder that isn't there now (an unplugged drive) is skipped, not an error, and its files stay
        // indexed. With --prune, files under none of the given folders leave the index (the app passes all its folders).
        return IndexService.Part(store: store, indexer: indexer, roots: roots,
                                 keepOnly: args.flags.contains("--prune") ? roots : nil)
    }
    var codeOptions = args.options
    codeOptions.code = true
    codeOptions.excludedFolders = (args.lists["--code-exclude"] ?? []).map { Arguments.folderURL($0).path }
    let codeRoots = (args.lists["--code-root"] ?? []).map(Arguments.folderURL)
    // A file whose reading runs away ends this helper, noted for the next one (`Watchdog`). The limit
    // is backend-aware: MLX's bf16 weights hold ~0.5 GB more than llama.cpp's Q8_0 mmaps.
    Watchdog(folder: args.indexFolder,
             memoryLimit: args.values["--memory-limit"].flatMap(UInt64.init).map { $0 << 20 }
                ?? Watchdog.memoryLimit(for: backend),
             timeLimit: args.values["--time-limit"].flatMap(Double.init) ?? Watchdog.defaultTimeLimit).start()
    let service = IndexService(main: try part(IndexStore.mainFile, options: args.options,
                                              roots: args.positional.map(Arguments.folderURL)),
                               code: codeRoots.isEmpty ? nil : try part(IndexStore.codeFile, options: codeOptions,
                                                                        roots: codeRoots),
                               folder: args.indexFolder,
                               model: model,
                               modelReady: { embedderCompat(backend: backend) == nil || modelComplete(backend: backend) },
                               backfill: args.flags.contains("--backfill"),
                               newFilesLimit: args.values["--new-limit"].flatMap(Int.init) ?? 50,
                               linger: args.values["--linger"].flatMap(Double.init) ?? 60)
    return service.serve()
}

/// `digup speed [--backend B] [--json]`: the speed check estimates start from, what indexing has
/// measured on this machine, and the selected engine's embedding throughput (texts/s, images/s per
/// budget, audio-seconds/s, load seconds, peak RSS). The engine benchmark embeds synthetic inputs
/// through the real towers, so it measures the model rather than the disk.
func speed(_ args: Arguments) throws {
    let json = args.flags.contains("--json")
    if !json {
        if let seconds = MeasuredCosts.Speed.measuredSeconds() {
            let speed = MeasuredCosts.Speed.compared(seconds)
            print(String(format: "20 float16 matrix multiplies (2048²) %.3f s · Vision on a made-up screenshot %.3f s",
                         seconds.multiplies, seconds.ocr))
            print(String(format: "next to the M4 Pro behind the fixed costs: the model's work ×%.2f, OCR ×%.2f (1 = as fast)",
                         speed.embedding, speed.ocr))
        } else {
            print("no speed check here (no Metal): the fixed costs stand in until indexing has measured this machine's")
        }
        let measured = MeasuredCosts.load(from: args.indexFolder)
        print("measured by indexing (\(tilde(MeasuredCosts.file(in: args.indexFolder).path))):"
              + (measured.tallies.isEmpty ? " nothing yet" : ""))
        for (name, tally) in measured.tallies.sorted(by: { $0.key < $1.key }) {
            print(String(format: "  %@ %.4f s each, over %.0f", name, tally.seconds / tally.units, tally.units))
        }
    }
    let backend = (try? args.backend()) ?? .automatic
    let benchmark = benchmarkEngine(backend: backend)
    if json {
        var payload: [String: Any] = ["backend": backend.rawValue, "resolved": backend.resolved.rawValue,
                                      "engine": benchmark]
        if let seconds = MeasuredCosts.Speed.measuredSeconds() {
            payload["matrix_multiplies_s"] = seconds.multiplies
            payload["ocr_s"] = seconds.ocr
        }
        emit(payload)
    } else {
        let picked = backend.rawValue == "automatic" ? "automatic" : "chosen"
        let resolved = (benchmark["resolved"] as? String) ?? "?"
        let detail = (benchmark["error"] as? String) ?? benchmarkLine(benchmark)
        print("engine (\(picked) → \(resolved)): " + detail)
    }
}

/// One engine's embedding throughput on synthetic inputs through the real towers. Never throws: a
/// missing model (or an MLX failure under an explicit `--backend mlx`) reports as `error`, so
/// `speed` stays a read-only probe.
func benchmarkEngine(backend: EmbeddingBackend) -> [String: Any] {
    let resolved = backend.resolved
    var peak: UInt64 = Watchdog.footprint() ?? 0
    func rss() {
        if let footprint = Watchdog.footprint() { peak = max(peak, footprint) }
    }
    // Text: 20 short queries, twice (the first pass warms the pipelines; the second is timed).
    // The text-only embedder is released before the full model loads: real processes hold one at a
    // time (query encoder vs indexing helper), and measuring both resident would overstate the peak.
    let queries = (1...20).map { "payment declined error \($0)" }
    do {
        let textResult: (runtime: String, load: Double, rate: Double)
        do {
            let textOnly = try makeEmbedder(textOnly: true, backend: backend, log: nil, qos: .utility)
            rss()
            let loadSeconds = textOnly.info.loadSeconds ?? 0
            _ = try textOnly.embedTexts(queries.map(Prompts.query))
            let started = Date()
            let vectors = try textOnly.embedTexts(queries.map(Prompts.query))
            let seconds = max(Date().timeIntervalSince(started), 0.001)
            rss()
            textResult = (textOnly.info.runtime, loadSeconds, Double(vectors.count) / seconds)
            textOnly.close()
        }
        var result: [String: Any] = ["resolved": resolved.rawValue, "runtime": textResult.runtime,
                                     "load_s": (textResult.load * 100).rounded() / 100,
                                     "texts_per_s": (textResult.rate * 10).rounded() / 10,
                                     "peak_rss_mb": Int64(peak) >> 20]
        // Media: one generated image at two budgets plus 8 s of sine-wave audio, through the full model.
        do {
            let full = try makeEmbedder(textOnly: false, backend: backend, log: nil, qos: .utility)
            rss()
            result["full_load_s"] = ((full.info.loadSeconds ?? 0) * 100).rounded() / 100
            let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: scratch) }
            let image = scratch.appendingPathComponent("speed.png")
            try writeSpeedImage(to: image)
            for budget in [70, 280] {
                let started = Date()
                let embeddings = try full.embedImages([image.path], budget: budget)
                let seconds = max(Date().timeIntervalSince(started), 0.001)
                rss()
                if embeddings.first?.vector != nil {
                    result["images_per_s_\(budget)"] = (1 / seconds * 10).rounded() / 10
                } else {
                    result["images_per_s_\(budget)"] = NSNull()
                    result["image_error"] = embeddings.first?.error ?? "no vector"
                }
            }
            let audio = scratch.appendingPathComponent("speed.wav")
            try writeSpeedWAV(to: audio, seconds: 8)
            let audioStarted = Date()
            let audioEmbeddings = try full.embedAudio([audio.path])
            let audioSeconds = max(Date().timeIntervalSince(audioStarted), 0.001)
            rss()
            if audioEmbeddings.first?.vector != nil {
                result["audio_seconds_per_s"] = (8 / audioSeconds * 10).rounded() / 10
            } else {
                result["audio_seconds_per_s"] = NSNull()
                result["audio_error"] = audioEmbeddings.first?.error ?? "no vector"
            }
            full.close()
            rss()
            result["peak_rss_mb"] = Int64(peak) >> 20
        } catch {
            result["media_error"] = String("\(error)".prefix(300))
        }
        return result
    } catch {
        return ["resolved": resolved.rawValue, "error": String("\(error)".prefix(300)),
                "peak_rss_mb": Int64(peak) >> 20]
    }
}

func benchmarkLine(_ benchmark: [String: Any]) -> String {
    var parts: [String] = []
    if let load = benchmark["load_s"] { parts.append(String(format: "load %.1f s", (load as? Double) ?? 0)) }
    if let rate = benchmark["texts_per_s"] { parts.append(String(format: "%.1f texts/s", (rate as? Double) ?? 0)) }
    for budget in [70, 280] {
        if let rate = benchmark["images_per_s_\(budget)"] as? Double {
            parts.append(String(format: "%.1f images/s @%d", rate, budget))
        }
    }
    if let rate = benchmark["audio_seconds_per_s"] as? Double {
        parts.append(String(format: "%.1f audio-s/s", rate))
    }
    parts.append("peak \(benchmark["peak_rss_mb"] ?? 0) MB")
    if let error = benchmark["media_error"] as? String { parts.append("(media: \(error))") }
    return parts.joined(separator: " · ")
}

/// A 512×384 gradient PNG for `speed`'s image benchmark: no disk dependency, same pixels every run.
func writeSpeedImage(to url: URL) throws {
    // Minimal PNG writer: 8-bit RGB, no interlace. Decoding goes through the same stb_image path as real files.
    let width = 512, height = 384
    var raw = Data()
    for y in 0..<height {
        raw.append(0)   // filter byte: none
        for x in 0..<width {
            raw.append(UInt8((x * 255) / width))
            raw.append(UInt8((y * 255) / height))
            raw.append(UInt8(((x + y) * 255) / (width + height)))
        }
    }
    // zlib-compress with compression level 6 via NSData is unavailable on Linux; use uncompressed DEFLATE
    // blocks instead (valid zlib stream: header 0x78 0x01, stored blocks, Adler-32).
    var zlib = Data([0x78, 0x01])
    var offset = raw.startIndex
    while offset < raw.endIndex {
        let end = min(offset + 65535, raw.endIndex)
        let chunk = raw[offset..<end]
        let final = end == raw.endIndex
        zlib.append(final ? 1 : 0)
        let count = UInt16(chunk.count)
        zlib.append(contentsOf: [UInt8(count & 0xFF), UInt8(count >> 8), UInt8(~count & 0xFF), UInt8(~count >> 8)])
        zlib.append(contentsOf: chunk)
        offset = end
    }
    var adler: UInt32 = 1
    _ = adler
    // Adler-32 of the raw scanlines (s1/s2 folded mod 65521 per byte).
    var s1: UInt32 = 1, s2: UInt32 = 0
    for byte in raw {
        s1 = (s1 + UInt32(byte)) % 65521
        s2 = (s2 + s1) % 65521
    }
    adler = (s2 << 16) | s1
    zlib.append(contentsOf: [UInt8(adler >> 24), UInt8((adler >> 16) & 0xFF), UInt8((adler >> 8) & 0xFF), UInt8(adler & 0xFF)])
    func chunk(_ type: String, _ payload: Data) -> Data {
        var out = Data()
        let count = UInt32(payload.count)
        out.append(contentsOf: [UInt8(count >> 24), UInt8((count >> 16) & 0xFF), UInt8((count >> 8) & 0xFF), UInt8(count & 0xFF)])
        out.append(contentsOf: type.utf8)
        out.append(payload)
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in type.utf8 + payload {
            crc ^= UInt32(byte)
            for _ in 0..<8 { crc = crc & 1 == 1 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1 }
        }
        crc ^= 0xFFFF_FFFF
        out.append(contentsOf: [UInt8(crc >> 24), UInt8((crc >> 16) & 0xFF), UInt8((crc >> 8) & 0xFF), UInt8(crc & 0xFF)])
        return out
    }
    var png = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    var ihdr = Data()
    for value in [UInt32(width), UInt32(height)] {
        ihdr.append(contentsOf: [UInt8(value >> 24), UInt8((value >> 16) & 0xFF), UInt8((value >> 8) & 0xFF), UInt8(value & 0xFF)])
    }
    ihdr.append(contentsOf: [8, 2, 0, 0, 0])   // 8-bit, RGB, deflate, no filter, no interlace
    png.append(chunk("IHDR", ihdr))
    png.append(chunk("IDAT", zlib))
    png.append(chunk("IEND", Data()))
    try png.write(to: url)
}

/// 16 kHz mono 16-bit WAV of a 440 Hz sine for `speed`'s audio benchmark.
func writeSpeedWAV(to url: URL, seconds: Int) throws {
    let rate = 16_000, count = rate * seconds
    var data = Data()
    func append(_ text: String) { data.append(contentsOf: text.utf8) }
    func append32(_ value: UInt32) {
        data.append(contentsOf: [UInt8(value & 0xFF), UInt8((value >> 8) & 0xFF), UInt8((value >> 16) & 0xFF), UInt8(value >> 24)])
    }
    func append16(_ value: UInt16) { data.append(contentsOf: [UInt8(value & 0xFF), UInt8(value >> 8)]) }
    append("RIFF"); append32(UInt32(36 + count * 2)); append("WAVE"); append("fmt ")
    append32(16); append16(1); append16(1); append32(UInt32(rate)); append32(UInt32(rate * 2)); append16(2); append16(16)
    append("data"); append32(UInt32(count * 2))
    for i in 0..<count {
        let sample = sin(2 * Double.pi * 440 * Double(i) / Double(rate)) * 0.5
        let clamped = max(-32768, min(32767, Int((sample * 32767).rounded())))
        append16(UInt16(bitPattern: Int16(clamped)))
    }
    try data.write(to: url)
}

// MARK: Eval

struct EvalFile: Decodable {
    struct Query: Decodable {
        let q: String
        let expect: [String]   // paths relative to `base`; "path@6-12" also requires the hit to start in [6, 12) s
        let tag: String?
        let kind: String?
    }
    let base: String
    let queries: [Query]
}

func evaluate(_ args: Arguments) throws {
    guard let file = args.positional.first else { throw CLIError("which eval file?") }
    let eval = try JSONDecoder().decode(EvalFile.self, from: Data(contentsOf: URL(fileURLWithPath: (file as NSString).expandingTildeInPath)))
    var base = (eval.base as NSString).expandingTildeInPath
    // The testbed kept elsewhere (another drive on Windows, wherever CI puts it): DIGUP_TESTBED stands for
    // ~/DigUpTestbed in the eval's base.
    if let testbed = ProcessInfo.processInfo.environment["DIGUP_TESTBED"], eval.base.hasPrefix("~/DigUpTestbed") {
        base = testbed + eval.base.dropFirst("~/DigUpTestbed".count)
    }
    // "code: …" queries search the code index (with its prompt), the others the main one.
    let isCode = eval.queries.map { CodeQuery.text(of: $0.q) != nil }
    let texts = eval.queries.map { CodeQuery.text(of: $0.q) ?? $0.q }
    var searchers: [Bool: Searcher] = [:]
    var vectors = [[Float]?](repeating: nil, count: texts.count)
    for code in Set(isCode) {
        let store = try IndexStore(directory: args.indexFolder, file: code ? IndexStore.codeFile : IndexStore.mainFile)
        let searcher = try Searcher(store: store)
        searchers[code] = searcher
        let indices = texts.indices.filter { isCode[$0] == code }
        let embedded = try queryVectors(indices.map { texts[$0] }, store: store, searcher: searcher,
                                        backend: try args.backend(),
                                        prompt: code ? Prompts.codeQuery : Prompts.query)
        for (index, vector) in zip(indices, embedded ?? []) { vectors[index] = vector }
    }
    let depth = args.values["-n"].flatMap(Int.init) ?? 10

    var byTag: [String: (n: Int, at1: Int, at5: Int, at10: Int, mrr: Double)] = [:]
    var shown = 0, listed = 0, folded = 0   // the app's weaker-matches fold (Searcher.strongCount), over all queries
    for (index, query) in eval.queries.enumerated() {
        let kinds = query.kind.flatMap { FileKind(rawValue: $0) }.map { Set([$0]) }
        let hits = try searchers[isCode[index]]!.search(texts[index], queryVector: vectors[index], kinds: kinds,
                                                       limit: depth, zGate: args.zGate, wordWeight: args.wordWeight)
        let strong = Searcher.strongCount(hits)
        shown += strong
        listed += hits.count
        let targets: [(path: String, range: ClosedRange<Double>?)] = query.expect.map { entry in
            let parts = entry.split(separator: "@", maxSplits: 1)
            let range = parts.count == 2 ? parts[1].split(separator: "-").compactMap { Double($0) } : []
            return (base + "/" + parts[0], range.count == 2 ? range[0]...range[1] : nil)
        }
        // A place to be at: a page or moment starts in the range; a stretch of code overlaps it (its lines or cells).
        func at(_ hit: SearchHit, _ range: ClosedRange<Double>) -> Bool {
            guard let loc = hit.loc else { return false }
            if hit.segment == .lines || hit.segment == .cells { return loc <= range.upperBound && (hit.locEnd ?? loc) >= range.lowerBound }
            return range.contains(loc)
        }
        let rank = hits.firstIndex { hit in
            targets.contains { target in
                hit.path == target.path && (target.range.map { at(hit, $0) } ?? true)
            }
        }.map { $0 + 1 }
        if let rank, rank > strong { folded += 1 }
        let tag = query.tag ?? "all"
        for key in Set([tag, "ALL"]) {
            var row = byTag[key] ?? (0, 0, 0, 0, 0)
            row.n += 1
            if let rank {
                if rank <= 1 { row.at1 += 1 }
                if rank <= 5 { row.at5 += 1 }
                if rank <= 10 { row.at10 += 1 }
                row.mrr += 1 / Double(rank)
            }
            byTag[key] = row
        }
        let top = hits.first.map { "\(($0.path as NSString).lastPathComponent)" + (location($0).isEmpty ? "" : " [\(location($0))]") } ?? "-"
        print("\(rank.map { lpad("#\($0)", 4) } ?? "miss")  \(pad(tag, 11)) \(pad("“\(query.q)”", 42)) top: \(top)")
    }
    print("")
    print("\(pad("tag", 12)) \(lpad("n", 3))  \(lpad("R@1", 5))  \(lpad("R@5", 5))  \(lpad("R@10", 5))  \(lpad("MRR", 5))")
    for (tag, row) in byTag.sorted(by: { ($0.key == "ALL" ? "~" : $0.key) < ($1.key == "ALL" ? "~" : $1.key) }) {
        let n = Double(row.n)
        print("\(pad(tag, 12)) \(lpad("\(row.n)", 3))  " + String(format: "%5.2f  %5.2f  %5.2f  %5.2f",
              Double(row.at1) / n, Double(row.at5) / n, Double(row.at10) / n, row.mrr / n))
    }
    let count = Double(max(1, eval.queries.count))
    print(String(format: "weaker-matches fold (below 0.5× the top score): %.1f of %.1f results shown per query; ",
                 Double(shown) / count, Double(listed) / count)
          + "\(folded) expected \(folded == 1 ? "answer" : "answers") folded")
}

// MARK: The MLX engine's weights

/// `digup convert-mlx [--reconvert]`: the MLX engine's weights, converted once from the GGUFs that are already
/// downloaded and verified (`ModelConvert`). The first MLX run does this by itself; this is for doing it now,
/// and for remaking a cache after a llama.cpp-unrelated model update.
func convertMLX(_ args: Arguments) throws {
    #if canImport(MLXRuntime)
    let models = try MLXModels.locate()
    if models.upToDate, !args.flags.contains("--reconvert") {
        print("MLX weights are ready: \(tilde(models.cache.path))")
        return
    }
    print("converting the GGUFs into MLX weights (once, \(ModelConvert.recipe)):")
    let started = Date()
    try ModelConvert.convert(models: models, reconvert: args.flags.contains("--reconvert")) { line in
        print("  \(line)")
    }
    print("done in \(duration(Date().timeIntervalSince(started))): \(tilde(models.cache.path))")
    #else
    throw WorkerError.failed("MLX needs macOS")
    #endif
}

/// `digup gguf-info <file>... [--tensors]`: a GGUF as the runtimes read it — its metadata, and its tensors'
/// names, shapes and types when --tensors is there (for checking a model update against what DigUp reads).
func ggufInfo(_ args: Arguments) throws {
    guard !args.positional.isEmpty else { throw CLIError("which file?") }
    for path in args.positional {
        let file = try GGUFFile(path: path)
        print(tilde(path))
        print("  architecture: \(file.architecture ?? "?"), \(file.tensors.count) tensors")
        for (key, value) in file.metadata.sorted(by: { $0.key < $1.key }) where key != "general.architecture" {
            print("  \(key) = \(summary(value))")
        }
        guard args.flags.contains("--tensors") else { continue }
        for tensor in file.tensors {
            let shape = tensor.shape.reversed().map(String.init).joined(separator: " × ")
            print("  \(pad("\(tensor.type)", 6)) \(pad(shape, 22)) \(tensor.name)")
        }
    }
}

/// A metadata value for reading: arrays fold after a few, everything else prints whole.
func summary(_ value: GGUFFile.Value) -> String {
    func describe(_ value: GGUFFile.Value) -> String {
        switch value {
        case .bool(let value): "\(value)"
        case .uint8(let value): "\(value)"
        case .int8(let value): "\(value)"
        case .uint16(let value): "\(value)"
        case .int16(let value): "\(value)"
        case .uint32(let value): "\(value)"
        case .int32(let value): "\(value)"
        case .float32(let value): "\(value)"
        case .uint64(let value): "\(value)"
        case .int64(let value): "\(value)"
        case .float64(let value): "\(value)"
        case .string(let value): value
        case .array(let values): "[" + values.map(describe).joined(separator: ", ") + "]"
        }
    }
    guard case .array(let values) = value, values.count > 8 else { return describe(value) }
    return "[" + values.prefix(8).map(describe).joined(separator: ", ")
        + ", … (\(values.count) items)]"
}

// MARK: Main

do {
    let raw = CommandLine.arguments.dropFirst()
    guard let command = raw.first, command != "-h", command != "--help", command != "help" else {
        print(usage)
        exit(0)
    }
    let args = try Arguments(raw.dropFirst())
    if ["index-service", "worker", "search-service", "download-model"].contains(command)
        || command == "estimate" && args.flags.contains("--json") {
        ParentWatch.start()   // an app's helper goes when the app does
    }
    switch command {
    case "estimate": try estimate(args)
    case "index": try index(args)
    case "search": try search(args)
    case "status": try status(args)
    case "speed": try speed(args)
    case "eval": try evaluate(args)
    case "index-service": exit(try indexService(args))
    case "worker": exit(workerService(args))
    case "search-service": exit(try searchService(args))
    case "download-model": exit(try downloadModel(args))
    case "convert-mlx": try convertMLX(args)
    case "gguf-info": try ggufInfo(args)
    default:
        print(usage)
        exit(2)
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
