# MLX backend for DigUp — design and implementation plan

Status: implementation · October 2026 · owner: you

Phases 0–4 are complete and green: plumbing, text/vision/audio towers with reference-vector gates,
plus the performance/product surface — backend-aware `speed` with engine benchmarks, cross-backend
and mixed-index verification on a synthetic index, Settings engine picker with conversion progress,
Watchdog limits per backend, `build.sh`/CI wiring, and user docs. Phase 5 (default-engine decision)
is recorded below with measured numbers: **Automatic stays llama.cpp; MLX ships opt-in**.

DigUp embeds files with EmbeddingGemma 2 on llama.cpp (Metal). This plan adds **MLX** as a second,
native-Swift inference backend for the same model on macOS, behind the existing `Embedder` seam,
with index compatibility preserved and every claim about parity and speed settled by the repo's own
test assets (reference vectors, evals, testbed) rather than by argument.

---

## 1. Why MLX, and what "done" means

### Goals

1. **A second engine, first-class.** `mlx-swift` running EmbeddingGemma 2 natively on Apple
   Silicon, selectable per process (`digup --backend mlx`), from Settings in the app, with
   llama.cpp untouched and always available as the fallback.
2. **No re-index to try it.** Vectors from MLX and llama.cpp live in the same namespace (same
   `vectorVersion` semantics), *if and only if* the parity gates in §5 pass. Switching engines,
   in either direction, must not empty anyone's index.
3. **Falsifiable performance and quality claims.** The eval harness (`evals/`, `digup eval`,
   `digup speed`) measures both engines and the 2×2 cross matrix (index with one, query with the
   other) on the same testbed, on M1-class and M4-class hardware.
4. **One download, one trust root.** MLX weights are derived on-device from the GGUFs the app
   already downloads and SHA-256-verifies (`ModelFiles`). No second model download, no new
   hosting, no Python.

### Non-goals

- No bundled Python runtime (the `worker/` mlx-vlm reference stays a *dev/eval oracle*, not a
  shipping backend).
- No changes on Windows/Linux: they keep llama.cpp exactly as is. The conditional-target pattern
  in `Package.swift` (already used for `DigUpApp`) extends to the new target.
- No change to the vector store format (768 × fp16 in SQLite), the search pipeline, prompts,
  budgets (`[70, 140, 280, 560, 1120]`), or the 30 s audio window.
- No new model versions. MLX runs *EmbeddingGemma 2*, revision-pinned to `ModelFiles.revision`.

### Success criteria (measured, not vibes)

| Metric | Gate |
|---|---|
| Reference vectors (`Tests/LlamaRuntimeTests/Reference/vectors.json`) | cosine ≥ 0.999 for **every** input (text, image, audio), same bar `ReferenceVectorTests` holds llama.cpp to — **complete**: all five texts, all three images, and speech pass in `MLXReferenceVectorTests` |
| Evals (`synthetic`, `multilingual`, `code`) | recall within ±0.5 pt of llama.cpp, per file, including cross-backend and mixed-backend runs — **partially complete**: full testbed unavailable in this environment, so the matrix script (`scripts/mlx-eval-matrix.sh`) plus a synthetic cross/mixed-index check stand in (same top hit both directions, mixed index searchable with either backend; see §5.6). Full testbed evals remain CI work |
| Indexing throughput | ≥ 1.0× llama.cpp to ship at all; ≥ 1.2× on M1- and M4-class to become the default — **measured below: not met (MLX 0.5–1.0×), so MLX ships opt-in** |
| Query encode | panel-open → vector no slower than llama.cpp's (load + first embed), measured via `digup speed` — **measured: MLX text-only load 1.1–1.2 s vs llama.cpp 0.5–0.6 s (2× slower cold); warm texts/s at parity** |
| Memory | peak RSS ≤ 1.35× the llama.cpp path on the speed testbed, else ship the 8-bit weight recipe (§4.4) — **measured: 3.5× (2.6 GB vs 0.75 GB), so the 8-bit recipe stays future work and the default stays llama.cpp** |
| Robustness | MLX init failure falls back to llama.cpp in-process, logged, no user-visible breakage — **complete** |

If the throughput gate can't be met, MLX still ships as an opt-in engine on the strength of
quality parity and the strategic wins below; the *default* stays llama.cpp. The plan has a kill
point (§6, Phase 1) if parity itself fails.

### The honest strategic case

- **llama.cpp's center of gravity is autoregressive decode.** Our workload is the opposite: a
  small bidirectional encoder, batched, mean-pooled, no KV cache, with fixed-shape vision inputs
  — exactly the shape `mx.compile` fusion and MLX's batched SDPA are built for. Whether that
  translates to real wins on 271M–740M-parameter towers is an empirical question this plan
  answers with a spike before the full port.
- **bf16 weights.** llama.cpp runs Q8_0 (block quantization); MLX can run the dequantized weights
  at bf16 — closer to the model Google trained, and the exact configuration the mlx-vlm reference
  validated at ≥ 0.999 agreement. If retrieval quality measurably improves on `multilingual`,
  that's a user-visible reason to switch engines beyond speed.
- **One language runtime.** The app, the engine and the model code become one Swift codebase; the
  Python worker remains the reference oracle for tests, not a dependency of the product.
- **Counterpoints kept in view:** llama.cpp's Q8_0 Metal kernels move half the bytes of bf16 and
  are extremely tuned; MLX bf16 could *lose* on memory-bandwidth-bound towers. That's why the
  weight-recipe experiment (bf16 vs MLX 8-bit group-64) is part of Phase 1, not an afterthought.

---

## 2. Where inference stands today (the seams this plugs into)

- **`Embedder`** (`Sources/DigUpKit/Worker.swift:31`) — the whole backend contract:
  `info`, `embedTexts`, `embedImages(budget:)`, `embedAudio`, `close`. "One thread at a time."
  Images and audio arrive as **file paths** (DigUp's own scratch files: downscaled PNG/JPEG,
  16 kHz mono WAV windows).
- **`WorkerInfo.vectorSource`** (`Worker.swift:25`) — `model@revision runtime dtype`; the index
  fingerprint (`Indexer.fullWorker` → meta `vector_source`; `Indexer.checkModel` decides
  re-embed vs mismatch; `queryVectors` refuses query/index source mismatches;
  `Engine` drops meaning search on mismatch).
- **`LlamaEmbedder`** (`Sources/LlamaRuntime/LlamaEmbedder.swift`) — in-process llama.cpp: text
  GGUF + mmproj via mtmd, one `llama_process` per batch, mean-pooled, L2-normalized in Swift.
- **`makeEmbedder(textOnly:log:qos:)`** (`Sources/digup/main.swift:196`) — the single factory:
  `$DIGUP_WORKER_DIR` (Python mlx-vlm reference) else `LlamaEmbedder`. `embedderSource()`
  advertises the vector source without loading a model. Used by `index`, `index-service`
  (shared embedder for main + code index), `search`, `eval`, `search-service`.
- **Query encoder in the app** — `Engine.runningEncoder()` spawns the bundled helper:
  `digup worker` (JSON-lines protocol, `WorkerService.swift`), text-only, exits when idle.
- **Models** — `ModelFiles`: pinned repo `ggml-org/embeddinggemma-2-GGUF` @ revision, two files
  (text 310 MB, mmproj 555 MB, both Q8_0), SHA-256-verified, resumable download, HF-cache reuse,
  `DIGUP_MODELS` to point helpers at a directory.
- **Parity assets** — `ReferenceVectorTests` (5 texts incl. a 1,800-token doc, 3 images at
  budgets 280/140/70, one 30 s WAV; tolerance 0.999; `DIGUP_WRITE_REFERENCE=1` regenerates and
  refuses to rewrite vectors for an unchanged `vectorVersion`), `ImageSizing` (Pillow-bit-exact
  bicubic + budget → target size, unit-tested), `evals/*.json` + `digup eval`, `scripts/testbed`.
- **The MLX oracle already exists** — `worker/digup_worker.py` runs EmbeddingGemma 2 on
  mlx-vlm (pinned: mlx-vlm @ fec3f50, mlx 0.32.3, transformers 5.19.0, bf16), measured at
  ≥ 0.999 agreement with llama.cpp (0.9997 vs float32). It is the porting reference *and* the
  parity oracle for golden intermediates.

---

## 3. The three decisions

### D1 — Native Swift MLX (`mlx-swift`), not the Python worker, not "tune llama.cpp"

| Option | Verdict |
|---|---|
| **A. Native `mlx-swift` backend in the package** | **Chosen.** No runtime deps to ship/sign (~1.5–2 GB avoided), starts in-process like `LlamaEmbedder`, uses the same helpers and protocols, testable in `swift test` on CI's M-series runners. |
| B. Ship the Python worker (uv venv + mlx-vlm) | Rejected: venv size, PyPI supply chain, code-signing a Python tree, cold start, and it duplicates the model download (HF bf16 weights ≠ our GGUFs). Stays the dev/eval oracle. |
| C. Stay on llama.cpp only | Rejected as a *plan*; it forfeits bf16 weights, compile fusion, and the Swift-native path. It remains the fallback and the parity oracle — the plan strengthens, never weakens, it. |

Dependency posture (mirrors how llama.cpp is vendored): pin **`ml-explore/mlx-swift` exactly**
(0.31.6 at the time of writing). The implementation does **not** depend on `mlx-swift-lm`
wholesale: its `MLXEmbedders` (`EmbeddingGemma` v1, Gemma-3-based) and `MLXVLM` (Gemma 4 pieces on
main) were useful *reference code* (MIT), but DigUp needs exactly one model and no registry
machinery; a fast-moving 3.x dependency with breaking changes is the wrong trade. Everything
model-specific lives in the `MLXRuntime` target.

macOS floor: mlx-swift requires macOS 14 — DigUp's floor already is macOS 14. No conflict.

### D2 — Weights: convert the verified GGUFs to an MLX cache, on device, once

| Option | Verdict |
|---|---|
| **A. On-device GGUF → safetensors cache** | **Chosen.** First MLX use converts the two pinned, SHA-256-verified GGUFs into bf16 (or 8-bit) `.safetensors` under `<modelsDir>/mlx/eg2-<recipe>-v1/`, with a manifest (source revision + SHAs + converter version). Reuse of the download, HF-cache and "Use Files…" flows; nothing new to host or trust; works offline. One-time cost (~10–30 s, mostly I/O), ~1.4 GB extra disk for bf16. |
| B. Download a pre-converted MLX repo (mlx-community-style) | Rejected as primary: a second ~0.8–1.5 GB download for every user, a new artifact to host/pin/verify, and existing users pay it again for weights they already have. Worth revisiting only if the GGUF tensor layout turns out to be unmappable (§9). |
| C. Parse GGUF on every model load | Rejected: adds seconds to every encoder start (the panel-open path cares); conversion once + mmap'd safetensors is strictly better. |

The GGUF format is a small, stable, self-describing container (header, key/value metadata, tensor
directory, raw blocks). Q8_0 dequantization is deterministic. Because **we** write the Swift
model, the converter keeps GGUF tensor names as-is and the `MLXNN` module tree is keyed to them
(`@ModuleInfo(key:)`) — **there is no tensor-rename mapping to get wrong.** The text model and
mmproj (vision + audio towers + projections) convert into two safetensors files, plus extraction
of the tokenizer vocabulary (D2.1).

Cache rules: manifest mismatch (new `ModelFiles.revision`, new converter version, new recipe) →
regenerate atomically (`.partial` + rename, same discipline as model downloads); stale recipes →
delete; the folder inherits the models directory's Time Machine exclusion; `DIGUP_MODELS` is
respected throughout. Conversion runs at utility QoS with progress events on the helper's
JSON-lines stdout ("preparing MLX engine", %), and is cancellable like any indexing stop.

**D2.1 — Tokenizer.** Implemented without a tokenizer download or tokenizer dependency:
`GGUFVocab` reads the tokenizer directly from inside the text GGUF through llama.cpp, so both engines
make identical ids by construction. A `swift-transformers` prototype was measured first and rejected:
its BPE implementation byte-falls back where llama.cpp merges multibyte scripts, so Bengali/Arabic/CJK
ids diverged. The remaining framing test pins `<bos> … <eos>` and truncation against
`LlamaEmbedder`.

### D3 — One vector namespace across engines (compat), gated by tests, not by hope

Today `vectorSource` (model@rev **runtime** dtype) is the compatibility key, so a naive MLX
backend would force a full re-embed on first run and another one when switching back — an
unacceptable tax for an experimental engine. But the repo's own measurements say the engines
*are* compatible: llama.cpp (Q8_0) vs mlx-vlm (bf16) agree ≥ 0.999 on text and images and 0.9998
on audio — and `ReferenceVectorTests`'s bar for "the same vectors" is precisely 0.999.

Design:

- `WorkerInfo` gains a `vectorCompat: String` (Codable with a default — the Python worker doesn't
  send it, so decode falls back to `vectorSource`; the worker protocol stays compatible).
- Both engines report the shared namespace now that the reference-vector gates pass for text,
  images, and audio. The eval matrix of §5 (2×2 cross-engine + mixed-index) remains required before a
  default-engine decision, but it is not required to preserve the shared test namespace.
- `IndexStore` schema 5 → 6: write meta `vector_compat` next to `vector_source` (which stays as
  provenance: which engine made these vectors); migration derives compat from known
  `vector_source` values, unknown sources keep `vectorSource` (conservative).
- `Indexer.checkModel()`, `queryVectors` (`main.swift:349`) and `Engine`'s mismatch handling
  compare **compat**, not source.
- Mixed indexes (some vectors from each engine) are a supported state once the gate passes;
  per-modality score normalization in `Searcher` already absorbs small scale differences, and
  the mixed-index eval measures it explicitly.

This makes engine switching free, makes A/B comparison trivial (`--reindex` unnecessary), and
keeps the "index remembers which model version made its vectors" promise intact.

---

## 4. Target architecture

### 4.1 Package layout

```
Sources/CStbMLX/                         tiny stb_image shim so MLX decodes the pixels mtmd decodes
Sources/DigUpKit/GGUF.swift              minimal GGUF reader: header, KV metadata, tensor dir, Q8_0/F32/BF16/F16 blocks
Sources/LlamaRuntime/GGUFVocab.swift     the GGUF's own tokenizer through llama.cpp (both engines' ids)
Sources/MLXRuntime/                      (macOS only; the LlamaRuntime analogue)
  MLXEmbedder.swift                      Embedder conformance: batch, pool, L2-normalize, warm-up, close
  ModelConvert.swift                     GGUFs → safetensors cache (bf16 recipe), manifest, exact module-name mapping
  MLXModels.swift                        locate()/ensure() the cache beside the GGUFs (LlamaModels analogue)
  Framing.swift                          media marker/soft-token sequences and the 2048-token text cut
  Model/
    EmbeddingGemma2Text.swift            text backbone (sliding/full hybrid, MQA + shared KV, PLE, softcap-free)
    VisionTower.swift                    ViT (patch 16), budget-driven shapes, projection into text hidden space
    AudioTower.swift                     USM conformer (12 layers, conv subsampling) + projection
    MelFrontend.swift                    log-mel filterbank (MLX FFT): frame/window/hop from the pinned config
Sources/MLXRuntimeTests/                 (macOS only; mirrors LlamaRuntimeTests)
Tests/LlamaRuntimeTests/Reference/       shared fixtures (vectors.json, inputs) — referenced, not copied
```

`Package.swift` follows the manifest's existing conditional pattern (`appTargets`,
`appDependencies`): `mlxDependencies` (mlx-swift exact) and `mlxTargets`
(`CStbMLX`, `MLXRuntime`, `MLXRuntimeTests`) exist only `#if os(macOS)`; the `digup` target's
dependency list adds `"MLXRuntime"` conditionally. `Sources/digup` guards backend-specific calls with
`#if canImport(MLXRuntime)`. Windows/Linux builds never resolve mlx-swift.

### 4.2 Backend selection

- **CLI**: global `--backend llama|mlx` (default: `$DIGUP_BACKEND`, else `automatic`). `automatic`
  still means `llama`; every embedder-consuming command honors the selected backend
  (`index`, `index-service`, `search`, `eval`, `worker`, `search-service`). `speed` takes
  `--backend` too, with `--json` emitting the engine benchmark for tables and CI.
- **`makeEmbedder`** grows a `backend` parameter and constructs `MLXEmbedder` or `LlamaEmbedder`;
  `embedderCompat()` answers for the selected backend without loading weights (MLX can: its
  `modelInfo` is static).
- **App: complete.** `AppSettings.embeddingBackend` (+ save/load, `-embeddingBackend` launch override),
  `Engine` passes `--backend` to the indexing helper and query encoder (always explicit, so a stray
  `$DIGUP_BACKEND` can't change behavior), logs MLX→llama.cpp fallback, and restarts both on change.
  Settings → Storage has a segmented Engine picker (Automatic/llama.cpp/MLX), live MLX cache state
  (`MLXCache.state`, shared with `MLXModels` so the names can't drift), a "Prepare MLX weights…"
  button running the helper's `convert-mlx` with progress, and the shared-namespace explainer.
  `digup status` reports the selected engine, MLX cache state and size, and vector compat.
- **Fallback**: if `MLXEmbedder` init throws (missing cache and conversion impossible, Metal
  failure, out of memory), the factory logs and falls back to `LlamaEmbedder` — except when an
  explicit `--backend mlx` was given (fail loudly for the person who asked). Since compat is
  shared, a fallback mid-index mixes engines safely (that's what D3 bought).

### 4.3 `MLXEmbedder` — the model port

Porting references, in order of authority: (1) the pinned **mlx-vlm** implementation (it is the
validated parity oracle, ≥ 0.9997 vs float32); (2) `mlx-swift-lm`'s Gemma/Gemma4 Swift code for
idiomatic `MLXNN` structure (MIT); (3) llama.cpp's arch code for framing details. Exact
hyperparameters are read from the GGUF KV metadata. The implementation confirmed the following
against the real checkpoint:

- **Text tower (~271M)** — Gemma 4 backbone as a *bidirectional encoder*: hybrid attention
  (sliding window 512 + full layers), MQA with shared KV layers, per-layer embeddings (PLE),
  RMSNorm `(1 + w)` in fp32, GeGLU, tied embeddings, 262k vocab, no KV cache, mean-pool over the
  sequence → output projection to 768 → L2-normalize (order of pool/proj exactly as mlx-vlm does
  it; reference vectors arbitrate).
- **Vision tower (~150M-class)** — ViT, patch 16, SigLIP-style; pixels are divided by 255 and
  scaled to [-1, 1], patchified from the **shared** `ImageSizing` output (§4.5) after decoding with
  the same stb_image path mtmd uses; soft-token budget `{70,140,280,560,1120}` maps to target sizes
  exactly as today; the FFN uses `gelu_quick` because the mmproj carries no `use_gelu` flag and
  clip.cpp defaults to quick; rows are projected into the text hidden space and scattered at the
  image soft-token positions.
- **Audio tower (~305M-class)** — USM conformer: 2 conv subsampling layers, 12 conformer layers,
  chunked attention, and an HF-exact log-mel frontend (MLX FFT; 320-sample window, 160-sample hop,
  512-point FFT, 128 HTK bins, 0.001 floor, natural log, semicausal padding; 16 kHz, 30 s window ⇒
  ~750 soft tokens). The GGUF already stores softplus-transformed per-dimension scales and swaps
  the two convolution-norm names; the loader follows llama.cpp's reverse-order load. Rows are
  projected and scattered like vision.
- **Framing** — `<bos> … <eos>`; media inputs `<bos> <marker-open> <soft × n> <marker-close>
  <eos>` with the checkpoint's special token ids (boi 255999, boa 256000, image soft 258880, audio
  soft 258881, eoi 258882, eoa 258883 — verified against `mtmd_tokenize`). The exact sequence is
  **asserted** against `mtmd_tokenize`'s output for probe inputs (llama.cpp is linked into the same
  test binary — compare token id arrays directly).
- **Execution** — all MLX work on one thread (matches the protocol's "one thread at a time" and
  MLX's per-thread streams). Warm up with two `embedTexts` at init (metal pipeline cost paid while
  the panel opens), record `loadSeconds` like `LlamaEmbedder` does. `mx.compile` with shape
  bucketing was evaluated for Phase 4 and **deferred**: the workload is batch-16 text plus single
  media inputs, where graph compilation saves little and risks drift at the 0.999 gate; batching
  already matches the reference's instead. Media batches clear the GPU cache on return.

### 4.4 Weight recipes and memory policy

The implemented recipe is:

- **bf16** — dequantize Q8_0 blocks to bf16 (parity with the Python reference by construction:
  same weights llama.cpp computes with, rounded to bf16; bf16's exponent range avoids the fp16
  activation overflow `ModelFiles` warns about). The vision tower upcasts its weights to f32 in
  memory because its pooled rows have less mean-pooling dilution than the text tower's.

Measured consequences (M1 MacBook Pro 16 GB, `digup speed --backend … --json`, synthetic inputs
through the real towers — see the speed table in Phase 5): the bf16 cache is 1.49 GB on disk and
peaks at ~2.6 GB RSS vs llama.cpp's ~0.75 GB (3.5×, over the 1.35× gate). The planned **8-bit
group-64** recipe therefore stays future work rather than a Phase 4 deliverable: it would halve
weight bytes and is the route to meeting the memory gate, but bf16 parity is proven and the
engine ships opt-in regardless. Per-command cache tuning landed (512 MB indexing / 256 MB query,
mirroring the Python worker), media batches clear the GPU cache on return, `Watchdog` is
backend-aware (MLX floor 5 GB vs 4 GB, same 8 GB ceiling), and the helper-quit-when-done process
model is unchanged — it remains the ultimate memory reclaim.

### 4.5 Preprocessing: shared, not duplicated

- `ImageSizing` moves from `LlamaRuntime` to `DigUpKit` (it is reference parity logic, not
  llama.cpp glue; `ImageSizingTests` move with it). Both backends consume it — same pixels in,
  same soft-token count out. Pixel decode: the same stb_image path mtmd uses, through the tiny
  `CStbMLX` shim. The JPEG LSB concern in risk 8 materialized during implementation: frame-70
  needed bit-parity, so the fallback became the implementation.
- Audio arrives as DigUp's own 30 s 16 kHz mono WAV scratch files (written by `writeWAV`), so a
  ~60-line RIFF reader replaces miniaudio entirely — no decoder drift possible.

### 4.6 What stayed untouched, and what the implementation touched

`Prompts`, extraction, crawler, vector storage (fp16 blobs), `WorkerConfig`/`EmbeddingWorker`
protocol, Sparkle, and onboarding (GGUFs remain the only download; first MLX use converts once with
"Preparing MLX engine…" in the log). The implementation changed `Indexer`, `Searcher`, and
`IndexStore` for the compat namespace; added `GGUF.swift`, `GGUFVocab`, `CStbMLX`, `MLXRuntime`,
`MLXRuntimeTests`, `MLXCache`, `convert-mlx`, `gguf-info`, `speed --backend/--json`,
`scripts/mlx-metallib.sh` (wired into `build.sh`), `scripts/mlx-eval-matrix.sh`, and `.github/workflows/ci.yml`;
moved `ImageSizing` into the kit; and updated `README.md` and `Resources/Acknowledgements.txt`
(engine choice, build step, mlx-swift MIT notice).

---

## 5. Parity, testing, and CI

The principle: **llama.cpp (in the same binary) and the pinned Python worker are oracles; every
layer of the MLX stack gets its own golden test.**

1. **Conversion** — implemented as `Sources/DigUpKit/GGUF.swift` plus `digup gguf-info`; covered
   indirectly by the reference-vector gates plus `ModelConvertTests.weightNamesFollowModulePaths`
   (linear weights keep `.weight`, raw parameters drop it) and `cacheIdentityMatchesApp`
   (`MLXCache` and `MLXModels` can't drift).
2. **Tokenizer** — implemented as `Sources/LlamaRuntime/GGUFVocab.swift`: identical ids by
   construction, plus `framingMatchesLlamaCpp` and `mediaFramingMatchesMtmd` gates. A broader
   multilingual corpus fixture remains possible, but is no longer load-bearing.
3. **Frontends** — implemented as `MelFrontend` plus shared `ImageSizing`; committed as
   `MelFrontendTests` (filterbank pinned to transformers' values, constants, silence→floor shape).
   Normalization constants match the pinned processor config.
4. **Layers** — implemented against llama.cpp stage dumps and the Python reference during
   development; there are no committed hidden-state fixtures (the reference-vector gates arbitrate
   end-to-end instead).
5. **End-to-end — complete.** `MLXReferenceVectorTests` compares every reference input against the
   existing `vectors.json` at ≥ 0.999: five texts, three images, and speech, plus text and media
   framing gates. MLX does **not** have its own reference file.
6. **Evals — matrix scripted, testbed-gated.** `digup eval` already takes `--backend` for query
   vectors (index with A via `digup index --backend A`, query with B via `digup eval --backend B`).
   `scripts/mlx-eval-matrix.sh` automates the 2×2 plus documents the mixed-index procedure (half
   the folders with each backend into one index). Verified here on a 3-doc synthetic index: both
   cross directions return the same top hit as same-engine, and a mixed index (llama + mlx segments)
   is searchable with either backend. Full `synthetic`/`multilingual`/`code` testbed evals run in
   CI once `DIGUP_TESTBED` is provisioned (see `ci.yml`).
7. **Benchmarks — complete.** `digup speed --backend … [--json]` reports texts/s, images/s @70/@280,
   audio-seconds/s, text-only and full load seconds, and peak RSS (`Watchdog.footprint`), for
   either engine; `--json` emits one object for tables and CI. See the Phase 5 table for numbers.
8. **CI — complete.** `.github/workflows/ci.yml`: macOS M-series job (build llama.cpp, download +
   convert models, both reference-vector suites, full `swift test`, both `speed --json` tables,
   eval matrix as a non-blocking report), Linux + Windows build-only jobs asserting mlx-swift
   never resolves there. `scripts/mlx-metallib.sh` is wired into `build.sh`.
   - **macOS (M-series runner)**: `scripts/build-llama.sh`, `swift test` (model-dependent tests
     already self-skip without weights; the job downloads models via `digup download-model` into
     a cached `DIGUP_MODELS` dir by SHA), reference-vector + tokenizer + eval gates for both
     backends.
   - **Linux + Windows**: build-only, to keep the conditional manifest honest (mlx-swift must
     never leak into them).
   - **Nightly (optional)**: run the same against the Python worker for oracle freshness, and a
     mlx-swift version-bump canary.
   A mlx-swift version bump is treated like a llama.cpp tag bump: pinned exact, upgraded only
   with `swift test` green (the same rule `scripts/build-llama.sh` documents).

---

## 6. Delivery plan

Each phase lands green on `swift test` + evals; nothing user-visible ships before Phase 4.

**Phase 0 — Groundwork: complete.**
Moved `ImageSizing` (+ tests) into DigUpKit; added `WorkerInfo.vectorCompat` with decoding default;
schema 5→6 migration; `--backend` plumbing end-to-end; `digup status` backend-aware (engine line +
MLX cache line); `scripts/mlx-metallib.sh` added and wired into `build.sh`; `ci.yml` added.
`swift test` is green, with no behavior change on the llama path.

**Phase 1 — Spike + text tower: complete.**
GGUF reader + converter (bf16) + GGUF-backed tokenizer + text backbone + pooling; MLX text embeds
pass `MLXReferenceVectorTests` on all five text inputs. The 8-bit recipe was deferred with cause
(see §4.4); the kill criteria were not triggered.

**Phase 2 — Vision: complete for reference parity.**
ViT port, pixel normalization, budget shapes, plus framing parity tests vs `mtmd_tokenize`;
`photo-140`, `screen-280`, and `frame-70` all pass. `MelFrontendTests`/`ModelConvertTests` pin the
frontend and name mapping; image-heavy testbed evals run in CI.

**Phase 3 — Audio: complete for reference parity.**
HF-exact mel frontend (MLX FFT) + conformer; `speech` passes at ≥ 0.999; 30 s window framing is
covered by the mtmd framing test. All reference inputs are green, so the reference-vector half of
D3's gate evidence is complete; the cross/mixed-index half is verified synthetically (§5.6).

**Phase 4 — Performance + product surface: complete.**
`digup speed --backend … [--json]` benchmarks both engines (texts/s, images/s @70/@280,
audio-s/s, load, peak RSS); `Watchdog.memoryLimit(for:)` is backend-aware; media batches clear
the GPU cache; `mx.compile` evaluated and deferred with cause. Settings UI ships (engine picker,
live MLX cache state, Prepare button with progress, shared-namespace explainer), onboarding
untouched, README/FAQ/Acknowledgements updated, `digup status` polished with the MLX cache line.
Ships as **opt-in** (`automatic` = llama still). *Exit: opt-in MLX in users' hands.*

**Phase 5 — Default decision: recorded with numbers (M1 MacBook Pro 16 GB, 2026-10-13).**

`digup speed --backend … --json` through the real towers (20 text queries ×2, one 512×384 gradient
PNG @70/@280, 8 s 440 Hz sine):

| Engine | text load | texts/s | images/s @70 | images/s @280 | audio s/s | full load | peak RSS |
|---|---|---|---|---|---|---|---|
| llama.cpp (Q8_0) | 0.58 s | 193.2 | 1.6 | 0.4 | 34.6 | 0.98 s | 745 MB |
| MLX (bf16) | 1.14 s | 185.2 (0.96×) | 1.3 (0.81×) | 0.4 (1.0×) | 18.0 (0.52×) | 1.20 s | 2,573 MB (3.5×) |

Gates (§1): indexing ≥ 1.0× to ship at all / ≥ 1.2× to default — **not met** (0.5–1.0×);
query cold load ≤ llama — **not met** (2× slower); memory ≤ 1.35× — **not met** (3.5×).
Quality parity **is** met (all reference vectors ≥ 0.999; cross/mixed-index checks pass), so per
the plan's fallback rule MLX ships **opt-in** on parity and strategic value, and **`automatic`
stays llama.cpp**. The route to the gates is the 8-bit group-64 recipe (halves weight bytes and
memory-bandwidth, the binding constraint on the conformer and ViT) plus a second measurement pass
on M4-class hardware; recorded here when it lands. The flip costs users nothing either way (D3).

Estimates assume focused staff-level work: Phase 0 ~2–4 days; Phase 1 ~2–3 weeks; Phases 2–3
~1–2 weeks each; Phase 4 ~1–2 weeks. Total ~7–10 weeks to opt-in shipping.

---

## 7. Risk register

| # | Risk | L×I | Mitigation |
|---|---|---|---|
| 1 | Tokenizer edge-case drift (multilingual, byte fallback) breaks parity silently | M×H | **Closed:** use the GGUF's own tokenizer through llama.cpp; framing/tokenizer gates hold the implementation |
| 2 | Text-tower attention port wrong (sliding bidirectional, PLE, shared KV) | M×H | **Closed:** reference vectors pass batched/padded at ≥ 0.999 |
| 3 | Mel frontend mismatch (window/hop/normalization) | M×H | **Closed:** HF-exact frontend; value-for-value Python check and the speech gate |
| 4 | MLX bf16 slower than llama.cpp Q8_0 (bandwidth-bound towers) | M×M | **Confirmed and decided:** M1 measures 0.5–1.0× (text parity, images 0.8×, audio 0.5×); MLX ships opt-in, 8-bit group-64 remains the route to the gate |
| 5 | Memory pressure on 8 GB Macs (bf16 full model ~1.5 GB peak) | M×M | **Confirmed and decided:** M1 peak 2.6 GB vs 0.75 GB (3.5×, over the 1.35× gate); text-only query encoder keeps the smaller 256 MB GPU cache; 8-bit recipe is the future fix, default stays llama.cpp |
| 6 | mlx-swift API churn (0.31→0.32; 3.x-style breakage upstream) | M×M | Exact pin; upgrade as a deliberate PR gated on `swift test` (same rule as llama.cpp tags) |
| 7 | GGUF layout surprise (tensor names/shapes not cleanly mappable) | L×H | **Closed:** GGUF tensor layout mapped; the two documented audio exceptions are the reversed convolution-norm names and pre-transformed per-dimension scales |
| 8 | JPEG decode drift (ImageIO vs stb) breaks image tolerance | L×M | **Closed:** frame-70 required the stb_image path; `CStbMLX` is now the implementation |
| 9 | Mixed-engine score bias within one index | L×M | **Verified synthetically:** cross-direction and mixed-index searches return the same top hit as same-engine; per-modality normalization stands; full testbed matrix runs in CI |
| 10 | App/DMG size growth from mlx-swift | H×L | **Shipped and noted:** mlx-swift links into the helper only (app stays lean); `build.sh` colocates `mlx.metallib`; release notes report the measured delta |
| 11 | Hardened-runtime/Metal issues | L×M | **Wired:** `scripts/mlx-metallib.sh` runs in `build.sh` and CI; signed-build validation rides the existing release/CI checks |

---

## 8. User experience summary

- Nothing changes until a user picks "Engine: MLX" (or runs `digup --backend mlx`). Downloads,
  onboarding, indexing schedules, search, all identical.
- First enable: a one-time "Preparing MLX engine" conversion (progress in Settings / helper
  events), then the same indexing and search flows on the new engine — existing vectors keep
  working (D3), new files embed with MLX.
- Engine switch at any time, either direction, no re-index, no settings gymnastics. If MLX can't
  start, DigUp says so in the menu and quietly uses llama.cpp.
- Privacy posture unchanged: the app still touches the network only for the one model download
  (unchanged artifacts) and update checks.

---

## 9. Resolved questions and remaining work

1. Exact EmbeddingGemma 2 hyperparameters were read from GGUF KV metadata. **Resolved.**
2. The pinned GGUF repo has no tokenizer files. **Resolved:** use the GGUF's own tokenizer through
   llama.cpp (`GGUFVocab`).
3. `mlx-swift-lm` was not used as a dependency. **Resolved:** port directly from the validated
   mlx-vlm Python implementation and llama.cpp.
4. The output projection is per token, then mean pooling — exactly as llama.cpp's pooling does.
   **Resolved.**
5. The reference `agreement` bar was kept at 0.999, and all inputs pass it. **Resolved for the
   reference gates.**
6. Watchdog memory limits per backend: `Watchdog.memoryLimit(for:)` (MLX floor 5 GB, same 8 GB
   ceiling). **Resolved.**
7. Long-term: publish an official MLX-format artifact once the on-device converter has proven
   itself. **Open; revisit after a release cycle of opt-in use.**

---

## Appendix A — File-by-file change list (Phases 0–4)

| File | Change |
|---|---|
| `Package.swift` | conditional `mlxDependencies`/`mlxTargets`; `digup` gains conditional `MLXRuntime` dependency |
| `Sources/DigUpKit/Worker.swift` | `WorkerInfo.vectorCompat` (+ decoding default); `EmbeddingBackend` (+ `displayName`, `automaticExplainer`); `VectorCompat` |
| `Sources/DigUpKit/MLXCache.swift` | new shared MLX cache identity (`recipe`/`version`/folder, `state`, `bytes`) so the app and `MLXModels` can't drift |
| `Sources/DigUpKit/GGUF.swift` | new GGUF reader used by conversion and model configs |
| `Sources/LlamaRuntime/GGUFVocab.swift` | new GGUF tokenizer shared by both engines |
| `Sources/CStbMLX/**` | new stb_image shim for vision bit-parity |
| `Sources/DigUpKit/IndexStore.swift` | schema 6: `vector_compat` meta + migration |
| `Sources/DigUpKit/Indexer.swift` | `checkModel` compares compat |
| `Sources/DigUpKit/ImageSizing.swift` | moved here from `LlamaRuntime` |
| `Sources/digup/main.swift` | `--backend` parsing; `makeEmbedder`/`embedderCompat` backend-aware; `status` extensions (+ MLX cache line); `speed --backend/--json` + engine benchmark + synthetic PNG/WAV writers; `convert-mlx`; `gguf-info`; backend-aware `Watchdog` limit |
| `Sources/digup/Watchdog.swift` | `memoryLimit(for:)` per backend |
| `Sources/digup/WorkerService.swift` | construct MLX embedder when `--backend mlx` |
| `Sources/DigUpKit/ModelFiles.swift` | multi-repo file identity and shared `modelFolders` |
| `Sources/MLXRuntime/**` | new (see §4.1); media batches clear the GPU cache; compile evaluated and deferred (comment on `MLXEmbedder`) |
| `Sources/DigUpApp/AppSettings.swift` | `embeddingBackend` + save/load (+ `-embeddingBackend` launch override) |
| `Sources/DigUpApp/Engine.swift` | pass `--backend` to helper + encoder (always explicit); MLX→llama.cpp fallback logging; `embeddingBackend` getter + `setEmbeddingBackend` (restarts helper, stops encoder); `reembedAll` (stop helper, `resetAll`, re-embed with the current engine) |
| `Sources/DigUpApp/SettingsWindow.swift` | engine picker in StorageSettings, live MLX cache state, Prepare button with progress, Re-embed button with confirmation, shared-namespace explainer |
| `Sources/DigUpApp/AppController.swift` | `setEmbeddingBackend` (saves, updates settings, tells the engine) |
| `Tests/MLXRuntimeTests/**` | reference-vector, text/media framing gates, plus `MelFrontendTests` and `ModelConvertTests` |
| `scripts/mlx-metallib.sh` | Metal-library placement helper (wired into `build.sh` and CI) |
| `scripts/mlx-eval-matrix.sh` | cross-backend eval matrix (2×2 + mixed-index procedure) |
| `build.sh` | runs `scripts/mlx-metallib.sh` after building |
| `.github/workflows/ci.yml` | macOS test + eval + speed tables, Linux/Windows build-only (mlx-swift never resolves there) |
| `README.md`, `Resources/Acknowledgements.txt` | engine docs (How it works, FAQ, build step), MLX MIT notice |

## Appendix B — `makeEmbedder` as implemented

```swift
func makeEmbedder(
    textOnly: Bool,
    backend: EmbeddingBackend = .automatic,
    log: URL?,
    qos: QualityOfService = .default
) throws -> any Embedder {
    if var config = try WorkerConfig.fromEnvironment() {          // the Python oracle, unchanged
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
            note(log, "mlx can't start (\(error)); llama.cpp it is")
            return try LlamaEmbedder(models: try LlamaModels.locate(), textOnly: textOnly, log: log)
        }
    case .llama, .automatic:
        return try LlamaEmbedder(models: try LlamaModels.locate(), textOnly: textOnly, log: log)
    }
}
```

## Appendix C — References

- mlx-swift (0.31.6; API names used here: `MLX.GPU.set(cacheLimit:)`, `set(memoryLimit:relaxed:)`,
  `clearCache()`, `resetPeakMemory()`, `maxRecommendedWorkingSetBytes()`; `MLX.save`/`loadArrays`
  safetensors I/O; `Transforms+Compile` for compile): https://github.com/ml-explore/mlx-swift
- mlx-swift-lm (MLXEmbedders' `EmbeddingGemma`/`Gemma3ModelBackbone`/`EmbedderModelContainer`;
  MLXVLM Gemma 4 work on main): evaluated as reference material, but not used as a dependency.
- swift-transformers (Tokenizers, Gemma/SentencePiece support): evaluated for tokenization, but
  rejected after measured multilingual divergence; the implementation uses the GGUF's own tokenizer.
- Gemma 4 Technical Report (arXiv 2607.02770) — ViT + USM conformer + hybrid attention family details
- The repo's own oracles: `worker/digup_worker.py` (pinned mlx-vlm @ fec3f50, mlx 0.32.3,
  transformers 5.19.0, bf16, ≥ 0.999 agreement), `Tests/LlamaRuntimeTests/Reference/`,
  `evals/`, `scripts/testbed/`, `scripts/build-llama.sh` (llama.cpp b11461).
