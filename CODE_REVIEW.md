# DigUp — Deep Code Review (post cross-platform pull)

**Date:** 2026-10-13
**Reviewer:** Buffy (macOS / llama.cpp / ML review)
**Supersedes:** the pre-pull review of commit `e7dfba4^` (kept in git history, commit `4ecbda0`).
**Scope:** full repository after pulling 109 changed files (12,259 insertions, 670 deletions): `Sources/DigUpKit` (+ the new `Platform/` layer), `Sources/DigUpApp`, `Sources/digup`, `Sources/LlamaRuntime`, `shell/` (new Tauri app), `scripts/`, `Tests/`.

**Verdict:** this pull adds a genuinely impressive cross-platform engine (Apple frameworks on macOS, a portable C-library layer on Linux/Windows) and a new Tauri shell. The engineering *around* the model remains excellent. But **every retrieval-quality defect from the previous review is still open**, and the new portable layer is **not compiled or tested by any CI**. The decision layer is still the weak point.

No source files were changed by this review: `git status` is clean.

---

## 1. What was verified (with exit status captured)

| Check | Command | Result |
|---|---|---|
| Kit builds | `swift build --target DigUpKit` | ✅ exit 0 |
| CLI builds | `swift build --product digup` | ✅ exit 0 |
| Full test suite | `swift test` (`set -o pipefail`, exit captured) | ✅ **exit 0** — `DigUpKitTests`: 86 tests / 9 suites; `LlamaRuntimeTests`: 6 tests / 2 suites |
| Reference vectors | `ReferenceVectorTests.vectorsMatchTheReference()` | ✅ **ran and passed in 8.92 s** (not skipped). The llama.cpp ↔ mlx-vlm cosine ≥ 0.999 claim holds on this machine after the pull |
| End-to-end CLI | `digup index` → `status` → `search` on a scratch corpus | ✅ works, including the vector-source line and the honest "no good matches" |
| P0 reproduction | 4-vector corpus, default gate vs `--zgate 1.0` | ✅ **still broken** — see §3 |

### What the new work gets right

- **A real platform seam.** `Platform/Apple/*` and `Platform/Portable/*` implement the same names (`FileSystem.info`, `Rules.classify`, `Vectors.scores`, `PDFium`/`PDFKit`, `FFmpeg`/`AVFoundation`). `Package.swift` selects targets per OS, and the portable C deps (SQLite+FTS5, stb, miniz, PDFium) are fetched at pinned versions.
- **Careful portability details**: Windows `stbi_*` UTF-8 file names, `FileReader` callback so Bengali/Arabic paths open through miniz/PDFium, FAT of `stat` vs `FindFirstFileExW`, `git` discovery that avoids Xcode's shim, a hand-written `DirectoryWalk` for a Foundation enumerator bug.
- **Speed calibration** (`MeasuredCosts` + `Platform/*/Speed*`): indexing now measures its own cost per kind and scales the M4-Pro baseline by a model-free speed check. This is a real improvement to estimates (though see §4).
- **`Watchdog`**: ends the indexing helper when one file runs away (memory footprint or a per-file time limit) and hands the offending path to the next helper — a direct answer to the "PDF text extraction grew to 30 GB" class of bug.
- **`SearchService` / `IndexService` / `DownloadModel`**: the app's engine as JSON-line helper processes, so the Tauri shell can reuse it. Encoder idle-release, per-request `render` on two slots, and a source-mismatch warning ("meaning search is off: the index was built with X… the encoder is Y…") are all thoughtful.
- **`FileFetcher`**: resumable ranged download that keeps the `Range` across Hugging Face's CDN redirect, writes to disk as it comes, and reports progress at most 5×/s.
- **`IndexSchedule` / `IndexHours`**: event-driven "index only at set hours" with correct across-midnight and DST-gap handling.

---

## 2. Prior findings — status after the pull

| # | Finding | Status |
|---|---|---|
| P0 | Small-index z-gate makes meaning search impossible | ❌ **Still open** (reproduced, §3) |
| P1 | Same gate too permissive on medium corpora | ❌ **Still open** (mean/SD unchanged, `Searcher.meaning`) |
| P1 | `Estimator` has no startup term | ❌ **Still open** (no constant in `Estimator.estimate`; `Estimates.swift` only decodes) |
| P1 | llama.cpp appends `<eos>` on truncation, MLX drops it | ❌ **Still open** (`LlamaEmbedder.tokenize` unchanged) |
| P1 | Integer division in `ImageSizing.targetSize` thin-image fallback | ❌ **Still open** (`width / height * side`) |
| P2 | unbatched media embedding, `rowSetsScale`, `minPageText` fingerprint, fp16 round-trip, Q8_0 rationale | ❌ Unchanged |
| — | `Searcher` scoring has no unit tests; reference test skips without model files | ❌ Unchanged |

The one improvement in this area: `MeasuredCosts` calibrates per-unit cost, but the *constant* model-load cost is still absent.

---

## 3. P0 (still) — meaning search is impossible on small indexes

`Searcher.meaning` (`Sources/DigUpKit/Searcher.swift:322–336`) standardizes each query's cosines by a mean and a
*population* standard deviation over the corpus, and `search` gates at `z >= zGate` with `zGate` defaulting to **2.0**
(`:116`, `:181`, `:199`).

**Corrected bound.** For a sample of size *n*, the largest attainable standardized score is **√(n − 1)** (put one value at
the top and the rest equal). (My earlier report stated `(n−1)/√n`; the measured data below shows the true bound is
`√(n−1)`.)

| n (vectors) | max attainable z | passes `z ≥ 2.0`? |
|---|---|---|
| 2 | 1.00 | never |
| 4 | **1.73** | never |
| 5 | 2.00 | only for a perfect single outlier |
| 6 | 2.24 | yes, barely |
| 20 | 4.36 | yes |

So for **n ≤ 4 the gate can never be passed**, and for n = 5 only in the degenerate case.

**Reproduced on the current build** (4 text docs, `--index /tmp/du-zidx`):

```
$ digup status
vectors: 4 in 4 segments
$ digup search "a vehicle taken in because something was rubbing and the friction material was worn"
no good matches
$ digup search "…" --zgate 1.0
 1. doc  01-car-service.txt  [part 1]   (z 1.7, words #1)   ← the correct file
```

**Impact.** A new user, or any folder with a handful of files, gets keyword-only results while the app reports
"No good matches" with the right file already indexed. `zGate` remains a CLI-only knob, so app users have no recourse.

**Fix.** Make the gate size-aware — `gate = min(2.0, √(n − 1) − margin)` while n is small, or drop the z-gate for a rank
margin (`(top1 − top2)/SD`) plus an absolute cosine floor. This is the single highest-value change in the codebase.

---

## 4. P1 — `Estimator` still has no startup term

`Sources/DigUpKit/Estimator.swift` models per-unit cost but has **no constant for model load / warm-up**. The CLI itself
reports start-up separately (`query encoder 0.56 s incl. start-up`), and `MeasuredCosts` calibrates only per-unit rates,
so first-launch time-to-finish remains optimistic by the full model start. `Indexer` already knows the model-load time
(`LlamaEmbedder.info.loadSeconds`, and `Indexer.swift:245` deliberately excludes it from cost measurement) — that value
should feed a fixed startup term in `Estimate`.

---

## 5. P1 — `<eos>` on truncation still diverges from the reference

`LlamaEmbedder.tokenize` (unchanged):

```swift
if tokens.count > maxTokens {
    tokens = Array(tokens[0..<(maxTokens - 1)]) + [llama_vocab_eos(vocab)]
}
```

The MLX reference drops the trailing token instead, so any input longer than `maxTokens` is framed differently, and the
≥ 0.999 claim is only checked on inputs that fit. Still an **untested path**.

---

## 6. P1 — integer division in `ImageSizing.targetSize` still present

`Sources/LlamaRuntime/ImageSizing.swift`:

```swift
if targetHeight == 0 {
    targetHeight = side
    targetWidth = min(width / height * side, maxSide)      // Int / Int
} else if targetWidth == 0 {
    targetWidth = side
    targetHeight = min(height / width * side, maxSide)     // Int / Int
}
```

`width / height` truncates to {0, 1, 2, …}. Should be `Double(width) / Double(height)`. The main path uses `Double`, so
this only bites on extreme aspect ratios — the exact case the branch exists for.

---

## 7. NEW — the portable layer is never compiled by CI

There is exactly one workflow, `.github/workflows/homebrew.yml`, and it runs on `macos-latest` only to update the cask
from a published release (it verifies the DMG's signature/notarization; it does not build or test the project).

Consequences:

- **`Platform/Portable/*` is `#if !canImport(...)`, so it does not compile on macOS at all** — and nothing else compiles
  it. The portable `SHA256`, `EXIF`, `PDFium`, `FFmpeg`, `RTF`, `miniz`, stb and `Math`/`Speed` code (several thousand
  lines, the bulk of this pull) could have compile errors or logic bugs that nothing catches until someone builds on
  Linux/Windows by hand. `Package.swift`'s non-macOS branches are likewise unexercised.
- **`swift test` is not run in CI on any OS.** The 92 passing tests are local evidence only.
- The `ReferenceVectorTests` model files are not in CI, so the one test that anchors llama.cpp to the reference is
  skipped wherever it does run unattended (a pre-existing gap, now more consequential).

**Fix.** Add a matrix workflow (`macos-latest`, `ubuntu-latest`, `windows-latest`) that runs `./scripts/fetch-vendor.sh`
(non-macOS) and `swift build` + `swift test`. This is the highest-value infrastructure change; without it the port is
unverifiable.

Related: `scripts/fetch-vendor.sh` pins versions/commits but **does not checksum the downloaded archives** (SQLite, stb,
miniz, PDFium). The model download is SHA-256-verified, so the vendor libs are the odd one out in the supply chain.

---

## 8. NEW — compile warnings worth fixing

`swift test` emits warnings (build succeeded, but these are latent bugs / debts):

- `Tests/LlamaRuntimeTests/ReferenceVectorTests.swift:85,88` — `#require(_:_:)` is redundant because `embedding.vector` /
  `speech.vector` *never* equals `nil` (the macro can't fail). The assertions are dead; the test is weaker than it looks.
- `Sources/DigUpApp/ModelDownload.swift:127,208,291` — `weak` capture of `self` differs from an implicitly-captured
  strong reference in the outer scope; the `weak self` does not do what it appears to.
- `Sources/DigUpApp/DebugHooks.swift:271` — passing a non-`Sendable` parameter `done` to a function expecting a
  `@Sendable` closure.

---

## 9. NEW — smaller correctness / robustness items

- **`ParentWatch` is a no-op on Windows** (`#if os(macOS)` / `#elseif os(Linux)` only), and the comment says the shell
  ends its helpers itself. But the shell only stops helpers on `RunEvent::Exit`; a **crash or force-quit of the Tauri
  shell on Windows orphans `index-service` / `search-service`**, which keep walking folders and holding the index. On
  macOS the kernel path and on Linux the `getppid` poll both catch this.
- **`Watchdog.stop` uses `exit()` on Windows but `_exit()` elsewhere.** `exit()` runs CRT atexit handlers and flushes
  stdio in a process being deliberately aborted mid-read; a single `FlushFileBuffers`/handler that blocks defeats the
  watchdog. Use the CRT `_exit` (or `TerminateProcess`) path on Windows too.
- **`Portable/SHA256` swaps in only where `CryptoKit` is absent**, so on this machine it is never compiled and the "the
  way CryptoKit's `SHA256` is used here" equivalence is not covered by any test that runs here. (See §7.)
- **`Portable/Math.scores` assumes `dimension % 8 == 0`** (`let lanes = dimension / 8`, SIMD8). True for 768, but the
  remainder is silently dropped if the dimension ever changes; a `precondition` would make that explicit.
- **`Portable/EstEXIF.bytes(_:_:)` reads byte-by-byte** through a block cache; fine for headers, but `bytes(at, 19)` for
  `DateTimeOriginal` and the `iinf`/`iloc` scans revisit the cache heavily — acceptable, just worth noting it is not a
  bulk reader.
- **`FFmpeg.run` reads stdout on the caller's thread** and sends nothing to stderr (null). Correct for the current
  callers, but any future tool that writes a large amount to stderr while stdout is read could still stall; a stderr
  reader thread would make it robust.

### On the Tauri shell (`shell/src-tauri`)

Reviewed `main.rs` and `helper.rs` by reading. The design — helpers as JSON-line child processes, `ask` whitelisting the
panel's ops, tray + global shortcut, single-instance plugin, `CREATE_NO_WINDOW`/`nice(10)` for background work, reply
routing per process with per-request `oneshot` senders so a dead helper fails only its own requests — is sound and
mirrors the Mac app. Not exercised here (no Linux/Windows host), so §7 is the governing risk.

---

## 10. Tuning / evals (unchanged concern)

The eval deltas in the comments (`24 vs 22`, `91 → 94`, R@1 `0.88 vs 0.73`) remain within noise at the sample sizes used,
and `wordWeight`, `codeWordWeight`, `zGate`, `codeStretch` and the `strongCount` ratio were tuned on the same small eval
set. Treat them as noise until they survive a hold-out with bootstrap intervals.

---

## 11. Limitations of this review

- **Only the macOS build path was compiled, run and tested.** The entire `Platform/Portable/*` layer, the non-macOS
  `Package.swift` branches, the Tauri shell, ffmpeg/PDFium/miniz/stb integrations, and the Windows/Linux `Watchdog` and
  `ParentWatch` paths were assessed **by reading only** and are unverified — see §7.
- The GUI app, Quick Look, FSEvents indexing, Sparkle updates, notifications/permissions, and the Tauri panel JS
  (`shell/ui/panel.js`, 605 lines) were read, not executed.
- The P0 reproduction uses a hand-built 4-document corpus; the medium-corpus (41-vector) false-positive case from the
  prior review was not re-run after the pull, though `Searcher.meaning` is unchanged so its cause stands.
- `ImageSizing`'s bit-for-bit Pillow claim was not re-derived against a real Pillow run; the passing `ImageSizingTests`
  plus the passing image reference vectors remain the evidence.

---

## 12. Bottom line

The port is a large, well-structured piece of engineering, and the `Watchdog` / `MeasuredCosts` / helper-process work is
exactly the kind of hardening the engine needed. Two things should happen before anything else:

1. **Fix the retrieval gate** (§3) — it is the same, reproduced P0, and it defeats meaning search for a whole class of
   indexes.
2. **Add a Linux/Windows CI matrix** (§7) — a cross-platform port with no CI on the non-primary platforms is a port you
   cannot trust.

Then the smaller latent fixes (§4–§9): the estimate startup term, the `<eos>` truncation mismatch, the `ImageSizing`
integer division, the compile warnings, and the Windows orphan-helper gap.
