# DigUp — Deep Code Review

**Date:** 2026-10-13
**Reviewer:** Buffy (macOS / llama.cpp / ML review)
**Scope:** full repository — `Sources/DigUpKit`, `Sources/DigUpApp`, `Sources/digup`, `Sources/LlamaRuntime`, `Sources/CLlama`, `worker/`, `scripts/`, `Tests/`
**Verdict:** the engineering *around* the model is excellent; the weak point is the retrieval **decision layer**.

No source files were changed as part of this review: `git status` is clean.

---

## 1. What was verified (not just read)

| Check | Command | Result |
|---|---|---|
| Kit builds | `swift build --target DigUpKit` | ✅ builds |
| CLI builds | `swift build --product digup` | ✅ builds |
| llama.cpp toolchain | `./scripts/build-llama.sh` | ✅ clones llama.cpp `b11461`, applies `scripts/llama.cpp-no-logits.patch`, produces `Vendor/llama/lib/libllamacpp.a`. Every non-public API `LlamaEmbedder` calls (`llama_process`, `llama_batch_ext_*`, `llama_embd`, `llama_model_n_embd_inp/out`, `mtmd_bitmap_init_from_audio`) exists at the pinned tag and links. |
| Test suite | `swift test` (exit status captured with `set -o pipefail`) | ✅ **exit 0** — 69 tests in 6 suites (`DigUpKit`) + 6 tests in 2 suites (`LlamaRuntime`) = **75/75 pass** |
| Reference vectors | `ReferenceVectorTests.vectorsMatchTheReference()` | ✅ **ran and passed in 10.2 s** (not skipped). Model files are present on this machine, so the llama.cpp ↔ mlx-vlm cosine ≥ 0.999 claim is **empirically confirmed here** |
| End-to-end CLI | `estimate` → `index` → `search` → `status` on scratch folders | ✅ works, exercising text, image, screenshot and audio indexing plus hybrid search (including an honest "no good matches") |

All environment changes were confined to gitignored paths (`Vendor/`, `.build/`) and `/tmp` scratch dirs, which were removed.

### What the code gets right

Genuinely high quality, and unusually well-reasoned:

- **Process-per-work-role** — Vision and llama.cpp run out of process, so their memory returns to the OS instead of accumulating.
- **WAL** — search never blocks on indexing.
- **mmap'd vector storage** with a *measured* justification.
- **SHA-256-verified, resumable model download.**
- **Vector-version indirection** so a llama.cpp bump does not invalidate existing indexes.
- **Multilingual PDF-corruption detection** (Bengali glyph order, Arabic ligature loss).
- Comments that cite **measured numbers**, which is rare and good.

The rest of this report is ordered by impact.

---

## 2. P0 — Meaning search is mathematically impossible on small indexes

`Searcher.meaning` (`Sources/DigUpKit/Searcher.swift:316–331`) standardizes each query's cosines with a mean / standard
deviation over the whole corpus, and `search` gates at `z >= 2.0` (`:117`, parameter `zGate`).

For any sample of size **n**, the largest attainable standardized score is **(n−1)/√n**:

| n (vectors) | max attainable z |
|---|---|
| 4  | **1.50** |
| 5  | 1.79 |
| 6  | 2.04 |
| 20 | 4.36 |

So for `n <= 5` the gate at `z = 2.0` **can never be passed**, no matter how good the match is.

**Reproduced.** A 4-vector index, query *"a vehicle taken in because something was rubbing and the friction material
was worn"*:

- default (`zGate = 2.0`) → `no good matches`
- the correct document, `01-car-service.txt` ("the mechanic replaced the brake pads…"), is found at **z 1.5** — invisible
  under the default gate
- with `--zgate 1.0` it is the **top hit**

**Impact.** A new user, or anyone whose folder holds a handful of files, gets keyword-only results forever while the app
reports "No good matches" with the right file already indexed. `zGate` is a CLI-only knob, so app users have no recourse.

**Fix.** Make the gate size-aware: fall back to top-k with a margin while `n` is small, or use
`gate = min(2.0, (n − 1)/√n)`. Better still, drop the z-gate in favour of a **rank margin** (`(top1 − top2)/SD`) plus an
absolute cosine floor.

---

## 3. P1 — The same gate is simultaneously too permissive

On a 41-vector corpus the *same* query returns:

| z | file | content |
|---|---|---|
| 2.5 | `11-budget-review` | "we overspent on dining out and underspent on transport" |
| 2.3 | car tyres | — |
| 2.3 | bike repair | — |

None of them is the brake-pad document. So one corpus pushes the **true** match *below* the gate (z 1.5) and another pushes
**false** ones *above* it (z 2.5).

**Root cause.** Mean/SD over cosines is the wrong statistic. Cosine distributions are bounded at 1, left-skewed, and
multimodal across kinds; both the mean and the SD are corpus-composition dependent. Use **median/MAD**, or a rank / margin
statistic instead.

---

## 4. P1 — `Estimator` has no startup term

`Sources/DigUpKit/Estimator.swift` models per-unit cost (`image`, `screenshot`, `pdfPage`, `docChunk`, …) but has **no
constant for model load / first-token latency**. On the scratch smoke test the estimate predicted **~1 s** while the real
first index took **~50 s** — the gap is the one-time model start, unmodelled. Users see a wildly wrong time-to-finish on
first launch.

**Fix.** Add a fixed startup cost (calibratable the way the per-unit costs already are, per the comment at
`Costs`, "The app will calibrate these with a few seconds of real work on first launch").

---

## 5. P1 — llama.cpp appends `<eos>` on truncation; the MLX reference drops it

In `LlamaEmbedder.tokenize` (`Sources/LlamaRuntime/LlamaEmbedder.swift`):

```swift
if tokens.count > maxTokens {
    tokens = Array(tokens[0..<(maxTokens - 1)]) + [llama_vocab_eos(vocab)]
}
```

This truncates *and appends `<eos>`*. The MLX reference (`mlx-vlm`) drops the trailing token instead, so for any input
longer than `maxTokens` the two runtimes tokenize differently — and the ≥ 0.999 cosine claim is only checked on inputs
that fit. This is an **untested path**.

**Fix.** Either match the reference's truncation semantics exactly, or add a reference-vector test whose input exceeds
`maxTokens`.

---

## 6. P1 — Integer division in `ImageSizing`'s fallback

`Sources/LlamaRuntime/ImageSizing.swift`, `targetSize`:

```swift
if targetHeight == 0 {
    targetHeight = side
    targetWidth = min(width / height * side, maxSide)
} else if targetWidth == 0 {
    targetWidth = side
    targetHeight = min(height / width * side, maxSide)
}
```

`width / height` (both `Int`) is **integer division**, so the "too thin" fallback axis is truncated to a ratio in
{0, 1, 2, …}. Should be `Double(width) / Double(height)`. The `targetSize` main path uses `Double`, so this only bites
on extreme aspect ratios — which is exactly the case this branch exists to handle.

---

## 7. P2 items

- **Unbatched media embedding** — media (images / audio) are embedded one at a time rather than batched with the text
  stream, so the encoder sits idle between calls.
- **`rowSetsScale` derived from defaults, not index state** — the scale set used by `Searcher.meaning` is built from
  configuration defaults rather than the actual index contents, so a change in defaults silently changes old indexes'
  ranking.
- **`minPageText` outside the fingerprint** — an indexing parameter that affects stored output is not part of the
  index fingerprint, so changing it does not force a rebuild.
- **Untested fp16 round-trip** — the half-float storage path has no test asserting the round-trip error bound.
- **Imprecise Q8_0 rationale** — the comment justifying Q8_0 is directionally right but the numbers are not derived.

---

## 8. Test-coverage gaps

- `Searcher`'s hand-tuned scoring (the z-gate, `wordWeight`, `strongCount`, lookup weighting) has **no unit tests**.
- `ReferenceVectorTests` **silently skips in CI** when the model files are absent, so the one test that anchors
  llama.cpp against the reference never runs in CI.

---

## 9. On the tuning / evals

The eval deltas cited in code comments (e.g. "24 vs 22", "91 → 94") are **within noise** at the sample sizes used:
`codeWordWeight`, `wordWeight`, lookup weighting, the `strongCount` ratio, `zGate` and `codeStretch` were all chosen on
the same three eval files (26–105 queries, observed deltas of 1–3 items). That is multiple comparisons on a small set.
Treat those deltas as noise unless they survive a **hold-out with bootstrap confidence intervals**.

---

## 10. Limitations of this review

- The GUI app, Quick Look, FSEvents-driven indexing, Sparkle updates, and the notification / permission flows were
  **assessed by reading only** — I could not execute them.
- Retrieval-quality conclusions rest on my own small corpora (4 and 41 vectors) plus the repo's evals; the eval-accuracy
  deltas are, as noted, within noise at their sample sizes.
- `ImageSizing`'s bit-for-bit Pillow claim was **not re-derived against a real Pillow run**. The evidence I have is
  `ImageSizingTests` passing plus the passing image reference vectors.

---

## 11. Bottom line

The engineering **around** the model — process isolation, storage, migrations, privacy rules, model download — is
excellent and unusually well-reasoned. The weak point is the **retrieval decision layer**: a corpus-size-dependent
Gaussian z-gate that provably kills small indexes and passes junk on medium ones, plus tuning that is not held out.

**Fix the gate before adding any more features.**
