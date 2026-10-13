# MLX vs llama.cpp: measured comparison

Date: 2026-10-14 · Machine: M1 MacBook Pro, 16 GB · Binaries: release (`swift build -c release`)
Method: `digup speed --backend … --json` (20 text queries ×2, one 512×384 gradient PNG @70/@280,
8 s 440 Hz sine) plus a worker-protocol batch test (8× `photo.jpg` in one image op, per budget).
The app's indexing helper was paused for the timed runs and resumed after.

## Verdict

No — MLX is not faster overall. It **wins text throughput (1.1×) and batched high-budget
images (1.15×)**, ties single high-budget images, trails low-budget/batched-small images (0.8×),
and trails badly on audio (0.17×), cold start (3–4× slower), and peak memory (3.4×).

## Numbers

| | llama.cpp Q8_0 | MLX bf16 | ratio |
|---|---|---|---|
| text-only cold load | 0.6–0.8 s | 2.4 s | 0.3× |
| texts/s (batch 20, warm) | 197–199 | 218 | 1.1× MLX |
| images/s @70, single input | 4.7–4.8 | 0.6 | 0.13× (cold-pipeline artifact; see vision) |
| images/s @70, batch 8 | 4.51 | 3.71 | 0.82× |
| images/s @280, single input | 0.8–0.9 | 0.9 | ~1.0× |
| images/s @280, batch 8 | 0.86 | 0.99 | 1.15× MLX |
| audio seconds per second (8 s clip) | 32.8–33 | 5.7 | 0.17× |
| full-model load (warm pipelines) | 1.0–1.5 s | 0.87 s | ~1× |
| peak RSS | ~760 MB | ~2,550 MB | 3.4× |

Caveats:

- Single MLX run (rerunning meant pausing the live re-embed again). llama.cpp ran twice and was
  stable to ±3%.
- Earlier debug-build numbers understated MLX: Swift `-O0` overhead falls almost entirely on
  MLX's Swift-driven graph, and the first debug runs were contended by the app's own helper.
- The MLX @70 single-image figure is polluted — it is the run that compiles the cold vision
  pipeline. Batch-8 @70 (0.82×) is the honest number.

## Text: MLX wins (1.1×)

Both engines are bandwidth-bound on big matmuls. Q8_0 moves 1 byte/param, bf16 moves 2 — MLX
starts with a ~2× bandwidth handicap and still wins, because batch-20 amortizes its Swift dispatch
and its fused SDPA beats llama.cpp's encode path per token. This is the workload MLX was
hypothesized for (`docs/mlx-backend-plan.md` §1), and it checks out.

## Vision: MLX wins at 280 tokens, loses at 70

Same bandwidth handicap, but the crossover is dispatch amortization. At 280 tokens — especially
batched ×8, which is how the indexer actually runs images (`Indexer.process` batches images
8 at a time) — each `matmul` does real work per dispatch and MLX's kernels win (1.15×). At 70
tokens single input, fixed costs dominate: per-op dispatch from Swift plus first-touch pipeline
compile, and llama.cpp's single fused graph wins. Practical upshot: screenshots (budget 280)
favor MLX; thumbnails and video frames (70) favor llama.cpp.

## Audio: MLX collapses (0.17×)

Structural, not tuning. The conformer does chunked attention by explicitly slicing 9 overlapping
key/value windows per block (`Sources/MLXRuntime/Model/AudioTower.swift:blockedContext`), plus a
pad/split/reshape chain for relative positions (`relativeScores`) — hundreds of tiny Metal
dispatches per 8 s clip, with the mask rebuilt per block. llama.cpp runs the same math as one
fused ggml graph. No weight format changes this; it needs `mx.compile` fusion of the attention
path or a rewritten block-sparse kernel.

## Cold load: MLX 3–4× slower (2.4 s vs 0.6 s)

Anatomy: 1.4 GB of safetensors parsed (vs 865 MB mmap'd GGUFs), vision f32 upcast, then the two
warm-up embeds compile cold Metal pipelines. The full-model load timed *faster* (0.87 s) only
because it reuses already-compiled pipelines in-process. The user-visible cost is the query
encoder: a separate text-only process, cold after every idle exit — panel-open-after-idle pays
~2.4 s on MLX vs ~0.6 s on llama.cpp. Warm queries then favor MLX (1.1×).

## Memory: 3.4× (2.55 GB vs 0.76 GB)

Anatomy: 1.42 GB bf16 weights resident (llama.cpp mmaps 865 MB Q8_0, mostly unresident) plus
~0.3 GB vision f32 upcast plus the 512 MB GPU cache limit plus f32 attention temporaries
(`Watchdog.footprint` = `phys_footprint`). Over the plan's 1.35× gate by a mile; matters most
on 8 GB Macs, where 2.5 GB plus OCR plus app starts to squeeze.

## End-to-end reality check

The live MLX re-embed measured ~1 file/s through screenshot-heavy folders, vs `costs.json`'s
llama-measured screenshot cost of 1.18 s each (~0.85 files/s). Rough parity — because screenshot
indexing is **OCR-bound** (Vision ~0.2 s runs beside the model and usually sets the pace,
`Sources/DigUpKit/Platform/Apple/Speed+Apple.swift:16`). When extraction dominates, the engine
barely moves wall-clock. Engine choice matters for audio/video-heavy folders and query latency,
not for screenshot backfills.

## What would flip each deficit

- Audio: `mx.compile` on the conformer attention path (biggest single win available), else a fused
  chunked-attention kernel.
- Memory and load bytes: 8-bit group-64 recipe — halves weight bytes, directly attacks both RSS
  and load time. Still the route to the default flip.
- Cold query latency: keep a warm text-only encoder (app design change, engine-independent) or
  accept ~2.4 s after idle.
- Nothing needed for text or @280 images — already at or above parity.

## Bottom line

`automatic` stays llama.cpp. The evidence is sharper than "0.5–1.0× everywhere": MLX leads text
and batched high-budget vision, ties @280 single, trails @70 (0.82× batched), and trails badly
on audio (0.17×), cold start (3–4×), and memory (3.4×). If the 8-bit + compile project is ever
scoped, audio and memory are the acceptance tests; text and screenshots already pass.
