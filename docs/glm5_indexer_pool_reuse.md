# GLM-5.3 Indexer pool reuse

GLM-5.3 decode runs the DSA Indexer over the whole context on every step. This
option approximates later selections by building a larger candidate pool on
one step and only re-ranking that pool on the next N steps. Quality depends on
how much the required tokens change between steps:

- **Refresh step:** score the full context (DeepGEMM `fp8_paged_mqa_logits`)
  and keep the top `GLM5_INDEXER_POOL_SIZE` (default 8192) token indices as the
  request's pool.
- **Refresh output:** return the original TopK selector's full-context result.
  Re-ranking the candidate pool can change near-ties and tie ordering, so it
  must not replace that result on a refresh step.
- **Reuse output:** score the pool plus the tokens appended since the refresh
  (at most 63) with a fused Triton kernel reading the paged FP8 Indexer cache,
  then select top-2048 with the existing TopK kernel.
- **Short rows:** rows shorter than `GLM5_INDEXER_POOL_MIN_KV_LEN` refresh on
  every step and return the original full-context TopK result without building
  a candidate pool.

Refresh rows skip candidate scoring/selection work. Row lengths select full
or candidate work on device, keeping mixed refresh/reuse rows compatible with
CUDA graphs. Performance depends on batch size, context length, and the
native TopK backend; reuse can be slower for short contexts or small batches.

## Controlled accuracy re-evaluation (2026-10-28)

The original uncapped run on 100 frozen, approximately 48K CWE prompts scored
9.2% answer-word recall for baseline and 1.6% for 8K/N15, with 67 and 69
confirmed generation loops counted as zero. That baseline is invalid for
judging approximation quality: its first eight prompts scored 87.5%, the
remaining 92 scored 2.39%.

An independent EP8 GPU reproducer identified a MoE chunk-scheduling problem
in the fused DP8/EP8 test deployment. The wrapper's rank-local capacity was
17280 tokens, so a 48010-token prefill invoked three MoE kernels per layer.
Idle ranks invoked one kernel per layer and advanced to different layer
weights while the active rank was still processing its earlier layer's
chunks. With three distinct layers, the misaligned result had relative L2
error 0.635; matching the chunk count on every rank produced bitwise-identical
outputs to the balanced-load reference.

The controlled rerun sets `DSV4_MOE_CHUNK_TOKENS=65536` on both services, above
this test's `MAX_BATCH_TOKENS_SIZE=57344`. This avoids chunking for this bounded
cohort; it is not a general fix for arbitrary DP/CP inputs. It keeps 1M context
capacity, greedy generation with no extra output cap, explicit DP backend
routing, and the same frozen 100 prompts. Confirmed loops count as failures
with zero score. Both runs use TP1/DP8/EP8, FP8 KV, eager execution, native
`dsv4_persistent` decode TopK, and `topk_v3_tie_break` prefill TopK. The same
private control hook is present in both runs; it reads configuration on each
forward and records prefill metadata, so these are not production latency
measurements.

| Metric | Baseline | Fixed pool 8K, N=15 |
|---|---:|---:|
| Frozen CWE prompts | 100 | 100 |
| Answer-word recall | 100% | 100% |
| All ten words correct | 100/100 | 100/100 |
| Natural completions | 100/100 | 100/100 |
| Confirmed loops / request errors | 0 / 0 | 0 / 0 |
| Mean output tokens | 239.96 | 237.33 |
| Maximum output tokens | 1740 | 953 |
| Median request TPOT, ms | 154.452 | 166.912 |
| P95 request TPOT, ms | 194.081 | 226.316 |

All 100 paired scores tie. Median TPOT increased 8.07% in this approximately
48K, one-request-per-DP-rank eager experiment. Output trajectories and mixed
prefill/decode scheduling differ, so this is a descriptive measurement rather
than an isolated performance comparison. It does not establish production
CUDA-graph latency or SWE quality.

A separate native profiler request recorded 651 launches each of the pool
plan, pool selection, candidate scoring, and result mapping kernels: 31 decode
steps across 21 Indexer layers. Candidate scoring takes the short refresh path
on steps 1 and 17, with active scoring on intervening steps, consistent with
one refresh followed by 15 reuse steps. This trace is a path check, excluded
from the accuracy and latency measurements.

The corrected environment and the pool fixes were evaluated together. Do not
attribute the change from the old corrupted run solely to the pool fixes;
there is no original-pool ablation in the corrected environment.

The pool changes separately fix request-page recycling and preserve the
native full-context TopK result on refresh steps. Both native decode TopK
backends pass the 19 GPU regression checks listed below. Accuracy aligns on
this frozen CWE cohort; broader task validation remains necessary. The
historical latency and quality tables below predate the fixes and the
corrected baseline environment.

## Historical end-to-end results (before the baseline correction)

These earlier runs predate the controlled 100-prompt validation above and
used an unhealthy baseline. Keep the approximation experimental and disabled
by default; one corrected CWE cohort does not establish general equivalence.

On eval80, the baseline and pool runs used the same GLM-5.3 model, FP8 KV,
DP8/EP8, CUDA graphs (capture sizes 1/2/4/8), and 47,740–48,124-token inputs.
The only difference in saved service settings was
`GLM5_INDEXER_POOL_REUSE_STEPS=15`. Logs contain 168 pool initialization
messages (21 Indexer layers × 8 ranks), with pool=8192, reuse=15,
min-KV=16384 and slots=128, plus CUDA graph capture messages for both runs.
These logs confirm configuration and initialization; they are not per-step
refresh/reuse counters.

| Metric | Baseline | Pool 8K, N=15 |
|---|---:|---:|
| RULER-CWE answer-word recall, 24 prompts | 0.5875 | 0.3417 |
| RULER request errors | 0/24 | 0/24 |
| Decode latency median, ms/token | 16.5654 | 19.6208 |
| Decode latency mean, ms/token | 16.5713 | 19.9204 |
| Decode request errors | 0/64 | 0/64 |

Decode used 64 concurrent requests across 8 DP frontends, each generating
exactly 512 tokens. Latency is the per-request server-reported
`(cost_time - first_token_cost_time) / (output_len - 1)`, then aggregated
across requests. Median latency increased 18.44%; mean increased 20.21%.
This is one end-to-end run, not an isolated kernel timing or a scaling study.

RULER used greedy generation limited to 120 new tokens. Scores measure the
fraction of ten answer words present in that output, not full-answer accuracy.
Paired scores improved for 6 prompts, tied for 4, and worsened for 14.
Eight pool outputs repeated `</think>` 120 times (none in baseline);
other pool outputs also included repetition or irrelevant answers. Therefore
the observed regression cannot be explained solely by the output length cap.
The MoE scheduling defect described above prevents this run from isolating
approximation loss. Successful requests and kernel tests do not establish
model quality; use the controlled rerun for the new comparison.

The tested remote commit was `346f1713c70b6fdb8d040a8af66e2b629935ffd2`;
its tree equals local commit `f714a35a4e305a26d8019647c4b68718cf00ef72`.
The subsequent formatting commit `15025aeda6c7928320394177b316aba2f4126d9a`
preserves the Python AST of every changed file. The run completed at
14:16 CST. Remote evidence is under `/opt/src/glm53-pool-build/e2e/`, in
`run-base` and `run-pool8k` (settings, logs, RULER outputs, and decode JSON).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GLM5_INDEXER_POOL_REUSE_STEPS` | `0` (off) | N: steps that reuse a pool after its refresh, 1 to 63 |
| `GLM5_INDEXER_POOL_SIZE` | `8192` | Pool size; a power of two of at least 2048 |
| `GLM5_INDEXER_POOL_MIN_KV_LEN` | `16384` | Rows shorter than this refresh every step |
| `GLM5_INDEXER_POOL_SLOTS` | `128` | Pools per Indexer layer, a power of two; larger decode batches use the original path |

Memory: each Indexer layer keeps `SLOTS * POOL_SIZE * 4` bytes. That is 4 MiB
at the defaults, 84 MiB over GLM-5.3's 21 Indexer layers. Each call also
needs temporary scratch of 32 bytes per row per pool entry.

Scope: the option applies to single-token decode only. Prefill, MTP
target-verify/draft-extend, and batches larger than the slot count use the
original path. It requires top-2048, 128-dim FP8 Indexer keys and 64-token
pages.

## CUDA graphs and batching

Everything is decided on the device, so one captured graph per batch size
serves both refresh and reuse steps. All batch sizes share the per-layer slot
state, which is allocated before capture.

Rows find their slot through KV pages rather than request ids, with no engine
changes:

- the row's first physical page;
- the page that held the refresh step's last token, which was private to the
  request when the pool was built.

A reordered or shrunk decode batch therefore finds its pools again. Physical
pages alone do not identify a request: a new request can recycle both pages.
Prefill resets pools, and decode's original input-length metadata forces
the first decode step to refresh, including on a separate P/D decode worker.
An oversized decode batch or MTP verify/extend path also resets pools, so later
supported batches cannot revive stale slots. This includes MTP's key-only fast
path: a new request can reach ordinary decode after its initial prompt+1 step.
Other live rows rebuild after a reset. Padded graph rows have zero length and
do no work.

## Kernels per step (per Indexer layer)

| Kernel | Notes |
|---|---|
| plan | One program. Matches rows to slots, assigns free slots, emits per-row lengths |
| DeepGEMM metadata + logits | Lengths are zero for rows that reuse their pool, so a reuse step does no work here |
| count / compact / select | Builds the pool for refresh rows at or above the minimum length. A sampled floor keeps about 2× the pool size per row, compacted in token order by 4096-token segments; multiway bisection then picks the pool, allowing near-tie swaps |
| candidate logits | On reuse rows only: fused gather + dot over FP8 pool keys and appended tokens |
| Full TopK | Existing selector on refresh rows; zero-length reuse rows are ignored |
| Candidate TopK + map | Existing selector, then candidate positions back to token indices; refresh rows retain full TopK |

## Kernel validation (eval80, L20D, DeepGEMM 2.8.0+6d6aaad)

`test/test_indexer_pool_reuse.py` (10 tests),
`test/test_indexer_pool_deepgemm.py` (3 tests), and the hybrid module's
`test/indexer_pool_lifecycle_test.py` (6 tests) pass together. They cover:

- candidate logits against a torch reference and against DeepGEMM, within 2e-5;
- refresh steps preserving the original selector, including a deterministic
  tie policy; independent persistent TopK calls may reorder an identical set;
- pools against the exact top-8192;
- reordered, new, short and changed-page rows;
- slot recycling;
- 4K and 16K pools;
- CUDA graph replay at batch sizes 1, 4, 8 and 16 against eager;
- page recycling at prefill, first decode without local prefill, and oversized
  batch fallback, including the production Indexer entry;
- MTP key-only and paged-backend fallback boundaries;
- crossing the minimum KV length after skipping pool construction.

## Isolated kernel timing with native topk_v3 (2026-10-28)

These measurements use an NVIDIA L20D with the evaluation model stopped.
Leaving the DP8 service running is not isolation: idle ranks still execute
dummy forwards and disturb timings. A 16-step CUDA graph cycle includes one
refresh and 15 reuse steps, reset, length updates, and DeepGEMM metadata.
The block table has 1M-token capacity. Results are synthetic Indexer-core
medians, not production model speedups.

| Decode rows | KV tokens | Baseline µs | Fixed 8K/N15 µs | Baseline / pool |
|---|---:|---:|---:|---:|
| 1 | 48000 | 18.183 | 31.219 | 0.58x |
| 16 | 48000 | 33.697 | 46.339 | 0.73x |
| 64 | 48000 | 92.111 | 98.529 | 0.93x |
| 1 | 128000 | 20.828 | 31.055 | 0.67x |
| 16 | 128000 | 67.714 | 51.090 | 1.33x |
| 64 | 128000 | 217.698 | 115.028 | 1.89x |

The 8K/N3 and 16K/N3 kernel variants were also measured, but are not quality
claims: the 100-prompt evaluation used 8K/N15. Shortening the reuse window
increases refresh cost. Keep default reuse disabled, and validate both task
quality and latency for the actual backend, context lengths, and batch sizes.

## Historical latency per Indexer layer, CUDA graph, N = 15

From `test/bench_indexer_pool_reuse.py`; "avg" is (refresh + 15 reuse) / 16.

| BS | ctx | original µs | pool 8K avg µs | 8K speedup | 16K speedup | 32K speedup |
|---|---|---|---|---|---|---|
| 1 | 128K | 34.5 | 42.3 | 0.82x | 0.72x | 0.54x |
| 8 | 128K | 56.2 | 42.9 | 1.31x | 1.08x | 0.78x |
| 32 | 64K | 88.9 | 57.9 | 1.54x | 1.11x | 0.67x |
| 32 | 128K | 137.1 | 63.5 | 2.16x | 1.57x | 0.97x |
| 64 | 64K | 130.3 | 82.9 | 1.57x | 1.10x | 0.72x |
| 64 | 128K | 238.2 | 99.0 | 2.41x | 1.77x | 1.19x |

## Recall on real decode logits

**Historical diagnostic only:** these logits were later found to come from
an older clustered-NVFP4 runtime whose generated outputs were corrupted.
Do not use the following numbers as validated model recall or accuracy.
The end-to-end results above supersede the earlier pending-evaluation status.

Run with `test/replay_indexer_pool_recall.py` on 3 RULER-CWE requests at about
130K context: 21 layers × 119 decode steps. Recall is the share of the
selected top-2048 whose logit reaches the exact 2048-th logit.

| Pool | N | Mean | p10 |
|---|---|---|---|
| 8192 | 15 | 0.804 | 0.373 |
| 8192 | 4 | 0.837 | 0.416 |
| 16384 | 15 | 0.884 | 0.587 |
| 32768 | 15 | 0.936 | 0.828 |
| stale top-2048 | 15 | 0.51 | 0.13 |

Recall barely depends on the distance from the refresh: one step after it, the
mean is already 0.82. The loss comes from steps where the Indexer's target
moves to a region outside the pool, so a larger pool helps more than a smaller
N within this historical dataset. This does not establish task accuracy.
