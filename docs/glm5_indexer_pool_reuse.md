# GLM-5.3 Indexer pool reuse

GLM-5.3 decode runs the DSA Indexer over the whole context on every step. Its
top-2048 changes slowly between neighbouring steps, so this option builds a
larger candidate pool on one step and only re-ranks that pool on the next N
steps:

- **Refresh step:** score the full context (DeepGEMM `fp8_paged_mqa_logits`)
  and keep the top `GLM5_INDEXER_POOL_SIZE` (default 8192) token indices as the
  request's pool.
- **Every step, refresh included:** score the pool plus the tokens appended
  since the refresh (at most 63) with a fused Triton kernel that reads the
  paged FP8 Indexer cache directly. Then select top-2048 with the existing
  TopK kernel. On a refresh step this is the full-context top-2048, up to
  near-ties, since Triton and DeepGEMM sum heads in a different order.
- **Short rows:** rows shorter than `GLM5_INDEXER_POOL_MIN_KV_LEN` refresh on
  every step. Rows of at most 8192 tokens keep all their tokens in the pool,
  so their results stay exact.

## End-to-end status (2026-10-27)

The 8K pool / N=15 implementation completes end-to-end inference, but has
**not passed quality validation**. Keep it experimental and disabled by default.

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
The root cause (approximation versus an implementation/state issue) remains
unresolved; successful requests and kernel tests do not prove model quality.

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

A reordered or shrunk batch therefore finds its pools again, and a new request
never inherits one. Padded graph rows have zero length and do no work.

## Kernels per step (per Indexer layer)

| Kernel | Notes |
|---|---|
| plan | One program. Matches rows to slots, assigns free slots, emits per-row lengths |
| DeepGEMM metadata + logits | Lengths are zero for rows that reuse their pool, so a reuse step does no work here |
| count / compact / select | Builds the pool. A sampled floor keeps about 2× the pool size per row, compacted in token order by 4096-token segments; an exact multiway bisection then picks the top pool size |
| candidate logits | Fused gather + FP8 dot over pool + appended tokens |
| TopK + map | Existing `dsv4_persistent_topk` (or `topk_v3`), then candidates back to token indices |

## Kernel validation (eval80, L20D, DeepGEMM 2.8.0+6d6aaad)

`test/test_indexer_pool_reuse.py` (7 tests) and
`test/test_indexer_pool_deepgemm.py` (3 tests) pass. They cover:

- candidate logits against a torch reference and against DeepGEMM, within 2e-5;
- refresh steps against the original Indexer;
- pools against the exact top-8192;
- reordered, new, short and changed-page rows;
- slot recycling;
- 4K and 16K pools;
- CUDA graph replay at batch sizes 1, 4, 8 and 16 against eager.

## Latency per Indexer layer, CUDA graph, N = 15

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
