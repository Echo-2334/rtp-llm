# GLM5 prefill PIVOT-Refine (experimental)

This opt-in path groups adjacent prefill queries, recalls a shared candidate
set with a mean proxy, and re-scores candidates independently for each query.
It returns the existing Top-2048 interface. It is an approximation to the
full-context Indexer, not a numerically equivalent optimization.

## Configuration

Set these on the **prefill** worker:

```json
[
  ["GLM5_PREFILL_PIVOT_MODE", "refine"],
  ["GLM5_PREFILL_PIVOT_GROUP_SIZE", "4"],
  ["GLM5_PREFILL_PIVOT_CANDIDATES", "4096"],
  ["GLM5_PREFILL_PIVOT_LOCAL_WINDOW", "4"],
  ["GLM5_PREFILL_PIVOT_MIN_KV_LEN", "32768"],
  ["DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS", "2048"],
  ["ENABLE_CUDA_GRAPH", "0"]
]
```

`MODE` defaults to `off`; omitting it preserves native Indexer selection.
The other four PIVOT values shown above are the initial defaults when
enabled. Group size supports 2 or 4. Candidates must exceed
2048 and the window union, and must not exceed 16384. Window must be at least
group size. Minimum visible KV length must be at least the candidate budget.
The minimum is per query, **not the batch's concatenated KV length**.

This version supports 32 Indexer heads, head dimension 128, Top-2048 and one
FP8 scale per head/key. Ordinary and CP prefill share the implementation.
Decode, target verify and draft extend retain their existing dispatch, and
the decode `GLM5_INDEXER_POOL_*` options remain independent.

## Algorithm and boundaries

1. Keep the existing Q/K projections, positional transforms, quantization,
   cache writes and full K assembly. CP collective ordering is unchanged,
   including ranks with no local queries.
2. Process at most 2048 original query rows per chunk. A smaller positive
   `DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS` is honored. Zero still uses 2048 for
   PIVOT to bound temporary logits memory.
3. Form fixed tiles of `g` rows. Only use a tile if all rows have equal
   ragged K starts/output offsets, consecutive exclusive K ends, and enough
   visible KV. Request boundaries, CP zigzag discontinuities, short prefixes
   and incomplete tails use the original full-context logits/TopK backend.
4. Dequantize Q with its original scale, average Q and the unscaled gate
   separately per head, then quantize the proxy with a power-of-two FP8
   scale. The input gate already contains Q scale; averaging that fused
   quantity directly is incorrect.
5. Reserve `W + g - 1` positions for the union of the group's local windows.
   Scan only the prefix preceding that union and select the remaining
   `C - (W + g - 1)` candidates. Excluding the window from the proxy prefix
   avoids duplicate candidate IDs. The native candidate TopK reads only the
   valid ragged interval, so this path does not need a full-logits mask.
6. Fuse relative candidate-ID conversion, validity checks and window append.
   On SM100-family GPUs with group size 4, pack each group's candidate K/scale
   into a 256-token-aligned interval and use DeepGEMM `fp8_mqa_logits` with
   compressed output (`max_seqlen_k`). A fused packing kernel also prepares
   interval metadata. Mask by each row's original `ks/ke`, including
   group-internal future tokens. Group size 2 and other GPU architectures
   retain the grouped Triton scorer: compressed MQA's four-query tile cannot
   mix two different packed K origins.
7. Run the configured native prefill Top-2048 on candidate scores, map the
   selected candidate slots to original K IDs, check finite scores and causal
   bounds, subtract each row's ragged K start to recover native request-relative
   indices, and scatter with output offsets in one kernel. Invalid slots remain
   `-1`. Explicit canonicalization retains the existing sort/write path and
   still applies offsets exactly once.

The local window guarantees candidate availability, not forced membership
in the final Top-2048. Mean proxies use all queries in a prefill group;
although final KV reads are causal, selection is approximate and may differ
with prefill chunk/group boundaries. No pool persists between forwards.

## Limitations and validation

- Group validity, compaction and fallback-row construction use one Triton
  kernel. Two counts are copied to the host together, requiring one host
  synchronization per nonempty chunk. CUDA graph capture remains rejected.
- Candidate recall uses native `dsv4_top_k_per_row_prefill` with radix sorting
  when the recall width is at most 4096 (4089 for the default config).
  Larger recall widths retain masked `torch.topk` to avoid exceeding the
  native kernel's dynamic shared-memory budget. The configured final
  Top-2048 backend remains unchanged. Equal-score candidate ties have no
  guaranteed index order.
- Default SM100 group-4 scoring adds approximately 264 MiB of packed K/scale
  workspace at 2048 queries and 4096 candidates. Workspace is chunk-local;
  no request state or persistent pool is introduced.
- Full CP K gathering/assembly is retained. This version does not reduce
  CP communication volume.
- This is a prototype for quality/performance evaluation, not evidence of
  GLM5.3 RULER/SWE parity or end-to-end speedup. Benchmark the entire new
  path, including planning, proxy construction, recall, refine and writes.

GPU tests:

```bash
python -m unittest \
  rtp_llm.models_py.triton_kernels.sparse_mla.test.test_prefill_pivot -v
```

The tests check unequal Q scales, production DeepGEMM score agreement,
causal masks, request/CP boundaries, local-window coverage, unique IDs,
native fallback, output offsets, both final TopK backends, and exclusion of
decode/MTP from PIVOT. Model evaluation should isolate prefill PIVOT by
keeping decode reuse disabled in both baseline and experiment.

## GLM5.3 validation (2026-10-29)

The fused implementation passed 49 GPU tests on L20D. A paired model run
used the same frozen RULER CWE 100 prompts (approximately 48K input tokens),
answers, generation settings and rank routing. Both arms disabled decode
pool reuse. PIVOT used the configuration above. There was no additional
output cap beyond the 1M context limit; confirmed repetition loops would
have been stopped and scored as failures.

| Metric | Native baseline | Fused prefill PIVOT |
|---|---:|---:|
| Answer-word recall | 100% | 99.9% |
| All ten words correct | 100/100 | 99/100 |
| Natural completions | 100 | 100 |
| Request errors / repetition loops / context-limited outputs | 0 / 0 / 0 | 0 / 0 / 0 |
| Median TTFT | 3.378 s | 3.793 s |

The sole regression was sample 032: `ephyra` became `eph(y)ra`, scoring
0.9 with the unchanged case-insensitive substring scorer. All eight ranks
recorded actual packed-DeepGEMM PIVOT execution. This is evidence of a
working model path, not accuracy equivalence or an end-to-end speedup.

The run used TP1/DP8/EP8 eager fused serving, not production CP8 with
separate prefill/decode workers. CP boundary unit tests do not replace that
deployment validation. TTFT includes JIT and concurrent scheduling effects;
the two arms also produced different output trajectories. No SWE evaluation
was performed for this prefill change.

## Ragged output-coordinate fix (2026-10-30)

The initial implementation returned assembled-K indices for refined rows,
while native TopK and fallback rows returned request-relative indices. For a
second request whose K starts at 40000, this incorrectly shifted every refined
index by 40000. CP gathered attention adds a request workspace offset itself,
so the initial output could read the wrong request region or go out of bounds.
The remap kernel now subtracts the row's K start before applying the caller's
output offset. Zero-origin single-request behavior is unchanged.

A reproducer with identical queries and sufficient candidates had 0% selected
set overlap for the second request before the fix and 100% after it. GPU
regression coverage includes nonzero request origins, refined/fallback rows,
4K/16K budgets, both final TopK backends, canonicalization, and the CP entry
with identity/permuted query layouts (transport mocked); 51 tests passed.

`MAX_CONTEXT_BATCH_SIZE=1` is not a singleton admission guarantee in this
branch's FIFO scheduler. With CP enabled and `CP_FORCE_SINGLE_PREFILL=0`,
multiple requests may be admitted subject to concurrency and token/KV limits.
The previous single-request-per-rank RULER run did not exercise this bug.
The user's SWE degradation has not yet been reproduced end-to-end, so this
confirmed bug is not proof that all observed quality loss has been explained.

A separate shadow run on eight approximately 48K RULER CWE prompts sampled
64 queries in each of 21 active indexer layers (10752 query/layer observations
per variant), keeping the model trajectory native. Mean native Top-2048
candidate coverage was 88.67% for g4/C4096, 95.86% for g4/C8192 and 98.93% for
g4/C16384. Final selected-set overlap matched coverage on every sampled g4
row; candidate scores matched the native full-context scores exactly. However,
the minimum g4/C16384 coverage was only 18.99%, at a final prompt query in
layer 0. These zero-origin measurements isolate approximation from the ragged
coordinate bug; they are neither SWE accuracy nor proof of CP8 correctness.
