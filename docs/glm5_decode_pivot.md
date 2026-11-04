# GLM-5 MTP3 decode PIVOT

Experimental, opt-in PIVOT-Refine for target verification with three draft tokens
and four target query rows per request. It is stateless across verification rounds.

```sh
GEN_NUM_PER_CIRCLE=3
GLM5_DECODE_PIVOT_MODE=refine
GLM5_DECODE_PIVOT_GROUP_SIZE=4
GLM5_DECODE_PIVOT_CANDIDATES=16384
GLM5_DECODE_PIVOT_LOCAL_WINDOW=4
GLM5_DECODE_PIVOT_MIN_KV_LEN=32768
```

`GLM5_DECODE_PIVOT_MODE=off` (the default) disables this path. To isolate its effect,
keep `GLM5_INDEXER_POOL_REUSE_STEPS=0` and `GLM5_PREFILL_PIVOT_MODE=off` in the comparison.
MTP model loading and engine configuration are still required separately.

The four rows must belong to the same physical page-table row. Their dequantized
queries and unscaled head gates are mean-pooled separately. A single paged FP8 MQA
query recalls historical candidates. The candidate budget includes a disjoint
local-window union: the final `LOCAL_WINDOW` positions visible to the first query
plus all three newly introduced positions. Each original query independently
rescores those candidates, masks tokens outside its own causal length, and selects
Top-2048 with the configured native decode TopK backend. Invalid positions remain
`-1`; output token indices remain request-local.

Short, padded, nonconsecutive, or out-of-capacity groups use native full-context
scoring. Ordinary single-token decode, MTP1, draft extension, and prefill do not
enter this path. There are no host reads of sequence lengths; the kernel path is
CUDA Graph compatible. Full serving-engine graph capture requires separate end-to-end
validation beyond the graph replay unit test.

For CUDA Graph serving, set the PIVOT configuration before starting the process:

```sh
ENABLE_CUDA_GRAPH=1
DECODE_CAPTURE_CONFIG=1,2,4,8
MAX_GENERATE_BATCH_SIZE=8
```

The capture sizes count requests, not query rows: an MTP3 batch of eight requests
has 32 target verification rows. The engine rounds a runtime batch up to the next
captured bucket (for example, three requests use the four-request graph). Keep
the maximum generation batch size within the largest capture bucket. Increase
both settings together when provisioning larger serving batches.

Each request has its own mean-pooled query, page table, candidates and causal
lengths. Eligibility and the full-context fallback are recomputed on the GPU on
every replay. Padded slots do not inherit candidates from previous requests.
No `GLM5_INDEXER_POOL_SLOTS` setting is needed for this stateless path.

PIVOT mode and candidate budget are fixed at graph capture time; changing a
Python attribute or environment variable afterwards does not change an already
captured graph. Restart and recapture when changing these settings.

The production-dispatch regression covers BS 1, 2, 3, 4, 5, 8, 16, 32, 64 and 128,
with updated Q/scales/page tables and long, short, padded and nonconsecutive
groups in the same captured graph. It also compares batched requests with
independent execution. Both native MTP layout with `dsv4_persistent` and flattened
MTP layout with `topk_v3` pass. This operator test does not by itself establish
full-model serving capacity at BS128.

For PD separation, enable `GLM5_DECODE_PIVOT_*` on the decode role only. The
prefill role keeps its existing CP and KV-transfer settings. Decode PIVOT uses
the receiving worker's physical page table and request-local token indices;
it neither transfers candidate pools nor changes the KV wire format. Existing
RPC/P2P handling restores the first target token and draft state before MTP
verification. A prefill CP shard must first be reconstructed by the existing
transfer path into the TP1 decode cache; PIVOT does not operate on an individual
prefill shard.

The accompanying graph routing fix checks the actual input's target-verify
flag. A model configured to capture verification graphs can still receive a
normal prompt prefill in a fused service; that prompt must execute eagerly.
The runner's capture role alone must not admit it to a four-query graph.
This fix also requires rebuilding `th_transformer`.

Local full-model testing uses one TP1/DP8/EP8 fused service on the available
eight-GPU machine. It does not validate a live PD network transfer. MTP draft
indexer sharing is separate from PIVOT; the current end-to-end configuration
uses `RTP_LLM_ENABLE_MTP_INDEXER_SHARE=0` and
`DISABLE_SP_PREFILL_CUDA_GRAPH=1`. The latter keeps draft extension eager while
allowing target verification and ordinary draft decode graphs.

This changes the target model's attention approximation. Mean pooling uses all
four queries to choose candidates; per-row causal masking does not make the
candidate choice independent of later draft queries. It therefore does not retain
the lossless-speculation guarantee of an unchanged target model. Candidate overlap
and end-to-end accuracy must both be measured.

Validation on L20D: eight new GPU/config/dispatch tests and 21 existing prefill
PIVOT tests passed. Early synthetic microbenchmarks show no speedup over native
four-query paged MQA, which already shares KV reads. No production performance or
accuracy claim is made from these tests.

The accompanying engine fix restores missing MTP GPU token mirrors from the
published CPU tokens. Previously the fallback filled them with token zero after
prefill dispatch, potentially corrupting the first target verification even with
PIVOT disabled. This C++ change requires rebuilding `th_transformer`; deploying
only the Python files does not apply it. Existing valid GPU mirrors remain
unchanged. The dedicated missing/partially missing mirror regression passes.

On 2026-11-02, a paired 100-request RULER CWE run on GLM-5.3 (about 48K input
tokens, greedy, no extra output cap) completed with the engine fix: native MTP3
answered all ten words correctly on 100/100 requests, versus 99/100 for g4p16
(99.9% answer-word recall). Both runs ended naturally without errors or detected
loops. The one miss was a spelling change; three isolated repeats per mode all
passed, but do not replace the original paired score. SWE has not been evaluated.

Mean output length was 253.34 tokens for native MTP3 and 267.86 for PIVOT (+5.73%).
Of the paired requests, 46 became longer, 48 shorter, and six were unchanged;
the median paired difference was zero. Inspected long responses enumerated word
positions, including long native responses. These results do not establish a
consistent tendency toward longer generation. Accuracy-run latency includes
diagnostic synchronization and must not be treated as a throughput benchmark.

Complete Indexer CUDA Graph microbenchmarks at BS1/128K took 29.0 microseconds
for native MTP3 versus 107.3 for PIVOT; at BS8/1M, 396.9 versus 432.1 microseconds.
This implementation remains experimental and disabled by default.


The 2026-11-02 graph follow-up completed 100/100 CWE requests correctly, with
100 natural stops and no errors or detected loops. Actual target graph replay
was observed at request buckets 1, 2, 4 and 8. The full-model BS4/8 smoke used
short fallback requests; the eligible long-request mixed-batch PIVOT path is
covered by the operator replay tests up to BS128. This does not establish
full-model long-context BS128 capacity or live PD transport correctness.

The performance revision retains group size four and the 16K candidate budget.
It fuses GPU eligibility planning, scale clamping and proxy construction;
uses a 128-column/four-warp candidate score tile with the same FP16 dot math;
and introduces exact, length-aware radix candidate selection for shapes where
it beats torch TopK. Large-batch matrices above 256K columns retain torch TopK.
The selection routing depends only on captured tensor shapes. Exact threshold
ties in the new selector use increasing token indices; torch's arbitrary tie
choice and candidate order are not promised to match. Per-query causal masks,
group-new tokens and native fallback are preserved. Prefill kernels are unchanged.

Both native-MTP/dsv4 and flattened-MTP/topk_v3 configurations pass 11 operator
regressions, including graph replay, changing physical pages, padded slots,
short rows and large tie sets. Eight synthetic complete-path cases preserve the
old final Top-2048 sets. These checks complement rather than replace model accuracy.
Optimized full-model graph accuracy also completed 100/100 CWE requests fully
correct, all naturally stopped, with no errors, loops or context truncation.
Average output was 244.23 tokens (max 1634), versus 223.70 before optimization.
Both graph runs use identical generation settings and prompt hashes. Prior eager
native MTP3 averaged 253.34 tokens; these variable-length accuracy runs are not
controlled service throughput comparisons. Optimized serving again exercised
actual request graph buckets 1, 2, 4 and 8.

Complete-path CUDA Graph timing on idle L20D improved from 202.8 to 93.4 us
at BS1 / actual KV48K / static capacity1M; 241.2 to 152.2 us at BS8 / KV128K;
and 430.8 to 401.5 us at BS8 / KV1M. Native MTP3 measured 88.1, 69.9 and
396.0 us respectively. The optimization reduces PIVOT overhead but does not
establish a gain over native or any full-model end-to-end speedup.


On 2026-11-03, a further kernel-only revision tuned radix histogram tiles and
replaced candidate rescoring for batches >=4 with packed scratch pages plus
native four-query paged FP8 MQA. It requires SM100+, 64-token pages, an aligned
candidate count, and native-MTP enabled; other shapes keep the Triton scorer.
Packing preserves candidate order and exact FP8 keys/scales. A subsequent mask
restores invalid/ineligible/future positions to -inf before the unchanged final
TopK. Scratch pages and metadata are independent per request and rebuilt on every
CUDA Graph replay. No persistent pool or PD transfer format is introduced.

Candidate-score arithmetic now follows the existing DeepGEMM reduction and can
round differently from the previous Triton FP16-dot path. Synthetic maximum
absolute score error was about 2e-6; the eight complete-path comparison cases
retained identical final Top2048 sets. Both native/dsv4 and flat/topk_v3 test
configurations passed 12 tests, including the new packed-score graph/causality
regression. These kernel-only checks do not replace a fresh model accuracy run;
the preceding 100/100 CWE results apply to the previous revision.

The previously measured prefill win used 2048 queries, group4, candidates4096,
and KV128K. Its MQA kernel decreased from approximately972 to237 us. Decode uses
four queries per request and candidates16384. On BS8/KV1M, the native four-query
MQA kernel took about189 us and the one-proxy MQA about179 us. Native decode
already amortizes KV reads across its four queries, so removing three query rows
does not remove three quarters of the kernel latency. The extra16K recall and
candidate scoring then consume much of the saved time. The prefill and decode
measurements also use different candidate budgets and are not like-for-like.


With the user's topk_v3 backend included in the same controlled comparison,
BS8/KV128K improved from142.8 to120.8 us, but native measured48.4 us;
BS8/KV1M improved from397.7 to377.6 us, but native measured274.2 us.
The small advantage over dsv4_persistent at BS8/KV1M is backend-specific and
does not establish a PIVOT advantage for topk_v3. Detailed per-kernel records
are in /Users/echo/glm53-decode-pivot-kernel-20261103/REPORT.md.
