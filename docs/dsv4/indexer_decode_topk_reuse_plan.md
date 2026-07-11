# Decode Indexer Top-K Reuse — Implementation Plan

GLM-5.1 / DeepSeek-V4 (DSA) decode 优化：对 indexer 的 top-k 计算做「16-step 分组复用」，把每步的**全上下文稠密打分**降为**8192 候选上的精算 + 每 16 步刷新一次粗池**。

目标模型：`glm_5`（`GlmMoeDsaForCausalLM`，FP8，78 层，`index_topk=2048`, `index_n_heads=32`, `index_head_dim=128`）。
部署形态：PD 分离，decode = DP/EP、`enable_cuda_graph=1`（本优化只动 decode 角色）。

---

## 1. 背景（现状代码）

decode indexer 单步单层路径：`rtp_llm/models_py/modules/dsv4/fp8/indexer.py:533` `IndexerFP8.forward_decode_vectorized`
1. `compressor.forward_decode_vectorized` 把新 token 压进 `INDEXER_KV` 池（每槽 132B），压缩比 `compress_ratio`。
2. 算 indexer query `q`（RoPE，`freqs_cis[start_pos]`），`weights = x @ weights_proj`。
3. `indexer_q_fp8_quant_fold(q, weights)` → `q_fp8, w_fold`。
4. **`fp8_paged_indexer_score(q_fp8, w_fold, pool_2d, block_table, ctx_lens, max_ctx_len=T_max)`** → `logits[B*q_len, T_max]`。这是 DeepGEMM FP8 kernel，对**全部压缩 key** 稠密打分。`T_max = ctx_len / compress_ratio`（`_kv_cache_t = max_seq_len // compress_ratio`）。**这是 decode indexer 的主要开销。**
5. Top-K：`rtp_llm_ops.dsv4_persistent_topk(score, lengths, out_topk, workspace, K=2048, T_max)`（radix-select，仅 K∈{512,1024,2048}）→ `out_topk_buffer`。
6. 返回 `out_topk_buffer`（2048 索引）→ 喂稀疏 MLA（`flash_mla_sparse_fwd` / TileLang `sparse_attn`）。

批量事实：decode 是多序列批量，`positions_d[M]` 逐 token 绝对位置（`fp8/indexer.py` prepare meta），每序列相位不同。
CUDA graph：decode 被捕获（`decode_capture_config 1,2,4,8`），主图内算子形状必须固定。

---

## 2. 算法规格

分组大小 `G=16`，粗池 `C=8192`，精 `K=2048`（可配）。对每一层、每一序列（row）维护：
- `coarse_idx[row, C]` int32：当前组复用的 8192 候选（压缩 key 索引）。
- `group_base[row]`：当前粗池服务窗口的起始压缩位置。
- 有效候选数 `coarse_len[row]`（上下文短于 8192 时 < C）。

**每步 fine（进主图，固定形状）**：
1. 用**该步精确 query**（内容 + 该步 RoPE `freqs_cis[pos]`）对 `coarse_idx[row]` 指向的 8192 个 key 做精算打分 → `logits_fine[B, C]`。
2. `dsv4_persistent_topk(logits_fine, coarse_len, out_topk, ws, K=2048, C)` → 最终 2048 索引（映射回全局压缩索引）。
3. 输出交给稀疏 MLA（不变）。

**每 16 步 coarse（图外，侧流，按行触发）**：
- 触发条件：`(pos - phase0(row)) % G == 0` 的行。
- 用一个代表性 query（见 §4 RoPE）对**全上下文**做 `fp8_paged_indexer_score`（现有 kernel）→ top-`C`（用 `dsv4_persistent_topk` K=C=8192，需支持 8192，见 §5）→ 写入下一组的 `coarse_idx[row]`。
- **预取**：不在组边界当步算，而在**上一组的第 12 步**用侧流提前算好，使每步都有就绪的 8192 池。
- group0 特例：step0 当步同步算一次（无上一组可预取）。

复用前提（正确性）：组内 16 步各自真实 top-2048 的并集 ⊆ 粗池 8192。靠 4× 余量 + coarse RoPE 用窗口中点保证。

---

## 3. 配置开关（`parallel`/新 group）

新增 server_args（`rtp_llm/server/server_args/` 里加一组或并入 hw_kernel/moe 组）：
- `--dsv4_indexer_reuse` / `DSV4_INDEXER_REUSE`（bool，默认 False）总开关。
- `--dsv4_indexer_reuse_group`（int，默认 16）分组大小 G。
- `--dsv4_indexer_reuse_coarse`（int，默认 8192）粗池 C。
- `--dsv4_indexer_reuse_prefetch_step`（int，默认 12）预取触发步。
- `--dsv4_indexer_reuse_rope_offset`（int，默认 G/2=8；-1=动态窗口中点）coarse RoPE 偏移。

只在 `role_type=DECODE` 生效；prefill 路径完全不变。默认关闭，A/B 可控。

---

## 4. RoPE 位置（修正后）

coarse 打分给 query 施加的 RoPE 与「何时计算」解耦（`freqs_cis` 查表可任取位置）。**统一用所服务 16 步窗口的中点**：
- 组服务窗口 `[P, P+15]` → coarse query RoPE 位置 = `P + rope_offset`（默认 8）。
- group0：`P=i`，位置 `i+8`。
- 后续组：`P` = 该组起始绝对压缩位置，位置 `P+8`（**不用计算时刻的 i+12**）。
- `rope_offset` 设为可调；用 NIAH 扫 {8, 12, dynamic} 定最优。

注意：coarse 只决定候选集合；fine 每步仍用**精确位置** RoPE，最终注意力位置无近似。

---

## 5. Kernel 工作

1. **新增 fine-score kernel**：对显式候选索引列表 `coarse_idx[B, C]` 做 gather + FP8 打分，输出 `logits_fine[B, C]`。
   - 输入：`q_fp8[B, n_heads, head_dim]`、`w_fold[B, n_heads]`、`INDEXER_KV` pool、`coarse_idx[B, C]`（全局压缩索引，-1 为 pad）、`coarse_len[B]`。
   - 实现选项 A：新 Triton kernel（gather 132B 槽 → dequant → q·k → 加权求和）。
   - 实现选项 B：给 `fp8_paged_indexer_score` 加一个「候选索引模式」入参（复用其 dequant/GEMM）。优先 B（少写一套）。
   - 形状固定（B×C），可进 cuda graph。
2. **`dsv4_persistent_topk` 支持 K=8192**：现仅 K∈{512,1024,2048}。coarse 需 top-8192。要么扩展该 op（radix-select 支持 8192），要么 coarse 用一次普通 `topk`（图外，允许）。粗刷新在图外 → 可直接用 torch `topk(8192)` 兜底，后续再优化成 kernel。
3. 索引映射：fine 的 top-2048 是「coarse_idx 内的局部下标」，需 gather 回全局压缩索引再交给稀疏 MLA。加一步 `global = coarse_idx[local]`。

---

## 6. 数据结构 / buffer

`AttentionFP8`（或 IndexerFP8）持有常驻 buffer（随 max_batch、层分配一次）：
- `coarse_idx[num_layers, max_bsz, C]` int32 ≈ 78×8×8192×4 ≈ **20 MB**。
- `coarse_len[num_layers, max_bsz]` int32。
- `group_phase0[max_bsz]` int32：每行组起始锚点（首个 decode step 的 pos 决定相位）。
- 侧流 `torch.cuda.Stream` + `torch.cuda.Event`（每层或全局共享）。
- top-k workspace 复用现有 `_get_topk_workspace`。

生命周期：请求进入 decode 时按其起始 pos 初始化 `group_phase0`；prefill→decode 切换时（首个 decode step）同步算一次 group0 粗池。

---

## 7. CUDA graph 集成

- **主图（每步 fine）**：`fine-score → persistent_topk(K=2048 over C) → 索引映射 → sparse MLA`。全部固定形状，读常驻 `coarse_idx` buffer（原地更新，语义等同 block_table，图安全）。
- **图外（coarse 刷新）**：host decode 循环每步检查哪些行到 `prefetch_step`；对这些行在**侧流**上 launch coarse（全量 score → top8192 → 写 `coarse_idx` 的「下一组」区）。用 `event.record(side)` / `main_stream.wait_event` 保证下一组边界步的 fine 读到就绪数据。
- 因相位逐行错开，任一步约 1/16 行触发；coarse kernel 带 **row-mask**（只算被触发行）。
- 与现有 capture/replay 的衔接点：`rtp_llm/models_py/.../decode/forward.py:forward_decode` 与 cuda_graph runner；粗刷新调度放在图 replay 之外的 host 步循环里。

---

## 8. 分阶段实施

### Phase 0 — Profiling（先做，定收益上限）
- 用 nsys/自埋点测 128k decode 下 `fp8_paged_indexer_score` 单层耗时、indexer 在 decode 单步的占比、MLA/MoE 占比。
- 产出：indexer 占比 X% → 整体 decode 理论加速 = 1/(1-X+X/speedup_indexer)。若 X 低则重估是否值得。

### Phase 1 — 算法正确性 + 上限（`enable_cuda_graph=0`，单流，同步）
- 在 `IndexerFP8` 加 `forward_decode_reuse`：拆 coarse（全量→top8192，torch topk 兜底）/ fine（候选打分→top2048）。
- 组相位、coarse RoPE（窗口中点）、每 16 步同步刷新（先不预取、不侧流）。
- 开关 `DSV4_INDEXER_REUSE=1` 走新路径，默认走旧路径。
- **验证**：128k 下跑 NIAH（single_1/2/3、multikey）+ ruler_cwe，对比 baseline 分数（要求基本无损）；测单流 decode tok/s（看上限）。

### Phase 2 — 预取重叠（性能）
- 引入侧流 + event，改「同步每 16 步」为「上一组 step12 预取」，row-mask 只算触发行。
- 仍 `enable_cuda_graph=0` 验证正确性与吞吐提升。

### Phase 3 — CUDA graph 集成（生产）
- fine 路径入主图；coarse 刷新图外侧流 + event；常驻 buffer。
- 回归：`enable_cuda_graph=1` 下 NIAH/cwe 分数 + 端到端 decode tok/s；确认无 capture 失败、无 replay 读到未就绪 coarse。

---

## 9. 验证方案

- **质量**（必须，防 recall 下降）：128k `niah_single_1/2/3`、`niah_multikey_1/2`、`ruler_cwe`，新旧对比。NIAH 对候选缺失敏感，是主判据；cwe 辅助。
- **性能**：单请求 `first_token`（应不变）与 `decode tok/s`（应显著上升）；`aux_info.cost_time - first_token_cost_time` / output_len。
- **消融**：`rope_offset ∈ {8,12,dynamic}`、`G ∈ {8,16,32}`、`C ∈ {4096,8192}` 对 质量/速度 的权衡。
- 工具链：lm_eval（`local-chat-completions`，`chat_template_kwargs.enable_thinking=false`，128k，多并发）已就绪。

## 10. 风险与回退

- recall 下降（突发关注点）：增大 C 或减小 G；`rope_offset` 调优；总开关一键回退旧路径。
- cuda graph 集成复杂：Phase 1/2 已能出「关图」下的收益与正确性；图集成失败可先以关图形态交付（decode 关图会损失一部分 launch 开销，但 indexer 复用收益可能覆盖）。
- `persistent_topk` 不支持 8192：coarse 图外用 torch topk 兜底，不阻塞。
- 批量逐行相位实现繁琐：Phase 1 先按「同步、全行同刷」简化（bsz=1 或统一相位）验证算法，再上 row-mask。

## 11. 改动文件清单

- `rtp_llm/models_py/modules/dsv4/fp8/indexer.py` — 新增 `forward_decode_reuse` + coarse/fine 拆分 + 组相位状态。
- `rtp_llm/models_py/modules/dsv4/fp8/attention.py` — 常驻 buffer、侧流/event、decode 分发到 reuse 路径。
- `rtp_llm/models_py/modules/dsv4/decode/forward.py` — host 步循环里的 coarse 预取调度。
- fine-score kernel：优先扩展 `fp8_paged_indexer_score`（候选索引模式）；否则新增 Triton kernel。
- `rtp_llm/server/server_args/*` — 新增 `DSV4_INDEXER_REUSE*` 开关。
- 测试：`rtp_llm/models_py/modules/dsv4/test/` 加 reuse 正确性单测（对拍 baseline top-2048 的召回率）。

---

## 12. CUDA graph 兼容设计（Phase 3 详解）

### 12.1 Phase 1 里为什么不兼容
decode 图捕获要求：捕获区内**算子序列固定、无 host 同步、无动态 shape**；replay 时只有**常驻 buffer 内容**变化。Phase 1 违反处：
- `bool(refresh_mask.any())` → D2H 同步 + 数据相关分支；
- `refresh_mask.nonzero()` / 动态子批 → 动态 shape；
- 「每 G 步刷新」本身是数据相关控制流；
- coarse 的 `topk(8192)`（torch）在图内也不合适。

### 12.2 核心原则：热路径进图，刷新出图，靠常驻 buffer 通信
- **fine 每步路径固定 shape → 进主图**：candidate-score(C) → `dsv4_persistent_topk`(K over C) → 索引映射 → 稀疏 MLA。读常驻 `coarse_idx` buffer。
- **coarse 刷新是动态/条件 → 出主图**，由 host decode 循环在两次 `graph.replay()` 之间编排，原地写 `coarse_idx`。
- `coarse_idx` 地址固定、被主图捕获，replay 前 host 更新其内容——与现有 block_table/lengths 原地更新同构，图安全。

### 12.3 组件
1. **常驻 buffer（固定地址，一次分配）**：`coarse_idx[L, max_bsz, C] int32`、`coarse_len[L, max_bsz]`、`q_persist[L, max_bsz, H, D]`（fine 路径每步把本步精确 q 写入，供 coarse 复用）。
2. **in-graph fine（新 kernel，Phase 1b 前置）**：`fp8_indexer_score_gather(q_fp8, w_fold, pool, coarse_idx, coarse_len) -> logits[bsz, C]`（按候选全局索引 gather 132B 槽 → dequant → q·k 加权 → 固定 [bsz,C]），随后 `persistent_topk(K over C)`。全固定 shape，可捕获；同时**这才真正省算力**。
3. **out-of-graph coarse 刷新（host 编排）**：
   - host 侧已知每序列 CPU 位置（scheduler 维护 seqlen），**无需 GPU 同步**即可算「哪些行到触发点」。
   - 触发：行 `pos ≡ (G-4) (mod G)` 时**预取下一组**的粗池（留 4 步 slack）；对触发行组成**紧凑子批**（动态，图外 OK），跑全量 `fp8_paged_indexer_score` → top-C → 写这些行的 `coarse_idx`「下一组」槽。
   - group0 冷启动：首步同步算一次。
4. **RoPE 与 q 复用**：coarse 直接**复用预取步已算好的精确 q**（`q_persist`）——预取发生在上一组第 12 步，其位置天然是 `i+12`（与最初设想一致，无需再 RoPE）；group0 用首步 q（或 re-RoPE 到 i+8）。这样避免跨图重算 q。

### 12.4 排序 / 重叠（两版）
- **v1（Phase 3a，先做，必正确）**：coarse 在**主流上、两次 replay 之间**由 host 直接 launch（不进图）。同流顺序天然保证「写 coarse_idx 在 fine 读之前」。无跨流 event，图安全。coarse 成本 ≈ `(触发行数)×O(T_max)` 摊到每步 ≈ `(B/G)×T_max`，仍远小于 baseline 的 `B×T_max`。
- **v2（Phase 3b，重叠提速）**：coarse 捕成**独立子图**在**侧流** replay，用 event 与主图同步；主图起点 `stream.wait_event` 等待「外部每轮重录的 event」（graph + external event 标准模式）。4 步 slack 掩盖 coarse 延迟。复杂度高，作为增量。

### 12.5 图安全性论证
捕获区（fine）内：无 `.item()/.any()/.nonzero()`、无动态 shape、只读写固定地址 buffer；coarse 全在捕获区外由 host 编排 → 满足 cuda graph 约束。`coarse_idx` 原地更新的可见性由**同流顺序（v1）**或 **event（v2）**保证。

### 12.6 落地顺序
Phase 1b（写 `fp8_indexer_score_gather` 候选 kernel，先在关图下验证省算力与数值一致）→ Phase 3a（fine 入主图 + coarse 图外同流，端到端图开验证精度/吞吐）→ Phase 3b（侧流 + event 重叠）。

### 12.8 关键发现（已核实代码）
- **图外钩子已存在，无需改 C++**：`CudaGraphRunner`（`rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc:624`）在每次 replay 前调用 Python `attn.prepare_cuda_graph(...)`，**在捕获区之外**。coarse 刷新放这里即可（条件/动态在图外合法），写常驻 `coarse_idx`；同流顺序保证主图 fine 读到就绪数据。q 通过 in-graph 每步写 `q_persist` 供 coarse 复用（预取步位置 = i+12）。
- **候选打分免重写 FP8**：`_reuse_candidate_logits` 按 `coarse_idx` **gather 原始 132B 槽**到紧凑 paged pool（`_reuse_compact_slot_index`: `abs=bt[b,t//eb]*eb+t%eb`），再复用 `fp8_paged_indexer_score(max_ctx_len=C)` —— 与全量 kernel **字节级一致**，只多一次 gather，固定 shape 可入图。

### 12.9 已实现（本次）
- Phase 1（默认 `DSV4_INDEXER_REUSE`）：full-score 复用，**精度验证通过**（ruler_cwe 128k = 0.964 vs baseline 0.962/0.966）。
- Phase 1b（`DSV4_INDEXER_REUSE_FINE_KERNEL`，默认关）：候选专用打分（省算力 + 图形状），`_forward_decode_reuse_kernel` 跳过非刷新步的全量打分。
- 单测 `test/test_indexer_decode_reuse.py`（7 项，纯 CPU）：env 解析、C≥T 逐位等价、短上下文、漂移召回、紧凑 gather 复现分页寻址。
- **待办（需上模型迭代）**：① Phase 1b 候选路径的 FP8 数值/紧凑池 scale 处理 on-model 校验（对拍 baseline top-2048）+ 测速；② Phase 3a 把 coarse 迁到 `prepare_cuda_graph`、fine 入图、`coarse_idx/q_persist` 常驻 buffer，图开端到端。

### 12.7 遗留风险
- v2 的「主图等待外部 event」需在捕获时用 `cudaStreamWaitEvent` 记录并每轮重录 event，实现需谨慎（否则 replay 读到未就绪 coarse）。
- 逐行触发的紧凑子批 gather/scatter 的 host 开销（每步少量）；
- 冷启动首组与 batch slot 复用：用 `coarse_group[row]` 版本号判定刷新（Phase 1 已如此），图外逻辑照搬。
