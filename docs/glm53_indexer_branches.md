# GLM-5.3 Indexer 分支记录

核对日期：2026-11-04。仓库：`Echo-2334/rtp-llm`。

| 功能 | 分支 | 对应提交 / 范围 |
| --- | --- | --- |
| 早期 decode indexer pool reuse | `feat/glm53-indexer-pool-reuse` | 分支 HEAD `f8680bd8167779575dcb93cec019586445dd6536`；初始实现 `db6bdb8f5`。这是历史版本，不包含后续请求边界修复。 |
| 修复版 decode indexer pool reuse | `fix/glm53-indexer-accuracy-20261027` | 修复提交 `2e8e120a6`，包含请求边界清理及 refresh 选择保留。 |
| Prefill PIVOT / indexer reuse | `fix/glm53-indexer-accuracy-20261027` | 初始融合实现 `270906429`；请求相对索引修复 `484d00b10c95f7f83a17a7ea2997f64dc35e077d`。 |
| MTP3 decode PIVOT | `fix/glm53-indexer-accuracy-20261027` | 随本记录提交；4-query mean pooling、候选精排、CUDA Graph、多 BS、MTP token mirror 和 graph routing 修复。 |

Prefill 和最新 decode 实现目前共用修复分支，没有另建独立的
prefill-only / decode-only 分支。需要固定版本时使用提交号，不要只依赖会移动的分支 HEAD。
`484d00b10` 是本次 MTP decode PIVOT 提交之前的版本，但已经包含修复版 decode pool reuse。

## 开关与功能区别

- 旧 decode pool reuse：`GLM5_INDEXER_POOL_REUSE_STEPS`，跨 decode 步复用候选池。
- Prefill PIVOT：`GLM5_PREFILL_PIVOT_MODE=refine`，配套 `GLM5_PREFILL_PIVOT_*` 参数。
- MTP3 decode PIVOT：`GLM5_DECODE_PIVOT_MODE=refine`，配套 `GLM5_DECODE_PIVOT_*` 参数；每轮 4 个验证 query 内重建候选，不跨轮复用。

这些是独立的 opt-in 路径。隔离测试 MTP decode PIVOT 时，保持
`GLM5_INDEXER_POOL_REUSE_STEPS=0`、`GLM5_PREFILL_PIVOT_MODE=off`。
PD 分离部署只在对应角色启用对应开关；具体配置、验证范围及需要重编 C++ 的说明见
[MTP decode PIVOT 文档](glm5_decode_pivot.md)。

## 本次交付验证范围

2026-11-04 从 L20D 测试机取回 2026-11-03 kernel overlay，三个实现文件与本地一致。
此前同一版 kernel 在 native-MTP/dsv4 与 flat-MTP/topk_v3 两种配置下各通过 12 个测试，
16 组完整 Indexer 对比保持最终 Top-2048 集合一致。MTP C++ 修复及已有 PD 状态回归
分别通过 2 个测试。本次提交前再次检查 Python 语法与 diff 格式。

最新 packed FP8 scorer 改变浮点归约方式，尚未重新运行全模型精度；此前 CWE 100/100
属于上一版。没有实测跨机器 PD 网络传输，也没有证明当前 topk_v3 配置的端到端性能收益。
