# Suffix Reoptimization v2.1 相对 v2.0 的关键核心更新

## 资料口径

本文以当前仓库中的 canonical 设计、v2.1 sidecar、独立 config、主流程接入、输出层和单元测试为准。核对范围包括：

- `suffix_v2.1_方法设计.md`；
- `method_versions/suffix_reoptimization_v2_1.py`；
- `configs/suffix_reoptimization_v2_1.json` 与组合 selector config；
- `invert.py`、`experiment_outputs.py`；
- v2.1 sidecar、integration、输出与回归测试。

经代码确认后的修正是：canonical 设计文档中“v2.1 尚未实现”的状态已被本次实现推进所取代，但当前仍没有 v2.1 的真实模型、真实数据或正式实验 artifact。本文因此只报告代码事实与 static/toy/mock/CPU 验证，不声称 accuracy、速度或稳定性已经提升，也不把 v2.0 的已知问题写成 v2.0 已被修改。

## 1. 发现了什么问题

v2.0 的连续优化、离散诊断、classifier 接口和累计 replay 共同存在于一条较长路径中，正式方法状态与 Ground Truth diagnostics 的 failure domain 也没有完全隔离。与此同时，mixed-context 提交前缀、最终 token 与主返回 embedding 的一致性，以及每个位置 repair 次数上界，都需要更直接的执行合同。

这些问题会增加以下风险：

1. diagnostics 异常影响正式 accepted/rollback 状态；
2. 连续 embedding、已提交 token 前缀和最终返回 embedding 的语义不一致；
3. local repair、expanded rerank 与累计 replay 形成难以审计的循环；
4. classifier、prox/range、MAD/CUSUM 等机制增加主流程分支和版本耦合；
5. selector、resolved config 与 JSONL 若未同步接入，运行时实际方法可能与实验命名不一致。

上述内容是 v2.1 新版本隔离实现的动因，不构成对 v2.0 文件的修改；v2.0 仍保留为可显式选择的旧版本。

## 2. 做了哪些改动

### 2.1 方法结构

| 改动维度 | v2.0 | v2.1 | 当前实现合同 |
|---|---|---|---|
| 连续优化 | 两阶段优化 | 单一 persistent global Adam | 只优化有效位置；anchor refresh 不重建 optimizer |
| 连续目标 | 方向/幅度目标并含 prox、range | 多层方向/幅度联合误差 + legal-vocab soft-min | 仅使用 Qwen2/Qwen2.5 causal LM，`use_cache=False` |
| 上下文 | 连续状态与离散状态路径较复杂 | committed-prefix mixed context | 严格从左到右提交 token |
| 向量修复 | 与后续多级诊断共同工作 | 每位置最多一次 vector repair | local Adam 只存在于该次 trial 内 |
| 离散候选 | 多来源候选与累计 replay | normal embedding + PPL；必要时一次 expanded rerank | legal-only、去重、按 `(d, token_id)` 决胜 |
| 失败语义 | diagnostics 与正式域存在耦合风险 | formal result 先冻结，diagnostics 后运行 | 单候选非有限丢弃；全部无效 hard fail；diagnostics 不改正式结果 |
| 返回状态 | token 与 embedding 可能不一致 | token-consistent final embedding | fixed structural 与 padding 位置保持入口 embedding |
| 删除机制 | classifier、prox/range、MAD、CUSUM、replay 等 | 全部移除 | v2.1 config/interface/result 不再承载这些机制 |

### 2.2 冻结默认参数

| 参数组 | v2.1 默认值 |
|---|---|
| 多层目标 | offsets `[0, 1, 2]`，weights `[1.0, 0.5, 0.25]` |
| 联合误差 | `alpha_dir=0.5`，`alpha_mag=0.5` |
| 词表正则 | `lambda_v=0.005`，`tau_v=0.01`，`K_v=10`，refresh interval `10` |
| Global Adam | `1000` steps，LR `1e-3`，betas `(0.9, 0.999)`，epsilon `1e-8` |
| Local Adam | `50` steps，LR `1e-3`，同一组 Adam 数值参数 |
| Gate | `tau_J=0.15`，`delta_c_max=0.01`，`tau_r=0.05` |
| 候选池 | normal embedding `10`，expanded embedding `20`，PPL `10` |
| 数值与过滤 | epsilon `1e-8`，`filter_nonascii=true` |
| Offline diagnostics | `accuracy_diagnostics_enabled=false` |

独立 v2.1 config 保持 `suffix_reoptimization_v2_1=false`，用于版本隔离和显式选择检查；组合默认 config 覆盖为 `suffix_version=v2.1` 且 `suffix_reoptimization_v2_1=true`。CGMR 组合默认改为 `cgmr_version=none`，v1.0/v1.1/v1.2 三个 enabled flag 均为 `false`。显式选择 v2.1 但 disabled，或 v2.1 与任意 CGMR 同时选择，都会 fail-fast。

### 2.3 主流程与输出

- `invert.py` 增加 v2.1 import、alias、strict selector、config 构造、Qwen2 causal model gate、多层 target hidden 收集、committed-prefix helper、独立执行分支和通用 fatal exit code。
- formal result 以 `suffix_reoptimization_v2_1_result` 和 canonical `suffix_reoptimization_result` 两个 JSONL 字段落盘；v2.1 不输出 classifier 顶层字段。
- formal result 返回后，主流程只读 frozen final tokens 计算顶层实验评价，并在 `advanced_method` evaluation view 保存 pre/post accuracy，供 `stage_accuracy` 与固定日志摘要读取；accuracy 不写回 v2.1 formal/diagnostics result，测试锁定 diagnostics、accepted 和 rollback 不被改变。
- `experiment_outputs.py` 记录 selected-only v2.1 resolved config；`experiment.log` 继续使用既有固定样本与平均 accuracy 摘要，不写入 repair、candidate 或 diagnostics 明细。

## 3. 为什么要做这些改动

1. 单一 persistent Adam 和固定的 fresh-retrieval gate 缩短了连续优化到正式判定的状态链，有助于审计 optimizer 生命周期；实际速度仍需真实实验测量。
2. 严格 committed-prefix mixed context 让当前位置始终看到已经提交的左侧 token，避免后续位置基于过期的全连续前缀进行判断。
3. 每位置最多一次 vector repair 和一次 expanded rerank 给出明确循环上界，降低隐式 replay 与重复修复造成的状态复杂度。
4. legal-only chunked top-k、special/pad 排除和确定性 tie-break 使候选合同可复现，同时避免完整词表距离矩阵的显存峰值；chunk 大小和真实耗时仍需实测。
5. formal result 先冻结、offline diagnostics 后运行，使 Ground Truth 读取和 diagnostics exception 无法改变正式 token、embedding、accepted 或 rollback。
6. 独立 sidecar/config/selector 保留 v2.0 可回退路径；v2.1 与 CGMR fail-fast 避免将两个尚未定义组合语义的方法静默串联。

## 4. 改动后的实验效果

当前没有可核准的 v2.1 `resolved_config.json`、`experiment.log` 或 `reconstructions.jsonl` 正式实验目录，因此没有样本级 accuracy、平均 accuracy、运行时间、吞吐量或 repair gain 可以报告。

| 验证层级 | 当前结果 | 可以支持的结论 | 不能支持的结论 |
|---|---:|---|---|
| v2.1 sidecar toy/mock/CPU 单元测试 | 24 项通过 | sidecar 的冻结合同在 toy/mock 条件下成立，包括 global nonfinite rollback 与 expanded pool 边界 | 真实 Qwen accuracy、速度、显存 |
| v2.1 integration 单元测试 | 12 项通过 | selector、CGMR fail-fast、Qwen gate、resolved config、JSONL schema/wiring、exit code 与 v2.0 显式选择回归通过 | 正式数据集效果 |
| 固定日志摘要回归 | 8 项通过 | v2.1 没有向 `experiment.log` 增加方法明细；formal accuracy 为 `None` 时可从顶层 evaluation view 输出实验数值 | 正式日志 artifact 已生成 |
| output/layout 回归 | 17 项通过 | 方法目录布局、组合默认 config 与 package layout 合同通过 | 正式输出目录已生成 |
| 全量 unittest | 229 项通过 | 当前仓库单元测试在本次 CPU 环境中全绿 | 真实模型端到端可运行性 |
| 静态编译 | 相关 Python 文件通过 `py_compile` | 当前接入文件无 Python 语法错误 | 真实模型端到端可运行性 |

这些结果仅属于 static/toy/mock/CPU 验证。尚需真实 Qwen2/Qwen2.5 模型、真实数据和正式 timestamp-run artifact 才能评价方法效果。

## 5. 后续改进方向

1. 在独立环境中先做单样本真实 Qwen2/Qwen2.5 smoke run，核对 `use_cache=False`、显存和 JSONL schema；该步骤尚未执行。
2. 生成正式 timestamp run，并逐项核对 `resolved_config.json`、固定 `experiment.log` 和 `reconstructions.jsonl`。
3. 在相同模型、数据、seed、层与运行时条件下公平比较 v2.1、v2.0 和 frozen baseline；分别报告样本级与平均 accuracy。
4. 对 `lambda_v`、`tau_v`、`K_v`、refresh interval、三个 gate threshold 和候选池大小做消融与校准。
5. 记录 global/local optimization、legal-vocab retrieval 与 rerank 的分阶段耗时和峰值显存，验证 chunked retrieval 的实际收益。
6. 保持 v2.1 与 CGMR 默认互斥；只有在单独设计并冻结组合语义、输出合同和回归测试后，才考虑新增组合版本。
