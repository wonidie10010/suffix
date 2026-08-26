# Suffix Reoptimization v2.1.1 相对 v2.1 的关键核心更新

## 资料口径

本文参考“2.1准确率下降原因”会话和“DEML实验审计计划”上下文提出的实施建议，同时以当前仓库代码、config、selector、测试和输出层为准。涉及正式实验效果的结论只以 `resolved_config.json`、`experiment.log` 和 `reconstructions.jsonl` 为准。

经代码确认后的关键修正是：原 v2.1 在 `invert.py` 的 `use_external_stage1` 集合中，因此 `part_epoch=0`，初始化后的 embedding 直接进入 v2.1 sidecar；本次没有覆盖旧 v2.1，而是新增 v2.1.1，使 legacy Stage-1 在进入 v2.1 全局/因果阶段前完整执行。当前没有为 v2.1.1 运行真实模型、真实数据或正式 timestamp run，因此本文不声称 accuracy 已经提升。

## 1. 发现了什么问题

原 v2.1 的主流程将该方法视为外部 Stage-1 已经完成的方法。代码通过 `use_external_stage1` 跳过初始化后的 legacy Stage-1，`part_epoch` 被置为 `0`。因此，v2.1 的入口不是“经过初始连续优化后的 embedding”，而是初始化 embedding 的直接快照。

这会使“初始优化阶段”和“v2.1 global/causal reoptimization 阶段”同时缺失前者的基线语义，导致准确率下降原因难以与后续候选、repair 或 commit 机制区分。该判断是代码路径层面的确认，不是尚未完成的正式实验因果结论。

## 2. 做了哪些改动

| 改动维度 | v2.1 | v2.1.1 | 说明 |
|---|---|---|---|
| 版本隔离 | 旧 sidecar/config | 新 sidecar/config | 保留 v2.1 作为可回退版本，不覆盖旧实现 |
| Stage-1 入口 | `use_external_stage1`，`part_epoch=0` | 不进入 `use_external_stage1`，`part_epoch=optimization.epoch` | 初始化后先执行 legacy Stage-1 |
| 入口 embedding | 初始化 embedding | Stage-1 完成后的 embedding | 以 Stage-1 输出作为 v2.1.1 entry embedding |
| token handoff | v2.1 原有入口快照 | Stage-1 输出后建立冻结 entry token snapshot | 不在两阶段之间插入额外 full hidden rerank |
| 后续方法 | v2.1 global Adam + causal candidate/repair/commit | 复制 v2.1 的同一 formal 逻辑 | candidate pool、top-k、gate、repair、acceptance 和 rollback 未另行改动 |
| 配置与 selector | `suffix_reoptimization_v2_1*` | 独立 `suffix_reoptimization_v2_1_1*` | resolved config 记录版本和入口 pipeline |
| 输出证据 | v2.1 result | 新增 v2.1.1 result 和 `legacy_stage1` 结果 | `experiment.log` 固定摘要格式不变，方法细节写入 JSONL |

新增和修改的主要文件包括：

- `suffix_optimization_methods/method_versions/suffix_reoptimization_v2_1_1.py`；
- `suffix_optimization_methods/configs/suffix_reoptimization_v2_1_1.json`；
- `suffix_optimization_methods/configs/advanced_methods.json`；
- `experiment_configs/l24_airport_medical_suffix_v2_1_1_no_cgmr.json`；
- `invert.py` 和 `experiment_outputs.py`；
- `test/test_suffix_reoptimization_v2_1_1.py` 与对应 integration test；
- 原文件名保留的 `实验/一键运行_suffix_v2_1.py`、
  `实验/环境和实验/内部文件/run_experiment_suffix_v2_1.sh`、
  `实验/环境和实验/内部文件/runner_suffix_v2_1.py` 及其测试和 README。

一键链路没有新增并行入口：原 `suffix_v2_1` 文件名仍作为用户入口，但其内部
配置、method 目录、smoke 配置、日志 provenance 和结果复制目标均已切换到
v2.1.1；共享 runtime、共享锁、单卡约束、模型准备和三类必需 artifact 校验保持不变。

组合默认 selector 已指向 v2.1.1；v2.1 仍可通过显式 `suffix_version=v2.1` 选择。v2.1.1 与 CGMR 继续保持互斥。

## 3. 为什么要做这些改动

1. 新建 v2.1.1 而不是覆盖 v2.1，符合 suffix 版本隔离和可回退要求，也让下降原因可以在相同 v2.1 formal 逻辑下单独比较 Stage-1 入口差异。
2. 把 legacy Stage-1 放回 v2.1.1 入口，恢复“初始化 → 初始连续优化 → v2.1 global/causal formal method”的预期流程。
3. 冻结 Stage-1 后的 embedding 和 token snapshot，再进入 sidecar，可以区分 Stage-1 输出与后续 token 提交；不额外加入 hidden rerank，避免同时改变第二个变量。
4. 将 Stage-1 的 `epoch/acc/tokens` 保存到 v2.1.1 JSONL 结果，避免后续 `pre_acc` 记录被 sidecar 入口指标覆盖，增强审计可追溯性。
5. 独立的 `suffix_v2_1_1_*` 配置前缀和 resolved entry pipeline 使实际运行版本、参数和 Stage-1 语义可从 artifact 复核。

## 4. 改动后的实验效果

本次没有运行正式实验，因此没有可核准的 v2.1.1 `resolved_config.json`、`experiment.log` 或 `reconstructions.jsonl`，不能报告样本级 accuracy、平均 accuracy、速度或显存收益。

| 验证层级 | 结果 | 可以支持的结论 |
|---|---:|---|
| v2.1/v2.1.1 sidecar 与 integration focused tests | 73 项通过 | 新 sidecar 的 toy/mock/CPU 合同、selector、配置、输出 wiring 和旧 v2.1 回归通过 |
| method package layout | 10 项通过 | 新版本配置合并与 package layout 通过 |
| experiment log summary | 8 项通过 | 没有向固定 `experiment.log` 摘要增加方法明细 |
| v2.1.1 一键入口与 runner 专项测试 | 17 项通过 | 原 v2.1 文件名链路可定位到 v2.1.1 配置、method 目录和 smoke 流程 |
| 全量 unittest | 266 项通过 | 当前仓库静态/toy/mock/CPU 单元测试全绿 |
| Python/JSON 静态检查 | 通过 | 相关源码可编译，三个新增/修改 JSON 可解析；直接 py_compile 的默认缓存目录受本机权限限制，已用不落盘 compile 校验 |

以上均不是正式模型实验结果，不能证明 v2.1.1 已经解决准确率下降；仍需真实 Qwen2/Qwen2.5 模型、相同数据和公平对照 run 验证。

## 5. 后续改进方向

1. 在明确授权后，使用相同模型、数据、seed、层数和运行时条件分别运行 v2.1、v2.1.1、v2.0 与 frozen baseline。
2. 每次正式 run 核对 `resolved_config.json` 中的 `entry_pipeline`、`legacy_stage1_enabled`、`optimization.epoch` 和实际 selector，并同步检查固定 `experiment.log` 与 `reconstructions.jsonl`。
3. 报告 Stage-1 `pre_acc`、v2.1.1 formal `post_acc` 和独立 baseline 最终 accuracy，避免混淆三种指标口径。
4. 对 Stage-1 epoch、global steps、candidate pool、repair gate 和 acceptance 条件做消融，判断准确率下降来自初始优化不足还是后续 formal 阶段。
5. 在正式结果稳定后，再补充多样本、多随机种子和分阶段耗时/显存分析；当前不把静态验证写成效果提升。
