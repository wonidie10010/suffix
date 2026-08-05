# Suffix Reoptimization v1.4.1：取消相对置信硬门控更新说明

> 更新状态：代码与单元测试已完成，尚未运行真实模型实验  
> 基线版本：Suffix Reoptimization v1.4  
> 新版本：Suffix Reoptimization v1.4.1  
> 设计来源：`outputs/suffix_reoptimization_v1_4_取消相对置信门控/Suffix_Reoptimization_v1.4_取消相对置信硬门控改动文档.md`

## 1. 资料与验证口径

本说明根据当前 v1.4/v1.4.1 sidecar、独立配置、`invert.py`、`experiment_outputs.py` 和测试代码核对。当前 v1.4 文件与配置继续保留，可由 selector 显式选择和回退。

本次只完成代码和自动化测试，没有运行真实模型或数据集实验。因此，下文区分“经代码确认的实现”和“尚需实验验证的效果”，不把设计预期写成 accuracy 提升事实。

## 2. 发现了什么问题

经代码确认，v1.4 的 `hybrid` 置信模式会对 continuous similarity、token-forward similarity、top-1/top-2 margin 和 discretization gap 分别进行样本内 percentile 排名，再取四项排名均值作为额外硬门槛。

相对排名只保留顺序，不保留绝对差距。当一组位置的分数都很高但彼此接近时，排名靠后的位置仍可能因 `percentile_confidence_below_min` 被划入低置信集合。此时位置自身的绝对信号没有变化，高低置信结论却可能因其他位置的分数、数量或排列发生变化。

这会使已经通过全部绝对条件的位置再次进入细化优化，也增加了 `confidence_percentile_min`、`confidence_min_points`、短序列 fallback 和 `adaptive_gate_applied` 等配置与诊断语义。

## 3. 做了哪些改动

### 3.1 新建 v1.4.1，保留 v1.4

- 新增独立 `suffix_reoptimization_v1_4_1.py` 和 `suffix_reoptimization_v1_4_1.json`；
- 新配置全部使用 `suffix_v1_4_1_*` 前缀；
- v1.4 sidecar 与配置未改写，原有 `hybrid`/`fixed` 行为仍可回退；
- selector 新增 `v1.4.1`、`1.4.1` 及两个标准方法名别名；
- 未显式指定版本时，fallback 优先级更新为 v1.4.1、v1.4、v1.3、v1.2.1 和旧版本；
- `advanced_methods.json` 的显式默认 selector 仍为 `v1.2`，本次没有切换默认方法。

### 3.2 改为逐位置绝对置信门控

经代码确认，v1.4.1 已删除 percentile rank、四项排名均值、样本数启用条件和短序列相对门控 fallback。每个有效位置只根据自身信号判断。

位置只有同时满足以下条件才属于高置信：

1. continuous forward similarity 达到固定下限；
2. token-forward similarity 达到固定下限；
3. top-1/top-2 margin 达到固定下限；
4. discretization gap 不超过固定上限；
5. 至少存在两个有效候选；
6. embedding top-1 与 hidden rerank top-1 一致；
7. 不存在 adaptive anomaly；
8. 所需相似度、margin、gap、候选和 token ID 完整有效。

任意条件失败都会进入低置信集合。单个位置的绝对输入不变时，其他位置的增删、排序或分数变化不再影响它的掩码结论。

### 3.3 配置和输出字段清理

- v1.4.1 标准配置不再包含 `confidence_percentile_min` 和 `confidence_min_points`；
- `confidence_mode` 标准值为 `absolute`，兼容输入 `hybrid` 和 `fixed` 会归一化为相同的绝对门控；
- v1.4.1 的 `confidence_mask.mode` 固定为 `absolute`；
- 新结果不再写入 `percentile_confidence`、顶层或逐位置 `adaptive_gate_applied`；
- 新结果不再写入 `thresholds.percentile_min`、`thresholds.min_points` 或 `percentile_confidence_below_min`；
- `resolved_config.json` 新增独立 v1.4.1 参数块，其中只记录实际生效的绝对门控参数；
- `reconstructions.jsonl` 增加 `suffix_reoptimization_v1_4_1_result`，同时继续写统一的 `suffix_reoptimization_result`；
- `experiment.log` 的固定样本和平均 accuracy 摘要格式未增加方法明细，也没有恢复 `.xlsx` 输出。

### 3.4 保持不变的流程

以下机制继续复用 v1.4 的已确认实现：

- 持久化 SGD 粗优化、余弦学习率和原地 clip；
- embedding 候选生成、可选 PPL 候选与 target hidden rerank；
- adaptive hidden/token/drop anomaly；
- 高置信 token 离散锚定和 embedding 冻结；
- 非连续低置信位置的独立 Parameter 与稀疏 Adam；
- masked-window hidden loss、proximal loss 和 range loss；
- 接受、拒绝和离散锚定基线回滚；
- 置信构建函数不接收 `total_input_ids`，oracle 只用于掩码完成后的评估与接受决策。

## 4. 为什么要做这些改动

绝对门槛保留了模型信号的实际数值意义。多个位置都明显超过门槛时，不再需要为了形成样本内排序而强行把其中一部分降为低置信。

逐位置独立判断也使失败原因更直接：低置信位置只能由绝对阈值、候选条件、adaptive anomaly 或无效输入解释，而不再依赖其他位置构成。这减少了配置分支，并避免 `adaptive_gate_applied` 与 adaptive anomaly 的命名混淆。

建立 v1.4.1 而不是覆盖 v1.4，则保证相对门控实现仍能用于回退和后续配对消融。

## 5. 改动后的实验效果

**尚需完整实验验证。**

本次没有运行真实模型或数据集实验，没有新的 timestamp run artifact，因此不能宣称：

- suffix `post_acc` 已提高；
- 高置信 token 精度或错误位置低置信召回率已提高；
- fine 阶段运行时间或 loss 波动已降低。

已经完成的是代码级验证：v1.4.1 定向测试共 15 项通过，完整既有测试共 86 项通过；相关 Python 文件通过 `py_compile`。测试覆盖高且接近的分数全部通过绝对门控、各类绝对失败条件、位置独立性、旧 percentile 属性不影响掩码、高置信 bitwise 冻结、非连续低置信更新、接受回滚、selector、resolved config、统一结果和固定日志格式。

这些测试证明实现符合设计约束，但不能替代真实模型 accuracy 实验。

## 6. 后续改进方向

下一步应在用户明确下达实验命令后，使用相同模型、seed、数据集、候选预算和 CGMR 设置，对 v1.4 与 v1.4.1 做配对实验。重点核对：

- suffix `pre_acc` / `post_acc`；
- 高置信 token 精度；
- 错误位置进入低置信集合的召回率；
- 高、低置信位置数量变化；
- 冻结位置破坏数；
- fine 阶段实际优化位置数、运行时间和 loss 尾段波动。

固定阈值现在承担全部门控责任。如果实验发现高置信误冻结上升，应优先做固定阈值敏感性分析，而不是直接恢复样本内相对排名硬门槛。
