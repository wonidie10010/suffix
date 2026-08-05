# suffix_reoptimization_v1.2.2 相对 v1.2.1 的关键核心更新

本文吸收用户提供的正式实施方案，并以当前代码
`suffix_optimization_methods/method_versions/suffix_reoptimization_v1_2_1.py`
为唯一直接基线核对实现。方法、配置与输出说明以 v1.2.2 sidecar、
独立配置、`invert.py` 和 `experiment_outputs.py` 的当前实现为准；尚未生成
正式实验 artifact，因此不把启发式阈值或预期收益写成实验事实。

## 1. 发现了什么问题

v1.2.1 在目标层 hidden-state 的异常检测、候选 forward 重排、状态汇总和
接受判断中使用 cosine similarity。cosine 更关注方向一致性，对 hidden
幅值误差不敏感；同一版本的 suffix 优化损失也只有 weighted cosine loss，
因此优化目标与后续离散决策都缺少相对能量误差视角。

另一方面，指标替换存在明确的兼容边界：输入 embedding-space cosine top-k
仍然是高效候选池检索手段，v1.2.1 的六个异常原因、threshold/adaptive 组合
规则、扫描顺序和回滚结构也不能随指标替换而改变。relative MSE 的方向与
旧 cosine 相反，如果复用 `hidden_mean`、`hidden_min` 或 `similarity` 字段，
会使跨版本结果读取产生歧义。

## 2. 做了哪些改动

v1.2.2 新建了独立 sidecar 和独立 config，未修改 v1.2.1，也未导入 v1.3/v1.4
机制。主要改动如下：

| 改动维度 | v1.2.1 | v1.2.2 |
|---|---|---|
| suffix hidden loss | weighted cosine loss | `0.1 × weighted cosine loss + 0.9 × weighted relative MSE loss` |
| 目标层异常指标 | 低 cosine / cosine drop | 高 relative MSE / relative MSE rise |
| candidate forward 重排 | cosine `argmax` | relative MSE `argmin` |
| 候选池检索 | embedding cosine top-k | 保持 embedding cosine top-k |
| suffix 接受指标 | 普通 suffix token-forward cosine mean | 普通 suffix token-forward relative MSE mean |
| 指标字段 | cosine 的 `hidden_*` / `similarity` | 明确的 `*_relative_mse_*` 字段 |

所有 relative MSE 路径共用唯一实现：两侧 hidden state 先转为 float32，再按
目标 hidden 能量归一化，稳定项固定为 `1e-8`。位置权重只用于原本加权的
suffix 优化损失，不用于异常、汇总、candidate 排序或接受。

六个异常信号按原顺序一一替换；threshold 模式仍需满足
`min_anomaly_reasons`，adaptive 模式仍是任一原因成立即触发，且没有新增
adaptive embedding rise。初始启发式阈值为：高 relative MSE `1.0`、
rise `0.30`、最小 suffix mean 改善 `0.01`、adaptive z-score `1.5`。
其中 rise `0.30` 是旧 cosine drop `0.15` 的严格近似方向换算起点，尚需验证。

此外，v1.2.2 可选择性地在 baseline `backward()` 后、`SGD.step()` 前只读
统计梯度趋势。EMA 固定为 `0.9`，局部位置通过 `eval_start_pos` 映射为完整
序列绝对位置；该统计不修改梯度，也不参与任何优化或决策。

## 3. 为什么要做这些改动

relative MSE 为目标层 hidden 恢复补充了幅值敏感的误差口径；将其作为联合
损失的主要项，有助于让优化、异常检测、forward 重排和接受判断使用一致的
数值定义。保留小权重 cosine 方向项，则避免完全放弃方向诊断。

保留 embedding cosine 候选池与 v1.2.1 的控制流，可以把实验变量限制在
目标层指标体系，便于后续归因。明确使用 lower-is-better 的 relative MSE
字段和 version-aware 读取逻辑，可避免把新旧版本的 mean/min/max 按错误方向
比较。梯度趋势仅作为只读 artifact，为后续判断难优化位置提供证据，同时
不改变本版本算法结果。

## 4. 改动后的实验效果

尚未执行正式模型实验，尚需完整实验验证。

已准备 airport 5 条与 medical 5 条的独立实验配置；正式运行时关闭 CGMR
和其他 suffix 版本。后续结果只以 timestamp run 中的
`resolved_config.json`、`experiment.log` 和 `reconstructions.jsonl` 为准，
不生成 Excel。当前没有可用于报告 accuracy、耗时或接受率的正式 artifact，
因此本节不提供推测数值。

## 5. 后续改进方向

1. 运行已准备的 10 条样本配置，首先核对 relative MSE 分布、六类触发原因、
   接受率与 accuracy。
2. 围绕 `1.0`、`0.30` 和 `0.01` 做消融，确认初始启发式阈值的稳定范围。
3. 在相同数据、种子、baseline 和关闭 CGMR 的条件下，与 v1.2.1 做公平对比。
4. 扩大样本量并增加随机种子，分别报告 oracle 与非 oracle 接受模式。
5. 只把梯度趋势用于离线分析；若未来考虑接入决策，应另建版本验证，不能
   改写 v1.2.2 的只读语义。
