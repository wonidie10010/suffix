# suffix_v1.2.3 相对 v1.2.2 的关键核心更新

本文吸收用户提供的实现要求，并以当前项目代码和配置为准进行核对。方法事实来自
`suffix_reoptimization_v1_2_2.py`、`suffix_v1_2_3.py`、
`invert.py`、`experiment_outputs.py` 和对应 JSON 配置；自动化验证事实来自本次新增的
mock/单元测试。按任务约束，本次没有加载真实模型、没有处理真实数据集样本、没有运行
正式实验，也没有生成新的 timestamp run，因此没有可供核对的正式实验 artifact。

原始 baseline 已独立冻结为 `frozen_original_baseline`，用于提供不受后续 suffix
演进影响的固定对照实现；`suffix_version=none` 唯一选择这条路径。

## 1. 发现了什么问题

经代码确认，原始 baseline 的连续 embedding 优化、词表检索和 token 重排分别使用
hidden cosine 目标、embedding cosine top-K 与 hidden cosine argmax。这条路径原先主要
以内联逻辑存在于 `invert.py`，后续若继续修改主流程或 suffix 第一阶段，固定对照的行为
边界不够独立。因此需要把原始行为复制到不依赖 suffix 版本的 sidecar，并移除冗余的
baseline selector。

v1.2.2 的第一阶段沿用原始 baseline；其第二阶段已经统一使用 relative MSE 参与
suffix 联合损失、候选重排、异常检测和接受判断。这样会形成第一阶段以 cosine 为主、
第二阶段以 relative MSE 为主的度量边界。v1.2.3 要验证的问题是：如果只把第一阶段的
连续优化、词表检索和候选重排依次换成 relative MSE、embedding MSE 和 hidden relative
MSE，同时完全保留 v1.2.2 第二阶段，能否得到更一致的第一阶段目标与离散映射。

上述动机属于方法设计和待验证假设，不代表 v1.2.3 已经优于 v1.2.2。

## 2. 做了哪些改动

| 改动维度 | 原始 baseline / v1.2.2 | 新实现 |
|---|---|---|
| 固定对照 | baseline 逻辑主要位于 `invert.py` | 新增独立 `frozen_original_baseline.py`；保留 cosine 目标、SGD、原学习率/epoch/初始化/clip/range、embedding cosine 检索、PPL 扩展和 hidden cosine argmax |
| v1.2.3 第一阶段连续目标 | hidden cosine loss + 原 range loss | 有效位置上的 float32 relative MSE 加权均值 + 原 range loss；epsilon 固定为 `1e-8`；SGD 每 epoch 重建 |
| v1.2.3 第一阶段词表检索 | embedding cosine top-K | 分块计算 raw float32 embedding MSE，按 argmin 语义排序；过滤整个候选池并尽可能补足 top-K |
| v1.2.3 第一阶段候选重排 | hidden cosine argmax | 完整离散序列 forward 后，以目标层 hidden relative MSE argmin 选择；保留 embedding/PPL/current-token 来源和从左到右顺序 |
| v1.2.3 第二阶段 | v1.2.2 第二阶段 | 原样保留：`0.1 × weighted cosine loss + 0.9 × weighted relative MSE loss`、prox、range、Adam、front-decay、异常检测、两轮扫描、接受/拒绝/回滚/重扫和 gradient trend 只读诊断 |
| 版本切换 | 旧 `suffix_reoptimization_version` selector | canonical selector 改为 `suffix_version`；继续接受旧 selector 和 v1.2.3 旧开关作为输入别名，冲突时直接报错 |
| 输出 | v1.2.2 结果字段 | 方法 ID 改为 `suffix_v1.2.3`；第一阶段与重优化详情分别写入 `stage1`、`reoptimization`，并保留通用 `pre_acc`、`post_acc` 和最终结果 |

第一阶段记录 relative MSE start/end/min、embedding-forward 与 token-forward relative
MSE、有效位置数、步数、停止原因、NaN 状态、优化器和学习率。候选诊断记录 embedding
MSE top-1、top-K token IDs、合法候选数、每个候选的 hidden relative MSE、best/second
best 与 margin。`oracle_*` 字段只在候选池与选择完成后计算，不参与优化、候选生成、
排序、异常检测或接受判断。

v1.2.3 的 canonical 开关为 `suffix_v1_2_3`。第二阶段参数使用 `suffix_v1_2_3_*`
前缀，并与 v1.2.2
默认值一致。`resolved_config.json` 构建逻辑记录第一阶段实际使用的全局优化参数、top-K、
PPL、过滤规则以及全部第二阶段参数。固定格式的 `experiment.log` 没有增加字段，未新增
或更新 Excel 输出。

## 3. 为什么要做这些改动

独立冻结 baseline 将固定对照从 suffix 版本演进中隔离出来，并删除重复的 legacy
baseline selector，便于以后确认方法收益来自 suffix 算法而不是 baseline 被同步改写。它仍然使用 cosine，
也不调用 v1.2.3 的任何 helper。

第一阶段的 relative MSE 使用目标 hidden 的平均平方能量归一化，预期可同时反映方向与
相对幅值误差；embedding MSE 和 hidden relative MSE 都使用 lower-is-better 的 argmin
语义，使第一阶段连续目标、词表映射和候选 forward 重排的度量方向一致。分块词表搜索
避免构造完整的 vocab × hidden 中间张量，完整候选过滤则避免只有首候选合法而其余
top-K 含非法 token。

保留 v1.2.2 第二阶段的全部算法和默认参数，是为了把本版本的实验变量限制在第一阶段。
这样后续正式实验可以更清楚地判断三处 MSE 改动的影响。代价是第一阶段搜索和候选
forward 仍可能增加计算量，实际耗时与准确率都尚需正式实验验证。

指标方向必须分开解释：embedding MSE、relative MSE、joint loss 与 total loss 都是越小
越好，accuracy 越大越好；不同定义或不同空间的 MSE 绝对值不能直接横向比较。

## 4. 改动后的实验效果

本次未运行正式模型实验，尚需完整实验验证。当前不能报告 accuracy、pre_acc、
suffix gain、接受率或耗时变化，也不能根据单元测试推断模型效果。

当前只能确认代码、config、selector、resolved config、reconstruction 结果字段以及
mock 行为满足设计约束：冻结 baseline 的连续更新、词表检索和 hidden 重排与旧逻辑在
相同模拟输入下等价；v1.2.3 的数值函数、候选排序、Ground Truth 隔离和 v1.2.2 第二
阶段等价性已由自动化测试覆盖。

已准备但未执行的正式配置为：

- `experiment_configs/l24_airport_medical_baseline_only.json`
- `experiment_configs/l24_airport_medical_suffix_v1_2_3_no_cgmr.json`

两份配置均使用 airport 5 条与 medical 5 条、相同模型/layer/seed/主优化参数/top-K/PPL
口径，并关闭 CGMR 和非目标 suffix。v1.2.3 是否优于 v1.2.2，仍需以后
在这两个数据集上运行正式实验并核对 timestamp run 中的 `resolved_config.json`、
`experiment.log` 与 `reconstructions.jsonl`。

## 5. 后续改进方向

1. 按已准备配置先完成 frozen baseline 与 v1.2.3 的 10 条样本同口径实验，再与现有
   v1.2.1、v1.2.2、v1.3.1 artifact 分开核对 baseline accuracy、suffix pre_acc 与
   final/post accuracy。
2. 使用 `oracle_stage1_embedding_candidate_hit`、
   `oracle_stage1_joint_candidate_hit` 和 ground-truth rank 做离线候选召回诊断，但
   不让这些字段进入候选选择或接受逻辑。
3. 对第一阶段的连续 relative MSE、embedding MSE 检索与 hidden relative MSE 重排做
   逐项消融，以判断收益或退化来自哪一处。
4. 在相同硬件与运行参数下比较 v1.2.2/v1.2.3 的词表搜索和 candidate forward 耗时，
   再决定是否需要不改变语义的批处理优化。
5. 扩大样本量和随机种子，并分别报告 `oracle_accuracy`、`hidden_anomaly` 与 `always`
   接受模式，避免把 oracle 接受结果外推为无 oracle 场景效果。
