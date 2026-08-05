# suffix_reoptimization_v1.3.1 相对 v1.2.1 的关键核心更新

本文吸收用户提供的正式实施方案，并以当前代码
`suffix_optimization_methods/method_versions/suffix_reoptimization_v1_2_1.py`
为唯一直接基线核对实现。方法与配置说明以 v1.3.1 sidecar、独立配置、
`invert.py` 和 `experiment_outputs.py` 的当前实现为准；尚未生成正式实验
artifact，因此锚定机制的预期收益仍属于待验证判断。

## 1. 发现了什么问题

v1.2.1 在发现异常位置后直接以当前连续 embedding 的整段前缀作为固定部分，
只优化异常点开始的 suffix。经过 baseline 连续优化和多轮 suffix 更新后，
待重构区间 `[eval_start_pos:suffix_start)` 的连续 embedding 不一定严格等于
当前已接受 token 经模型 input embedding layer 得到的离散表示。这会使下一轮
suffix 优化建立在连续前缀与当前 token 重构不完全一致的上下文上。

修正范围必须严格受限：`[0:eval_start_pos)` 是 v1.2.1 已知固定输入前缀，
应继续保留 current embedding；锚定区间只能来自最新已接受重构 token，
严禁从 Ground Truth 回填。异常、候选池、cosine rerank、接受、拒绝扫描和
回滚都不应借此升级而改变。

## 2. 做了哪些改动

v1.3.1 新建了独立 sidecar 和独立 config，未修改 v1.2.1，也未导入既有
v1.3/v1.4 的 manifold、nearest-vocabulary 或其他锚定机制。每轮 suffix
优化前按以下三段构造输入：

| 区间 | v1.3.1 来源 | 梯度语义 |
|---|---|---|
| `[0:eval_start_pos)` | 当前 embedding 原样保留 | 固定 |
| `[eval_start_pos:suffix_start)` | 最新已接受 `current_tokens` 经模型 input embedding layer 重嵌入 | `no_grad` / detached |
| `[suffix_start:]` | 当前连续 suffix embedding | 唯一 Adam 参数 |

实现会检查 token 数量与 embedding 序列长度一致、anchor 长度等于
`suffix_start-eval_start_pos`，以及拼接后的总长度不变。接受候选时同步提交
candidate embedding 与 token；拒绝时不覆盖当前状态，并继续使用 v1.2.1 的
扫描位置更新方式。

event 中记录的 anchor 诊断只统计待重构锚定区间，包含 token 来源、是否实际
调用 input embedding layer、本轮接受状态以及锚定前后同口径的 weighted
cosine similarity/loss。锚定前值复用扫描结果，锚定后值复用优化的首次
forward，未增加诊断专用 forward。

## 3. 为什么要做这些改动

离散锚定使每轮 suffix 优化看到的待重构前缀与当前已接受 token 序列一致，
预期有助于减少连续 embedding 漂移造成的上下文错位。只重嵌入
`[eval_start_pos:suffix_start)`，同时保留已知固定输入前缀和当前连续 suffix，
可以避免 Ground Truth 泄漏，也不改变 v1.2.1 对固定输入前缀的处理。

把 Adam 参数严格限制在 suffix，能够保证 anchor 不被梯度更新。复用既有
forward 记录锚定诊断，则可以观察锚定本身对目标层 weighted cosine 的影响，
而不引入额外推理成本或改变随机/状态轨迹。

## 4. 改动后的实验效果

尚未执行正式模型实验，尚需完整实验验证。

已准备 airport 5 条与 medical 5 条的独立实验配置；正式运行时关闭 CGMR
和其他 suffix 版本。后续结果只以 timestamp run 中的
`resolved_config.json`、`experiment.log` 和 `reconstructions.jsonl` 为准，
不生成 Excel。当前没有正式 artifact，因此不报告未经验证的 accuracy、
anchor 收益或耗时变化。

## 5. 后续改进方向

1. 运行已准备的 10 条样本配置，逐轮核对 anchor 来源、数量、接受状态和
   weighted cosine 变化。
2. 在相同数据、种子和关闭 CGMR 的条件下，与 v1.2.1 做公平对比。
3. 专门检查第二轮使用最新已接受 token、拒绝后状态不变及固定前缀逐位一致性。
4. 扩大样本量并增加随机种子，报告触发率、接受率、accuracy 与运行耗时。
5. 如果后续需要其他 anchor 范围或边界策略，应建立新版本，不在 v1.3.1
   中混入 stable/full anchor 或 Ground Truth 回填。
