# suffix v2.0 相对 v1.x 的方向—尺寸联合误差与在线修复更新

## 资料口径

本文按用户已经确定的 suffix v2.0 方法设计实施，并以当前代码、独立 JSON 配置、selector 接线、父/worker runner 和测试为准核对事实。本文没有引用正式实验 artifact：本次按要求没有加载真实模型、没有访问真实数据集，也没有执行正式实验，因此不报告或推测 accuracy。涉及实现效果的结论仅表示代码和合成测试已覆盖，不表示模型实验效果已经得到验证。

## 1. 发现了什么问题

1. 旧 suffix 版本的误差体系、候选重排、异常判断和接受策略不能直接代表已经确定的 v2.0 方法；若直接修改旧 sidecar，会破坏历史 selector、回退能力和实验可比性。
2. v2.0 要求把连续优化、连续诊断、当前位置候选评分和 repair 判断统一到 float32 的方向—尺寸联合误差，并且只使用目标层 L 及其后续 L+1、L+2。旧版本路径不能在不改变语义的情况下隐式复用。
3. Stage 4、局部 repair、累计区间 repair 的候选池和接受规则不同。Stage 4 将进入当前位置选择前的当前 token 作为普通候选，与 Embedding/PPL 候选统一评分；repair 只有在当前位置严格改善超过 `replace_epsilon=1e-8` 时才替换。
4. 全局联合误差和 Ground Truth accuracy 都只能作为离线诊断，不能用于 token 选择、repair、最终接受或回滚。最终接受必须只受硬失败控制。
5. 十样本正式实验需要四张 GPU 做样本级并行，但四个 worker 不能并发写同一 artifact，也不能产生四份正式 run；旧的一键串行汇总结构无法满足这一约束。

## 2. 做了哪些改动

| 改动维度 | 旧路径 | suffix v2.0 | 说明 |
|---|---|---|---|
| 版本隔离 | v1.x sidecar/config | 新增独立 v2.0 sidecar/config/selector | 旧文件、配置与 fallback 保留 |
| 误差 | 各旧版本既有指标 | float32 方向—尺寸联合误差 | Phase 1 为 0.9/0.1，Phase 2 为 0.1/0.9，评分为 0.5/0.5 |
| 多层范围 | 旧版本各自定义 | L、L+1、L+2，原始权重 1、0.5、0.25 | 越界层删除后重新归一化，不读取 L 之前的层 |
| 连续优化 | 旧版本各自流程 | Phase 1 每步重建 SGD；Phase 2 持久 Adam | 默认分别为 1000 步/0.01 和 50 步/0.001；冻结特殊前缀与 padding |
| Stage 3 | 无 v2.0 冻结诊断 | 一次性计算并冻结 `c_i` 与 `tau_c` | `tau_c = median(c) + 3 * max(MAD(c), 1e-8)` |
| Stage 4 候选 | 旧版本候选规则 | 正常 Embedding 10 + PPL 10 + 当前 token；扩展 Embedding 20 + PPL 10 + 当前 token | classifier 当前关闭；当前 token 无特殊优先级 |
| 当前位置评分 | 旧版本各自评分 | 只评分位置 i 的多层联合误差 | 不读取 i+1、i+2 或短窗口 |
| 在线异常 | 旧版本离线/其他门控 | 前缀 median/MAD 与 warmup `d_i > tau_c` | 当前点不进入自身阈值历史 |
| 累计触发 | 无此 v2.0 状态 | robust 单侧 CUSUM | 默认 `kappa=0.5`、阈值 5.0，属于尚需实验校准的初始启发式值 |
| repair | 旧版本各自接受规则 | 局部与累计区间均要求严格当前位置改善 | 累计区间按起点到触发点从左到右重算 PPL 和全部历史状态 |
| 最终接受 | 旧版本各自逻辑 | 无硬失败即接受 | 全局联合误差与 accuracy 不参与接受或回滚 |
| classifier | 无统一可注入接口 | 保留严格 provider 协议 | 当前固定关闭、零调用；未来开启时缺失或非法结果立即硬失败 |
| artifact | 单进程 timestamp run | 四 worker 合并成唯一正式父 run | 原子 JSONL、统一 experiment.log、唯一 resolved config 与 manifest |
| 多 GPU | 非正式四卡样本调度 | GPU 0–3 四进程样本并行 | 固定 shards 为 `[0,4,8]`、`[1,5,9]`、`[2,6]`、`[3,7]` |

硬失败会回滚 sidecar 入口 token/embedding 快照，并记录 `rollback_reason`、`failed_stage`、`failed_position` 和 `failed_segment`。`stage4_unrepaired_global_joint_error`、`final_repaired_global_joint_error` 与 `global_joint_error_delta` 仅写入重构记录供分析。

父 runner 在任何模型或数据访问前校验四张物理 GPU、正式配置、十样本映射、固定 shards 和唯一输出路径。worker 仅写独立 shard/status/stdout/stderr；四份状态、配置指纹和模型元数据一致后，父进程才排序并原子生成唯一 `reconstructions.jsonl`，再复用现有摘要 writer 重建固定格式 `experiment.log`。正式父 run 经核心 artifact SHA-256 校验后只复制一次到 `实验/结果/suffix_v2_0_<timestamp>/`。

## 3. 为什么要做这些改动

1. 独立 sidecar 与显式 selector 使 v2.0 可以完整实现既定设计，同时不改变 v1.x 的方法语义和回退路径。
2. 统一方向—尺寸联合误差能让连续优化、离散化与 repair 使用同一量纲明确的目标；分阶段权重保留了先方向、后尺寸的既定优化重点。
3. 只在当前位置评分并使用已提交左前缀生成 PPL 候选，保证 Stage 4 真正在线、从左到右，避免未来 token 或 Ground Truth 信息泄漏。
4. 严格 repair 改善条件保证每一次替换都由当前位置联合误差下降支持；取消全局误差门控避免已完成的局部严格改善因全序列诊断微小波动而整体回滚。
5. robust CUSUM 用于识别连续多个中等 gap；区间顺序 repair 会在较早 token 改变后重算后续 hidden、PPL 和历史状态，保持在线状态一致。
6. 四卡样本并行避免了输入 embedding 反向传播、多层 hook 与动态候选 forward 在同一样本跨卡切分时的未验证风险，同时提高十样本总体吞吐量，并保留单一正式实验口径。

## 4. 改动后的实验效果

尚需完整实验验证。

本次只完成单元测试、mock runner、dry-run、配置自检、静态检查和 Python 编译检查，没有生成正式 `results/invert_timestamp_runs/suffix_reoptimization_v2.0/<timestamp>/` artifact。因此目前没有可以从 `resolved_config.json`、`experiment.log` 和 `reconstructions.jsonl` 核对的真实样本 accuracy，也不应填写样本级结果表或平均提升。

已确认的是代码级行为：v2.0 方法测试覆盖联合误差、层权重、候选池、tie-break、PPL 左前缀、classifier 严格接口、repair 严格改善、CUSUM、硬失败回滚和诊断隔离；runner mock 测试覆盖四进程隔离、固定 shard、唯一合并、失败保留、单次复制与哈希一致性。以上不能替代正式模型实验。

## 5. 后续改进方向

1. 在满足四张可用 CUDA GPU 的正式环境运行唯一一键入口，随后从正式父 run 的三个核心实验 artifact 与 manifest 核对十条样本和运行时配置。
2. 对 `cumulative_kappa=0.5`、`cumulative_threshold=5.0` 及 MAD 倍数做固定数据、固定 seed 的消融；在实验完成前保持当前既定默认值不变。
3. classifier provider 实现后，先补 provider 数值/排序一致性和候选覆盖测试，再显式将 `classifier_enabled` 切换为 true；不得静默降级。
4. 对 Phase 1、Phase 2、Stage 4、局部 repair 和累计区间 repair 分别做耗时剖析，在不改变公式、候选池或在线顺序的前提下定位性能热点。
5. 扩大样本和 seed 后，继续以同一正式 artifact 口径报告效果，明确区分 suffix 内部 `pre_acc` 与最终 `post_acc`。

当前正式实现采用四卡样本并行以加速十样本总体实验，并合并为唯一正式结果。同一样本 Tensor Parallel 因尚未完成输入 embedding 反向传播、多层 hook、数值一致性和性能验证，本版本未启用。
