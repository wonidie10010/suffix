# suffix v2.1 相对 v2.0 的关键核心更新与 canonical 方法设计

## 0. 资料口径与状态边界

本文吸收了用户提出的 v2.1 idea、`C:\Users\wonidie\Desktop\v2.0.drawio` 的流程结构，以及 ChatGPT `suffix` 项目中的三轮设计讨论；最终事实口径以当前项目代码、config 和项目级协作规则为准。

- 经代码确认，v2.0 当前仍采用 Phase 1/Phase 2 双阶段连续优化，并保留 prox、range、classifier 接口和多套离散/累计诊断状态。
- 经代码确认，v2.0 当前存在 diagnostics failure domain、`cumulative_max_repairs_per_trigger` 未实际约束循环、单卡 runner 与历史四卡报告不一致等实现问题。它们不是 v2.1 的设计依据，也不会在本文中写成“已修复”。
- 本文仅冻结 suffix v2.1 的方法设计，不代表 v2.1 已实现。
- 当前没有 v2.1 正式实验 artifact；本文不声称 accuracy、速度或稳定性已有提升。
- 本文不修改 v2.0，不删除任何旧版本，也不处理 runner/GPU 拓扑问题。

设计状态：**canonical spec 已冻结；代码、config、测试和正式实验均尚未执行。**

## 1. 发现了什么问题

### 1.1 两阶段连续优化将一个目标拆成多套权重与优化语义

v2.0 先用方向主导的 SGD，再用尺寸主导的 Adam，并在后续离散评分中使用第三套方向/尺寸权重。这使连续优化、连续诊断和离散评分的口径不完全统一，`d_i-c_i` 也可能同时混入“状态变化”和“度量定义变化”。

### 1.2 v2.0 的连续诊断没有使用真实已提交离散前缀

v2.0 先完成全局连续优化与全位置连续诊断，再进入左到右离散化。这样，位置 `i` 的连续 embedding 质量没有在 `0...i-1` 已经离散并提交的 token embedding 上重新检查，不能直接回答“前缀离散化后，当前位置连续状态是否仍合格”。

### 1.3 离散误差与累计误差控制链过长

v2.0 同时维护 `c`、`d`、`g`、`delta_g`、`G`、`z`、`S`、局部阈值、CUSUM 段起点和 replay 历史。累计 replay 一旦修改前缀，又必须重建后续连续基线与历史状态，容易形成复杂循环和实现偏差。

### 1.4 连续目标中的“靠近词表”语义不够直接

v2.0 的 prox 约束强调靠近 Phase 1 输出，range 约束强调坐标范围；它们都不等价于“靠近某个真实合法词表 embedding”。v2.1 需要把词表邻近性直接写入优化目标，同时避免不可微 argmin 和每步全词表反向传播。

### 1.5 正式返回 embedding 与最终 token 可能语义不一致

当前 v2.0 成功路径返回连续的 Phase 2 embedding，同时在结果对象中另存最终离散 token。v2.1 的 mixed-context 循环会逐位置改用已提交 token embedding；若仍返回旧连续张量，主返回值与正式 token 状态不一致。

## 2. 做了哪些改动

| 改动维度 | v2.0 | v2.1 canonical | 状态 |
|---|---|---|---|
| 连续优化 | Phase 1 SGD + Phase 2 Adam | 单阶段 persistent Adam | 结构冻结；步数/LR 待校准 |
| 连续目标 | 不同阶段使用不同方向/尺寸权重，并含 prox/range | 同一方向/尺寸联合误差 + legal-vocab local soft-min | 结构冻结；权重待校准 |
| 词表正则 | prox/range 间接约束 | 对合法词表 top-k anchor 的 normalized soft-min | 结构冻结 |
| 连续诊断上下文 | 全连续序列的一次性全位置诊断 | 已提交 token 前缀 + 当前/未来连续 embedding 的 mixed context | 结构冻结 |
| 不合格连续 embedding | 主要扩大离散候选池 | 先做一次 `e_i`-only 向量修复；仍不合格才扩大候选池 | 结构冻结 |
| 离散诊断 | 多套局部量与阈值 | 单一局部退化 `r_i=max(0,d_i-c_i)` | 结构冻结 |
| 累计诊断 | CUSUM、区间起点、replay、历史重建 | 删除 | 结构冻结 |
| classifier | 接口保留、默认关闭 | 从 v2.1 config/interface/candidate branch 中移除 | 结构冻结 |
| 离散 repair | local + cumulative replay | 最多一次 expanded rerank | 结构冻结 |
| 返回 embedding | 可与 `final_tokens` 不一致 | 主返回 embedding 与 `final_tokens` 一致 | 结构冻结 |
| diagnostics | v2.0 当前仍与正式 rollback 域耦合 | 正式结果冻结后再进入独立 diagnostics domain | 结构冻结 |

## 3. 为什么要做这些改动

1. **统一度量口径。** Global optimization、online continuous error `c_i` 和 candidate error `d_i` 使用相同的层、层权重、方向/尺寸定义及权重，`d_i-c_i` 才能解释为当前位置从连续 embedding 到 token embedding 的新增退化。
2. **让前缀离散化真正进入后续连续状态。** 位置 `i` 的 mixed context 直接使用已经提交的 token embedding，因此历史 token 对当前位置 hidden state 的所有可观察影响都会进入 `c_i`。
3. **用因果向量修复代替累计 replay。** 如果前缀影响能由当前位置 `e_i` 补偿，向量修复会完成补偿；如果无法补偿，`J_i` 会保持偏高并转入 expanded pool。基于 residual 的 CUSUM 无法定位旧 token 根因，因此没有充分理由保留复杂 replay。
4. **直接约束词表邻近性。** Local top-k soft-min 把 embedding 拉向真实合法词表 anchor，同时把离散检索与反向传播分开：anchor index 和 anchor embedding 均 stop-gradient。
5. **限制循环。** 每个位置最多一次 vector repair、最多一次 expanded rerank；candidate 阶段开始后连续 baseline 冻结，禁止 `d_i -> e_i -> d_i` 循环。
6. **明确 failure domain。** 可恢复的局部 trial 失败只回退局部候选；只有正式状态无法安全继续才整体 rollback；Ground Truth diagnostics 永远不能改变正式结果。

以上收益均为结构与可审计性层面的设计理由。对 accuracy、耗时和稳定性的影响仍是实验假设。

## 4. 改动后的实验效果

**尚需完整实验验证。**

当前没有 suffix v2.1 代码、resolved config、`experiment.log` 或 `reconstructions.jsonl`，因此不能提供样本级 accuracy、平均 accuracy、运行时间、repair gain 或吞吐量结论。

正式实验前必须先完成：

1. 独立 v2.1 sidecar/config/selector；
2. mixed `inputs_embeds` candidate scorer；
3. 正式方法与 offline diagnostics 的 failure-domain 隔离；
4. 最小针对性单元测试与 mock/dry-run；
5. 明确并记录全部待校准参数的初始值。

## 5. 后续改进与验证方向

1. 对 `lambda_v`、soft-min temperature、anchor top-k/refresh interval 做消融，检查词表邻近约束是否压过 hidden matching。
2. 对 `tau_J`、`tau_r`、global/local steps 与 LR 做校准；`tau_J` 必须绑定模型和 embedding 尺度，不能作为跨模型常数。
3. 对比 hard nearest anchor 与 local soft-min；hard nearest 只作为消融，不进入 canonical 主线。
4. 统计 `unresolved_continuous_quality` 与 `unresolved_local_degradation` 的分布，判断删除累计 replay 后是否出现稳定的未解决错误模式。
5. 检查逐位置向量补偿是否导致后部 embedding 距离词表越来越远。
6. 在相同模型、数据、随机种子、候选预算和运行时口径下对比 v2.0/v2.1。
7. runner/GPU 拓扑作为独立实现议题处理；不得用方法实验替代 runner 契约修正。

## 6. v2.1 canonical 方法设计

### 6.1 版本隔离

- 新建独立 sidecar：建议命名 `suffix_optimization_methods/method_versions/suffix_reoptimization_v2_1.py`。
- 新建独立 config：建议命名 `suffix_optimization_methods/configs/suffix_reoptimization_v2_1.json`。
- 参数统一使用 `suffix_v2_1_*` 前缀。
- 版本选择仍由 `suffix_optimization_methods/configs/advanced_methods.json` 的 `suffix_version` 显式控制。
- `suffix_version="v2.1"` 但 `suffix_reoptimization_v2_1=false` 时 fail-fast。
- v2.0 与全部 v1.x 文件、config 和 selector 路径保留。

### 6.2 输入、快照与 preflight

方法入口冻结：

- `entry_embedding_snapshot`；
- `entry_token_snapshot`；
- special/fixed prefix；
- padding 与 attention/eval mask；
- 有效位置集合 `I`；
- external hidden collection index `L`。

有效层：

\[
\mathcal K=\{L,L+1,L+2\}\cap[0,N_{hidden})
\]

越界层删除后，对剩余非负层权重重新归一化，使 `sum_k w_k=1`。

合法词表 `V_legal` 使用与离散候选相同的 legality policy，并显式排除：

- tokenizer special token IDs；
- `pad_token_id`；
- 配置要求排除的非法或非 ASCII token。

preflight 至少要求：

- `I`、`K` 非空；
- `|V_legal| >= max(K_v,K_e_expanded)`；
- 有效位置的 pre-selection working token 合法；
- target hidden keys、shape、mask、长度一致；
- causal attention 与 mixed `inputs_embeds` forward 可用；
- 必需 helper 可用。

### 6.3 统一的多层方向—尺寸误差

稳定余弦：

\[
\cos_\epsilon(h,h^*)=
\frac{h^\top h^*}{\max(\lVert h\rVert_2\lVert h^*\rVert_2,\epsilon)}
\]

\[
D_{dir}(h,h^*)=
\frac{1-\operatorname{clamp}(\cos_\epsilon(h,h^*),-1,1)}{2}
\]

\[
D_{mag}(h,h^*)=
\left(
\frac{\lVert h\rVert_2-\lVert h^*\rVert_2}
{\lVert h\rVert_2+\lVert h^*\rVert_2+\epsilon}
\right)^2
\]

方向与尺寸权重非负并归一化：

\[
\bar\alpha_{dir}+\bar\alpha_{mag}=1
\]

\[
D_{joint}=\bar\alpha_{dir}D_{dir}+\bar\alpha_{mag}D_{mag}
\]

同一 `D_joint`、有效层与层权重必须同时用于：

- global hidden loss；
- local vector repair hidden loss；
- online continuous hidden error `c_i`；
- discrete candidate hidden error `d_i(q)`。

在上述归一化条件下：

\[
0\le c_i\le1,\quad 0\le d_i\le1
\]

### 6.4 Legal-vocab local soft-min

embedding 距离：

\[
q(e,v)=\frac{\lVert e-v\rVert_2^2}{d_e}
\]

对当前 embedding 检索 `K_v` 个合法 anchor，anchor index 和 anchor embedding 均 stop-gradient。正式评价量使用 fresh retrieval：

\[
R_i(e)=-\tau_v\left[
\operatorname{logsumexp}_{k}\left(-\frac{q(e,v_{i,k})}{\tau_v}\right)
-\log K_v
\right]
\]

`-log K_v` 等价于 `1/K_v` normalization，保证 `R_i>=0`。

优化内部允许使用最近一次 refresh 得到的 detached anchor cache，形成区间固定的 local surrogate；正式 gate 与 repair acceptance 必须对 old/new embedding 分别 fresh retrieve 后重算，不能直接复用 stale cache 指标。

### 6.5 单阶段全局连续优化

\[
\mathcal L_{hidden}=
\frac{1}{|I|}\sum_{i\in I}\sum_{k\in\mathcal K}w_kD_{joint}(h_{k,i},h^*_{k,i})
\]

\[
\mathcal L_{vocab}=\frac{1}{|I|}\sum_{i\in I}R_i
\]

\[
\mathcal L_{global}=\mathcal L_{hidden}+\lambda_v\mathcal L_{vocab}
\]

冻结规则：

- 只有 `I` 中的位置成为 trainable parameter；
- fixed prefix、padding 和非 `I` 位置从不进入 optimizer parameter；
- forward 时把 trainable positions scatter/index-copy 回 detached full-sequence base；
- 整个 global stage 只创建一个 persistent Adam；
- no weight decay；
- no scheduler；
- constant configured LR；
- anchor refresh 不重置 Adam moments。

输出 `E_work` 只是 causal loop 的 continuous warm-start，不是正式最终 embedding。

### 6.6 严格左到右 mixed-context 循环

访问有效位置 `i` 时构造：

\[
M_i[j]=
\begin{cases}
\operatorname{Emb}(t_j^{commit}), & j<i\\
e_j^{work}, & j\ge i
\end{cases}
\]

其中：

- `j<i` 必须使用当前已提交 token embedding；
- `j>i` 的 continuous embedding detached；
- special/fixed prefix 和 padding 保持入口不变量；
- 禁止复用与当前 committed prefix 不一致的 stale KV cache；
- 只读取位置 `i` 的 `L/L+1/L+2` hidden。

#### 6.6.1 Continuous quality gate

fresh 计算：

\[
c_i=D_{hidden,multi}(M_i,i)
\]

\[
J_i=c_i+\lambda_vR_i
\]

若 `J_i<=tau_J`，直接冻结该位置的 continuous baseline。

若 `J_i>tau_J`，只允许一次 local vector repair：

- 只把 `e_i` 设为 trainable；
- committed prefix、future continuous embeddings、padding 全部 frozen/detached；
- 每个位置新建一个只包含 `e_i` 的 Adam，trial 内 persistent；
- 使用与 global stage 相同的 hidden/vocab objective family；
- local steps/LR 独立配置。

trial 完成后，对 old/new embedding 都做 fresh anchor retrieval 和 fresh mixed forward。仅当：

\[
J_{new}<J_{old}-\epsilon_J
\]

且：

\[
c_{new}\le c_{old}+\delta_{c,max}
\]

才接受 trial；否则恢复旧 `e_i`。

非有限 trial 若旧 formal state 未污染，则拒绝本次 repair 并继续，不做 sample rollback。

#### 6.6.2 Continuous baseline freeze

repair 决策后 fresh 重算并冻结：

- `c_i`：最终 continuous hidden-only error；
- `R_i`：最终 fresh vocabulary soft-min；
- `J_i=c_i+lambda_v R_i`。

candidate 阶段开始后 `e_i` 不得再次变化。

如果最终 `J_i>tau_J`：

- 初始候选池设为 expanded；
- 记录 `unresolved_continuous_quality=true`；
- 不视为 hard failure。

### 6.7 离散候选、评分与一次 local degradation repair

Normal pool 来源：

- final continuous `e_i` 的 legal embedding neighbors，深度 `K_e_normal`；
- `PPL(committed_tokens[:i])` 的 top `K_p`；
- pre-selection current working token。

Expanded embedding source 使用 `K_e_expanded>K_e_normal`。最终候选集合满足：

\[
C_i^{normal}\subseteq C_i^{expanded}
\]

由于跨来源去重，expanded 后候选数不一定严格增加；若新增数为 0，记录 `expanded_added_count=0`，仍视为 expanded attempt 已完成。

候选处理：

- legal filter；
- 按 token ID 去重；
- 合并 source metadata；
- 不存在 classifier branch；
- current token 是普通候选，无特殊优先级。

对候选 token `q` 构造：

\[
[\operatorname{Emb}(t_{<i}^{commit}),\operatorname{Emb}(q),E_{>i}^{work}]
\]

future continuous embeddings 保持 detached；只读取位置 `i` 的有效层 hidden。单个候选 score 非有限时丢弃该候选；全部候选均非有限才是 hard failure。

确定性排序键：

```text
(d_i(q), token_id)
```

首轮选择 `d_old` 后计算：

\[
r_i=\max(0,d_i-c_i),\qquad 0\le r_i\le1
\]

若 `r_i<=tau_r`，直接 commit。

若 `r_i>tau_r` 且首轮为 normal：

1. 只执行一次 expanded rerank；
2. expanded pool 是首轮完整 pool 与 expanded embedding source 的 union，必须保留首轮 selected token；
3. 只有 `d_new<d_old-epsilon_d` 才替换；
4. 更新 `d_i` 与 `r_i`，但不改变 frozen `c_i`。

若首轮已 expanded，或 expanded 后仍 `r_i>tau_r`：

- 记录 `unresolved_local_degradation=true`；
- 不返回 vector repair；
- 不二次 rerank；
- 不累计 residual；
- 不维护 CUSUM；
- 不 replay；
- 不 rollback。

随后 commit token，进入 `i+1`。下一位置的 mixed context 会自然接收当前 token 对后续 hidden 的可观察影响。

### 6.8 明确删除的 v2.0 机制

v2.1 canonical 中不存在：

- Phase 1/Phase 2；
- per-step SGD；
- prox loss；
- range loss；
- classifier provider/config/top-k；
- sample-wide continuous median/MAD threshold；
- `g_i`、`delta_g`、`G_i`、`z_i`、`S_i`；
- local gap threshold；
- CUSUM；
- cumulative segment start；
- replay/checkpoint/history rebuild；
- `cumulative_max_repairs_per_trigger`。

### 6.9 正式返回状态

成功后：

- `final_tokens` 为左到右 commit 后的 token sequence；
- 主返回 `final_embedding` 与 `final_tokens` 一致：有效 committed 位置写入 `Emb(final_token[i])`；fixed structural/padding 位置保持入口冻结 embedding；
- `E_work` 和每个位置 repair 后的 continuous embedding 只作为内部状态或 JSONL 诊断，不作为主返回 embedding；
- 如后续 CGMR 需要 continuous reference，必须通过显式命名字段和独立接口设计决定，不能把主返回 embedding 的语义再次模糊化。

### 6.10 Failure domain

| Hard failure：整体回滚入口快照 | Local fallback：保持正式旧状态继续 | Offline-only：不得改变正式结果 |
|---|---|---|
| v2.1 显式选择但 disabled | vector repair trial 非有限但旧 `e_i` 完整 | accuracy |
| 输入长度、shape、mask 合同破坏 | repair 无严格 `J` 改善 | Ground Truth token correctness |
| external layer/target hidden key 不匹配 | hidden cap 不通过 | oracle prefix/per-position accuracy |
| 无有效位置或有效层 | repair 后 `J_i>tau_J` | oracle aggregate statistics |
| legal vocab 不足 top-k 合同 | `unresolved_continuous_quality` | diagnostics exception |
| 必需 helper 缺失 | 单个 candidate 非有限 | `diagnostics_failed` |
| causal attention 或 stale-cache 合同无法保证 | expanded 没有新增 token | 任何 accuracy-derived report |
| global formal state/loss/update 非有限 | expanded 未达到 `epsilon_d` 改善 | 任何 Ground Truth-based comparison |
| 正式 candidate pool 为空 | `unresolved_local_degradation` |  |
| 全部 candidate score 非有限 | 部分 candidate source 为空但 pool 非空 |  |
| committed state/final invariant 破坏 |  |  |

总原则：只有能够证明异常发生在未提交 local trial 中、且旧 formal state 未被污染时，才允许 local fallback；否则按 hard failure。

### 6.11 Formal result 与 offline diagnostics 隔离

正式方法 domain 完成后先冻结：

- `final_tokens`；
- `final_embedding`；
- `accepted/rollback`；
- 正式事件与失败语义。

随后才进入独立 diagnostics domain：

- `accuracy_diagnostics_enabled=false` 时 zero Ground Truth read；
- enabled 时可读取 Ground Truth 计算离线 accuracy；
- diagnostics exception 只记录 `diagnostics_failed=true`；
- diagnostics 不得改变 token、embedding、accepted 或 rollback。

### 6.12 Artifact 口径

`resolved_config.json` 必须记录 selector、结构常量、数值容差与全部待校准参数。

`reconstructions.jsonl` 至少记录：

- per-position `c_i/R_i/J_i`；
- vector repair 是否触发、trial 是否安全、是否接受、old/new 指标；
- 初始 pool mode；
- candidate token/source/score；
- nonfinite candidate 丢弃；
- `d_old/d_final`；
- `r_old/r_final`；
- expanded attempt 与 `expanded_added_count`；
- unresolved flags；
- hard failure/local fallback；
- 独立 offline diagnostics 与 `diagnostics_failed`。

`experiment.log` 保持项目既有固定摘要，不增加 v2.1 方法明细。

## 7. 最终 config 字段分组

### 7.1 结构常量或结构开关

| 字段 | 语义 |
|---|---|
| `suffix_reoptimization_v2_1` | 独立 v2.1 sidecar enable；显式选择但关闭时 fail-fast |
| `suffix_v2_1_layer_offsets` | 固定 external hidden collection offsets `[0,1,2]` |
| `suffix_v2_1_global_optimizer` | 固定 persistent Adam |
| `suffix_v2_1_local_optimizer` | 固定每位置新建、trial 内 persistent Adam |
| `suffix_v2_1_weight_decay_enabled` | 固定 false |
| `suffix_v2_1_scheduler_mode` | 固定 none |
| `suffix_v2_1_vocab_distance_mode` | 固定 mean squared L2 |
| `suffix_v2_1_vocab_softmin_mode` | 固定 normalized stable log-sum-exp soft-min |
| `suffix_v2_1_candidate_tie_break_mode` | 固定 `(hidden_error,token_id)` |
| `suffix_v2_1_accuracy_diagnostics_enabled` | formal domain 后是否运行 oracle diagnostics |
| `suffix_v2_1_filter_nonascii` | anchors 与全部 candidate source 共用的 legality policy |

### 7.2 数值稳定与严格改善容差

| 字段 | 语义 |
|---|---|
| `suffix_v2_1_hidden_epsilon` | cosine denominator 与 norm metric 的稳定量 |
| `suffix_v2_1_epsilon_J` | vector repair 的最小严格 `J` 改善 |
| `suffix_v2_1_epsilon_d` | expanded rerank 的最小严格 `d` 改善 |

### 7.3 待实验校准

| 字段 | 语义 |
|---|---|
| `suffix_v2_1_layer_weights` | 原始层权重；越界过滤后归一化 |
| `suffix_v2_1_alpha_dir` / `suffix_v2_1_alpha_mag` | 方向/尺寸权重；运行时归一化 |
| `suffix_v2_1_vocab_weight` | `lambda_v` |
| `suffix_v2_1_vocab_temperature` | soft-min temperature `tau_v` |
| `suffix_v2_1_vocab_anchor_top_k` | anchor 数 `K_v` |
| `suffix_v2_1_vocab_anchor_refresh_interval` | global/local 共用 anchor refresh cadence |
| `suffix_v2_1_global_steps` / `suffix_v2_1_global_lr` | single global Adam 预算 |
| `suffix_v2_1_local_steps` / `suffix_v2_1_local_lr` | 单位置 vector repair 预算 |
| `suffix_v2_1_adam_beta1` / `beta2` / `adam_epsilon` | Adam 数值参数 |
| `suffix_v2_1_tau_J` | 模型/embedding-scale/vocab-definition specific continuous gate |
| `suffix_v2_1_delta_c_max` | repair acceptance 允许的 hidden-only 最大恶化 |
| `suffix_v2_1_tau_r` | `[0,1]` 内的 local discretization degradation threshold |
| `suffix_v2_1_embedding_top_k_normal` | normal embedding candidate depth |
| `suffix_v2_1_embedding_top_k_expanded` | expanded embedding candidate depth |
| `suffix_v2_1_ppl_top_k` | PPL candidate depth |

不存在 v2.1 config 字段：classifier、prox、range、MAD、CUSUM、cumulative、replay。

## 8. 最终幻觉与信息泄漏审查

1. 本文所有 v2.1 机制均是已冻结设计，不是已实现代码事实。
2. 当前没有 v2.1 accuracy、耗时、repair gain 或稳定性证据。
3. `L` 仅表示 external hidden collection index；不推断 transformer block 编号。
4. 当前样本 future `c/J/d/r`、全位置 median/MAD、未来 token 与 Ground Truth 不得参与当前位置决策。
5. future continuous tensor 可以存在，但必须 detached，并依赖 causal attention 阻止 `j>i -> i`；无法保证时 hard failure。
6. `c_i` 必须是 vector decision 后、token candidate 前冻结的 final continuous hidden baseline。
7. `d_i` 必须与 `c_i` 使用相同 committed prefix、层、层权重和方向/尺寸定义。
8. expanded rerank 只更新 `d_i/r_i`，不能移动 `c_i`。
9. candidate outcome 不能返回修改 `e_i`；每位置最多一次 vector repair 和一次 expanded rerank。
10. 唯一跨位置传播是 `commit(t_i) -> Emb(t_i) -> c_{i+1}`，不存在 residual history 或 replay feedback loop。
11. `tau_J` 绑定 `(model,V_legal,q,K_v,tau_v,lambda_v)`，不能写成跨模型理论常数。
12. 正式结果必须先于 Ground Truth diagnostics 冻结。

## 9. 最终冻结结论

经三轮项目内讨论和对抗审查后，suffix v2.1 已足够冻结为 canonical 方法规范；当前没有未决的结构性问题。所有未冻结内容均为明确列出的实验校准数值，而不是方法流程歧义。

配套架构图：

- `suffix_v2.1_方法架构图.drawio`：可编辑源文件；
- `suffix_v2.1_方法架构图.svg`：可直接预览的矢量图。
