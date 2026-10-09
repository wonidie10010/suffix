# suffix v2.5 相对 v2.4 的关键核心更新

日期：2026-10-10。v2.5 已实现，完成本地验证、真实独立标定及服务器 Off/On 配对实验，产物已回传校验。

本文以 `suffix_optimization_methods/方法描述/v2.4/方法修复.md` 为需求依据，以新增的 v2.5 代码、配置和测试为实施依据。工作区中已删除的版本升级流程、写作模板及数据核对清单从 Git HEAD 只读核对，保留其现有删除状态。

## 1. 发现了什么问题

v2.4 在初始离散化和每轮 R 后的离散化中执行回溯。回溯选中的路径可能继续参与后续 R 的前缀锚定，也可能随所在 R 被整体回滚。`small_gap` 可以独立触发搜索，且搜索遇到首条严格改善路线即停止。

## 2. 做了哪些改动

v2.5 的顺序固定为：

```text
Stage1 → 初始贪心锚定 → 全部 R 优化、贪心锚定及接受/回滚
       → 冻结最终连续嵌入和目标隐藏状态
       → 独立最终离散化、低拟合触发、有限回溯及贪心重走
       → 最终 token
```

| 机制 | v2.4 | v2.5 |
|---|---|---|
| 回溯阶段 | 初始离散化及每轮 R | 全部 R 结束后的最终离散化 |
| 必要触发条件 | 低拟合或小分差 | `s1 < tau_fit(p)`；小分差保留诊断 |
| 最大回退距离 | 2 | 4 |
| 每处备选 | 当前排名与历史赢家组合，最多 2 个 | 当前合法候选排名前 4 个不同的非原选 token ID |
| 每事件路线 / 每样本事件 | 6 / 2 | 16 / 2 |
| 每样本额外前向预算 | 64 | 640，最终阶段重新初始化并共享 |
| 向前看 | 2 个位置 | 2 个位置 |
| 选路 | 首条严格改善 | 比较完整评分路线，选择最小窗口误差的严格改善路线 |

每条路线只在一个位置改选，后续按当前前缀贪心重走。保留非负误差下界剪枝；固定前缀不回退。无合法回退位置时试探当前位置。预算中断时撤销本事件的试探，恢复普通贪心基准路径及对应决策表；未完成评分的路线不能被选中。

连续损失、R 接受条件、窗口误差 `sum((1-clamp(cosine,-1,1))/2)`、阈值标定方法和严格改善判据均沿用原定义；测试核对了 Stage1、R 优化、连续隐藏损失与窗口评分函数的 AST 一致性。

最终阶段仅将 Stage1 和已接受 R 的 Top1 轨迹作为候选 ID 提案，按其来源、位置、模型身份及锚定前缀契约校验，并在当前前缀下重新评分。拒绝的 R 不更新最终轨迹来源。前缀改选后重新生成后续 PPL 候选和评分。

`freeze_checks` 记录连续嵌入和目标的前后哈希、最终阶段前向调用的梯度状态、优化阶段回溯次数及最终 token/决策一致性。新增阶段与搜索明细写入 `suffix_reoptimization_v2_5_result`，随主流程写入 `reconstructions.jsonl`；`experiment.log` 保持固定样本及平均正确率摘要。

### 新增文件与使用方式

所有原有程序和配置保持不变；任务前记录的 566 个原有文件均通过 SHA-256 核对。为满足该隔离要求，v2.5 使用独立入口与结果输出模块。

| 新增文件 | 用途 |
|---|---|
| `suffix_optimization_methods/method_versions/suffix_reoptimization_v2_5.py` | 独立方法实现 |
| `suffix_optimization_methods/configs/suffix_reoptimization_v2_5.json` | 独立版本参数，使用 `suffix_v2_5_*` 前缀 |
| `suffix_optimization_methods/configs/advanced_methods_v2_5.json` | 独立 selector：`suffix_version=v2.5` |
| `experiment_configs/l24_deml3x4_suffix_v2_5.json` | v2.5 实验配置，保持原模型、数据及优化设置 |
| `invert_suffix_v2_5.py` | 独立主入口，调用 v2.5 sidecar |
| `experiment_outputs_suffix_v2_5.py` | v2.5 配置落盘、离线评估和固定日志输出 |
| `test/test_suffix_reoptimization_v2_5.py` | 阶段、搜索、冻结、配置与接入测试 |
| `test/test_suffix_v2_5_entry.py` | 本地 tiny Qwen2 CLI Off/On 冒烟测试 |
| `实验/环境和实验/内部文件/runner_suffix_v2_5.py` | 独立真实标定、预算检查、共享快照 Off/On 运行与回传校验 |
| `test/test_suffix_v2_5_runner.py` | 真实 R 配对、冻结边界、最终决策及完整路线选路校验 |
| `suffix_optimization_methods/更新/suffix_v2.5_更新报告.md` | 本报告 |

在项目根目录运行：

```powershell
D:\conda_envs\deml\python.exe -B .\invert_suffix_v2_5.py --config .\experiment_configs\l24_deml3x4_suffix_v2_5.json
```

当前独立配置的 `suffix_v2_5_backtracking_calibration` 为空，沿用原标定契约：`calibration_ready=false` 时不执行回溯。正式启用前，需要按目标模型和推理设置取得独立冻结标定，并写入新配置的该字段。Off 消融使用 `suffix_v2_5_backtracking_enabled=false`。`collect_backtracking_calibration` 使用独立贪心样本的最终离散决策和重复窗口前向，沿用原分位数、数值容差定义。

回退 v2.4 使用原入口和原配置：

```powershell
D:\conda_envs\deml\python.exe -B .\invert.py --config .\experiment_configs\l24_deml3x4_suffix_v2_4.json
```

## 3. 为什么要做这些改动

阶段隔离使 Off/On 共享相同的连续优化结果，回溯的差异从最终离散选择开始。必要低拟合门控约束搜索触发，小分差继续用于诊断。更大的候选与回退范围提供更多可行路线，完整路线比较避免停留在先遇到的改善路径；额外计算由每样本预算约束。效果和运行成本需要通过正式配对实验评估。

## 4. 改动后的实验效果

### 本地及服务器代码验证

| 验证范围 | 实际结果 |
|---|---|
| v2.5 方法及接入 | 21 项测试通过 |
| 独立 CLI 冒烟 | 1 项通过；tiny Qwen2，CPU，Off/On 各 2 条，合成标定仅用于测试 |
| 旧版本回归 | v2.4 方法与 runner、v2.3、v2.2.2(2) 共 56 项通过 |
| 独立 v2.5 运行器 | 本地及服务器各 8 项测试通过，服务器 dry-run 通过 |
| 编译及配置 | 新增 Python 文件编译、JSON 解析通过 |
| 原文件保护 | 实现时 566 个原有文件哈希一致，旧程序与旧配置未修改 |

Windows 混合梯度及 tiny Qwen2 fixture 测试曾在断言通过后异常退出；CLI 冒烟已隔离到子解释器并校验退出码。该处理仅涉及测试，方法逻辑未改变。正式实验使用独立真实冻结标定，未使用合成测试阈值。

### 真实配对实验

本次 run 为 `20261009-213018-v25-paired`，实验代码 commit 为 `445a9d887af13c73a42a283d61015c5f1679dd94`。模型固定 `/mnt/my_disk/tch/models/Qwen2.5-1.5B`，层索引 24、Stage1 epoch=1000、seed=0、CGMR=none、GPU1、float16、无量化。Skytrax、CMS、ECHR_Law 各 4 条，共 12 条、534 个评估 token。

2026-10-09 21:30 开始准备时，服务器项目磁盘未挂载；管理员恢复后，于 22:28 启动，2026-10-10 02:39 正常完成。四组均退出 0：独立贪心标定 6 条、独立预算检查 3 条、正式 Off 12 条、正式 On 12 条。

标定与正式、预算样本的文本及 token 哈希互不重叠。真实冻结标定 ID 为 `2c084634682789696f341825ba9ab253e233cb33acae5cc31d51bf3a683fe8c4`，普通位置拟合阈值为 0.8751738429069519，首位置阈值为 0.1608821451663971，严格改善 epsilon 与数值容差均为 1.3312687485722563e-07。默认独立配置仍保留空标定；本次实际标定记录在结果目录的 `frozen_calibration.json` 和 `effective_off.json` / `effective_on.json` 中。

两组共享 12 份 Stage1 快照及该标定。全部 R 的损失、接受/回滚、优化 token、最终连续嵌入与目标哈希完全一致；均为 17 轮接受、7 轮拒绝。优化阶段回溯、冻结后的 R 更新及开启梯度的最终前向均为 0，最终 token、保留决策及前缀指纹检查通过。

| 样本（索引从 0 开始） | token 数 | Off 正确 / accuracy | On 正确 / accuracy | 修复 | 损伤 | 净增 |
|---|---:|---:|---:|---:|---:|---:|
| Skytrax:0 | 115 | 105 / 0.913043 | 107 / 0.930435 | 4 | 2 | +2 |
| Skytrax:1 | 39 | 36 / 0.923077 | 36 / 0.923077 | 0 | 0 | +0 |
| Skytrax:2 | 64 | 59 / 0.921875 | 58 / 0.906250 | 0 | 1 | -1 |
| Skytrax:3 | 97 | 89 / 0.917526 | 91 / 0.938144 | 2 | 0 | +2 |
| CMS:0 | 16 | 9 / 0.562500 | 13 / 0.812500 | 4 | 0 | +4 |
| CMS:1 | 5 | 2 / 0.400000 | 2 / 0.400000 | 0 | 0 | +0 |
| CMS:2 | 11 | 7 / 0.636364 | 7 / 0.636364 | 0 | 0 | +0 |
| CMS:3 | 20 | 16 / 0.800000 | 17 / 0.850000 | 1 | 0 | +1 |
| ECHR_Law:0 | 26 | 24 / 0.923077 | 24 / 0.923077 | 0 | 0 | +0 |
| ECHR_Law:1 | 26 | 24 / 0.923077 | 24 / 0.923077 | 0 | 0 | +0 |
| ECHR_Law:2 | 29 | 26 / 0.896552 | 28 / 0.965517 | 2 | 0 | +2 |
| ECHR_Law:3 | 86 | 80 / 0.930233 | 79 / 0.918605 | 0 | 1 | -1 |

| 汇总口径 | Off | On | 差值 |
|---|---:|---:|---:|
| 正确 token / 总 token | 477 / 534 | 486 / 534 | +9 |
| micro accuracy（合计正确数 / 合计 token 数） | 0.893258 | 0.910112 | +1.6854 个百分点 |
| macro accuracy（12 条 accuracy 等权平均） | 0.812277 | 0.843920 | +3.1644 个百分点 |
| 相对共享 Stage1 初始离散输出的修复 / 损伤 | 74 / 21 | 79 / 17 | 净增益 53 → 62 |
| 第二阶段合计耗时（秒） | 2528.759 | 2628.042 | +99.283 |
| 额外搜索前向 | 0 | 2111 | +2111 |

直接比较最终 Off → On，修复 13 个、损伤 4 个，净增 9 个正确 token；5 条样本提升、2 条下降、5 条持平。Off 是 v2.5 的关闭回溯对照，并非单独原论文 baseline，也不是重新运行的 v2.4。内部 `pre_acc` 不作为原论文 baseline 结果。

On 共记录 17 次低拟合触发、启动 16 次搜索、接受 12 次路线，其中 10 次最终保留；没有预算耗尽。每样本最多 457 次额外前向、2 次事件，每事件最多 16 条路线，均符合 4 位置、4 备选、16 路线、2 事件、640 前向、向前看 2 的上限。5 次接受选择了首条改善之后更优的完整路线，核对所选误差等于完整路线最小值且严格低于原窗口误差减 epsilon。最终候选覆盖为 Off 499/534、On 500/534；新增覆盖 3 处、丢失覆盖 2 处，候选覆盖不等于 token 正确率。

On 的第二阶段耗时增加约 3.93%。两组不是等计算预算实验，且运行于共享服务器；On 复用 Stage1，不能直接用进程总耗时相除解释加速。当前结论限于这 12 条、seed=0 的配对验证；三项机制各自的贡献和跨样本稳定性仍需消融及多种子实验。

### 回传与复算

结果保存于 `实验/结果/suffix_v2_5/20261009-213018-v25-paired/`。四组均保留 `resolved_config.json`、固定格式 `experiment.log` 和 `reconstructions.jsonl`；同时保留 manifest、comparison、plan、launch、有效配置、样本拆分、真实标定、12 份共享快照、6 份标定快照、3 份预算快照及日志。54 个服务器源文件的字节数和 SHA256 全部与本地一致。

服务器及本地 Python 3.10.20 的运行器 verify 均通过，`comparison.json` 从原始 artifact 精确重算一致。另用回传的真实 Qwen2Tokenizer 重新编码四组 33 条 `original` 文本，逐条核对原文/参考 token 哈希、评估区间、逐位置正确性、macro/micro、修复/损伤及耗时。四组 `experiment.log` 的每条与总体平均值均核对到其 6 位小数，格式保持固定摘要。未生成 Excel。

复验已回传结果：

```powershell
D:\conda_envs\deml\python.exe -B .\实验\结果\suffix_v2_5\20261009-213018-v25-paired\verify_real_tokens.py
D:\conda_envs\deml\python.exe -B .\实验\环境和实验\内部文件\runner_suffix_v2_5.py verify --project . --runtime .\实验\环境和实验\.runtime --model-path /mnt/my_disk/tch/models/Qwen2.5-1.5B --bundle .\实验\结果\suffix_v2_5\20261009-213018-v25-paired --gpu 1
```

原文复算无需加载模型权重或 GPU。精确汇总校验使用与服务器相同的 Python 3.10.20；Python 3.13 的内置浮点求和使 Off macro 末位相差 1.11e-16，不涉及 token 或实验逻辑差异。验证证据见 `local_bundle_verification.json`、`independent_token_metrics.json`、`final_validation_details.json` 和 `sha256_manifest.json`。

## 5. 后续改进方向

1. 扩大样本并增加随机种子，检验当前净增 9 个正确 token 的稳定性。
2. 在冻结配对条件下分别消融必要低拟合触发、搜索规模和完整路线选优，区分机制贡献；增加等计算预算对照。
3. 针对本次 4 个损伤位置及剩余 48 个错误，结合候选覆盖和窗口评分追踪原因；修改门控前重新独立标定，避免依据正式标签调参。
