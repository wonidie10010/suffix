# suffix v2.5 相对 v2.4 的关键核心更新

日期：2026-10-09。v2.5 已实现并完成本地验证。

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

**尚需完整实验验证。** 本次完成代码与本地验证，未启动真实数据服务器实验。

| 验证范围 | 实际结果 |
|---|---|
| v2.5 方法及接入 | 21 项测试通过 |
| 独立 CLI 冒烟 | 1 项通过；本地生成的 tiny Qwen2，CPU，Off/On 各 2 条样本 |
| 旧版本回归 | v2.4 方法与 runner、v2.3、v2.2.2(2) 共 56 项通过 |
| 编译及配置 | 5 个新增 Python 文件通过 `py_compile`；3 个新增 JSON 可解析 |
| 原文件保护 | 566 个原有文件哈希一致 |
| 独立 v2.5 实验运行器 | 8 项测试通过；2 个新增 Python 文件通过编译 |

CLI 冒烟核对了实际写出的 `resolved_config.json`、`experiment.log`、`reconstructions.jsonl`，以及 canonical v2.5 字段、离线评估和固定日志格式。On 组使用明确标注的合成测试标定，实际启动回溯。两组复用相同 Stage1 快照，R 损失与接受结果、优化阶段 token、冻结连续嵌入哈希一致。最终阶段的嵌入和目标哈希不变，梯度开启的前向调用为 0。临时模型、快照和产物在验证后清理，测试可重复运行。

Windows 本地环境中，同一测试进程先执行真实梯度 R，再加载 tiny Qwen2 推理 fixture，会在断言通过后出现 `0xC0000005` 退出异常；单独的 CLI 冒烟正常退出。CLI 冒烟现运行在独立解释器，父测试同时校验子进程退出码。该隔离处理针对测试运行环境，方法和主入口的计算逻辑保持不变。

| 正式效果口径 | 状态 |
|---|---|
| 真实样本逐条 accuracy 与平均值 | 未运行 |
| v2.5 Off/On 的修复、损伤及净增益 | 未运行 |
| 真实模型独立阈值标定、GPU 前向预算与耗时 | 未运行 |

2026-10-09 21:30 开始正式实验准备。本次任务 ID 为 `20261009-213018-v25-paired`，持久状态与计划位于 `实验/结果/suffix_v2_5/20261009-213018-v25-paired/`。服务器预检发现 `/mnt/my_disk` 为空、2TB 分区 `/dev/sdb1` 未挂载；现有 SSH 账号恢复挂载需要 sudo 密码。已请求管理员恢复挂载，尚未启动真实标定及正式 Off/On。该状态不构成实验效果结果。

复验本地测试：

```powershell
D:\conda_envs\deml\python.exe -B -m unittest test.test_suffix_reoptimization_v2_5 test.test_suffix_v2_5_entry test.test_suffix_reoptimization_v2_4 test.test_suffix_v2_4_runner test.test_suffix_reoptimization_v2_3 test.test_suffix_reoptimization_v2_2_2_2
```

## 5. 后续改进方向

1. 在目标真实模型上完成独立冻结标定，再使用相同 Stage1 快照进行 v2.5 Off/On 配对实验，核对 R 和冻结哈希一致。
2. 从真实 artifact 统计逐样本 accuracy、最终修复/损伤、候选覆盖、路线接受与最终保留，以及实际额外前向次数和耗时。
3. 分别消融阶段顺序、必要低拟合触发及搜索规模，确认各项修复的贡献。
