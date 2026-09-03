# DEML suffix v2.2.1：执行版方案

状态：实现、针对性测试、GitHub/服务器同步、smoke 和两组 formal 已完成。正式结果以服务器上的三件 artifact 与 bundle manifest 为准。

## 1. 本次已经确定的范围

- 正式数据集：Skytrax、CMS、ECHR Law。
- 每个数据集只取前 4 条，正式实验共 12 条；不做论文数据的逐条完全复现。
- 两次正式运行使用完全相同的 12 条样本、顺序、模型、seed 和实验环境。
- 固定前缀暂按攻击者已知处理；recover 区域的 GT 不进入 formal 运行过程。
- 候选参数采用 v1.3.1 候选值。
- 不设置 `eps` 数值门槛；只要 R 让 hidden loss 真正下降，就接受，否则回滚。

## 2. 数据和 `EN_test`

| 数据集 | 服务器资源 | formal 样本 |
|---|---|---|
| Skytrax | `skytrax-reviews-dataset/data/airline.csv` | 前 4 条 `content`；不能换成 `airport.csv` |
| CMS | `data/cms.json` | 前 4 条文本 |
| ECHR Law | `data/ECHR-ACL2019/EN_test` | 按文件名稳定排序后的前 4 个 JSON，读取 `CONCLUSION` |

`EN_test` 不是新方法，也不是第四个数据集。它只是 ECHR-ACL2019 中的 “English test split”（英文测试集）目录。本次从这个目录取前 4 个文件即可。

当前服务器资源已经足够支撑 12 条小规模 formal 实验：Skytrax 上游约 4.1 万条 airline review，CMS 文本适配文件有 5,566 条，ECHR `EN_test` 有 2,998 个 JSON。CMS 当前文件不再追求与论文历史快照逐条一致，因为本次目标不是完整复现论文数据。

来源：[DEML 论文](https://arxiv.org/html/2503.09022v3)、[CMS 官方数据页](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service)、[ECHR-ACL2019 资源](https://archive.org/details/ECHR-ACL2019)。

本方案按“总共只有 12 条”理解：不再额外取 `[4:8]` 做 calibration。为了不凭 12 条正式样本调阈值，v2.2.1 采用固定的无阈值触发策略：按位置顺序尝试 R，直到达到每样本 2 次 attempt 预算；因此本版不定义 `tau_s`，也不读取 formal GT 来决定是否触发。

## 3. 两次正式实验

本次不需要跑三组，也不需要把 B0、E0、E1 当成新算法。只做以下两次独立运行：

| 运行 | 初始优化 | R 重优化 | 目的 |
|---|---:|---:|---|
| Run A：baseline | 1000 次 | 关闭 | 得到 baseline 结果 |
| Run B：baseline + R | 1000 次 | 开启；每次 50 次 | 测量重优化带来的变化 |

Run B 必须重新从相同 seed 和相同初始条件开始，不能把 Run A 的结果接着拿来重优化。这样 Run B 在开启 R 之前应得到同一份 baseline 状态；实现时保存这个状态，作为 Run A 与 Run B 的内部一致性检查即可，不需要额外跑第三组实验。

这里的“1000 epoch”对应初始 continuous optimization 的 `epoch=1000`；“50 epoch”在代码中对应每次重优化的 `suffix_v2_2_1_steps=50`，本质上是 50 次梯度更新。两次运行的 loss、候选、模型和其他参数保持一致，唯一的正式差异是 R 开关。

当前仓库已新增 `suffix_reoptimization_v2_2_1` sidecar、selector、独立 config、两份 formal config、runner 和针对性测试；已有 v1.3.1、v2.1 和 v2.1.1 只作为代码参考。这里“采用 v1.3.1 候选参数”只表示采用候选阶段的参数，不表示把 v1.3.1 的全部触发、接受或诊断逻辑搬进来。此前的 E0/E1/E2 只是方案讨论中的临时标签，本版正式报告不再使用它们。

## 4. v2.2.1 只实现 R 的最小合同

### 不变部分

- 保留 Original DEML 的 Stage-1 continuous optimization。
- 候选仍为 `EmbeddingTop10 + PPLTop10 + single-layer hidden cosine scorer`。
- 初始优化结束后的离散化，以及 R 连续重优化之后把 embedding 映射回 token 的步骤，都复用 Original DEML / Historical B0 的候选构建、排序和选 token 逻辑。
- 不把 v2.1.1 的多层 scorer、expanded pool、dedupe 或新 candidate policy 带进来。
- v2.0 的 CUSUM、累计 repair、v2.0 触发/修复循环等其他设计，本版不启用。

### R 的优化部分

- 触发后只优化当前位置及其后面的 suffix `[i:T)`；`<i` 的已提交 token 和 embedding 冻结。
- suffix 参数使用 FP32。
- `Adam`，`50` steps，`lr=0.03`。
- 前端衰减 `0.90`，下限 `0.20`；`prox_weight=0.005`；`range_weight=0.001`。
- trial 前保存完整 suffix snapshot；loss 没有下降或运行失败时完整回滚。
- 每个位置最多 1 次 attempt，每个样本最多 2 次 attempt；被拒绝的 trial 也消耗预算。

### 接受规则和信息边界

```text
accept iff H_post < H_pre
```

这里的 `H` 是 R 使用的 hidden loss。接受判断只读 hidden states/loss，不读 GT、目标 token ID、accuracy 或 `oracle_accuracy`。全部 12 条跑完后，才由独立 offline evaluator 读取 GT 计算准确率。

## 5. 服务器环境：已经找到以前真正跑通的环境

用户判断“之前已经跑过，所以肯定能跑”是有 artifact 支持的。服务器实际成功运行 v2.1/v2.1.1 使用的是：

```text
项目：/mnt/my_disk/tch/suffix
Python：/mnt/my_disk/tch/suffix/实验/环境和实验/.runtime/envs/deml-02d13e38b205/bin/python
模型：/mnt/my_disk/tch/models/Qwen2.5-1.5B
```

当前环境核验到：Python 3.10.20、PyTorch 2.11.0+cu126、Transformers 4.37.2、PEFT 0.7.1，CUDA 可用；机器有 2 张 RTX 3090。历史一键运行日志明确记录了该 Python 路径，并有 v2.1.1 完成后的结果复制记录。

因此，之前查到的 `/mnt/my_disk/tch/deml_experiment_setup/envs/deml-py310` 是旧的准备目录，不是本次 runner 应使用的环境。v2.2.1 runner 应直接复用上面的 `.runtime/envs/deml-02d13e38b205/bin/python`，并使用实际项目和模型路径；不需要重新安装一套 PyTorch。

本轮没有对服务器做 pull、覆盖、commit、push，也没有启动实验。服务器上本次准备的 ECHR/CMS 数据文件保留在原位置。

## 6. 一键运行流程

```text
检查数据、模型、环境和 Git SHA
  -> 固定 12 条样本及顺序
  -> Run A：baseline，初始优化 1000 次，R 关闭
  -> Run B：baseline + R，初始优化 1000 次，R 开启，每次 50 次
  -> 比较 Run A 与 Run B
  -> offline evaluator 读取 GT
  -> 检查每组 resolved_config.json / experiment.log / reconstructions.jsonl
```

每组保存上述三件 artifact，R 的明细只写入 `reconstructions.jsonl`，不扩展固定格式的 `experiment.log`。本地先做 `py_compile`、JSON 校验、mock 和 rollback/预算测试；不在本地下载模型或运行 formal。

## 7. 程序完成后的同步与执行操作

以下顺序就是本轮实际执行顺序；所有相对路径均以项目根目录为基准。

### 7.1 本地实现和验证

1. 新增 v2.2.1 sidecar、独立 config、selector 接入、两份 formal config、runner 和针对性测试。
2. 本地只运行 `py_compile`、JSON 语法校验、sidecar/mock/rollback/预算测试、selector/config/output contract 测试；不下载模型、不跑 12 条 formal。
3. 检查 `git diff --stat`、`git diff`，确认没有把模型、缓存、日志、结果或本次任务之外的删除文件加入提交。

### 7.2 GitHub 同步

1. 在提交前执行 `git status`、`git branch --show-current`、`git remote -v`、`git fetch origin`；如远程出现本地没有的提交，先停下处理，不覆盖远程。
2. 只暂存本次 v2.2.1 文件，检查 `git diff --cached` 后提交；禁止 `git add .`、force push、`git reset --hard` 和 `git clean -fd`。
3. 用户已要求同步，因此本轮测试通过后执行 `git push origin main`，并记录提交 SHA。

### 7.3 服务器同步

1. 在 `/mnt/my_disk/tch/suffix` 先检查 `git status`、当前分支和 `git remote -v`；未知 tracked 修改时停止，不覆盖。
2. `git fetch origin` 后只做可快进同步（`git pull --ff-only origin main`），不删除服务器已有的 `data/`、模型缓存或结果。
3. 对照本地 HEAD、`origin/main` 和服务器 HEAD；三者一致后再运行。服务器继续使用已核实的 `.runtime` Python 和 `/mnt/my_disk/tch/models/Qwen2.5-1.5B`。

### 7.4 smoke、正式运行与结果验收

1. 先执行 `python 实验/一键运行_suffix_v2_2_1.py --smoke-test`；smoke 只验证真实模型加载、selector、输出目录和三件 artifact，不计入正式结果。
2. smoke 通过后，runner 顺序启动 Run A baseline 和 Run B baseline+R；每组均重新从相同 seed/初始条件开始，Run B 的 R 只使用 hidden loss。
3. 每组检查 `resolved_config.json`、`experiment.log`、`reconstructions.jsonl`，确认 3 个数据集、每个 4 条、总计 12 条、CGMR 为 `none`、epoch=1000、R steps=50（Run B）。
4. 运行结束后再用 GT 做 offline 汇总，只报告每个数据集和 overall 的 token accuracy、样本均值，以及 R 的 trigger/attempt/accept/reject/预算统计。smoke、失败或不完整目录不进入最终结果。
5. 对两组 artifact 计算 SHA-256，并把运行 SHA、目录、配置摘要和最终统计写入 runner 的 bundle manifest；不生成新的 Excel 汇总表。

### 7.5 本轮实际执行记录

- 本地实现与静态/JSON/runner 合同测试完成；本地无 PyTorch，因此真实 sidecar 测试在服务器环境执行。
- GitHub 已同步到提交 `993f25443aec39b965fe0673a6271834369ad3da`；服务器已快进到同一提交，保留服务器原有数据、模型缓存和历史结果。
- smoke 已通过；正式运行在服务器完成。Run A 与 Run B 都生成了 12 条记录，且三件核心 artifact 均通过验收。
- 本轮正式结果目录和汇总 manifest：
  - `results/invert_timestamp_runs/frozen_original_baseline/20260903-185506`
  - `results/invert_timestamp_runs/suffix_reoptimization_v2.2.1/20260903-193500`
  - `实验/结果/suffix_v2.2.1_bundle/ablation_manifest.json`

## 8. 运行后状态与仍需注意的限制

1. **`eval_start_pos` 已实测确定**：12 条正式记录均为 `0`；没有额外插入 EOS，Run A/Run B 输入保持一致。
2. **formal 结果已实测完成**：Run A baseline 总体宏平均为 `0.6679977279`，Run B baseline+R 最终总体宏平均为 `0.7495538158`；逐数据集与 R 诊断见 bundle manifest 和最终汇报。
3. **仍需注意的限制**：正式样本只有 12 条（每个数据集前 4 条），因此结果只能说明本次固定样本上的行为，不能外推为论文规模结论；R 的接受判断仍然只依据 hidden loss，accuracy 仅在运行完成后离线统计。

除此之外，数据集、样本数量、样本位置、模型、服务器实际运行环境、v1.3.1 候选参数、触发策略和 GT 信息边界都已经确定。

## 9. 执行前 checklist

- [x] 冻结 fixed prefix 处理、运行时 `eval_start_pos`、模型路径和 runner 入口。
- [x] 实现 `suffix_reoptimization_v2_2_1`、selector 和独立 config。
- [x] 完成 Run B 重优化前后状态记录、GT 泄漏、rollback、预算的 mock 测试。
- [x] 服务器 PyTorch/CUDA 测试通过后，只提交本次相关文件并推送 GitHub。
- [x] 服务器 fast-forward 到同一 SHA，保留原有 data/模型/结果。
- [x] smoke 通过后运行两次 12 条 formal：每个数据集前 4 条。
- [x] 比较 Run A 与 Run B，并只用 offline GT 报告最终准确率。

正式结果和 SHA-256 manifest 已写入 `实验/结果/suffix_v2.2.1_bundle/ablation_manifest.json`；三件原始 artifact 仍以 `results/invert_timestamp_runs/<method>/<timestamp>/` 为准。
