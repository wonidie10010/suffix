# suffix v2.2.2 相对 v2.2.1 的关键核心更新

日期：2026-09-16。状态：已实施并完成本机 CPU 合成/mock 验证；真实模型验证与完整实验尚未执行。

资料口径：以本次工作树代码、配置与测试为准；方法依据为用户确认的[实现规格与实验方案](../方法描述/v2.2.1/检查点诊断（未实现）/检查点诊断_实现规格与实验方案.md)。没有使用新实验 artifact 推断方法效果。项目引用的升级流程和更新模板在工作树中为既有删除，本次完整读取 HEAD 历史内容作为流程参考，未恢复这些文件。

## 1. 发现了什么问题

旧 v2.2.1 先生成完整暂定离散序列，再逐位置执行 R。R 未触发或预算耗尽时直接跳过，R 内部又可能重排未来后缀，因此不能把“已有 token”或内层 rerank 当作正式 checkpoint 提交边界。

旧实现没有候选质量下限、有效候选表索引或段级审查和单点修复。旧一键程序比较 baseline 与 baseline+R，不能直接作为 CP-off/on 的共享起点对照。

## 2. 做了哪些改动

| 改动 | 本次实现 |
| --- | --- |
| 独立版本 | 新增 `method_versions/suffix_reoptimization_v2_2_2.py` 及独立 config；旧 v2.2.1 保留；总 selector 默认仍为 v2.1.1 |
| 完整两阶段 | 新 sidecar 承担 legacy Stage-1 无 GT 优化计算，再运行 R+CP；提供共享快照写入/读取入口 |
| 提交调度 | 外层位置统一收尾，固定 5-token、不重叠、尾段跳过；R=0 仍完成离散恢复与 CP |
| 段级诊断 | 原始离散 hidden 求和 cosine <0.90 时触发；eta=0.05；优先最早持续下降，否则选最大孤立下降 |
| 候选复查 | 有效旧表按单点 cosine >=0.90、合法 ID/special/ASCII 筛选；full-prefix 顺序前向，无 KV cache |
| 事务处理 | 所有候选成功评分后才允许选优；任一候选失败或新分数不可用，整个 CP 保留旧状态；系统致命错误和适配契约错误终止运行 |
| 接受与连续性 | 最佳新段级 cosine 严格改善才替换一个 ID，不加 margin、不递归；同步 embedding 并刷新未来失效候选 |
| 结果记录 | 配置、环境、CP 明细与离线得失进入 JSON artifact；固定 experiment.log 不变；不生成 Excel |
| 一键程序 | 新增 `实验/一键运行_suffix_v2_2_2.py`、内部 runner 和两组配置，支持 dry-run、smoke、正式模式 |

R 正式配置保持：always、每样本 2 次、每位置 1 次、50 步、lr=0.03、front-decay=0.9/floor=0.2、prox=0.005、range=0.001、range_top_k=10。smoke 的小预算单独写入有效配置。候选下限 0.90 和 eta=0.05 为用户指定，未实验定标；两个 0.90 使用独立字段。

## 3. 为什么要做这些改动

统一收尾避免 R 跳过分支漏掉 checkpoint；有效表和事务隔离避免使用被拒绝试算的候选或提交部分评估的赢家。未来刷新保证修复影响后续恢复，同时不重做已完成段。

完整前缀前向明确离散因果评分语义，代价是额外计算。段级 hidden 改善只是代理目标，不能据此宣称 token 更正确。

两组复用同一份 Stage-1 快照及随机状态，runner 核对样本顺序、pair_id、快照 SHA-256、参数和真实 timestamp artifact 路径。离线统计区分直接替换得失与最终全序列得失，避免把后续路径变化全部归为单点替换。

## 4. 改动后的实验效果与实际验证

**尚需完整实验验证。** 本次没有运行真实 smoke 或正式 12 样本对照，没有推送 GitHub 或同步服务器，因此没有实验样本准确率或平均提升可报告。

实际验证：

- **61 项测试通过**：新版 sidecar、配置/selector/目标采集及 Stage-1 集成、runner mock；旧 v2.2.1 sidecar/runner、输出目录和固定日志回归。
- **8 个 Python 文件编译通过**：新版 sidecar、一键入口、内部 runner、3 个新测试文件、invert.py、experiment_outputs.py。
- 同一合成输入下 CP-off 与旧 v2.2.1 的 R 结果及 embedding 一致；Stage-1 新函数与直接执行现有 legacy 循环 AST 的输出一致。
- 覆盖 R 各分支、零预算/零恢复长度/尾段、整体回滚、严格选优、未来刷新、tuple hidden、hook 清理和在线/离线标签隔离。
- runner mock 验证共享快照 write/read、失败停止与清理；实际从非项目根目录调用 dry-run 通过，没有启动实验。
- `git diff --check` 通过；旧版本算法和配置未修改。

验证环境是 Windows、Python 3.13.12、任务专用临时 CPU PyTorch 2.14.0+cpu，不是服务器 requirements 的 CUDA/Transformers 环境。测试使用合成小模型，未加载 Qwen 权重。涉及 invert 的集成直接执行真实函数/legacy 循环 AST，不等同于导入完整 Hugging Face 入口并运行真实 smoke。

模型文件预检发现并复用了现有缓存：

- ID：`Qwen/Qwen2.5-1.5B`。
- 路径：`D:/cache/huggingface/transformers/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323`。
- revision：`8faed761d45a263340a0528343f099c05c9a4323`；`model_cache_status=hit`、`download_performed=false`。
- 已检查 config、tokenizer 文件及权重存在且非空，权重大小 3,087,467,144 字节；这是文件预检，不是加载或推理验证。
- 本机缺少 `data/cms.json` 与 `data/ECHR-ACL2019/EN_test`，列为服务器运行前待核对项，不自动下载。

新版使用显式路径、环境变量及既有共享缓存；多个来源不明确时要求显式路径，不新建版本专属缓存、不修改旧入口。真实运行离线复用已核对的本地模型。

## 5. 后续执行顺序

当前实施阶段到此停止。收到用户后续指令后才执行：更新 GitHub → 服务器同步同一提交 → 核对环境/数据/模型来源 → 真实 smoke → 通过后运行正式 CP-off/on 两组。

```text
# 仅布局、配置与文件检查，不加载模型
python 实验/一键运行_suffix_v2_2_2.py --dry-run

# 以下等待后续运行指令
python 实验/一键运行_suffix_v2_2_2.py --smoke-test
python 实验/一键运行_suffix_v2_2_2.py
```

无参数入口会执行正式实验，不能作为当前实施检查命令。服务器阶段仍需验证真实 Qwen 层输出、dtype/attention 行为、快照一致性、成本和恢复效果；本机合成测试不替代这些验证。
