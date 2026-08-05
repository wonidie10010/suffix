# AGENTS.md

本文件记录 `D:\matt\DEML-main-v0` 的项目级协作规则。全局规则仍然适用；如果与本文件冲突，以更严格、更贴近本项目的规则为准。

## 更新文档写作约定

当用户要求写“更新文档”“改动说明”“版本更新”“实验更新 Word”“方法更新说明”“vX 相对 vY 的改动”等文档时，必须按本节执行。

1. 先根据方法读取对应模板，并按其中的固定结构组织内容：
   - suffix 方法：`suffix_optimization_methods/更新/更新文档写作模板.md`
   - CGMR 方法：`CGMR/更新/更新文档写作模板.md`
2. 涉及实验结果时，必须从真实 artifact 核对数据，优先检查：
   - `results/invert_timestamp_runs/<method_version>/<timestamp>/resolved_config.json`
   - `results/invert_timestamp_runs/<method_version>/<timestamp>/experiment.log`
   - `results/invert_timestamp_runs/<method_version>/<timestamp>/reconstructions.jsonl`
3. 涉及方法、损失函数、配置或流程时，必须从当前代码和 config 核对事实，不要只根据文件名或历史印象推断。
4. 文档中要明确区分：
   - 参考分析或用户草稿中的观点
   - 当前代码中已确认的真实实现
   - 基于实验 artifact 得到的结果
5. 对不确定点必须写明“经代码确认”“尚需验证”或“参考分析推测”，不能把推测写成事实。
6. Word 文档的内容顺序固定为：
   - 发现了什么问题
   - 做了哪些改动
   - 为什么要做这些改动
   - 改动后的实验效果
   - 后续改进方向
7. 如果生成 `.docx`，应按 documents skill 做结构检查；若本机有 LibreOffice/`soffice`，再做渲染 QA。若无法渲染，最终说明中必须注明只完成结构 QA。

## DEML artifact 口径

- 当前项目的实验产物主要位于 `results/invert_timestamp_runs/<method_version>/<timestamp>/`。
- `output_dir` 主要作为实验命名语义，真实文件写入 timestamp run 目录。
- 以后运行实验不再生成或更新 Excel / `.xlsx` 汇总表；实验结果以 `resolved_config.json`、`experiment.log` 和 `reconstructions.jsonl` 为准。
- 汇报实验时不要只看历史文件夹名称；要核对 `resolved_config.json` 中的方法开关、参数和运行时配置。
- 对 suffix 方法，注意区分：
  - 单独 baseline run 的最终 `accuracy`
  - suffix 方法内部的 `pre_acc` / `before_accuracy`
  - suffix 方法最终 `post_acc` / `final_accuracy`

## 实验日志维护约定

1. `experiment.log` 由主流程通过 `experiment_outputs.py` 统一维护，只保留固定的样本阶段正确率摘要和最终平均正确率；方法 sidecar 不得直接写入该文件。
2. 以后进行代码更新或方法迭代时，除非用户明确要求修改 `experiment.log` 的格式或内容，新增和调整的方法明细只能写入 `reconstructions.jsonl`，不得向 `experiment.log` 增加字段、事件或中间状态。
3. 历史 config 中的 `*_log` 字段仅为兼容旧配置保留，不得用它们重新开启详细 `experiment.log`。

## 临时文件与原文件编辑约定

1. 任务过程中产生的临时文件，例如测试工作簿、渲染预览、中间转换结果、调试输出和一次性脚本，统一放在项目根目录的 `outputs/` 中；必要时可按任务建立临时子目录，避免污染主目录。
2. `outputs/` 默认是临时工作区，不是用户要求的最终交付位置。任务完成前，必须删除本次任务在 `outputs/` 中新建的全部临时文件和目录，并确认没有遗留。
3. 清理时只能删除当前任务明确产生的临时内容；不得删除 `outputs/` 中原有的、来源不明的或用户保留的文件。
4. 当用户指定修改某个已有文件时，例如要求在 `111.xlsx` 中进行操作，默认直接编辑该原文件，保持原路径和原文件名；不得在 `outputs/` 或其他位置另行生成一份同内容的交付文件。
5. 只有在用户明确要求保留原件、创建副本、输出新版本，或原地编辑在技术上无法完成时，才可以新建文件。如果必须新建，应先说明原因和最终保存位置。
6. 对原文件进行修改后，应直接验证原文件的内容、格式和可读性；若验证过程产生预览图、缓存或中间文件，仍须放入 `outputs/` 并在任务结束前清理。

## Suffix 方法版本升级流程

当用户要求修改 suffix 方法、推出新版本、相对旧版本升级或保留可回退实现时，必须按本节执行。

1. 不直接覆盖当前最高版本；先在 `suffix_optimization_methods/method_versions/` 中新建下一版本完整 sidecar 文件，例如 `suffix_v1_5_0.py`。v1.2.3 及后续新版本由 sidecar 自己运行 Stage 1 初始重构与 Stage 2 suffix reoptimization。
2. 旧版本文件必须保留，除非用户明确要求删除；旧版本应继续可以被 config 选择和运行。
3. 每个新结构 suffix 版本必须有独立 config 文件，例如 `suffix_optimization_methods/configs/suffix_v1_5_0.json`，参数命名使用版本前缀 `suffix_v1_5_0_*`；v1.0–v1.2.2、v1.3–v1.4.1 的旧文件名和实现路径保持不变。
4. 版本切换必须通过显式 selector 或等价 config 机制完成；当前约定是 `suffix_optimization_methods/configs/advanced_methods.json` 中的 `suffix_version`。
5. `invert.py` 只做轻量 orchestration：import、config dataclass 初始化、版本选择、run 分支和结果落盘；复杂方法逻辑放在 sidecar 文件中。
6. `experiment_outputs.py` 必须同步更新，使新版本参数写入 `resolved_config.json`；方法细节和新增结果字段写入 `reconstructions.jsonl`，固定的 `experiment.log` 摘要保持不变。
7. 新版本必须补最小可运行测试，优先测试新增纯函数逻辑和版本选择逻辑。
8. 实现后必须按更新文档模板在 `suffix_optimization_methods/更新/` 下写版本更新报告；若没有完成完整实验，报告的实验效果部分必须写明“尚需完整实验验证”。
9. 更详细步骤参见 `suffix_optimization_methods/更新/方法版本升级流程.md`。

## CGMR 方法版本升级流程

当用户要求修改 Confidence-Gated Multi-Layer Reranking（CGMR）方法、推出新版本、相对旧版本升级或保留可回退实现时，必须按本节执行。

1. 不直接覆盖当前最高版本；在 `CGMR/method_versions/` 中新建下一版本 sidecar 文件。
2. 旧版本文件和 config 必须保留，并继续可由 config selector 选择。
3. 每个版本必须有独立 config 文件，参数使用版本前缀 `cgmr_vX_Y_*`。
4. 版本切换通过 `CGMR/configs/candidate_reranking_methods.json` 中的 `cgmr_version` 完成。
5. CGMR 与 suffix 使用独立 selector；执行顺序固定为完整 suffix 或冻结 baseline 上游方法完成后，再运行 CGMR token 候选后处理。
6. `invert.py` 只做 import、config 初始化、版本选择、调用和结果落盘；候选构建、多层评分、置信门控和接受逻辑放在 sidecar 中。
7. `experiment_outputs.py` 必须同步记录 selector 和版本配置；方法细节和新增结果字段写入 `reconstructions.jsonl`，固定的 `experiment.log` 摘要保持不变。
8. 新版本必须补最小可运行测试；实验与验证可以按用户指令延后，但不得把未验证结果写成已确认事实。
9. 更新 Word 写入 `CGMR/更新/`；详细步骤参见 `CGMR/更新/方法版本升级流程.md`。
