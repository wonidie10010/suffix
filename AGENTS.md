# AGENTS.md

本文件记录 `D:\matt\DEML-main-v0` 的项目级协作规则。全局规则仍然适用；如果与本文件冲突，以更严格、更贴近本项目的规则为准。

## 更新文档写作约定

### Suffix 方法

当用户要求为 suffix 方法写“更新文档”“改动说明”“版本更新”“实验更新 Word”“方法更新说明”或“vX 相对 vY 的改动”时，必须先完整读取并执行：

- `suffix_optimization_methods/更新/更新文档写作模板.md`
- `suffix_optimization_methods/更新/更新文档数据核对清单.md`

这两个文件分别是 suffix 更新文档结构与实验数据核对的单一事实来源，本文件不重复其细则。

### CGMR 方法及无独立模板的方法

CGMR 的独立提示词文件已移除。为 CGMR 或其他没有独立模板的方法撰写更新文档时，按以下规则执行：

1. 涉及实验结果时，必须从真实 artifact 核对数据，优先检查：
   - `results/invert_timestamp_runs/<method_version>/<timestamp>/resolved_config.json`
   - `results/invert_timestamp_runs/<method_version>/<timestamp>/experiment.log`
   - `results/invert_timestamp_runs/<method_version>/<timestamp>/reconstructions.jsonl`
2. 涉及方法、损失函数、配置或流程时，必须从当前代码和 config 核对事实，不要只根据文件名或历史印象推断。
3. 文档中要明确区分：
   - 参考分析或用户草稿中的观点
   - 当前代码中已确认的真实实现
   - 基于实验 artifact 得到的结果
4. 对不确定点必须写明“经代码确认”“尚需验证”或“参考分析推测”，不能把推测写成事实。
5. Word 文档的内容顺序固定为：
   - 发现了什么问题
   - 做了哪些改动
   - 为什么要做这些改动
   - 改动后的实验效果
   - 后续改进方向
6. 如果生成 `.docx`，应按 documents skill 做结构检查；若本机有 LibreOffice/`soffice`，再做渲染 QA。若无法渲染，最终说明中必须注明只完成结构 QA。

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

当用户要求修改 suffix 方法、推出新版本、相对旧版本升级或保留可回退实现时，必须先完整读取并严格执行 `suffix_optimization_methods/更新/方法版本升级流程.md`。该文件是 suffix 版本隔离、配置与 selector、主流程接入、输出、测试和更新报告要求的单一事实来源；本文件只负责触发，不重复具体步骤。

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
9. 更新 Word 写入 `CGMR/更新/`。
