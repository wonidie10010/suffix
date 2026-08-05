# 内部文件

这里存放一键实验入口所依赖的 bootstrap、suffix v2.0 四卡 runner、旧版兼容
runner 及其单元测试。请从上一级目录运行 `一键运行_suffix_v2_0.py`；正式入口不会
调用旧版 runner，也不会选择 suffix v2.0 之外的方法。

启动器直接使用仓库根目录中的项目源码、`requirements.txt` 和实验配置，不再维护
内嵌项目副本。运行环境与日志仍写入上一级目录的 `.runtime/`，
实验结果仍保留在仓库根目录的 `results/` 并额外复制到 `实验/结果/`。
