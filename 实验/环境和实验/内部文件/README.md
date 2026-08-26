# 内部文件

这里存放一键实验入口所依赖的 Linux bootstrap、单卡 runner 及其单元测试。

- `实验/一键运行_suffix_v2_0.py`：suffix v2.0 独立入口。
- `实验/一键运行_suffix_v2_1.py`：原 suffix v2.1 文件名保留，现执行 suffix v2.1.1 独立实验。

两个入口都只启动一个实验进程，只使用 `DEML_GPU_ID` 指定的一张 GPU；原
v2.1 一键链路现不会选择 v2.0 或任何 CGMR 方法，而是运行 v2.1.1。v2.1.1
的正式配置是仓库根目录中的
`experiment_configs/l24_airport_medical_suffix_v2_1_1_no_cgmr.json`。

启动器直接使用仓库根目录中的项目源码、`requirements.txt` 和实验配置，不维护
内嵌项目副本。v2.0/v2.1.1 复用上一级目录的共享 `.runtime/`、Conda 环境和模型
缓存，但通过共享锁禁止并发一键启动；日志按版本命名，实验结果保留在仓库根目录
`results/invert_timestamp_runs/`，并额外复制到 `实验/结果/` 对应版本目录。

正式运行：

```bash
python 实验/一键运行_suffix_v2_1.py
```

服务器环境验证：

```bash
DEML_GPU_ID=1 python 实验/一键运行_suffix_v2_1.py --smoke-test
```
