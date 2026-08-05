#!/usr/bin/env python3
"""suffix v2.0 四卡正式实验的唯一顶层一键入口。

该入口调用内部 bootstrap。bootstrap 会先检查 Linux/磁盘/Conda/Python/
锁定依赖/CUDA；环境缺失时会在本目录的 .runtime 下安装隔离环境，随后只启动
suffix v2.0 的四卡样本并行实验。
"""

import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys


BUNDLE_DIR = Path(__file__).resolve().parent
INTERNAL_DIR = BUNDLE_DIR / "内部文件"
BOOTSTRAP_SCRIPT = INTERNAL_DIR / "run_experiment.sh"
V2_RUNNER = INTERNAL_DIR / "suffix_v2_0_parallel_runner.py"
PROJECT_DIR = BUNDLE_DIR.parent.parent


def validate_layout():
    required = (
        BOOTSTRAP_SCRIPT,
        V2_RUNNER,
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / "invert.py",
        PROJECT_DIR
        / "experiment_configs"
        / "l24_airport_medical_suffix_v2_0_no_cgmr.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("一键实验文件不完整：{}".format(", ".join(missing)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="检查/安装隔离环境并运行唯一的 suffix v2.0 四卡实验"
    )
    return parser


def main(argv=None, platform_name=None, machine=None, which=shutil.which,
         run=subprocess.run):
    build_parser().parse_args(argv)
    try:
        validate_layout()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    platform_name = platform_name or sys.platform
    machine = (machine or platform.machine()).lower()
    if platform_name != "linux" or machine not in ("x86_64", "amd64"):
        print(
            "正式一键实验要求 Linux x86_64、4 张可用 NVIDIA GPU；"
            "当前平台不满足，未启动实验。",
            file=sys.stderr,
        )
        return 3

    bash = which("bash")
    if not bash:
        print("未找到 bash，无法启动环境准备程序。", file=sys.stderr)
        return 3

    completed = run(
        [bash, str(BOOTSTRAP_SCRIPT)],
        cwd=str(BUNDLE_DIR),
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
