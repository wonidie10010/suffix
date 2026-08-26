#!/usr/bin/env python3
"""suffix v2.1.1 单卡正式实验的唯一顶层一键入口。

该入口保留原 v2.1 一键链路的文件名，并调用其 bootstrap；bootstrap
复用实验目录中的共享 runtime/Conda 环境与模型缓存，但只启动
suffix v2.1.1 的单进程单卡实验。
"""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
ENVIRONMENT_DIR = EXPERIMENT_DIR / "环境和实验"
INTERNAL_DIR = ENVIRONMENT_DIR / "内部文件"
BOOTSTRAP_SCRIPT = INTERNAL_DIR / "run_experiment_suffix_v2_1.sh"
SINGLE_GPU_RUNNER = INTERNAL_DIR / "runner_suffix_v2_1.py"
PROJECT_DIR = EXPERIMENT_DIR.parent
EXPERIMENT_CONFIG = (
    PROJECT_DIR
    / "experiment_configs"
    / "l24_airport_medical_suffix_v2_1_1_no_cgmr.json"
)


def validate_layout():
    required = (
        BOOTSTRAP_SCRIPT,
        SINGLE_GPU_RUNNER,
        PROJECT_DIR / "requirements.txt",
        PROJECT_DIR / "invert.py",
        EXPERIMENT_CONFIG,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("一键实验文件不完整：{}".format(", ".join(missing)))


def build_parser():
    parser = argparse.ArgumentParser(
        description="检查/安装共享环境并运行唯一的 suffix v2.1.1 单卡实验"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行一个极短样本的真实全流程测试，不启动正式 10 样本实验",
    )
    return parser


def main(argv=None, platform_name=None, machine=None, which=shutil.which,
         run=subprocess.run):
    options = build_parser().parse_args(argv)
    try:
        validate_layout()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    platform_name = platform_name or sys.platform
    machine = (machine or platform.machine()).lower()
    if platform_name != "linux" or machine not in ("x86_64", "amd64"):
        print(
            "正式一键实验要求 Linux x86_64、1 张可用 NVIDIA GPU；"
            "当前平台不满足，未启动实验。",
            file=sys.stderr,
        )
        return 3

    bash = which("bash")
    if not bash:
        print("未找到 bash，无法启动环境准备程序。", file=sys.stderr)
        return 3

    run_kwargs = {
        "cwd": str(ENVIRONMENT_DIR),
        "check": False,
    }
    if options.smoke_test:
        child_env = os.environ.copy()
        child_env["DEML_SMOKE_TEST"] = "1"
        run_kwargs["env"] = child_env
    completed = run([bash, str(BOOTSTRAP_SCRIPT)], **run_kwargs)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
