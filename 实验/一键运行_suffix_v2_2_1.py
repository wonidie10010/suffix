#!/usr/bin/env python3
"""suffix v2.2.1 one-click entry point.

Run this file from the project root with the shared Python environment on the
experiment machine.  ``--smoke-test`` executes one tiny real end-to-end run;
without it the runner executes baseline followed by baseline+R.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent
INTERNAL_RUNNER = (
    EXPERIMENT_DIR
    / "环境和实验"
    / "内部文件"
    / "runner_suffix_v2_2_1.py"
)
DEFAULT_RUNTIME = EXPERIMENT_DIR / "环境和实验" / ".runtime"
DEFAULT_RESULT_ROOT = EXPERIMENT_DIR / "结果" / "suffix_v2.2.1_bundle"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run suffix v2.2.1 smoke or two-run formal experiment"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one short real end-to-end sample",
    )
    parser.add_argument("--runtime", default=str(DEFAULT_RUNTIME))
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--python", dest="python_executable", default=None)
    return parser


def validate_layout():
    required = (
        INTERNAL_RUNNER,
        PROJECT_DIR / "invert.py",
        PROJECT_DIR / "experiment_configs" / "l24_deml3x4_baseline.json",
        PROJECT_DIR / "experiment_configs" / "l24_deml3x4_suffix_v2_2_1.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("one-click bundle is incomplete: {}".format(", ".join(missing)))


def main(argv=None, run=subprocess.run):
    options = build_parser().parse_args(argv)
    try:
        validate_layout()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    python_executable = (
        options.python_executable
        or os.environ.get("DEML_SUFFIX_PYTHON")
        or sys.executable
    )
    command = [
        str(python_executable),
        str(INTERNAL_RUNNER),
        "smoke" if options.smoke_test else "formal",
        "--project",
        str(PROJECT_DIR),
        "--runtime",
        str(Path(options.runtime).resolve()),
        "--result-root",
        str(Path(options.result_root).resolve()),
        "--log-file",
        str(
            Path(options.result_root).resolve()
            / "logs"
            / "suffix_v2.2.1_runner.log"
        ),
        "--python",
        str(python_executable),
    ]
    completed = run(command, cwd=str(PROJECT_DIR), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
