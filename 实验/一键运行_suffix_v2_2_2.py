#!/usr/bin/env python3
"""v2.2.2 paired CP-off/on launcher. Use --dry-run during implementation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
INTERNAL_RUNNER = PROJECT_DIR / "实验/环境和实验/内部文件/runner_suffix_v2_2_2.py"


def main(argv=None, run=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--python", default=os.environ.get("DEML_SUFFIX_PYTHON", sys.executable))
    parser.add_argument("--runtime", default=str(PROJECT_DIR / "实验/环境和实验/.runtime"))
    parser.add_argument("--result-root", default=str(PROJECT_DIR / "实验/结果/suffix_v2.2.2_bundle"))
    parser.add_argument("--model-path", default=os.environ.get("DEML_MODEL_PATH"))
    options = parser.parse_args(argv)
    if not INTERNAL_RUNNER.is_file():
        parser.error("missing internal runner: " + str(INTERNAL_RUNNER))
    command = [options.python, str(INTERNAL_RUNNER),
               "dry-run" if options.dry_run else "smoke" if options.smoke_test else "formal",
               "--project", str(PROJECT_DIR), "--python", options.python,
               "--runtime", str(Path(options.runtime).resolve()),
               "--result-root", str(Path(options.result_root).resolve())]
    if options.model_path:
        command.extend(["--model-path", options.model_path])
    return int(run(command, cwd=str(PROJECT_DIR), check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
