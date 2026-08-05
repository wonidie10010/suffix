#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


MODEL_ID = "Qwen/Qwen2.5-1.5B"
EXPERIMENT_CONFIG = (
    "experiment_configs/"
    "l24_airport_medical_suffix_v2_0_no_cgmr.json"
)
METHOD_DIRECTORY = "suffix_reoptimization_v2.0"
REQUIRED_ARTIFACTS = (
    "resolved_config.json",
    "experiment.log",
    "reconstructions.jsonl",
)


def prepare_smoke_experiment(project_dir, runtime_dir):
    project_dir = Path(project_dir).resolve()
    smoke_root = Path(runtime_dir).resolve() / "smoke-test"
    run_root = smoke_root / "runs"
    copy_root = smoke_root / "smoke-copied-results"
    smoke_root.mkdir(parents=True, exist_ok=True)
    data_path = smoke_root / "smoke_data.json"
    config_path = smoke_root / "suffix_v2_0_smoke.json"
    official_config = (
        project_dir
        / "experiment_configs"
        / "l24_airport_medical_suffix_v2_0_no_cgmr.json"
    )
    if not official_config.is_file():
        raise FileNotFoundError(
            "official suffix v2.0 config is missing: {}".format(
                official_config
            )
        )
    data_path.write_text(
        json.dumps(["Test."], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config = {
        "include_configs": [str(official_config)],
        "datasets": [{
            "name": "suffix_v2_0_smoke",
            "path": str(data_path),
            "type": "local",
            "len": 1,
        }],
        "dataset_path": str(data_path),
        "dataset_type": "local",
        "dataset_len": 1,
        "epoch": 1,
        "suffix_v2_0_phase1_epoch": 1,
        "suffix_v2_0_phase2_epoch": 1,
        "suffix_v2_0_normal_embedding_top_k": 1,
        "suffix_v2_0_expanded_embedding_top_k": 1,
        "suffix_v2_0_ppl_top_k": 1,
        "suffix_v2_0_classifier_top_k": 1,
        "device_map": "single_gpu",
        "log_dir": str(run_root),
        "output_dir": "suffix_v2_0_one_click_smoke",
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "config": config_path,
        "data": data_path,
        "run_root": run_root,
        "copy_root": copy_root,
    }

EXIT_MODEL_FAILURE = 20
EXIT_EXPERIMENT_FAILURE = 30
EXIT_RESULT_FAILURE = 40
EXIT_UNEXPECTED_FAILURE = 50


def normalized_package_name(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def pinned_requirements(requirements_path):
    requirements = {}
    for raw_line in Path(requirements_path).read_text(
            encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if "==" not in line:
            raise ValueError("unsupported requirement: {}".format(line))
        name, expected_version = line.split("==", 1)
        requirements[normalized_package_name(name)] = (
            name.strip(),
            expected_version.strip(),
        )
    return requirements


def environment_mismatches(project_dir):
    mismatches = []
    expected_python = (3, 10, 20)
    actual_python = tuple(sys.version_info[:3])
    if actual_python != expected_python:
        mismatches.append(
            "python expected {} but found {}".format(
                ".".join(map(str, expected_python)),
                ".".join(map(str, actual_python)),
            )
        )
    requirements_path = Path(project_dir) / "requirements.txt"
    for _, (package_name, expected_version) in pinned_requirements(
            requirements_path).items():
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append("{} is missing".format(package_name))
            continue
        if actual_version != expected_version:
            mismatches.append(
                "{} expected {} but found {}".format(
                    package_name,
                    expected_version,
                    actual_version,
                )
            )
    return mismatches


class ProgressPrinter:
    def __init__(self, stream=None, initial=0):
        self.stream = stream or sys.stdout
        self.last_percent = -1
        self.update(initial)

    def update(self, percent):
        percent = max(0, min(100, int(percent)))
        if percent <= self.last_percent:
            return
        self.last_percent = percent
        self.stream.write("\r实验总进度：{}%".format(percent))
        self.stream.flush()


def append_log(log_file, text):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with Path(log_file).open("a", encoding="utf-8") as handle:
        handle.write("[{}] {}\n".format(timestamp, text))


def run_with_heartbeat(
        command,
        log_file,
        progress,
        start_percent,
        end_percent,
        cwd=None,
        env=None,
        interval=2.0):
    progress.update(start_percent)
    with Path(log_file).open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        current = int(start_percent)
        while process.poll() is None:
            time.sleep(interval)
            if current < int(end_percent) - 1:
                current += 1
                progress.update(current)
        return_code = process.wait()
    if return_code == 0:
        progress.update(end_percent)
    return return_code


def model_download_command():
    script = (
        "from huggingface_hub import snapshot_download\n"
        "try:\n"
        "    snapshot_download("
        "repo_id={!r}, revision='main', local_files_only=True)\n"
        "except Exception:\n"
        "    snapshot_download(repo_id={!r}, revision='main')\n"
    ).format(MODEL_ID, MODEL_ID)
    return [sys.executable, "-c", script]


def experiment_percent(state):
    phase = state.get("phase")
    total_samples = max(1, int(state.get("total_samples") or 1))
    if phase == "sample_optimization":
        sample_index = max(0, int(state.get("sample_index") or 0))
        total_steps = max(1, int(state.get("total_steps") or 1))
        completed_steps = max(
            0,
            min(total_steps, int(state.get("completed_steps") or 0)),
        )
        sample_fraction = 0.9 * completed_steps / total_steps
        experiment_fraction = (
            sample_index + sample_fraction
        ) / total_samples
    elif phase == "sample_completed":
        completed_samples = max(
            0,
            min(
                total_samples,
                int(state.get("completed_samples") or 0),
            ),
        )
        experiment_fraction = completed_samples / total_samples
    elif phase == "experiment_completed":
        experiment_fraction = 1.0
    else:
        sample_index = max(0, int(state.get("sample_index") or 0))
        experiment_fraction = sample_index / total_samples
    return min(99, max(50, int(50 + 49 * experiment_fraction)))


def read_progress_state(progress_file):
    try:
        with Path(progress_file).open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if isinstance(state, dict):
            return state
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return None


def run_experiment(
        project_dir,
        runtime_dir,
        log_file,
        progress,
        env,
        interval=2.0,
        experiment_config=EXPERIMENT_CONFIG,
        run_root=None):
    project_dir = Path(project_dir).resolve()
    method_root = (
        Path(run_root).resolve() / METHOD_DIRECTORY
        if run_root is not None
        else project_dir
        / "results"
        / "invert_timestamp_runs"
        / METHOD_DIRECTORY
    )
    before_runs = (
        {item.resolve() for item in method_root.iterdir() if item.is_dir()}
        if method_root.exists()
        else set()
    )
    command = [
        sys.executable,
        str(project_dir / "invert.py"),
        "--config",
        str(experiment_config),
    ]
    append_log(log_file, "starting experiment: {}".format(" ".join(command)))
    tracked_run_dir = None
    current_percent = 50
    progress.update(current_percent)
    with Path(log_file).open("ab", buffering=0) as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(project_dir),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            time.sleep(interval)
            if current_percent < 98:
                current_percent += 1
                progress.update(current_percent)
        return_code = process.wait()

    if return_code != 0:
        append_log(
            log_file,
            "experiment exited with code {}".format(return_code),
        )
        return return_code, tracked_run_dir

    progress.update(99)
    if tracked_run_dir is None:
        after_runs = (
            {item.resolve() for item in method_root.iterdir() if item.is_dir()}
            if method_root.exists()
            else set()
        )
        new_runs = sorted(after_runs - before_runs)
        if len(new_runs) == 1:
            tracked_run_dir = new_runs[0]
    return 0, tracked_run_dir


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def copy_and_verify_results(
        project_dir, result_root, run_dir, source_root=None):
    if run_dir is None:
        raise RuntimeError("experiment run directory was not reported")
    project_dir = Path(project_dir).resolve()
    source = Path(run_dir).resolve()
    expected_parent = (
        Path(source_root).resolve()
        if source_root is not None
        else project_dir / "results" / "invert_timestamp_runs"
    )
    expected_root = (expected_parent / METHOD_DIRECTORY).resolve()
    if not _is_relative_to(source, expected_root):
        raise RuntimeError("run directory is outside the expected result root")
    if not source.is_dir():
        raise RuntimeError("experiment run directory does not exist")
    for artifact in REQUIRED_ARTIFACTS:
        if not (source / artifact).is_file():
            raise RuntimeError("missing artifact: {}".format(artifact))

    destination_root = Path(result_root).resolve() / METHOD_DIRECTORY
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise RuntimeError(
            "destination result already exists: {}".format(destination)
        )

    try:
        shutil.copytree(source, destination)
        for artifact in REQUIRED_ARTIFACTS:
            source_hash = sha256_file(source / artifact)
            destination_hash = sha256_file(destination / artifact)
            if source_hash != destination_hash:
                raise RuntimeError(
                    "artifact hash mismatch: {}".format(artifact)
                )
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    return destination


def runtime_environment(runtime_dir):
    env = os.environ.copy()
    gpu_id = str(env.get("DEML_GPU_ID", "0")).strip()
    if not re.fullmatch(r"[0-9]+", gpu_id):
        raise ValueError("DEML_GPU_ID must be one non-negative GPU index")
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "HF_HOME": str(Path(runtime_dir) / "hf-cache"),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
    })
    return env


def run_bundle(args):
    project_dir = Path(args.project).resolve()
    runtime_dir = Path(args.runtime).resolve()
    result_root = Path(args.result_root).resolve()
    log_file = Path(args.log_file).resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressPrinter(stream=sys.stderr, initial=35)
    env = runtime_environment(runtime_dir)
    smoke_test = bool(getattr(args, "smoke_test", False))
    experiment_config = EXPERIMENT_CONFIG
    experiment_run_root = None
    copy_root = result_root
    if smoke_test:
        smoke = prepare_smoke_experiment(project_dir, runtime_dir)
        experiment_config = smoke["config"]
        experiment_run_root = smoke["run_root"]
        copy_root = smoke["copy_root"]
        append_log(args.log_file, "running explicit one-click smoke test")

    append_log(args.log_file, "checking/downloading model {}".format(MODEL_ID))
    model_status = run_with_heartbeat(
        model_download_command(),
        args.log_file,
        progress,
        35,
        50,
        cwd=str(project_dir),
        env=env,
    )
    if model_status != 0:
        append_log(args.log_file, "model preparation failed")
        return EXIT_MODEL_FAILURE

    offline_env = env.copy()
    offline_env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    experiment_status, run_dir = run_experiment(
        project_dir,
        runtime_dir,
        args.log_file,
        progress,
        offline_env,
        experiment_config=experiment_config,
        run_root=experiment_run_root,
    )
    if experiment_status != 0:
        return EXIT_EXPERIMENT_FAILURE

    try:
        destination = copy_and_verify_results(
            project_dir,
            copy_root,
            run_dir,
            source_root=experiment_run_root,
        )
    except Exception:
        append_log(args.log_file, traceback.format_exc())
        return EXIT_RESULT_FAILURE
    append_log(
        args.log_file,
        "verified result copy: {}".format(destination),
    )
    progress.update(100)
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check-env")
    check_parser.add_argument("--project", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--project", required=True)
    run_parser.add_argument("--runtime", required=True)
    run_parser.add_argument("--result-root", required=True)
    run_parser.add_argument("--log-file", required=True)
    run_parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "check-env":
        mismatches = environment_mismatches(args.project)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch, file=sys.stderr)
            return 1
        return 0
    try:
        return run_bundle(args)
    except KeyboardInterrupt:
        return 130
    except Exception:
        try:
            append_log(args.log_file, traceback.format_exc())
        except Exception:
            pass
        return EXIT_UNEXPECTED_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
