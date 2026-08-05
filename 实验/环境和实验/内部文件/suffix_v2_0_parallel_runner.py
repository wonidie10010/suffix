#!/usr/bin/env python3
"""Four-GPU sample-parallel runner for the single official suffix v2.0 run."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


METHOD = "suffix_reoptimization_v2.0"
VERSION = "v2.0"
METHOD_DIRECTORY = "suffix_reoptimization_v2.0"
EXECUTION_MODE = "sample_parallel_4gpu"
PHYSICAL_GPU_IDS = [0, 1, 2, 3]
SHARD_MAP = {
    "worker_0": [0, 4, 8],
    "worker_1": [1, 5, 9],
    "worker_2": [2, 6],
    "worker_3": [3, 7],
}
SAMPLE_MAP = {
    index: ("airport", index) for index in range(5)
}
SAMPLE_MAP.update({index: ("medical", index - 5) for index in range(5, 10)})
CORE_ARTIFACTS = (
    "resolved_config.json",
    "experiment.log",
    "reconstructions.jsonl",
    "run_manifest.json",
)
CONFIG_RELATIVE_PATH = Path(
    "experiment_configs/l24_airport_medical_suffix_v2_0_no_cgmr.json"
)
EXIT_PREFLIGHT_FAILURE = 10
EXIT_MODEL_FAILURE = 20
EXIT_WORKER_FAILURE = 30
EXIT_MERGE_FAILURE = 40
EXIT_COPY_FAILURE = 50


class ParallelRunError(RuntimeError):
    pass


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
            name.strip(), expected_version.strip()
        )
    return requirements


def environment_mismatches(project_dir):
    """Return exact Python/package mismatches for the v2.0 bundle."""
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
                    package_name, expected_version, actual_version
                )
            )
    return mismatches


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ParallelRunError("recursive config include: {}".format(path))
    seen.add(path)
    with path.open("r", encoding="utf-8") as source:
        current = json.load(source)
    merged = {}
    includes = current.pop("include_configs", [])
    includes = [includes] if isinstance(includes, str) else includes
    for include in includes:
        include_path = Path(include)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged.update(load_config(include_path, seen))
    merged.update(current)
    seen.remove(path)
    return merged


def validate_static_shards():
    if list(SHARD_MAP) != ["worker_0", "worker_1", "worker_2", "worker_3"]:
        raise ParallelRunError("worker ids are not canonical")
    flattened = [value for shard in SHARD_MAP.values() for value in shard]
    if len(flattened) != len(set(flattened)):
        raise ParallelRunError("shard map contains duplicate global indices")
    if sorted(flattened) != list(range(10)):
        raise ParallelRunError("shard map omits or adds global indices")
    if len(PHYSICAL_GPU_IDS) != len(set(PHYSICAL_GPU_IDS)):
        raise ParallelRunError("physical GPU ids contain duplicates")


def validate_experiment_config(config):
    validate_static_shards()
    if str(config.get("suffix_version")).lower() != "v2.0":
        raise ParallelRunError("selector must be v2.0")
    legacy_selector = config.get("suffix_reoptimization_version")
    if legacy_selector is not None and str(legacy_selector).lower() != "v2.0":
        raise ParallelRunError("legacy suffix selector must also be v2.0")
    if config.get("suffix_reoptimization_v2_0") is not True:
        raise ParallelRunError("suffix v2.0 must be enabled")
    if config.get("suffix_v2_0_classifier_enabled") is not False:
        raise ParallelRunError("classifier_enabled must be false")
    if int(config.get("num_invert_layers", -1)) != 24:
        raise ParallelRunError("num_invert_layers must be 24")
    datasets = config.get("datasets") or []
    if [item.get("name") for item in datasets] != ["airport", "medical"]:
        raise ParallelRunError("dataset order must be airport then medical")
    if [item.get("len") for item in datasets] != [5, 5]:
        raise ParallelRunError("airport and medical must each contain five samples")
    if config.get("cgmr_version") != "none":
        raise ParallelRunError("CGMR selector must be none")
    for key in ("cgmr_v1_0", "cgmr_v1_1", "cgmr_v1_2"):
        if config.get(key) is not False:
            raise ParallelRunError("{} must be disabled".format(key))
    old_suffix_flags = (
        "suffix_reoptimization_v1_0", "suffix_reoptimization_v1_1",
        "suffix_reoptimization_v1_2", "suffix_reoptimization_v1_2_1",
        "suffix_reoptimization_v1_2_2", "suffix_reoptimization_v1_2_3",
        "suffix_v1_2_3",
        "suffix_reoptimization_v1_3", "suffix_reoptimization_v1_3_1",
        "suffix_reoptimization_v1_4", "suffix_reoptimization_v1_4_1",
    )
    for key in old_suffix_flags:
        if config.get(key) is not False:
            raise ParallelRunError("{} must be disabled".format(key))
    if config.get("local_embedding_repair", False) is not False:
        raise ParallelRunError("legacy local repair must be disabled")
    if config.get("device_map") != "single_gpu":
        raise ParallelRunError("worker device_map must be single_gpu")
    return config


def probe_physical_gpus(python_executable=sys.executable, run=subprocess.run):
    query = run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    if query.returncode != 0:
        raise ParallelRunError("nvidia-smi GPU inventory failed")
    try:
        available = [int(line.strip()) for line in query.stdout.splitlines() if line.strip()]
    except ValueError as error:
        raise ParallelRunError("nvidia-smi returned invalid GPU indices") from error
    if any(gpu_id not in available for gpu_id in PHYSICAL_GPU_IDS):
        raise ParallelRunError("physical CUDA GPUs 0,1,2,3 are required")
    probe_code = (
        "import torch; raise SystemExit(0 if torch.cuda.is_available() "
        "and torch.cuda.device_count()==1 else 1)"
    )
    for gpu_id in PHYSICAL_GPU_IDS:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        result = run(
            [str(python_executable), "-c", probe_code],
            env=environment, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise ParallelRunError("physical GPU {} is not usable".format(gpu_id))
    return list(PHYSICAL_GPU_IDS)


def preflight(project_dir, repo_root, python_executable, timestamp,
              gpu_probe=probe_physical_gpus):
    project_dir = Path(project_dir).resolve()
    repo_root = Path(repo_root).resolve()
    config_path = project_dir / CONFIG_RELATIVE_PATH
    config = validate_experiment_config(load_config(config_path))
    gpu_ids = gpu_probe(python_executable)
    if gpu_ids != PHYSICAL_GPU_IDS:
        raise ParallelRunError("GPU probe must return physical ids [0,1,2,3]")
    parent_root = repo_root / "results" / "invert_timestamp_runs" / METHOD_DIRECTORY
    parent_run = parent_root / timestamp
    expected_parent = (
        repo_root / "results" / "invert_timestamp_runs"
        / "suffix_reoptimization_v2.0" / timestamp
    )
    if parent_run != expected_parent:
        raise ParallelRunError("official parent path is not the v2.0 method path")
    if parent_run.exists():
        raise ParallelRunError("official parent timestamp already exists")
    temporary_root = repo_root / "outputs" / "suffix_v2_0_parallel" / timestamp
    if temporary_root.exists():
        raise ParallelRunError("worker temporary timestamp already exists")
    result_dir = repo_root / "实验" / "结果" / ("suffix_v2_0_" + timestamp)
    staging_dir = result_dir.with_name("." + result_dir.name + ".tmp")
    if result_dir.exists() or staging_dir.exists():
        raise ParallelRunError("collected result timestamp already exists")
    return {
        "project_dir": project_dir,
        "repo_root": repo_root,
        "config": config,
        "config_path": config_path,
        "parent_run": parent_run,
        "temporary_root": temporary_root,
        "result_dir": result_dir,
        "staging_dir": staging_dir,
    }


def worker_spec(worker_id, physical_gpu_id, timestamp):
    indices = list(SHARD_MAP["worker_{}".format(worker_id)])
    return {
        "worker_id": int(worker_id),
        "physical_gpu_id": int(physical_gpu_id),
        "local_device": "cuda:0",
        "parent_timestamp": str(timestamp),
        "assigned_global_indices": indices,
        "assigned_samples": [
            {
                "global_index": index,
                "dataset_name": SAMPLE_MAP[index][0],
                "dataset_sample_index": SAMPLE_MAP[index][1],
            }
            for index in indices
        ],
    }


def preflight_summary(state, python_executable, project_dir, timestamp):
    del project_dir, timestamp
    commands = {}
    for worker_id in range(4):
        worker_dir = state["temporary_root"] / "worker_{}".format(worker_id)
        commands["worker_{}".format(worker_id)] = build_worker_command(
            python_executable,
            state["project_dir"],
            state["config_path"],
            worker_dir / "worker_spec.json",
            worker_dir,
        )
    return {
        "selected_method": METHOD_DIRECTORY,
        "dataset_count": 10,
        "num_invert_layers": 24,
        "classifier_enabled": False,
        "cgmr_enabled": False,
        "local_repair_enabled": False,
        "other_suffix_versions_enabled": False,
        "physical_gpu_ids": PHYSICAL_GPU_IDS,
        "shard_map": SHARD_MAP,
        "parent_run": str(state["parent_run"]),
        "worker_commands": commands,
    }


def print_run_confirmation(stream):
    print("selected method = suffix v2.0", file=stream)
    print("dataset count = 10", file=stream)
    print("num_invert_layers = 24", file=stream)
    print("CGMR disabled", file=stream)
    print("local repair disabled", file=stream)
    print("other suffix versions disabled", file=stream)


def build_worker_command(python_executable, project_dir, config_path,
                         spec_path, worker_dir):
    return [
        str(python_executable), str(Path(project_dir) / "invert.py"),
        "--config", str(config_path),
        "--parallel-worker-spec", str(spec_path),
        "--worker-output-dir", str(worker_dir),
        "--device-map", "single_gpu",
    ]


def prepare_model_cache(python_executable, project_dir, environment,
                        run=subprocess.run):
    code = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id='Qwen/Qwen2.5-1.5B')"
    )
    result = run(
        [str(python_executable), "-c", code], cwd=str(project_dir),
        env=environment, check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ParallelRunError("model cache preparation failed")


def launch_workers(python_executable, project_dir, config_path,
                   temporary_root, timestamp, popen=subprocess.Popen):
    processes = []
    for worker_id, physical_gpu_id in enumerate(PHYSICAL_GPU_IDS):
        worker_dir = Path(temporary_root) / "worker_{}".format(worker_id)
        worker_dir.mkdir(parents=True, exist_ok=False)
        spec = worker_spec(worker_id, physical_gpu_id, timestamp)
        spec_path = worker_dir / "worker_spec.json"
        atomic_json(spec_path, spec)
        stdout_path = worker_dir / "stdout.log"
        stderr_path = worker_dir / "stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": str(physical_gpu_id),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        command = build_worker_command(
            python_executable, project_dir, config_path, spec_path, worker_dir
        )
        try:
            process = popen(
                command, cwd=str(project_dir), env=environment,
                stdout=stdout_handle, stderr=stderr_handle,
            )
        except Exception as error:
            stdout_handle.close()
            stderr_handle.close()
            exit_codes = wait_workers(processes) if processes else {}
            failure = ParallelRunError(
                "worker {} launch failed: {}".format(worker_id, error)
            )
            failure.worker_exit_codes = exit_codes
            raise failure from error
        processes.append({
            "worker_id": worker_id,
            "process": process,
            "stdout": stdout_handle,
            "stderr": stderr_handle,
            "worker_dir": worker_dir,
            "environment": environment,
            "command": command,
        })
    return processes


def wait_workers(processes):
    exit_codes = {}
    for item in processes:
        try:
            exit_codes[str(item["worker_id"])] = int(
                item["process"].wait()
            )
        except Exception:
            exit_codes[str(item["worker_id"])] = -1
        finally:
            item["stdout"].close()
            item["stderr"].close()
    return exit_codes


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ParallelRunError(
                    "invalid JSONL at {}:{}".format(path, line_number)
                ) from error
    return records


def collect_worker_logs(temporary_root, parent_run):
    destination = Path(parent_run) / "worker_logs"
    destination.mkdir(parents=True, exist_ok=True)
    for worker_id in range(4):
        worker_dir = Path(temporary_root) / "worker_{}".format(worker_id)
        if not worker_dir.is_dir():
            continue
        stdout = (worker_dir / "stdout.log").read_text(
            encoding="utf-8", errors="replace"
        )
        stderr = (worker_dir / "stderr.log").read_text(
            encoding="utf-8", errors="replace"
        )
        (destination / "worker_{}.log".format(worker_id)).write_text(
            "===== STDOUT =====\n{}\n===== STDERR =====\n{}".format(
                stdout, stderr
            ),
            encoding="utf-8",
        )


def available_worker_record_counts(temporary_root):
    counts = {}
    for worker_id in range(4):
        status_path = (
            Path(temporary_root) / "worker_{}".format(worker_id)
            / "worker_status.json"
        )
        if not status_path.is_file():
            continue
        try:
            counts[str(worker_id)] = int(
                load_json(status_path).get("record_count", -1)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            counts[str(worker_id)] = -1
    return counts


def validate_worker_outputs(temporary_root, exit_codes):
    statuses = {}
    records = []
    fingerprints = []
    for worker_id in range(4):
        worker_dir = Path(temporary_root) / "worker_{}".format(worker_id)
        status_path = worker_dir / "worker_status.json"
        shard_path = worker_dir / "shard_reconstructions.jsonl"
        if not status_path.is_file() or not shard_path.is_file():
            raise ParallelRunError("worker {} is missing status or shard".format(worker_id))
        status = load_json(status_path)
        shard = load_jsonl(shard_path)
        expected = SHARD_MAP["worker_{}".format(worker_id)]
        if exit_codes[str(worker_id)] != 0 or status.get("exit_code") != 0:
            raise ParallelRunError("worker {} exited unsuccessfully".format(worker_id))
        if not status.get("success"):
            raise ParallelRunError("worker {} reported failure".format(worker_id))
        if status.get("assigned_global_indices") != expected:
            raise ParallelRunError("worker {} assignment changed".format(worker_id))
        if status.get("completed_global_indices") != expected:
            raise ParallelRunError("worker {} completion is incomplete".format(worker_id))
        if int(status.get("record_count", -1)) != len(expected) or len(shard) != len(expected):
            raise ParallelRunError("worker {} record count mismatch".format(worker_id))
        if status.get("local_device") != "cuda:0" or status.get("device_map") != {"": 0}:
            raise ParallelRunError("worker {} device isolation changed".format(worker_id))
        fingerprint = status.get("runtime_config_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ParallelRunError("worker {} config fingerprint is missing".format(worker_id))
        model_metadata = status.get("model_metadata") or {}
        if int(model_metadata.get("num_invert_layers", -1)) != 24:
            raise ParallelRunError("worker {} model metadata is invalid".format(worker_id))
        fingerprints.append(fingerprint)
        statuses[str(worker_id)] = status
        records.extend(shard)
    if len(set(fingerprints)) != 1:
        raise ParallelRunError("worker runtime config fingerprints differ")
    return statuses, records


def validate_merged_records(records):
    if len(records) != 10:
        raise ParallelRunError("merged record count must be ten")
    indices = [record.get("global_index") for record in records]
    if len(indices) != len(set(indices)) or sorted(indices) != list(range(10)):
        raise ParallelRunError("merged global indices are duplicated or incomplete")
    for record in records:
        global_index = int(record["global_index"])
        expected_dataset, expected_sample = SAMPLE_MAP[global_index]
        if record.get("dataset_name") != expected_dataset:
            raise ParallelRunError("global index dataset mapping mismatch")
        if int(record.get("dataset_sample_index", -1)) != expected_sample:
            raise ParallelRunError("global index sample mapping mismatch")
        expected_worker = next(
            worker_id for worker_id in range(4)
            if global_index in SHARD_MAP["worker_{}".format(worker_id)]
        )
        if int(record.get("assigned_worker_id", -1)) != expected_worker:
            raise ParallelRunError("record worker assignment mismatch")
        if int(record.get("assigned_physical_gpu_id", -1)) != expected_worker:
            raise ParallelRunError("record physical GPU assignment mismatch")
        if record.get("selected_advanced_method") != METHOD_DIRECTORY:
            raise ParallelRunError("record did not select suffix v2.0")
        if record.get("method") != METHOD_DIRECTORY or record.get("version") != VERSION:
            raise ParallelRunError("record method/version mismatch")
        if record.get("classifier_enabled") is not False:
            raise ParallelRunError("classifier unexpectedly enabled")
        if record.get("classifier_provider_available") is not False:
            raise ParallelRunError("classifier provider unexpectedly available")
        if int(record.get("classifier_candidate_count", -1)) != 0:
            raise ParallelRunError("classifier candidates unexpectedly generated")
        if int(record.get("num_invert_layers", -1)) != 24:
            raise ParallelRunError("record inversion layer mismatch")
        if record.get("selected_candidate_reranking_method") != "none":
            raise ParallelRunError("CGMR unexpectedly ran")
        if not record.get("accepted") or record.get("rollback") or record.get("fatal_failure"):
            raise ParallelRunError("record is not a successful v2.0 sample")
        for key, value in record.items():
            if key.startswith("suffix_reoptimization_v1_") and isinstance(value, dict):
                if value.get("enabled") and not value.get("skipped", False):
                    raise ParallelRunError("legacy suffix unexpectedly ran")
        local = record.get("local_embedding_repair_result") or {}
        if local.get("enabled") and not local.get("skipped", False):
            raise ParallelRunError("legacy local repair unexpectedly ran")
    ordered = sorted(
        records,
        key=lambda record: (
            0 if record["dataset_name"] == "airport" else 1,
            int(record["dataset_sample_index"]),
        ),
    )
    expected_order = [
        ("airport", index) for index in range(5)
    ] + [("medical", index) for index in range(5)]
    actual_order = [
        (record["dataset_name"], int(record["dataset_sample_index"]))
        for record in ordered
    ]
    if actual_order != expected_order:
        raise ParallelRunError("merged record order is not canonical")
    return ordered


def atomic_write_jsonl(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    reparsed = load_jsonl(temporary)
    validate_merged_records(reparsed)
    os.replace(temporary, path)


def load_experiment_outputs(project_dir):
    path = Path(project_dir) / "experiment_outputs.py"
    spec = importlib.util.spec_from_file_location(
        "suffix_v2_parallel_experiment_outputs", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_resolved_config(statuses, parent_run, timestamp):
    configs = [statuses[str(worker_id)].get("resolved_config") for worker_id in range(4)]
    if any(config is None for config in configs):
        raise ParallelRunError("worker runtime metadata is missing resolved config")
    normalized = []
    for config in configs:
        clone = json.loads(json.dumps(config, ensure_ascii=False))
        clone.get("artifacts", {}).clear()
        clone.get("outputs", {}).clear()
        clone.get("run", {}).pop("timestamp", None)
        normalized.append(clone)
    if any(value != normalized[0] for value in normalized[1:]):
        raise ParallelRunError("worker resolved configurations differ")
    resolved = configs[0]
    resolved["run"]["timestamp"] = timestamp
    resolved["run"]["execution_mode"] = EXECUTION_MODE
    resolved["dataset"]["len_setting"] = 10
    resolved["parallel_execution"] = {
        "enabled": True,
        "mode": "sample_parallel",
        "worker_count": 4,
        "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
        "global_sample_count": 10,
        "dataset_order": ["airport", "medical"],
        "shard_map": {key: list(value) for key, value in SHARD_MAP.items()},
        "future_multi_gpu_mode": "tensor_parallel",
        "future_multi_gpu_enabled": False,
    }
    resolved["classifier_enabled"] = False
    resolved["classifier_provider_available"] = False
    resolved["classifier_candidate_count"] = 0
    resolved["sample_counts"] = {"airport": 5, "medical": 5, "total": 10}
    resolved["method_exclusivity"] = {
        "suffix_v2_0_enabled": True,
        "legacy_suffix_enabled": False,
        "cgmr_enabled": False,
        "legacy_local_repair_enabled": False,
    }
    resolved["artifacts"] = {
        "run_dir": str(parent_run),
        "experiment_log": str(Path(parent_run) / "experiment.log"),
        "reconstructions": str(Path(parent_run) / "reconstructions.jsonl"),
        "resolved_config": str(Path(parent_run) / "resolved_config.json"),
        "run_manifest": str(Path(parent_run) / "run_manifest.json"),
    }
    resolved["outputs"] = {
        "run_dir": str(parent_run),
        "experiment_log": str(Path(parent_run) / "experiment.log"),
        "reconstructions": str(Path(parent_run) / "reconstructions.jsonl"),
        "resolved_config": str(Path(parent_run) / "resolved_config.json"),
        "run_manifest": str(Path(parent_run) / "run_manifest.json"),
    }
    return resolved


def base_manifest(parent_run, temporary_root, timestamp):
    return {
        "method": METHOD,
        "version": VERSION,
        "execution_mode": EXECUTION_MODE,
        "parent_run_path": str(parent_run),
        "collection_timestamp": timestamp,
        "worker_count": 4,
        "physical_gpu_ids": list(PHYSICAL_GPU_IDS),
        "shard_map": {key: list(value) for key, value in SHARD_MAP.items()},
        "airport_sample_count": 5,
        "medical_sample_count": 5,
        "total_sample_count": 10,
        "num_invert_layers": 24,
        "classifier_enabled": False,
        "classifier_provider_available": False,
        "temporary_worker_path": str(temporary_root),
        "worker_started_at": None,
        "worker_finished_at": None,
        "worker_exit_codes": {},
        "worker_record_counts": {},
        "merge_started_at": None,
        "merge_finished_at": None,
        "merge_success": False,
        "output_validation_success": False,
        "overall_success": False,
        "failure_reason": None,
    }


def copy_and_verify(parent_run, staging_dir, result_dir):
    shutil.copytree(parent_run, staging_dir)
    source_hashes = {
        name: sha256_file(Path(parent_run) / name) for name in CORE_ARTIFACTS
    }
    copied_hashes = {
        name: sha256_file(Path(staging_dir) / name) for name in CORE_ARTIFACTS
    }
    if source_hashes != copied_hashes:
        raise ParallelRunError("core artifact hash mismatch after copy")
    records = load_jsonl(Path(staging_dir) / "reconstructions.jsonl")
    validate_merged_records(records)
    for worker_id in range(4):
        if not (Path(staging_dir) / "worker_logs" / "worker_{}.log".format(worker_id)).is_file():
            raise ParallelRunError("copied worker log is missing")
    os.replace(staging_dir, result_dir)
    return result_dir


def validate_official_artifacts(parent_run):
    parent_run = Path(parent_run)
    required = [parent_run / name for name in CORE_ARTIFACTS[:-1]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ParallelRunError(
            "official artifact is missing: {}".format(", ".join(missing))
        )
    records = load_jsonl(parent_run / "reconstructions.jsonl")
    ordered = validate_merged_records(records)
    if records != ordered:
        raise ParallelRunError("official reconstruction order changed")
    resolved = load_json(parent_run / "resolved_config.json")
    if (resolved.get("advanced_method") or {}).get("name") != METHOD_DIRECTORY:
        raise ParallelRunError("official resolved config did not select v2.0")
    v2_config = (resolved.get("advanced_methods") or {}).get(
        "suffix_reoptimization_v2_0"
    ) or {}
    if v2_config.get("enabled") is not True:
        raise ParallelRunError("official resolved config did not enable v2.0")
    if (
        v2_config.get("classifier_enabled") is not False
        or v2_config.get("classifier_provider_available") is not False
        or int(v2_config.get("classifier_candidate_count", -1)) != 0
    ):
        raise ParallelRunError("official resolved classifier metadata is invalid")
    if int((resolved.get("model") or {}).get("num_invert_layers", -1)) != 24:
        raise ParallelRunError("official resolved inversion layer mismatch")
    if (resolved.get("candidate_reranking_method") or {}).get("name") != "none":
        raise ParallelRunError("official resolved CGMR selector is not none")
    if resolved.get("method_exclusivity") != {
        "suffix_v2_0_enabled": True,
        "legacy_suffix_enabled": False,
        "cgmr_enabled": False,
        "legacy_local_repair_enabled": False,
    }:
        raise ParallelRunError("official method exclusivity metadata is invalid")
    if (resolved.get("outputs") or {}).get("run_dir") != str(parent_run):
        raise ParallelRunError("official resolved output path leaked a worker path")
    dataset_specs = resolved.get("datasets") or []
    if (
        [item.get("name") for item in dataset_specs] != ["airport", "medical"]
        or [int(item.get("len", -1)) for item in dataset_specs] != [5, 5]
    ):
        raise ParallelRunError("official resolved dataset mapping is invalid")
    parallel = resolved.get("parallel_execution") or {}
    if (
        parallel.get("enabled") is not True
        or parallel.get("mode") != "sample_parallel"
        or parallel.get("worker_count") != 4
        or parallel.get("physical_gpu_ids") != PHYSICAL_GPU_IDS
        or parallel.get("shard_map") != SHARD_MAP
    ):
        raise ParallelRunError("official parallel_execution metadata is invalid")
    log_text = (parent_run / "experiment.log").read_text(
        encoding="utf-8", errors="strict"
    )
    if log_text.count("===== overall average accuracy =====") != 1:
        raise ParallelRunError("experiment.log must contain one overall average")
    if log_text.count("===== dataset airport sample ") != 5:
        raise ParallelRunError("experiment.log airport sample count mismatch")
    if log_text.count("===== dataset medical sample ") != 5:
        raise ParallelRunError("experiment.log medical sample count mismatch")
    return records


def run_parallel(project_dir, repo_root, python_executable, timestamp=None,
                 gpu_probe=probe_physical_gpus, popen=subprocess.Popen,
                 prepare_model=True,
                 experiment_outputs_loader=load_experiment_outputs,
                 confirmation_stream=None):
    timestamp = timestamp or default_timestamp()
    project_dir = Path(project_dir).resolve()
    repo_root = Path(repo_root).resolve()
    state = preflight(
        project_dir, repo_root, python_executable, timestamp, gpu_probe=gpu_probe
    )
    if confirmation_stream is not None:
        print_run_confirmation(confirmation_stream)
    parent_run = state["parent_run"]
    temporary_root = state["temporary_root"]
    parent_run.mkdir(parents=True, exist_ok=False)
    temporary_root.mkdir(parents=True, exist_ok=False)
    manifest = base_manifest(parent_run, temporary_root, timestamp)
    manifest_path = parent_run / "run_manifest.json"
    atomic_json(manifest_path, manifest)
    try:
        if prepare_model:
            prepare_model_cache(
                python_executable, project_dir, os.environ.copy()
            )
        manifest["worker_started_at"] = utc_now()
        processes = launch_workers(
            python_executable, project_dir, state["config_path"],
            temporary_root, timestamp, popen=popen,
        )
        exit_codes = wait_workers(processes)
        manifest["worker_finished_at"] = utc_now()
        manifest["worker_exit_codes"] = exit_codes
        collect_worker_logs(temporary_root, parent_run)
        manifest["worker_record_counts"] = available_worker_record_counts(
            temporary_root
        )
        statuses, records = validate_worker_outputs(temporary_root, exit_codes)
        manifest["worker_record_counts"] = {
            worker_id: int(status["record_count"])
            for worker_id, status in statuses.items()
        }
        manifest["merge_started_at"] = utc_now()
        ordered = validate_merged_records(records)
        atomic_write_jsonl(parent_run / "reconstructions.jsonl", ordered)
        resolved = parent_resolved_config(
            statuses, parent_run, timestamp
        )
        atomic_json(parent_run / "resolved_config.json", resolved)
        outputs = experiment_outputs_loader(project_dir)
        outputs.rebuild_experiment_log(
            parent_run / "experiment.log", ordered
        )
        validate_official_artifacts(parent_run)
        manifest["merge_finished_at"] = utc_now()
        manifest["merge_success"] = True
        manifest["output_validation_success"] = True
        manifest["overall_success"] = True
        atomic_json(manifest_path, manifest)
        copy_and_verify(
            parent_run, state["staging_dir"], state["result_dir"]
        )
        shutil.rmtree(temporary_root)
        return parent_run, state["result_dir"]
    except Exception as error:
        if manifest.get("worker_finished_at") is None and hasattr(
                error, "worker_exit_codes"):
            manifest["worker_finished_at"] = utc_now()
            manifest["worker_exit_codes"] = error.worker_exit_codes
            collect_worker_logs(temporary_root, parent_run)
            manifest["worker_record_counts"] = available_worker_record_counts(
                temporary_root
            )
        manifest["failure_reason"] = "{}: {}".format(type(error).__name__, error)
        manifest["overall_success"] = False
        atomic_json(manifest_path, manifest)
        if state["staging_dir"].exists():
            shutil.rmtree(state["staging_dir"])
        raise


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("run", "dry-run", "mock-dry-run", "check-env")
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timestamp", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "check-env":
        mismatches = environment_mismatches(args.project)
        for mismatch in mismatches:
            print(mismatch, file=sys.stderr)
        return 1 if mismatches else 0
    if not args.repo_root:
        print("--repo-root is required for {}".format(args.command), file=sys.stderr)
        return EXIT_PREFLIGHT_FAILURE
    timestamp = args.timestamp or default_timestamp()
    try:
        if args.command in ("dry-run", "mock-dry-run"):
            gpu_probe = (
                (lambda python: list(PHYSICAL_GPU_IDS))
                if args.command == "mock-dry-run" else probe_physical_gpus
            )
            state = preflight(
                args.project, args.repo_root, args.python, timestamp,
                gpu_probe=gpu_probe,
            )
            print(json.dumps(
                preflight_summary(
                    state, args.python, args.project, timestamp
                ),
                indent=2,
                ensure_ascii=False,
            ))
            return 0
        parent_run, copied = run_parallel(
            args.project, args.repo_root, args.python, timestamp=timestamp,
            confirmation_stream=sys.stderr,
        )
        repo_root = Path(args.repo_root).resolve()
        print("Experiment completed")
        print("Original result:")
        print(parent_run.relative_to(repo_root).as_posix() + "/")
        print("Copied result:")
        print(copied.relative_to(repo_root).as_posix() + "/")
        return 0
    except ParallelRunError as error:
        print(str(error), file=sys.stderr)
        return EXIT_PREFLIGHT_FAILURE if args.command != "run" else EXIT_WORKER_FAILURE
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return EXIT_WORKER_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
