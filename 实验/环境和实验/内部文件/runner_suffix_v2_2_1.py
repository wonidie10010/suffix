#!/usr/bin/env python3
"""Run and verify the suffix v2.2.1 two-run experiment bundle.

The runner keeps the official JSON configs immutable.  It creates effective
configs in the bundle directory, invokes ``invert.py`` with the shared
runtime Python, and validates the three canonical artifacts produced by the
main pipeline.  No ground-truth value is read until the suffix sidecar has
finished; the GT-backed accuracy fields are used only for the final offline
summary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


MODEL_ID = "Qwen/Qwen2.5-1.5B"
MODEL_PATH_CANDIDATES = (
    "/mnt/my_disk/tch/models/Qwen2.5-1.5B",
    "models/Qwen2.5-1.5B",
)
FORMAL_RUNS = (
    {
        "label": "baseline",
        "config": "experiment_configs/l24_deml3x4_baseline.json",
        "method": "frozen_original_baseline",
        "expected_advanced": "frozen_original_baseline",
    },
    {
        "label": "baseline_plus_R",
        "config": "experiment_configs/l24_deml3x4_suffix_v2_2_1.json",
        "method": "suffix_reoptimization_v2.2.1",
        "expected_advanced": "suffix_reoptimization_v2.2.1",
    },
)
REQUIRED_ARTIFACTS = (
    "resolved_config.json",
    "experiment.log",
    "reconstructions.jsonl",
)
EXPECTED_DATASETS = ("Skytrax", "CMS", "ECHR_Law")
EXPECTED_SAMPLES_PER_DATASET = 4

EXIT_EXPERIMENT_FAILURE = 30
EXIT_RESULT_FAILURE = 40
EXIT_UNEXPECTED_FAILURE = 50


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_log(log_file, text):
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("[{}] {}\n".format(utc_now(), text))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(project_dir):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_dir),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def resolve_model_path():
    configured = os.environ.get("DEML_MODEL_PATH")
    if configured:
        return str(Path(configured).expanduser())
    for candidate in MODEL_PATH_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return MODEL_ID


def build_runtime_environment(runtime_dir):
    env = os.environ.copy()
    gpu_id = str(env.get("DEML_GPU_ID", "0")).strip()
    if not gpu_id.isdigit():
        raise ValueError("DEML_GPU_ID must be a non-negative GPU index")
    runtime_dir = Path(runtime_dir).resolve()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "HF_HOME": str(runtime_dir / "hf-cache"),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
    })
    return env


def official_config_path(project_dir, relative_path):
    path = Path(relative_path)
    if not path.is_absolute():
        path = Path(project_dir) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError("missing experiment config: {}".format(path))
    return path


def write_effective_config(
        project_dir, bundle_root, label, official_config, model_path,
        log_dir, smoke=False):
    config_dir = Path(bundle_root) / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "include_configs": [str(Path(official_config).resolve())],
        "base_model_name": model_path,
        "log_dir": str(Path(log_dir).resolve()),
        "output_dir": "suffix_v2.2.1_{}".format(label),
    }
    if smoke:
        smoke_root = Path(bundle_root) / "smoke"
        smoke_root.mkdir(parents=True, exist_ok=True)
        data_path = smoke_root / "smoke_data.json"
        data_path.write_text(
            json.dumps(["A short smoke-test sentence."], ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        config.update({
            "datasets": [{
                "name": "suffix_v2_2_1_smoke",
                "path": str(data_path),
                "type": "local",
                "len": 1,
            }],
            "dataset_path": str(data_path),
            "dataset_type": "local",
            "dataset_len": 1,
            "epoch": 1,
            "suffix_v2_2_1_max_attempts": 1,
            "suffix_v2_2_1_max_attempts_per_position": 1,
            "suffix_v2_2_1_steps": 1,
            "suffix_v2_2_1_range_top_k": 1,
            "top_k_cos": 1,
            "top_k_ppl": 1,
            "device_map": "single_gpu",
            "seed": 0,
        })
    config_path = config_dir / "{}.json".format(label)
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def method_run_root(project_dir, method):
    return (
        Path(project_dir).resolve()
        / "results"
        / "invert_timestamp_runs"
        / method
    )


def wait_for_timestamp_slot(method_root, sleep_fn=time.sleep):
    """Avoid invert.py's second-resolution timestamp collision."""
    method_root = Path(method_root)
    method_root.mkdir(parents=True, exist_ok=True)
    while (method_root / time.strftime("%Y%m%d-%H%M%S")).exists():
        sleep_fn(0.25)


def discover_new_run(method_root, before):
    method_root = Path(method_root)
    after = {
        item.resolve()
        for item in method_root.iterdir()
        if item.is_dir()
    }
    new_runs = sorted(after - set(before), key=lambda item: item.stat().st_mtime)
    if len(new_runs) != 1:
        raise RuntimeError(
            "expected exactly one new run under {}, found {}".format(
                method_root, len(new_runs)
            )
        )
    return new_runs[0]


def run_invert(
    project_dir, python_executable, effective_config, method, log_file,
        env, log_dir=None, sleep_fn=time.sleep):
    method_root = (
        Path(log_dir).resolve() / method
        if log_dir is not None
        else method_run_root(project_dir, method)
    )
    wait_for_timestamp_slot(method_root, sleep_fn=sleep_fn)
    before = {
        item.resolve()
        for item in method_root.iterdir()
        if item.is_dir()
    }
    command = [
        str(python_executable),
        str(Path(project_dir).resolve() / "invert.py"),
        "--config",
        str(Path(effective_config).resolve()),
    ]
    append_log(log_file, "starting: {}".format(" ".join(command)))
    with Path(log_file).open("ab", buffering=0) as output:
        completed = subprocess.run(
            command,
            cwd=str(Path(project_dir).resolve()),
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        append_log(
            log_file,
            "invert.py exited with code {}".format(completed.returncode),
        )
        raise RuntimeError("experiment failed for {}".format(method))
    run_dir = discover_new_run(method_root, before)
    append_log(log_file, "new run: {}".format(run_dir))
    return run_dir


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSONL at {}:{}: {}".format(path, line_number, error)
                ) from error
    return records


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def record_suffix_result(record):
    return (
        record.get("suffix_reoptimization_result")
        or record.get("suffix_reoptimization_v2_2_1_result")
        or {}
    )


def validate_artifacts(
        run_dir, expected_advanced, expected_count=12,
        expected_dataset_names=EXPECTED_DATASETS,
        expected_per_dataset=EXPECTED_SAMPLES_PER_DATASET,
        expected_epoch=1000):
    run_dir = Path(run_dir).resolve()
    missing = [name for name in REQUIRED_ARTIFACTS
               if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing canonical artifacts in {}: {}".format(
                run_dir, ", ".join(missing)
            )
        )
    resolved = load_json(run_dir / "resolved_config.json")
    records = load_jsonl(run_dir / "reconstructions.jsonl")
    if len(records) != int(expected_count):
        raise ValueError(
            "{} expected {} records, found {}".format(
                run_dir, expected_count, len(records)
            )
        )
    actual_names = [
        str((record.get("dataset") or {}).get("name", ""))
        for record in records
    ]
    counts = {name: actual_names.count(name) for name in expected_dataset_names}
    if set(actual_names) != set(expected_dataset_names):
        raise ValueError(
            "unexpected dataset names in {}: {}".format(run_dir, actual_names)
        )
    if any(value != int(expected_per_dataset) for value in counts.values()):
        raise ValueError("dataset sample counts are not 4 each: {}".format(counts))
    if resolved.get("advanced_method", {}).get("name") != expected_advanced:
        raise ValueError(
            "resolved advanced method mismatch: {}".format(
                resolved.get("advanced_method", {}).get("name")
            )
        )
    if resolved.get("candidate_reranking_method", {}).get("name") != "none":
        raise ValueError("formal run unexpectedly enabled candidate reranking")
    if resolved.get("optimization", {}).get("epoch") != int(expected_epoch):
        raise ValueError(
            "run is not using the expected initial stage epoch {}".format(
                expected_epoch
            )
        )
    for record in records:
        if record.get("selected_candidate_reranking_method") != "none":
            raise ValueError("record has a non-none candidate reranker")
        if record.get("selected_advanced_method") != expected_advanced:
            raise ValueError("record selected method mismatch")
    return {
        "run_dir": str(run_dir),
        "resolved_config": resolved,
        "records": records,
        "dataset_counts": counts,
        "artifact_sha256": {
            name: sha256_file(run_dir / name)
            for name in REQUIRED_ARTIFACTS
        },
    }


def validate_smoke_artifacts(run_dir):
    return validate_artifacts(
        run_dir,
        expected_advanced="suffix_reoptimization_v2.2.1",
        expected_count=1,
        expected_dataset_names=("suffix_v2_2_1_smoke",),
        expected_per_dataset=1,
        expected_epoch=1,
    )


def accuracy_summary(records, use_pre=False):
    values = []
    by_dataset = {}
    for record in records:
        result = record_suffix_result(record)
        value = result.get("pre_acc") if use_pre else record.get("accuracy")
        value = numeric(value)
        if value is None:
            continue
        values.append(value)
        dataset_name = str((record.get("dataset") or {}).get("name", ""))
        by_dataset.setdefault(dataset_name, []).append(value)
    return {
        "sample_count": len(values),
        "macro_accuracy": (
            sum(values) / len(values) if values else None
        ),
        "by_dataset": {
            name: {
                "sample_count": len(items),
                "macro_accuracy": sum(items) / len(items),
            }
            for name, items in sorted(by_dataset.items())
        },
    }


def suffix_diagnostics(records):
    result_records = [record_suffix_result(record) for record in records]
    accepted = sum(bool(item.get("accepted")) for item in result_records)
    attempts = sum(int(item.get("attempt_count", 0) or 0)
                   for item in result_records)
    triggered = sum(int(item.get("trigger_count", 0) or 0)
                    for item in result_records)
    budget_exhausted = sum(
        int(item.get("budget_exhausted_count", 0) or 0)
        for item in result_records
    )
    hidden_pre = [numeric(item.get("pre_hidden_loss"))
                  for item in result_records]
    hidden_final = [numeric(item.get("final_hidden_loss"))
                    for item in result_records]
    hidden_pre = [item for item in hidden_pre if item is not None]
    hidden_final = [item for item in hidden_final if item is not None]
    return {
        "sample_count": len(result_records),
        "accepted_sample_count": accepted,
        "attempt_count": attempts,
        "trigger_count": triggered,
        "budget_exhausted_count": budget_exhausted,
        "hidden_loss_mean_before": (
            sum(hidden_pre) / len(hidden_pre) if hidden_pre else None
        ),
        "hidden_loss_mean_after": (
            sum(hidden_final) / len(hidden_final) if hidden_final else None
        ),
        "formal_gt_blind": all(
            bool(item.get("formal_gt_blind")) and not bool(item.get("gt_accessed"))
            for item in result_records
        ),
    }


def compare_runs(baseline_records, suffix_records):
    baseline_by_key = {
        (
            (record.get("dataset") or {}).get("name"),
            (record.get("dataset") or {}).get("sample_index"),
        ): record
        for record in baseline_records
    }
    pre_matches = 0
    comparable = 0
    for record in suffix_records:
        key = (
            (record.get("dataset") or {}).get("name"),
            (record.get("dataset") or {}).get("sample_index"),
        )
        baseline = baseline_by_key.get(key)
        pre_tokens = record_suffix_result(record).get("pre_tokens")
        if baseline:
            baseline_result = baseline.get("frozen_original_baseline_result") or {}
            baseline_tokens = baseline_result.get("final_tokens")
            if baseline_tokens is None:
                baseline_tokens = baseline.get("optimization_result", {}).get("tokens")
        else:
            baseline_tokens = None
        if baseline is None or pre_tokens is None or baseline_tokens is None:
            continue
        comparable += 1
        if list(pre_tokens) == list(baseline_tokens):
            pre_matches += 1
    return {
        "comparable_samples": comparable,
        "pre_tokens_equal_to_baseline_final_tokens": pre_matches,
    }


def build_manifest(project_dir, model_path, runs, baseline_records=None):
    manifest = {
        "generated_at": utc_now(),
        "project_dir": str(Path(project_dir).resolve()),
        "git_revision": git_revision(project_dir),
        "model": {
            "configured_path_or_id": model_path,
            "model_id": MODEL_ID,
        },
        "experiment_contract": {
            "datasets": list(EXPECTED_DATASETS),
            "samples_per_dataset": EXPECTED_SAMPLES_PER_DATASET,
            "total_samples": 12,
            "initial_optimization_epochs": 1000,
            "top_k_cos": 10,
            "top_k_ppl": 10,
            "candidate_reranking": "none",
            "discretization": "Original DEML embedding-top10 + PPL-top10 + hidden-cosine",
        },
        "runs": {},
    }
    for label, info in runs.items():
        records = info["records"]
        entry = {
            "run_dir": info["run_dir"],
            "dataset_counts": info["dataset_counts"],
            "artifact_sha256": info["artifact_sha256"],
            "accuracy": accuracy_summary(records),
        }
        if label == "baseline_plus_R":
            entry["pre_accuracy"] = accuracy_summary(records, use_pre=True)
            entry["suffix_diagnostics"] = suffix_diagnostics(records)
        manifest["runs"][label] = entry
    if baseline_records is not None and "baseline_plus_R" in runs:
        manifest["comparison"] = compare_runs(
            baseline_records,
            runs["baseline_plus_R"]["records"],
        )
    return manifest


def run_bundle(
        project_dir, runtime_dir, result_root, log_file, python_executable,
        smoke=False, sleep_fn=time.sleep):
    project_dir = Path(project_dir).resolve()
    runtime_dir = Path(runtime_dir).resolve()
    result_root = Path(result_root).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    env = build_runtime_environment(runtime_dir)
    model_path = resolve_model_path()
    canonical_log_dir = project_dir / "results" / "invert_timestamp_runs"
    run_infos = {}

    if smoke:
        official = official_config_path(
            project_dir,
            "experiment_configs/l24_deml3x4_suffix_v2_2_1.json",
        )
        config = write_effective_config(
            project_dir,
            result_root,
            "smoke",
            official,
            model_path,
            result_root / "smoke-runs",
            smoke=True,
        )
        run_dir = run_invert(
            project_dir,
            python_executable,
            config,
            "suffix_reoptimization_v2.2.1",
            log_file,
            env,
            log_dir=result_root / "smoke-runs",
            sleep_fn=sleep_fn,
        )
        info = validate_smoke_artifacts(run_dir)
        manifest = build_manifest(project_dir, model_path, {"smoke": info})
        manifest["mode"] = "smoke"
    else:
        for index, spec in enumerate(FORMAL_RUNS):
            official = official_config_path(project_dir, spec["config"])
            config = write_effective_config(
                project_dir,
                result_root,
                spec["label"],
                official,
                model_path,
                canonical_log_dir,
            )
            run_dir = run_invert(
                project_dir,
                python_executable,
                config,
                spec["method"],
                log_file,
                env,
                sleep_fn=sleep_fn,
            )
            run_infos[spec["label"]] = validate_artifacts(
                run_dir,
                expected_advanced=spec["expected_advanced"],
            )
            if index == 0:
                # invert.py names runs to the second; reserve a new second
                # before launching the suffix run.
                sleep_fn(1.1)
        manifest = build_manifest(
            project_dir,
            model_path,
            run_infos,
            baseline_records=run_infos["baseline"]["records"],
        )
        manifest["mode"] = "formal"

    manifest_path = result_root / "ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_log(log_file, "manifest: {}".format(manifest_path))
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run and verify suffix v2.2.1 smoke/formal experiments"
    )
    parser.add_argument("command", choices=("smoke", "formal"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        manifest = run_bundle(
            project_dir=args.project,
            runtime_dir=args.runtime,
            result_root=args.result_root,
            log_file=args.log_file,
            python_executable=args.python_executable,
            smoke=args.command == "smoke",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        append_log(args.log_file, "{}: {}".format(type(error).__name__, error))
        return EXIT_RESULT_FAILURE if isinstance(
            error, (FileNotFoundError, ValueError, RuntimeError)
        ) else EXIT_UNEXPECTED_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
