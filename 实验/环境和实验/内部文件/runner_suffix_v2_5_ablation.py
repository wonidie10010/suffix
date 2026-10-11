#!/usr/bin/env python3
"""Cumulative R/E/B ablation, with verified shared Stage1 snapshots and fresh B calibration."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import runner_suffix_v2_5 as paired

ROOT, KEY, PREFIX = paired.ROOT, paired.KEY, paired.PREFIX
ARTIFACTS = paired.ARTIFACTS
read_json, dump, digest, records_at = paired.read_json, paired.dump, paired.digest, paired.records_at
ARMS = dict(baseline=(False, False, False), r=(True, False, False),
            re=(True, True, False), reb=(True, True, True))
SWITCH_KEYS = tuple(PREFIX + name for name in
                    ("reoptimization_enabled", "expansion_enabled", "backtracking_enabled"))


def arm_configs(project):
    configs = {label: paired.common.load_config(project / "experiment_configs" /
               ("l24_deml3x4_suffix_v2_5_ablation_" + label + ".json")) for label in ARMS}
    controls = []
    for label, config in configs.items():
        if tuple(config[key] for key in SWITCH_KEYS) != ARMS[label]:
            raise ValueError("wrong component switches: " + label)
        controls.append({k: v for k, v in config.items() if k not in (*SWITCH_KEYS, "output_dir")})
    if any(control != controls[0] for control in controls[1:]):
        raise ValueError("non-component configuration differs between arms")
    return configs


def validate_components(result, switches):
    expected = dict(zip(("R", "E", "B"), switches))
    if result["component_switches"] != expected:
        raise ValueError("result component switches differ")
    paired.validate_frozen_result(result)
    if not expected["R"] and (result["attempt_count"] or result["events"]
                             or result["freeze_checks"]["r_optimizer_calls_before_freeze"]):
        raise ValueError("R executed in disabled arm")
    search = result["backtracking"]
    if not expected["B"] and (search["event_count"] or search["compute"]["forward_calls"]):
        raise ValueError("B executed in disabled arm")
    if result["discretization"]["r_trajectory_candidates_enabled"] is not False:
        raise ValueError("R trajectory injection enabled")
    for row in result["discretization"]["events"]:
        if row["expansion_enabled"] is not expected["E"]:
            raise ValueError("decision expansion switch differs")
        for candidate in row["base_candidates"]:
            for source in candidate["sources"]:
                if source["source"] == "trajectory" and source["call_id"].startswith("R_"):
                    raise ValueError("R step 30/40 trajectory entered candidate pool")
                if not expected["E"] and source["source"] not in ("embedding", "ppl"):
                    raise ValueError("trajectory source injected with E disabled")
        if not expected["E"]:
            if row["trajectory_candidate_ids"] or row["seeds"] or row["variant_candidates"]:
                raise ValueError("E candidates present in disabled arm")
            if row["original_candidate_ids"] != [c["token_id"] for c in row["base_candidates"]]:
                raise ValueError("disabled E changed original candidate pool")


def validate_group(path, count, switches, calibration):
    records = paired.validate_run(path, count, switches[2], calibration)
    settings = read_json(Path(path) / "resolved_config.json")["advanced_methods"]["suffix_reoptimization_v2_5"]
    if settings["component_switches"] != dict(zip(("R", "E", "B"), switches)):
        raise ValueError("resolved component switches differ")
    if settings["r_trajectory_candidates_enabled"] is not False:
        raise ValueError("resolved R trajectory policy differs")
    for record in records:
        validate_components(record[KEY], switches)
        if record[KEY]["stage1_reused"] is not True:
            raise ValueError("all groups must read the shared Stage1 snapshot")
    return records


def prepare_snapshots(project, bundle, source, plan):
    """Validate current texts/model and every serialized Stage1 contract before reuse."""
    import torch
    if read_json(source / "manifest.json")["status"] != "complete":
        raise ValueError("source snapshot bundle incomplete")
    paired.prepare_calibration(project, bundle, plan)
    identities = read_json(bundle / "sample_manifest.json")
    if identities != read_json(source / "sample_manifest.json"):
        raise ValueError("source/current evaluation or calibration data/order changed")
    source_manifest = read_json(source / "manifest.json")
    for key in ("resolved_model_path", "config_sha256", "tokenizer_config_sha256", "weight_files"):
        if source_manifest["model_cache"][key] != plan["model_cache"][key]:
            raise ValueError("source/current model differs: " + key)
    verified = []
    for label, directory, split, id_key in (("off", "snapshots", "evaluation", "pair_id"),
            ("calibration_greedy", "calibration_snapshots", "calibration", "sample_id"),
            ("calibration_budget", "budget_snapshots", "budget", "sample_id")):
        old = records_at(source / "runs" / label)
        if len(old) != len(identities[split]):
            raise ValueError("source snapshot count mismatch")
        for index, (record, identity) in enumerate(zip(old, identities[split])):
            src = source / directory / ("sample_%06d.pt" % index)
            checksum = digest(src)
            if checksum != record[KEY]["stage1_snapshot_sha256"]:
                raise ValueError("source Stage1 SHA256 mismatch")
            snapshot = torch.load(src, map_location="cpu", weights_only=False)
            contract = snapshot["contract"]
            if snapshot["pair_id"] != identity[id_key]:
                raise ValueError("source Stage1 identity mismatch")
            for field in ("lr", "epoch", "alpha", "clip", "init_method", "init_param", "optim_method", "seed"):
                if contract[field] != plan["config"][field]:
                    raise ValueError("source Stage1 parameter mismatch: " + field)
            if contract["model"] != plan["model_path"] or contract["block"] != plan["config"]["num_invert_layers"]:
                raise ValueError("source Stage1 model/layer mismatch")
            if snapshot["stage1"]["completed_steps"] != plan["config"]["epoch"]:
                raise ValueError("source Stage1 did not complete planned steps")
            group = snapshot["trajectory_groups"]["v2.5"]
            if group["sampling_steps"] != [500, 600, 700, 800, 900]:
                raise ValueError("source Stage1 sampling differs")
            destination = bundle / directory / src.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and digest(destination) != checksum:
                raise ValueError("existing destination snapshot differs")
            if not destination.exists():
                shutil.copy2(src, destination)
            verified.append(dict(path=destination.relative_to(bundle).as_posix(), sha256=checksum,
                                 pair_id=snapshot["pair_id"], contract=contract))
    dump(bundle / "stage1_reuse_validation.json", dict(passed=True, source=str(source),
        source_commit=source_manifest["commit"], current_commit=plan["commit"], snapshots=verified,
        data_order_and_tokenization_identical=True, model_cache_identical=True,
        stage1_parameters_identical=True, injection_policy="E gates consumption; shared snapshots retain Stage1 metadata"))


def launch(project, bundle, manifest, plan, label, switches, calibration, split="evaluation"):
    count = dict(evaluation=12, calibration=6, budget=3)[split]
    retained = bundle / "runs" / label
    if label in manifest["groups"]:
        return validate_group(retained, count, switches, calibration)
    config = dict(arm_configs(project).get(label, plan["config"]))
    config.update(zip(SWITCH_KEYS, switches))
    snapshots = bundle / dict(evaluation="snapshots", calibration="calibration_snapshots", budget="budget_snapshots")[split]
    config.update(base_model_name=plan["model_path"], log_dir=str(project / "results/invert_timestamp_runs"),
        suffix_v222_snapshot_dir=str(snapshots), suffix_v222_snapshot_mode="read",
        suffix_v222_run_kind="formal" if split == "evaluation" else "independent_calibration")
    config[PREFIX + "backtracking_calibration"] = calibration
    for key in ("fit_threshold", "first_fit_threshold", "acceptance_epsilon", "consistency_tolerance"):
        config[PREFIX + "backtracking_" + key] = calibration.get(key)
    if split != "evaluation":
        path = str(bundle / ("calibration_samples.json" if split == "calibration" else "budget_samples.json"))
        config.update(datasets=[dict(name="Calibration_v25" if split == "calibration" else "Budget_v25",
                      path=path, type="local", len=count)], dataset_path=path, dataset_type="local", dataset_len=count)
    config_path = bundle / ("effective_" + label + ".json")
    dump(config_path, config)
    method_root = project / "results/invert_timestamp_runs" / paired.METHOD
    before = set(method_root.iterdir()) if method_root.exists() else set()
    manifest.update(stage=label, stage_started_at=datetime.now(timezone.utc).isoformat())
    dump(bundle / "manifest.json", manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    begun = time.perf_counter()
    with (bundle / "runner.log").open("a", encoding="utf-8") as log:
        command = [sys.executable, "-B", str(project / "invert_suffix_v2_5.py"), "--config", str(config_path)]
        process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(active_child_pid=process.pid, active_command=command)
        dump(bundle / "manifest.json", manifest)
        code = process.wait()
    wall_seconds = time.perf_counter() - begun
    manifest.pop("active_child_pid", None)
    created = [p for p in set(method_root.iterdir()) - before if p.is_dir()]
    if len(created) == 1:
        source = created[0]
        retained.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACTS:
            if (source / name).is_file():
                shutil.copy2(source / name, retained / name)
        manifest.setdefault("attempts", {}).setdefault(label, []).append(
            dict(run_dir=str(source), exit_code=code, wall_seconds=wall_seconds))
        dump(bundle / "manifest.json", manifest)
    if code or len(created) != 1:
        raise RuntimeError(label + " exited " + str(code) + "; timestamp directories=" + str(len(created)))
    records = validate_group(retained, count, switches, calibration)
    manifest["groups"][label] = dict(run_dir=str(source), retained=str(retained), sample_count=count,
        wall_seconds=wall_seconds, component_switches=dict(zip(("R", "E", "B"), switches)),
        config_sha256=digest(config_path), artifacts={name: digest(retained / name) for name in ARTIFACTS})
    dump(bundle / "manifest.json", manifest)
    return records


def compare(groups):
    order = [r["pair_id"] for r in groups["baseline"]]
    if not order or len(set(order)) != len(order) or any([r["pair_id"] for r in rows] != order for rows in groups.values()):
        raise ValueError("group identity/order mismatch")
    samples = []
    for rows in zip(*(groups[label] for label in ARMS)):
        first = rows[0]
        offline = first["discretization_offline_evaluation"]
        sample = dict(pair_id=first["pair_id"], dataset=first["dataset"]["name"],
                      token_count=offline["evaluated_token_count"], groups={})
        for label, record in zip(ARMS, rows):
            result, evaluated = record[KEY], record["discretization_offline_evaluation"]
            validate_components(result, ARMS[label])
            for field in ("reference_sha256", "eval_start_pos", "evaluated_token_count"):
                if evaluated[field] != offline[field]:
                    raise ValueError("paired evaluation differs: " + field)
            if (record["original"] != first["original"] or result["stage1"] != first[KEY]["stage1"]
                    or result["stage1_snapshot_sha256"] != first[KEY]["stage1_snapshot_sha256"]):
                raise ValueError("paired shared Stage1 differs")
            flags = evaluated["final_correctness"]
            if len(flags) != sample["token_count"] or sum(flags) != evaluated["correct_token_count"]:
                raise ValueError("correctness numerator/denominator mismatch")
            accuracy = sum(flags) / len(flags)
            if not math.isclose(accuracy, evaluated["accuracy"], abs_tol=1e-12) or not math.isclose(accuracy, record["accuracy"], abs_tol=1e-12):
                raise ValueError("stored accuracy differs from token flags")
            coverage = evaluated["final_candidate_coverage"]
            sample["groups"][label] = dict(correct=sum(flags), accuracy=accuracy,
                second_stage_seconds=result["second_stage_seconds"], comparable_end_to_end_seconds=result["comparable_end_to_end_seconds"],
                stage1_seconds=result["stage1"]["elapsed_seconds"],
                r_attempts=result["attempt_count"], r_accepted=result["accepted_round_count"], r_rejected=result["rejected_round_count"],
                b_events=result["backtracking"]["event_count"], b_accepted=result["backtracking"]["accepted_count"],
                b_forwards=result["backtracking"]["compute"]["forward_calls"],
                final_covered=sum(c["pool_contains_gt"] for c in coverage),
                final_decisions=len(coverage), pre_hidden_loss=result["pre_hidden_loss"], final_hidden_loss=result["final_hidden_loss"])
        sample["B_pairing"] = paired.validate_optimization_pair(rows[2][KEY], rows[3][KEY])
        samples.append(sample)
    totals = {}
    for label in ARMS:
        arm = [row["groups"][label] for row in samples]
        total = {key: sum(r[key] for r in arm) for key in arm[0] if key not in ("accuracy", "pre_hidden_loss", "final_hidden_loss")}
        total.update(sample_count=len(samples), token_count=sum(r["token_count"] for r in samples),
                     macro_accuracy=sum(r["accuracy"] for r in arm) / len(arm))
        total["micro_accuracy"] = total["correct"] / total["token_count"]
        totals[label] = total
    adjacent = []
    for before, after in zip(ARMS, list(ARMS)[1:]):
        changes = []
        for left, right in zip(groups[before], groups[after]):
            a, b = left["discretization_offline_evaluation"], right["discretization_offline_evaluation"]
            repairs = [i + a["eval_start_pos"] for i, (x, y) in enumerate(zip(a["final_correctness"], b["final_correctness"])) if not x and y]
            damage = [i + a["eval_start_pos"] for i, (x, y) in enumerate(zip(a["final_correctness"], b["final_correctness"])) if x and not y]
            changes.append(dict(pair_id=left["pair_id"], repair_positions=repairs, damage_positions=damage,
                repairs=len(repairs), damage=len(damage), net=len(repairs)-len(damage)))
        adjacent.append(dict(before=before, after=after, repairs=sum(c["repairs"] for c in changes),
            damage=sum(c["damage"] for c in changes), net_correct=totals[after]["correct"]-totals[before]["correct"],
            micro_gain_pp=100*(totals[after]["micro_accuracy"]-totals[before]["micro_accuracy"]),
            macro_gain_pp=100*(totals[after]["macro_accuracy"]-totals[before]["macro_accuracy"]), samples=changes))
    return dict(baseline="v2.5 base flow with R/E/B disabled", equal_compute_budget=False,
        R_trajectory_injection=False, shared_stage1=True, B_pair_all_R_and_frozen_states_identical=True,
        timing="second_stage excludes shared Stage1; comparable_end_to_end adds its original measured cost", totals=totals,
        adjacent=adjacent, samples=samples)


def verify(bundle, retokenize=False):
    manifest = read_json(bundle / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("experiment incomplete")
    hashes = read_json(bundle / "sha256_manifest.json")
    for name, checksum in hashes.items():
        if digest(bundle / name) != checksum:
            raise ValueError("bundle SHA256 mismatch: " + name)
    calibration = read_json(bundle / "frozen_calibration.json")
    frozen = {k: v for k, v in calibration.items() if k != "calibration_id"}
    if hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() != calibration["calibration_id"]:
        raise ValueError("calibration SHA256 mismatch")
    groups = {}
    for label, group in manifest["groups"].items():
        records = validate_group(bundle / "runs" / label, group["sample_count"],
                                 tuple(group["component_switches"][k] for k in ("R", "E", "B")),
                                 {} if label == "calibration_greedy" else calibration)
        if label in ARMS:
            groups[label] = records
        for index, record in enumerate(records):
            directory = "snapshots" if label in ARMS else "calibration_snapshots" if label == "calibration_greedy" else "budget_snapshots"
            if digest(bundle / directory / ("sample_%06d.pt" % index)) != record[KEY]["stage1_snapshot_sha256"]:
                raise ValueError("record/shared snapshot SHA256 mismatch")
    comparison = compare(groups)
    if comparison != read_json(bundle / "comparison.json"):
        raise ValueError("recomputed comparison differs")
    count = 0
    if retokenize:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], trust_remote_code=True, use_fast=False, local_files_only=True)
        for group in manifest["groups"]:
            for record in records_at(bundle / "runs" / group):
                ids = tokenizer(str(record["original"]))["input_ids"]
                start = record["eval_start_pos"]
                if start:
                    ids = [tokenizer.bos_token_id] + ids
                evaluated = record["discretization_offline_evaluation"]
                if paired.sha_text(json.dumps(ids)) != evaluated["reference_sha256"]:
                    raise ValueError("original retokenization reference differs")
                predicted = record[KEY]["final_tokens"]
                flags = [a == b for a, b in zip(ids[start:], predicted[start:])]
                if len(ids) != len(predicted) or flags != evaluated["final_correctness"]:
                    raise ValueError("independent exact token-ID accuracy differs")
                count += 1
    return dict(passed=True, artifacts_sha256_verified=len(hashes), comparison_recomputed=True,
        shared_stage1_verified=True, component_gates_verified=True, R_trajectory_injection=False,
        B_pair_all_R_and_frozen_states_identical=True, original_records_retokenized=count,
        sample_counts={label: len(records) for label, records in groups.items()})


def run(args):
    import fcntl
    project, runtime, bundle, source = (Path(getattr(args, k)).resolve() for k in ("project", "runtime", "bundle", "source_bundle"))
    lock = (runtime / "suffix-experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = paired.preflight(project, runtime, args.model_path, args.gpu)
    arm_configs(project)
    if (bundle / "manifest.json").exists():
        manifest = read_json(bundle / "manifest.json")
        if manifest["status"] == "complete":
            print(json.dumps(verify(bundle), indent=2))
            return
        if not args.resume or manifest["commit"] != plan["commit"]:
            raise ValueError("explicit resume at the same commit required")
        manifest.update(status="running", errors=[])
    else:
        bundle.mkdir(parents=True, exist_ok=True)
        manifest = dict(status="running", groups={}, errors=[], commit=plan["commit"], model_path=plan["model_path"],
            model_cache=plan["model_cache"], python=sys.executable, gpu=plan["gpu"], parent_pid=os.getpid(),
            started_at=datetime.now(timezone.utc).isoformat(), source_bundle=str(source), shared_gpu=True,
            execution_order=["calibration_greedy", "calibration_budget", *ARMS], equal_compute_budget=False)
    dump(bundle / "manifest.json", manifest)
    dump(bundle / "plan.json", plan)
    try:
        if not (bundle / "stage1_reuse_validation.json").exists():
            prepare_snapshots(project, bundle, source, plan)
        launch(project, bundle, manifest, plan, "calibration_greedy", ARMS["re"], {}, "calibration")
        if not (bundle / "frozen_calibration.json").exists():
            manifest["stage"] = "freeze_calibration"
            dump(bundle / "manifest.json", manifest)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with (bundle / "runner.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-B", str(Path(paired.__file__)), "freeze", "--project", str(project),
                    "--runtime", str(runtime), "--model-path", plan["model_path"], "--bundle", str(bundle), "--gpu", args.gpu],
                    cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        calibration = read_json(bundle / "frozen_calibration.json")
        manifest["calibration_id"] = calibration["calibration_id"]
        budget = launch(project, bundle, manifest, plan, "calibration_budget", ARMS["reb"], calibration, "budget")
        dump(bundle / "budget_preflight.json", dict(forward_budget=640, frozen_before_evaluation=True,
            extra_forward_calls=[r[KEY]["backtracking"]["compute"]["forward_calls"] for r in budget],
            event_counts=[r[KEY]["backtracking"]["event_count"] for r in budget]))
        groups = {label: launch(project, bundle, manifest, plan, label, switches, calibration) for label, switches in ARMS.items()}
        dump(bundle / "comparison.json", compare(groups))
        manifest.update(status="complete", stage="complete", completed_at=datetime.now(timezone.utc).isoformat())
        dump(bundle / "manifest.json", manifest)
        dump(bundle / "sha256_manifest.json", {p.relative_to(bundle).as_posix(): digest(p)
            for p in sorted(bundle.rglob("*")) if p.is_file() and p.name not in
            ("runner.log", "launcher.log", "sha256_manifest.json", "server_validation.json", "local_bundle_verification.json")})
        dump(bundle / "server_validation.json", verify(bundle, retokenize=True))
    except Exception as error:
        manifest.update(status="failed", errors=[type(error).__name__ + ": " + str(error)], traceback=traceback.format_exc())
        dump(bundle / "manifest.json", manifest)
        raise
    print(bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "run", "verify"))
    for name in ("project", "runtime", "model-path", "bundle", "source-bundle"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", default="1", choices=("0", "1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retokenize", action="store_true")
    args = parser.parse_args()
    if args.mode == "verify":
        print(json.dumps(verify(Path(args.bundle).resolve(), args.retokenize), indent=2))
    elif args.mode == "dry-run":
        plan = paired.preflight(args.project, args.runtime, args.model_path, args.gpu)
        configs = arm_configs(Path(args.project))
        print(json.dumps(dict(commit=plan["commit"], model=plan["model_cache"], gpu=plan["gpu_preflight"],
            arms={k: {key: c[key] for key in SWITCH_KEYS} for k, c in configs.items()}), indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
