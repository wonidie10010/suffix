#!/usr/bin/env python3
"""Freeze independent v2.5 calibration, then compare shared-snapshot search off/on."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import runner_suffix_v2_2_2 as common

METHOD = "suffix_reoptimization_v2.5"
KEY = "suffix_reoptimization_v2_5_result"
PREFIX = "suffix_v2_5_"
ARTIFACTS = common.ARTIFACTS
ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "experiment_configs/l24_deml3x4_suffix_v2_5.json"
digest = common.digest


def read_json(path):
    def invalid(value):
        raise ValueError("nonstandard JSON: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha_text(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


def records_at(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / "reconstructions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_frozen_result(result):
    """Reject results that break the continuous/discrete phase boundary."""
    checks = result["freeze_checks"]
    if not checks["passed"] or not checks["final_decisions_match_tokens"]:
        raise ValueError("freeze or final decision check failed")
    for field in ("r_optimization_calls_after_freeze", "optimization_backtracking_events",
                  "optimization_backtracking_forward_calls", "grad_enabled_forward_calls",
                  "requires_grad_embedding_forward_calls"):
        if checks[field] != 0:
            raise ValueError("freeze boundary: " + field)
    for kind in ("embedding", "target"):
        if (not checks[kind + "_unchanged"] or not checks[kind + "_sha256_before"]
                or checks[kind + "_sha256_before"] != checks[kind + "_sha256_after"]):
            raise ValueError("frozen " + kind + " hash mismatch")
    search = result["backtracking"]
    limits = dict(extra_forward_budget=640, max_events=2, max_distance=4,
                  alternatives=4, max_trials=16, lookahead=2)
    if any(search[key] != value for key, value in limits.items()):
        raise ValueError("v2.5 search configuration mismatch")
    if search["compute"]["forward_calls"] > 640 or search["event_count"] > 2:
        raise ValueError("v2.5 search budget exceeded")
    for event in search["events"]:
        if event["call_id"] != "final_discretization" or not event["low_fit"]:
            raise ValueError("backtracking phase/necessary low-fit trigger mismatch")
        if len(event["trials"]) > 16:
            raise ValueError("route budget exceeded")
        for proposal in event["proposals"]:
            ids = [row["token_id"] for row in proposal["alternatives"]]
            if len(ids) > 4 or len(ids) != len(set(ids)):
                raise ValueError("alternative budget/identity mismatch")
        if event["accepted"]:
            completed = [trial for trial in event["trials"] if trial["completed"]]
            chosen = event["trials"][event["accepted_trial_index"]]
            if (event["budget_exhausted"] or not chosen["completed"]
                    or event["E_selected"] != min(trial["E_trial"] for trial in completed)
                    or not event["E_selected"] < event["E_base"] - event["epsilon"]):
                raise ValueError("minimum complete route/strict improvement mismatch")
    import hashlib
    tokens, start = result["final_tokens"], result["eval_start_pos"]
    decisions = [row for row in result["discretization"]["events"] if row["final_retained"]]
    if sorted(row["position"] for row in decisions) != list(range(start, len(tokens))):
        raise ValueError("final decision positions mismatch")
    for row in decisions:
        # Same canonical integer-list serialization as the sidecar.
        prefix = hashlib.sha256(json.dumps(tokens[:row["position"]], separators=(",", ":")).encode()).hexdigest()
        if (row["selected_token_id"] != tokens[row["position"]]
                or row["call_id"] != "final_discretization" or row["prefix_fingerprint"] != prefix):
            raise ValueError("final token/decision/prefix mismatch")


def validate_optimization_pair(a, b):
    for result in (a, b):
        validate_frozen_result(result)
    fields = ("pre_tokens", "optimization_final_tokens", "pre_hidden_loss", "final_hidden_loss",
              "attempt_count", "accepted_round_count", "rejected_round_count", "final_trajectory_contracts")
    for field in fields:
        if a[field] != b[field]:
            raise ValueError("paired optimization differs: " + field)
    r_fields = ("position", "triggered", "attempted", "accepted", "pre_hidden_loss",
                "post_hidden_loss", "changed_positions", "reason", "call_id", "trial_tokens",
                "entrance_tokens", "rolled_back")
    r_rows = lambda result: [{field: row.get(field) for field in r_fields} for row in result["events"]]
    if r_rows(a) != r_rows(b):
        raise ValueError("paired R losses/acceptance/tokens differ")
    for field in ("embedding_sha256_before", "target_sha256_before", "r_optimizer_calls_before_freeze",
                  "r_attempts_completed_before_freeze"):
        if a["freeze_checks"][field] != b["freeze_checks"][field]:
            raise ValueError("paired frozen state differs: " + field)
    if a["backtracking"]["calibration"] != b["backtracking"]["calibration"]:
        raise ValueError("paired frozen calibration differs")
    return dict(passed=True, all_R_identical=True, optimization_tokens_identical=True,
                embedding_sha256=a["freeze_checks"]["embedding_sha256_before"],
                target_sha256=a["freeze_checks"]["target_sha256_before"])


def preflight(project, runtime, model_path, gpu):
    project, runtime, model = Path(project).resolve(), Path(runtime).resolve(), Path(model_path).resolve()
    config = common.load_config(project / CONFIG.relative_to(ROOT))
    if (config.get("suffix_version") != "v2.5" or not config.get("suffix_reoptimization_v2_5")
            or config.get("cgmr_version") != "none" or config.get("epoch") != 1000
            or config.get("num_invert_layers") != 24 or config.get(PREFIX + "steps") != 50
            or config.get("seed") != 0):
        raise ValueError("formal selector/Stage-1/R configuration mismatch")
    if [(d["name"], d["len"]) for d in config["datasets"]] != [("Skytrax", 4), ("CMS", 4), ("ECHR_Law", 4)]:
        raise ValueError("formal sample configuration mismatch")
    limits = dict(max_distance=4, alternatives=4, max_trials=16, max_events=2,
                  forward_budget=640, lookahead=2)
    if any(config.get(PREFIX + "backtracking_" + key) != value for key, value in limits.items()):
        raise ValueError("formal v2.5 search configuration mismatch")
    cache = common.inspect_model_cache(model)
    if cache["model_cache_status"] != "hit":
        raise ValueError("existing local model cache unavailable")
    for dataset in config["datasets"]:
        if not (project / dataset["path"]).exists():
            raise FileNotFoundError(dataset["path"])
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    selected = next((line.split(",") for line in rows.splitlines() if line.split(",")[0].strip() == str(gpu)), None)
    if selected is None or int(selected[1]) < 12000:
        raise RuntimeError("selected GPU needs at least 12 GiB free before this run")
    return dict(config=config, model_path=str(model), model_cache=cache, gpu=str(gpu),
                gpu_preflight=rows.splitlines(), shared_gpu=True,
                calibration_samples=6, budget_samples=3, evaluation_samples=12,
                calibration_rule="first two distinct samples after evaluation indices plus one held-out budget sample, 8..192 tokens, per dataset",
                equal_compute_budget=False, backtracking_extra_forward_budget=640,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip())


def prepare_calibration(project, bundle, plan):
    sys.path.insert(0, str(project))
    from dataset import Dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan["model_path"], trust_remote_code=True, use_fast=False, local_files_only=True)
    evaluation, pending = [], []
    for spec in plan["config"]["datasets"]:
        data = list(Dataset(str(project / spec["path"]), spec["type"]).get_data())
        pending.append((spec, data))
        for index, text in enumerate(data[:spec["len"]]):
            tokens = tokenizer(str(text))["input_ids"]
            evaluation.append(dict(pair_id=spec["name"] + ":" + str(index), dataset=spec["name"],
                source_path=spec["path"], source_index=index, text_sha256=sha_text(text),
                token_sha256=sha_text(json.dumps(tokens)), token_count=len(tokens)))
    excluded_text = {row["text_sha256"] for row in evaluation}
    excluded_tokens = {row["token_sha256"] for row in evaluation}
    texts, identities, budget_texts, budget_identities = [], [], [], []
    for spec, data in pending:
        selected = 0
        for index in range(spec["len"], len(data)):
            text = str(data[index])
            tokens = tokenizer(text)["input_ids"]
            text_hash, token_hash = sha_text(text), sha_text(json.dumps(tokens))
            if not 8 <= len(tokens) <= 192 or text_hash in excluded_text or token_hash in excluded_tokens:
                continue
            is_budget = selected == 2
            sample_id = ("Budget_v25:" + str(len(budget_texts)) if is_budget else "Calibration_v25:" + str(len(texts)))
            identity = dict(sample_id=sample_id, dataset=spec["name"], source_path=spec["path"],
                source_index=index, text_sha256=text_hash, token_sha256=token_hash, token_count=len(tokens))
            (budget_identities if is_budget else identities).append(identity)
            (budget_texts if is_budget else texts).append(text)
            excluded_text.add(text_hash)
            excluded_tokens.add(token_hash)
            selected += 1
            if selected == 3:
                break
        if selected != 3:
            raise ValueError("insufficient independent calibration samples: " + spec["name"])
    dump(bundle / "calibration_samples.json", texts)
    dump(bundle / "budget_samples.json", budget_texts)
    dump(bundle / "sample_manifest.json", dict(evaluation=evaluation, calibration=identities, budget=budget_identities,
        overlap_text_sha256=[], overlap_token_sha256=[], selection_rule=plan["calibration_rule"]))
    return identities


def validate_run(run_dir, count, enabled, calibration):
    run_dir = Path(run_dir)
    for name in ARTIFACTS:
        if not (run_dir / name).is_file():
            raise ValueError("missing artifact: " + name)
    resolved, records = read_json(run_dir / "resolved_config.json"), records_at(run_dir)
    method = resolved["advanced_methods"]["suffix_reoptimization_v2_5"]
    if (resolved["advanced_method"]["name"] != METHOD or method["backtracking_enabled"] is not enabled
            or method["backtracking_calibration"] != calibration or len(records) != count):
        raise ValueError("run configuration/calibration/sample count mismatch")
    for record in records:
        result = record[KEY]
        if (record["selected_advanced_method"] != METHOD or record["selected_candidate_reranking_method"] != "none"
                or result.get("gt_accessed") is not False or not result.get("formal_gt_blind")
                or result.get("anomaly_reasons")):
            raise ValueError("method/GT boundary or failed R trial")
        search = result["backtracking"]
        if (search["compute"]["forward_calls"] > 640 or search["event_count"] > 2
                or (enabled and not search["calibration_ready"])):
            raise ValueError("backtracking budget or calibration violated")
        validate_frozen_result(result)
    return records


def launch_group(project, bundle, manifest, plan, label, calibration, enabled, snapshots, mode, calibration_data=False):
    count = (3 if label == "calibration_budget" else 6) if calibration_data else 12
    retained = bundle / "runs" / label
    if label in manifest["groups"]:
        return validate_run(retained, count, enabled, calibration)
    config = dict(plan["config"])
    config.update(base_model_name=plan["model_path"], log_dir=str(project / "results/invert_timestamp_runs"),
                  suffix_v222_snapshot_dir=str(snapshots), suffix_v222_snapshot_mode=mode,
                  suffix_v222_run_kind="independent_calibration" if calibration_data else "formal")
    config[PREFIX + "backtracking_enabled"] = enabled
    config[PREFIX + "backtracking_calibration"] = calibration
    for key in ("fit_threshold", "first_fit_threshold", "acceptance_epsilon", "consistency_tolerance"):
        config[PREFIX + "backtracking_" + key] = calibration.get(key)
    if calibration_data:
        budget = label == "calibration_budget"
        path = str(bundle / ("budget_samples.json" if budget else "calibration_samples.json"))
        config.update(datasets=[dict(name="Budget_v25" if budget else "Calibration_v25", path=path, type="local", len=count)],
                      dataset_path=path, dataset_type="local", dataset_len=count)
    config_path = bundle / ("effective_" + label + ".json")
    dump(config_path, config)
    method_root = project / "results/invert_timestamp_runs" / METHOD
    before = set(method_root.iterdir()) if method_root.exists() else set()
    manifest.update(stage=label, stage_started_at=datetime.now(timezone.utc).isoformat())
    dump(bundle / "manifest.json", manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (bundle / "runner.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-B", str(project / "invert_suffix_v2_5.py"), "--config", str(config_path)],
                                   cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest["active_child_pid"] = process.pid
        dump(bundle / "manifest.json", manifest)
        code = process.wait()
    manifest.pop("active_child_pid", None)
    created = [p for p in set(method_root.iterdir()) - before if p.is_dir()]
    if len(created) == 1:
        source = created[0]
        retained.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACTS:
            if (source / name).is_file():
                shutil.copy2(source / name, retained / name)
        manifest.setdefault("attempts", {})[label] = dict(run_dir=str(source), retained=str(retained), exit_code=code)
        dump(bundle / "manifest.json", manifest)
    if code or len(created) != 1:
        raise RuntimeError(label + " exited " + str(code) + "; timestamp directories=" + str(len(created)))
    records = validate_run(retained, count, enabled, calibration)
    if not calibration_data and Counter(r["dataset"]["name"] for r in records) != Counter(Skytrax=4, CMS=4, ECHR_Law=4):
        raise ValueError("formal dataset counts differ")
    manifest["groups"][label] = dict(run_dir=str(source), retained=str(retained), config_sha256=digest(config_path),
                                    sample_count=count, artifacts={name: digest(retained / name) for name in ARTIFACTS})
    dump(bundle / "manifest.json", manifest)
    return records


def freeze_calibration(project, bundle, model_path):
    sys.path.insert(0, str(project))
    import torch
    import invert_suffix_v2_5 as invert
    from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_5 as method
    torch.set_num_threads(1)
    tokenizer, model = invert.get_model(model_path, None, device_map="single_gpu", offload_state_dict=True, quantization="none")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    samples = []
    for index, record in enumerate(records_at(bundle / "runs/calibration_greedy")):
        snapshot = torch.load(bundle / "calibration_snapshots" / ("sample_%06d.pt" % index), map_location="cpu", weights_only=False)
        samples.append(dict(sample_id=record["pair_id"], result=record[KEY],
            target_hidden_state=snapshot["target"].to(invert.get_model_device(model)),
            attention_mask=torch.tensor(snapshot["contract"]["mask"], device=invert.get_model_device(model))))
    with torch.no_grad():
        calibration = method.collect_backtracking_calibration(samples, model, invert.get_input_embedding_layer(model), tokenizer,
            24, invert.forward_and_get_last_hidden_state, repeats=3, quantile=0.05)
    calibration.update(source_samples=read_json(bundle / "sample_manifest.json")["calibration"],
                       source_artifacts_sha256={name: digest(bundle / "runs/calibration_greedy" / name) for name in ARTIFACTS})
    calibration.pop("calibration_id")
    calibration["calibration_id"] = hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    dump(bundle / "frozen_calibration.json", calibration)


def compare_pair(off, on):
    if not off or [r["pair_id"] for r in off] != [r["pair_id"] for r in on]:
        raise ValueError("paired identity/order mismatch")
    samples = []
    for left, right in zip(off, on):
        a, b = left[KEY], right[KEY]
        x, y = left["discretization_offline_evaluation"], right["discretization_offline_evaluation"]
        if (not a.get("stage1_snapshot_sha256") or a["stage1_snapshot_sha256"] != b["stage1_snapshot_sha256"]
                or a["stage1_reused"] is not False or b["stage1_reused"] is not True):
            raise ValueError("shared snapshot write/read mismatch")
        for field in ("reference_sha256", "eval_start_pos", "evaluated_token_count"):
            if x[field] != y[field]:
                raise ValueError("paired evaluation differs: " + field)
        paired_freeze = validate_optimization_pair(a, b)
        common_initial = x["pre_correctness"]
        item = dict(pair_id=left["pair_id"], dataset=left["dataset"]["name"],
                    token_count=x["evaluated_token_count"], snapshot_sha256=a["stage1_snapshot_sha256"], freeze=paired_freeze, groups={})
        for label, result, offline in (("off", a, x), ("on", b, y)):
            final, coverage = offline["final_correctness"], offline["candidate_coverage"]
            if len(final) != len(common_initial):
                raise ValueError("paired correctness length mismatch")
            search = result["backtracking"]
            events = offline.get("backtracking_events", [])
            selected = [event["selected_changes"] for event in events if event["accepted"] and event["selected_changes"]]
            costs = result["compute_stats"]
            item["groups"][label] = dict(correct=sum(final), accuracy=offline["accuracy"],
                repaired=sum(not pre and post for pre, post in zip(common_initial, final)),
                damaged=sum(pre and not post for pre, post in zip(common_initial, final)),
                coverage=dict(decisions=len(coverage), covered=sum(row["pool_contains_gt"] for row in coverage),
                    final_decisions=sum(row["final_retained"] for row in coverage),
                    final_covered=sum(row["final_retained"] and row["pool_contains_gt"] for row in coverage),
                    history_newly_covered=sum(row.get("history_newly_covered", False) for row in coverage),
                    retained_history_newly_covered=sum(row["final_retained"] and row.get("history_newly_covered", False) for row in coverage)),
                backtracking={key: search[key] for key in ("triggered_count", "event_count", "accepted_count", "formal_accepted_count",
                    "final_retained_count", "budget_exhausted_count")},
                local_accepted_path_repairs=sum(row["repairs"] for row in selected),
                local_accepted_path_damage=sum(row["damage"] for row in selected),
                r_accepted=result["accepted_round_count"], r_rejected=result["rejected_round_count"],
                extra_compute=search["compute"], second_stage_seconds=result["second_stage_seconds"],
                second_stage_peak_memory_bytes=result["second_stage_peak_memory_bytes"],
                normal_forward_calls=sum(costs[key]["forward_calls"] for key in ("candidate", "ppl", "observation", "r_optimization")))
        item["direct_final_repairs"] = sum(not pre and post for pre, post in zip(x["final_correctness"], y["final_correctness"]))
        item["direct_final_damage"] = sum(pre and not post for pre, post in zip(x["final_correctness"], y["final_correctness"]))
        off_tables = {row["position"]: row for row in x["final_candidate_coverage"]}
        on_tables = {row["position"]: row for row in y["final_candidate_coverage"]}
        item["newly_final_covered"] = sum(row["pool_contains_gt"] and not off_tables.get(pos, {}).get("pool_contains_gt", False) for pos, row in on_tables.items())
        item["lost_final_coverage"] = sum(row["pool_contains_gt"] and not on_tables.get(pos, {}).get("pool_contains_gt", False) for pos, row in off_tables.items())
        samples.append(item)
    totals = {}
    token_count = sum(row["token_count"] for row in samples)
    for label in ("off", "on"):
        rows = [row["groups"][label] for row in samples]
        total = dict(sample_count=len(rows), token_count=token_count, correct=sum(r["correct"] for r in rows),
            macro_accuracy=sum(r["accuracy"] for r in rows) / len(rows),
            repaired=sum(r["repaired"] for r in rows), damaged=sum(r["damaged"] for r in rows),
            second_stage_seconds=sum(r["second_stage_seconds"] for r in rows), normal_forward_calls=sum(r["normal_forward_calls"] for r in rows),
            extra_forward_calls=sum(r["extra_compute"]["forward_calls"] for r in rows),
            extra_token_count=sum(r["extra_compute"]["token_count"] for r in rows),
            local_accepted_path_repairs=sum(r["local_accepted_path_repairs"] for r in rows),
            local_accepted_path_damage=sum(r["local_accepted_path_damage"] for r in rows),
            r_accepted=sum(r["r_accepted"] for r in rows), r_rejected=sum(r["r_rejected"] for r in rows))
        total["micro_accuracy"] = total["correct"] / token_count
        for field in ("coverage", "backtracking"):
            total[field] = {key: sum(row[field][key] for row in rows) for key in rows[0][field]}
        totals[label] = total
    return dict(equal_compute_budget=False, phase_pairing="all R and frozen continuous state identical; only final discretization differs", baseline="off-arm initial discrete output from shared continuous Stage-1",
        coverage_unit="candidate decision; final coverage paired by absolute position",
        shared_gpu=True, samples=samples, totals=totals,
        direct_final_repairs=sum(r["direct_final_repairs"] for r in samples),
        direct_final_damage=sum(r["direct_final_damage"] for r in samples),
        newly_final_covered=sum(r["newly_final_covered"] for r in samples),
        lost_final_coverage=sum(r["lost_final_coverage"] for r in samples))


def verify_bundle(bundle):
    """Recheck returned artifact hashes and pairing without loading the model."""
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("experiment is incomplete")
    calibration = read_json(bundle / "frozen_calibration.json")
    frozen = {key: value for key, value in calibration.items() if key != "calibration_id"}
    checksum = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if checksum != calibration["calibration_id"] or checksum != manifest["calibration_id"]:
        raise ValueError("frozen calibration identity mismatch")
    samples = read_json(bundle / "sample_manifest.json")
    for field in ("text_sha256", "token_sha256"):
        groups = [{row[field] for row in samples[key]} for key in ("evaluation", "calibration", "budget")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("calibration/evaluation/budget split overlap")
    records = {}
    for label, count, enabled in (("calibration_greedy", 6, False), ("calibration_budget", 3, True),
                                   ("off", 12, False), ("on", 12, True)):
        path = bundle / "runs" / label
        group = manifest["groups"][label]
        for name in ARTIFACTS:
            if digest(path / name) != group["artifacts"][name]:
                raise ValueError("returned artifact SHA256 mismatch: " + label + "/" + name)
        records[label] = validate_run(path, count, enabled, {} if label == "calibration_greedy" else calibration)
    if [r["pair_id"] for r in records["off"]] != [row["pair_id"] for row in samples["evaluation"]]:
        raise ValueError("formal sample manifest/order mismatch")
    for label, directory in (("calibration_greedy", "calibration_snapshots"),
                             ("calibration_budget", "budget_snapshots"), ("off", "snapshots")):
        for index, record in enumerate(records[label]):
            path = bundle / directory / ("sample_%06d.pt" % index)
            if digest(path) != record[KEY]["stage1_snapshot_sha256"]:
                raise ValueError("returned shared snapshot SHA256 mismatch: " + str(path))
    for name in ARTIFACTS:
        if digest(bundle / "runs/calibration_greedy" / name) != calibration["source_artifacts_sha256"][name]:
            raise ValueError("calibration source artifact mismatch")
    comparison = compare_pair(records["off"], records["on"])
    if comparison != read_json(bundle / "comparison.json"):
        raise ValueError("recomputed comparison differs")
    return dict(passed=True, calibration_id=checksum, sample_counts={k: len(v) for k, v in records.items()},
                all_R_and_frozen_states_identical=True, artifacts_sha256_verified=True,
                comparison_recomputed=True)


def run_bundle(args):
    import fcntl
    project, runtime, bundle = Path(args.project).resolve(), Path(args.runtime).resolve(), Path(args.bundle).resolve()
    lock = (runtime / "suffix-experiment.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = preflight(project, runtime, args.model_path, args.gpu)
    if (bundle / "manifest.json").exists():
        manifest = read_json(bundle / "manifest.json")
        if manifest["status"] == "complete":
            print("Already complete: " + str(bundle))
            return
        if not args.resume:
            raise FileExistsError("existing bundle; inspect it before explicit --resume")
        if manifest["commit"] != plan["commit"]:
            raise ValueError("resume commit mismatch")
        manifest.update(status="running", errors=[])
    else:
        bundle.mkdir(parents=True, exist_ok=True)
        manifest = dict(status="running", run_kind="formal", groups={}, errors=[], commit=plan["commit"],
            model_path=plan["model_path"], model_cache=plan["model_cache"], python=sys.executable,
            gpu=plan["gpu"], gpu_preflight=plan["gpu_preflight"], shared_gpu=True,
            started_at=datetime.now(timezone.utc).isoformat(), execution_order=["calibration_greedy", "calibration_budget", "off", "on"],
            equal_compute_budget=False, parent_pid=os.getpid())
    dump(bundle / "manifest.json", manifest)
    dump(bundle / "plan.json", plan)
    try:
        if not (bundle / "sample_manifest.json").exists():
            prepare_calibration(project, bundle, plan)
        launch_group(project, bundle, manifest, plan, "calibration_greedy", {}, False, bundle / "calibration_snapshots", "write", True)
        if not (bundle / "frozen_calibration.json").exists():
            manifest["stage"] = "freeze_calibration"
            dump(bundle / "manifest.json", manifest)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu"], HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            with (bundle / "runner.log").open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "freeze", "--project", str(project),
                    "--runtime", str(runtime), "--model-path", plan["model_path"], "--bundle", str(bundle), "--gpu", plan["gpu"]],
                    cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        calibration = read_json(bundle / "frozen_calibration.json")
        manifest["calibration_id"] = calibration["calibration_id"]
        budget_records = launch_group(project, bundle, manifest, plan, "calibration_budget", calibration, True, bundle / "budget_snapshots", "write", True)
        budget_check = dict(forward_budget=640, frozen_before_evaluation=True,
            event_counts=[r[KEY]["backtracking"]["event_count"] for r in budget_records],
            extra_forward_calls=[r[KEY]["backtracking"]["compute"]["forward_calls"] for r in budget_records],
            exhausted_counts=[r[KEY]["backtracking"]["budget_exhausted_count"] for r in budget_records],
            policy="retain design budget 640; interruption uses original greedy fallback; no evaluation-driven tuning")
        dump(bundle / "budget_preflight.json", budget_check)
        off = launch_group(project, bundle, manifest, plan, "off", calibration, False, bundle / "snapshots", "write")
        on = launch_group(project, bundle, manifest, plan, "on", calibration, True, bundle / "snapshots", "read")
        dump(bundle / "comparison.json", compare_pair(off, on))
        manifest.update(status="complete", stage="complete", completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="failed", errors=[type(error).__name__ + ": " + str(error)], traceback=traceback.format_exc())
        raise
    finally:
        dump(bundle / "manifest.json", manifest)
    print(bundle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "run", "freeze", "verify"))
    for name in ("project", "runtime", "model-path", "bundle"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", default="1", choices=("0", "1"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "dry-run":
        plan = preflight(args.project, args.runtime, args.model_path, args.gpu)
        print(json.dumps({k: v for k, v in plan.items() if k != "config"}, ensure_ascii=True, indent=2))
    elif args.mode == "verify":
        print(json.dumps(verify_bundle(Path(args.bundle).resolve()), ensure_ascii=False, indent=2))
    elif args.mode == "freeze":
        freeze_calibration(Path(args.project).resolve(), Path(args.bundle).resolve(), args.model_path)
    else:
        run_bundle(args)


if __name__ == "__main__":
    main()
