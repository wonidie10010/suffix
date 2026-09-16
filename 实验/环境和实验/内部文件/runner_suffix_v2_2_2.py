#!/usr/bin/env python3
"""Standard-library-only preflight and paired Stage-1 snapshot orchestration."""
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
import tempfile
import time
import uuid

METHOD = "suffix_reoptimization_v2.2.2"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
PREFIX = "suffix_v2_2_2_"
CONFIGS = {label: "experiment_configs/l24_deml3x4_suffix_v2_2_2_cp_{}.json".format(label)
           for label in ("off", "on")}
ARTIFACTS = ("resolved_config.json", "experiment.log", "reconstructions.jsonl")
FROZEN = dict(checkpoint_size=5, checkpoint_stride=5, checkpoint_trigger_cosine=0.90,
              checkpoint_diagnostic_tolerance=0.05, checkpoint_candidate_min_cosine=0.90,
              checkpoint_candidate_threshold_source="user_fixed_2026_09_16",
              checkpoint_forward_mode="full_prefix",
              checkpoint_candidate_failure_policy="abort_checkpoint_keep_state",
              checkpoint_tail_policy="skip_incomplete", checkpoint_max_repairs=1,
              checkpoint_recursive=False, checkpoint_numeric_norm_epsilon=1e-8,
              checkpoint_score_dtype="float32", checkpoint_schema_version=1)


def load_config(path, stack=()):
    path = Path(path).resolve()
    if path in stack:
        raise ValueError("configuration include cycle")
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = {}
    for included in data.get("include_configs", []):
        merged.update(load_config(path.parent / included, (*stack, path)))
    merged.update({k: v for k, v in data.items() if k not in ("include_configs", "__comments")})
    if "suffix_reoptimization_version" in merged and merged["suffix_reoptimization_version"] != merged.get("suffix_version"):
        raise ValueError("conflicting suffix selectors")
    return merged


def digest(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def validate_pair(configs):
    differences = {key for key in set(configs["off"]) | set(configs["on"])
                   if configs["off"].get(key) != configs["on"].get(key)}
    if differences != {PREFIX+"checkpoint_enabled", "output_dir"}:
        raise ValueError("unexpected paired configuration differences: " + repr(differences))
    for label, config in configs.items():
        if config.get("suffix_version") != "v2.2.2" or config.get("suffix_reoptimization_v2_2_2") is not True:
            raise ValueError("v2.2.2 selector/enable required")
        if config.get("cgmr_version") != "none":
            raise ValueError("paired experiment requires CGMR none")
        if config.get(PREFIX+"checkpoint_enabled") is not (label == "on"):
            raise ValueError("wrong checkpoint switch")
        for key, value in FROZEN.items():
            actual = config.get(PREFIX+key)
            if type(actual) is not type(value) or actual != value:
                raise ValueError("invalid frozen setting: " + PREFIX+key)
    return True


def inspect_model_cache(path):
    """Read existing local assets only. Never fall back to a remote model ID."""
    path=Path(path).resolve()
    info=dict(model_id=MODEL_ID, resolved_model_path=str(path), model_cache_status="invalid",
              download_performed=False, model_revision=None, missing=[])
    for name in ("config.json","tokenizer_config.json"):
        if not (path/name).is_file(): info["missing"].append(name)
    if not (path/"tokenizer.json").is_file() and not all((path/name).is_file() for name in ("vocab.json","merges.txt")):
        info["missing"].append("tokenizer.json or vocab.json+merges.txt")
    weights=[]
    index=next((path/name for name in ("model.safetensors.index.json","pytorch_model.bin.index.json") if (path/name).is_file()),None)
    if index:
        metadata=json.loads(index.read_text(encoding="utf-8"))
        weights=[path/name for name in sorted(set(metadata["weight_map"].values()))]
    else:
        weights=[p for p in (path/"model.safetensors",path/"pytorch_model.bin") if p.is_file()]
    if not weights:info["missing"].append("model weights")
    for weight in weights:
        if weight.parent.resolve()!=path or not weight.is_file() or weight.stat().st_size==0:
            info["missing"].append(weight.name)
    if info["missing"]:return info
    config=json.loads((path/"config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in ("qwen2","qwen2_5","qwen2.5"):
        raise ValueError("unexpected cached model family")
    info.update(model_cache_status="hit", model_revision=config.get("_commit_hash") or (
        path.name if path.parent.name=="snapshots" else None),
        config_sha256=digest(path/"config.json"), tokenizer_config_sha256=digest(path/"tokenizer_config.json"),
        weight_files=[dict(name=p.name,bytes=p.stat().st_size) for p in weights],
        total_weight_bytes=sum(p.stat().st_size for p in weights))
    return info


def preflight(project, model_path=None, runtime=None):
    project = Path(project).resolve()
    for required in ("invert.py", "实验/一键运行_suffix_v2_2_2.py",
                     "suffix_optimization_methods/method_versions/suffix_reoptimization_v2_2_2.py"):
        if not (project / required).is_file():
            raise FileNotFoundError(project / required)
    configs = {label: load_config(project / path) for label, path in CONFIGS.items()}
    validate_pair(configs)
    selected = model_path or os.environ.get("DEML_MODEL_PATH")
    if not selected:
        # Same shared locations as existing launchers; no version-specific cache.
        candidates = [Path("/mnt/my_disk/tch/models/Qwen2.5-1.5B"), project / "models/Qwen2.5-1.5B"]
        shared_runtime=Path(runtime) if runtime else project/"实验/环境和实验/.runtime"
        cache_roots=[shared_runtime/"hf-cache/hub"]
        cache_roots.extend(Path(os.environ[name]) for name in ("HF_HUB_CACHE","TRANSFORMERS_CACHE") if os.environ.get(name))
        if os.environ.get("HF_HOME"): cache_roots.append(Path(os.environ["HF_HOME"])/"hub")
        for cache in cache_roots:
            repo=cache/"models--Qwen--Qwen2.5-1.5B"
            ref=repo/"refs/main"
            if ref.is_file(): candidates.append(repo/"snapshots"/ref.read_text().strip())
        present=[path.resolve() for path in candidates if (path/"config.json").is_file()]
        present=list(dict.fromkeys(present))
        if len(present)>1:
            raise ValueError("multiple existing model sources; set --model-path explicitly: "+repr([str(p) for p in present]))
        selected=str(present[0]) if present else None
    model = Path(selected).expanduser() if selected else project / "models/Qwen2.5-1.5B"
    if not model.is_absolute():
        model = project / model
    pending = []
    cache_info=inspect_model_cache(model)
    if cache_info["model_cache_status"]!="hit":
        pending.append("local model: " + str(model)+" missing "+repr(cache_info["missing"]))
    for dataset in configs["off"]["datasets"]:
        path = Path(dataset["path"])
        if not path.is_absolute():
            path = project / path
        if not path.exists():
            pending.append("dataset: " + str(path))
    return dict(configs=configs, model_path=str(model.resolve()), model_cache=cache_info, pending_server_checks=pending,
                groups=["cp_off", "cp_on"], stage1="write once; read same snapshots in second group",
                real_model_loaded=False)


def read_artifacts(run_dir, enabled, expected_count):
    run_dir = Path(run_dir)
    for name in ARTIFACTS:
        if not (run_dir / name).is_file():
            raise ValueError("missing artifact: " + str(run_dir / name))
    def strict_json(text):
        return json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("nonstandard JSON: "+v)))
    resolved = strict_json((run_dir/"resolved_config.json").read_text(encoding="utf-8"))
    records = [strict_json(line) for line in (run_dir/"reconstructions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    advanced = resolved.get("advanced_methods", {}).get("suffix_reoptimization_v2_2_2", {})
    if advanced.get("checkpoint_enabled") is not enabled or resolved.get("advanced_method", {}).get("name") != METHOD:
        raise ValueError("artifact selector/switch mismatch")
    for key, value in FROZEN.items():
        if advanced.get(key) != value:
            raise ValueError("artifact contract mismatch: " + key)
    if len(records) != expected_count:
        raise ValueError("unexpected sample count")
    keys = []
    for record in records:
        result = record.get("suffix_reoptimization_v2_2_2_result", {})
        if record.get("selected_advanced_method") != METHOD or record.get("selected_candidate_reranking_method") != "none":
            raise ValueError("unexpected online method")
        if not result.get("formal_gt_blind") or result.get("gt_accessed") is not False:
            raise ValueError("invalid GT boundary flags")
        if not result.get("stage1_snapshot_sha256") or not result.get("pair_id"):
            raise ValueError("missing snapshot identity")
        keys.append(result["pair_id"])
        checkpoint = result["reoptimization"]["checkpoint"]
        expected_events = checkpoint["complete_segment_count"] if enabled else 0
        if len(checkpoint["events"]) != expected_events:
            raise ValueError("incomplete checkpoint events")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample identity")
    return records


def compare_pair(off_records, on_records):
    if [r["pair_id"] for r in off_records] != [r["pair_id"] for r in on_records]:
        raise ValueError("paired sample identity/order mismatch")
    rows = []
    for off, on in zip(off_records, on_records):
        before = off["suffix_reoptimization_v2_2_2_result"]
        after = on["suffix_reoptimization_v2_2_2_result"]
        if before["stage1_snapshot_sha256"] != after["stage1_snapshot_sha256"]:
            raise ValueError("paired Stage-1 snapshots differ")
        row=dict(pair_id=on["pair_id"], dataset=on.get("dataset",{}).get("name"), off_accuracy=off["accuracy"], on_accuracy=on["accuracy"],
                         delta=on["accuracy"]-off["accuracy"], snapshot_sha256=after["stage1_snapshot_sha256"],
                         off_second_stage_seconds=before["second_stage_seconds"],
                         on_second_stage_seconds=after["second_stage_seconds"])
        left,right=off.get("checkpoint_offline_evaluation"),on.get("checkpoint_offline_evaluation")
        if left is not None and right is not None:
            if left["evaluated_token_count"] != right["evaluated_token_count"] or left["eval_start_pos"] != right["eval_start_pos"]:
                raise ValueError("offline evaluation positions differ")
            row.update(evaluated_token_count=right["evaluated_token_count"],
                       off_correct=left["correct_token_count"],on_correct=right["correct_token_count"],
                       final_repairs=sum(not a and b for a,b in zip(left["final_correctness"],right["final_correctness"])),
                       final_damage=sum(a and not b for a,b in zip(left["final_correctness"],right["final_correctness"])),
                       direct_repairs=right["direct_repairs"],direct_damage=right["direct_damage"])
            if not row["evaluated_token_count"]:
                row.update(off_accuracy=None,on_accuracy=None,delta=None)
        checkpoint=after["checkpoint"]
        events=checkpoint["events"]
        row["checkpoint"]={key:value for key,value in checkpoint.items() if key!="events"}
        def ratio(numerator,denominator):return numerator/denominator if denominator else None
        triggered=sum(e.get("triggered") is True for e in events)
        localized=sum(e.get("selected_position") is not None for e in events)
        stale=sum(e.get("reason")=="missing_or_stale_candidate_table" for e in events)
        attempts=sum(e.get("repair_attempt_count",0) for e in events)
        accepted=sum(e.get("accepted") is True for e in events)
        row["checkpoint_rates"]=dict(
            triggered=ratio(triggered,checkpoint["complete_segment_count"]),
            localized=ratio(localized,triggered),
            no_alternative=ratio(sum(e.get("reason")=="no_eligible_alternative" for e in events),localized-stale),
            accepted=ratio(accepted,attempts),
            candidate_failure=ratio(sum(e.get("reason") in ("candidate_forward_failed","invalid_candidate_score") for e in events),attempts),
            hidden_improved_but_token_damaged=ratio(row.get("direct_damage",0),accepted))
        def ratio(numerator,denominator):return numerator/denominator if denominator else None
        triggered=sum(e.get("triggered") is True for e in events)
        localized=sum(e.get("selected_position") is not None for e in events)
        stale=sum(e.get("reason")=="missing_or_stale_candidate_table" for e in events)
        attempts=sum(e.get("repair_attempt_count",0) for e in events)
        accepted=sum(e.get("accepted") is True for e in events)
        row["checkpoint_rates"]=dict(
            triggered=ratio(triggered,checkpoint["complete_segment_count"]),
            localized=ratio(localized,triggered),
            no_alternative=ratio(sum(e.get("reason")=="no_eligible_alternative" for e in events),localized-stale),
            accepted=ratio(accepted,attempts),
            candidate_failure=ratio(sum(e.get("reason") in ("candidate_forward_failed","invalid_candidate_score") for e in events),attempts),
            hidden_improved_but_token_damaged=ratio(row.get("direct_damage",0),accepted))
        row["cp_forward_calls"]=sum(e.get("observation_forward_calls",0)+e.get("candidate_forward_calls",0) for e in events)
        row["cp_forward_token_count"]=sum(e.get("forward_token_count",0) for e in events)
        row["peak_memory_bytes"]=dict(off=before.get("second_stage_peak_memory_bytes"),on=after.get("second_stage_peak_memory_bytes"))
        row["comparable_end_to_end_seconds"]=dict(off=before.get("comparable_end_to_end_seconds"),on=after.get("comparable_end_to_end_seconds"))
        rows.append(row)
    valid=[row for row in rows if row["delta"] is not None]
    def aggregate(items):
        denominator=sum(row.get("evaluated_token_count",0) for row in items)
        summary={}
        for label in ("off","on"):
            summary[label+"_macro_accuracy"]=sum(r[label+"_accuracy"] for r in items)/len(items) if items else None
            summary[label+"_micro_accuracy"]=sum(r.get(label+"_correct",0) for r in items)/denominator if denominator else None
        return summary
    return dict(samples=rows, mean_delta=sum(row["delta"] for row in valid)/len(valid) if valid else None,
                accuracy=aggregate(valid), by_dataset={name:aggregate([r for r in valid if r["dataset"]==name]) for name in {r["dataset"] for r in valid}},
                improved=sum(row["delta"] > 0 for row in valid), worsened=sum(row["delta"] < 0 for row in valid),
                unchanged=sum(row["delta"] == 0 for row in valid), not_applicable=len(rows)-len(valid))


def run_bundle(project, runtime, result_root, python, model_path=None, smoke=False, run=subprocess.run):
    project, runtime = Path(project).resolve(), Path(runtime).resolve()
    plan = preflight(project, model_path, runtime)
    pending=[item for item in plan["pending_server_checks"] if not smoke or not item.startswith("dataset:")]
    if pending:
        raise ValueError("server preflight incomplete: " + repr(pending))
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    gpu = os.environ.get("DEML_GPU_ID", "0")
    if not gpu.isdigit():
        raise ValueError("invalid DEML_GPU_ID")
    bundle = Path(result_root).resolve()/tag
    bundle.mkdir(parents=True, exist_ok=False)
    temp_root = project/"outputs"
    temp_root.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="v222_", dir=temp_root))
    manifest = dict(status="running", run_kind="smoke" if smoke else "formal", groups={}, errors=[])
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1",
               HF_HOME=str(runtime/"hf-cache"), TOKENIZERS_PARALLELISM="false")
    env["CUDA_VISIBLE_DEVICES"] = gpu
    records_by_label = {}
    try:
        revision = run(["git", "rev-parse", "HEAD"], cwd=str(project), capture_output=True, text=True, check=True)
        status = run(["git", "status", "--porcelain"], cwd=str(project), capture_output=True, text=True, check=True)
        manifest.update(commit=revision.stdout.strip(), worktree_status=status.stdout,
                        model_path=plan["model_path"], model_cache=plan["model_cache"], python=python, gpu=gpu)
        # Pin an unpacked local directory by content when no HF commit metadata exists.
        if not manifest["model_cache"]["model_revision"] and not smoke:
            weights=manifest["model_cache"].get("weight_files",[])
            checksums={item["name"]:digest(Path(plan["model_path"])/item["name"]) for item in weights}
            manifest["model_cache"]["weight_sha256"]=checksums
            manifest["model_cache"]["model_revision"]="local-sha256:"+hashlib.sha256(json.dumps(checksums,sort_keys=True).encode()).hexdigest()
        canonical_root = project/"results/invert_timestamp_runs"
        method_root = canonical_root/METHOD
        # Alternate which condition writes the shared snapshot across bundles.
        order = ["on", "off"] if int(tag[-1], 16) % 2 else ["off", "on"]
        manifest["execution_order"] = order
        for index, label in enumerate(order):
            config = dict(plan["configs"][label])
            config.update(base_model_name=plan["model_path"], log_dir=str(canonical_root),
                          suffix_v222_snapshot_dir=str(temporary/"snapshots"),
                          suffix_v222_snapshot_mode="write" if index == 0 else "read",
                          suffix_v222_run_kind="smoke" if smoke else "formal")
            expected_count = 12
            if smoke:
                data = temporary/"smoke.json"
                dump(data, ["A short smoke test sentence with enough tokens for two checkpoint segments."])
                config.update(datasets=[dict(name="checkpoint_smoke", path=str(data), type="local", len=1)],
                              dataset_path=str(data), dataset_type="local", dataset_len=1, epoch=1,
                              suffix_v2_2_2_steps=1, suffix_v2_2_2_max_attempts=1)
                expected_count=1
            config_path = bundle/("effective_cp_"+label+".json")
            dump(config_path, config)
            # The shared output layer uses second-resolution timestamp directories.
            while (method_root/time.strftime("%Y%m%d-%H%M%S")).exists():
                time.sleep(0.1)
            before = set(method_root.iterdir()) if method_root.exists() else set()
            with (bundle/"runner.log").open("a", encoding="utf-8") as log:
                completed = run([python, str(project/"invert.py"), "--config", str(config_path)],
                                cwd=str(project), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError("CP-{} exited {}".format(label, completed.returncode))
            created = set(method_root.iterdir())-before
            runs = [p for p in created if p.is_dir() and (p/"reconstructions.jsonl").exists()]
            if len(runs) != 1:
                raise ValueError("cannot identify one new timestamp run")
            records = read_artifacts(runs[0], label == "on", expected_count)
            if not smoke:
                counts=Counter(r["dataset"]["name"] for r in records)
                if counts != Counter(Skytrax=4, CMS=4, ECHR_Law=4):
                    raise ValueError("formal dataset counts differ")
            elif any(r["suffix_reoptimization_v2_2_2_result"]["checkpoint"]["complete_segment_count"] < 1 for r in records):
                raise ValueError("smoke did not contain one full segment")
            records_by_label[label] = records
            manifest["groups"][label] = dict(run_dir=str(runs[0]), config_sha256=digest(config_path),
                                             jsonl_bytes=(runs[0]/"reconstructions.jsonl").stat().st_size,
                                             artifacts={name:digest(runs[0]/name) for name in ARTIFACTS})
            dump(bundle/"manifest.json", manifest)
        summary=compare_pair(records_by_label["off"], records_by_label["on"])
        dump(bundle/"paired_summary.json", summary)
        manifest.update(status="complete", pairs=summary["samples"])
    except Exception as error:
        manifest.update(status="failed", errors=[type(error).__name__+": "+str(error)])
        raise
    finally:
        # Delete only this invocation's mkdtemp child, never an existing directory.
        if temporary.parent.resolve() != temp_root.resolve() or not temporary.name.startswith("v222_"):
            raise RuntimeError("unexpected temporary path")
        shutil.rmtree(temporary)
        dump(bundle/"manifest.json", manifest)
    return bundle


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dry-run", "smoke", "formal"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path")
    args=parser.parse_args(argv)
    try:
        if args.mode == "dry-run":
            plan=preflight(args.project, args.model_path, args.runtime)
            plan.pop("configs")
            plan.update(result_root=str(Path(args.result_root).resolve()), validation="layout/config only; no real model or experiment")
            print(json.dumps(plan, ensure_ascii=True, indent=2))
        else:
            print(run_bundle(args.project,args.runtime,args.result_root,args.python,args.model_path,args.mode=="smoke"))
        return 0
    except Exception as error:
        print(type(error).__name__+": "+str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
