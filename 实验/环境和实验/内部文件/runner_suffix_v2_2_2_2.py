#!/usr/bin/env python3
"""Compare checkpoint expansion with discretization expansion from one Stage-1 snapshot."""
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
import uuid

import runner_suffix_v2_2_2 as control

METHODS = {"checkpoint": "suffix_reoptimization_v2.2.2", "discretization": "suffix_reoptimization_v2.2.2(2)"}
KEYS = {"checkpoint": "suffix_reoptimization_v2_2_2_result", "discretization": "suffix_reoptimization_v2_2_2_2_result"}
CONFIG = "experiment_configs/l24_deml3x4_suffix_v2_2_2_2.json"
COMMON_R = ("max_attempts", "max_attempts_per_position", "steps", "lr", "trigger_mode", "trigger_threshold",
            "hidden_weight_mode", "hidden_weight_decay", "hidden_weight_floor", "prox_weight", "range_weight",
            "range_top_k", "accept_mode", "filter_nonascii")
dump, digest = control.dump, control.digest


def preflight(project, model_path=None, runtime=None):
    plan = control.preflight(project, model_path, runtime)
    old = plan["configs"]["on"]
    new = control.load_config(Path(project)/CONFIG)
    if (new.get("suffix_version") != "v2.2.2(2)" or new.get("suffix_reoptimization_v2_2_2_2") is not True
            or new.get("suffix_v2_2_2_2_checkpoint_enabled") is not False or new.get("cgmr_version") != "none"
            or new.get("suffix_v2_2_2_2_expansion_policy") != "checkpoint_sources_2_plus_4"):
        raise ValueError("invalid discretization arm configuration")
    # All non-method settings, including data selection and Stage-1, must be identical.
    def common(config):
        return {k:v for k,v in config.items() if not k.startswith(("suffix_", "__")) and k != "output_dir"}
    if common(old) != common(new):
        raise ValueError("paired sample/Stage-1 configuration mismatch")
    for field in COMMON_R:
        if old["suffix_v2_2_2_"+field] != new["suffix_v2_2_2_2_"+field]:
            raise ValueError("paired R configuration mismatch: "+field)
    plan.update(configs={"checkpoint":old, "discretization":new}, groups=list(METHODS),
                stage1="checkpoint writes; discretization reads exactly the same snapshots",
                equal_compute_budget=False, comparison_baseline="checkpoint arm initial discrete tokens")
    return plan


def compare_pair(control_records, expanded_records):
    if len(control_records) != len(expanded_records) or not control_records:
        raise ValueError("paired sample count mismatch or empty run")
    samples, seen = [], set()
    for left, right in zip(control_records, expanded_records):
        a, b = left[KEYS["checkpoint"]], right[KEYS["discretization"]]
        x, y = left["discretization_offline_evaluation"], right["discretization_offline_evaluation"]
        identity = left["pair_id"]
        if identity in seen or identity != right["pair_id"] or left["dataset"] != right["dataset"]:
            raise ValueError("paired sample identity/order mismatch")
        seen.add(identity)
        if not a.get("stage1_snapshot_sha256") or a["stage1_snapshot_sha256"] != b.get("stage1_snapshot_sha256"):
            raise ValueError("paired Stage-1 snapshot mismatch")
        if a.get("stage1_reused") is not False or b.get("stage1_reused") is not True:
            raise ValueError("paired Stage-1 write/read contract mismatch")
        for field in ("reference_sha256", "eval_start_pos", "evaluated_token_count"):
            if x[field] != y[field]:
                raise ValueError("paired reference/evaluation mismatch: "+field)
        count, start = x["evaluated_token_count"], x["eval_start_pos"]
        baseline = x["pre_correctness"]
        item = dict(pair_id=identity, dataset=left["dataset"]["name"], evaluated_token_count=count,
                    common_initial_correct=sum(baseline), stage1_snapshot_sha256=a["stage1_snapshot_sha256"], groups={})
        for label, result, offline in (("checkpoint",a,x), ("discretization",b,y)):
            final = offline["final_correctness"]
            if len(final) != count or len(baseline) != count or len(result["final_tokens"]) != count+start:
                raise ValueError("paired correctness alignment mismatch")
            coverage = offline["candidate_coverage"]
            phases = {}
            for phase in ("initial", "R"):
                rows = [row for row in coverage if (row["phase"] == "initial") == (phase == "initial")]
                phases[phase] = dict(decisions=len(rows), contains_gt=sum(r["pool_contains_gt"] for r in rows),
                    validly_scored_gt=sum(r["gt_validly_scored"] for r in rows),
                    retained_decisions=sum(r["retained_in_formal_state"] for r in rows))
            additions = offline["expansion_coverage"]
            item["groups"][label] = dict(correct=sum(final), accuracy=sum(final)/count if count else None,
                repaired=sum(not pre and post for pre,post in zip(baseline,final)),
                damaged=sum(pre and not post for pre,post in zip(baseline,final)),
                wrong_to_wrong_changes=sum(not pre and not post and old != new for pre,post,old,new in
                    zip(baseline, final, a["pre_tokens"][start:], result["final_tokens"][start:])),
                candidate_coverage=phases,
                expansion_newly_covered=sum(r["newly_covered"] for r in additions),
                retained_expansion_newly_covered=sum(r["newly_covered"] and r["retained_in_formal_state"] for r in additions),
                second_stage_seconds=result.get("second_stage_seconds"))
        cp_rows = [row for event in left.get("checkpoint_offline_evaluation", {}).get("checkpoint_events", [])
                   for row in event.get("candidate_coverage", [])]
        item["groups"]["checkpoint"]["checkpoint_coverage"] = dict(decisions=len(cp_rows),
            selected_pool_contains_gt=sum(r["selected_pool_contains_gt"] for r in cp_rows),
            old_pool_contains_gt=sum(r["old_pool_contains_gt"] for r in cp_rows),
            validly_scored_gt=sum(r["gt_validly_scored"] for r in cp_rows))
        samples.append(item)
    totals = {}
    for label in METHODS:
        rows = [s["groups"][label] for s in samples]
        count = sum(s["evaluated_token_count"] for s in samples)
        valid = [r["accuracy"] for r in rows if r["accuracy"] is not None]
        totals[label] = dict(sample_count=len(rows), evaluated_token_count=count,
            correct=sum(r["correct"] for r in rows),
            micro_accuracy=sum(r["correct"] for r in rows)/count if count else None,
            macro_accuracy=sum(valid)/len(valid) if valid else None,
            repaired=sum(r["repaired"] for r in rows), damaged=sum(r["damaged"] for r in rows),
            wrong_to_wrong_changes=sum(r["wrong_to_wrong_changes"] for r in rows))
        totals[label]["candidate_coverage"] = {}
        for phase in ("initial", "R"):
            combined = {key:sum(r["candidate_coverage"][phase][key] for r in rows)
                        for key in ("decisions", "contains_gt", "validly_scored_gt", "retained_decisions")}
            combined["coverage_rate"] = combined["contains_gt"]/combined["decisions"] if combined["decisions"] else None
            totals[label]["candidate_coverage"][phase] = combined
        totals[label]["expansion_newly_covered"] = sum(r["expansion_newly_covered"] for r in rows)
        totals[label]["retained_expansion_newly_covered"] = sum(r["retained_expansion_newly_covered"] for r in rows)
        if label == "checkpoint":
            totals[label]["checkpoint_coverage"] = {key:sum(r["checkpoint_coverage"][key] for r in rows)
                for key in ("decisions", "selected_pool_contains_gt", "old_pool_contains_gt", "validly_scored_gt")}
    return dict(baseline="checkpoint arm initial discrete output", equal_compute_budget=False,
                common_initial_correct=sum(s["common_initial_correct"] for s in samples),
                coverage_unit="candidate decision; prefixes and decision counts may differ between arms",
                samples=samples, totals=totals)


def read_expanded(run_dir, expected):
    for name in control.ARTIFACTS:
        if not (run_dir/name).is_file():
            raise ValueError("missing artifact: "+name)
    resolved = json.loads((run_dir/"resolved_config.json").read_text(encoding="utf-8"))
    if (resolved["advanced_method"]["name"] != METHODS["discretization"] or
            resolved["advanced_methods"]["suffix_reoptimization_v2_2_2_2"]["checkpoint_enabled"] is not False):
        raise ValueError("expanded artifact selector mismatch")
    records = [json.loads(line) for line in (run_dir/"reconstructions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != expected:
        raise ValueError("expanded artifact sample count mismatch")
    for record in records:
        result = record[KEYS["discretization"]]
        if (record["selected_advanced_method"] != METHODS["discretization"] or
                record["selected_candidate_reranking_method"] != "none" or
                result.get("gt_accessed") is not False or not result.get("formal_gt_blind") or "checkpoint" in result or
                not record["discretization_offline_evaluation"]["evaluated_after_online_return"]):
            raise ValueError("expanded artifact method/GT boundary mismatch")
        if result.get("anomaly_reasons"):
            raise ValueError("expanded arm contains failed R trials")
    return records


def run_bundle(project, runtime, result_root, python, model_path=None, smoke=False, run=subprocess.run):
    project, runtime = Path(project).resolve(), Path(runtime).resolve()
    plan = preflight(project, model_path, runtime)
    pending = [p for p in plan["pending_server_checks"] if not smoke or not p.startswith("dataset:")]
    if pending:
        raise ValueError("server preflight incomplete: "+repr(pending))
    gpu = os.environ.get("DEML_GPU_ID", "0")
    if not gpu.isdigit():
        raise ValueError("invalid DEML_GPU_ID")
    control.ensure_gpu_idle(gpu, run)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    bundle = Path(result_root).resolve()/tag
    bundle.mkdir(parents=True, exist_ok=False)
    temp_root = project/"outputs"
    temp_root.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="v222_2_", dir=temp_root))
    manifest = dict(status="running", run_kind="smoke" if smoke else "formal", groups={}, errors=[],
                    execution_order=list(METHODS), equal_compute_budget=False,
                    model_path=plan["model_path"], model_cache=plan["model_cache"], python=python)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1",
               HF_HOME=str(runtime/"hf-cache"), TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES=gpu)
    records = {}
    try:
        manifest["source_sha256"] = {str(path.relative_to(project)):digest(path) for path in
            [project/"invert.py", project/"experiment_outputs.py",
             project/"实验/环境和实验/内部文件/runner_suffix_v2_2_2_2.py",
             project/"suffix_optimization_methods/method_versions/suffix_reoptimization_v2_2_2.py",
             project/"suffix_optimization_methods/method_versions/suffix_reoptimization_v2_2_2_2.py"]}
        if smoke:
            data = temporary/"smoke.json"
            dump(data, ["A short smoke test sentence with enough tokens for two checkpoint segments."])
        for label in METHODS:
            config = dict(plan["configs"][label])
            config.update(base_model_name=plan["model_path"], log_dir=str(project/"results/invert_timestamp_runs"),
                          suffix_v222_snapshot_dir=str(bundle/"snapshots"),
                          suffix_v222_snapshot_mode="write" if label == "checkpoint" else "read",
                          suffix_v222_run_kind=manifest["run_kind"])
            if smoke:
                config.update(datasets=[dict(name="checkpoint_smoke",path=str(data),type="local",len=1)],
                              dataset_path=str(data),dataset_type="local",dataset_len=1,epoch=3,
                              suffix_v2_2_2_steps=3,suffix_v2_2_2_max_attempts=1,
                              suffix_v2_2_2_2_steps=3,suffix_v2_2_2_2_max_attempts=1)
            config_path = bundle/("effective_"+label+".json")
            dump(config_path,config)
            method_root = project/"results/invert_timestamp_runs"/METHODS[label]
            before = set(method_root.iterdir()) if method_root.exists() else set()
            with (bundle/"runner.log").open("a",encoding="utf-8") as log:
                completed=run([python,str(project/"invert.py"),"--config",str(config_path)],cwd=str(project),
                              env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
            if completed.returncode:
                raise RuntimeError(label+" arm exited "+str(completed.returncode))
            created=[p for p in set(method_root.iterdir())-before if p.is_dir() and (p/"reconstructions.jsonl").exists()]
            if len(created)!=1:
                raise ValueError("cannot identify one timestamp run for "+label)
            run_dir=created[0]
            records[label]=(control.read_artifacts(run_dir,True,1 if smoke else 12) if label=="checkpoint"
                            else read_expanded(run_dir,1 if smoke else 12))
            if not smoke and Counter(r["dataset"]["name"] for r in records[label]) != Counter(Skytrax=4,CMS=4,ECHR_Law=4):
                raise ValueError("formal dataset counts differ")
            manifest["groups"][label]=dict(run_dir=str(run_dir),config_sha256=digest(config_path),
                artifacts={name:digest(run_dir/name) for name in control.ARTIFACTS})
            dump(bundle/"manifest.json",manifest)
        dump(bundle/"comparison.json",compare_pair(records["checkpoint"],records["discretization"]))
        manifest.update(status="complete",snapshot_dir=str(bundle/"snapshots"))
    except Exception as error:
        manifest.update(status="failed",errors=[type(error).__name__+": "+str(error)])
        raise
    finally:
        if temporary.parent.resolve()!=temp_root.resolve() or not temporary.name.startswith("v222_2_"):
            raise RuntimeError("unexpected temporary path")
        shutil.rmtree(temporary)
        dump(bundle/"manifest.json",manifest)
    return bundle


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("dry-run","smoke","formal"))
    for name in ("project","runtime","result-root"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--model-path")
    args=parser.parse_args(argv)
    try:
        if args.mode=="dry-run":
            print(json.dumps(preflight(args.project,args.model_path,args.runtime),ensure_ascii=True,indent=2))
        else:
            print(run_bundle(args.project,args.runtime,args.result_root,args.python,args.model_path,args.mode=="smoke"))
        return 0
    except Exception as error:
        print(type(error).__name__+": "+str(error),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
