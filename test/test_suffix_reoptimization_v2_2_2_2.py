"""Candidate expansion, fresh R trajectories, and cross-version Stage-1 pairing."""
import copy
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from experiment_outputs import discretization_offline_evaluation, build_stage_accuracy, write_experiment_sample_summary
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_2_2 as cp
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_2_2_2 as expanded
from test.test_suffix_reoptimization_v2_2_2 import fixture, Design4Tokenizer, _register_layer_hooks
from test.test_suffix_reoptimization_v2_2_2_integration import load_functions

ROOT=Path(__file__).resolve().parents[1]
RUNNER_PATH=ROOT/"实验/环境和实验/内部文件"
sys.path.insert(0,str(RUNNER_PATH))
import runner_suffix_v2_2_2_2 as runner
sys.path.pop(0)


def arguments(attempts=0):
    kwargs=fixture(6)
    kwargs["tokenizer"]=Design4Tokenizer()
    kwargs["config"]=expanded.SuffixReoptimizationV222_2Config(enabled=True,steps=3,max_attempts=attempts,range_top_k=2)
    return kwargs


class ExpansionTests(unittest.TestCase):
    def test_real_generator_merge_winner_and_next_prefix(self):
        kwargs=arguments()
        prefixes=[]
        def ppl(ids,*args,**kw):
            prefixes.append(list(ids))
            return None,torch.tensor([7])
        def forward(model,ids,*args,**kw):
            ids=torch.tensor(ids)
            scores=torch.zeros((*ids.shape,4))
            scores[:,:,0]=(ids==8).float()
            scores[:,:,1]=(ids!=8).float()
            return scores
        kwargs["target_hidden_state"]=torch.tensor([[[1.,0,0,0]]*6])
        kwargs["get_perplexity"]=ppl
        kwargs["forward_and_get_last_hidden_state"]=forward
        trajectory={i:dict(call_id="stage1",stages=[dict(step=1,token_ids=[8]),dict(step=2,token_ids=[9])]) for i in range(1,6)}
        kwargs["stage_candidates"]=trajectory
        with mock.patch.object(expanded,"_candidate_token_ids",return_value=(1,[1,1,7])):
            _,result=expanded.run_suffix_reoptimization_v2_2_2_2(**kwargs)
        events=result["candidate_expansion"]["events"]
        self.assertEqual([0,8,8,8,8,8],result["final_tokens"])
        self.assertEqual([0,8],prefixes[1])
        for event,table in zip(events,result["initial_candidate_rerank"]):
            self.assertEqual([1,1,7,7],event["original_candidate_ids"])
            self.assertEqual([1,1,7,7],table["candidate_token_ids"][:4])
            self.assertIn(8,event["added_candidate_ids"])
            self.assertLessEqual(len(event["added_candidate_ids"]),6)
            self.assertEqual(len(set(event["added_candidate_ids"])),len(event["added_candidate_ids"]))
            self.assertTrue(set(event["added_candidate_ids"]).isdisjoint(event["original_candidate_ids"]))
        self.assertNotIn("checkpoint",result)

    def test_ties_preserve_original_and_nonfinite_additions_are_skipped(self):
        for invalid in (False,True):
            kwargs=arguments()
            def forward(model,ids,*args,**kw):
                states=torch.ones((len(ids),len(ids[0]),4))
                if invalid:
                    for index,row in enumerate(ids):
                        if 8 in row: states[index]=float('nan')
                return states
            kwargs["forward_and_get_last_hidden_state"]=forward
            pool=dict(candidates=[dict(token_id=8,quota_source="case")])
            with mock.patch.object(expanded,"_candidate_token_ids",return_value=(1,[1])), \
                 mock.patch.object(expanded,"build_checkpoint_candidates",return_value=pool):
                _,result=expanded.run_suffix_reoptimization_v2_2_2_2(**kwargs)
            self.assertTrue(all(e["selected_token_id"]==e["original_selected_token_id"] for e in result["candidate_expansion"]["events"]))

    def test_r_uses_own_completed_trajectory_and_rejection_provenance(self):
        kwargs=arguments(1)
        real=expanded._optimize_suffix
        def reject(*args,**kw):
            embedding,pre,post,summary=real(*args,**kw)
            return embedding,pre,pre,summary
        with mock.patch.object(expanded,"_optimize_suffix",side_effect=reject), \
             mock.patch.object(expanded,"_rerank_positions",wraps=expanded._rerank_positions) as rerank:
            _,result=expanded.run_suffix_reoptimization_v2_2_2_2(**kwargs)
        self.assertEqual([],result["anomaly_reasons"])
        self.assertEqual(2,rerank.call_count)
        trajectory=rerank.call_args_list[1].kwargs["stage_candidates"]
        self.assertTrue(trajectory)
        for row in trajectory.values():
            self.assertEqual("R_1",row["call_id"])
            self.assertEqual([1,2],[s["step"] for s in row["stages"]])
        trials=[e for e in result["candidate_expansion"]["events"] if e["phase"]=="R_1"]
        self.assertTrue(trials)
        self.assertTrue(all(not e["retained_in_formal_state"] for e in trials))
        self.assertEqual(result["pre_tokens"],result["final_tokens"])

    def test_cross_version_real_stage1_snapshot_is_reused_and_target_checked(self):
        kwargs=arguments(1)
        initial=kwargs.pop("optimized_embedding")
        stage1=dict(model=kwargs["model"],initial_embedding=initial[:,1:],prefix_embedding=initial[:,:1],
            target_hidden_state=kwargs["target_hidden_state"],attention_mask=kwargs["attention_mask"],
            layer_id=0,register_layer_hooks=_register_layer_hooks,weight_mask=torch.ones(6),
            right_range=torch.ones(4),lr=.01,epoch=3,alpha=.001)
        original=dict(kwargs,config=cp.SuffixReoptimizationV222Config(enabled=True,checkpoint_enabled=True,
            steps=3,max_attempts=1,range_top_k=2))
        cp.calibrate_checkpoint(kwargs["model"],kwargs["embed_layer"],kwargs["tokenizer"],0,_register_layer_hooks,original["config"])
        with tempfile.TemporaryDirectory(prefix="v222_2_test_",dir=ROOT/"outputs") as directory:
            shared=dict(stage1_kwargs=stage1,snapshot_path=str(Path(directory)/"s.pt"),pair_id="tiny:0",snapshot_contract={"block":0})
            _,a=cp.run_two_stage(**shared,stage2_kwargs=original,snapshot_mode="write")
            with mock.patch.object(expanded,"optimize_stage1",side_effect=AssertionError("must reuse Stage-1")):
                _,b=expanded.run_two_stage(**shared,stage2_kwargs=kwargs,snapshot_mode="read")
            self.assertEqual(a["stage1_snapshot_sha256"],b["stage1_snapshot_sha256"])
            self.assertEqual([1,2],b["stage1"]["checkpoint_trajectory"]["captured_steps"])
            self.assertFalse(a["stage1_reused"])
            self.assertTrue(b["stage1_reused"])
            bad=dict(kwargs,target_hidden_state=kwargs["target_hidden_state"]+1)
            with self.assertRaisesRegex(ValueError,"observation differs"):
                expanded.run_two_stage(**shared,stage2_kwargs=bad,snapshot_mode="read")
        json.dumps(b,allow_nan=False)


class WiringTests(unittest.TestCase):
    def test_fixed_log_summary_has_no_added_diagnostics(self):
        record=dict(selected_advanced_method=expanded.METHOD_NAME,selected_candidate_reranking_method="none",
                    accuracy=.75,suffix_reoptimization_v2_2_2_2_result=dict(pre_acc=.5,post_acc=.75))
        self.assertEqual({"pre_suffix":.5,"suffix_v2_2_2_2":.75},build_stage_accuracy(record))
        buffer=io.StringIO()
        write_experiment_sample_summary(buffer,record,1,1,4)
        self.assertNotIn("candidate",buffer.getvalue())
        self.assertNotIn("expansion",buffer.getvalue())

    def test_preflight_selector_and_config(self):
        plan=runner.preflight(ROOT,model_path=ROOT/"models/Qwen2.5-1.5B")
        self.assertEqual(["checkpoint","discretization"],plan["groups"])
        cfg=expanded.config_from_mapping(plan["configs"]["discretization"])
        self.assertTrue(cfg.enabled)
        funcs=load_functions("normalize_suffix_version","select_advanced_method")
        disabled=types.SimpleNamespace(enabled=False)
        for alias in ("v2.2.2(2)","suffix_reoptimization_v2_2_2_2"):
            self.assertEqual(expanded.METHOD_NAME,funcs["select_advanced_method"](alias,disabled,disabled,disabled,disabled,
                             suffix_reopt_v2_2_2_2_config=cfg))
        with self.assertRaises(ValueError):
            expanded.SuffixReoptimizationV222_2Config(checkpoint_enabled=True)

    def test_common_baseline_repairs_and_pair_mismatch(self):
        def record(label,pre,final):
            result=dict(pre_tokens=pre,final_tokens=final,stage1_snapshot_sha256="shared",stage1_reused=label=="discretization")
            return dict(pair_id="tiny:0",dataset={"name":"tiny"},**{runner.KEYS[label]:result},
                        discretization_offline_evaluation=discretization_offline_evaluation(result,[0,1,2,3],1))
        a=record("checkpoint",[0,1,8,9],[0,1,2,9])
        b=record("discretization",[0,8,2,3],[0,8,2,3])
        comparison=runner.compare_pair([a],[b])
        self.assertEqual(2,comparison["totals"]["discretization"]["repaired"])
        self.assertEqual(1,comparison["totals"]["discretization"]["damaged"])
        for where,key in ((b,"pair_id"),(b[runner.KEYS["discretization"]],"stage1_snapshot_sha256"),
                          (b["discretization_offline_evaluation"],"reference_sha256")):
            previous=where[key]; where[key]="wrong"
            with self.assertRaises(ValueError): runner.compare_pair([a],[b])
            where[key]=previous

    def test_bundle_two_groups_share_snapshots_and_failure_stops(self):
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(prefix="v222_2_test_",dir=ROOT/"outputs") as directory:
                project=Path(directory)
                plan=runner.preflight(ROOT,model_path=project/"unused_model")
                plan["pending_server_checks"]=[]
                calls=[]
                emitted={}
                def run(command,**kw):
                    config=runner.control.load_config(command[-1]); calls.append(config)
                    if fail: return types.SimpleNamespace(returncode=9)
                    label=list(runner.METHODS)[len(calls)-1]
                    snapshot=Path(config["suffix_v222_snapshot_dir"])/"fixture.pt"
                    if label=="checkpoint":
                        snapshot.parent.mkdir(parents=True)
                        snapshot.write_bytes(b"shared")
                    else:
                        self.assertEqual(b"shared",snapshot.read_bytes())
                    result=dict(pre_tokens=[0,8,2],final_tokens=[0,1,2],stage1_snapshot_sha256="shared",
                                formal_gt_blind=True,gt_accessed=False,stage1_reused=label=="discretization")
                    record=dict(pair_id="smoke:0",dataset={"name":"smoke"},selected_advanced_method=runner.METHODS[label],
                        selected_candidate_reranking_method="none",**{runner.KEYS[label]:result},
                        discretization_offline_evaluation=discretization_offline_evaluation(result,[0,1,2],1))
                    emitted[label]=[record]
                    output=project/"results/invert_timestamp_runs"/runner.METHODS[label]/"timestamp"
                    output.mkdir(parents=True)
                    runner.dump(output/"resolved_config.json",dict(advanced_method=dict(name=runner.METHODS[label]),
                        advanced_methods={"suffix_reoptimization_v2_2_2_2":dict(checkpoint_enabled=False)}))
                    (output/"experiment.log").write_text("summary",encoding="utf-8")
                    (output/"reconstructions.jsonl").write_text(json.dumps(record)+"\n",encoding="utf-8")
                    return types.SimpleNamespace(returncode=0)
                with mock.patch.object(runner,"preflight",return_value=plan), \
                     mock.patch.object(runner.control,"ensure_gpu_idle"), \
                     mock.patch.object(runner,"digest",return_value="hash"), \
                     mock.patch.object(runner.control,"read_artifacts",side_effect=lambda *args:emitted["checkpoint"]):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            runner.run_bundle(project,project/"runtime",project/"bundle","python",smoke=True,run=run)
                        self.assertEqual(1,len(calls))
                    else:
                        bundle=runner.run_bundle(project,project/"runtime",project/"bundle","python",smoke=True,run=run)
                        self.assertEqual(["write","read"],[c["suffix_v222_snapshot_mode"] for c in calls])
                        self.assertEqual(calls[0]["suffix_v222_snapshot_dir"],calls[1]["suffix_v222_snapshot_dir"])
                        self.assertEqual("complete",json.loads((bundle/"manifest.json").read_text(encoding="utf-8"))["status"])
                        self.assertTrue((bundle/"comparison.json").is_file())
                self.assertEqual([],list((project/"outputs").iterdir()))


if __name__=="__main__":
    unittest.main()
