"""Execute entry/config and legacy-loop AST without optional HF imports/models."""
import ast
import io
import json
from pathlib import Path
import time
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from experiment_outputs import checkpoint_offline_evaluation, _resolved_suffix_v222_config, extract_experiment_stage_summary, write_experiment_sample_summary
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_2_2 as cp
from test.test_suffix_reoptimization_v2_2_2 import fixture, _register_layer_hooks
from test.test_suffix_v2_2_2_runner import runner

ROOT=Path(__file__).resolve().parents[1]
TREE=ast.parse((ROOT/"invert.py").read_text(encoding="utf-8"))


def load_functions(*names,extra=None):
    namespace=dict(extra or {})
    nodes=[node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/"invert.py"),"exec"),namespace)
    return namespace


class IntegrationTests(unittest.TestCase):
    def test_offline_direct_and_final_effects_are_separate(self):
        result={"final_tokens":[0,3,8,5,6,7],"checkpoint":{"events":[dict(
            checkpoint_id=0,a=1,b=5,accepted=True,triggered=True,selected_position=2,
            old_token_id=4,new_token_id=8,segment_tokens_before=[3,4,5,6,7])]}}
        evaluation=checkpoint_offline_evaluation(result,[0,3,4,5,6,7],1)
        self.assertEqual(1,evaluation["direct_damage"])
        self.assertEqual(4,evaluation["correct_token_count"])
        self.assertEqual(0,evaluation["checkpoint_events"][0]["segment_errors_before"])
        self.assertNotIn("reference_ids",result)

    def test_exact_target_branch_and_tuple_hook(self):
        kwargs=fixture(6)
        token_ids=torch.tensor([[0,3,4,5,6,7]])
        class Tokenizer:
            def __call__(self,*args,**kw):return {"input_ids":token_ids,"attention_mask":torch.ones_like(token_ids)}
        functions=load_functions("get_hidden_state",extra={
            "torch":torch,"get_model_device":lambda model:torch.device("cpu"),
            "get_input_embedding_layer":lambda model:model.embed_tokens,
            "register_layer_hooks":_register_layer_hooks,
            "suffix_v222_forward_discrete":cp.forward_discrete})
        ids,mask,emb,hidden=functions["get_hidden_state"](Tokenizer(),kwargs["model"],0,prompt="fixture",
                                                            selected_advanced_method=runner.METHOD)
        self.assertTrue(torch.equal(hidden[0],kwargs["target_hidden_state"]))
        self.assertTrue(torch.equal(ids,token_ids))

    def test_offline_gt_changes_do_not_enter_online_sidecar(self):
        kwargs=fixture(6)
        state=cp.rng_state()
        _,first=cp.run_suffix_reoptimization_v2_2_2(**kwargs)
        original=json.dumps(first,sort_keys=True)
        checkpoint_offline_evaluation(first,[0,3,4,5,6,7],1)
        checkpoint_offline_evaluation(first,[0,8,8,8,8,8],1)
        self.assertEqual(original,json.dumps(first,sort_keys=True))
        cp.restore_rng(state)
        _,second=cp.run_suffix_reoptimization_v2_2_2(**kwargs)
        self.assertEqual(first["final_tokens"],second["final_tokens"])
        self.assertEqual([e["reason"] for e in first["checkpoint"]["events"]],
                         [e["reason"] for e in second["checkpoint"]["events"]])

    def test_selector_aliases_enabled_and_rollback(self):
        functions=load_functions("normalize_suffix_version","select_advanced_method")
        normalize=functions["normalize_suffix_version"]
        select=functions["select_advanced_method"]
        disabled=types.SimpleNamespace(enabled=False)
        enabled=types.SimpleNamespace(enabled=True)
        for alias in ("v2.2.2","2.2.2","suffix_reoptimization_v2_2_2"):
            self.assertEqual("v2.2.2",normalize(alias))
            self.assertEqual(runner.METHOD,select(alias,disabled,disabled,disabled,disabled,suffix_reopt_v2_2_2_config=enabled))
        with self.assertRaises(ValueError):select("v2.2.2",disabled,disabled,disabled,disabled,suffix_reopt_v2_2_2_config=disabled)
        self.assertEqual("suffix_reoptimization_v2.2.1",select("v2.2.1",disabled,disabled,disabled,disabled,suffix_reopt_v2_2_1_config=enabled))

    def test_config_registry_and_fixed_summary(self):
        config=runner.load_config(ROOT/runner.CONFIGS["on"])
        parsed=cp.config_from_mapping(config)
        resolved=_resolved_suffix_v222_config(types.SimpleNamespace(**config))
        self.assertTrue(parsed.checkpoint_enabled)
        self.assertEqual(.05,resolved["checkpoint_diagnostic_tolerance"])
        record=dict(selected_advanced_method=runner.METHOD,selected_candidate_reranking_method="none",
                    accuracy=.8,suffix_reoptimization_v2_2_2_result=dict(pre_acc=.5,post_acc=.8))
        summary=extract_experiment_stage_summary(record)
        buffer=io.StringIO()
        write_experiment_sample_summary(buffer,record,1,1,10)
        self.assertNotIn("checkpoint",buffer.getvalue())
        self.assertNotIn("candidate",buffer.getvalue())
        self.assertTrue(summary)

    def test_stage1_matches_actual_legacy_loop(self):
        kwargs=fixture(6)
        model=kwargs["model"]
        initial=kwargs["optimized_embedding"][:,1:].clone()
        prefix=kwargs["optimized_embedding"][:,:1].clone()
        args=types.SimpleNamespace(clip=True,optim_method="cosine",num_invert_layers=0,
                                   invert_method="cosine",filter_nonascii=True)
        main=next(n for n in TREE.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="epoch_idx")
        ns=dict(torch=torch,F=F,args=args,use_v1_4_coarse_stage=False,part_epoch=2,
                new_input_embed_0=initial.clone().requires_grad_(True),prefix_embed=prefix,
                fixed_prefix=True,lr=.1,alpha=.001,model=model,
                target_attention_mask=kwargs["attention_mask"],next_hidden_states_last=kwargs["target_hidden_state"],
                register_layer_hooks=_register_layer_hooks,loss_func=torch.nn.MSELoss(),
                weight_mask=torch.ones(6),right_range=torch.ones(4),
                selected_advanced_method="suffix_reoptimization_v2.2.1",
                epochs=[],loss_lst=[],cos_sim_lst=[],last_optimization_percent=0,
                console_finish_progress=lambda:None,console_update=lambda *x:None,
                console_safe_text=str,format_progress_bar=lambda x:"",sample_idx=0,time=time,start=time.time(),
                tokenizer=kwargs["tokenizer"],embed_layer=kwargs["embed_layer"],
                total_input_ids=None,eval_start_pos=1,invert_embedding=lambda *a,**kw:(None,None,None))
        exec(compile(ast.Module(body=[loop],type_ignores=[]),str(ROOT/"invert.py"),"exec"),ns)
        actual,summary=cp.optimize_stage1(model,initial,prefix,kwargs["target_hidden_state"],kwargs["attention_mask"],
                                        0,_register_layer_hooks,torch.ones(6),torch.ones(4),.1,2,.001,True,"cosine")
        self.assertTrue(torch.equal(ns["final_input_embed"],actual))
        self.assertEqual(2,summary["completed_steps"])


if __name__ == "__main__":unittest.main()
