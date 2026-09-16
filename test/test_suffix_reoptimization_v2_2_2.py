import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_2_2 as cp
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_2_1 as old
from test.test_suffix_reoptimization_v2_2_1 import (
    _Model, _Tokenizer, _register_layer_hooks, _forward_tokens,
    _embedding_top_indices, _select_candidate, _get_perplexity,
)


ROOT = Path(__file__).resolve().parents[1]


class Model(_Model):
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False):
        assert use_cache is False
        return super().forward(input_ids, inputs_embeds, attention_mask)


class Tokenizer(_Tokenizer):
    bos_token_id = 0


def fixture(length=11, prefix=1, **config):
    model = Model().eval()
    tokens = ([0] if prefix else []) + [3+i%6 for i in range(length-prefix)]
    embedding = model.embed_tokens(torch.tensor([tokens], dtype=torch.long)).detach()
    mask = torch.ones((1, length), dtype=torch.long)
    target = _forward_tokens(model, tokens, mask, 0).detach()
    cfg = dict(enabled=True, checkpoint_enabled=True, steps=1, range_top_k=2, max_attempts=0)
    cfg.update(config)
    kwargs=dict(model=model, embed_layer=model.embed_tokens, optimized_embedding=embedding,
                target_hidden_state=target, attention_mask=mask, layer_id=0,
                register_layer_hooks=_register_layer_hooks, tokenizer=Tokenizer(),
                config=cp.SuffixReoptimizationV222Config(**cfg), fixed_prefix_tokens=[0] if prefix else [],
                eval_start_pos=prefix, top_k_ppl=2, top_k_cos=2,
                embedding_top_indices=_embedding_top_indices, select_candidate_from_top_indices=_select_candidate,
                get_perplexity=_get_perplexity, forward_and_get_last_hidden_state=_forward_tokens)
    return kwargs


class ContractTests(unittest.TestCase):
    def test_fixed_parameters_reject_changes(self):
        for field, value in (("checkpoint_diagnostic_tolerance", .01), ("checkpoint_candidate_min_cosine", .89),
                             ("checkpoint_forward_mode", "cached_prefix"), ("checkpoint_max_repairs", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                cp.SuffixReoptimizationV222Config(**{field:value})

    def test_missing_explicit_config_fails(self):
        with self.assertRaises(ValueError):
            cp.config_from_mapping({})

    def test_online_signature_has_no_private_labels(self):
        for function in (cp.run_suffix_reoptimization_v2_2_2, cp.run_checkpoint, cp.optimize_stage1):
            names=inspect.signature(function).parameters
            for forbidden in ("total_input_ids", "target_token_ids", "accuracy", "oracle"):
                self.assertNotIn(forbidden,names)

    def test_padding_and_private_prefix_rejected(self):
        kwargs=fixture()
        kwargs["attention_mask"][0,-1]=0
        with self.assertRaises(ValueError):
            cp.run_suffix_reoptimization_v2_2_2(**kwargs)
        kwargs=fixture()
        kwargs["fixed_prefix_tokens"]=[3]
        with self.assertRaises(ValueError):
            cp.run_suffix_reoptimization_v2_2_2(**kwargs)

    def test_segmentation_zero_budget_and_tail(self):
        for prefix in (0,1):
            for length in (0,1,4,5,6,9,10,11):
                with self.subTest(prefix=prefix,length=length):
                    kwargs=fixture(length+prefix,prefix)
                    _,result=cp.run_suffix_reoptimization_v2_2_2(**kwargs)
                    events=result["checkpoint"]["events"]
                    self.assertEqual(length//5,len(events))
                    self.assertEqual(length%5,result["checkpoint"]["tail_skipped_token_count"])
                    self.assertEqual([(prefix+g*5,prefix+g*5+4) for g in range(length//5)],[(e["a"],e["b"]) for e in events])

    def test_off_has_no_checkpoint_forward(self):
        with mock.patch.object(cp,"forward_discrete",side_effect=AssertionError("CP must be off")):
            _,result=cp.run_suffix_reoptimization_v2_2_2(**fixture(checkpoint_enabled=False))
        self.assertEqual([],result["checkpoint"]["events"])

    def test_old_new_off_equivalence(self):
        kwargs=fixture(max_attempts=1,checkpoint_enabled=False)
        old_config={k:v for k,v in vars(kwargs["config"]).items() if not k.startswith("checkpoint_")}
        old_kwargs=dict(kwargs,config=old.SuffixReoptimizationV221Config(**old_config))
        state=cp.rng_state()
        expected_emb, expected=old.run_suffix_reoptimization_v2_2_1(**old_kwargs)
        cp.restore_rng(state)
        actual_emb,actual=cp.run_suffix_reoptimization_v2_2_2(**kwargs)
        self.assertTrue(torch.equal(expected_emb,actual_emb))
        for key in ("final_tokens","attempt_count","accepted_round_count","pre_hidden_loss","final_hidden_loss"):
            self.assertEqual(expected[key],actual[key],key)

    def test_all_r_paths_reach_checkpoint(self):
        for mode in ("accept","reject","error","threshold"):
            kwargs=fixture(max_attempts=1)
            if mode=="threshold":
                kwargs["config"].trigger_mode="threshold"
                kwargs["config"].trigger_threshold=-1
            patch=mock.patch.object(cp,"_optimize_suffix",side_effect=RuntimeError("local trial")) if mode=="error" else mock.patch.object(
                cp,"_optimize_suffix",return_value=(kwargs["optimized_embedding"].clone(),1.,.5 if mode=="accept" else 1.,{}))
            with patch:
                _,result=cp.run_suffix_reoptimization_v2_2_2(**kwargs)
            self.assertEqual(2,len(result["checkpoint"]["events"]))


class DiagnosticTests(unittest.TestCase):
    def test_trajectory_cases(self):
        cases=[([.98,.86,.96,.82,.81],3),([.88,.76,.87,.87,.86],1),
               ([.98,.95,.92,.89,.80],None),([.85]*5,None),([.97,.84,.86,.88,.89],1),
               ([.95,.8,.82,.83,.94],None)]
        for values,position in cases:
            self.assertEqual(position,cp.diagnose_segment(values)["selected_position"])

    def test_exact_diagnostic_boundaries(self):
        # Binary-exact tolerance isolates the comparator from decimal rounding.
        self.assertIsNone(cp.diagnose_segment([1,.875,.875,.875,.875],eta=.125)["selected_position"])
        self.assertEqual("isolated_drop",cp.diagnose_segment([1,.75,.875,.875,.875],eta=.125)["diagnosis_type"])

    def test_sum_cosine_is_not_mean_cosine(self):
        a=torch.tensor([[[10.,0.],[0.,1.]]])
        b=torch.tensor([[[1.,0.],[1.,0.]]])
        self.assertGreater(cp.segment_cosine(a,b),.9)
        self.assertIsNone(cp.segment_cosine(torch.zeros_like(a),b))

    def test_threshold_filter_duplicates_and_current_retention(self):
        table=dict(candidate_token_ids=[0,1,2,2,3,4,5],candidate_hidden_cosine=[1.,.899,.9,.95,float('nan'),.99,.92])
        ids,excluded,mapping=cp.filter_existing_candidates(table,4,Tokenizer(),12,True,.9)
        self.assertEqual([2,5],ids)
        self.assertEqual(mapping[2],mapping[3])
        self.assertIn("nonfinite_old_score",[e["reason"] for e in excluded])


class TransactionTests(unittest.TestCase):
    def test_contract_errors_are_fatal(self):
        with self.assertRaises(cp.CheckpointContractError):
            cp.raise_if_fatal(cp.CheckpointContractError("wrong exact-layer shape"))

    def test_tuple_hook_extracts_hidden_only(self):
        kwargs=fixture(6)
        block=kwargs["model"].layers[0]
        original=block.forward
        def tuple_forward(hidden):return (original(hidden), "cache sentinel")
        def model_forward(input_ids=None,attention_mask=None,use_cache=False):
            return block(kwargs["embed_layer"](input_ids))[0]
        with mock.patch.object(block,"forward",side_effect=tuple_forward), \
             mock.patch.object(kwargs["model"],"forward",side_effect=model_forward):
            hidden=cp.forward_discrete(kwargs["model"],[0,3,4,5,6,7],0,_register_layer_hooks)
        self.assertEqual((1,6,4),tuple(hidden.shape))
        self.assertEqual(0,len(block._forward_hooks))

    def setup_trial(self):
        kwargs=fixture(6)
        tokens=[0,3,4,5,6,7]
        table={2:dict(candidate_token_ids=[4,8,9],candidate_hidden_cosine=[.99,.9,.95],
                      prefix_fingerprint=cp.prefix_fingerprint(tokens[:2]),generation=0)}
        args=(kwargs["model"],kwargs["embed_layer"],tokens,kwargs["optimized_embedding"],
              kwargs["target_hidden_state"],0,_register_layer_hooks,Tokenizer(),kwargs["config"],table,1,5,1)
        return args

    def test_candidate_failure_rolls_back_even_after_improvement(self):
        args=self.setup_trial()
        original=list(args[2]); embedding=args[3].clone()
        with mock.patch.object(cp,"diagnose_segment",return_value=dict(selected_position=2)), \
             mock.patch.object(cp,"segment_cosine",side_effect=[.8]+[.8]*5+[.95]), \
             mock.patch.object(cp,"forward_discrete",side_effect=[args[4],args[4],RuntimeError("trial failed")]):
            event=cp.run_checkpoint(*args)
        self.assertEqual("candidate_forward_failed",event["reason"])
        self.assertEqual(original,args[2]); self.assertTrue(torch.equal(embedding,args[3]))
        self.assertFalse(event["accepted"]); self.assertEqual(.8,event["group_cosine_after"])

    def test_best_later_candidate_commits_only_one_id(self):
        args=self.setup_trial(); original=list(args[2]); old_embedding=args[3].clone()
        with mock.patch.object(cp,"diagnose_segment",return_value=dict(selected_position=2)), \
             mock.patch.object(cp,"segment_cosine",side_effect=[.8]+[.8]*5+[.81,.86]):
            event=cp.run_checkpoint(*args)
        self.assertEqual("accepted",event["reason"])
        self.assertEqual(9,args[2][2]); self.assertEqual(1,event["repair_attempt_count"])
        self.assertEqual([2],[i for i,(a,b) in enumerate(zip(original,args[2])) if a!=b])
        self.assertTrue(torch.equal(old_embedding[:,3:],args[3][:,3:]))
        json.dumps(event,allow_nan=False)

    def test_invalid_candidate_and_equal_score_do_not_commit(self):
        for score,reason in ((None,"invalid_candidate_score"),(.8,"no_improvement")):
            args=self.setup_trial(); original=list(args[2])
            with mock.patch.object(cp,"diagnose_segment",return_value=dict(selected_position=2)), \
                 mock.patch.object(cp,"segment_cosine",side_effect=[.8]+[.8]*5+[score,score]):
                event=cp.run_checkpoint(*args)
            self.assertEqual(reason,event["reason"]); self.assertEqual(original,args[2])

    def test_stale_and_rejected_tables_are_not_reused(self):
        args=self.setup_trial(); args[9][2]["prefix_fingerprint"]="rejected state"
        with mock.patch.object(cp,"diagnose_segment",return_value=dict(selected_position=2)), \
             mock.patch.object(cp,"segment_cosine",return_value=.8):
            self.assertEqual("missing_or_stale_candidate_table",cp.run_checkpoint(*args)["reason"])

    def test_full_prefix_is_causal_and_hooks_removed(self):
        kwargs=fixture()
        before=[0,3,4,5,6,7]; after=[0,3,8,5,6,7]
        a=cp.forward_discrete(kwargs["model"],before,0,_register_layer_hooks)
        b=cp.forward_discrete(kwargs["model"],after,0,_register_layer_hooks)
        self.assertTrue(torch.equal(a[:,:2],b[:,:2]))
        self.assertFalse(torch.equal(a[:,3:],b[:,3:]))
        with mock.patch.object(kwargs["model"],"forward",side_effect=RuntimeError("fail")):
            with self.assertRaises(RuntimeError):
                cp.forward_discrete(kwargs["model"],before,0,_register_layer_hooks)
        self.assertEqual(0,len(kwargs["model"].layers[0]._forward_hooks))

    def test_repair_refreshes_future_without_r_budget(self):
        kwargs=fixture(11,max_attempts=0)
        real=cp.run_checkpoint
        def repair(*args,**kw):
            event=real(*args,**kw)
            if args[10]==1:
                args[2][2]=10
                event["accepted"]=True
                event["reason"]="accepted"
            return event
        with mock.patch.object(cp,"run_checkpoint",side_effect=repair):
            _,result=cp.run_suffix_reoptimization_v2_2_2(**kwargs)
        self.assertEqual(5,result["checkpoint"]["future_refresh_count"])
        self.assertEqual(10,result["final_tokens"][2])


class Stage1Tests(unittest.TestCase):
    def test_shared_snapshot_runs_stage1_once_and_restores_rng(self):
        kwargs=fixture(6,max_attempts=0)
        first=kwargs["optimized_embedding"].clone()
        stage2=dict(kwargs); stage2.pop("optimized_embedding")
        stage1=dict(initial_embedding=first)
        temp_root=ROOT/"outputs/checkpoint_v222_impl"
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp, \
             mock.patch.object(cp,"optimize_stage1",return_value=(first,{"elapsed_seconds":0.})) as optimize:
            args=dict(stage1_kwargs=stage1,stage2_kwargs=stage2,snapshot_path=str(Path(temp)/"s.pt"),
                      pair_id="sample:0",snapshot_contract={"block":0})
            a,ra=cp.run_two_stage(**args,snapshot_mode="write")
            torch.rand(50)
            b,rb=cp.run_two_stage(**args,snapshot_mode="read")
            self.assertEqual(1,optimize.call_count)
            self.assertTrue(torch.equal(a,b))
            self.assertEqual(ra["stage1_snapshot_sha256"],rb["stage1_snapshot_sha256"])
            self.assertTrue(rb["stage1_reused"])


if __name__ == "__main__":
    unittest.main()
