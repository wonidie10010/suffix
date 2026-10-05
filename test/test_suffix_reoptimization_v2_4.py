"""Search outcomes, actual forward budgets, rollback, calibration and wiring."""
import ast
import copy
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import torch

from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_3 as v23
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_4 as v24
from experiment_outputs import (build_resolved_config, build_stage_accuracy,
    discretization_offline_evaluation, write_experiment_sample_summary)
from test.test_suffix_reoptimization_v2_3 import Tokenizer, fixture as real_fixture, functions
from test.test_suffix_reoptimization_v2_2_1 import _forward_tokens, _register_layer_hooks

ROOT = Path(__file__).resolve().parents[1]


class PathModel(torch.nn.Module):
    """Causal hidden-score fixture whose helpers execute real model forwards."""
    def __init__(self, vocabulary, scorer):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocabulary, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.scorer, self.mode = scorer, 'initial'
        self.eval()
        with torch.no_grad():
            self.embed_tokens.weight.zero_()
            self.embed_tokens.weight[:, 0] = torch.arange(vocabulary)/100
            self.embed_tokens.weight[:, 1] = 1
        for parameter in self.parameters(): parameter.requires_grad_(False)
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        del attention_mask
        if input_ids is None:
            hidden = inputs_embeds
        else:
            rows = []
            for tokens in input_ids.tolist():
                row = []
                for position in range(len(tokens)):
                    score = float(self.scorer(tokens[:position+1], position, self.mode))
                    row.append([score, max(0, 1-score*score)**.5, 0, 0])
                rows.append(row)
            hidden = torch.tensor(rows, dtype=torch.float32)
        for layer in self.layers: hidden = layer(hidden)
        return types.SimpleNamespace(logits=torch.zeros((*hidden.shape[:2], self.embed_tokens.num_embeddings)))


def frozen(model, tokenizer, fit=.1, first=.1):
    identity = v24.trajectory_identity(model, model.embed_tokens, tokenizer)
    observations = [dict(sample_id='cal:only', has_prefix=True, top1_score=fit),
                    dict(sample_id='cal:only', has_prefix=False, top1_score=first)]
    return v24.build_backtracking_calibration(observations,
        [dict(sample_id='cal:only', window_errors=[.25, .25])], ['cal:only'],
        v24.backtracking_inference_signature(model, identity, 0))


def path_fixture(scorer, *, length=6, prefix=1, fit=.1, first=.1, calibrated=True, **overrides):
    tokenizer = Tokenizer()
    model = PathModel(len(tokenizer.texts), scorer)
    ids = ([0] if prefix else [])+[1]*(length-prefix)
    embedding = model.embed_tokens(torch.tensor([ids])).detach()
    target = torch.zeros_like(embedding)
    target[:, :, 0] = 1
    values = dict(enabled=True, max_attempts=0, seed_top_k=0,
        case_quota=0, space_quota=0, prefix_quota=0, suffix_quota=0,
        backtracking_calibration=frozen(model, tokenizer, fit, first) if calibrated else {})
    values.update(overrides)
    prefixes = []
    def ppl(tokens, selected_model, layer_id, top_k):
        prefixes.append(list(tokens))
        selected_model(input_ids=torch.tensor([tokens]))
        return torch.ones(1), [3] if 2 in tokens else [1]
    return dict(model=model, embed_layer=model.embed_tokens, optimized_embedding=embedding,
        target_hidden_state=target, attention_mask=torch.ones((1, length), dtype=torch.long),
        layer_id=0, register_layer_hooks=_register_layer_hooks, tokenizer=tokenizer,
        config=v24.SuffixReoptimizationV24Config(**values), fixed_prefix_tokens=[0] if prefix else [],
        eval_start_pos=prefix, top_k_cos=3, top_k_ppl=1,
        embedding_top_indices=lambda *_:torch.tensor([1,2,3]),
        select_candidate_from_top_indices=lambda ids,*_:(1,ids.tolist()),
        get_perplexity=ppl, forward_and_get_last_hidden_state=_forward_tokens), prefixes


def immediate(tokens, position, mode):
    if position == 1: return {1:.9,2:.8995,3:.6}.get(tokens[-1],.1)
    if position in (2,3):
        return ({1:.7,2:.5,3:.99} if tokens[1]==2 else {1:.3,2:.2995,3:.1}).get(tokens[-1],.1)
    return {1:.99,2:.5,3:.3}.get(tokens[-1],.1)


def farther(tokens, position, mode):
    if position == 3:
        return ({1:.999,2:.8,3:.7} if tokens[2]==2 else {1:.9,2:.8995,3:.6}).get(tokens[-1],.1)
    if position in (4,5): return {1:.99 if tokens[2]==2 else .7,2:.5,3:.4}.get(tokens[-1],.1)
    return {1:.95,2:.8,3:.7}.get(tokens[-1],.1)


class CalibrationTests(unittest.TestCase):
    def test_independent_quantiles_empty_prefix_and_repeated_numerical_drift(self):
        rows = [dict(sample_id='cal:0',has_prefix=True,top1_score=s) for s in (.2,.6,1.)]
        rows.append(dict(sample_id='cal:0',has_prefix=False,top1_score=.9))
        payload = v24.build_backtracking_calibration(rows,
            [dict(sample_id='cal:0',window_errors=[1.,1.0001])],['cal:0'],{'model':'tiny'})
        self.assertAlmostEqual(.24,payload['fit_threshold'])
        self.assertAlmostEqual(.9,payload['first_fit_threshold'])
        self.assertGreater(payload['acceptance_epsilon'],.0001)
        self.assertTrue(payload['fitted_without_reference_tokens'])
    def test_missing_calibration_retains_greedy_and_reports_skips(self):
        kwargs,_ = path_fixture(immediate,calibrated=False)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertEqual([0,1,1,1,1,1],result['final_tokens'])
        self.assertFalse(result['backtracking']['calibration_ready'])
        self.assertEqual(0,result['backtracking']['compute']['forward_calls'])
        self.assertEqual({'calibration_missing'},{e['reason'] for e in result['backtracking']['events']})
    def test_frozen_settings_calibration_split_and_mutation_are_enforced(self):
        kwargs,_ = path_fixture(immediate)
        with self.assertRaises(v24.DiscretizationContractError):
            v24.run_suffix_reoptimization_v2_4(**dict(kwargs,sample_id='cal:only'))
        kwargs['model'].config = types.SimpleNamespace(use_cache=False)
        with self.assertRaises(v24.DiscretizationContractError): v24.run_suffix_reoptimization_v2_4(**kwargs)
        payload = copy.deepcopy(kwargs['config'].backtracking_calibration)
        payload['fit_threshold'] = .2
        with self.assertRaises(ValueError): v24.SuffixReoptimizationV24Config(backtracking_calibration=payload)
        with self.assertRaises(ValueError): v24.SuffixReoptimizationV24Config(backtracking_acceptance_epsilon=0.)
    def test_duplicate_ids_no_second_place_and_low_fit_priority(self):
        config = v24.SuffixReoptimizationV24Config()
        row = dict(base_candidates=[dict(token_id=1,score=.9),dict(token_id=1,score=.9)],variant_candidates=[])
        self.assertIsNone(v24._backtracking_signal(row,1,config)['type'])
        row['variant_candidates'] = [dict(token_id=2,score=.8995)]
        self.assertEqual('small_gap',v24._backtracking_signal(row,1,config)['type'])
        kwargs,_ = path_fixture(immediate,fit=.95)
        self.assertEqual('low_fit',v24._backtracking_signal(row,1,kwargs['config'])['type'])
        row['base_candidates'] = [dict(token_id=1,score=float('nan'))]
        row['variant_candidates'] = []
        with self.assertRaises(v24.DiscretizationContractError): v24._backtracking_signal(row,1,config)
    def test_invalid_budgets_and_foreign_calibration_observations(self):
        for key,value in [('backtracking_max_distance',3),('backtracking_max_events',3),
                ('backtracking_alternatives',3),('backtracking_max_trials',7),
                ('backtracking_forward_budget',-1),('backtracking_gap_threshold',float('nan'))]:
            with self.subTest(key=key),self.assertRaises(ValueError): v24.SuffixReoptimizationV24Config(**{key:value})
        with self.assertRaises(ValueError):
            v24.build_backtracking_calibration([dict(sample_id='eval:0',has_prefix=True,top1_score=.5)],
                [dict(sample_id='cal:0',window_errors=[.1,.1])],['cal:0'],{'model':'tiny'})

    def test_collects_independent_greedy_scores_and_real_repeated_window_forwards(self):
        kwargs,_ = path_fixture(immediate,backtracking_enabled=False)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        payload = v24.collect_backtracking_calibration([dict(sample_id='independent:0',result=result,
            target_hidden_state=kwargs['target_hidden_state'],attention_mask=kwargs['attention_mask'])],
            kwargs['model'],kwargs['embed_layer'],kwargs['tokenizer'],0,_forward_tokens)
        self.assertEqual(5,payload['score_counts']['with_prefix'])
        self.assertEqual(15,payload['numerical_repeat_count'])
        self.assertGreater(payload['acceptance_epsilon'],0)


class SearchTests(unittest.TestCase):
    def test_first_improvement_fixed_window_prefix_refresh_and_no_nested_search(self):
        kwargs,prefixes = path_fixture(immediate)
        entry = kwargs['optimized_embedding'].clone()
        embedding,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertEqual([0,2,3,3,1,1],result['final_tokens'])
        self.assertTrue(torch.equal(entry,embedding))
        self.assertEqual(1,result['backtracking']['event_count'])
        event = result['backtracking']['events'][0]
        self.assertEqual([1,3],event['window'])
        self.assertEqual(4,event['next_scan_position'])
        self.assertLess(event['E_selected'],event['E_base']-event['epsilon'])
        self.assertEqual(1,len(event['trials']))
        self.assertTrue(event['final_retained'])
        self.assertTrue(result['triggered'])
        self.assertTrue(result['accepted'])
        self.assertIn([0,2],prefixes)
        self.assertIn([0,2,3],prefixes)
        self.assertTrue(all(e['anomaly_trigger_suppressed'] for e in result['discretization']['events'] if e['path_kind']!='normal'))
        for table in result['initial_candidate_rerank']:
            self.assertEqual(v24.prefix_fingerprint(result['pre_tokens'][:table['position']]),table['prefix_fingerprint'])
    def test_rejected_current_choices_then_nearest_previous_position(self):
        kwargs,_ = path_fixture(farther)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        event = result['backtracking']['events'][0]
        self.assertEqual([3,2,1],event['split_order'])
        self.assertEqual([3,3,2],[t['split_position'] for t in event['trials']])
        self.assertEqual(1,event['retreat_depth'])
        self.assertEqual([0,1,2,1,1,1],result['final_tokens'])
        self.assertTrue(all(t['active_changes']==1 and t['window']==event['window'] for t in event['trials']))

    def test_reaches_two_positions_back_and_never_exceeds_six_trials(self):
        def scorer(tokens,position,mode):
            if position==3:
                return ({1:.999,2:.8,3:.7} if tokens[1]==3 else {1:.9,2:.8995,3:.6}).get(tokens[-1],.1)
            if position in (4,5): return {1:.99 if tokens[1]==3 else .7,2:.5,3:.4}.get(tokens[-1],.1)
            return {1:.95,2:.8,3:.7}.get(tokens[-1],.1)
        kwargs,_ = path_fixture(scorer)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        event = result['backtracking']['events'][0]
        self.assertTrue(event['accepted'])
        self.assertEqual(2,event['retreat_depth'])
        self.assertEqual(6,len(event['trials']))
        self.assertEqual([3,3,2,2,1,1],[t['split_position'] for t in event['trials']])
        self.assertLessEqual(result['backtracking']['compute']['forward_calls'],64)

    def test_history_id_absent_from_current_pool_is_scored_under_the_current_prefix(self):
        def scorer(tokens,position,mode):
            if position==1: return {1:.9,2:.8995,3:.2,4:.8994}.get(tokens[-1],.1)
            if position in (2,3): return {1:.99 if tokens[1]==4 else .3,2:.1,3:.05}.get(tokens[-1],.1)
            return {1:.99,2:.5,3:.3}.get(tokens[-1],.1)
        kwargs,_ = path_fixture(scorer)
        identity = v24.trajectory_identity(kwargs['model'],kwargs['embed_layer'],kwargs['tokenizer'])
        state = v24._new_backtracking_state(kwargs['config'],kwargs['model'],identity,0,'sample:1')
        state['history'] = [dict(token_id=4,position=1,identity=identity,sample_id='sample:1',
                                call_id='R_rejected',clock=-1,score=-.9)]
        decisions = []
        common = {k:kwargs[k] for k in ('tokenizer','model','embed_layer','target_hidden_state','layer_id',
            'eval_start_pos','embedding_top_indices','select_candidate_from_top_indices','get_perplexity',
            'forward_and_get_last_hidden_state','config','attention_mask')}
        tokens,_,_ = v24._rerank_positions(kwargs['optimized_embedding'],None,1,[0],
            invert_method='cosine',filter_nonascii=True,add_perplexity=True,top_k_ppl=1,top_k_cos=3,
            backtracking_state=state,identity=identity,discretization_events=decisions,**common)
        event = state['events'][0]
        self.assertEqual(4,tokens[1])
        self.assertEqual('historical_winner',event['trials'][1]['alternative']['source'])
        selected = decisions[event['trials'][1]['decision_indices'][0]]
        self.assertAlmostEqual(.8994,selected['selected_score'],places=5)
        self.assertTrue(any(row['quota_source']=='history' and row['token_id']==4 for row in selected['base_candidates']))
        synthetic = dict(version='v2.4',pre_tokens=tokens,final_tokens=tokens,events=[],
                         discretization={'events':decisions},backtracking={'events':state['events']})
        v24._mark_final_decisions(decisions)
        evaluated = discretization_offline_evaluation(synthetic,[0,4,1,1,1,1],1)
        final = next(row for row in evaluated['final_candidate_coverage'] if row['position']==1)
        self.assertTrue(final['pool_contains_gt'])
        self.assertTrue(final['history_newly_covered'])
    def test_low_fit_priority_and_suffix_left_boundary(self):
        def scorer(tokens,position,mode):
            if position==3 and tokens[2]!=2: return {1:.79,2:.7895,3:.6}.get(tokens[-1],.1)
            return farther(tokens,position,mode)
        kwargs,_ = path_fixture(scorer,fit=.8)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        event = result['backtracking']['events'][0]
        self.assertEqual('low_fit',event['type'])
        self.assertEqual([2,1],event['split_order'])
        self.assertEqual(2,event['trials'][0]['split_position'])
        identity = v24.trajectory_identity(kwargs['model'],kwargs['embed_layer'],kwargs['tokenizer'])
        state = v24._new_backtracking_state(kwargs['config'],kwargs['model'],identity,0)
        common = {k:kwargs[k] for k in ('tokenizer','model','embed_layer','target_hidden_state','layer_id',
            'eval_start_pos','embedding_top_indices','select_candidate_from_top_indices','get_perplexity',
            'forward_and_get_last_hidden_state','config','attention_mask')}
        tokens,_,_ = v24._rerank_positions(kwargs['optimized_embedding'],[0,1,1,1,1,1],3,[0],
            invert_method='cosine',filter_nonascii=True,add_perplexity=True,top_k_ppl=1,top_k_cos=3,
            call_id='R_1',backtracking_state=state,identity=identity,**common)
        self.assertEqual([0,1,1],tokens[:3])
        self.assertEqual([3],state['events'][0]['split_order'])
        self.assertEqual([3,5],state['events'][0]['window'])
    def test_empty_prefix_and_sequence_boundaries(self):
        for prefix,length in ((0,1),(0,2),(1,1),(1,2)):
            with self.subTest(prefix=prefix,length=length):
                kwargs,_ = path_fixture(lambda t,p,m:{1:.7,2:.6995,3:.5}.get(t[-1],.1),
                    prefix=prefix,length=length,fit=.8,first=.8)
                _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
                self.assertEqual(length,len(result['final_tokens']))
                if prefix: self.assertEqual(0,result['final_tokens'][0])
                for event in result['backtracking']['events']:
                    self.assertGreaterEqual(event['window'][0],prefix)
                    self.assertLess(event['window'][1],length)
                    self.assertTrue(all(p>=prefix for p in event['split_order']))
    def test_nonnegative_partial_sum_prunes_without_full_trial_forward(self):
        kwargs,_ = path_fixture(lambda t,p,m:{1:.999,2:.9985,3:.3}.get(t[-1],.1),length=4)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertEqual([0,1,1,1],result['final_tokens'])
        event = result['backtracking']['events'][0]
        self.assertTrue(all(t['reason']=='nonnegative_error_lower_bound' for t in event['trials']))
        self.assertEqual(1,result['backtracking']['compute']['by_category']['window']['forward_calls'])
    def test_budget_interruptions_preserve_original_path_and_count_actual_calls(self):
        for budget in (0,1,2,4,5,6,7,8,64):
            with self.subTest(budget=budget):
                kwargs,_ = path_fixture(immediate,backtracking_forward_budget=budget)
                observed = []
                handle = kwargs['model'].register_forward_pre_hook(lambda *a:observed.append(1))
                try: _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
                finally: handle.remove()
                extra = result['backtracking']['compute']['forward_calls']
                self.assertLessEqual(extra,budget)
                if budget<8: self.assertEqual([0,1,1,1,1,1],result['final_tokens'])
                self.assertEqual(0,len(kwargs['model']._forward_pre_hooks))
                self.assertEqual(0,len(kwargs['model'].layers[0]._forward_hooks))
                if budget==64:
                    normal = sum(result['compute_stats'][k]['forward_calls'] for k in ('candidate','ppl','observation','r_optimization'))
                    self.assertEqual(len(observed),normal+extra)
                    self.assertGreater(result['backtracking']['compute']['candidate_count'],0)
    def test_no_legal_backup_skips_without_consuming_event_budget(self):
        kwargs,_ = path_fixture(lambda *_:.9)
        kwargs['embedding_top_indices'] = lambda *_:torch.tensor([1,0])
        kwargs['add_perplexity'] = False
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertEqual(0,result['backtracking']['event_count'])
        self.assertEqual({'no_legal_alternative'},{e['reason'] for e in result['backtracking']['events']})

    def test_recoverable_baseline_forward_failure_retains_original_and_removes_hooks(self):
        kwargs,_ = path_fixture(immediate)
        original = kwargs['get_perplexity']
        calls = []
        def failing_once(tokens,*args,**options):
            if len(tokens)==2 and not calls:
                calls.append(True)
                kwargs['model'].layers[0].register_forward_hook(lambda *_:None)
                kwargs['model'](input_ids=torch.tensor([tokens]))
                raise RuntimeError('transient speculative failure')
            return original(tokens,*args,**options)
        kwargs['get_perplexity'] = failing_once
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertEqual([0,1,1,1,1,1],result['final_tokens'])
        self.assertFalse(result['backtracking']['events'][0]['accepted'])
        self.assertEqual(0,len(kwargs['model'].layers[0]._forward_hooks))

    def test_acceptance_epsilon_rejects_numerically_insufficient_improvement(self):
        kwargs,_ = path_fixture(immediate)
        old = kwargs['config']
        identity = v24.trajectory_identity(kwargs['model'],kwargs['embed_layer'],kwargs['tokenizer'])
        payload = v24.build_backtracking_calibration(
            [dict(sample_id='independent:0',has_prefix=True,top1_score=.1)],
            [dict(sample_id='independent:0',window_errors=[0.,1.])], ['independent:0'],
            v24.backtracking_inference_signature(kwargs['model'],identity,0))
        fields = {key:getattr(old,key) for key in old.__dataclass_fields__}
        for key in ('fit_threshold','first_fit_threshold','acceptance_epsilon','consistency_tolerance'):
            fields['backtracking_'+key] = None
        fields['backtracking_calibration'] = payload
        kwargs['config'] = v24.SuffixReoptimizationV24Config(**fields)
        _,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertEqual([0,1,1,1,1,1],result['final_tokens'])
        self.assertTrue(any(trial['completed'] and not trial['accepted'] for trial in result['backtracking']['events'][0]['trials']))
    def test_recent_history_filters_sample_time_special_ids_and_never_reuses_scores(self):
        kwargs,_ = path_fixture(immediate)
        identity = v24.trajectory_identity(kwargs['model'],kwargs['embed_layer'],kwargs['tokenizer'])
        state = v24._new_backtracking_state(kwargs['config'],kwargs['model'],identity,0,'sample:1')
        def old(token,clock):
            return dict(token_id=token,clock=clock,position=1,identity=identity,sample_id='sample:1',call_id='R_rejected')
        state['history'] = [old(4,1),old(5,2),dict(old(6,3),sample_id='another:1'),old(0,4),old(2,5),old(7,8)]
        row = dict(prefix_fingerprint=v24.prefix_fingerprint([0]),
            base_candidates=[dict(token_id=t,score=s) for t,s in ((1,.9),(2,.8),(3,.7))],variant_candidates=[])
        items = v24._backtracking_alternatives(row,[0,1,1],1,state,8,kwargs['tokenizer'],kwargs['embed_layer'].num_embeddings)
        self.assertEqual([2,5],[item['token_id'] for item in items])
        self.assertEqual('historical_winner',items[1]['source'])
        self.assertNotIn('score',items[1])
    def test_r_rollback_keeps_formal_state_and_search_budget_is_not_refunded(self):
        def scorer(tokens,position,mode):
            return {1:.99,2:.5,3:.3}.get(tokens[-1],.1) if mode=='initial' else immediate(tokens,position,mode)
        kwargs,_ = path_fixture(scorer,max_attempts=2,steps=1,backtracking_max_events=1)
        entry = kwargs['optimized_embedding'].clone()
        def optimize(*args,**options):
            kwargs['model'].mode = 'R_trial'
            return args[1]+.01,1.,2.,dict(completed_steps=1)
        with mock.patch.object(v24,'_optimize_suffix',side_effect=optimize):
            embedding,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertTrue(torch.equal(entry,embedding))
        self.assertEqual([0,1,1,1,1,1],result['final_tokens'])
        self.assertEqual(2,result['rejected_round_count'])
        self.assertEqual(1,result['backtracking']['event_count'])
        event = next(e for e in result['backtracking']['events'] if e['launched'])
        self.assertTrue(event['accepted'])
        self.assertFalse(event['r_accepted'])
        self.assertFalse(event['accepted_into_formal_state'])
        self.assertFalse(event['final_retained'])
        self.assertTrue(event['rolled_back'])
        self.assertGreater(result['backtracking']['compute']['forward_calls'],0)
        self.assertTrue(any(e['reason']=='event_budget_exhausted' for e in result['backtracking']['events']))
        self.assertTrue(all(not e['final_retained'] for e in result['discretization']['events'] if e['call_id']!='stage1'))


class IntegrationTests(unittest.TestCase):
    def test_configs_aliases_v23_rollback_resolved_config_and_fixed_log(self):
        fn = functions('load_config','merge_config','canonicalize_config_aliases','normalize_suffix_version','select_advanced_method')
        values = fn['load_config'](str(ROOT/'experiment_configs/l24_deml3x4_suffix_v2_4.json'))
        cfg = v24.config_from_mapping(values)
        for alias in ('2.4','v2.4','suffix_v2_4','suffix_reoptimization_v2_4','suffix_reoptimization_v2.4'):
            self.assertEqual(v24.METHOD_NAME,fn['select_advanced_method'](alias,None,None,None,None,suffix_reopt_v2_4_config=cfg))
        old_values = fn['load_config'](str(ROOT/'experiment_configs/l24_deml3x4_suffix_v2_3.json'))
        self.assertEqual(v23.METHOD_NAME,fn['select_advanced_method']('v2.3',None,None,None,None,suffix_reopt_v2_3_config=v23.config_from_mapping(old_values)))
        class Args(types.SimpleNamespace):
            def __getattr__(self,name): return None
        args = Args(**values,selected_advanced_method=v24.METHOD_NAME,selected_candidate_reranking_method='none',config='v24.json')
        resolved = build_resolved_config(args,'time','run','experiment.log','reconstructions.jsonl',None,1,24,'qwen2')
        self.assertEqual(['suffix_reoptimization_v2_4'],list(resolved['advanced_methods']))
        settings = resolved['advanced_methods']['suffix_reoptimization_v2_4']
        self.assertEqual(64,settings['backtracking_forward_budget'])
        self.assertFalse(settings['backtracking_calibration_ready'])
        record = dict(selected_advanced_method=v24.METHOD_NAME,selected_candidate_reranking_method='none',
            accuracy=.75,suffix_reoptimization_v2_4_result=dict(pre_acc=.5,post_acc=.75))
        self.assertEqual(dict(pre_suffix=.5,suffix_v2_4=.75),build_stage_accuracy(record))
        buffer = io.StringIO()
        write_experiment_sample_summary(buffer,record,1,1,4)
        self.assertNotIn('backtrack',buffer.getvalue())
        self.assertNotIn('candidate',buffer.getvalue())
        with self.assertRaises(ValueError): v24.config_from_mapping({})
    def test_disabled_search_matches_v23_candidate_and_continuous_flow(self):
        torch.set_num_threads(1)
        kwargs = real_fixture(attempts=1,steps=5)
        before,old = v23.run_suffix_reoptimization_v2_3(**kwargs)
        fields = {k:getattr(kwargs['config'],k) for k in kwargs['config'].__dataclass_fields__}
        kwargs['config'] = v24.SuffixReoptimizationV24Config(**fields,backtracking_enabled=False)
        after,result = v24.run_suffix_reoptimization_v2_4(**kwargs)
        self.assertTrue(torch.equal(before,after))
        self.assertEqual(old['final_tokens'],result['final_tokens'])
        self.assertEqual(old['pre_tokens'],result['pre_tokens'])
        self.assertEqual(old['accepted_round_count'],result['accepted_round_count'])
        self.assertEqual(0,result['backtracking']['compute']['forward_calls'])
    def test_real_stage1_grouped_snapshot_v24_and_v23_readers(self):
        torch.set_num_threads(1)
        kwargs = real_fixture(attempts=0)
        entry = kwargs.pop('optimized_embedding')
        fields = {k:getattr(kwargs['config'],k) for k in kwargs['config'].__dataclass_fields__}
        kwargs['config'] = v24.SuffixReoptimizationV24Config(**fields,backtracking_enabled=False)
        stage1 = dict(model=kwargs['model'],initial_embedding=entry[:,1:],prefix_embedding=entry[:,:1],
            target_hidden_state=kwargs['target_hidden_state'],attention_mask=kwargs['attention_mask'],
            layer_id=0,register_layer_hooks=_register_layer_hooks,weight_mask=torch.ones(4),
            right_range=torch.ones(4),lr=.01,epoch=901,alpha=.001)
        with tempfile.TemporaryDirectory(prefix='suffix_v24_snapshot_',dir=ROOT/'outputs') as directory:
            path = str(Path(directory)/'shared.pt')
            shared = dict(stage1_kwargs=stage1,stage2_kwargs=kwargs,snapshot_path=path,pair_id='tiny:0',snapshot_contract={'block':0})
            final,a = v24.run_two_stage(**shared,snapshot_mode='write')
            with mock.patch.object(v24,'optimize_stage1',side_effect=AssertionError('must reuse Stage-1')):
                repeated,b = v24.run_two_stage(**shared,snapshot_mode='read')
            self.assertTrue(torch.equal(final,repeated))
            self.assertEqual(a['final_tokens'],b['final_tokens'])
            self.assertEqual([500,600,700,800,900],a['stage1']['discretization_trajectory']['retained_steps'])
            old_kwargs = dict(kwargs,config=v23.SuffixReoptimizationV23Config(**fields))
            _,old = v23.run_two_stage(**dict(shared,stage2_kwargs=old_kwargs),snapshot_mode='read')
            self.assertEqual(a['stage1_snapshot_sha256'],old['stage1_snapshot_sha256'])
            snapshot = torch.load(path,weights_only=False)
            snapshot['trajectory_groups'].pop('v2.4')
            torch.save(snapshot,path)
            _,from_v23 = v24.run_two_stage(**shared,snapshot_mode='read')
            self.assertEqual(a['final_tokens'],from_v23['final_tokens'])
    def test_main_stage_branch_runs_v24_and_records_stage1(self):
        kwargs = real_fixture(attempts=0)
        entry = kwargs['optimized_embedding']
        fields = {k:getattr(kwargs['config'],k) for k in kwargs['config'].__dataclass_fields__}
        config = v24.SuffixReoptimizationV24Config(**fields,backtracking_enabled=False)
        args = types.SimpleNamespace(base_model_name='tiny',num_invert_layers=0,lr=.01,epoch=3,
            alpha=.001,clip=True,init_method='uniform',init_param=.1,optim_method='cosine',seed=0,
            filter_nonascii=True,perplexity=True,top_k_ppl=2,top_k_cos=2,invert_method='cosine')
        namespace = dict(selected_advanced_method=v24.METHOD_NAME,args=args,tokenizer=kwargs['tokenizer'],
            model=kwargs['model'],embed_layer=kwargs['embed_layer'],eval_start_pos=1,
            dataset_metadata={'name':'tiny','sample_index':0},sample_idx=0,
            next_hidden_states_last=kwargs['target_hidden_state'],new_input_embed_0=entry[:,1:],
            prefix_embed=entry[:,:1],total_attention_mask=kwargs['attention_mask'],
            weight_mask=torch.ones(4),right_range=torch.ones(4),register_layer_hooks=_register_layer_hooks,
            suffix_reopt_v2_4_config=config,run_suffix_v24_two_stage=v24.run_two_stage,
            embedding_top_indices=kwargs['embedding_top_indices'],select_candidate_from_top_indices=kwargs['select_candidate_from_top_indices'],
            get_perplexity=kwargs['get_perplexity'],forward_and_get_last_hidden_state=_forward_tokens,os=__import__('os'))
        tree = ast.parse((ROOT/'invert.py').read_text(encoding='utf-8'))
        branches = [node for node in ast.walk(tree) if isinstance(node,ast.If) and isinstance(node.test,ast.Compare)
            and any(isinstance(part,ast.Constant) and part.value==v24.METHOD_NAME for part in node.test.comparators)]
        branch = next(node for node in branches if any(isinstance(child,ast.Name) and child.id=='run_suffix_v24_two_stage'
            for statement in node.body for child in ast.walk(statement)))
        exec(compile(ast.Module(body=branch.body,type_ignores=[]),str(ROOT/'invert.py'),'exec'),namespace)
        self.assertEqual('v2.4',namespace['optimization_result']['version'])
        self.assertEqual('stage1_then_suffix_v2_4',namespace['optimization_result']['stage'])
    def test_labels_only_enter_offline_repairs_damage_and_retention(self):
        kwargs,_ = path_fixture(immediate)
        buffer = io.StringIO()
        _,result = v24.run_suffix_reoptimization_v2_4(**dict(kwargs,log_file=buffer))
        self.assertEqual('',buffer.getvalue())
        self.assertFalse(result['gt_accessed'])
        evaluated = discretization_offline_evaluation(result,[0,2,3,3,1,1],1)
        self.assertEqual(3,evaluated['backtracking_events'][0]['selected_changes']['repairs'])
        self.assertEqual(0,evaluated['backtracking_events'][0]['selected_changes']['damage'])
        self.assertEqual([],evaluated['final_error_types'])
        self.assertEqual(5,len(evaluated['final_candidate_coverage']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
