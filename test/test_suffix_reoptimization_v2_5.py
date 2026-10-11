"""v2.5 phase separation, exhaustive bounded search, and independent wiring."""
import ast
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch

from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_4 as v24
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_5 as v25
from test.test_suffix_reoptimization_v2_3 import fixture as real_fixture
from test.test_suffix_reoptimization_v2_4 import path_fixture as old_path_fixture, immediate
from test.test_suffix_reoptimization_v2_2_1 import _forward_tokens, _register_layer_hooks
from experiment_outputs_suffix_v2_5 import (
    build_resolved_config, build_stage_accuracy, discretization_offline_evaluation,
    write_experiment_sample_summary, write_experiment_average_summary)

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'invert_suffix_v2_5.py').read_text(encoding='utf-8'))
torch.set_num_threads(1)


def path_fixture(scorer=immediate, **options):
    v25_keys = {k: options.pop(k) for k in list(options)
                if k.startswith('backtracking_') or k in ('reoptimization_enabled', 'expansion_enabled')}
    kwargs, prefixes = old_path_fixture(scorer, **options)
    fields = vars(kwargs['config']).copy()
    for key in ('max_distance', 'alternatives', 'max_trials', 'forward_budget'):
        fields.pop('backtracking_' + key)
    fields.update(v25_keys)
    kwargs['config'] = v25.SuffixReoptimizationV25Config(**fields)
    return kwargs, prefixes


def functions(*names):
    namespace = dict(os=os, SCRIPT_DIR=str(ROOT), json=json, copy=copy)
    nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'invert_suffix_v2_5.py', 'exec'), namespace)
    return namespace


def run(kwargs):
    return v25.run_suffix_reoptimization_v2_5(**kwargs)


def best_later(tokens, position, mode):
    if position == 1:
        return {1: .95, 2: .94, 3: .93, 4: .92, 5: .91}.get(tokens[-1], .05)
    if position in (2, 3, 4):
        gain = {1: .3, 2: .7, 3: .8, 4: .9, 5: .99}[tokens[1]]
        return gain if tokens[-1] == 1 else .05
    return .99 if tokens[-1] == 1 else .05


class TrajectoryCandidateTests(unittest.TestCase):
    def test_stage1_still_adds_top1_candidates(self):
        kwargs, _ = path_fixture()
        trajectory = dict(call_id='stage1', stages=[dict(step=500, token_ids=[7]),
                                                   dict(step=600, token_ids=[2])])
        base = v25.build_discretization_base([1, 2], [3], kwargs['tokenizer'],
            kwargs['embed_layer'].num_embeddings, trajectory=trajectory)
        self.assertEqual([1, 2, 3, 7], [row['token_id'] for row in base['candidates']])
        self.assertEqual([7], base['trajectory_candidate_ids'])
        self.assertEqual(['embedding', 'trajectory'],
                         [s['source'] for s in base['candidates'][1]['sources']])

    def test_r_samples_do_not_add_or_blacklist_base_candidates(self):
        kwargs, _ = path_fixture()
        trajectory = dict(call_id='R_1', stages=[dict(step=30, token_ids=[2]),
                                                dict(step=40, token_ids=[7])])
        for ppl in ([3], [3, 7]):
            with self.subTest(ppl=ppl):
                base = v25.build_discretization_base([1, 2], ppl, kwargs['tokenizer'],
                    kwargs['embed_layer'].num_embeddings, trajectory=trajectory)
                self.assertEqual([1, 2]+ppl, [row['token_id'] for row in base['candidates']])
                self.assertEqual([], base['trajectory_candidate_ids'])
                self.assertTrue(all(s['source'] != 'trajectory'
                                    for row in base['candidates'] for s in row['sources']))
                self.assertEqual(['r_trajectory_candidates_disabled']*2,
                                 [row['reason'] for row in base['exclusions']])

    def test_r_trial_and_final_search_exclude_r_samples_without_stage1_fallback(self):
        for accepted in (False, True):
            with self.subTest(accepted=accepted):
                kwargs, _ = path_fixture(fit=.8, max_attempts=1, steps=50)
                identity = v25.trajectory_identity(kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer'])
                stage1 = v25.DiscretizationTrajectoryCollector(1000, 1, kwargs['embed_layer'],
                    'cosine', 'stage1', [0], identity, kwargs['config'].stage1_sampling_steps)
                stage1.snapshots[500] = [[7] for _ in range(5)]
                kwargs['stage_candidates'] = stage1.finalize(1000, 6)
                def optimize(*args, **options):
                    callback = options['trajectory_callback']
                    with mock.patch.object(v25, 'stage_neighbours', return_value=[[8] for _ in range(5)]):
                        callback(30, args[1])
                    with mock.patch.object(v25, 'stage_neighbours', return_value=[[2] for _ in range(5)]):
                        callback(40, args[1])
                    return args[1] + .01, 2., 1. if accepted else 3., dict(completed_steps=50)
                with mock.patch.object(v25, '_optimize_suffix', side_effect=optimize):
                    _, result = run(kwargs)
                self.assertEqual(accepted, result['events'][0]['accepted'])
                self.assertEqual([30, 40], result['events'][0]['anchor']['discretization_trajectory']['retained_steps'])
                self.assertFalse(result['discretization']['r_trajectory_candidates_enabled'])
                initial = [d for d in result['discretization']['events'] if d['phase'] == 'initial']
                trial = [d for d in result['discretization']['events'] if d['phase'] == 'R']
                final = [d for d in result['discretization']['events'] if d['phase'] == 'final']
                self.assertTrue(initial and trial and final)
                self.assertTrue(all(d['trajectory_candidate_ids'] == [7] for d in initial))
                self.assertTrue(all(not d['trajectory_candidate_ids'] for d in trial))
                for decision in trial + (final if accepted else []):
                    self.assertEqual('R_1', decision['trajectory']['call_id'])
                    self.assertNotIn(7, decision['original_candidate_ids'])
                    self.assertNotIn(8, [row['token_id'] for row in decision['base_candidates']])
                    self.assertEqual(['r_trajectory_candidates_disabled']*2,
                                     [e['reason'] for e in decision['exclusions']])
                if not accepted:
                    self.assertTrue(all(d['trajectory_candidate_ids'] == [7] for d in final))
                self.assertTrue(result['freeze_checks']['passed'])


class ComponentSwitchTests(unittest.TestCase):
    def test_all_eight_combinations_gate_r_expansion_and_final_backtracking(self):
        for r_enabled in (False, True):
            for e_enabled in (False, True):
                for b_enabled in (False, True):
                    with self.subTest(R=r_enabled, E=e_enabled, B=b_enabled):
                        kwargs, _ = path_fixture(fit=.8, max_attempts=1, steps=50,
                            seed_top_k=5, case_quota=3, space_quota=3, prefix_quota=1, suffix_quota=3,
                            reoptimization_enabled=r_enabled, expansion_enabled=e_enabled,
                            backtracking_enabled=b_enabled)
                        identity = v25.trajectory_identity(kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer'])
                        stage1 = v25.DiscretizationTrajectoryCollector(1000, 1, kwargs['embed_layer'],
                            'cosine', 'stage1', [0], identity, kwargs['config'].stage1_sampling_steps)
                        stage1.snapshots[500] = [[7] for _ in range(5)]
                        kwargs['stage_candidates'] = stage1.finalize(1000, 6)
                        entrance = kwargs['optimized_embedding'].clone()
                        def optimize(*args, **options):
                            with mock.patch.object(v25, 'stage_neighbours', return_value=[[8] for _ in range(5)]):
                                options['trajectory_callback'](30, args[1])
                                options['trajectory_callback'](40, args[1])
                            return args[1] + .01, 2., 1., dict(completed_steps=50)
                        with mock.patch.object(v25, '_optimize_suffix', side_effect=optimize) as optimizer, \
                                mock.patch.object(v25, '_position_similarity', wraps=v25._position_similarity) as similarity:
                            embedding, result = run(kwargs)
                        self.assertEqual(int(r_enabled), optimizer.call_count)
                        self.assertEqual(int(r_enabled), result['attempt_count'])
                        self.assertEqual(dict(R=r_enabled, E=e_enabled, B=b_enabled), result['component_switches'])
                        self.assertEqual(b_enabled, bool(result['backtracking']['events']))
                        if not r_enabled:
                            similarity.assert_not_called()
                            self.assertEqual([], result['events'])
                            self.assertTrue(torch.equal(entrance, embedding))
                        decisions = result['discretization']['events']
                        self.assertTrue(all(d['expansion_enabled'] == e_enabled for d in decisions))
                        initial = [d for d in decisions if d['phase'] == 'initial']
                        if e_enabled:
                            self.assertTrue(all(d['trajectory_candidate_ids'] == [7] for d in initial))
                            self.assertTrue(any(d['variant_candidates'] for d in decisions))
                        else:
                            for decision in decisions:
                                self.assertEqual([], decision['trajectory_candidate_ids'])
                                self.assertEqual([], decision['seeds'])
                                self.assertEqual([], decision['variant_candidates'])
                                self.assertEqual(decision['original_candidate_ids'],
                                    [row['token_id'] for row in decision['base_candidates']])
                                self.assertTrue(all(s['source'] in ('embedding', 'ppl')
                                    for row in decision['base_candidates'] for s in row['sources']))
                        self.assertFalse(result['discretization']['r_trajectory_candidates_enabled'])
                        self.assertTrue(result['freeze_checks']['passed'])
                        self.assertEqual(5, kwargs['config'].seed_top_k)
                        self.assertEqual(5, kwargs['config'].trajectory_max_new)

    def test_e_off_leaves_original_ids_and_skips_variant_generation(self):
        kwargs, _ = path_fixture(expansion_enabled=False)
        for call_id in ('stage1', 'R_1'):
            trajectory = dict(call_id=call_id, stages=[dict(step=30, token_ids=[2]), dict(step=40, token_ids=[7])])
            base = v25.build_discretization_base([1, 2], [3, 7], kwargs['tokenizer'],
                kwargs['embed_layer'].num_embeddings, trajectory=trajectory, expansion_enabled=False)
            self.assertEqual([1, 2, 3, 7], [row['token_id'] for row in base['candidates']])
            self.assertEqual([], base['trajectory_candidate_ids'])
            self.assertTrue(all(s['source'] != 'trajectory' for row in base['candidates'] for s in row['sources']))
        with mock.patch.object(v25, '_variant_texts', side_effect=AssertionError('E is disabled')):
            variants = v25.build_discretization_variants([], kwargs['tokenizer'],
                kwargs['embed_layer'].num_embeddings, True, kwargs['config'])
        self.assertEqual([], variants['candidates'])
        self.assertEqual([], variants['seeds'])

    def test_config_switches_are_boolean_and_legacy_configs_default_on(self):
        fn = functions('load_config', 'merge_config', 'canonicalize_config_aliases')
        values = fn['load_config'](str(ROOT / 'experiment_configs/l24_deml3x4_suffix_v2_5.json'))
        self.assertTrue(v25.config_from_mapping(values).reoptimization_enabled)
        self.assertTrue(v25.config_from_mapping(values).expansion_enabled)
        for name in ('reoptimization_enabled', 'expansion_enabled'):
            key = 'suffix_v2_5_' + name
            self.assertFalse(getattr(v25.config_from_mapping(dict(values, **{key: False})), name))
            for invalid in (0, 1, 'false', None):
                with self.subTest(name=name, value=invalid), self.assertRaises(TypeError):
                    v25.config_from_mapping(dict(values, **{key: invalid}))
            values.pop(key)
        legacy = v25.config_from_mapping(values)
        self.assertTrue(legacy.reoptimization_enabled and legacy.expansion_enabled)

    def test_resolved_config_reports_disabled_components_and_effective_pool_bounds(self):
        fn = functions('load_config', 'merge_config', 'canonicalize_config_aliases')
        values = fn['load_config'](str(ROOT / 'experiment_configs/l24_deml3x4_suffix_v2_5.json'))
        values.update(suffix_v2_5_reoptimization_enabled=False, suffix_v2_5_expansion_enabled=False)
        class Args(types.SimpleNamespace):
            def __getattr__(self, name):
                return None
        args = Args(**values, selected_advanced_method=v25.METHOD_NAME,
            selected_candidate_reranking_method='none', config='v25.json')
        resolved = build_resolved_config(args, 'time', 'run', 'experiment.log', 'reconstructions.jsonl', None, 1, 24, 'qwen2')
        settings = resolved['advanced_methods']['suffix_reoptimization_v2_5']
        self.assertEqual(dict(R=False, E=False, B=True), settings['component_switches'])
        self.assertEqual(0, settings['effective_max_attempts'])
        self.assertEqual([], settings['expansion_stages'])
        self.assertFalse(settings['stage1_trajectory_candidates_enabled'])
        self.assertFalse(settings['r_trajectory_candidates_enabled'])
        self.assertEqual((20, 10), (settings['maximum_initial_candidates'], settings['maximum_initial_no_prefix_candidates']))
        self.assertEqual((20, 10), (settings['maximum_r_candidates'], settings['maximum_r_no_prefix_candidates']))
        self.assertEqual(20, settings['maximum_final_candidates'])
        self.assertEqual(5, settings['seed_top_k'])
        self.assertEqual(5, settings['trajectory_max_new'])


class TriggerAndSearchTests(unittest.TestCase):
    def test_small_gap_is_diagnostic_and_low_fit_is_strictly_required(self):
        kwargs, _ = path_fixture(lambda t, p, m: {1: .9, 2: .8995, 3: .2}.get(t[-1], .1), fit=.8)
        _, result = run(kwargs)
        self.assertEqual([], result['backtracking']['events'])
        signals = [d['candidate_signal'] for d in result['discretization']['events']]
        self.assertTrue(all(s['small_gap'] and not s['low_fit'] and s['type'] is None for s in signals))
        cfg = kwargs['config']
        row = dict(base_candidates=[dict(token_id=1, score=.8)], variant_candidates=[])
        self.assertIsNone(v25._backtracking_signal(row, 1, cfg)['type'])

    def test_alternatives_are_current_rank_top_four_distinct_ids(self):
        kwargs, _ = path_fixture(fit=.8)
        identity = v25.trajectory_identity(kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer'])
        state = v25._new_backtracking_state(kwargs['config'], kwargs['model'], identity, 0)
        state['history'] = [dict(token_id=9, score=1.)]
        row = dict(prefix_fingerprint=v25.prefix_fingerprint([0]),
            base_candidates=[dict(token_id=t, score=s) for t, s in
                             ((1, .99), (3, .8), (0, 1.), (2, .9), (3, .8), (5, .6), (4, .7), (6, .5))],
            variant_candidates=[dict(token_id=7, score=float('nan'))])
        actual = v25._backtracking_alternatives(row, [0, 1, 1], 1, state, 0,
                                               kwargs['tokenizer'], kwargs['embed_layer'].num_embeddings)
        self.assertEqual([2, 3, 4, 5], [a['token_id'] for a in actual])
        self.assertTrue(all(a['source'] == 'current_ranking' for a in actual))
        with self.assertRaises(v25.DiscretizationContractError):
            v25._backtracking_alternatives(row, [2, 1, 1], 1, state, 0,
                                          kwargs['tokenizer'], kwargs['embed_layer'].num_embeddings)

    def test_search_continues_after_improvement_and_selects_best_complete_route(self):
        kwargs, prefixes = path_fixture(best_later, fit=.8, backtracking_prune=False)
        kwargs['embedding_top_indices'] = lambda *_: torch.tensor([1, 2, 3, 4, 5])
        _, result = run(kwargs)
        event = next(e for e in result['backtracking']['events'] if e['launched'])
        completed = [t for t in event['trials'] if t['completed']]
        self.assertEqual(4, len(completed))
        self.assertTrue(completed[0]['improves_baseline'])
        self.assertFalse(completed[0]['accepted'])
        self.assertEqual(3, event['accepted_trial_index'])
        self.assertAlmostEqual(min(t['E_trial'] for t in completed), event['E_selected'])
        self.assertEqual(5, result['final_tokens'][1])
        self.assertTrue(any(5 in prefix for prefix in prefixes))
        retained = [d for d in result['discretization']['events'] if d['final_retained']]
        for decision in retained:
            self.assertEqual(v25.prefix_fingerprint(result['final_tokens'][:decision['position']]),
                             decision['prefix_fingerprint'])
        for event in result['backtracking']['events']:
            self.assertEqual('final_discretization', event['call_id'])
            self.assertIsNone(event['r_accepted'])

    def test_four_position_retreat_and_sixteen_routes(self):
        def scorer(tokens, position, mode):
            if position < 5:
                return {1: .95, 2: .9, 3: .89, 4: .88, 5: .87}.get(tokens[-1], .05)
            return (.99 if tokens[1] == 5 else .3) if tokens[-1] == 1 else .05
        kwargs, _ = path_fixture(scorer, length=8, fit=.8, backtracking_prune=False)
        kwargs['embedding_top_indices'] = lambda *_: torch.tensor([1, 2, 3, 4, 5])
        _, result = run(kwargs)
        event = next(e for e in result['backtracking']['events'] if e['launched'])
        self.assertEqual([4, 3, 2, 1], event['split_order'])
        self.assertEqual([1, 7], event['window'])
        self.assertEqual(16, len(event['trials']))
        self.assertEqual(4, event['retreat_depth'])
        self.assertEqual(5, result['final_tokens'][1])
        self.assertTrue(all(t['active_changes'] == 1 for t in event['trials']))
        self.assertLessEqual(result['backtracking']['compute']['forward_calls'], 640)

    def test_budget_interruptions_restore_greedy_tokens_and_decisions(self):
        saw_discarded_improvement = False
        for budget in (0, 1, 2, 4, 5, 7, 10, 12, 16, 640):
            with self.subTest(budget=budget):
                kwargs, _ = path_fixture(best_later, fit=.8, backtracking_prune=False,
                                         backtracking_forward_budget=budget)
                kwargs['embedding_top_indices'] = lambda *_: torch.tensor([1, 2, 3, 4, 5])
                _, result = run(kwargs)
                self.assertLessEqual(result['backtracking']['compute']['forward_calls'], budget)
                for event in result['backtracking']['events']:
                    if event['budget_exhausted']:
                        self.assertFalse(event['accepted'])
                        self.assertEqual(event['base_tokens'], event['selected_tokens'])
                        self.assertTrue(all(not t['accepted'] for t in event['trials']))
                        saw_discarded_improvement |= any(t.get('improves_baseline') for t in event['trials'])
                self.assertTrue(result['freeze_checks']['final_decisions_match_tokens'])
                self.assertEqual(0, len(kwargs['model']._forward_pre_hooks))
                self.assertEqual(0, len(kwargs['model'].layers[0]._forward_hooks))
        self.assertTrue(saw_discarded_improvement)

    def test_pruning_and_failed_routes_never_enter_complete_route_selection(self):
        kwargs, _ = path_fixture(lambda t, p, m: {1: .79, 2: .5, 3: .2}.get(t[-1], .1), fit=.8)
        _, result = run(kwargs)
        self.assertEqual([0, 1, 1, 1, 1, 1], result['final_tokens'])
        trials = [t for e in result['backtracking']['events'] for t in e['trials']]
        self.assertTrue(any(t['reason'] == 'nonnegative_error_lower_bound' for t in trials))
        self.assertTrue(all(not t['accepted'] for t in trials if not t['completed']))

    def test_empty_prefix_single_token_and_fixed_prefix_boundaries(self):
        for prefix, length in ((0, 1), (0, 2), (1, 1), (1, 2), (1, 8)):
            with self.subTest(prefix=prefix, length=length):
                kwargs, _ = path_fixture(lambda t, p, m: {1: .7, 2: .6, 3: .5}.get(t[-1], .1),
                                         prefix=prefix, length=length, fit=.8, first=.8)
                _, result = run(kwargs)
                self.assertEqual(length, len(result['final_tokens']))
                self.assertLessEqual(result['backtracking']['event_count'], 2)
                if prefix:
                    self.assertEqual(0, result['final_tokens'][0])
                for event in result['backtracking']['events']:
                    self.assertTrue(all(p >= prefix for p in event['split_order']))
                    if event['position'] == prefix:
                        self.assertEqual([prefix], event['split_order'])
                if length == prefix:
                    self.assertEqual([], result['final_candidate_rerank'])

    def test_missing_calibration_never_launches_search(self):
        kwargs, _ = path_fixture(fit=.8, calibrated=False)
        _, result = run(kwargs)
        self.assertFalse(result['backtracking']['calibration_ready'])
        self.assertEqual(0, result['backtracking']['event_count'])
        self.assertEqual(0, result['backtracking']['compute']['forward_calls'])

    def test_strict_original_epsilon_is_used(self):
        kwargs, _ = path_fixture(best_later, fit=.8, backtracking_prune=False)
        identity = v25.trajectory_identity(kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer'])
        calibration = v25.build_backtracking_calibration(
            [dict(sample_id='cal:other', has_prefix=True, top1_score=.8)],
            [dict(sample_id='cal:other', window_errors=[0., 1.])], ['cal:other'],
            v25.backtracking_inference_signature(kwargs['model'], identity, 0))
        fields = vars(kwargs['config']).copy()
        for name in ('fit_threshold', 'first_fit_threshold', 'acceptance_epsilon', 'consistency_tolerance'):
            fields['backtracking_' + name] = None
        fields['backtracking_calibration'] = calibration
        kwargs['config'] = v25.SuffixReoptimizationV25Config(**fields)
        _, result = run(kwargs)
        self.assertEqual(0, result['backtracking']['accepted_count'])
        self.assertTrue(all(e['epsilon'] == calibration['acceptance_epsilon'] for e in result['backtracking']['events']))


class PhaseAndFreezeTests(unittest.TestCase):
    def test_real_r_off_on_share_continuous_result_and_freeze_embedding(self):
        kwargs = real_fixture(attempts=2, steps=5)
        fields = vars(kwargs['config']).copy()
        outputs = []
        for enabled in (False, True):
            kwargs['config'] = v25.SuffixReoptimizationV25Config(**fields, backtracking_enabled=enabled)
            embedding, result = run(kwargs)
            outputs.append((embedding, result))
            checks = result['freeze_checks']
            self.assertTrue(checks['passed'])
            self.assertEqual(0, checks['optimization_backtracking_events'])
            self.assertEqual(0, checks['r_optimization_calls_after_freeze'])
            self.assertEqual(0, checks['grad_enabled_forward_calls'])
            self.assertEqual(0, checks['requires_grad_embedding_forward_calls'])
            for d in result['discretization']['events'][:result['optimization_decision_count']]:
                self.assertFalse(d['accepted_into_formal_state'])
                self.assertFalse(d['final_retained'])
        self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]))
        a, b = outputs[0][1], outputs[1][1]
        self.assertEqual(a['optimization_final_tokens'], b['optimization_final_tokens'])
        self.assertEqual(a['freeze_checks']['embedding_sha256_before'], b['freeze_checks']['embedding_sha256_before'])
        for left, right in zip(a['events'], b['events']):
            for key in ('pre_hidden_loss', 'post_hidden_loss', 'accepted', 'trial_tokens'):
                self.assertEqual(left.get(key), right.get(key))

    def test_all_r_acceptance_and_rejection_complete_before_final_search(self):
        kwargs, _ = path_fixture(fit=.8, max_attempts=2, steps=1)
        entry = kwargs['optimized_embedding'].clone()
        order = []
        original = v25._rerank_positions
        def scan(*args, **options):
            order.append(('scan', options.get('call_id', 'stage1'), options['config'].backtracking_enabled))
            return original(*args, **options)
        def optimize(*args, **options):
            order.append(('R', len([x for x in order if x[0] == 'R'])))
            index = order[-1][1]
            return args[1] + .01, 2., 1. if index == 0 else 3., dict(completed_steps=1)
        with mock.patch.object(v25, '_rerank_positions', side_effect=scan), \
                mock.patch.object(v25, '_optimize_suffix', side_effect=optimize):
            embedding, result = run(kwargs)
        self.assertEqual(['stage1', 'R_1', 'R_2', 'final_discretization'], [x[1] for x in order if x[0] == 'scan'])
        self.assertTrue(all(not x[2] for x in order if x[0] == 'scan' and x[1] != 'final_discretization'))
        self.assertTrue(torch.equal(entry + .01, embedding))
        self.assertEqual((1, 1), (result['accepted_round_count'], result['rejected_round_count']))
        self.assertIn('R_1', result['final_trajectory_contracts'])
        self.assertNotIn('R_2', result['final_trajectory_contracts'])
        self.assertTrue(all(e['call_id'] == 'final_discretization' for e in result['backtracking']['events']))
        for decision in result['discretization']['events'][result['optimization_decision_count']:]:
            source = decision['trajectory']
            self.assertEqual('R_1', source['call_id'])

    def test_final_phase_has_no_backward_or_optimizer_step(self):
        kwargs, _ = path_fixture(fit=.8)
        with mock.patch.object(torch.Tensor, 'backward', side_effect=AssertionError('final backward')), \
                mock.patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('final Adam step')), \
                mock.patch.object(v25, '_optimize_suffix', side_effect=AssertionError('final R')):
            _, result = run(kwargs)
        self.assertTrue(result['freeze_checks']['passed'])

    def test_mutated_frozen_embedding_or_target_is_fatal(self):
        for mutated in ('embedding', 'target'):
            kwargs, _ = path_fixture(fit=.8)
            original = v25._rerank_positions
            def scan(*args, **options):
                if options.get('call_id') == 'final_discretization':
                    value = args[0] if mutated == 'embedding' else options['target_hidden_state']
                    value.add_(.01)
                return original(*args, **options)
            with self.subTest(mutated=mutated), mock.patch.object(v25, '_rerank_positions', side_effect=scan):
                with self.assertRaisesRegex(v25.DiscretizationContractError, 'freeze boundary'):
                    run(kwargs)

    def test_final_trajectory_contract_rejects_stale_source_prefix(self):
        kwargs = real_fixture(attempts=0)
        kwargs['config'] = v25.SuffixReoptimizationV25Config(**vars(kwargs['config']), backtracking_enabled=False)
        identity = v25.trajectory_identity(kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer'])
        collector = v25.DiscretizationTrajectoryCollector(1000, 1, kwargs['embed_layer'], 'cosine', 'stage1', [0], identity, (500, 600, 700, 800, 900))
        kwargs['stage_candidates'] = collector.finalize(1000, 4)
        original = v25._rerank_positions
        def scan(*args, **options):
            if options.get('call_id') == 'final_discretization':
                options['stage_candidates'][1]['prefix_fingerprint'] = 'stale'
            return original(*args, **options)
        with mock.patch.object(v25, '_rerank_positions', side_effect=scan):
            with self.assertRaisesRegex(v25.DiscretizationContractError, 'trajectory identity'):
                run(kwargs)

    def test_discrete_failure_restores_matching_tokens_text_and_decision_state(self):
        kwargs, _ = path_fixture(best_later, fit=.8, backtracking_prune=False)
        original = kwargs['get_perplexity']
        def failing_trial(tokens, *args, **options):
            if 2 in tokens:
                raise RuntimeError('speculative forward failure')
            return original(tokens, *args, **options)
        kwargs['get_perplexity'] = failing_trial
        _, result = run(kwargs)
        self.assertTrue(any(t['reason'].startswith('trial_failed:') for e in result['backtracking']['events'] for t in e['trials']))
        self.assertTrue(result['freeze_checks']['final_decisions_match_tokens'])
        self.assertEqual(result['final_text'], kwargs['tokenizer'].decode(result['final_tokens'][1:]))


class IntegrationTests(unittest.TestCase):
    def test_continuous_objectives_and_window_metric_are_identical_to_v24(self):
        old = ast.parse((ROOT / 'suffix_optimization_methods/method_versions/suffix_reoptimization_v2_4.py').read_text(encoding='utf-8'))
        new = ast.parse((ROOT / 'suffix_optimization_methods/method_versions/suffix_reoptimization_v2_5.py').read_text(encoding='utf-8'))
        for name in ('optimize_stage1', '_optimize_suffix', '_hidden_cosine_loss', '_window_error'):
            with self.subTest(name=name):
                before = next(node for node in old.body if isinstance(node, ast.FunctionDef) and node.name == name)
                after = next(node for node in new.body if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertEqual(ast.dump(before), ast.dump(after))

    def test_calibration_uses_final_decisions_only_and_preserves_split_boundary(self):
        kwargs, _ = path_fixture(best_later, fit=.8, backtracking_enabled=False)
        _, result = run(kwargs)
        payload = v25.collect_backtracking_calibration(
            [dict(sample_id='independent:cal', result=result,
                  target_hidden_state=kwargs['target_hidden_state'], attention_mask=kwargs['attention_mask'])],
            kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer'], 0, _forward_tokens)
        self.assertEqual(dict(with_prefix=5, empty_prefix=0), payload['score_counts'])
        self.assertEqual(15, payload['numerical_repeat_count'])
        fields = vars(kwargs['config']).copy()
        for name in ('fit_threshold', 'first_fit_threshold', 'acceptance_epsilon', 'consistency_tolerance'):
            fields['backtracking_' + name] = None
        fields['backtracking_calibration'] = payload
        kwargs['config'] = v25.SuffixReoptimizationV25Config(**fields)
        with self.assertRaisesRegex(v25.DiscretizationContractError, 'calibration split'):
            run(dict(kwargs, sample_id='independent:cal'))

    def test_explicit_config_aliases_and_original_entry_rollback(self):
        fn = functions('load_config', 'merge_config', 'canonicalize_config_aliases', 'normalize_suffix_version', 'select_advanced_method')
        values = fn['load_config'](str(ROOT / 'experiment_configs/l24_deml3x4_suffix_v2_5.json'))
        cfg = v25.config_from_mapping(values)
        self.assertEqual((4, 4, 16, 2, 640), (cfg.backtracking_max_distance, cfg.backtracking_alternatives,
            cfg.backtracking_max_trials, cfg.backtracking_max_events, cfg.backtracking_forward_budget))
        for alias in ('2.5', 'v2.5', 'suffix_v2_5', 'suffix_reoptimization_v2_5', 'suffix_reoptimization_v2.5'):
            self.assertEqual(v25.METHOD_NAME, fn['select_advanced_method'](alias, None, None, None, None, suffix_reopt_v2_5_config=cfg))
        old_values = fn['load_config'](str(ROOT / 'experiment_configs/l24_deml3x4_suffix_v2_4.json'))
        self.assertTrue(v24.config_from_mapping(old_values).enabled)
        self.assertEqual('v2.4', old_values['suffix_version'])
        with self.assertRaises(ValueError):
            v25.config_from_mapping({})
        for name, value in (('max_distance', 5), ('alternatives', 5), ('max_trials', 17), ('forward_budget', 641)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                v25.SuffixReoptimizationV25Config(**{'backtracking_' + name: value})

    def test_resolved_config_canonical_offline_and_fixed_logs(self):
        fn = functions('load_config', 'merge_config', 'canonicalize_config_aliases')
        values = fn['load_config'](str(ROOT / 'experiment_configs/l24_deml3x4_suffix_v2_5.json'))
        class Args(types.SimpleNamespace):
            def __getattr__(self, name):
                return None
        args = Args(**values, selected_advanced_method=v25.METHOD_NAME, selected_candidate_reranking_method='none', config='v25.json')
        resolved = build_resolved_config(args, 'time', 'run', 'experiment.log', 'reconstructions.jsonl', None, 1, 24, 'qwen2')
        self.assertEqual(['suffix_reoptimization_v2_5'], list(resolved['advanced_methods']))
        settings = resolved['advanced_methods']['suffix_reoptimization_v2_5']
        self.assertEqual(640, settings['backtracking_forward_budget'])
        self.assertEqual('final_discretization_only_fresh_per_sample', settings['backtracking_extra_budget_scope'])
        self.assertFalse(settings['r_trajectory_candidates_enabled'])
        self.assertEqual((35, 25), (settings['maximum_initial_candidates'], settings['maximum_initial_no_prefix_candidates']))
        self.assertEqual((30, 20), (settings['maximum_r_candidates'], settings['maximum_r_no_prefix_candidates']))
        self.assertEqual(30, settings['maximum_backtracking_r_candidates'])
        kwargs, _ = path_fixture(best_later, fit=.8)
        buffer = io.StringIO()
        _, result = run(dict(kwargs, log_file=buffer))
        self.assertEqual('', buffer.getvalue())
        self.assertFalse(result['gt_accessed'])
        reference = result['final_tokens']
        evaluated = discretization_offline_evaluation(result, reference, 1)
        self.assertEqual(1., evaluated['accuracy'])
        self.assertEqual(5, len(evaluated['final_candidate_coverage']))
        self.assertTrue(all(d['call_id'] == 'final_discretization' for d in evaluated['final_candidate_coverage']))
        record = dict(selected_advanced_method=v25.METHOD_NAME, selected_candidate_reranking_method='none', accuracy=1.,
                      suffix_reoptimization_v2_5_result=dict(result, pre_acc=.5, post_acc=1.))
        self.assertEqual(dict(pre_suffix=.5, suffix_v2_5=1.), build_stage_accuracy(record))
        write_experiment_sample_summary(buffer, record, 1, 1, 5)
        write_experiment_average_summary(buffer, [record])
        self.assertNotIn('backtrack', buffer.getvalue())
        self.assertNotIn('freeze', buffer.getvalue())
        json.loads(json.dumps(result, allow_nan=False))

    def test_new_main_branch_executes_real_stage1_and_method(self):
        kwargs = real_fixture(attempts=1, steps=3)
        entry = kwargs['optimized_embedding']
        cfg = v25.SuffixReoptimizationV25Config(**vars(kwargs['config']), backtracking_enabled=False)
        args = types.SimpleNamespace(base_model_name='tiny', num_invert_layers=0, lr=.01, epoch=3,
            alpha=.001, clip=True, init_method='uniform', init_param=.1, optim_method='cosine', seed=0,
            filter_nonascii=True, perplexity=True, top_k_ppl=2, top_k_cos=2, invert_method='cosine')
        namespace = dict(selected_advanced_method=v25.METHOD_NAME, args=args, tokenizer=kwargs['tokenizer'],
            model=kwargs['model'], embed_layer=kwargs['embed_layer'], eval_start_pos=1,
            dataset_metadata={'name': 'tiny', 'sample_index': 0}, sample_idx=0,
            next_hidden_states_last=kwargs['target_hidden_state'], new_input_embed_0=entry[:, 1:],
            prefix_embed=entry[:, :1], total_attention_mask=kwargs['attention_mask'],
            weight_mask=torch.ones(4), right_range=torch.ones(4), register_layer_hooks=_register_layer_hooks,
            suffix_reopt_v2_5_config=cfg, run_suffix_v25_two_stage=v25.run_two_stage,
            embedding_top_indices=kwargs['embedding_top_indices'], select_candidate_from_top_indices=kwargs['select_candidate_from_top_indices'],
            get_perplexity=kwargs['get_perplexity'], forward_and_get_last_hidden_state=_forward_tokens, os=os)
        branches = [node for node in ast.walk(TREE) if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                    and any(isinstance(part, ast.Constant) and part.value == v25.METHOD_NAME for part in node.test.comparators)]
        branch = next(node for node in branches if any(isinstance(child, ast.Name) and child.id == 'run_suffix_v25_two_stage'
                      for statement in node.body for child in ast.walk(statement)))
        exec(compile(ast.Module(body=branch.body, type_ignores=[]), 'invert_suffix_v2_5.py', 'exec'), namespace)
        self.assertEqual('v2.5', namespace['optimization_result']['version'])
        self.assertTrue(namespace['suffix_reopt_v2_5_result']['freeze_checks']['passed'])

    def test_shared_stage1_snapshot_off_on_and_v24_reader(self):
        kwargs = real_fixture(attempts=1, steps=3)
        entry = kwargs.pop('optimized_embedding')
        fields = vars(kwargs['config']).copy()
        kwargs['config'] = v25.SuffixReoptimizationV25Config(**fields, backtracking_enabled=False)
        stage1 = dict(model=kwargs['model'], initial_embedding=entry[:, 1:], prefix_embedding=entry[:, :1],
            target_hidden_state=kwargs['target_hidden_state'], attention_mask=kwargs['attention_mask'],
            layer_id=0, register_layer_hooks=_register_layer_hooks, weight_mask=torch.ones(4),
            right_range=torch.ones(4), lr=.01, epoch=901, alpha=.001)
        with tempfile.TemporaryDirectory(prefix='suffix_v25_test_', dir=ROOT / 'outputs') as directory:
            shared = dict(stage1_kwargs=stage1, stage2_kwargs=kwargs, snapshot_path=str(Path(directory) / 'shared.pt'),
                          pair_id='tiny:0', snapshot_contract={'block': 0})
            _, off = v25.run_two_stage(**shared, snapshot_mode='write')
            kwargs['config'].backtracking_enabled = True
            with mock.patch.object(v25, 'optimize_stage1', side_effect=AssertionError('must reuse Stage1')):
                _, on = v25.run_two_stage(**shared, snapshot_mode='read')
            self.assertEqual(off['stage1_snapshot_sha256'], on['stage1_snapshot_sha256'])
            self.assertEqual(off['freeze_checks']['embedding_sha256_before'], on['freeze_checks']['embedding_sha256_before'])
            self.assertEqual([500, 600, 700, 800, 900], off['stage1']['discretization_trajectory']['retained_steps'])
            old_kwargs = dict(kwargs, config=v24.SuffixReoptimizationV24Config(**fields, backtracking_enabled=False))
            _, old = v24.run_two_stage(**dict(shared, stage2_kwargs=old_kwargs), snapshot_mode='read')
            self.assertEqual(off['stage1_snapshot_sha256'], old['stage1_snapshot_sha256'])
            snapshot = torch.load(shared['snapshot_path'], weights_only=False)
            snapshot['trajectory_groups'].pop('v2.5')
            torch.save(snapshot, shared['snapshot_path'])
            _, from_v24 = v25.run_two_stage(**shared, snapshot_mode='read')
            self.assertEqual(on['final_tokens'], from_v24['final_tokens'])


if __name__ == '__main__':
    unittest.main()
