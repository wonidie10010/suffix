"""v2.3 contracts, actual tiny-model stages, output wiring, and rollback."""
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_3 as v23
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_2_2_2 as old
from experiment_outputs import (build_resolved_config, build_stage_accuracy,
    discretization_offline_evaluation, write_experiment_sample_summary)
from test.test_suffix_reoptimization_v2_2_1 import (
    _Block, _register_layer_hooks, _forward_tokens, _embedding_top_indices,
    _select_candidate, _get_perplexity)

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT/'invert.py').read_text(encoding='utf-8'))


def functions(*names):
    namespace = dict(os=__import__('os'), SCRIPT_DIR=str(ROOT), json=json, copy=copy)
    nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/'invert.py'), 'exec'), namespace)
    return namespace


class Tokenizer:
    all_special_ids, bos_token_id = [0], 0
    def __init__(self):
        self.texts = ['<bos>']
        for word in ('alpha', 'bravo', 'charlie', 'delta', 'echo'):
            self.texts.extend([word, word.upper(), word.capitalize(), ' '+word, word[:-1], word[1:]])
        self.texts.extend(['  alpha', '\talpha', ' \talpha', '中', '', 'broken', 'alias'])
        self.ids = {text: i for i, text in enumerate(self.texts)}
    def decode(self, ids, clean_up_tokenization_spaces=False):
        assert clean_up_tokenization_spaces is False
        return ''.join(self.texts[int(i)] for i in ids)
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == 'broken':
            return [self.ids['alias']]
        return [self.ids[text]] if text in self.ids else [1, 2]
    def get_vocab(self):
        return self.ids


class Model(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, 4)
        self.layers = torch.nn.ModuleList([_Block()])
        self.lm_head = torch.nn.Linear(4, vocab, bias=False)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(0)
            self.embed_tokens.weight.copy_(torch.randn(vocab, 4, generator=generator)*.1)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        del attention_mask
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(logits=self.lm_head(hidden))


def fixture(length=4, prefix=1, attempts=0, **overrides):
    tokenizer = Tokenizer()
    model = Model(len(tokenizer.texts))
    tokens = ([0] if prefix else [])+[1]*(length-prefix)
    embedding = model.embed_tokens(torch.tensor([tokens], dtype=torch.long)).detach()
    mask = torch.ones((1, length), dtype=torch.long)
    target = _forward_tokens(model, tokens, mask, 0).detach()
    config = dict(enabled=True, steps=50, max_attempts=attempts, range_top_k=2)
    config.update(overrides)
    return dict(model=model, embed_layer=model.embed_tokens, optimized_embedding=embedding,
        target_hidden_state=target, attention_mask=mask, layer_id=0,
        register_layer_hooks=_register_layer_hooks, tokenizer=tokenizer,
        config=v23.SuffixReoptimizationV23Config(**config), fixed_prefix_tokens=[0] if prefix else [],
        eval_start_pos=prefix, top_k_cos=2, top_k_ppl=2,
        embedding_top_indices=_embedding_top_indices, select_candidate_from_top_indices=_select_candidate,
        get_perplexity=_get_perplexity, forward_and_get_last_hidden_state=_forward_tokens)


def collector(kwargs, call='stage1', start=1, steps=1000, prefix=None):
    return v23.DiscretizationTrajectoryCollector(steps, start, kwargs['embed_layer'], 'cosine', call,
        kwargs['fixed_prefix_tokens'] if prefix is None else prefix,
        v23.trajectory_identity(kwargs['model'], kwargs['embed_layer'], kwargs['tokenizer']),
        (500,600,700,800,900) if call == 'stage1' else (30,40))


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()
        self.vocab = len(self.tokenizer.texts)
    def base(self, ids, ppl=(), stages=(), limit=5, ascii=True):
        return v23.build_discretization_base(list(ids), list(ppl), self.tokenizer, self.vocab, ascii,
            dict(call_id='stage1', stages=[dict(step=step, token_ids=tokens) for step,tokens in stages]), limit)
    def variants(self, ids, scores=None, **overrides):
        rows = [dict(token_id=i, score=s, order=r, sources=[]) for r,(i,s) in enumerate(
            zip(ids, scores if scores is not None else [1.-.1*r for r in range(len(ids))]))]
        return v23.build_discretization_variants(rows, self.tokenizer, self.vocab, True,
                                                v23.SuffixReoptimizationV23Config(**overrides))
    def test_stable_original_merge_top1_only_and_duplicate_provenance(self):
        result = self.base([0,1,1,34], [1,7], [(700,[13,8]),(500,[7,8]),(600,[0,9]),(800,[13]),(900,[19])])
        self.assertEqual([0,1,34,7], result['original_candidate_ids'])  # B0 specials/nonascii stay.
        self.assertEqual([13,19], result['trajectory_candidate_ids'])
        self.assertNotIn(8, [row['token_id'] for row in result['candidates']])
        trajectory = next(row for row in result['candidates'] if row['token_id'] == 13)
        self.assertEqual([700,800], [label['step'] for label in trajectory['sources']])
    def test_invalid_top1_does_not_backfill_top2_and_ascii_uses_raw_decode(self):
        stages = [(500,[0,7]),(600,[-1,7]),(700,[self.tokenizer.ids['中'],7]),(800,[True,7]),(900,[1,7])]
        result = self.base([1], stages=stages)
        self.assertEqual([], result['trajectory_candidate_ids'])
        self.assertEqual(5, len(result['exclusions']))
        result = self.base([1], stages=[(500,[self.tokenizer.ids['中']])], ascii=False)
        self.assertEqual([self.tokenizer.ids['中']], result['trajectory_candidate_ids'])
    def test_initial_five_r_two_and_no_quota_borrowing(self):
        result = self.base([1], stages=[(s,[i]) for s,i in zip((500,600,700,800,900),(2,3,7,8,9))])
        self.assertEqual(5, len(result['trajectory_candidate_ids']))
        result = self.base([1], stages=[(30,[2]),(40,[3])])
        self.assertEqual(2, len(result['trajectory_candidate_ids']))
        variants = self.variants([1])
        self.assertEqual(4, len(variants['candidates']))
        self.assertEqual(dict(case=2,space=2,prefix=0,suffix=2), variants['quota_gaps'])
    def test_filter_then_top5_and_finite_scores_stable_seed_order(self):
        invalid = [0,self.tokenizer.ids['中'],self.tokenizer.ids[''],self.tokenizer.ids['broken']]
        result = self.variants(invalid+[1,7,13,19,25,2], [1.]*10)
        self.assertEqual([1,7,13,19,25], [row['token_id'] for row in result['seeds']])
        result = self.variants([1,7,13], [float('nan'),.5,.5])
        self.assertEqual([7,13], [row['token_id'] for row in result['seeds']])
        self.assertIn('nonfinite_seed_score', [row['reason'] for row in result['exclusions']])
    def test_category_order_quotas_one_per_seed_and_roundtrip(self):
        result = self.variants([1,7,13,19,25])
        self.assertEqual(10, len(result['candidates']))
        categories = [row['quota_source'] for row in result['candidates']]
        self.assertEqual(['case']*3+['space']*3+['prefix']+['suffix']*3, categories)
        for category in result['quotas']:
            seeds = [row['sources'][0]['seed_token_id'] for row in result['candidates'] if row['quota_source']==category]
            self.assertEqual(len(seeds),len(set(seeds)))
        for row in result['candidates']:
            self.assertEqual([row['token_id']], self.tokenizer.encode(row['sources'][0]['text']))
    def test_duplicate_invalid_variants_continue_and_cross_category_merges(self):
        result = self.variants([1,2])
        self.assertIn(3, [row['token_id'] for row in result['candidates']])  # lower/upper duplicates, capitalize succeeds.
        tokenizer = self.tokenizer
        seed = tokenizer.ids['  alpha']
        result = self.variants([seed], case_quota=0)
        space = next(row for row in result['candidates'] if row['quota_source']=='space')
        self.assertEqual(' alpha', space['sources'][0]['text'])
        self.assertTrue(any(label['source']=='suffix' for label in space['sources']))
        self.assertEqual('alpha', next(row['sources'][0]['text'] for row in result['candidates'] if row['quota_source']=='suffix'))
    def test_literal_space_and_case_whitespace_are_preserved(self):
        self.assertEqual(['\talpha','\tALPHA','\tAlpha'],list(v23._variant_texts('case','\talpha')))
        self.assertEqual([' \talpha'],list(v23._variant_texts('space','\talpha')))
        self.assertEqual([' alpha'],list(v23._variant_texts('space','  alpha')))
        self.assertEqual(['Abc'],list(v23._variant_texts('case','aBc'))[-1:])
        self.assertEqual([' a',' '],list(v23._variant_texts('prefix',' ab')))


class TrajectoryTests(unittest.TestCase):
    def test_completed_samples_strict_endpoint_offset_and_absent_points(self):
        kwargs = fixture(length=5)
        capture = collector(kwargs)
        for step in (334,500,600,700,800,900,1000):
            capture(step, kwargs['optimized_embedding'])
        rows = capture.finalize(900,5)
        self.assertEqual([500,600,700,800], [s['step'] for s in rows[1]['stages']])
        self.assertEqual([500,600,700,800,900], capture.metadata()['captured_steps'])
        self.assertTrue(all(len(s['token_ids'])==1 for row in rows.values() for s in row['stages']))
        r = collector(kwargs, 'R_1', 3, 50, [0,1,1])
        for step in (17,30,34,40,50):
            r(step, kwargs['optimized_embedding'])
        self.assertEqual([3,4], list(r.finalize(40,5)))
        self.assertEqual([30], [s['step'] for s in r.finalize(40,5)[3]['stages']])
        missing = collector(kwargs, steps=10)
        self.assertEqual([], missing.finalize(10,5)[1]['stages'])
    def test_top1_distance_ties_use_smallest_id(self):
        embedding = torch.nn.Embedding(3,2)
        with torch.no_grad():
            embedding.weight.copy_(torch.tensor([[1.,0.],[1.,0.],[0.,1.]]))
        for metric in ('cosine','L2'):
            self.assertEqual([[0]],v23.stage_neighbours(torch.tensor([[[1.,0.]]]),embedding,1,metric))
    def test_identity_call_prefix_and_position_fail_as_contract_errors(self):
        kwargs = fixture()
        capture = collector(kwargs)
        rows = capture.finalize(10,4)
        for field,value in [('call_id','R_1'),('position',2),('source_start',0),('prefix_fingerprint','other'),
                            ('identity',{}),('planned_sampling_steps',[334,667])]:
            broken = copy.deepcopy(rows)
            broken[1][field] = value
            with self.subTest(field=field),self.assertRaises(v23.DiscretizationContractError):
                v23.run_suffix_reoptimization_v2_3(**dict(kwargs,stage_candidates=broken))


class FlowTests(unittest.TestCase):
    def test_next_prefix_and_score_reuse_batching_and_empty_variants(self):
        kwargs = fixture(prefix=0, length=3, seed_top_k=0, candidate_batch_size=1)
        calls, prefixes = [], []
        def forward(model, ids, mask, layer_id):
            self.assertFalse(torch.is_grad_enabled())
            calls.extend(ids)
            values = torch.tensor(ids)
            scores = torch.zeros(len(ids),len(ids[0]),4)
            scores[:,:,0] = (values==7).float()
            scores[:,:,1] = (values!=7).float()
            return scores
        def ppl(prefix,*args,**kw):
            prefixes.append(list(prefix))
            return None,torch.tensor([1,7])
        kwargs.update(target_hidden_state=torch.tensor([[[1.,0.,0.,0.]]*3]),
                      get_perplexity=ppl,forward_and_get_last_hidden_state=forward)
        with mock.patch.object(v23,'_candidate_token_ids',return_value=(1,[1,1,7])):
            _,result = v23.run_suffix_reoptimization_v2_3(**kwargs)
        self.assertEqual([[7],[7,7]],prefixes)
        self.assertEqual([7,7,7],result['final_tokens'])
        self.assertEqual(6,len(calls))
        self.assertEqual(6,result['compute_stats']['candidate']['candidate_count'])
        self.assertEqual(2,result['compute_stats']['ppl']['forward_calls'])
        self.assertEqual(18,result['compute_stats']['candidate']['token_count'])
        self.assertTrue(all(not e['variant_candidates'] for e in result['discretization']['events']))
    def test_tie_preserves_base_nonfinite_new_excluded_nonfinite_original_fatal(self):
        for mode in ('tie','nan_variant','nan_original','nan_trajectory'):
            kwargs = fixture(length=2)
            kwargs['target_hidden_state'] = torch.tensor([[[1.,0.,0.,0.]]*2])
            capture = collector(kwargs)
            capture.snapshots[500] = [[7]]
            kwargs['stage_candidates'] = capture.finalize(1000,2)
            def forward(model, ids, *args, **kw):
                states = torch.tensor([[[1.,0.,0.,0.]]*2]*len(ids))
                for i,row in enumerate(ids):
                    invalid = ((mode=='nan_variant' and row[1] not in (1,7)) or
                               (mode=='nan_original' and row[1]==1) or (mode=='nan_trajectory' and row[1]==7))
                    if invalid: states[i,1] = float('nan')
                return states
            kwargs.update(forward_and_get_last_hidden_state=forward,add_perplexity=False)
            with mock.patch.object(v23,'_candidate_token_ids',return_value=(1,[1])):
                if mode=='nan_original':
                    with self.assertRaises(v23.DiscretizationContractError):
                        v23.run_suffix_reoptimization_v2_3(**kwargs)
                else:
                    _,result = v23.run_suffix_reoptimization_v2_3(**kwargs)
                    self.assertEqual(1,result['final_tokens'][1])
                    json.dumps(result,allow_nan=False)
                    if mode=='tie': self.assertTrue(result['discretization']['events'][0]['tie_kept_base'])
    def test_r_entire_rollback_budget_and_final_retention(self):
        for mode in ('reject','accept','error','oom','contract'):
            kwargs = fixture(attempts=2)
            entry = kwargs['optimized_embedding'].clone()
            def optimize(*args,**kw):
                if mode in ('oom','contract'):
                    raise (torch.cuda.OutOfMemoryError('oom') if mode=='oom' else v23.DiscretizationContractError('bad offset'))
                if mode=='error': raise RuntimeError('ordinary trial')
                kw['trajectory_callback'](30,args[1])
                kw['trajectory_callback'](40,args[1])
                return args[1]+.01,1.,.5 if mode=='accept' else 1.,dict(completed_steps=50)
            with mock.patch.object(v23,'_optimize_suffix',side_effect=optimize):
                if mode in ('oom','contract'):
                    with self.assertRaises((torch.cuda.OutOfMemoryError,v23.DiscretizationContractError)):
                        v23.run_suffix_reoptimization_v2_3(**kwargs)
                    continue
                final,result = v23.run_suffix_reoptimization_v2_3(**kwargs)
            self.assertEqual(2,result['attempt_count'])
            if mode in ('reject','error'):
                self.assertTrue(torch.equal(entry,final))
                self.assertEqual(result['pre_tokens'],result['final_tokens'])
                self.assertEqual(2,result['rejected_round_count'])
                self.assertTrue(all(not e['accepted_into_formal_state'] and e['rolled_back']
                    for e in result['discretization']['events'] if e['phase']=='R'))
            final_events = [e for e in result['discretization']['events'] if e['final_retained']]
            self.assertEqual(3,len(final_events))
            if mode=='accept':
                self.assertEqual(['R_1','R_2','R_2'],[e['call_id'] for e in final_events])
                self.assertTrue(any(e['accepted_into_formal_state'] and not e['final_retained']
                                    for e in result['discretization']['events']))
            offline = discretization_offline_evaluation(result,[0,1,1,1],1)
            self.assertEqual(3,len(offline['final_candidate_coverage']))
    def test_real_r_uses_own_absolute_30_40_trajectory_and_no_log_write(self):
        kwargs = fixture(attempts=1)
        buffer = io.StringIO()
        _,result = v23.run_suffix_reoptimization_v2_3(**dict(kwargs,log_file=buffer))
        self.assertEqual('',buffer.getvalue())
        self.assertEqual([],result['anomaly_reasons'])
        self.assertEqual([30,40],result['events'][0]['anchor']['discretization_trajectory']['retained_steps'])
        self.assertEqual(52,result['compute_stats']['r_optimization']['forward_calls'])
    def test_recoverable_r_forward_failure_removes_leaked_hooks(self):
        kwargs = fixture(attempts=1)
        real = kwargs['forward_and_get_last_hidden_state']
        calls = 0
        def forward(*args,**kw):
            nonlocal calls
            calls += 1
            if calls > 6:
                kwargs['model'].layers[0].register_forward_hook(lambda *a: None)
                raise RuntimeError('local candidate forward')
            return real(*args,**kw)
        with mock.patch.object(v23,'_candidate_token_ids',return_value=(1,[1])), \
             mock.patch.object(v23,'_optimize_suffix',return_value=(kwargs['optimized_embedding']+.01,1.,.5,dict(completed_steps=50))):
            _,result = v23.run_suffix_reoptimization_v2_3(**dict(kwargs,forward_and_get_last_hidden_state=forward))
        self.assertEqual({},kwargs['model'].layers[0]._forward_hooks)
        self.assertEqual(result['pre_tokens'],result['final_tokens'])


class IntegrationTests(unittest.TestCase):
    def test_configs_selection_aliases_and_rollback(self):
        fn = functions('load_config','merge_config','canonicalize_config_aliases','normalize_suffix_version','select_advanced_method')
        config = fn['load_config'](str(ROOT/'experiment_configs/l24_deml3x4_suffix_v2_3.json'))
        cfg = v23.config_from_mapping(config)
        select = fn['select_advanced_method']
        for alias in ('2.3','v2.3','suffix_v2_3','suffix_reoptimization_v2_3','suffix_reoptimization_v2.3'):
            self.assertEqual(v23.METHOD_NAME,select(alias,None,None,None,None,suffix_reopt_v2_3_config=cfg))
        self.assertEqual(old.METHOD_NAME,select('v2.2.2(2)',None,None,None,None,
            suffix_reopt_v2_2_2_2_config=old.SuffixReoptimizationV222_2Config(enabled=True)))
        with self.assertRaises(ValueError): select('v2.3',None,None,None,None)
        with self.assertRaises(ValueError): v23.config_from_mapping({})
        with self.assertRaises(ValueError): v23.SuffixReoptimizationV23Config(stage1_sampling_steps=[334,667])
        for name,value in [('trajectory_max_new',6),('seed_top_k',True),('case_quota',4),('candidate_batch_size',0)]:
            with self.assertRaises(ValueError): v23.SuffixReoptimizationV23Config(**{name:value})
    def test_resolved_config_and_fixed_accuracy_log(self):
        fn = functions('load_config','merge_config','canonicalize_config_aliases')
        config = fn['load_config'](str(ROOT/'experiment_configs/l24_deml3x4_suffix_v2_3.json'))
        class Args(types.SimpleNamespace):
            def __getattr__(self,name): return None
        args = Args(**config, selected_advanced_method=v23.METHOD_NAME,
            selected_candidate_reranking_method='none',config='v23.json')
        resolved = build_resolved_config(args,'time','run','experiment.log','reconstructions.jsonl',None,1,24,'qwen2')
        self.assertEqual(['suffix_reoptimization_v2_3'],list(resolved['advanced_methods']))
        self.assertEqual([500,600,700,800,900],resolved['advanced_methods']['suffix_reoptimization_v2_3']['stage1_sampling_steps'])
        settings = resolved['advanced_methods']['suffix_reoptimization_v2_3']
        self.assertEqual((35,32,25,22),tuple(settings[name] for name in (
            'maximum_initial_candidates','maximum_r_candidates',
            'maximum_initial_no_prefix_candidates','maximum_r_no_prefix_candidates')))
        args.top_k_cos, args.perplexity, args.epoch = 2, False, 3
        args.suffix_v2_3_seed_top_k, args.suffix_v2_3_steps = 1, 20
        smaller = build_resolved_config(args,'time','run','experiment.log','reconstructions.jsonl',None,1,24,'qwen2')
        self.assertEqual(6,smaller['advanced_methods']['suffix_reoptimization_v2_3']['maximum_initial_candidates'])
        self.assertEqual(6,smaller['advanced_methods']['suffix_reoptimization_v2_3']['maximum_r_candidates'])
        record = dict(selected_advanced_method=v23.METHOD_NAME,selected_candidate_reranking_method='none',
                      accuracy=.75,suffix_reoptimization_v2_3_result=dict(pre_acc=.5,post_acc=.75))
        self.assertEqual(dict(pre_suffix=.5,suffix_v2_3=.75),build_stage_accuracy(record))
        buffer = io.StringIO()
        write_experiment_sample_summary(buffer,record,1,1,4)
        self.assertNotIn('candidate',buffer.getvalue())
        self.assertNotIn('trajectory',buffer.getvalue())
    def test_new_version_uses_existing_model_and_cgmr_contracts(self):
        fn = functions('validate_advanced_candidate_combination','validate_suffix_v21_model_config')
        self.assertTrue(fn['validate_advanced_candidate_combination'](v23.METHOD_NAME,'none'))
        with self.assertRaises(ValueError):
            fn['validate_advanced_candidate_combination'](v23.METHOD_NAME,'CGMR_v1.2')
        model = types.SimpleNamespace(model_type='qwen2',architectures=['Qwen2ForCausalLM'])
        self.assertTrue(fn['validate_suffix_v21_model_config'](v23.METHOD_NAME,model))
        with self.assertRaises(ValueError):
            fn['validate_suffix_v21_model_config'](v23.METHOD_NAME,types.SimpleNamespace(model_type='llama'))
    def test_real_stage1_grouped_snapshot_new_and_old_readers(self):
        torch.set_num_threads(1)
        kwargs = fixture(attempts=1)
        entry = kwargs.pop('optimized_embedding')
        stage1 = dict(model=kwargs['model'],initial_embedding=entry[:,1:],prefix_embedding=entry[:,:1],
            target_hidden_state=kwargs['target_hidden_state'],attention_mask=kwargs['attention_mask'],
            layer_id=0,register_layer_hooks=_register_layer_hooks,weight_mask=torch.ones(4),
            right_range=torch.ones(4),lr=.01,epoch=1000,alpha=.001)
        (ROOT/'outputs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='suffix_v23_snapshot_',dir=ROOT/'outputs') as directory:
            path = str(Path(directory)/'shared.pt')
            shared = dict(stage1_kwargs=stage1,stage2_kwargs=kwargs,snapshot_path=path,pair_id='tiny:0',snapshot_contract={'block':0})
            final,a = v23.run_two_stage(**shared,snapshot_mode='write')
            snapshot = torch.load(path,weights_only=False)
            self.assertEqual([334,667],snapshot['stage1']['checkpoint_trajectory']['captured_steps'])
            self.assertEqual([500,600,700,800,900],a['stage1']['discretization_trajectory']['retained_steps'])
            with mock.patch.object(v23,'optimize_stage1',side_effect=AssertionError('must reuse Stage-1')):
                repeated,b = v23.run_two_stage(**shared,snapshot_mode='read')
            self.assertTrue(torch.equal(final,repeated))
            self.assertEqual(a['final_tokens'],b['final_tokens'])
            self.assertTrue(b['stage1_reused'])
            old_kwargs = dict(kwargs,config=old.SuffixReoptimizationV222_2Config(enabled=True,max_attempts=0))
            _,control = old.run_two_stage(**dict(shared,stage2_kwargs=old_kwargs),snapshot_mode='read')
            self.assertEqual(a['stage1_snapshot_sha256'],control['stage1_snapshot_sha256'])
            snapshot['trajectory_groups'].pop('v2.3')
            torch.save(snapshot,path)
            with self.assertRaises(v23.DiscretizationContractError): v23.run_two_stage(**shared,snapshot_mode='read')
    def test_main_stage_branch_runs_v23_and_records_stage1(self):
        kwargs = fixture(attempts=0)
        args = types.SimpleNamespace(base_model_name='tiny',num_invert_layers=0,lr=.01,epoch=3,
            alpha=.001,clip=True,init_method='uniform',init_param=.1,optim_method='cosine',seed=0,
            filter_nonascii=True,perplexity=True,top_k_ppl=2,top_k_cos=2,invert_method='cosine')
        namespace = dict(selected_advanced_method=v23.METHOD_NAME,args=args,tokenizer=kwargs['tokenizer'],
            model=kwargs['model'],embed_layer=kwargs['embed_layer'],eval_start_pos=1,
            dataset_metadata={'name':'tiny','sample_index':0},sample_idx=0,
            next_hidden_states_last=kwargs['target_hidden_state'],new_input_embed_0=kwargs['optimized_embedding'][:,1:],
            prefix_embed=kwargs['optimized_embedding'][:,:1],total_attention_mask=kwargs['attention_mask'],
            weight_mask=torch.ones(4),right_range=torch.ones(4),register_layer_hooks=_register_layer_hooks,
            suffix_reopt_v2_3_config=kwargs['config'],run_suffix_v23_two_stage=v23.run_two_stage,
            embedding_top_indices=_embedding_top_indices,select_candidate_from_top_indices=_select_candidate,
            get_perplexity=_get_perplexity,forward_and_get_last_hidden_state=_forward_tokens,os=__import__('os'))
        branches = [node for node in ast.walk(TREE) if isinstance(node,ast.If) and
            isinstance(node.test,ast.Compare) and any(isinstance(part,ast.Constant) and
            part.value==v23.METHOD_NAME for part in node.test.comparators)]
        branch = next(node for node in branches if any(isinstance(child,ast.Name) and
            child.id=='run_suffix_v23_two_stage' for child in ast.walk(node)))
        exec(compile(ast.Module(body=branch.body,type_ignores=[]),str(ROOT/'invert.py'),'exec'),namespace)
        self.assertEqual('v2.3',namespace['optimization_result']['version'])
        self.assertEqual('stage1_then_suffix_v2_3',namespace['optimization_result']['stage'])
        self.assertEqual(4,len(namespace['suffix_reopt_v2_3_result']['final_tokens']))


if __name__ == '__main__':
    unittest.main()
