"""Offline CPU CLI smoke with a locally generated tiny Qwen2 checkpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


class EntrySmokeTests(unittest.TestCase):
    def test_cli_off_on_artifacts_and_frozen_r_pair(self):
        # A process that performs autograd R tests and then loads the tiny Qwen
        # inference fixture can exit with 0xC0000005 on this Windows runtime.
        # Keep the model fixture in its own interpreter and check its exit code.
        if os.environ.get('DEML_V25_CLI_SMOKE_WORKER') != '1':
            completed = subprocess.run(
                [sys.executable, '-B', '-m', 'unittest', 'test.test_suffix_v2_5_entry'],
                cwd=ROOT, env=dict(os.environ, DEML_V25_CLI_SMOKE_WORKER='1',
                    PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1'),
                capture_output=True, text=True, encoding='utf-8', timeout=180)
            self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
            return
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_5 as method
        import invert_suffix_v2_5 as entry
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='suffix_v25_cli_', dir=ROOT / 'outputs') as directory:
            base = Path(directory)
            (base / 'temp').mkdir()
            model_path = base / 'tiny_qwen2'
            vocabulary = {'[UNK]': 0, '[BOS]': 1, '[EOS]': 2, '[PAD]': 3,
                          'alpha': 4, 'bravo': 5, 'charlie': 6, 'delta': 7, 'echo': 8,
                          'foxtrot': 9, 'golf': 10, 'hotel': 11, 'india': 12, 'juliet': 13}
            backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            backend.post_processor = TemplateProcessing(single='[BOS] $A', special_tokens=[('[BOS]', 1)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                bos_token='[BOS]', eos_token='[EOS]', pad_token='[PAD]', clean_up_tokenization_spaces=False)
            tokenizer.save_pretrained(model_path)
            torch.manual_seed(0)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=3, use_cache=False))
            model.save_pretrained(model_path)
            # Synthetic observations exercise the CLI's calibrated-On wiring.
            # These fixture thresholds are not a real-data calibration result.
            tokenizer, model = entry.get_model(str(model_path), None, device_map='cpu', quantization='none')
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            identity = method.trajectory_identity(model, entry.get_input_embedding_layer(model), tokenizer)
            calibration = method.build_backtracking_calibration(
                [dict(sample_id='synthetic_calibration:0', has_prefix=True, top1_score=1.)],
                [dict(sample_id='synthetic_calibration:0', window_errors=[.25, .25])],
                ['synthetic_calibration:0'], method.backtracking_inference_signature(model, identity, 0))
            data_path = base / 'samples.json'
            data_path.write_text(json.dumps(['alpha bravo charlie', 'delta echo alpha']), encoding='utf-8')
            pairs = []
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                PYTHONUTF8='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                TEMP=str(base / 'temp'), TMP=str(base / 'temp'))
            for enabled in (False, True):
                label = 'on' if enabled else 'off'
                config = dict(include_configs=[str(ROOT / 'experiment_configs/l24_deml3x4_suffix_v2_5.json')],
                    base_model_name=str(model_path), device_map='cpu', num_invert_layers=0, offload_state_dict=False,
                    epoch=3, lr=.001, top_k_cos=3, top_k_ppl=2, alpha=.001,
                    datasets=[dict(name='Tiny', path=str(data_path), type='local', len=2)],
                    dataset_path=str(data_path), dataset_type='local', dataset_len=2,
                    log_dir=str(base / label), output_dir='tiny_' + label,
                    suffix_v222_snapshot_dir=str(base / 'snapshots'),
                    suffix_v222_snapshot_mode='read' if enabled else 'write',
                    suffix_v2_5_steps=3, suffix_v2_5_max_attempts=1, suffix_v2_5_lr=.001,
                    suffix_v2_5_backtracking_enabled=enabled,
                    suffix_v2_5_backtracking_calibration=calibration)
                path = base / (label + '.json')
                path.write_text(json.dumps(config), encoding='utf-8')
                completed = subprocess.run([sys.executable, '-B', str(ROOT / 'invert_suffix_v2_5.py'), '--config', str(path)],
                    cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
                self.assertEqual(0, completed.returncode, completed.stdout + completed.stderr)
                resolved_files = list((base / label).rglob('resolved_config.json'))
                self.assertEqual(1, len(resolved_files), completed.stdout + completed.stderr)
                run_dir = resolved_files[0].parent
                resolved = json.loads(resolved_files[0].read_text(encoding='utf-8'))
                self.assertEqual(method.METHOD_NAME, resolved['advanced_method']['name'])
                settings = resolved['advanced_methods']['suffix_reoptimization_v2_5']
                self.assertEqual(enabled, settings['backtracking_enabled'])
                self.assertEqual(640, settings['backtracking_forward_budget'])
                records = [json.loads(line) for line in (run_dir / 'reconstructions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                self.assertEqual(2, len(records))
                self.assertFalse(list(run_dir.glob('*.xlsx')))
                log = (run_dir / 'experiment.log').read_text(encoding='utf-8')
                self.assertNotIn('backtracking', log)
                self.assertNotIn('freeze', log)
                for record in records:
                    result = record['suffix_reoptimization_v2_5_result']
                    self.assertTrue(result['freeze_checks']['passed'])
                    self.assertEqual(0, result['freeze_checks']['optimization_backtracking_events'])
                    self.assertEqual(0, result['freeze_checks']['grad_enabled_forward_calls'])
                    self.assertFalse(result['gt_accessed'])
                    self.assertTrue(record['discretization_offline_evaluation']['evaluated_after_online_return'])
                    self.assertEqual(len(result['final_tokens']) - result['eval_start_pos'],
                                     len(record['discretization_offline_evaluation']['final_candidate_coverage']))
                pairs.append(records)
            self.assertGreater(sum(r['suffix_reoptimization_v2_5_result']['backtracking']['event_count'] for r in pairs[1]), 0)
            for off, on in zip(*pairs):
                a, b = off['suffix_reoptimization_v2_5_result'], on['suffix_reoptimization_v2_5_result']
                self.assertEqual(a['stage1_snapshot_sha256'], b['stage1_snapshot_sha256'])
                self.assertEqual(a['freeze_checks']['embedding_sha256_before'], b['freeze_checks']['embedding_sha256_before'])
                self.assertEqual(a['optimization_final_tokens'], b['optimization_final_tokens'])
                self.assertEqual([(e['pre_hidden_loss'], e['post_hidden_loss'], e['accepted']) for e in a['events']],
                                 [(e['pre_hidden_loss'], e['post_hidden_loss'], e['accepted']) for e in b['events']])


if __name__ == '__main__':
    unittest.main()
