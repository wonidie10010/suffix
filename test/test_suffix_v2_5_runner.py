"""Validate the paired runner against actual sidecar outputs and shared snapshots."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from experiment_outputs_suffix_v2_5 import discretization_offline_evaluation
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_5 as method
from test.test_suffix_reoptimization_v2_5 import immediate, path_fixture

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "实验/环境和实验/内部文件"
sys.path.insert(0, str(RUNNERS))
spec = importlib.util.spec_from_file_location("v25_runner", RUNNERS / "runner_suffix_v2_5.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PairedRunnerTests(unittest.TestCase):
    def make_pair(self):
        kwargs, _ = path_fixture(immediate, fit=.8)
        entry = kwargs.pop("optimized_embedding")
        stage1 = dict(model=kwargs["model"], initial_embedding=entry[:, 1:], prefix_embedding=entry[:, :1],
            target_hidden_state=kwargs["target_hidden_state"], attention_mask=kwargs["attention_mask"],
            layer_id=0, register_layer_hooks=kwargs["register_layer_hooks"], weight_mask=torch.ones(entry.shape[1]),
            right_range=torch.ones(4), lr=.01, epoch=3, alpha=.001)
        kwargs["config"].backtracking_enabled = False
        with tempfile.TemporaryDirectory(prefix="suffix_v25_runner_", dir=ROOT / "outputs") as directory:
            shared = dict(stage1_kwargs=stage1, stage2_kwargs=kwargs, snapshot_path=str(Path(directory) / "shared.pt"),
                          pair_id="tiny:0", snapshot_contract={"block": 0})
            _, off = method.run_two_stage(**shared, snapshot_mode="write")
            kwargs["config"].backtracking_enabled = True
            _, on = method.run_two_stage(**shared, snapshot_mode="read")
        reference = [0, 2, 3, 3, 1, 1]
        records = []
        for result in (off, on):
            records.append(dict(pair_id="tiny:0", dataset={"name": "tiny"},
                suffix_reoptimization_v2_5_result=result,
                discretization_offline_evaluation=discretization_offline_evaluation(result, reference, 1)))
        return records

    def test_actual_off_on_outputs_and_compute_fields(self):
        off, on = self.make_pair()
        compared = runner.compare_pair([off], [on])
        self.assertEqual(5, compared["totals"]["on"]["token_count"])
        self.assertEqual(0, compared["totals"]["off"]["extra_forward_calls"])
        self.assertGreater(compared["totals"]["on"]["extra_forward_calls"], 0)
        self.assertGreater(compared["totals"]["on"]["correct"], compared["totals"]["off"]["correct"])
        self.assertFalse(compared["equal_compute_budget"])

    def test_rejects_changed_reference(self):
        off, on = self.make_pair()
        on = copy.deepcopy(on)
        on["discretization_offline_evaluation"]["reference_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "reference_sha256"):
            runner.compare_pair([off], [on])

    def test_rejects_changed_shared_snapshot(self):
        off, on = self.make_pair()
        on[runner.KEY]["stage1_snapshot_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "snapshot"):
            runner.compare_pair([off], [on])

    def test_rejects_changed_frozen_hash(self):
        off, on = self.make_pair()
        on[runner.KEY]['freeze_checks'].update(embedding_sha256_before='changed', embedding_sha256_after='changed')
        with self.assertRaisesRegex(ValueError, 'paired frozen state'):
            runner.compare_pair([off], [on])

    def test_rejects_changed_optimization_tokens(self):
        off, on = self.make_pair()
        on[runner.KEY]['optimization_final_tokens'][1] += 1
        with self.assertRaisesRegex(ValueError, 'optimization_final_tokens'):
            runner.compare_pair([off], [on])

    def test_rejects_changed_final_decision(self):
        off, on = self.make_pair()
        decision = next(d for d in on[runner.KEY]['discretization']['events'] if d['final_retained'])
        decision['selected_token_id'] += 1
        with self.assertRaisesRegex(ValueError, 'token/decision'):
            runner.compare_pair([off], [on])

    def test_rejects_incorrect_complete_route_selection(self):
        off, on = self.make_pair()
        event = next(e for e in on[runner.KEY]['backtracking']['events'] if e['accepted'])
        event['E_selected'] = event['E_base']
        with self.assertRaisesRegex(ValueError, 'minimum complete route'):
            runner.compare_pair([off], [on])

    def test_real_R_pair_and_changed_R_loss(self):
        from test.test_suffix_reoptimization_v2_3 import fixture
        kwargs = fixture(attempts=2, steps=3)
        fields = vars(kwargs['config'])
        kwargs['config'] = method.SuffixReoptimizationV25Config(**fields, backtracking_enabled=False)
        _, off = method.run_suffix_reoptimization_v2_5(**kwargs)
        kwargs['config'].backtracking_enabled = True
        _, on = method.run_suffix_reoptimization_v2_5(**kwargs)
        self.assertTrue(runner.validate_optimization_pair(off, on)['all_R_identical'])
        on['events'][0]['post_hidden_loss'] += .01
        with self.assertRaisesRegex(ValueError, 'R losses'):
            runner.validate_optimization_pair(off, on)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
