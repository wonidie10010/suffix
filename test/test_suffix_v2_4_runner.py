"""Validate the paired runner against actual sidecar outputs and shared snapshots."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from experiment_outputs import discretization_offline_evaluation
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_4 as method
from test.test_suffix_reoptimization_v2_4 import immediate, path_fixture

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ROOT / "实验/环境和实验/内部文件"
sys.path.insert(0, str(RUNNERS))
spec = importlib.util.spec_from_file_location("v24_runner", RUNNERS / "runner_suffix_v2_4.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PairedRunnerTests(unittest.TestCase):
    def make_pair(self):
        kwargs, _ = path_fixture(immediate)
        entry = kwargs.pop("optimized_embedding")
        stage1 = dict(model=kwargs["model"], initial_embedding=entry[:, 1:], prefix_embedding=entry[:, :1],
            target_hidden_state=kwargs["target_hidden_state"], attention_mask=kwargs["attention_mask"],
            layer_id=0, register_layer_hooks=kwargs["register_layer_hooks"], weight_mask=torch.ones(entry.shape[1]),
            right_range=torch.ones(4), lr=.01, epoch=3, alpha=.001)
        kwargs["config"].backtracking_enabled = False
        with tempfile.TemporaryDirectory(prefix="suffix_v24_runner_", dir=ROOT / "outputs") as directory:
            shared = dict(stage1_kwargs=stage1, stage2_kwargs=kwargs, snapshot_path=str(Path(directory) / "shared.pt"),
                          pair_id="tiny:0", snapshot_contract={"block": 0})
            _, off = method.run_two_stage(**shared, snapshot_mode="write")
            kwargs["config"].backtracking_enabled = True
            _, on = method.run_two_stage(**shared, snapshot_mode="read")
        reference = [0, 2, 3, 3, 1, 1]
        records = []
        for result in (off, on):
            records.append(dict(pair_id="tiny:0", dataset={"name": "tiny"},
                suffix_reoptimization_v2_4_result=result,
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


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
