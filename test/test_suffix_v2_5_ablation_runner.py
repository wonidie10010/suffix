"""Check ablation pairing with real tiny-model outputs and intentional contract violations."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from experiment_outputs_suffix_v2_5 import discretization_offline_evaluation
from suffix_optimization_methods.method_versions import suffix_reoptimization_v2_5 as method
from test.test_suffix_reoptimization_v2_5 import immediate, path_fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "实验/环境和实验/内部文件"))
import runner_suffix_v2_5_ablation as runner


class AblationRunnerTests(unittest.TestCase):
    def make_groups(self):
        kwargs, _ = path_fixture(immediate, fit=.8)
        entry = kwargs.pop("optimized_embedding")
        stage1 = dict(model=kwargs["model"], initial_embedding=entry[:, 1:], prefix_embedding=entry[:, :1],
            target_hidden_state=kwargs["target_hidden_state"], attention_mask=kwargs["attention_mask"],
            layer_id=0, register_layer_hooks=kwargs["register_layer_hooks"], weight_mask=torch.ones(entry.shape[1]),
            right_range=torch.ones(4), lr=.01, epoch=3, alpha=.001)
        groups = {}
        with tempfile.TemporaryDirectory(prefix="suffix_v25_ablation_", dir=ROOT / "outputs") as directory:
            shared = dict(stage1_kwargs=stage1, stage2_kwargs=kwargs, snapshot_path=str(Path(directory) / "shared.pt"),
                          pair_id="tiny:0", snapshot_contract={"block": 0})
            for label, (r, e, b) in runner.ARMS.items():
                kwargs["config"].reoptimization_enabled = r
                kwargs["config"].expansion_enabled = e
                kwargs["config"].backtracking_enabled = b
                _, result = method.run_two_stage(**shared, snapshot_mode="write" if label == "baseline" else "read")
                offline = discretization_offline_evaluation(result, [0, 2, 3, 3, 1, 1], 1)
                groups[label] = [dict(pair_id="tiny:0", original="fixed original", dataset={"name": "tiny"},
                    accuracy=offline["accuracy"], suffix_reoptimization_v2_5_result=result,
                    discretization_offline_evaluation=offline)]
        return groups

    def test_actual_cumulative_groups_reconcile_repairs_damage_and_shared_snapshot(self):
        result = runner.compare(self.make_groups())
        self.assertTrue(result["B_pair_all_R_and_frozen_states_identical"])
        self.assertEqual(5, result["totals"]["baseline"]["token_count"])
        for delta in result["adjacent"]:
            self.assertEqual(delta["net_correct"], delta["repairs"] - delta["damage"])

    def test_changed_snapshot_or_reference_is_rejected(self):
        original = self.make_groups()
        for kind in ("snapshot", "reference"):
            groups = copy.deepcopy(original)
            row = groups["r"][0]
            if kind == "snapshot":
                row[runner.KEY]["stage1_snapshot_sha256"] = "changed"
            else:
                row["discretization_offline_evaluation"]["reference_sha256"] = "changed"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.compare(groups)

    def test_enabled_expansion_leak_in_baseline_is_rejected(self):
        result = self.make_groups()["baseline"][0][runner.KEY]
        result["discretization"]["events"][0]["trajectory_candidate_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "E candidates"):
            runner.validate_components(result, runner.ARMS["baseline"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
