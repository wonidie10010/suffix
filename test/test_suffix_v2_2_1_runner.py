import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    PROJECT_DIR
    / "实验"
    / "环境和实验"
    / "内部文件"
    / "runner_suffix_v2_2_1.py"
)
SPEC = importlib.util.spec_from_file_location("runner_suffix_v2_2_1", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def record(dataset_name, sample_index, method, accuracy=0.5, pre_acc=None):
    suffix = {
        "name": method,
        "version": "v2.2.1" if "2.2.1" in method else "frozen-original-v1",
        "pre_acc": pre_acc,
        "formal_gt_blind": method == "suffix_reoptimization_v2.2.1",
        "gt_accessed": False,
    }
    if method == "suffix_reoptimization_v2.2.1":
        suffix.update({
            "pre_tokens": [1, 2],
            "final_tokens": [1, 2],
            "attempt_count": 1,
            "trigger_count": 1,
            "accepted": True,
            "pre_hidden_loss": 0.3,
            "final_hidden_loss": 0.2,
        })
    return {
        "dataset": {"name": dataset_name, "sample_index": sample_index},
        "selected_advanced_method": method,
        "selected_candidate_reranking_method": "none",
        "accuracy": accuracy,
        "token_length": 2,
        "eval_start_pos": 0,
        "optimization_result": {
            "tokens": [1, 2],
        },
        "suffix_reoptimization_result": suffix,
    }


class RunnerTests(unittest.TestCase):
    def test_effective_config_does_not_modify_official_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            official = root / "official.json"
            official.write_text('{"epoch": 1000}\n', encoding="utf-8")
            output = runner.write_effective_config(
                root,
                root / "bundle",
                "baseline",
                official,
                "/model",
                root / "runs",
            )
            self.assertEqual({"epoch": 1000}, json.loads(official.read_text()))
            effective = json.loads(output.read_text())
            self.assertEqual([str(official.resolve())], effective["include_configs"])
            self.assertEqual("/model", effective["base_model_name"])

    def test_validate_artifacts_checks_canonical_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            resolved = {
                "advanced_method": {"name": "frozen_original_baseline"},
                "candidate_reranking_method": {"name": "none"},
                "optimization": {"epoch": 1000},
            }
            records = [
                record(name, index, "frozen_original_baseline")
                for name in runner.EXPECTED_DATASETS
                for index in range(4)
            ]
            (run_dir / "resolved_config.json").write_text(
                json.dumps(resolved), encoding="utf-8"
            )
            (run_dir / "experiment.log").write_text("ok\n", encoding="utf-8")
            (run_dir / "reconstructions.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            checked = runner.validate_artifacts(
                run_dir,
                expected_advanced="frozen_original_baseline",
            )
            self.assertEqual(12, len(checked["records"]))
            self.assertEqual(3, len(checked["artifact_sha256"]))

    def test_suffix_summary_is_gt_blind_and_pre_post_are_separate(self):
        records = [
            record("Skytrax", index, "suffix_reoptimization_v2.2.1", 0.75, 0.5)
            for index in range(4)
        ]
        summary = runner.suffix_diagnostics(records)
        self.assertTrue(summary["formal_gt_blind"])
        self.assertEqual(4, summary["accepted_sample_count"])
        self.assertEqual(4, runner.accuracy_summary(records, use_pre=True)["sample_count"])


if __name__ == "__main__":
    unittest.main()
