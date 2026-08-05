import io
import unittest

from experiment_outputs import (
    annotate_candidate_events_for_offline_evaluation,
    build_stage_accuracy,
    extract_experiment_stage_summary,
    write_experiment_average_summary,
    write_experiment_sample_summary,
)


def make_record(
    advanced_method="frozen_original_baseline",
    candidate_method="none",
    accuracy=0.5,
    baseline_accuracy=None,
    suffix_accuracy=None,
    cgmr_accuracy=None,
    dataset_name="airport",
    dataset_sample_number=1,
    dataset_sample_count=5,
):
    record = {
        "dataset": {
            "name": dataset_name,
            "sample_number": dataset_sample_number,
            "sample_count": dataset_sample_count,
        },
        "selected_advanced_method": advanced_method,
        "selected_candidate_reranking_method": candidate_method,
        "accuracy": accuracy,
        "suffix_reoptimization_result": {},
        "candidate_reranking_result": {},
    }
    if advanced_method.startswith(("suffix_reoptimization_v", "suffix_v")):
        result_key = (
            "suffix_v1_2_3_result"
            if advanced_method == "suffix_v1.2.3"
            else "suffix_reoptimization_result"
        )
        record[result_key] = {
            "pre_acc": baseline_accuracy,
            "post_acc": suffix_accuracy,
        }
    if candidate_method != "none":
        record["candidate_reranking_result"] = {
            "pre_acc": (
                baseline_accuracy
                if advanced_method == "frozen_original_baseline"
                else accuracy
            ),
            "post_acc": cgmr_accuracy,
        }
    record["stage_accuracy"] = build_stage_accuracy(record)
    return record


class ExperimentLogSummaryTests(unittest.TestCase):
    def test_suffix_and_cgmr_sample_has_dataset_and_stage_summary(self):
        record = make_record(
            advanced_method="suffix_reoptimization_v1.2",
            candidate_method="CGMR_v1.1",
            accuracy=0.826531,
            baseline_accuracy=0.765306,
            suffix_accuracy=0.826531,
            cgmr_accuracy=0.826531,
        )
        output = io.StringIO()

        write_experiment_sample_summary(output, record, 1, 10, 98)

        self.assertEqual(
            output.getvalue(),
            "===== dataset airport sample 1/5 =====\n"
            "  dataset: airport\n"
            "  dataset_sample: 1/5\n"
            "  global_sample: 1/10\n"
            "  token_length: 98\n"
            "  selected_method: suffix_reoptimization_v1.2 + CGMR_v1.1\n"
            "  pre_suffix_accuracy: 0.765306\n"
            "  suffix_v1_2_accuracy: 0.826531\n"
            "  CGMR_v1_1_accuracy: 0.826531\n",
        )

    def test_baseline_and_disabled_paths_use_expected_sources(self):
        baseline_cgmr = make_record(
            candidate_method="CGMR_v1.0",
            accuracy=0.65,
            baseline_accuracy=0.55,
            cgmr_accuracy=0.65,
        )
        disabled = make_record(accuracy=0.25)

        self.assertEqual(
            extract_experiment_stage_summary(baseline_cgmr),
            {
                "selected_advanced_method": (
                    "frozen_original_baseline + CGMR_v1.0"
                ),
                "baseline_accuracy": 0.55,
                "suffix_enabled": False,
                "suffix_accuracy": None,
                "cgmr_enabled": True,
                "cgmr_accuracy": 0.65,
            },
        )
        output = io.StringIO()
        write_experiment_sample_summary(output, disabled, 1, 1, 20)
        self.assertIn("  standalone_baseline_accuracy: 0.250000\n", output.getvalue())
        self.assertNotIn("suffix_accuracy", output.getvalue())
        self.assertNotIn("CGMR_accuracy", output.getvalue())

    def test_average_blocks_are_grouped_by_dataset_and_include_overall(self):
        records = [
            make_record(
                advanced_method="suffix_reoptimization_v1.2.1",
                candidate_method="CGMR_v1.1",
                baseline_accuracy=0.4,
                suffix_accuracy=0.6,
                cgmr_accuracy=0.7,
                dataset_name="airport",
            ),
            make_record(
                advanced_method="suffix_reoptimization_v1.2.1",
                candidate_method="CGMR_v1.1",
                baseline_accuracy=0.6,
                suffix_accuracy=0.8,
                cgmr_accuracy=0.9,
                dataset_name="medical",
            ),
        ]
        output = io.StringIO()
        write_experiment_average_summary(output, records)
        text = output.getvalue()

        self.assertIn("===== dataset airport average accuracy =====", text)
        self.assertIn("===== dataset medical average accuracy =====", text)
        self.assertIn("===== overall average accuracy =====", text)
        self.assertIn("  pre_suffix_average_accuracy: 0.500000\n", text)
        self.assertIn("  suffix_v1_2_1_average_accuracy: 0.700000\n", text)
        self.assertIn("  CGMR_v1_1_average_accuracy: 0.800000\n", text)

    def test_single_dataset_baseline_average_is_not_duplicated(self):
        output = io.StringIO()
        write_experiment_average_summary(output, [make_record(accuracy=0.5)])

        self.assertEqual(
            output.getvalue(),
            "===== dataset airport average accuracy =====\n"
            "  sample_count: 1\n"
            "  standalone_baseline_average_accuracy: 0.500000\n",
        )

    def test_cgmr_details_stay_out_of_log_and_offline_truth_stays_in_json_data(self):
        record = make_record(
            candidate_method="CGMR_v1.1",
            accuracy=0.7,
            baseline_accuracy=0.6,
            cgmr_accuracy=0.7,
        )
        record["candidate_reranking_result"].update({
            "risk_top_k": 20,
            "events": [{
                "position": 3,
                "old_token": 10,
                "new_token": 11,
                "accepted": True,
                "candidate_scores": [{"token_id": 10}],
            }],
        })
        annotate_candidate_events_for_offline_evaluation(
            record["candidate_reranking_result"],
            [0, 0, 0, 11],
        )
        output = io.StringIO()

        write_experiment_sample_summary(output, record, 1, 1, 20)

        text = output.getvalue()
        self.assertNotIn("risk_top_k", text)
        self.assertNotIn("candidate_score", text)
        self.assertNotIn("cgmr_position", text)
        evaluation = record["candidate_reranking_result"]["events"][0][
            "offline_evaluation"
        ]
        self.assertTrue(evaluation["evaluation_only"])
        self.assertEqual("correct_repair", evaluation["outcome"])

    def test_cgmr_v1_2_log_adds_only_compact_online_summary(self):
        record = make_record(
            candidate_method="CGMR_v1.2",
            accuracy=0.7,
            baseline_accuracy=0.6,
            cgmr_accuracy=0.7,
        )
        record["candidate_reranking_result"].update({
            "name": "CGMR_v1.2",
            "enabled": True,
            "log_enabled": True,
            "effective_layers": [24, 25],
            "effective_weights": [0.625, 0.375],
            "entropy_temperature": 0.05,
            "effective_candidate_threshold": 1.5,
            "high_entropy_position_count": 2,
            "multilayer_positions": [4],
            "multilayer_accepted_positions": [4],
            "multilayer_accepted_count": 1,
            "elapsed_seconds": 0.125,
            "events": [{"candidate_scores": [{"token_id": 10}]}],
        })
        output = io.StringIO()

        write_experiment_sample_summary(output, record, 1, 1, 20)

        text = output.getvalue()
        self.assertIn("  cgmr_v1_2_input_accuracy: 0.600000\n", text)
        self.assertNotIn("standalone_baseline_accuracy", text)
        self.assertIn("  cgmr_effective_layers: [24, 25]\n", text)
        self.assertIn("  cgmr_multilayer_accepted_positions: [4]\n", text)
        self.assertIn("  cgmr_pre_accuracy: 0.600000\n", text)
        self.assertIn("  cgmr_post_accuracy: 0.700000\n", text)
        self.assertNotIn("candidate_scores", text)
        self.assertNotIn("token_id", text)


if __name__ == "__main__":
    unittest.main()
