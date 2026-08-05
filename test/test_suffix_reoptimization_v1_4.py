import inspect
import math
import os
import types
import unittest
from unittest import mock

import torch

import invert
import suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4 as v14
from experiment_outputs import _suffix_result, build_resolved_config
from invert import load_config, normalize_suffix_version, select_advanced_method
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2 import (
    SuffixReoptimizationV12Config,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2_1 import (
    SuffixReoptimizationV121Config,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_3 import (
    SuffixReoptimizationV13Config,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4 import (
    METHOD_NAME,
    SuffixReoptimizationV14Config,
    _accept_candidate,
    _build_confidence_mask,
    _build_fine_observation_weights,
    _optimize_masked_positions,
    run_suffix_reoptimization_v1_4,
    scheduled_learning_rate,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate(position, margin=0.04, count=2, agreement=True):
    return {
        "position": position,
        "embedding_top1_token_id": 10 + position,
        "top1_token_id": 10 + position,
        "top2_token_id": 20 + position if count > 1 else None,
        "margin": margin if count > 1 else None,
        "candidate_count": count,
        "candidate_token_ids": list(range(count)),
        "candidate_agreement": agreement,
    }


def _state(accuracy, hidden_mean=0.90, anomaly_count=0, length=4):
    return {
        "accuracy": accuracy,
        "text": "decoded",
        "hidden_mean": hidden_mean,
        "hidden_min": hidden_mean,
        "embedding_hidden_mean": hidden_mean,
        "embedding_hidden_min": hidden_mean,
        "anomaly_count": anomaly_count,
        "first_anomaly_position": None,
        "first_anomaly_reasons": [],
        "_embedding_scores": [hidden_mean] * length,
        "_token_scores": [hidden_mean] * length,
        "_anomalies": [],
    }


class SuffixReoptimizationV14Tests(unittest.TestCase):
    def test_cosine_schedule_endpoints_and_monotonicity(self):
        values = [
            scheduled_learning_rate(step, 50, 0.01, 0.001, "cosine")
            for step in range(50)
        ]
        self.assertAlmostEqual(0.01, values[0])
        self.assertAlmostEqual(0.001, values[-1])
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))
        self.assertAlmostEqual(0.10, scheduled_learning_rate(0, 1, 0.10, 0.03))

    def test_mixed_fixed_confidence_gate(self):
        config = SuffixReoptimizationV14Config(confidence_mode="fixed")
        diagnostics = {
            0: _candidate(0),
            1: _candidate(1),
            2: _candidate(2),
            3: _candidate(3, agreement=False),
        }
        mask = _build_confidence_mask(
            [0.90, 0.79, 0.90, 0.90],
            [0.90, 0.90, 0.90, 0.90],
            diagnostics,
            [0, 1, 2, 3],
            {2: ["token_forward_adaptive_low"]},
            config,
        )

        self.assertEqual([0], mask["high_confidence_positions"])
        self.assertEqual([1, 2, 3], mask["low_confidence_positions"])
        failures = {item["position"]: item["gate_failures"] for item in mask["per_position"]}
        self.assertIn("continuous_similarity_below_min", failures[1])
        self.assertIn("adaptive_anomaly", failures[2])
        self.assertIn("candidate_disagreement", failures[3])

    def test_hybrid_percentile_gate_and_short_sequence_fallback(self):
        config = SuffixReoptimizationV14Config(
            confidence_mode="hybrid",
            confidence_continuous_min=0.70,
            confidence_token_min=0.70,
            confidence_margin_min=0.0,
            confidence_gap_max=0.20,
            confidence_percentile_min=0.60,
        )
        diagnostics = {
            index: _candidate(index, margin=0.01 * (index + 1))
            for index in range(4)
        }
        mask = _build_confidence_mask(
            [0.90, 0.91, 0.92, 0.93],
            [0.80, 0.85, 0.90, 0.93],
            diagnostics,
            [0, 1, 2, 3],
            {},
            config,
        )
        self.assertTrue(mask["adaptive_gate_applied"])
        self.assertEqual([2, 3], mask["high_confidence_positions"])

        short_mask = _build_confidence_mask(
            [0.90, 0.91, 0.92],
            [0.90, 0.91, 0.92],
            {index: _candidate(index) for index in range(3)},
            [0, 1, 2],
            {},
            config,
        )
        self.assertFalse(short_mask["adaptive_gate_applied"])
        self.assertEqual([0, 1, 2], short_mask["high_confidence_positions"])

    def test_single_candidate_and_nonfinite_signal_are_low_confidence(self):
        config = SuffixReoptimizationV14Config(confidence_mode="fixed")
        diagnostics = {0: _candidate(0, count=1), 1: _candidate(1)}
        mask = _build_confidence_mask(
            [0.90, math.nan],
            [0.90, 0.90],
            diagnostics,
            [0, 1],
            {},
            config,
        )
        self.assertEqual([], mask["high_confidence_positions"])
        failures = {item["position"]: item["gate_failures"] for item in mask["per_position"]}
        self.assertIn("fewer_than_two_candidates", failures[0])
        self.assertIn("nonfinite_or_missing_signal", failures[1])

    def test_confidence_builder_has_no_oracle_input(self):
        signature = inspect.signature(_build_confidence_mask)
        source = inspect.getsource(_build_confidence_mask)
        self.assertNotIn("total_input_ids", signature.parameters)
        self.assertNotIn("total_input_ids", source)

    def test_window_overlap_uses_maximum_weight(self):
        weights = _build_fine_observation_weights(
            6, [1, 2], list(range(6)), window=2, decay=0.5
        )
        self.assertTrue(torch.equal(
            weights,
            torch.tensor([[0.0, 1.0, 1.0, 0.5, 0.25, 0.0]]),
        ))

    def test_sparse_adam_updates_only_noncontiguous_low_positions(self):
        anchored = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        coarse = anchored.clone()
        target = torch.tensor([[[1.0, 0.0]] * 4])
        config = SuffixReoptimizationV14Config(
            fine_epoch=1,
            fine_lr_max=0.01,
            fine_lr_min=0.01,
            fine_window=0,
            prox_weight=0.0,
            range_weight=0.0,
        )
        embed_layer = types.SimpleNamespace(weight=torch.eye(2))
        with mock.patch.object(
            v14,
            "_forward_embedding_hidden",
            side_effect=lambda model, current, attention, layer_id, hooks: current,
        ):
            refined, summary = _optimize_masked_positions(
                None,
                anchored,
                coarse,
                target,
                None,
                0,
                None,
                [1, 3],
                [0, 1, 2, 3],
                config,
                embed_layer,
            )

        self.assertTrue(torch.equal(anchored[:, 0, :], refined[:, 0, :]))
        self.assertTrue(torch.equal(anchored[:, 2, :], refined[:, 2, :]))
        self.assertFalse(torch.equal(anchored[:, 1, :], refined[:, 1, :]))
        self.assertFalse(torch.equal(anchored[:, 3, :], refined[:, 3, :]))
        self.assertEqual([1, 3], summary["updated_positions"])
        self.assertEqual(1, summary["executed_steps"])

    def test_no_low_positions_skips_optimizer_and_all_low_is_supported(self):
        embedding = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        config = SuffixReoptimizationV14Config(
            fine_epoch=1,
            fine_lr_max=0.01,
            fine_lr_min=0.01,
            fine_window=0,
            prox_weight=0.0,
            range_weight=0.0,
        )
        layer = types.SimpleNamespace(weight=torch.eye(2))
        skipped, skipped_summary = _optimize_masked_positions(
            None, embedding, embedding, embedding, None, 0, None, [], [0, 1], config, layer
        )
        self.assertTrue(torch.equal(embedding, skipped))
        self.assertTrue(skipped_summary["skipped"])

        target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        with mock.patch.object(
            v14,
            "_forward_embedding_hidden",
            side_effect=lambda model, current, attention, layer_id, hooks: current,
        ):
            refined, summary = _optimize_masked_positions(
                None,
                embedding,
                embedding,
                target,
                None,
                0,
                None,
                [0, 1],
                [0, 1],
                config,
                layer,
            )
        self.assertEqual([0, 1], summary["updated_positions"])
        self.assertFalse(torch.equal(embedding, refined))

    def test_acceptance_improvement_tie_break_and_rollback(self):
        config = SuffixReoptimizationV14Config(accept_mode="oracle_accuracy")
        baseline = _state(0.50, hidden_mean=0.80, anomaly_count=1)
        accepted, reason = _accept_candidate(
            baseline, _state(0.75, hidden_mean=0.80, anomaly_count=1), [2], config
        )
        self.assertTrue(accepted)
        self.assertEqual("accuracy_improved", reason)

        accepted, reason = _accept_candidate(
            baseline, _state(0.25, hidden_mean=0.90, anomaly_count=0), [2], config
        )
        self.assertFalse(accepted)
        self.assertEqual("accuracy_decreased", reason)

        accepted, reason = _accept_candidate(
            baseline, _state(0.50, hidden_mean=0.806, anomaly_count=1), [2], config
        )
        self.assertTrue(accepted)
        self.assertEqual("hidden_improved", reason)

        always = SuffixReoptimizationV14Config(accept_mode="always")
        nonfinite = _state(0.50)
        nonfinite["hidden_mean"] = math.nan
        accepted, reason = _accept_candidate(baseline, nonfinite, [2], always)
        self.assertFalse(accepted)
        self.assertEqual("nonfinite_candidate_metrics", reason)
        self.assertEqual(
            (False, "reconstruction_unchanged"),
            _accept_candidate(baseline, _state(0.75), [], config),
        )

    def _run_with_candidate_accuracy(self, candidate_accuracy):
        config = SuffixReoptimizationV14Config(
            enabled=True,
            confidence_mode="fixed",
            fine_epoch=1,
            range_weight=0.0,
        )
        layer = torch.nn.Embedding(120, 2)
        optimized = torch.zeros(1, 4, 2)
        target_hidden = torch.zeros(1, 4, 2)
        target_ids = torch.tensor([[0, 10, 99, 30]])
        coarse_tokens = [0, 10, 20, 30]
        candidate_tokens = [0, 10, 99, 30]
        mask_result = {
            "mode": "fixed",
            "valid_positions": [1, 2, 3],
            "high_confidence_positions": [1, 3],
            "low_confidence_positions": [2],
            "high_confidence_count": 2,
            "low_confidence_count": 1,
            "per_position": [],
        }

        def fake_optimize(model, anchored, coarse, *args, **kwargs):
            candidate = anchored.clone()
            candidate[:, 2, :] = coarse[:, 2, :] + 1.0
            return candidate, {
                "skipped": False,
                "stopped_reason": "completed",
                "updated_positions": [2],
            }

        with mock.patch.object(
            v14,
            "_rerank_positions_with_diagnostics",
            side_effect=[(coarse_tokens, {}), (candidate_tokens, {})],
        ), mock.patch.object(
            v14,
            "_evaluate_state",
            side_effect=[_state(0.50), _state(0.50), _state(candidate_accuracy)],
        ), mock.patch.object(
            v14, "_build_confidence_mask", return_value=mask_result
        ), mock.patch.object(
            v14, "_optimize_masked_positions", side_effect=fake_optimize
        ), mock.patch.object(
            v14,
            "_oracle_mask_diagnostics",
            return_value={"evaluation_only": True},
        ):
            return run_suffix_reoptimization_v1_4(
                None,
                layer,
                optimized,
                target_hidden,
                torch.ones(1, 4),
                0,
                None,
                types.SimpleNamespace(decode=lambda ids: "decoded"),
                target_ids,
                config,
                eval_start_pos=1,
                embedding_top_indices=lambda *args: None,
                select_candidate_from_top_indices=lambda *args: None,
                get_perplexity=lambda *args, **kwargs: None,
                forward_and_get_last_hidden_state=lambda *args, **kwargs: None,
                coarse_stage_summary={"optimizer": "SGD", "lr_start": 0.10},
            )

    def test_run_accepts_only_low_position_change_and_keeps_anchors_bitwise(self):
        final_embedding, result = self._run_with_candidate_accuracy(0.75)
        self.assertTrue(result["accepted"])
        self.assertEqual([2], result["changed_positions"])
        self.assertEqual([0, 1, 3], result["frozen_positions"])
        self.assertEqual(0, result["frozen_position_mutation_count"])
        self.assertEqual(0, result["frozen_embedding_mutation_count"])
        self.assertEqual([0, 10, 99, 30], result["final_tokens"])
        self.assertEqual(1, len(result["events"]))
        self.assertNotIn("max_rounds", result["events"][0])
        self.assertEqual(result["pre_acc"], result["oracle_pre_acc"])
        self.assertEqual(result["post_acc"], result["oracle_post_acc"])
        self.assertFalse(torch.equal(final_embedding[:, 2, :], torch.zeros(1, 2)))

    def test_run_rolls_back_low_refinement_when_accuracy_decreases(self):
        _, result = self._run_with_candidate_accuracy(0.25)
        self.assertFalse(result["accepted"])
        self.assertEqual([], result["changed_positions"])
        self.assertEqual([0, 10, 20, 30], result["final_tokens"])
        self.assertEqual("accuracy_decreased", result["events"][0]["accept_reason"])

    def test_selector_fallback_resolved_config_and_unified_result(self):
        disabled = types.SimpleNamespace(enabled=False)
        v14_config = SuffixReoptimizationV14Config(enabled=True)
        v13_config = SuffixReoptimizationV13Config(enabled=True)
        v121_config = SuffixReoptimizationV121Config(enabled=True)
        v12_config = SuffixReoptimizationV12Config(enabled=True)
        for alias in (
            "1.4",
            "v1.4",
            "suffix_reoptimization_v1.4",
            "suffix_reoptimization_v1_4",
        ):
            self.assertEqual("v1.4", normalize_suffix_version(alias))
        self.assertEqual(
            METHOD_NAME,
            select_advanced_method(
                None,
                v121_config,
                v12_config,
                disabled,
                disabled,
                suffix_reopt_v1_3_config=v13_config,
                suffix_reopt_v1_4_config=v14_config,
            ),
        )
        self.assertEqual(
            "suffix_reoptimization_v1.2.1",
            select_advanced_method(
                "v1.2.1",
                v121_config,
                v12_config,
                disabled,
                disabled,
                suffix_reopt_v1_3_config=v13_config,
                suffix_reopt_v1_4_config=v14_config,
            ),
        )

        config_path = os.path.join(
            ROOT, "suffix_optimization_methods", "configs", "advanced_methods.json"
        )
        merged = load_config(config_path)
        merged.update({
            "config": config_path,
            "suffix_version": "v1.4",
            "output_dir": "vscode_deml_inversion",
            "seed": 0,
            "dataset_path": "data/medical.json",
            "dataset_type": "local",
            "dataset_len": 5,
            "base_model_name": "Qwen/Qwen2.5-1.5B",
            "lora_model_name": None,
            "num_invert_layers": 24,
            "quantization": "none",
            "lr": 0.1,
            "epoch": 1000,
            "alpha": 0.001,
            "clip": True,
            "init_method": "uniform",
            "init_param": 0.1,
            "optim_method": "cosine",
            "invert_method": "cosine",
            "filter_nonascii": True,
            "top_k_cos": 10,
            "perplexity": True,
            "top_k_ppl": 10,
            "selected_advanced_method": METHOD_NAME,
            "selected_candidate_reranking_method": "none",
            "device_map": "manual",
            "offload_folder": None,
            "offload_state_dict": False,
            "max_memory": None,
            "log_dir": "results/invert_timestamp_runs",
        })
        resolved = build_resolved_config(
            types.SimpleNamespace(**merged),
            "timestamp",
            "run_dir",
            "experiment.log",
            "reconstructions.jsonl",
            "unused.xlsx",
            5,
            28,
            "qwen2",
            28,
        )
        method = resolved["advanced_methods"]["suffix_reoptimization_v1_4"]
        self.assertEqual(METHOD_NAME, resolved["advanced_method"]["name"])
        self.assertEqual("v1.4", resolved["advanced_method"]["suffix_version"])
        self.assertEqual(0.10, method["coarse_lr_max"])
        self.assertEqual(50, method["fine_epoch"])
        self.assertNotIn("max_rounds", method)
        self.assertIn("one sparse Adam", method["optimizer_scope"])

        result = {"name": METHOD_NAME, "pre_acc": 0.5, "post_acc": 0.75}
        self.assertIs(result, _suffix_result({"suffix_reoptimization_v1_4_result": result}))

    def test_paired_experiment_configs_only_change_suffix_version(self):
        v121_path = os.path.join(
            ROOT,
            "experiment_configs",
            "l24_airport_medical_suffix_v1_2_1_no_cgmr.json",
        )
        v14_path = os.path.join(
            ROOT,
            "experiment_configs",
            "l24_airport_medical_suffix_v1_4_no_cgmr.json",
        )
        v121_config = load_config(v121_path)
        v14_config = load_config(v14_path)
        self.assertEqual("v1.2.1", v121_config["suffix_version"])
        self.assertEqual("v1.4", v14_config["suffix_version"])
        self.assertEqual("none", v121_config["cgmr_version"])
        self.assertEqual("none", v14_config["cgmr_version"])
        ignored = {"suffix_version", "output_dir"}
        self.assertEqual(
            {key: value for key, value in v121_config.items() if key not in ignored},
            {key: value for key, value in v14_config.items() if key not in ignored},
        )

    def test_coarse_branch_uses_persistent_optimizer_and_in_place_clip(self):
        source = inspect.getsource(invert.main)
        self.assertIn("v1_4_coarse_optimizer = torch.optim.SGD", source)
        self.assertIn("new_input_embed_0.clamp_(-clip_range, clip_range)", source)
        self.assertIn("suffix_v1_4_scheduled_learning_rate", source)
        self.assertIn('optimization_result["oracle_accuracy"]', source)


if __name__ == "__main__":
    unittest.main()
