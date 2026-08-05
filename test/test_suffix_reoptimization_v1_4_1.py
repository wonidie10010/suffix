import inspect
import json
import math
import os
import types
import unittest
from unittest import mock

import torch

import invert
import suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4 as v14
import suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4_1 as v141
from experiment_outputs import (
    _suffix_result,
    build_resolved_config,
    build_stage_accuracy,
    write_experiment_sample_summary,
)
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
    SuffixReoptimizationV14Config,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4_1 import (
    METHOD_NAME,
    SuffixReoptimizationV141Config,
    _build_confidence_mask,
    _optimize_masked_positions,
    run_suffix_reoptimization_v1_4_1,
    validate_suffix_reoptimization_v1_4_1_config,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate(position, margin=0.04, count=2, agreement=True):
    embedding_token = 10 + position
    selected_token = embedding_token if agreement else 30 + position
    candidate_ids = [selected_token]
    if count > 1:
        candidate_ids.append(20 + position)
    return {
        "position": position,
        "embedding_top1_token_id": embedding_token,
        "top1_token_id": selected_token,
        "top2_token_id": candidate_ids[1] if count > 1 else None,
        "margin": margin if count > 1 else None,
        "candidate_count": count,
        "candidate_token_ids": candidate_ids,
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


class SuffixReoptimizationV141ConfidenceTests(unittest.TestCase):
    def test_close_high_scores_all_pass_without_relative_fields(self):
        config = SuffixReoptimizationV141Config()
        continuous = [0.995, 0.996, 0.997, 0.998, 0.999]
        token = [value - 0.001 for value in continuous]
        diagnostics = {index: _candidate(index) for index in range(5)}

        mask = _build_confidence_mask(
            continuous, token, diagnostics, list(range(5)), {}, config
        )

        self.assertEqual("absolute", mask["mode"])
        self.assertEqual(list(range(5)), mask["high_confidence_positions"])
        self.assertEqual([], mask["low_confidence_positions"])
        self.assertNotIn("adaptive_gate_applied", mask)
        self.assertNotIn("percentile_min", mask["thresholds"])
        self.assertNotIn("min_points", mask["thresholds"])
        for entry in mask["per_position"]:
            self.assertNotIn("percentile_confidence", entry)
            self.assertNotIn("adaptive_gate_applied", entry)
            self.assertNotIn("percentile_confidence_below_min", entry["gate_failures"])

    def test_each_absolute_failure_remains_low_confidence(self):
        config = SuffixReoptimizationV141Config()
        cases = [
            ([0.79], [0.85], _candidate(0), {}, "continuous_similarity_below_min"),
            ([0.80], [0.79], _candidate(0), {}, "token_forward_similarity_below_min"),
            ([0.90], [0.90], _candidate(0, margin=0.019), {}, "margin_below_min"),
            ([0.95], [0.84], _candidate(0), {}, "discretization_gap_above_max"),
            ([0.90], [0.90], _candidate(0, count=1), {}, "fewer_than_two_candidates"),
            ([0.90], [0.90], _candidate(0, agreement=False), {}, "candidate_disagreement"),
            ([0.90], [0.90], _candidate(0), {0: ["hidden_adaptive_low"]}, "adaptive_anomaly"),
        ]
        for continuous, token, candidate, anomalies, expected in cases:
            with self.subTest(expected=expected):
                mask = _build_confidence_mask(
                    continuous, token, {0: candidate}, [0], anomalies, config
                )
                self.assertEqual([0], mask["low_confidence_positions"])
                self.assertIn(expected, mask["per_position"][0]["gate_failures"])

    def test_missing_nonfinite_and_invalid_token_ids_are_low_confidence(self):
        config = SuffixReoptimizationV141Config()
        diagnostics = {
            0: _candidate(0),
            1: _candidate(1),
            2: _candidate(2),
            3: _candidate(3),
        }
        diagnostics[3]["embedding_top1_token_id"] = None
        mask = _build_confidence_mask(
            [math.nan, math.inf, 0.90, 0.90],
            [0.90, 0.90, 0.90, 0.90],
            diagnostics,
            [0, 1, 2, 3],
            {},
            config,
        )
        failures = {entry["position"]: entry["gate_failures"] for entry in mask["per_position"]}
        self.assertIn("nonfinite_or_missing_signal", failures[0])
        self.assertIn("nonfinite_or_missing_signal", failures[1])
        self.assertEqual([], failures[2])
        self.assertIn("missing_candidate_token_id", failures[3])

        missing = _build_confidence_mask([], [], {}, [0], {}, config)
        self.assertIn(
            "nonfinite_or_missing_signal", missing["per_position"][0]["gate_failures"]
        )

    def test_absolute_threshold_boundaries_pass(self):
        config = SuffixReoptimizationV141Config()
        mask = _build_confidence_mask(
            [0.80, 0.90],
            [0.80, 0.80],
            {
                0: _candidate(0, margin=0.02),
                1: _candidate(1, margin=0.02),
            },
            [0, 1],
            {},
            config,
        )
        self.assertEqual([0, 1], mask["high_confidence_positions"])

    def test_mask_is_invariant_to_peer_scores_order_and_legacy_attributes(self):
        config = SuffixReoptimizationV141Config()
        base = _build_confidence_mask(
            [0.90], [0.90], {0: _candidate(0)}, [0], {}, config
        )
        expanded = _build_confidence_mask(
            [0.90, 0.999, 0.10],
            [0.90, 0.999, 0.10],
            {0: _candidate(0), 1: _candidate(1), 2: _candidate(2)},
            [2, 0, 1],
            {},
            config,
        )
        by_position = {
            entry["position"]: entry["high_confidence"]
            for entry in expanded["per_position"]
        }
        self.assertEqual(base["per_position"][0]["high_confidence"], by_position[0])

        config.confidence_percentile_min = 1.0
        config.confidence_min_points = 100
        legacy_changed = _build_confidence_mask(
            [0.90], [0.90], {0: _candidate(0)}, [0], {}, config
        )
        self.assertEqual(base, legacy_changed)

    def test_mode_aliases_normalize_to_absolute(self):
        for alias in ("absolute", "hybrid", "fixed"):
            config = validate_suffix_reoptimization_v1_4_1_config(
                SuffixReoptimizationV141Config(confidence_mode=alias)
            )
            self.assertEqual("absolute", config.confidence_mode)

    def test_confidence_builder_has_no_oracle_input(self):
        signature = inspect.signature(_build_confidence_mask)
        source = inspect.getsource(_build_confidence_mask)
        self.assertNotIn("total_input_ids", signature.parameters)
        self.assertNotIn("total_input_ids", source)


class SuffixReoptimizationV141FlowTests(unittest.TestCase):
    def test_sparse_adam_keeps_high_positions_bitwise_frozen(self):
        anchored = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        coarse = anchored.clone()
        target = torch.tensor([[[1.0, 0.0]] * 4])
        config = SuffixReoptimizationV141Config(
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
        self.assertEqual("v1.4.1", summary["version"])

    def test_no_low_positions_and_all_low_positions_are_supported(self):
        embedding = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
        config = SuffixReoptimizationV141Config(
            fine_epoch=1,
            fine_lr_max=0.01,
            fine_lr_min=0.01,
            fine_window=0,
            prox_weight=0.0,
            range_weight=0.0,
        )
        layer = types.SimpleNamespace(weight=torch.eye(2))
        skipped, summary = _optimize_masked_positions(
            None, embedding, embedding, embedding, None, 0, None, [], [0, 1], config, layer
        )
        self.assertTrue(torch.equal(embedding, skipped))
        self.assertTrue(summary["skipped"])
        self.assertEqual("v1.4.1", summary["version"])

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
        self.assertEqual("v1.4.1", summary["version"])
        self.assertFalse(torch.equal(embedding, refined))

    def _run_with_candidate_accuracy(self, candidate_accuracy):
        config = SuffixReoptimizationV141Config(
            enabled=True,
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
            "mode": "absolute",
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
            v141,
            "_rerank_positions_with_diagnostics",
            side_effect=[(coarse_tokens, {}), (candidate_tokens, {})],
        ), mock.patch.object(
            v141,
            "_evaluate_state",
            side_effect=[_state(0.50), _state(0.50), _state(candidate_accuracy)],
        ), mock.patch.object(
            v141, "_build_confidence_mask", return_value=mask_result
        ), mock.patch.object(
            v141, "_optimize_masked_positions", side_effect=fake_optimize
        ), mock.patch.object(
            v141,
            "_oracle_mask_diagnostics",
            return_value={"evaluation_only": True},
        ):
            return run_suffix_reoptimization_v1_4_1(
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

    def test_acceptance_keeps_anchors_and_reports_v141(self):
        final_embedding, result = self._run_with_candidate_accuracy(0.75)
        self.assertTrue(result["accepted"])
        self.assertEqual(METHOD_NAME, result["name"])
        self.assertEqual("v1.4.1", result["version"])
        self.assertEqual("v1.4.1", result["fine_stage"]["version"])
        self.assertEqual("absolute", result["confidence_mode"])
        self.assertEqual([2], result["changed_positions"])
        self.assertEqual([0, 1, 3], result["frozen_positions"])
        self.assertEqual(0, result["frozen_position_mutation_count"])
        self.assertEqual(0, result["frozen_embedding_mutation_count"])
        self.assertEqual([0, 10, 99, 30], result["final_tokens"])
        self.assertFalse(torch.equal(final_embedding[:, 2, :], torch.zeros(1, 2)))

    def test_accuracy_decrease_rolls_back_low_position(self):
        _, result = self._run_with_candidate_accuracy(0.25)
        self.assertFalse(result["accepted"])
        self.assertEqual([], result["changed_positions"])
        self.assertEqual([0, 10, 20, 30], result["final_tokens"])
        self.assertEqual("accuracy_decreased", result["events"][0]["accept_reason"])


class SuffixReoptimizationV141IntegrationTests(unittest.TestCase):
    def test_selector_aliases_fallback_and_v14_rollback(self):
        disabled = types.SimpleNamespace(enabled=False)
        v141_config = SuffixReoptimizationV141Config(enabled=True)
        v14_config = SuffixReoptimizationV14Config(enabled=True)
        v13_config = SuffixReoptimizationV13Config(enabled=True)
        v121_config = SuffixReoptimizationV121Config(enabled=True)
        v12_config = SuffixReoptimizationV12Config(enabled=True)
        for alias in (
            "1.4.1",
            "v1.4.1",
            "suffix_reoptimization_v1.4.1",
            "suffix_reoptimization_v1_4_1",
        ):
            self.assertEqual("v1.4.1", normalize_suffix_version(alias))

        selected = select_advanced_method(
            None,
            v121_config,
            v12_config,
            disabled,
            disabled,
            suffix_reopt_v1_3_config=v13_config,
            suffix_reopt_v1_4_config=v14_config,
            suffix_reopt_v1_4_1_config=v141_config,
        )
        self.assertEqual(METHOD_NAME, selected)
        selected_v14 = select_advanced_method(
            "v1.4",
            v121_config,
            v12_config,
            disabled,
            disabled,
            suffix_reopt_v1_3_config=v13_config,
            suffix_reopt_v1_4_config=v14_config,
            suffix_reopt_v1_4_1_config=v141_config,
        )
        self.assertEqual(v14.METHOD_NAME, selected_v14)

    def test_canonical_config_and_resolved_config_exclude_percentile_fields(self):
        config_path = os.path.join(
            ROOT,
            "suffix_optimization_methods",
            "configs",
            "suffix_reoptimization_v1_4_1.json",
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            direct = json.load(handle)
        self.assertNotIn("suffix_v1_4_1_confidence_percentile_min", direct)
        self.assertNotIn("suffix_v1_4_1_confidence_min_points", direct)

        advanced_path = os.path.join(
            ROOT, "suffix_optimization_methods", "configs", "advanced_methods.json"
        )
        merged = load_config(advanced_path)
        merged.update({
            "config": advanced_path,
            "suffix_version": "v1.4.1",
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
        method = resolved["advanced_methods"]["suffix_reoptimization_v1_4_1"]
        self.assertEqual(METHOD_NAME, resolved["advanced_method"]["name"])
        self.assertEqual("v1.4.1", resolved["advanced_method"]["suffix_version"])
        self.assertEqual("absolute", method["confidence_mode"])
        self.assertNotIn("confidence_percentile_min", method)
        self.assertNotIn("confidence_min_points", method)

        result = {"name": METHOD_NAME, "pre_acc": 0.5, "post_acc": 0.75}
        self.assertIs(result, _suffix_result({"suffix_reoptimization_v1_4_1_result": result}))

    def test_experiment_log_keeps_fixed_summary_format(self):
        record = {
            "dataset": {"name": "airport", "sample_number": 1, "sample_count": 1},
            "selected_advanced_method": METHOD_NAME,
            "selected_candidate_reranking_method": "none",
            "accuracy": 0.75,
            "suffix_reoptimization_result": {"pre_acc": 0.50, "post_acc": 0.75},
            "candidate_reranking_result": {},
        }
        record["stage_accuracy"] = build_stage_accuracy(record)
        import io

        output = io.StringIO()
        write_experiment_sample_summary(output, record, 1, 1, 4)
        text = output.getvalue()
        self.assertIn("  selected_method: suffix_reoptimization_v1.4.1\n", text)
        self.assertIn("  suffix_v1_4_1_accuracy: 0.750000\n", text)
        self.assertNotIn("confidence", text)
        self.assertNotIn("percentile", text)

    def test_coarse_branch_supports_v141_persistent_optimizer(self):
        source = inspect.getsource(invert.main)
        self.assertIn('"suffix_reoptimization_v1.4.1"', source)
        self.assertIn("suffix_v1_4_1_scheduled_learning_rate", source)
        self.assertIn("new_input_embed_0.clamp_(-clip_range, clip_range)", source)


if __name__ == "__main__":
    unittest.main()
