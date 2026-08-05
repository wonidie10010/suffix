import inspect
import io
import os
import types
import unittest
from unittest import mock

import torch

import suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2_2 as v122
from experiment_outputs import (
    build_resolved_config,
    suffix_hidden_metric_view,
)
from invert import load_config, normalize_suffix_version, select_advanced_method
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2_1 import (
    SuffixReoptimizationV121Config,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVANCED_CONFIG = os.path.join(
    ROOT,
    "suffix_optimization_methods",
    "configs",
    "advanced_methods.json",
)


class _Tokenizer:
    def decode(self, token_ids):
        return " ".join(str(int(item)) for item in token_ids)


class SuffixReoptimizationV122Tests(unittest.TestCase):
    def test_relative_mse_is_the_single_float32_definition(self):
        current = torch.tensor([[[3.0, 1.0]]], dtype=torch.float16)
        target = torch.tensor([[[1.0, 1.0]]], dtype=torch.float16)

        actual = v122._relative_mse(current, target)
        expected = (
            (current.float() - target.float()).pow(2).mean(dim=-1)
            / (
                target.float().pow(2).mean(dim=-1)
                + v122.RELATIVE_MSE_EPSILON
            )
        )

        self.assertEqual(torch.float32, actual.dtype)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(1e-8, v122.RELATIVE_MSE_EPSILON)
        for path in (
            v122._embedding_forward_relative_mse,
            v122._token_forward_relative_mse,
            v122._rerank_positions,
            v122._optimize_suffix,
        ):
            self.assertIn("_relative_mse(", inspect.getsource(path))

    def test_threshold_requires_minimum_reason_count(self):
        config = v122.SuffixReoptimizationV122Config(
            anomaly_detection_mode="threshold",
            embedding_relative_mse_high_threshold=1.0,
            relative_mse_rise_threshold=0.30,
            token_relative_mse_high_threshold=1.0,
            min_anomaly_reasons=3,
        )
        embedding = [0.0, 1.2, 1.6]
        token = [0.0, 1.1, 0.0]

        anomalies = v122._find_anomalies(embedding, token, 0, config)

        self.assertEqual([1], [item["position"] for item in anomalies])
        self.assertEqual(
            [
                "embedding_forward_relative_mse_high",
                "embedding_forward_relative_mse_rise",
                "token_forward_relative_mse_high",
            ],
            anomalies[0]["reasons"],
        )

    def test_adaptive_uses_any_reason_and_has_no_embedding_rise_signal(self):
        config = v122.SuffixReoptimizationV122Config(
            anomaly_detection_mode="adaptive",
            min_anomaly_reasons=99,
            adaptive_z_threshold=1.0,
            adaptive_rise_z_threshold=1.0,
            adaptive_min_points=2,
        )

        anomalies = v122._find_anomalies(
            [0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 4.0],
            0,
            config,
        )

        self.assertEqual(3, anomalies[0]["position"])
        self.assertIn(
            "adaptive_embedding_forward_relative_mse_high",
            anomalies[0]["reasons"],
        )
        self.assertIn(
            "adaptive_token_forward_relative_mse_high",
            anomalies[0]["reasons"],
        )
        self.assertIn(
            "adaptive_token_forward_relative_mse_rise",
            anomalies[0]["reasons"],
        )
        self.assertNotIn(
            "adaptive_embedding_forward_relative_mse_rise",
            inspect.getsource(v122._find_adaptive_anomalies),
        )

    def test_candidate_pool_stays_embedding_cosine_and_hidden_rerank_uses_argmin(self):
        calls = []

        def embedding_top_indices(embed, embed_layer, top_k, invert_method):
            calls.append((embed.clone(), top_k, invert_method))
            return torch.tensor([1, 2])

        def select_candidates(top_indices, tokenizer, filter_nonascii):
            return None, [int(item) for item in top_indices]

        def forward_tokens(model, token_lists, attention_mask, layer_id):
            return torch.tensor(token_lists, dtype=torch.float32).unsqueeze(-1)

        accuracy, _, tokens, diagnostics = v122._rerank_positions(
            input_embed=torch.zeros((1, 2, 1)),
            current_tokens=None,
            rerank_start=0,
            tokenizer=_Tokenizer(),
            model=None,
            embed_layer=types.SimpleNamespace(weight=torch.zeros((3, 1))),
            target_hidden_state=torch.tensor([[[2.0], [1.0]]]),
            total_input_ids=torch.tensor([[2, 1]]),
            layer_id=0,
            invert_method="cosine",
            filter_nonascii=True,
            add_perplexity=False,
            top_k_ppl=0,
            top_k_cos=2,
            eval_start_pos=0,
            embedding_top_indices=embedding_top_indices,
            select_candidate_from_top_indices=select_candidates,
            get_perplexity=None,
            forward_and_get_last_hidden_state=forward_tokens,
        )

        self.assertEqual([2, 1], tokens)
        self.assertEqual(1.0, accuracy)
        self.assertEqual(2, len(calls))
        self.assertTrue(all(call[2] == "cosine" for call in calls))
        self.assertEqual(
            [0.0, 0.0],
            [item["candidate_relative_mse"] for item in diagnostics],
        )
        self.assertIn(
            "torch.argmin(candidate_relative_mse)",
            inspect.getsource(v122._rerank_positions),
        )

    def test_joint_loss_has_fixed_weighted_terms_and_no_manifold(self):
        config = v122.SuffixReoptimizationV122Config(
            epoch=1,
            lr=0.0,
            range_weight=0.0,
            cosine_loss_weight=0.1,
            relative_mse_loss_weight=0.9,
        )
        input_embed = torch.tensor(
            [[[1.0, 0.0], [0.8, 0.2], [0.2, 0.8]]]
        )
        target_hidden = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
        )
        embed_layer = types.SimpleNamespace(weight=torch.eye(2))

        with mock.patch.object(
            v122,
            "_forward_embedding_hidden",
            side_effect=lambda model, current, attention, layer_id, hooks: current,
        ):
            _, summary = v122._optimize_suffix(
                model=None,
                input_embed=input_embed,
                target_hidden_state=target_hidden,
                attention_mask=None,
                layer_id=0,
                register_layer_hooks=None,
                suffix_start=1,
                config=config,
                embed_layer=embed_layer,
            )

        self.assertAlmostEqual(
            summary["scaled_cosine_loss_term_start"],
            0.1 * summary["weighted_cosine_loss_start"],
        )
        self.assertAlmostEqual(
            summary["scaled_relative_mse_loss_term_start"],
            0.9 * summary["weighted_relative_mse_loss_start"],
        )
        self.assertAlmostEqual(
            summary["joint_hidden_loss_start"],
            summary["scaled_cosine_loss_term_start"]
            + summary["scaled_relative_mse_loss_term_start"],
        )
        self.assertFalse(summary["manifold_enabled"])
        self.assertNotIn("manifold_loss", inspect.getsource(v122._optimize_suffix))

    def test_acceptance_uses_unweighted_suffix_token_forward_mean(self):
        current = {
            "accuracy": 0.5,
            "relative_mse_anomaly_count": 1,
            "_token_forward_relative_mse": [9.0, 0.60, 0.40],
        }
        candidate = {
            "accuracy": 0.5,
            "relative_mse_anomaly_count": 1,
            "_token_forward_relative_mse": [9.0, 0.40, 0.30],
        }
        config = v122.SuffixReoptimizationV122Config(
            accept_mode="hidden_anomaly",
            min_relative_mse_improvement=0.01,
        )

        accepted, reason, relative_mse_improved = v122._accept_candidate(
            current,
            candidate,
            [1],
            1,
            config,
        )

        self.assertTrue(accepted)
        self.assertTrue(relative_mse_improved)
        self.assertEqual("suffix_relative_mse_improved", reason)
        self.assertNotIn(
            "_build_suffix_hidden_weights",
            inspect.getsource(v122._accept_candidate),
        )

    def test_gradient_tracker_maps_absolute_positions_without_mutating_gradients(self):
        tracker = v122.BaselineGradientTrendTracker(
            enabled=True,
            position_offset=4,
        )
        first = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        second = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        first_before = first.clone()
        second_before = second.clone()

        tracker.observe(first)
        first_summary = tracker.summary()
        tracker.observe(second)
        summary = tracker.summary()

        self.assertTrue(torch.equal(first, first_before))
        self.assertTrue(torch.equal(second, second_before))
        self.assertEqual([4, 5], [item["position"] for item in summary["positions"]])
        self.assertNotIn("local_position", summary["positions"][0])
        self.assertIsNone(
            first_summary["positions"][0]["current_vs_history_cosine"]
        )
        self.assertEqual(
            1,
            summary["positions"][0]["valid_step_count"],
        )
        self.assertAlmostEqual(
            1.0,
            summary["positions"][0]["current_vs_history_cosine"],
        )
        self.assertEqual(0, summary["positions"][1]["valid_step_count"])
        self.assertIsNone(
            summary["positions"][1]["current_vs_history_cosine"]
        )

    def test_relative_mse_fields_do_not_reuse_cosine_names(self):
        public = v122._public_metrics({
            "accuracy": 0.5,
            "token_forward_relative_mse_mean": 0.2,
            "token_forward_relative_mse_max": 0.4,
            "embedding_forward_relative_mse_mean": 0.3,
            "embedding_forward_relative_mse_max": 0.5,
            "relative_mse_anomaly_count": 1,
            "first_anomaly_position": 2,
            "first_anomaly_reasons": ["token_forward_relative_mse_high"],
        })
        forbidden = {
            "hidden_mean",
            "hidden_min",
            "embedding_hidden_mean",
            "embedding_hidden_min",
            "hidden_similarity",
            "token_forward_similarity",
        }

        self.assertTrue(forbidden.isdisjoint(public))
        view = suffix_hidden_metric_view(
            {
                "version": "v1.2.2",
                "after": {
                    "hidden_mean": 0.99,
                    "token_forward_relative_mse_mean": 0.2,
                    "token_forward_relative_mse_max": 0.4,
                },
            }
        )
        self.assertEqual("relative_mse", view["metric_system"])
        self.assertEqual(0.2, view["mean"])
        self.assertEqual(0.4, view["worst"])

    def test_selector_is_explicit_and_new_versions_do_not_enter_fallback(self):
        disabled = types.SimpleNamespace(enabled=False)
        v121_config = SuffixReoptimizationV121Config(enabled=True)
        v122_config = v122.SuffixReoptimizationV122Config(enabled=True)
        v131_config = types.SimpleNamespace(enabled=True)

        for alias in (
            "1.2.2",
            "v1.2.2",
            "suffix_reoptimization_v1.2.2",
            "suffix_reoptimization_v1_2_2",
        ):
            self.assertEqual("v1.2.2", normalize_suffix_version(alias))
        self.assertEqual(
            "suffix_reoptimization_v1.2.2",
            select_advanced_method(
                "v1.2.2",
                v121_config,
                disabled,
                disabled,
                disabled,
                disabled,
                suffix_reopt_v1_2_2_config=v122_config,
                suffix_reopt_v1_3_1_config=v131_config,
            ),
        )
        self.assertEqual(
            "suffix_reoptimization_v1.2.1",
            select_advanced_method(
                None,
                v121_config,
                disabled,
                disabled,
                disabled,
                disabled,
                suffix_reopt_v1_2_2_config=v122_config,
                suffix_reopt_v1_3_1_config=v131_config,
            ),
        )

    def test_resolved_config_is_selected_only_and_has_no_excel_output(self):
        merged = load_config(ADVANCED_CONFIG)
        merged.update({
            "config": ADVANCED_CONFIG,
            "output_dir": "v122",
            "seed": 0,
            "dataset_path": "airport.csv",
            "dataset_type": "local",
            "dataset_len": 5,
            "base_model_name": "model",
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
            "selected_advanced_method": v122.METHOD_NAME,
            "selected_candidate_reranking_method": "none",
            "suffix_version": "v1.2.2",
            "cgmr_version": "none",
            "device_map": "auto",
            "offload_folder": None,
            "offload_state_dict": True,
            "max_memory": None,
            "log_dir": "results/invert_timestamp_runs",
        })
        resolved = build_resolved_config(
            types.SimpleNamespace(**merged),
            "timestamp",
            "run_dir",
            "experiment.log",
            "reconstructions.jsonl",
            None,
            10,
            28,
            "qwen2",
            28,
        )

        self.assertEqual(
            ["suffix_reoptimization_v1_2_2"],
            list(resolved["advanced_methods"]),
        )
        self.assertEqual({}, resolved["candidate_reranking_methods"])
        self.assertNotIn("selection_rule", resolved["advanced_method"])
        self.assertNotIn("summary_excel", resolved["outputs"])

    def test_sidecar_ignores_nonempty_log_file(self):
        log_file = io.StringIO()
        _, result = v122.run_suffix_reoptimization_v1_2_2(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            v122.SuffixReoptimizationV122Config(enabled=False),
            log_file=log_file,
        )

        self.assertEqual("", log_file.getvalue())
        self.assertEqual("v1.2.2", result["version"])
        self.assertNotIn("from experiment_outputs import", inspect.getsource(v122))

    def test_formal_main_does_not_call_excel_writer(self):
        with open(os.path.join(ROOT, "invert.py"), encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertNotIn("write_summary_excel", source)
        self.assertIn("summary_excel_path=None", source)
        self.assertNotIn(".xlsx", source)


if __name__ == "__main__":
    unittest.main()
