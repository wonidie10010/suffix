import inspect
import io
import json
import os
import types
import unittest
from unittest import mock

import torch

import suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2_2 as v122
import suffix_optimization_methods.method_versions.suffix_v1_2_3 as v123
from experiment_outputs import build_resolved_config, suffix_hidden_metric_view
from invert import (
    canonicalize_config_aliases,
    load_config,
    normalize_suffix_version,
    resolve_cli_suffix_version,
    select_advanced_method,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVANCED_CONFIG = os.path.join(
    ROOT,
    "suffix_optimization_methods",
    "configs",
    "advanced_methods.json",
)
FORMAL_CONFIG = os.path.join(
    ROOT,
    "experiment_configs",
    "l24_airport_medical_suffix_v1_2_3_no_cgmr.json",
)


class _Tokenizer:
    def __init__(self, special_ids=(), nonascii_ids=()):
        self.all_special_ids = list(special_ids)
        self.nonascii_ids = set(nonascii_ids)

    def decode(self, token_ids):
        values = [int(item) for item in token_ids]
        if len(values) == 1 and values[0] in self.nonascii_ids:
            return "é"
        return " ".join(str(item) for item in values)


def _disabled():
    return types.SimpleNamespace(enabled=False)


def _metrics(anomalies=None):
    anomalies = list(anomalies or [])
    first = anomalies[0] if anomalies else None
    return {
        "accuracy": 0.0,
        "text": "",
        "token_forward_relative_mse_mean": 0.3,
        "token_forward_relative_mse_max": 0.4,
        "embedding_forward_relative_mse_mean": 0.2,
        "embedding_forward_relative_mse_max": 0.3,
        "relative_mse_anomaly_count": len(anomalies),
        "first_anomaly_position": (
            first["position"] if first is not None else None
        ),
        "first_anomaly_reasons": (
            first["reasons"] if first is not None else []
        ),
        "_embedding_forward_relative_mse": [0.2, 0.3],
        "_token_forward_relative_mse": [0.3, 0.4],
        "_anomalies": anomalies,
    }


class SuffixV123MathTests(unittest.TestCase):
    def test_relative_mse_manual_zero_near_zero_and_float16(self):
        current = torch.tensor(
            [[[3.0, 1.0], [0.0, 0.0]]],
            dtype=torch.float16,
        )
        target = torch.tensor(
            [[[1.0, 1.0], [0.0, 0.0]]],
            dtype=torch.float16,
        )

        actual = v123._relative_mse(current, target)
        expected = (
            (current.float() - target.float()).pow(2).mean(dim=-1)
            / (
                target.float().pow(2).mean(dim=-1)
                + 1e-8
            )
        )
        zeros = v123._relative_mse(target, target)

        self.assertEqual(torch.float32, actual.dtype)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(torch.zeros_like(zeros), zeros))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(1e-8, v123.RELATIVE_MSE_EPSILON)

    def test_weighted_relative_mse_matches_manual_and_uniform_mean(self):
        current = torch.tensor([[[3.0, 1.0], [2.0, 0.0]]])
        target = torch.ones_like(current)
        values = v123._relative_mse(current, target).squeeze(0)
        weights = torch.tensor([1.0, 3.0])

        weighted = v123._weighted_relative_mse(
            current,
            target,
            position_weights=weights,
        )
        uniform = v123._weighted_relative_mse(current, target)

        self.assertAlmostEqual(
            float((values * weights).sum() / weights.sum()),
            float(weighted),
        )
        self.assertAlmostEqual(float(values.mean()), float(uniform))

    def test_stage1_mask_and_range_constraint_match_required_scope(self):
        prefix = torch.tensor([[[2.0, 2.0]]])
        optimizable = torch.tensor([[[3.0, 3.0], [9.0, 9.0]]])
        target = torch.ones((1, 3, 2))
        attention_mask = torch.tensor([[1, 1, 0]])

        with mock.patch.object(
            v123,
            "_forward_embedding_hidden",
            side_effect=lambda model, current, attention, layer, hooks: current,
        ):
            _, summary, _ = v123._stage1_optimize(
                model=None,
                initial_optimizable_embedding=optimizable,
                prefix_embedding=prefix,
                target_hidden_state=target,
                attention_mask=attention_mask,
                layer_id=0,
                register_layer_hooks=None,
                right_range=torch.tensor(0.0),
                lr=0.0,
                epoch=1,
                range_weight=0.0,
                clip=False,
                eval_start_pos=1,
                gradient_trend_stats_enabled=False,
            )

        self.assertEqual(1, summary["valid_position_count"])
        self.assertAlmostEqual(4.0, summary["relative_mse_loss_start"])
        self.assertAlmostEqual(28.0, summary["range_loss_start"])
        self.assertEqual("SGD", summary["optimizer"])
        self.assertTrue(summary["optimizer_recreated_each_epoch"])
        self.assertEqual("completed", summary["stopped_reason"])
        self.assertFalse(summary["nan_detected"])


class SuffixV123CandidateTests(unittest.TestCase):
    def test_embedding_mse_search_is_chunked_argmin_and_top_k_safe(self):
        weight = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ])
        embed_layer = types.SimpleNamespace(weight=weight)
        query = torch.tensor([1.8, 0.0], dtype=torch.float16)
        tokenizer = _Tokenizer()

        actual, diagnostics = v123._embedding_mse_candidates(
            query,
            embed_layer,
            top_k=10,
            tokenizer=tokenizer,
            filter_nonascii=True,
            chunk_size=2,
        )
        full_distances = (
            weight.float() - query.float()
        ).pow(2).mean(dim=-1)
        expected = torch.argsort(full_distances).tolist()

        self.assertEqual(expected, actual)
        self.assertEqual(expected, diagnostics["top_k_token_ids"])
        self.assertAlmostEqual(
            float(full_distances[expected[0]]),
            diagnostics["top1_mse"],
        )
        source = inspect.getsource(v123._embedding_mse_candidates)
        self.assertIn("torch.argsort", source)
        self.assertNotIn("cosine", source.lower())

    def test_embedding_mse_filters_entire_pool_and_tops_up(self):
        embed_layer = types.SimpleNamespace(
            weight=torch.tensor([[0.0], [0.1], [0.2], [0.3], [0.4]])
        )
        tokenizer = _Tokenizer(special_ids=[0], nonascii_ids=[1])

        candidates, diagnostics = v123._embedding_mse_candidates(
            torch.tensor([0.0]),
            embed_layer,
            top_k=2,
            tokenizer=tokenizer,
            filter_nonascii=True,
            chunk_size=1,
        )

        self.assertEqual([2, 3], candidates)
        self.assertEqual(2, diagnostics["valid_candidate_count"])

    def test_embedding_mse_empty_legal_pool_uses_nearest_fallback(self):
        embed_layer = types.SimpleNamespace(
            weight=torch.tensor([[0.0], [1.0], [2.0]])
        )
        tokenizer = _Tokenizer(special_ids=[0, 1, 2])

        candidates, diagnostics = v123._embedding_mse_candidates(
            torch.tensor([0.2]),
            embed_layer,
            top_k=3,
            tokenizer=tokenizer,
            filter_nonascii=True,
            chunk_size=2,
        )

        self.assertEqual([0], candidates)
        self.assertEqual(1, diagnostics["valid_candidate_count"])

    def test_hidden_rerank_uses_minimum_keeps_ppl_and_gt_is_oracle_only(self):
        embed_layer = types.SimpleNamespace(
            weight=torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        )
        input_embed = torch.tensor([[[0.1], [2.1]]])
        target_hidden = torch.tensor([[[0.0], [1.0]]])
        forward_batch_sizes = []

        def forward_tokens(model, token_lists, attention_mask, layer_id):
            del model, attention_mask, layer_id
            forward_batch_sizes.append(len(token_lists))
            return torch.tensor(token_lists, dtype=torch.float32).unsqueeze(-1)

        def perplexity(input_ids, model, layer_id, top_k):
            del input_ids, model, layer_id, top_k
            return torch.tensor([1.0]), torch.tensor([1])

        common = dict(
            input_embed=input_embed,
            tokenizer=_Tokenizer(),
            model=None,
            embed_layer=embed_layer,
            target_hidden_state=target_hidden,
            layer_id=0,
            filter_nonascii=True,
            add_perplexity=True,
            top_k_ppl=1,
            top_k_embedding=2,
            eval_start_pos=0,
            get_perplexity=perplexity,
            forward_and_get_last_hidden_state=forward_tokens,
        )
        first = v123._stage1_rerank_positions(
            total_input_ids=torch.tensor([[0, 1]]),
            **common,
        )
        second = v123._stage1_rerank_positions(
            total_input_ids=torch.tensor([[3, 3]]),
            **common,
        )

        self.assertEqual([0, 1], first[2])
        self.assertEqual(first[2], second[2])
        self.assertEqual(
            [item["candidate_token_ids"] for item in first[3]],
            [item["candidate_token_ids"] for item in second[3]],
        )
        self.assertEqual([2, 3, 1], first[3][1]["candidate_token_ids"])
        self.assertEqual(1, first[3][1]["selected_token_id"])
        self.assertAlmostEqual(1.0, first[3][1]["mse_margin"])
        self.assertTrue(all(size > 1 for size in forward_batch_sizes))
        self.assertNotEqual(
            [
                item["oracle_stage1_selected_token_correct"]
                for item in first[3]
            ],
            [
                item["oracle_stage1_selected_token_correct"]
                for item in second[3]
            ],
        )
        rerank_source = inspect.getsource(v123._stage1_rerank_positions)
        self.assertIn("torch.argmin", rerank_source)
        self.assertNotIn("torch.argmax", rerank_source)


class SuffixV123Stage2Tests(unittest.TestCase):
    def test_stage2_configuration_and_decision_helpers_are_v122(self):
        config = v123.SuffixV123Config()
        converted = v123._v122_config(config)

        self.assertIs(v122._find_anomalies, v123._find_anomalies)
        self.assertIs(v122._accept_candidate, v123._accept_candidate)
        self.assertIs(v122._optimize_suffix, v123._optimize_suffix)
        self.assertIs(v122._evaluate_state, v123._evaluate_state)
        self.assertEqual(0.1, converted.cosine_loss_weight)
        self.assertEqual(0.9, converted.relative_mse_loss_weight)
        self.assertEqual(config.__dict__, converted.__dict__)

    def test_stage2_mock_matches_v122_reject_rollback_and_rescan(self):
        embedding = torch.zeros((1, 2, 1))
        stage1_tokens = [0, 1]
        stage1_diagnostics = [{"position": 0}, {"position": 1}]
        target_hidden = torch.zeros_like(embedding)
        config = v122.SuffixReoptimizationV122Config(
            enabled=True,
            max_rounds=2,
            accept_mode="oracle_accuracy",
        )
        anomaly = {
            "position": 1,
            "reasons": ["mock"],
            "anomaly_detection_mode": "threshold",
        }

        def evaluate(
                model, candidate_embedding, tokens, target, attention,
                layer, hooks, total_ids, tokenizer, eval_start, scan_pos,
                candidate_config, forward):
            del (
                model, tokens, target, attention, layer, hooks, total_ids,
                tokenizer, eval_start, candidate_config, forward,
            )
            has_anomaly = (
                scan_pos == 1
                and float(candidate_embedding.sum()) == 0.0
            )
            return _metrics([anomaly] if has_anomaly else [])

        def rerank(
                candidate_embedding, current_tokens, rerank_start, *args,
                **kwargs):
            del rerank_start, args, kwargs
            if current_tokens is None:
                return (
                    0.5,
                    "0 1",
                    list(stage1_tokens),
                    list(stage1_diagnostics),
                )
            return 0.0, "1 0", [1, 0], [{"candidate": True}]

        with (
            mock.patch.object(v122, "_evaluate_state", side_effect=evaluate),
            mock.patch.object(v122, "_rerank_positions", side_effect=rerank),
            mock.patch.object(
                v122,
                "_optimize_suffix",
                return_value=(torch.ones_like(embedding), {"mock": True}),
            ),
        ):
            _, expected = v122.run_suffix_reoptimization_v1_2_2(
                model=None,
                embed_layer=None,
                optimized_embedding=embedding,
                target_hidden_state=target_hidden,
                attention_mask=None,
                layer_id=0,
                register_layer_hooks=None,
                tokenizer=_Tokenizer(),
                total_input_ids=torch.tensor([[0, 1]]),
                config=config,
                add_perplexity=False,
                embedding_top_indices=lambda *args: None,
                select_candidate_from_top_indices=lambda *args: None,
                get_perplexity=lambda *args: None,
                forward_and_get_last_hidden_state=lambda *args: None,
                baseline_gradient_trend_stats={"same": True},
            )
            _, actual = v123._run_reoptimization_v1_2_2(
                model=None,
                embed_layer=None,
                stage1_embedding=embedding,
                stage1_tokens=stage1_tokens,
                stage1_text="0 1",
                stage1_accuracy=0.5,
                stage1_candidate_diagnostics=stage1_diagnostics,
                target_hidden_state=target_hidden,
                attention_mask=None,
                layer_id=0,
                register_layer_hooks=None,
                tokenizer=_Tokenizer(),
                total_input_ids=torch.tensor([[0, 1]]),
                config=config,
                filter_nonascii=True,
                add_perplexity=False,
                top_k_ppl=1,
                top_k_cos=1,
                invert_method="cosine",
                eval_start_pos=0,
                embedding_top_indices=lambda *args: None,
                select_candidate_from_top_indices=lambda *args: None,
                get_perplexity=lambda *args: None,
                forward_and_get_last_hidden_state=lambda *args: None,
                stage1_gradient_trend_stats={"same": True},
            )

        comparable_fields = (
            "accept_mode",
            "pre_acc",
            "post_acc",
            "accuracy_gain",
            "triggered",
            "accepted",
            "accepted_round_count",
            "rejected_round_count",
            "reason",
            "before",
            "after",
            "events",
            "changed_positions",
            "final_tokens",
            "final_text",
            "final_accuracy",
        )
        self.assertEqual(
            {key: expected[key] for key in comparable_fields},
            {key: actual[key] for key in comparable_fields},
        )
        self.assertEqual(
            expected["baseline_gradient_trend_stats"],
            actual["stage1_gradient_trend_stats"],
        )
        self.assertFalse(actual["accepted"])
        self.assertEqual(1, actual["rejected_round_count"])
        self.assertEqual(stage1_tokens, actual["final_tokens"])


class SuffixV123WiringTests(unittest.TestCase):
    def test_legacy_config_aliases_canonicalize_and_conflicts_fail(self):
        canonical = canonicalize_config_aliases({
            "suffix_reoptimization_version": "v1.2.3",
            "suffix_reoptimization_v1_2_3": True,
            "suffix_reoptimization_v1_2_3_log": False,
        })

        self.assertEqual("v1.2.3", canonical["suffix_version"])
        self.assertTrue(canonical["suffix_v1_2_3"])
        self.assertFalse(canonical["suffix_v1_2_3_log"])
        self.assertNotIn("suffix_reoptimization_version", canonical)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            canonicalize_config_aliases({
                "suffix_version": "v1.2.3",
                "suffix_reoptimization_version": "v1.2.2",
            })

    def test_legacy_cli_alias_overrides_config_but_cli_conflicts_fail(self):
        self.assertEqual(
            "v1.2.3",
            resolve_cli_suffix_version(
                "v1.2",
                "v1.2.3",
                canonical_option_present=False,
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_cli_suffix_version(
                "v1.2",
                "v1.2.3",
                canonical_option_present=True,
            )

    def test_selector_aliases_are_strict_and_v123_has_no_fallback(self):
        config = v123.SuffixV123Config(enabled=True)
        for alias in (
            "1.2.3",
            "v1.2.3",
            "suffix_v1.2.3",
            "suffix_v1_2_3",
            "suffix_reoptimization_v1.2.3",
            "suffix_reoptimization_v1_2_3",
        ):
            self.assertEqual("v1.2.3", normalize_suffix_version(alias))

        self.assertEqual(
            v123.METHOD_NAME,
            select_advanced_method(
                "v1.2.3",
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                suffix_v1_2_3_config=config,
            ),
        )
        with self.assertRaisesRegex(ValueError, "v1.2.3.*disabled"):
            select_advanced_method(
                "v1.2.3",
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                suffix_v1_2_3_config=(
                    v123.SuffixV123Config(enabled=False)
                ),
            )
        self.assertEqual(
            "frozen_original_baseline",
            select_advanced_method(
                None,
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                suffix_v1_2_3_config=config,
            ),
        )

    def test_formal_config_selects_only_v123_without_loading_data(self):
        merged = load_config(FORMAL_CONFIG)
        suffix_method_keys = {
            "suffix_reoptimization_v1_0",
            "suffix_reoptimization_v1_1",
            "suffix_reoptimization_v1_2",
            "suffix_reoptimization_v1_2_1",
            "suffix_reoptimization_v1_2_2",
            "suffix_v1_2_3",
            "suffix_reoptimization_v1_3",
            "suffix_reoptimization_v1_3_1",
            "suffix_reoptimization_v1_4",
            "suffix_reoptimization_v1_4_1",
        }
        suffix_flags = [
            key
            for key, value in merged.items()
            if key in suffix_method_keys and value is True
        ]

        self.assertEqual(["suffix_v1_2_3"], suffix_flags)
        self.assertEqual("v1.2.3", merged["suffix_version"])
        self.assertNotIn("baseline_implementation", merged)
        self.assertNotIn("frozen_original_baseline", merged)
        self.assertNotIn("local_embedding_repair", merged)
        self.assertEqual("none", merged["cgmr_version"])
        self.assertEqual([5, 5], [item["len"] for item in merged["datasets"]])
        self.assertEqual(
            v123.METHOD_NAME,
            select_advanced_method(
                merged["suffix_version"],
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                suffix_v1_2_3_config=(
                    v123.SuffixV123Config(
                        enabled=merged["suffix_v1_2_3"]
                    )
                ),
            ),
        )

    def test_runner_emits_stage1_and_stage2_contract_without_logging(self):
        embedding = torch.zeros((1, 2, 1))
        candidate_diagnostics = [{
            "embedding_mse_top1_distance": 0.25,
            "selected_hidden_relative_mse": 0.5,
        }]
        stage2 = {
            "pre_acc": 0.5,
            "post_acc": 1.0,
            "accuracy_gain": 0.5,
            "final_accuracy": 1.0,
            "final_tokens": [0, 1],
            "final_text": "0 1",
            "triggered": False,
            "accepted": False,
            "accepted_round_count": 0,
            "rejected_round_count": 0,
            "reason": "no anomaly found",
            "events": [],
            "changed_positions": [],
        }
        log_file = io.StringIO()

        with (
            mock.patch.object(
                v123,
                "_stage1_optimize",
                return_value=(
                    embedding,
                    {"metric": "relative_mse"},
                    {"enabled": True},
                ),
            ),
            mock.patch.object(
                v122,
                "_embedding_forward_relative_mse",
                return_value=[0.1, 0.2],
            ),
            mock.patch.object(
                v123,
                "_stage1_rerank_positions",
                return_value=(
                    0.5,
                    "0 1",
                    [0, 1],
                    candidate_diagnostics,
                    [1],
                ),
            ),
            mock.patch.object(
                v122,
                "_token_forward_relative_mse",
                return_value=[0.3, 0.4],
            ),
            mock.patch.object(
                v123,
                "_run_reoptimization_v1_2_2",
                return_value=(embedding, stage2),
            ),
        ):
            _, stage1_embedding, result = v123.run_suffix_v1_2_3(
                model=None,
                embed_layer=None,
                initial_optimizable_embedding=embedding,
                prefix_embedding=None,
                target_hidden_state=embedding,
                attention_mask=torch.tensor([[1, 1]]),
                layer_id=0,
                register_layer_hooks=None,
                tokenizer=_Tokenizer(),
                total_input_ids=torch.tensor([[0, 1]]),
                right_range=torch.tensor(1.0),
                config=v123.SuffixV123Config(enabled=True),
                stage1_lr=0.1,
                stage1_epoch=1000,
                stage1_range_weight=0.001,
                stage1_clip=True,
                stage1_init_method="uniform",
                stage1_init_param=0.1,
                embedding_top_indices=lambda *args: None,
                select_candidate_from_top_indices=lambda *args: None,
                get_perplexity=lambda *args: None,
                forward_and_get_last_hidden_state=lambda *args: None,
                log_file=log_file,
            )

        required = {
            "version", "method", "stage1", "reoptimization",
            "pre_acc", "final_accuracy", "post_acc", "suffix_gain",
        }
        self.assertTrue(required.issubset(result))
        self.assertIs(embedding, stage1_embedding)
        self.assertEqual("relative_mse", result["stage1"]["metric"])
        self.assertEqual("embedding_mse", result["stage1"]["vocab_metric"])
        self.assertEqual(
            "hidden_relative_mse",
            result["stage1"]["candidate_rerank_metric"],
        )
        self.assertEqual(
            v123.REOPTIMIZATION_SOURCE,
            result["reoptimization"]["source"],
        )
        self.assertEqual("", log_file.getvalue())
        json.dumps(result)
        self.assertFalse(
            any(key.startswith("_") for key in result)
        )
        self.assertFalse(
            any(
                "cosine" in key
                for key in result["stage1"]["optimization"]
            )
        )

    def test_resolved_config_has_all_effective_v123_parameters(self):
        merged = load_config(ADVANCED_CONFIG)
        merged.update({
            "config": ADVANCED_CONFIG,
            "output_dir": "v123",
            "seed": 0,
            "dataset_path": "not_loaded.csv",
            "dataset_type": "local",
            "dataset_len": 5,
            "base_model_name": "not_loaded_model",
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
            "selected_advanced_method": v123.METHOD_NAME,
            "suffix_version": "v1.2.3",
            "selected_candidate_reranking_method": "none",
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
        selected = resolved["advanced_methods"]["suffix_v1_2_3"]

        self.assertEqual(
            ["suffix_v1_2_3"],
            list(resolved["advanced_methods"]),
        )
        self.assertEqual(0.1, selected["stage1"]["lr"])
        self.assertEqual(1000, selected["stage1"]["epoch"])
        self.assertEqual(0.001, selected["stage1"]["range_weight"])
        self.assertEqual(10, selected["stage1"]["top_k"])
        self.assertEqual(0.1, selected["reoptimization"]["cosine_loss_weight"])
        self.assertEqual(
            0.9,
            selected["reoptimization"]["relative_mse_loss_weight"],
        )
        self.assertEqual(
            v123.REOPTIMIZATION_SOURCE,
            selected["reoptimization"]["source"],
        )
        self.assertNotIn("summary_excel", resolved["outputs"])
        self.assertEqual(
            "relative_mse",
            suffix_hidden_metric_view({
                "version": "v1.2.3",
                "reoptimization": {
                    "after": {
                        "token_forward_relative_mse_mean": 0.2,
                        "token_forward_relative_mse_max": 0.4,
                    },
                },
            })["metric_system"],
        )

    def test_sidecar_does_not_write_experiment_log_or_excel(self):
        source = inspect.getsource(v123)

        self.assertNotIn("experiment.log", source)
        self.assertNotIn("experiment_outputs", source)
        self.assertNotIn(".xlsx", source)
        self.assertNotIn("openpyxl", source)


if __name__ == "__main__":
    unittest.main()
