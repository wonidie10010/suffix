import inspect
import os
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import baseline_methods.method_versions.frozen_original_baseline as frozen
import invert
from experiment_outputs import extract_experiment_stage_summary
from invert import (
    load_config,
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
    "l24_airport_medical_baseline_only.json",
)


class _Tokenizer:
    all_special_ids = [0]

    def decode(self, token_ids):
        values = [int(item) for item in token_ids]
        if len(values) == 1 and values[0] == 4:
            return "é"
        return " ".join(str(item) for item in values)


class _Model:
    def __init__(self, embed_layer):
        self._embed_layer = embed_layer

    def get_input_embeddings(self):
        return self._embed_layer


class FrozenOriginalBaselineTests(unittest.TestCase):
    def test_suffix_none_has_one_fixed_baseline(self):
        disabled = types.SimpleNamespace(enabled=False)
        self.assertEqual(
            "frozen_original_baseline",
            select_advanced_method(
                "none",
                disabled,
                disabled,
                disabled,
                disabled,
            ),
        )
        merged = load_config(ADVANCED_CONFIG)
        self.assertNotIn("baseline_implementation", merged)
        self.assertNotIn("frozen_original_baseline", merged)
        self.assertIn("suffix_v1_2_3", merged)

    def test_formal_config_selects_only_frozen_baseline(self):
        merged = load_config(FORMAL_CONFIG)

        self.assertNotIn("baseline_implementation", merged)
        self.assertNotIn("frozen_original_baseline", merged)
        self.assertEqual("none", merged["suffix_version"])
        self.assertNotIn("local_embedding_repair", merged)
        self.assertEqual("none", merged["cgmr_version"])
        self.assertEqual(
            ["airport", "medical"],
            [item["name"] for item in merged["datasets"]],
        )
        self.assertEqual([5, 5], [item["len"] for item in merged["datasets"]])
        self.assertEqual(
            "frozen_original_baseline",
            select_advanced_method(
                merged["suffix_version"],
                types.SimpleNamespace(enabled=False),
                types.SimpleNamespace(enabled=False),
                types.SimpleNamespace(enabled=False),
                types.SimpleNamespace(enabled=False),
            ),
        )

    def test_experiment_summary_labels_frozen_baseline_independently(self):
        summary = extract_experiment_stage_summary({
            "selected_advanced_method": "frozen_original_baseline",
            "selected_candidate_reranking_method": "none",
            "accuracy": 0.5,
        })

        self.assertEqual(
            "frozen_original_baseline",
            summary["selected_advanced_method"],
        )
        self.assertEqual(0.5, summary["baseline_accuracy"])

    def test_cosine_vocabulary_search_matches_legacy_helper(self):
        embed_layer = types.SimpleNamespace(
            weight=torch.tensor([
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.5, 0.5],
            ])
        )
        query = torch.tensor([0.9, 0.1])

        expected = invert.embedding_top_indices(
            query,
            embed_layer,
            4,
            "cosine",
            chunk_size=2,
        )
        actual = frozen._embedding_top_indices_cosine(
            query,
            embed_layer,
            4,
            chunk_size=2,
        )

        self.assertEqual(expected.tolist(), actual.tolist())
        self.assertEqual(
            invert.select_candidate_from_top_indices(
                expected,
                _Tokenizer(),
                True,
            ),
            frozen._select_candidate_from_top_indices(
                actual,
                _Tokenizer(),
                True,
            ),
        )

    def test_hidden_cosine_rerank_matches_legacy_mock_output(self):
        embed_layer = types.SimpleNamespace(
            weight=torch.tensor([
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [-1.0, 0.0],
            ])
        )
        model = _Model(embed_layer)
        optimized = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
        target_hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        target_ids = torch.tensor([[1, 2]])

        def forward_tokens(model, token_lists, attention_mask, layer_id):
            del model, attention_mask, layer_id
            rows = []
            for token_list in token_lists:
                rows.append([
                    embed_layer.weight[int(token_id)]
                    for token_id in token_list
                ])
            return torch.stack([torch.stack(row) for row in rows])

        def perplexity(input_ids, model, layer_id, top_k):
            del input_ids, model, layer_id, top_k
            return torch.tensor([1.0]), torch.tensor([2])

        with (
            mock.patch.object(
                invert,
                "forward_and_get_last_hidden_state",
                side_effect=forward_tokens,
            ),
            mock.patch.object(
                invert,
                "get_perplexity",
                side_effect=perplexity,
            ),
            mock.patch.object(invert, "console_update"),
        ):
            legacy = invert.invert_and_find_best(
                optimized,
                target_hidden,
                _Tokenizer(),
                model,
                target_ids,
                0,
                invert_method="cosine",
                filter_nonascii=True,
                add_perplexity=True,
                top_k_ppl=1,
                top_k_cos=3,
                eval_start_pos=0,
            )

        frozen_result = frozen._rerank_hidden_cosine(
            optimized,
            _Tokenizer(),
            model,
            embed_layer,
            target_hidden,
            target_ids,
            0,
            True,
            True,
            1,
            3,
            0,
            perplexity,
            forward_tokens,
        )

        self.assertEqual(legacy, frozen_result[:3])

    def test_continuous_cosine_update_matches_legacy_equation(self):
        initial = torch.tensor(
            [[[0.8, 0.2], [0.2, 0.8]]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        lr = 0.05

        expected = initial.clone().requires_grad_(True)
        optimizer = torch.optim.SGD([expected], lr=lr)
        cosine = F.cosine_similarity(
            expected.float(),
            target.float(),
            dim=-1,
        )
        optimizer.zero_grad()
        (-cosine * torch.ones(2)).sum().backward(inputs=[expected])
        optimizer.step()

        embed_layer = types.SimpleNamespace(weight=torch.eye(2))
        with mock.patch.object(
            frozen,
            "_forward_embedding_hidden",
            side_effect=lambda model, current, attention, layer, hooks: current,
        ):
            actual, _, summary = frozen._optimize_continuous_embedding(
                model=None,
                initial_optimizable_embedding=initial.clone(),
                prefix_embedding=None,
                target_hidden_state=target,
                attention_mask=None,
                layer_id=0,
                register_layer_hooks=None,
                tokenizer=_Tokenizer(),
                embed_layer=embed_layer,
                total_input_ids=torch.tensor([[0, 1]]),
                right_range=torch.tensor(100.0),
                lr=lr,
                epoch=1,
                alpha=0.0,
                clip=False,
                filter_nonascii=False,
                top_k=1,
                eval_start_pos=0,
            )

        self.assertTrue(torch.allclose(expected.detach(), actual))
        self.assertEqual("SGD", summary["optimizer"])
        self.assertTrue(summary["optimizer_recreated_each_epoch"])

    def test_frozen_sidecar_is_isolated_from_v123_and_keeps_cosine(self):
        source = inspect.getsource(frozen)

        self.assertIn("F.cosine_similarity", source)
        self.assertIn("torch.argmax", source)
        self.assertIn("torch.optim.SGD", source)
        self.assertNotIn("suffix_reoptimization_v1_2_3", source)
        self.assertNotIn("from suffix_optimization_methods", source)
        self.assertNotIn("experiment_outputs", source)

    def test_runner_has_no_baseline_toggle_or_selector_config(self):
        parameters = inspect.signature(
            frozen.run_frozen_original_baseline
        ).parameters

        self.assertNotIn("config", parameters)
        self.assertNotIn("enabled", parameters)


if __name__ == "__main__":
    unittest.main()
