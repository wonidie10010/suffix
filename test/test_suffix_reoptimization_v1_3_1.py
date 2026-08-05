import inspect
import io
import types
import unittest
from unittest import mock

import torch

import suffix_optimization_methods.method_versions.suffix_reoptimization_v1_3_1 as v131
from invert import normalize_suffix_version, select_advanced_method


class SuffixReoptimizationV131Tests(unittest.TestCase):
    def _embed_layer(self):
        layer = torch.nn.Embedding(8, 2)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [[0.0, 0.0]]
                    + [[float(index), -float(index)] for index in range(1, 8)]
                )
            )
        return layer

    def test_anchor_preserves_fixed_prefix_and_continuous_suffix(self):
        embed_layer = self._embed_layer()
        current_embedding = torch.tensor(
            [[[90.0, 91.0], [80.0, 81.0], [70.0, 71.0], [60.0, 61.0]]],
            requires_grad=True,
        )
        current_tokens = [7, 2, 3, 6]

        anchored = v131._build_anchored_base_embedding(
            current_embedding,
            current_tokens,
            suffix_start=3,
            eval_start_pos=1,
            embed_layer=embed_layer,
        )

        self.assertTrue(torch.equal(anchored[:, :1], current_embedding[:, :1]))
        self.assertTrue(
            torch.equal(
                anchored[:, 1:3],
                embed_layer(torch.tensor([[2, 3]])).detach(),
            )
        )
        self.assertTrue(torch.equal(anchored[:, 3:], current_embedding[:, 3:]))
        self.assertFalse(anchored.requires_grad)
        self.assertEqual(current_embedding.shape, anchored.shape)

        suffix_parameter = anchored[:, 3:].clone().requires_grad_(True)
        joined = torch.cat((anchored[:, :3], suffix_parameter), dim=1)
        joined.sum().backward()
        self.assertIsNone(current_embedding.grad)
        self.assertIsNone(embed_layer.weight.grad)
        self.assertIsNotNone(suffix_parameter.grad)

    def test_anchor_uses_latest_accepted_tokens_each_round(self):
        embed_layer = self._embed_layer()
        embedding = torch.zeros((1, 4, 2))

        first = v131._build_anchored_base_embedding(
            embedding,
            [0, 1, 2, 0],
            suffix_start=3,
            eval_start_pos=1,
            embed_layer=embed_layer,
        )
        second = v131._build_anchored_base_embedding(
            embedding,
            [0, 4, 5, 0],
            suffix_start=3,
            eval_start_pos=1,
            embed_layer=embed_layer,
        )

        self.assertFalse(torch.equal(first[:, 1:3], second[:, 1:3]))
        self.assertTrue(
            torch.equal(
                second[:, 1:3],
                embed_layer(torch.tensor([[4, 5]])).detach(),
            )
        )

    def test_token_embedding_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "token length"):
            v131._build_anchored_base_embedding(
                torch.zeros((1, 4, 2)),
                [1, 2, 3],
                suffix_start=3,
                eval_start_pos=1,
                embed_layer=self._embed_layer(),
            )

    def test_anchor_diagnostic_reuses_first_optimization_forward(self):
        embed_layer = self._embed_layer()
        current_embedding = torch.tensor(
            [[[1.0, 0.5], [2.0, -2.0], [3.0, -3.0], [1.0, 1.0]]]
        )
        current_tokens = [0, 2, 3, 1]
        anchored = v131._build_anchored_base_embedding(
            current_embedding,
            current_tokens,
            suffix_start=3,
            eval_start_pos=1,
            embed_layer=embed_layer,
        )
        forward_count = 0

        def forward(model, current, attention_mask, layer_id, hooks):
            nonlocal forward_count
            forward_count += 1
            return current

        config = v131.SuffixReoptimizationV131Config(
            epoch=2,
            lr=0.0,
            range_weight=0.0,
        )
        with mock.patch.object(
            v131,
            "_forward_embedding_hidden",
            side_effect=forward,
        ):
            _, summary = v131._optimize_suffix(
                model=None,
                input_embed=current_embedding,
                current_tokens=current_tokens,
                target_hidden_state=anchored,
                attention_mask=None,
                layer_id=0,
                register_layer_hooks=None,
                suffix_start=3,
                eval_start_pos=1,
                pre_anchor_embedding_scores=[0.0, 0.0, 0.0, 0.5],
                config=config,
                embed_layer=embed_layer,
            )

        self.assertEqual(2, forward_count)
        self.assertEqual(2, summary["anchor_count"])
        self.assertEqual(
            "current_accepted_reconstruction",
            summary["anchor_token_source"],
        )
        self.assertTrue(summary["anchor_uses_model_input_embedding_layer"])
        self.assertFalse(summary["anchor_uses_ground_truth_reconstruction"])
        self.assertEqual(0.5, summary["pre_anchor_weighted_cosine_similarity"])
        self.assertIsNotNone(summary["post_anchor_weighted_cosine_similarity"])

    def test_v131_retains_v121_cosine_decision_paths(self):
        source = inspect.getsource(v131)
        optimize_source = inspect.getsource(v131._optimize_suffix)

        self.assertIn("F.cosine_similarity", source)
        self.assertIn("torch.argmax", source)
        self.assertNotIn("relative_mse", source.lower())
        self.assertNotIn("manifold_loss", optimize_source)
        self.assertNotIn("_nearest_vocab_embeddings", optimize_source)

    def test_selector_aliases_and_explicit_enable(self):
        disabled = types.SimpleNamespace(enabled=False)
        enabled = v131.SuffixReoptimizationV131Config(enabled=True)

        for alias in (
            "1.3.1",
            "v1.3.1",
            "suffix_reoptimization_v1.3.1",
            "suffix_reoptimization_v1_3_1",
        ):
            self.assertEqual("v1.3.1", normalize_suffix_version(alias))
        self.assertEqual(
            "suffix_reoptimization_v1.3.1",
            select_advanced_method(
                "v1.3.1",
                disabled,
                disabled,
                disabled,
                disabled,
                disabled,
                suffix_reopt_v1_3_1_config=enabled,
            ),
        )

    def test_sidecar_ignores_nonempty_log_file(self):
        log_file = io.StringIO()
        _, result = v131.run_suffix_reoptimization_v1_3_1(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            v131.SuffixReoptimizationV131Config(enabled=False),
            log_file=log_file,
        )

        self.assertEqual("", log_file.getvalue())
        self.assertEqual("v1.3.1", result["version"])
        self.assertNotIn("from experiment_outputs import", inspect.getsource(v131))


if __name__ == "__main__":
    unittest.main()
