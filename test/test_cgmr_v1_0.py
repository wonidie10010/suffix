import unittest
from unittest import mock

import torch

import CGMR.method_versions.CGMR_v1_0 as cgmr


class CGMRV10Tests(unittest.TestCase):
    def test_default_config_uses_three_layers(self):
        config = cgmr.CGMRV10Config()

        self.assertEqual((0, 1, 2), config.layer_offsets)
        self.assertEqual((0.5, 0.3, 0.2), config.layer_weights)

    def test_config_rejects_invalid_layer_layout(self):
        with self.assertRaises(ValueError):
            cgmr.CGMRV10Config(layer_offsets=(1, 2), layer_weights=(0.6, 0.4))
        with self.assertRaises(ValueError):
            cgmr.CGMRV10Config(layer_offsets=(0, 1), layer_weights=(1.0,))
        with self.assertRaises(ValueError):
            cgmr.CGMRV10Config(layer_offsets=(0, -1), layer_weights=(0.5, 0.5))

    def test_layer_resolution_uses_24_25_26_and_filters_overflow(self):
        layers, weights, filtered = cgmr.resolve_effective_layers(
            24, 28, (0, 1, 2), (0.5, 0.3, 0.2)
        )
        self.assertEqual([24, 25, 26], layers)
        self.assertEqual([], filtered)
        self.assertAlmostEqual(1.0, sum(weights))

        layers, weights, filtered = cgmr.resolve_effective_layers(
            27, 28, (0, 1, 2), (0.5, 0.3, 0.2)
        )
        self.assertEqual([27], layers)
        self.assertEqual([28, 29], filtered)
        self.assertEqual([1.0], weights)

    def test_multilayer_aggregation_normalizes_each_layer(self):
        raw = {
            24: [0.1, 0.2, 0.3],
            25: [0.4, 0.5, 0.6],
            26: [0.7, 0.7, 0.7],
        }
        normalized, enhanced, penalty = cgmr._normalize_and_aggregate(
            raw, [24, 25, 26], [0.5, 0.3, 0.2], 0.0
        )

        self.assertTrue(torch.all(normalized[26] == 0))
        self.assertEqual(2, int(torch.argmax(enhanced)))
        self.assertTrue(torch.all(penalty == 0))

    def test_insufficient_layers_skip_without_model_access(self):
        config = cgmr.CGMRV10Config(enabled=True)
        tokens, result = cgmr.run_cgmr_v1_0(
            model=None,
            embed_layer=None,
            optimized_embedding=None,
            current_tokens=[10, 20],
            target_hidden_states={},
            attention_mask=None,
            target_layer=27,
            model_layer_count=28,
            model_device=None,
            config=config,
            tokenizer=None,
            filter_nonascii=True,
            add_perplexity=False,
            top_k_ppl=0,
            top_k_cos=0,
            invert_method="cosine",
            eval_start_pos=0,
            embedding_top_indices=None,
            select_candidate_from_top_indices=None,
            get_perplexity=None,
        )

        self.assertEqual([10, 20], tokens)
        self.assertTrue(result["skipped"])
        self.assertEqual("insufficient_valid_layers", result["reason"])

    def test_accepted_earlier_token_is_visible_to_later_position(self):
        config = cgmr.CGMRV10Config(
            enabled=True,
            strong_margin_threshold=0.01,
            weak_margin_threshold=0.02,
            low_score_threshold=-1.0,
            min_enhanced_gain=0.05,
            min_enhanced_margin=0.05,
            max_layer_l_drop=0.0,
        )
        prefixes = []

        def candidate_pool(position, current_tokens, *args, **kwargs):
            del args, kwargs
            prefixes.append((position, list(current_tokens)))
            current = int(current_tokens[position])
            alternative = 11 if position == 0 else 21
            candidates = [current, alternative]
            return candidates, {
                current: ["current"],
                alternative: ["embedding"],
            }, alternative

        def layer_scores(model, current_tokens, position, candidate_ids,
                         target_hidden_states, layer_ids, attention_mask,
                         batch_size, model_device):
            del model, current_tokens, candidate_ids, target_hidden_states
            del attention_mask, batch_size, model_device
            if position == 0 and layer_ids == [0]:
                return {0: [0.5, 0.5]}
            if position == 0:
                return {layer_id: [0.0, 1.0] for layer_id in layer_ids}
            return {layer_id: [1.0, 0.0] for layer_id in layer_ids}

        with mock.patch.object(cgmr, "_build_candidate_pool", side_effect=candidate_pool), \
                mock.patch.object(cgmr, "_score_candidates_by_layers", side_effect=layer_scores):
            tokens, result = cgmr.run_cgmr_v1_0(
                model=None,
                embed_layer=None,
                optimized_embedding=None,
                current_tokens=[10, 20],
                target_hidden_states={0: object(), 1: object(), 2: object()},
                attention_mask=None,
                target_layer=0,
                model_layer_count=3,
                model_device=None,
                config=config,
                tokenizer=None,
                filter_nonascii=True,
                add_perplexity=False,
                top_k_ppl=0,
                top_k_cos=0,
                invert_method="cosine",
                eval_start_pos=0,
                embedding_top_indices=None,
                select_candidate_from_top_indices=None,
                get_perplexity=None,
            )

        self.assertEqual([11, 20], tokens)
        self.assertEqual([0], result["changed_positions"])
        self.assertEqual((1, [11, 20]), prefixes[1])


if __name__ == "__main__":
    unittest.main()
