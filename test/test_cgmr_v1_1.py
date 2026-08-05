import unittest
from unittest import mock

import CGMR.method_versions.CGMR_v1_1 as cgmr


class CGMRV11Tests(unittest.TestCase):
    def test_default_config_uses_relative_risk_and_accepted_budget(self):
        config = cgmr.CGMRV11Config()

        self.assertEqual((0, 1, 2), config.layer_offsets)
        self.assertEqual((0.5, 0.3, 0.2), config.layer_weights)
        self.assertEqual(20, config.risk_top_k)
        self.assertEqual(10, config.max_accepted_repairs)
        self.assertEqual(0.0, config.score_drop_risk_weight)

    def test_relative_margin_distinguishes_high_and_low_cosine_ties(self):
        high_scores = cgmr._relative_margin(0.999, 0.998, 1e-6)
        low_scores = cgmr._relative_margin(0.650, 0.645, 1e-6)

        self.assertAlmostEqual(0.5, high_scores, places=6)
        self.assertAlmostEqual(0.005 / 0.355, low_scores, places=6)
        self.assertGreater(high_scores, low_scores)

    def test_config_rejects_invalid_risk_controls(self):
        with self.assertRaises(ValueError):
            cgmr.CGMRV11Config(relative_margin_epsilon=0.0)
        with self.assertRaises(ValueError):
            cgmr.CGMRV11Config(
                relative_margin_risk_weight=0.0,
                low_score_risk_weight=0.0,
                score_drop_risk_weight=0.0,
            )
        with self.assertRaises(ValueError):
            cgmr.CGMRV11Config(risk_top_k=0)

    def test_risk_top_k_uses_risk_rank_then_position_processing_order(self):
        records = [
            {"position": 8, "risk_score": 0.9},
            {"position": 2, "risk_score": 0.8},
            {"position": 5, "risk_score": 0.7},
        ]

        selected, _ = cgmr._rank_and_select_risk_records(records, 0.65, 2)

        self.assertEqual([8, 2], [record["position"] for record in selected])
        self.assertEqual([2, 8], sorted(record["position"] for record in selected))
        self.assertFalse(records[2]["selected_for_rerank"])

    def test_earlier_accepted_token_is_visible_to_later_selected_position(self):
        config = cgmr.CGMRV11Config(
            enabled=True,
            min_risk_score=0.0,
            risk_top_k=2,
            max_accepted_repairs=10,
            min_enhanced_gain=0.05,
            min_enhanced_margin=0.05,
            max_layer_l_drop=0.0,
        )
        prefixes = []
        layer_l_calls = {0: 0, 1: 0}

        def candidate_pool(position, current_tokens, *args, **kwargs):
            del args, kwargs
            prefixes.append((position, list(current_tokens)))
            current = int(current_tokens[position])
            alternative = 11 if position == 0 else 21
            return [current, alternative], {
                current: ["current"],
                alternative: ["embedding"],
            }, alternative

        def layer_scores(model, current_tokens, position, candidate_ids,
                         target_hidden_states, layer_ids, attention_mask,
                         batch_size, model_device):
            del model, current_tokens, candidate_ids, target_hidden_states
            del attention_mask, batch_size, model_device
            if layer_ids == [0]:
                layer_l_calls[position] += 1
                if layer_l_calls[position] == 1:
                    return {0: [0.6, 0.6] if position == 0 else [0.4, 0.4]}
                return {0: [0.5, 0.5]}
            if position == 0:
                return {layer_id: [0.0, 1.0] for layer_id in layer_ids}
            return {layer_id: [1.0, 0.0] for layer_id in layer_ids}

        with mock.patch.object(cgmr, "_build_candidate_pool", side_effect=candidate_pool), \
                mock.patch.object(cgmr, "_score_candidates_by_layers", side_effect=layer_scores):
            tokens, result = cgmr.run_cgmr_v1_1(
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
        self.assertEqual([1, 0], result["selected_risk_positions"])
        self.assertEqual([0, 1], result["processing_order"])
        self.assertEqual([0], result["changed_positions"])
        self.assertEqual((1, [11, 20]), prefixes[3])

    def test_rejected_positions_do_not_consume_accepted_budget(self):
        config = cgmr.CGMRV11Config(
            enabled=True,
            min_risk_score=0.0,
            risk_top_k=3,
            max_accepted_repairs=1,
            min_enhanced_gain=0.05,
            min_enhanced_margin=0.05,
            max_layer_l_drop=0.0,
        )
        layer_l_calls = {0: 0, 1: 0, 2: 0}

        def candidate_pool(position, current_tokens, *args, **kwargs):
            del args, kwargs
            current = int(current_tokens[position])
            alternative = current + 100
            return [current, alternative], {
                current: ["current"],
                alternative: ["embedding"],
            }, alternative

        def layer_scores(model, current_tokens, position, candidate_ids,
                         target_hidden_states, layer_ids, attention_mask,
                         batch_size, model_device):
            del model, current_tokens, candidate_ids, target_hidden_states
            del attention_mask, batch_size, model_device
            if layer_ids == [0]:
                layer_l_calls[position] += 1
                return {0: [0.5, 0.5]}
            if position == 0:
                return {layer_id: [1.0, 0.0] for layer_id in layer_ids}
            return {layer_id: [0.0, 1.0] for layer_id in layer_ids}

        with mock.patch.object(cgmr, "_build_candidate_pool", side_effect=candidate_pool), \
                mock.patch.object(cgmr, "_score_candidates_by_layers", side_effect=layer_scores):
            tokens, result = cgmr.run_cgmr_v1_1(
                model=None,
                embed_layer=None,
                optimized_embedding=None,
                current_tokens=[10, 20, 30],
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

        self.assertEqual([10, 120, 30], tokens)
        self.assertEqual([0, 1], result["processed_positions"])
        self.assertEqual(1, result["accepted_count"])
        self.assertEqual(1, result["rejected_count"])
        self.assertEqual([1], result["changed_positions"])


if __name__ == "__main__":
    unittest.main()
