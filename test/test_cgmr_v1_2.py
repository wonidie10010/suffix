import unittest
from unittest import mock

import torch

import CGMR.method_versions.CGMR_v1_2 as cgmr


class FakeTokenizer:
    all_special_ids = [0]

    def decode(self, token_ids):
        token_id = int(token_ids[0])
        if token_id == 2:
            return "é"
        return chr(64 + token_id) if 0 < token_id < 27 else str(token_id)


class CGMRV12PureLogicTests(unittest.TestCase):
    def test_default_config_and_validation(self):
        config = cgmr.CGMRV12Config()

        self.assertEqual((0, 1, 2), config.layer_offsets)
        self.assertEqual((0.5, 0.3, 0.2), config.layer_weights)
        self.assertEqual(0.05, config.entropy_temperature)
        self.assertEqual(1.5, config.effective_candidate_threshold)
        self.assertEqual(6, config.max_multilayer_candidates)

        invalid_configs = [
            {"layer_offsets": (1, 2), "layer_weights": (0.5, 0.5)},
            {"layer_offsets": (0, -1), "layer_weights": (0.5, 0.5)},
            {"layer_offsets": (0, 0), "layer_weights": (0.5, 0.5)},
            {"layer_offsets": (0, 1), "layer_weights": (1.0,)},
            {"layer_weights": (0.0, 0.0, 0.0)},
            {"layer_weights": (float("nan"), 0.3, 0.2)},
            {"entropy_temperature": 0.0},
            {"effective_candidate_threshold": 0.99},
            {"max_multilayer_candidates": 1},
            {"lookahead_window": 0},
            {"improvement_epsilon": 0.0},
            {"relative_mse_epsilon": 0.0},
            {"max_candidates": 1},
            {"candidate_batch_size": 0},
        ]
        for overrides in invalid_configs:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    cgmr.CGMRV12Config(**overrides)

    def test_relative_mse_matches_required_float32_definition(self):
        candidate = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
        target = torch.tensor([[1.0, 1.0]])

        actual = cgmr._relative_mse(candidate, target, 1e-8)
        expected = (
            (candidate.float() - target.float()).pow(2).mean(dim=-1)
            / (target.float().pow(2).mean(dim=-1) + 1e-8)
        )

        self.assertTrue(torch.allclose(actual, expected))
        self.assertLess(float(actual[1]), float(actual[0]))

    def test_entropy_threshold_and_conventional_rounding(self):
        _, low_effective, _ = cgmr._distance_entropy_and_effective_count(
            [0.0, 10.0],
            0.05,
        )
        _, high_effective, _ = cgmr._distance_entropy_and_effective_count(
            [0.0, 0.0],
            0.05,
        )

        self.assertLessEqual(low_effective, 1.5)
        self.assertGreater(high_effective, 1.5)
        self.assertEqual(
            3,
            cgmr._multilayer_candidate_count(2.5, 10, 10),
        )
        self.assertEqual(
            2,
            cgmr._multilayer_candidate_count(1.6, 10, 10),
        )
        self.assertEqual(
            4,
            cgmr._multilayer_candidate_count(8.2, 4, 10),
        )
        self.assertEqual(
            3,
            cgmr._multilayer_candidate_count(8.2, 10, 3),
        )

    def test_raw_multilayer_cost_only_normalizes_effective_weights(self):
        raw = {
            24: [1.0, 2.0],
            25: [9.0, 0.0],
        }

        costs = cgmr._raw_multilayer_costs(
            raw,
            [24, 25],
            [0.5, 0.3],
            [0, 1],
        )

        self.assertAlmostEqual((0.5 / 0.8) * 1.0 + (0.3 / 0.8) * 9.0, costs[0])
        self.assertAlmostEqual((0.5 / 0.8) * 2.0, costs[1])
        self.assertLess(costs[1], costs[0])

    def test_generated_candidates_are_filtered_and_current_replaces_last(self):
        tokenizer = FakeTokenizer()
        baseline = torch.zeros(1, 2, 3)

        candidates, sources, embedding_top1 = cgmr._build_candidate_pool_v1_2(
            position=1,
            current_tokens=[9, 6],
            upstream_optimized_embedding=baseline,
            embed_layer=None,
            tokenizer=tokenizer,
            filter_nonascii=True,
            add_perplexity=True,
            top_k_ppl=3,
            top_k_cos=4,
            invert_method="cosine",
            max_candidates=3,
            embedding_top_indices=lambda *args: torch.tensor([0, 2, 3, 4]),
            get_perplexity=lambda *args, **kwargs: (
                None,
                torch.tensor([0, 2, 5]),
            ),
            model=None,
            layer_id=0,
        )

        self.assertEqual([3, 4, 6], candidates)
        self.assertEqual(3, embedding_top1)
        self.assertEqual(["embedding"], sources[3])
        self.assertEqual(["current"], sources[6])
        self.assertNotIn(0, candidates)
        self.assertNotIn(2, candidates)
        self.assertLessEqual(len(candidates), 3)

    def test_current_keeps_stable_position_and_source_or_appends_when_space_exists(self):
        candidates, sources = cgmr._merge_candidate_sources(
            [3, 4],
            [3, 5],
            current_token=3,
            max_candidates=4,
        )
        self.assertEqual([3, 4, 5], candidates)
        self.assertEqual(["embedding", "perplexity", "current"], sources[3])

        candidates, sources = cgmr._merge_candidate_sources(
            [3],
            [],
            current_token=6,
            max_candidates=3,
        )
        self.assertEqual([3, 6], candidates)
        self.assertEqual(["current"], sources[6])

    def test_filter_nonascii_false_still_filters_special_tokens(self):
        filtered = cgmr._filter_generated_candidates(
            [0, 2, 3],
            FakeTokenizer(),
            filter_nonascii=False,
        )
        self.assertEqual([2, 3], filtered)

    def test_hybrid_embeddings_use_discrete_prefix_candidate_and_baseline_tail(self):
        embed_layer = torch.nn.Embedding(6, 2)
        with torch.no_grad():
            embed_layer.weight.copy_(torch.tensor([
                [0.0, 0.5],
                [1.0, 1.5],
                [2.0, 2.5],
                [3.0, 3.5],
                [4.0, 4.5],
                [5.0, 5.5],
            ]))
        baseline = torch.tensor([[
            [10.0, 10.5],
            [11.0, 11.5],
            [12.0, 12.5],
        ]])

        hybrid = cgmr._build_hybrid_candidate_embeddings(
            baseline,
            committed_tokens=[1, 4, 5],
            position=1,
            candidate_ids=[2, 3],
            embed_layer=embed_layer,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(hybrid[:, 0], embed_layer.weight[1].expand(2, -1)))
        self.assertTrue(torch.equal(hybrid[0, 1], embed_layer.weight[2]))
        self.assertTrue(torch.equal(hybrid[1, 1], embed_layer.weight[3]))
        self.assertTrue(torch.equal(hybrid[:, 2], baseline[0, 2].expand(2, -1)))

    def test_frozen_baseline_snapshot_is_not_changed_by_later_repairs(self):
        embed_layer = torch.nn.Embedding(4, 2)
        optimized = torch.tensor([[
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
        ]])
        frozen = optimized.detach().clone()
        optimized.add_(100.0)

        hybrid = cgmr._build_hybrid_candidate_embeddings(
            frozen,
            committed_tokens=[1, 2, 3],
            position=1,
            candidate_ids=[0],
            embed_layer=embed_layer,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(hybrid[0, 2], frozen[0, 2]))
        self.assertFalse(torch.equal(hybrid[0, 2], optimized[0, 2]))

    def test_attention_mask_expansion_and_shape_validation(self):
        mask = torch.tensor([[1, 1, 0]])
        expanded = cgmr._expand_attention_mask_for_embeddings(
            mask,
            batch_size=2,
            sequence_length=3,
            device=torch.device("cpu"),
        )

        self.assertEqual((2, 3), tuple(expanded.shape))
        self.assertEqual(torch.device("cpu"), expanded.device)
        with self.assertRaises(ValueError):
            cgmr._expand_attention_mask_for_embeddings(
                torch.ones(3, 3),
                batch_size=2,
                sequence_length=3,
                device=torch.device("cpu"),
            )
        with self.assertRaises(ValueError):
            cgmr._expand_attention_mask_for_embeddings(
                torch.ones(1, 2),
                batch_size=2,
                sequence_length=3,
                device=torch.device("cpu"),
            )


class CGMRV12OnlineSelectionTests(unittest.TestCase):
    def _run_mocked(
            self,
            entry_token=10,
            sequence_length=2,
            layer_zero=None,
            layer_one=None,
            lookahead=None,
            model_layer_count=2,
            eval_start_pos=0,
            fixed_prefix_tokens=None,
            prefixes=None,
            input_token_source="embedding_top1"):
        config = cgmr.CGMRV12Config(
            enabled=True,
            entropy_temperature=0.05,
            effective_candidate_threshold=1.5,
            improvement_epsilon=1e-6,
        )
        current_tokens = [entry_token] + [30] * (sequence_length - 1)
        if eval_start_pos:
            current_tokens[0] = 999
        layer_zero = layer_zero or [0.1, 0.11]
        layer_one = layer_one or [1.0, 0.0]
        lookahead = lookahead or [0.2, 0.1]
        prefixes = prefixes if prefixes is not None else []

        def candidate_pool(position, tokens, *args, **kwargs):
            del args, kwargs
            prefixes.append((position, list(tokens)))
            current = int(tokens[position])
            if position == eval_start_pos:
                alternative = 10 if current == 20 else 20
            else:
                alternative = current + 1
            return [current, alternative], {
                current: ["current"],
                alternative: ["embedding"],
            }, alternative

        def scores(
                model,
                embed_layer,
                baseline_embedding,
                tokens,
                position,
                candidates,
                target_hidden_states,
                effective_layers,
                attention_mask,
                batch_size,
                device,
                target_layer,
                lookahead_positions,
                epsilon):
            del model, embed_layer, baseline_embedding, tokens, candidates
            del target_hidden_states, attention_mask, batch_size, device
            del target_layer, epsilon
            if position == eval_start_pos:
                by_layer = {0: list(layer_zero)}
                if 1 in effective_layers:
                    by_layer[1] = list(layer_one)
                ahead = {
                    next_position: list(lookahead)
                    for next_position in lookahead_positions
                }
                return by_layer, ahead
            by_layer = {
                layer_id: [0.0, 10.0]
                for layer_id in effective_layers
            }
            ahead = {
                next_position: [0.0, 10.0]
                for next_position in lookahead_positions
            }
            return by_layer, ahead

        target_hidden_states = {0: object()}
        if model_layer_count > 1:
            target_hidden_states[1] = object()
        with mock.patch.object(
                cgmr,
                "_build_candidate_pool_v1_2",
                side_effect=candidate_pool), mock.patch.object(
                    cgmr,
                    "_score_hybrid_candidates",
                    side_effect=scores):
            return cgmr.run_cgmr_v1_2(
                model=None,
                embed_layer=None,
                upstream_optimized_embedding=None,
                current_tokens=current_tokens,
                target_hidden_states=target_hidden_states,
                attention_mask=None,
                target_layer=0,
                model_layer_count=model_layer_count,
                model_device=None,
                config=config,
                tokenizer=None,
                filter_nonascii=True,
                add_perplexity=False,
                top_k_ppl=0,
                top_k_cos=0,
                invert_method="cosine",
                eval_start_pos=eval_start_pos,
                fixed_prefix_tokens=fixed_prefix_tokens or [],
                input_token_source=input_token_source,
                embedding_top_indices=None,
                select_candidate_from_top_indices=None,
                get_perplexity=None,
            )

    def test_single_effective_layer_runs_and_rejects_high_entropy_multilayer(self):
        prefixes = []
        tokens, result = self._run_mocked(
            entry_token=10,
            sequence_length=2,
            layer_zero=[0.1, 0.1],
            model_layer_count=1,
            eval_start_pos=1,
            fixed_prefix_tokens=[7],
            prefixes=prefixes,
        )

        self.assertEqual(7, tokens[0])
        self.assertEqual([(1, [7, 30])], prefixes)
        self.assertEqual(1, result["evaluated_position_count"])
        self.assertEqual([], result["multilayer_positions"])
        event = result["events"][0]
        self.assertTrue(event["high_entropy"])
        self.assertEqual("layer_l_multilayer_rejected", event["selection_source"])
        self.assertEqual("insufficient_effective_layers", event["selection_reason"])

    def test_multilayer_and_lookahead_improvements_accept_candidate(self):
        tokens, result = self._run_mocked()

        self.assertEqual(20, tokens[0])
        first = result["events"][0]
        self.assertTrue(first["accepted"])
        self.assertTrue(first["multilayer_accepted"])
        self.assertEqual(
            "multilayer_and_lookahead_accepted",
            first["selection_source"],
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["multilayer_accepted"])
        self.assertEqual(1, result["multilayer_accepted_count"])

    def test_lookahead_without_improvement_keeps_layer_l_best(self):
        tokens, result = self._run_mocked(lookahead=[0.1, 0.2])

        self.assertEqual(10, tokens[0])
        first = result["events"][0]
        self.assertFalse(first["multilayer_accepted"])
        self.assertEqual("layer_l_multilayer_rejected", first["selection_source"])
        self.assertEqual(
            "lookahead_improvement_not_above_epsilon",
            first["selection_reason"],
        )

    def test_last_position_can_accept_without_lookahead(self):
        tokens, result = self._run_mocked(sequence_length=1)

        self.assertEqual([20], tokens)
        first = result["events"][0]
        self.assertFalse(first["lookahead_available"])
        self.assertEqual(
            "multilayer_without_lookahead_accepted",
            first["selection_source"],
        )

    def test_multilayer_acceptance_is_distinct_from_entry_token_change(self):
        tokens, result = self._run_mocked(
            entry_token=20,
            sequence_length=1,
            layer_zero=[0.11, 0.1],
            layer_one=[0.0, 1.0],
        )

        self.assertEqual([20], tokens)
        first = result["events"][0]
        self.assertFalse(first["accepted"])
        self.assertTrue(first["multilayer_accepted"])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["multilayer_accepted"])
        self.assertEqual(0, result["accepted_count"])
        self.assertEqual(1, result["multilayer_accepted_count"])

    def test_committed_token_is_visible_to_next_position(self):
        prefixes = []
        self._run_mocked(prefixes=prefixes)

        self.assertEqual((0, [10, 30]), prefixes[0])
        self.assertEqual((1, [20, 30]), prefixes[1])

    def test_all_input_token_sources_are_preserved_in_results_and_events(self):
        for source in (
                "embedding_top1",
                "suffix_output"):
            with self.subTest(source=source):
                _, result = self._run_mocked(
                    sequence_length=1,
                    input_token_source=source,
                )
                self.assertEqual(source, result["input_token_source"])
                self.assertEqual(source, result["events"][0]["input_token_source"])


if __name__ == "__main__":
    unittest.main()
