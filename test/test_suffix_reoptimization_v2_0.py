import inspect
import math
import os
import types
import unittest

import torch

from experiment_outputs import _resolved_suffix_v20_config
from invert import (
    ensure_suffix_v2_committed_prefix,
    experiment_exit_code_for_records,
    load_config,
    normalize_suffix_version,
    select_advanced_method,
)
from suffix_optimization_methods.method_versions import (
    suffix_reoptimization_v2_0 as v20,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Tokenizer:
    vocab_size = 32
    all_special_ids = [0]

    def decode(self, token_ids):
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return " ".join(str(int(value)) for value in values)


class _Provider:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = 0

    def top_candidates(self, **kwargs):
        self.calls += 1
        return self.candidates[:kwargs["top_k"]]


class _ToyBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _ToyBody(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=4):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([_ToyBlock()])


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ToyBody()
        self.lm_head = torch.nn.Linear(4, 32, bias=False)
        with torch.no_grad():
            values = torch.arange(128, dtype=torch.float32).reshape(32, 4)
            self.model.embed_tokens.weight.copy_(values / 32.0 - 2.0)
            self.lm_head.weight.copy_(values / 64.0 - 1.0)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        del attention_mask
        hidden = (
            self.model.embed_tokens(input_ids)
            if inputs_embeds is None else inputs_embeds
        )
        for layer in self.model.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(logits=self.lm_head(hidden))


class JointErrorTests(unittest.TestCase):
    def test_config_rejects_wrong_types_ranges_and_nonfinite_values(self):
        with self.assertRaises(TypeError):
            v20.SuffixReoptimizationV20Config(phase1_epoch=1.5)
        with self.assertRaises(TypeError):
            v20.SuffixReoptimizationV20Config(classifier_enabled="false")
        with self.assertRaises(ValueError):
            v20.SuffixReoptimizationV20Config(phase1_lr=math.inf)
        with self.assertRaises(ValueError):
            v20.SuffixReoptimizationV20Config(layer_offsets=(0, -1))

    def test_equal_direction_magnitude_and_joint_are_zero(self):
        hidden = torch.tensor([[1.0, 2.0]])
        joint, direction, magnitude = v20.direction_magnitude_joint_error(
            hidden, hidden
        )
        self.assertAlmostEqual(0.0, float(joint), places=7)
        self.assertAlmostEqual(0.0, float(direction), places=7)
        self.assertAlmostEqual(0.0, float(magnitude), places=7)

    def test_direction_and_magnitude_cases_match_definition(self):
        _, same_direction, different_magnitude = (
            v20.direction_magnitude_joint_error(
                torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, 0.0]])
            )
        )
        _, orthogonal, same_magnitude = v20.direction_magnitude_joint_error(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
        )
        _, opposite, opposite_magnitude = v20.direction_magnitude_joint_error(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
        )
        self.assertAlmostEqual(0.0, float(same_direction), places=7)
        self.assertGreater(float(different_magnitude), 0.0)
        self.assertAlmostEqual(0.5, float(orthogonal), places=7)
        self.assertAlmostEqual(0.0, float(same_magnitude), places=7)
        self.assertAlmostEqual(1.0, float(opposite), places=7)
        self.assertAlmostEqual(0.0, float(opposite_magnitude), places=7)

    def test_zero_and_near_zero_vectors_remain_finite(self):
        for current, target in (
            (torch.zeros(2), torch.zeros(2)),
            (torch.tensor([1e-30, 0.0]), torch.tensor([0.0, 1e-30])),
        ):
            values = v20.direction_magnitude_joint_error(current, target)
            self.assertTrue(all(torch.isfinite(value).all() for value in values))

    def test_effective_layers_prune_and_renormalize(self):
        layers, weights, filtered = v20.resolve_effective_layers(
            2, 4, (0, 1, 2), (1.0, 0.5, 0.25)
        )
        self.assertEqual([2, 3], layers)
        self.assertEqual([4], filtered)
        self.assertAlmostEqual(1.0, sum(weights))
        self.assertAlmostEqual(2.0 / 3.0, weights[0])


class V20OrchestrationRegressionTests(unittest.TestCase):
    def test_v2_prepends_eos_when_prompt_has_no_special_prefix(self):
        tokenizer = types.SimpleNamespace(eos_token_id=31, all_special_ids=[0, 31])
        input_ids = torch.tensor([[7, 8]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        actual_ids, actual_mask = ensure_suffix_v2_committed_prefix(
            input_ids,
            attention_mask,
            tokenizer,
            "suffix_reoptimization_v2.0",
        )

        self.assertEqual([[31, 7, 8]], actual_ids.tolist())
        self.assertEqual([[1, 1, 1]], actual_mask.tolist())

    def test_v2_does_not_duplicate_an_existing_special_prefix(self):
        tokenizer = types.SimpleNamespace(eos_token_id=31, all_special_ids=[0, 31])
        input_ids = torch.tensor([[0, 7, 8]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        actual_ids, actual_mask = ensure_suffix_v2_committed_prefix(
            input_ids,
            attention_mask,
            tokenizer,
            "suffix_reoptimization_v2.0",
        )

        self.assertIs(actual_ids, input_ids)
        self.assertIs(actual_mask, attention_mask)

    def test_other_methods_keep_original_target_tokens(self):
        tokenizer = types.SimpleNamespace(eos_token_id=31, all_special_ids=[0, 31])
        input_ids = torch.tensor([[7, 8]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        actual_ids, actual_mask = ensure_suffix_v2_committed_prefix(
            input_ids,
            attention_mask,
            tokenizer,
            "suffix_v1.2.3",
        )

        self.assertIs(actual_ids, input_ids)
        self.assertIs(actual_mask, attention_mask)

    def test_v2_requires_eos_to_be_a_special_token(self):
        tokenizer = types.SimpleNamespace(eos_token_id=31, all_special_ids=[0])
        with self.assertRaisesRegex(ValueError, "EOS special token"):
            ensure_suffix_v2_committed_prefix(
                torch.tensor([[7]], dtype=torch.long),
                torch.ones((1, 1), dtype=torch.long),
                tokenizer,
                "suffix_reoptimization_v2.0",
            )

    def test_fatal_sample_makes_non_worker_experiment_fail(self):
        self.assertEqual(
            2,
            experiment_exit_code_for_records([
                {"global_index": 0, "fatal_failure": True},
            ]),
        )
        self.assertEqual(
            0,
            experiment_exit_code_for_records([
                {"global_index": 0, "fatal_failure": False},
            ]),
        )


class V20HalfPrecisionRegressionTests(unittest.TestCase):
    def test_phase2_adam_keeps_zero_gradient_fp16_coordinates_finite(self):
        model = _ToyModel().half()
        embed_layer = model.model.embed_tokens
        total_ids = torch.tensor([[0, 2, 3]], dtype=torch.long)
        full_embedding = embed_layer(total_ids).detach()
        target_hidden = {0: full_embedding.clone()}

        phase2_full, phase2_optimizable, _ = v20._optimize_phase(
            model=model,
            optimizable=full_embedding[:, 1:, :],
            prefix_embedding=full_embedding[:, :1, :],
            targets=target_hidden,
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            layers=[0],
            weights=[1.0],
            valid_mask=torch.tensor([False, True, True]),
            epoch=1,
            lr=1e-3,
            direction_weight=0.1,
            magnitude_weight=0.9,
            range_bound=torch.tensor(10.0, dtype=torch.float16),
            range_weight=0.001,
            prox_weight=0.005,
            prox_reference=full_embedding[:, 1:, :],
            optimizer_name="Adam",
            clip=False,
        )

        self.assertTrue(torch.isfinite(phase2_full).all())
        self.assertTrue(torch.isfinite(phase2_optimizable).all())


class CandidateAndDecisionTests(unittest.TestCase):
    def test_current_deduplicates_and_merges_all_candidate_sources(self):
        pool = v20.merge_candidate_sources(
            [
                ("embedding", [7, 2]),
                ("perplexity", [7, 3]),
                ("classifier", [
                    v20.ClassifierCandidate(token_id=7, score=0.2, rank=1),
                ]),
                ("current", [7]),
            ],
            _Tokenizer(), True,
        )
        self.assertEqual([7, 2, 3], [item["token_id"] for item in pool])
        current = pool[0]
        self.assertEqual(
            ["embedding", "perplexity", "classifier", "current"],
            current["sources"],
        )
        self.assertEqual(
            {"embedding": 1, "perplexity": 1, "classifier": 1, "current": 1},
            current["source_ranks"],
        )

    def test_classifier_provider_validation_is_strict(self):
        valid = _Provider([
            v20.ClassifierCandidate(token_id=2, score=0.2, rank=1),
        ])
        returned = v20.validate_classifier_candidates(
            valid, 1, [0], torch.zeros(1), 1
        )
        self.assertEqual(1, len(returned))
        invalid = _Provider([
            v20.ClassifierCandidate(token_id=2, score=math.inf, rank=1),
        ])
        with self.assertRaises(v20.SuffixV20FatalError):
            v20.merge_candidate_sources(
                [("classifier", invalid.top_candidates(top_k=10))],
                _Tokenizer(), True,
            )
        duplicate_rank = _Provider([
            v20.ClassifierCandidate(token_id=2, score=0.2, rank=1),
            v20.ClassifierCandidate(token_id=3, score=0.3, rank=1),
        ])
        with self.assertRaisesRegex(
                v20.SuffixV20FatalError, "duplicate_classifier_result"):
            v20.validate_classifier_candidates(
                duplicate_rank, 1, [0], torch.zeros(1), 2
            )

    def test_classifier_disabled_is_not_called_and_enabled_missing_is_fatal(self):
        engine = v20._V20Engine.__new__(v20._V20Engine)
        engine.config = v20.SuffixReoptimizationV20Config(
            classifier_enabled=False
        )
        engine.tokenizer = _Tokenizer()
        engine.filter_nonascii = True
        engine.classifier_provider = _Provider([])
        engine.classifier_candidate_count = 0
        engine.continuous = torch.zeros((1, 2, 2))
        engine.embedding_candidates = lambda position, top_k: [2, 3]
        engine.ppl_candidates = lambda position, committed: [4]
        candidates, _ = engine.candidate_pool(
            1, [0, 2], 20, include_classifier=True
        )
        self.assertEqual(0, engine.classifier_provider.calls)
        self.assertEqual([2, 3, 4], [item["token_id"] for item in candidates])
        engine.config.classifier_enabled = True
        engine.classifier_provider = None
        with self.assertRaisesRegex(
                v20.SuffixV20FatalError, "classifier_provider_unavailable"):
            engine.candidate_pool(1, [0, 2], 20, include_classifier=True)

    def test_current_outside_embedding_and_ppl_topk_enters_normal_pool(self):
        engine = v20._V20Engine.__new__(v20._V20Engine)
        engine.config = v20.SuffixReoptimizationV20Config()
        engine.tokenizer = _Tokenizer()
        engine.filter_nonascii = True
        engine.classifier_provider = None
        engine.classifier_candidate_count = 0
        engine.continuous = torch.zeros((1, 3, 2))
        calls = []
        engine.embedding_candidates = lambda position, top_k: (
            calls.append(("embedding", top_k)) or list(range(1, top_k + 1))
        )
        engine.ppl_candidates = lambda position, committed: (
            calls.append(("ppl", engine.config.ppl_top_k)) or [21, 22]
        )
        normal, _ = engine.candidate_pool(
            1, [0, 7, 8], 10, include_current=31
        )
        self.assertEqual(
            [("embedding", 10), ("ppl", 10)],
            calls,
        )
        current = next(item for item in normal if item["token_id"] == 31)
        self.assertEqual(["current"], current["sources"])
        self.assertNotIn("embedding", current["source_ranks"])
        self.assertNotIn("perplexity", current["source_ranks"])

    def _run_stage4_with_fake_scores(self, entry_tokens, continuous_values,
                                     score_by_position):
        engine = types.SimpleNamespace()
        engine.config = v20.SuffixReoptimizationV20Config()
        calls = []

        def candidate_pool(position, working_tokens, embedding_top_k,
                           include_current=None, include_classifier=False):
            calls.append({
                "position": position,
                "working_token": working_tokens[position],
                "embedding_top_k": embedding_top_k,
                "include_current": include_current,
                "include_classifier": include_classifier,
            })
            candidates = v20.merge_candidate_sources(
                [("embedding", [2]), ("perplexity", []),
                 ("current", [include_current])],
                _Tokenizer(), True,
            )
            return candidates, False

        def score_candidates(position, working_tokens, candidates):
            del working_tokens
            for entry in candidates:
                score = float(score_by_position[position][entry["token_id"]])
                entry["score"] = score
                entry["target_layer_score"] = score
                entry["layer_scores"] = {"0": score}
            return sorted(candidates, key=v20.candidate_tie_break_key)

        engine.candidate_pool = candidate_pool
        engine.score_candidates = score_candidates
        result = v20._run_discrete(
            engine,
            entry_tokens,
            sorted(continuous_values),
            continuous_values,
            enable_repairs=False,
        )
        return result, calls

    def test_current_enters_expanded_stage4_pool(self):
        (_, _, _, events, _, _), calls = self._run_stage4_with_fake_scores(
            [0, 7, 8],
            {1: 0.0, 2: 10.0},
            {1: {2: 0.1, 7: 0.2}, 2: {2: 0.1, 8: 0.2}},
        )
        self.assertEqual(20, calls[1]["embedding_top_k"])
        self.assertEqual(8, calls[1]["working_token"])
        self.assertEqual(8, calls[1]["include_current"])
        self.assertTrue(calls[1]["include_classifier"])
        current = next(
            item for item in events[1]["stage4_candidates"]
            if item["token_id"] == 8
        )
        self.assertIn("current", current["sources"])

    def test_current_can_remain_when_its_stage4_score_is_lowest(self):
        (tokens, _, _, events, _, _), calls = self._run_stage4_with_fake_scores(
            [0, 7], {1: 0.0}, {1: {2: 0.2, 7: 0.1}}
        )
        self.assertEqual(7, calls[0]["include_current"])
        self.assertEqual([0, 7], tokens)
        self.assertEqual(7, events[0]["stage4_selected_token"])
        self.assertEqual(2, events[0]["candidate_count"])

    def test_current_can_be_replaced_when_another_stage4_score_is_lower(self):
        (tokens, _, _, events, _, _), _ = self._run_stage4_with_fake_scores(
            [0, 7], {1: 0.0}, {1: {2: 0.1, 7: 0.2}}
        )
        self.assertEqual([0, 2], tokens)
        self.assertEqual(2, events[0]["stage4_selected_token"])

    def test_legal_current_prevents_empty_pool_fallback(self):
        engine = v20._V20Engine.__new__(v20._V20Engine)
        engine.config = v20.SuffixReoptimizationV20Config()
        engine.tokenizer = _Tokenizer()
        engine.filter_nonascii = True
        engine.classifier_provider = None
        engine.classifier_candidate_count = 0
        engine.continuous = torch.zeros((1, 2, 2))
        embedding_calls = []
        engine.embedding_candidates = lambda position, top_k: (
            embedding_calls.append(top_k) or []
        )
        engine.ppl_candidates = lambda position, committed: []
        candidates, generation_failed = engine.candidate_pool(
            1, [0, 7], 10, include_current=7
        )
        self.assertFalse(generation_failed)
        self.assertEqual([10], embedding_calls)
        self.assertEqual([7], [item["token_id"] for item in candidates])
        self.assertEqual(["current"], candidates[0]["sources"])

    def test_stage4_current_snapshot_does_not_read_ground_truth_tokens(self):
        source = inspect.getsource(v20._run_discrete)
        self.assertIn("current_token_i = int(tokens[position])", source)
        self.assertIn("include_current=current_token_i", source)
        self.assertNotIn("total_input_ids", source)
        self.assertNotIn("ground_truth", source.lower())

    def test_ppl_receives_only_committed_left_prefix(self):
        engine = v20._V20Engine.__new__(v20._V20Engine)
        engine.target_layer = 24
        engine.config = v20.SuffixReoptimizationV20Config()
        engine.model = object()
        seen = []

        def get_perplexity(prefix, model, layer_id, top_k):
            seen.append((list(prefix), model, layer_id, top_k))
            return torch.ones(2), torch.tensor([2, 3])

        engine.get_perplexity = get_perplexity
        self.assertEqual([2, 3], engine.ppl_candidates(2, [0, 7, 99, 100]))
        self.assertEqual(([0, 7], engine.model, 24, 10), seen[0])

    def test_tie_break_is_score_layer_source_ranks_then_token(self):
        base = {
            "score": 1.0,
            "target_layer_score": 1.0,
            "source_ranks": {"embedding": 2, "perplexity": 2},
            "token_id": 9,
        }
        by_layer = dict(base, target_layer_score=0.5, token_id=20)
        by_embedding = dict(base, source_ranks={"embedding": 1}, token_id=30)
        by_token = dict(base, token_id=8)
        ordered = sorted(
            [base, by_token, by_embedding, by_layer],
            key=v20.candidate_tie_break_key,
        )
        self.assertIs(by_layer, ordered[0])
        self.assertIs(by_embedding, ordered[1])
        self.assertIs(by_token, ordered[2])

    def test_repair_requires_strict_epsilon_improvement(self):
        self.assertTrue(v20.should_replace(1.0, 0.9, 1e-8))
        self.assertFalse(v20.should_replace(1.0, 1.0 - 1e-9, 1e-8))
        source = inspect.getsource(v20.should_replace)
        self.assertIn("<", source)
        self.assertNotIn("<=", source)

    def test_local_threshold_uses_only_prior_history(self):
        config = v20.SuffixReoptimizationV20Config(local_min_points=4)
        suspicious, info = v20.local_anomaly_decision(
            10.0, 0.0, [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0], 20.0, config,
        )
        self.assertTrue(suspicious)
        self.assertEqual("prefix_median_mad", info["mode"])
        warmup, warmup_info = v20.local_anomaly_decision(
            2.0, 0.0, [], [], 1.5, config,
        )
        self.assertTrue(warmup)
        self.assertEqual("continuous_threshold_warmup", warmup_info["mode"])

    def test_cusum_uses_robust_history_and_strict_threshold(self):
        config = v20.SuffixReoptimizationV20Config(
            cumulative_min_points=4,
            cumulative_kappa=0.5,
            cumulative_threshold=5.0,
        )
        state = v20.update_cumulative_state(
            1.0, [0.0, 0.0, 0.0, 0.0], 0.0, config, 4
        )
        self.assertGreater(state["S"], 5.0)
        self.assertTrue(state["triggered"])
        self.assertEqual(4, state["segment_start"])


class WiringAndFailureTests(unittest.TestCase):
    def test_selector_aliases_and_strict_disabled_failure(self):
        for alias in (
            "2.0", "v2.0", "suffix_v2_0",
            "suffix_reoptimization_v2.0", "suffix_reoptimization_v2_0",
        ):
            self.assertEqual("v2.0", normalize_suffix_version(alias))
        disabled = types.SimpleNamespace(enabled=False)
        enabled = v20.SuffixReoptimizationV20Config(enabled=True)
        self.assertEqual(
            v20.METHOD_NAME,
            select_advanced_method(
                "v2.0", disabled, disabled, disabled, disabled,
                suffix_reopt_v2_0_config=enabled,
            ),
        )
        with self.assertRaisesRegex(ValueError, "v2.0.*disabled"):
            select_advanced_method(
                "v2.0", disabled, disabled, disabled, disabled,
                suffix_reopt_v2_0_config=v20.SuffixReoptimizationV20Config(),
            )

    def test_formal_config_is_v2_only_and_classifier_disabled(self):
        path = os.path.join(
            ROOT, "experiment_configs",
            "l24_airport_medical_suffix_v2_0_no_cgmr.json",
        )
        config = load_config(path)
        self.assertEqual("v2.0", config["suffix_version"])
        self.assertTrue(config["suffix_reoptimization_v2_0"])
        self.assertFalse(config["suffix_v2_0_classifier_enabled"])
        self.assertEqual("none", config["cgmr_version"])
        self.assertEqual(24, config["num_invert_layers"])
        self.assertEqual([5, 5], [item["len"] for item in config["datasets"]])
        old_flags = [
            key for key in config
            if (
                key.startswith("suffix_reoptimization_v1_")
                and not key.endswith("_log")
            )
            or key == "suffix_v1_2_3"
        ]
        self.assertTrue(old_flags)
        self.assertTrue(all(config[key] is False for key in old_flags))

    def test_resolved_defaults_record_classifier_interface(self):
        resolved = _resolved_suffix_v20_config(types.SimpleNamespace())
        self.assertFalse(resolved["classifier_enabled"])
        self.assertFalse(resolved["classifier_provider_available"])
        self.assertEqual(0, resolved["classifier_candidate_count"])
        self.assertEqual("hard_failure_only_rollback", resolved["final_acceptance"])

    def test_missing_first_position_prefix_rolls_back_entry_snapshot(self):
        config = v20.SuffixReoptimizationV20Config(
            enabled=True, phase1_epoch=1, phase2_epoch=1
        )
        embedding = torch.zeros((1, 2, 2))
        final_embedding, result = v20.run_suffix_reoptimization_v2_0(
            model=None,
            embed_layer=None,
            initial_optimizable_embedding=embedding,
            prefix_embedding=None,
            target_hidden_states={0: torch.zeros_like(embedding)},
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            layer_id=0,
            model_layer_count=1,
            tokenizer=_Tokenizer(),
            total_input_ids=torch.tensor([[2, 3]]),
            right_range=torch.tensor(1.0),
            config=config,
            embedding_top_indices=lambda *args: None,
            select_candidate_from_top_indices=lambda *args: (None, [2]),
            get_perplexity=lambda *args, **kwargs: None,
            entry_tokens=[4, 5],
            eval_start_pos=0,
        )
        self.assertTrue(result["rollback"])
        self.assertTrue(result["fatal_failure"])
        self.assertEqual("missing_committed_prefix_for_ppl", result["rollback_reason"])
        self.assertEqual([4, 5], result["final_tokens"])
        self.assertTrue(torch.equal(embedding, final_embedding))

    def test_hard_failure_invariants_cover_length_prefix_and_padding(self):
        snapshot = v20._invariant_snapshot(
            [0, 2, 3, 4], torch.tensor([[1, 1, 1, 0]]), 1
        )
        cases = (
            ([0, 2, 3], "token_sequence_length_changed"),
            ([9, 2, 3, 4], "special_prefix_changed"),
            ([0, 2, 3, 8], "padding_changed"),
        )
        for tokens, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(v20.SuffixV20FatalError, reason):
                    v20._assert_invariants(tokens, snapshot, "test")

    def test_nonfinite_hidden_is_a_hard_failure(self):
        with self.assertRaisesRegex(v20.SuffixV20FatalError, "nonfinite"):
            v20.direction_magnitude_joint_error(
                torch.tensor([[math.nan, 0.0]]), torch.zeros((1, 2))
            )

    def test_legal_candidate_pool_total_failure_is_fatal(self):
        engine = v20._V20Engine.__new__(v20._V20Engine)
        engine.config = v20.SuffixReoptimizationV20Config()
        engine.tokenizer = _Tokenizer()
        engine.filter_nonascii = True
        engine.classifier_provider = None
        engine.classifier_candidate_count = 0
        engine.continuous = torch.zeros((1, 2, 2))
        engine.embedding_candidates = lambda position, top_k: [0]
        engine.ppl_candidates = lambda position, committed: [0]
        with self.assertRaisesRegex(
                v20.SuffixV20FatalError, "legal_candidate_pool_generation_failed"):
            engine.candidate_pool(1, [0, 2], 10)

    def test_full_toy_flow_accepts_and_diagnostics_do_not_change_tokens(self):
        model = _ToyModel()
        embed_layer = model.model.embed_tokens
        total_ids = torch.tensor([[0, 2, 3]], dtype=torch.long)
        target_hidden = {0: embed_layer(total_ids).detach()}
        prefix = embed_layer(total_ids[:, :1]).detach()
        initial = embed_layer(total_ids[:, 1:]).detach().clone()

        def embedding_top_indices(embed, layer, top_k, invert_method):
            self.assertEqual("cosine", invert_method)
            scores = torch.nn.functional.cosine_similarity(
                embed.float().unsqueeze(0), layer.weight.detach().float(), dim=-1
            )
            return torch.topk(scores, min(top_k, scores.numel())).indices

        def select_candidates(indices, tokenizer, filter_nonascii):
            values = [int(value) for value in indices.detach().cpu().tolist()]
            legal = [
                value for value in values
                if value not in tokenizer.all_special_ids
                and (not filter_nonascii or tokenizer.decode([value]).isascii())
            ]
            return (legal[0] if legal else None), values

        def ppl(prefix_tokens, current_model, layer_id, top_k):
            del prefix_tokens, current_model, layer_id
            values = torch.arange(1, top_k + 1, dtype=torch.long)
            return torch.ones(top_k), values

        results = []
        for diagnostics_enabled in (True, False):
            config = v20.SuffixReoptimizationV20Config(
                enabled=True,
                phase1_epoch=1,
                phase2_epoch=1,
                accuracy_diagnostics_enabled=diagnostics_enabled,
            )
            _, result = v20.run_suffix_reoptimization_v2_0(
                model=model,
                embed_layer=embed_layer,
                initial_optimizable_embedding=initial,
                prefix_embedding=prefix,
                target_hidden_states=target_hidden,
                attention_mask=torch.ones((1, 3), dtype=torch.long),
                layer_id=0,
                model_layer_count=1,
                tokenizer=_Tokenizer(),
                total_input_ids=total_ids,
                right_range=torch.tensor(10.0),
                config=config,
                embedding_top_indices=embedding_top_indices,
                select_candidate_from_top_indices=select_candidates,
                get_perplexity=ppl,
                entry_tokens=[0, 4, 5],
                eval_start_pos=1,
            )
            self.assertTrue(result["accepted"])
            self.assertFalse(result["rollback"])
            self.assertEqual("SGD", result["stage1"]["optimizer"])
            self.assertTrue(
                result["stage1"]["optimizer_recreated_each_step"]
            )
            self.assertEqual("Adam", result["reoptimization"]["optimizer"])
            self.assertFalse(
                result["reoptimization"]["optimizer_recreated_each_step"]
            )
            self.assertTrue(math.isfinite(
                result["final_repaired_global_joint_error"]
            ))
            results.append(result)
        self.assertEqual(results[0]["final_tokens"], results[1]["final_tokens"])
        self.assertTrue(results[0]["diagnostics"])
        self.assertFalse(results[1]["diagnostics"])

    def test_v2_source_has_no_relmse_or_global_acceptance_gate(self):
        source = inspect.getsource(v20)
        self.assertNotIn("relative_mse", source.lower())
        self.assertNotIn("global_final <=", source)
        self.assertNotIn("accuracy_tolerance", source)


if __name__ == "__main__":
    unittest.main()
