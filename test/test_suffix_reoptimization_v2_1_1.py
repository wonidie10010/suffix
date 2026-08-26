import inspect
import json
import math
import os
import types
import unittest
from unittest import mock

import torch

from suffix_optimization_methods.method_versions import (
    suffix_reoptimization_v2_1_1 as v21,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Tokenizer:
    vocab_size = 32
    all_special_ids = [0]
    pad_token_id = 1

    def decode(self, token_ids):
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return "".join(
            "é" if int(value) == 2 else "t{}".format(int(value))
            for value in values
        )


class _PaddedTokenizer:
    vocab_size = 5
    all_special_ids = [0, 1, 6]
    pad_token_id = 1

    def __len__(self):
        return 7

    def decode(self, token_ids):
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return "".join(
            "special" if int(value) == 6 else "t{}".format(int(value))
            for value in values
        )


class _OverfullPaddedTokenizer(_PaddedTokenizer):
    def __len__(self):
        return 11


class _ToyBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 0.1 * hidden.cumsum(dim=1)


class _ToyBody(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=4, layer_count=3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [_ToyBlock() for _ in range(layer_count)]
        )


class _ToyModel(torch.nn.Module):
    def __init__(
            self, *, model_type="qwen2", architecture="Qwen2ForCausalLM",
            vocab_size=32):
        super().__init__()
        self.config = types.SimpleNamespace(
            model_type=model_type,
            architectures=[architecture],
        )
        self.model = _ToyBody(vocab_size=vocab_size)
        self.lm_head = torch.nn.Linear(4, vocab_size, bias=False)
        self.forward_use_cache = []
        with torch.no_grad():
            values = torch.arange(
                vocab_size * 4, dtype=torch.float32
            ).reshape(vocab_size, 4)
            self.model.embed_tokens.weight.copy_(values / 40.0 - 1.5)
            self.lm_head.weight.copy_(values / 80.0 - 0.75)

    def forward(
            self, input_ids=None, inputs_embeds=None, attention_mask=None,
            use_cache=None):
        del attention_mask
        self.forward_use_cache.append(use_cache)
        hidden = (
            self.model.embed_tokens(input_ids)
            if inputs_embeds is None else inputs_embeds
        )
        for layer in self.model.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(logits=self.lm_head(hidden))


class _GroundTruthBomb:
    @property
    def detach(self):
        raise AssertionError("Ground Truth was read")


def _config(**overrides):
    values = {
        "enabled": True,
        "vocab_anchor_top_k": 2,
        "vocab_anchor_refresh_interval": 1,
        "global_steps": 1,
        "local_steps": 1,
        "embedding_top_k_normal": 2,
        "embedding_top_k_expanded": 4,
        "ppl_top_k": 2,
        "tau_J": 10.0,
        "tau_r": 1.0,
    }
    values.update(overrides)
    return v21.SuffixReoptimizationV211Config(**values)


def _fixture(*, custom_fixed=False):
    model = _ToyModel()
    embed_layer = model.model.embed_tokens
    tokens = [0, 3, 4, 1]
    token_tensor = torch.tensor([tokens], dtype=torch.long)
    entry = embed_layer(token_tensor).detach().clone()
    if custom_fixed:
        entry[:, 0, :] += 0.375
        entry[:, 3, :] -= 0.25
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    collected = v21._forward_hidden(
        model,
        [0, 1, 2],
        mask,
        inputs_embeds=entry,
        inference=True,
    )
    targets = {key: value.detach().clone() for key, value in collected.items()}
    model.forward_use_cache.clear()
    return model, embed_layer, tokens, entry, mask, targets


def _run_fixture(config=None, total_input_ids=None, custom_fixed=False):
    model, embed_layer, tokens, entry, mask, targets = _fixture(
        custom_fixed=custom_fixed
    )
    final_embedding, result = v21.run_suffix_reoptimization_v2_1_1(
        model=model,
        embed_layer=embed_layer,
        entry_embedding_snapshot=entry,
        entry_token_snapshot=tokens,
        target_hidden_states=targets,
        attention_mask=mask,
        layer_id=0,
        model_layer_count=3,
        tokenizer=_Tokenizer(),
        config=config or _config(),
        total_input_ids=total_input_ids,
        eval_start_pos=1,
    )
    return (
        model,
        embed_layer,
        tokens,
        entry,
        mask,
        targets,
        final_embedding,
        result,
    )


def _padded_preflight_fixture(logits_vocab_size=10):
    model = _ToyModel(vocab_size=10)
    if logits_vocab_size != 10:
        model.lm_head = torch.nn.Linear(4, logits_vocab_size, bias=False)
    embed_layer = model.model.embed_tokens
    tokens = [0, 3, 4, 1]
    entry = embed_layer(torch.tensor([tokens], dtype=torch.long)).detach()
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    targets = v21._forward_hidden(
        model,
        [0, 1, 2],
        mask,
        inputs_embeds=entry,
        inference=True,
    )
    return model, embed_layer, tokens, entry, mask, targets


class ConfigAndPureFunctionTests(unittest.TestCase):
    def test_frozen_defaults_match_the_canonical_design(self):
        config = v21.SuffixReoptimizationV211Config()
        self.assertEqual((0, 1, 2), config.layer_offsets)
        self.assertEqual((1.0, 0.5, 0.25), config.layer_weights)
        self.assertEqual((0.5, 0.5), (config.alpha_dir, config.alpha_mag))
        self.assertEqual((0.005, 0.01), (
            config.vocab_weight, config.vocab_temperature
        ))
        self.assertEqual((10, 10), (
            config.vocab_anchor_top_k,
            config.vocab_anchor_refresh_interval,
        ))
        self.assertEqual((1000, 1e-3), (
            config.global_steps, config.global_lr
        ))
        self.assertEqual((50, 1e-3), (
            config.local_steps, config.local_lr
        ))
        self.assertEqual((0.9, 0.999, 1e-8), (
            config.adam_beta1, config.adam_beta2, config.adam_epsilon
        ))
        self.assertEqual((0.15, 0.01, 0.05), (
            config.tau_J, config.delta_c_max, config.tau_r
        ))
        self.assertEqual((10, 20, 10), (
            config.embedding_top_k_normal,
            config.embedding_top_k_expanded,
            config.ppl_top_k,
        ))
        self.assertFalse(config.accuracy_diagnostics_enabled)
        self.assertTrue(config.filter_nonascii)

    def test_config_rejects_wrong_types_nonfinite_and_changed_structure(self):
        with self.assertRaises(TypeError):
            v21.SuffixReoptimizationV211Config(global_steps=1.0)
        with self.assertRaises(ValueError):
            v21.SuffixReoptimizationV211Config(global_lr=math.inf)
        with self.assertRaises(ValueError):
            v21.SuffixReoptimizationV211Config(layer_offsets=(0, 1))
        with self.assertRaises(ValueError):
            v21.SuffixReoptimizationV211Config(global_optimizer="sgd")
        with self.assertRaises(ValueError):
            v21.SuffixReoptimizationV211Config(weight_decay_enabled=True)
        with self.assertRaises(ValueError):
            v21.SuffixReoptimizationV211Config(
                embedding_top_k_normal=20,
                embedding_top_k_expanded=20,
            )

    def test_effective_layers_prune_and_renormalize(self):
        layers, weights, filtered = v21.resolve_effective_layers(
            1, 3, (0, 1, 2), (1.0, 0.5, 0.25)
        )
        self.assertEqual([1, 2], layers)
        self.assertEqual([3], filtered)
        self.assertAlmostEqual(1.0, sum(weights))
        self.assertAlmostEqual(2.0 / 3.0, weights[0])

    def test_joint_error_is_finite_and_bounded(self):
        config = _config()
        for current, target in (
            (torch.zeros(2), torch.zeros(2)),
            (torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])),
            (torch.tensor([1e-30, 0.0]), torch.tensor([0.0, 1e-30])),
        ):
            value = v21._joint_error(current, target, config)
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0)

    def test_config_file_has_only_v21_prefixed_method_fields(self):
        path = os.path.join(
            ROOT,
            "suffix_optimization_methods",
            "configs",
            "suffix_reoptimization_v2_1_1.json",
        )
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        keys = [key for key in payload if key != "__comments"]
        self.assertTrue(all(
            key == "suffix_reoptimization_v2_1_1"
            or key == "suffix_reoptimization_v2_1_1_log"
            or key.startswith("suffix_v2_1_1_")
            for key in keys
        ))
        forbidden = ("classifier", "prox", "mad", "cusum", "replay", "cumulative")
        self.assertTrue(all(
            term not in key.lower() for key in keys for term in forbidden
        ))


class LegalVocabularyTests(unittest.TestCase):
    def test_padded_model_vocab_separates_tokenizer_ids_and_logits_width(self):
        (
            model,
            embed_layer,
            tokens,
            entry,
            mask,
            targets,
        ) = _padded_preflight_fixture()
        state = v21._preflight(
            model,
            embed_layer,
            entry,
            tokens,
            targets,
            mask,
            0,
            3,
            _PaddedTokenizer(),
            _config(),
            1,
        )
        legal = state["legal_vocab"]

        self.assertEqual(7, legal.tokenizer_vocab_size)
        self.assertEqual(10, legal.vocab_size)
        self.assertIn(5, legal.ids)
        self.assertNotIn(6, legal.ids)
        self.assertTrue(all(token_id < 7 for token_id in legal.ids))
        self.assertFalse(legal.is_legal(7))
        self.assertFalse(legal.is_legal(9))

    def test_tokenizer_domain_cannot_exceed_model_embedding_domain(self):
        with self.assertRaisesRegex(
                v21.SuffixV211FatalError, "vocab_size_mismatch"):
            v21._LegalVocabulary(
                torch.nn.Embedding(10, 4),
                _OverfullPaddedTokenizer(),
                True,
                chunk_size=4,
            )

    def test_logits_contract_still_uses_model_embedding_width(self):
        (
            model,
            embed_layer,
            tokens,
            entry,
            mask,
            targets,
        ) = _padded_preflight_fixture(logits_vocab_size=9)
        with self.assertRaisesRegex(
                v21.SuffixV211FatalError,
                "causal_lm_logits_contract_failed",
        ):
            v21._preflight(
                model,
                embed_layer,
                entry,
                tokens,
                targets,
                mask,
                0,
                3,
                _PaddedTokenizer(),
                _config(),
                1,
            )

    def test_entry_snapshot_excludes_special_pad_nonascii_and_keeps_padding(self):
        model = _ToyModel()
        embed_layer = model.model.embed_tokens
        embedding = embed_layer(torch.tensor([[0, 2, 5]])).detach()
        actual = v21.build_entry_snapshot_from_embedding(
            embedding,
            embed_layer,
            _Tokenizer(),
            eval_start_pos=1,
            fixed_prefix_tokens=[0],
            attention_mask=torch.tensor([[1, 1, 0]]),
            filter_nonascii=True,
            chunk_size=3,
        )
        self.assertEqual(0, actual[0])
        self.assertNotIn(actual[1], {0, 1, 2})
        self.assertEqual(1, actual[2])

    def test_chunked_neighbor_tie_is_resolved_by_token_id(self):
        model = _ToyModel()
        embed_layer = model.model.embed_tokens
        with torch.no_grad():
            embed_layer.weight[4].copy_(embed_layer.weight[3])
        legal = v21._LegalVocabulary(
            embed_layer, _Tokenizer(), True, chunk_size=2
        )
        ids, _, distances = legal.nearest(embed_layer.weight[3], 2)
        self.assertEqual([3, 4], ids)
        self.assertEqual([0.0, 0.0], distances)

    def test_legal_count_preflight_contract_is_hard(self):
        model, embed_layer, tokens, entry, mask, targets = _fixture()
        tokenizer = _Tokenizer()
        tokenizer.vocab_size = 32
        config = _config(embedding_top_k_expanded=31)
        _, result = v21.run_suffix_reoptimization_v2_1_1(
            model, embed_layer, entry, tokens, targets, mask, 0, 3,
            tokenizer, config, eval_start_pos=1,
        )
        self.assertTrue(result["rollback"])
        self.assertEqual("legal_vocab_too_small", result["rollback_reason"])


class OptimizationAndCausalFlowTests(unittest.TestCase):
    def test_full_cpu_flow_returns_token_consistent_embedding_and_fixed_values(self):
        (
            model,
            embed_layer,
            tokens,
            entry,
            _,
            _,
            final_embedding,
            result,
        ) = _run_fixture(custom_fixed=True)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["rollback"])
        self.assertTrue(all(value is False for value in model.forward_use_cache))
        self.assertTrue(torch.equal(final_embedding[:, 0], entry[:, 0]))
        self.assertTrue(torch.equal(final_embedding[:, 3], entry[:, 3]))
        self.assertEqual(tokens[0], result["final_tokens"][0])
        self.assertEqual(tokens[3], result["final_tokens"][3])
        ids = torch.tensor(result["final_tokens"][1:3], dtype=torch.long)
        expected = embed_layer(ids).detach()
        self.assertTrue(torch.equal(final_embedding[0, 1:3], expected))
        self.assertEqual(1, result["global_optimization"]["optimizer_instances"])
        self.assertTrue(result["global_optimization"]["optimizer_persistent"])

    def test_global_anchor_refresh_does_not_recreate_adam(self):
        model, embed_layer, _, entry, mask, targets = _fixture()
        legal = v21._LegalVocabulary(embed_layer, _Tokenizer(), True, 4)
        config = _config(global_steps=3, vocab_anchor_refresh_interval=1)
        real_adam = torch.optim.Adam
        with mock.patch.object(
                v21.torch.optim,
                "Adam",
                side_effect=lambda *args, **kwargs: real_adam(*args, **kwargs),
        ) as constructor:
            _, summary = v21._global_optimize(
                model,
                entry,
                targets,
                mask,
                [1, 2],
                [0, 1, 2],
                [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0],
                legal,
                config,
            )
        self.assertEqual(1, constructor.call_count)
        self.assertEqual(3, summary["anchor_refresh_count"])
        self.assertFalse(summary["anchor_refresh_recreated_optimizer"])

    def test_mixed_context_uses_only_committed_effective_prefix(self):
        model, embed_layer, tokens, work, _, _ = _fixture(custom_fixed=True)
        del model
        committed = list(tokens)
        committed[1] = 8
        mixed = v21._mixed_context(
            work, committed, [1], embed_layer
        )
        self.assertTrue(torch.equal(mixed[:, 0], work[:, 0]))
        self.assertTrue(torch.equal(
            mixed[:, 1], embed_layer(torch.tensor([8])).detach()
        ))
        self.assertTrue(torch.equal(mixed[:, 2], work[:, 2]))
        self.assertTrue(torch.equal(mixed[:, 3], work[:, 3]))

    def test_each_position_has_at_most_one_vector_trial_and_one_expanded_attempt(self):
        result = _run_fixture(config=_config(tau_J=0.0, tau_r=0.0))[-1]
        self.assertTrue(result["accepted"])
        for event in result["events"]:
            self.assertTrue(event["vector_repair"]["triggered"])
            self.assertEqual(1, event["vector_repair"]["optimizer_instances"])
            self.assertIn(event["initial_pool_mode"], {"normal", "expanded"})
            self.assertIsInstance(event["expanded_attempt"], bool)

    def test_expanded_pool_can_add_zero_and_keeps_normal_selected_token(self):
        model, embed_layer, tokens, entry, _, targets = _fixture()
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        legal = v21._LegalVocabulary(embed_layer, _Tokenizer(), True, 4)
        state = {
            "entry_embedding": entry,
            "entry_tokens": tokens,
            "attention_mask": mask,
            "positions": [1],
            "layers": [0, 1, 2],
            "weights": [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0],
            "legal_vocab": legal,
        }
        sources = {
            "embedding_normal": [3],
            "embedding_expanded": [3, 4],
            "perplexity": [4],
            "current": [3],
        }
        scored_token_ids = []

        def score_candidates(*args):
            candidates = args[5]
            scored_token_ids.append({
                int(candidate["token_id"]) for candidate in candidates
            })
            scored = []
            for candidate in candidates:
                current = dict(candidate)
                token_id = int(current["token_id"])
                current["hidden_error"] = 0.3 if token_id == 3 else 0.4
                current["layer_hidden_errors"] = {"0": current["hidden_error"]}
                scored.append(current)
            scored.sort(key=lambda item: (
                item["hidden_error"], item["token_id"]
            ))
            return scored, []

        with mock.patch.object(
                v21, "_global_optimize", return_value=(entry, {})), mock.patch.object(
                v21, "_continuous_metrics", return_value=(0.1, 0.1, 0.1)), mock.patch.object(
                v21, "_candidate_sources", return_value=sources), mock.patch.object(
                v21, "_score_candidates", side_effect=score_candidates):
            _, final_tokens, _, events, _ = v21._formal_method(
                model,
                embed_layer,
                targets,
                _config(tau_J=1.0, tau_r=0.05),
                state,
            )

        self.assertEqual(2, len(scored_token_ids))
        normal_ids, expanded_ids = scored_token_ids
        self.assertTrue(normal_ids.issubset(expanded_ids))
        self.assertEqual(set(), expanded_ids - normal_ids)
        self.assertEqual(0, events[0]["expanded_added_count"])
        self.assertEqual(3, events[0]["selected_initial_token"])
        self.assertEqual(3, events[0]["selected_final_token"])
        self.assertEqual(3, final_tokens[1])

    def test_uncommitted_nonfinite_local_trial_falls_back_without_pollution(self):
        model, embed_layer, tokens, work, mask, targets = _fixture()
        legal = v21._LegalVocabulary(embed_layer, _Tokenizer(), True, 4)
        original = work.detach().clone()
        with mock.patch.object(
                v21,
                "_forward_hidden",
                side_effect=v21.SuffixV211FatalError(
                    "nonfinite_local_loss", "vector_repair", 1
                ),
        ):
            updated, record = v21._try_vector_repair(
                model,
                work,
                tokens,
                [],
                1,
                embed_layer,
                targets,
                mask,
                [0, 1, 2],
                [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0],
                legal,
                _config(),
                (0.2, 0.3, 0.2015),
            )
        self.assertFalse(record["trial_safe"])
        self.assertFalse(record["accepted"])
        self.assertTrue(torch.equal(updated, original))


class CandidateAndFailureDomainTests(unittest.TestCase):
    def _score_with_totals(self, totals):
        model, embed_layer, tokens, work, mask, targets = _fixture()
        candidates = [
            {"token_id": 4, "sources": ["embedding_normal"], "source_ranks": {}},
            {"token_id": 3, "sources": ["current"], "source_ranks": {}},
        ][:len(totals)]
        with mock.patch.object(
                v21,
                "_multi_hidden_error",
                return_value=torch.tensor(totals, dtype=torch.float32).reshape(-1, 1),
        ):
            return v21._score_candidates(
                model,
                work,
                tokens,
                [],
                1,
                candidates,
                embed_layer,
                targets,
                mask,
                [0, 1, 2],
                [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0],
                _config(),
            )

    def test_candidate_tie_break_is_exactly_hidden_error_then_token_id(self):
        scored, dropped = self._score_with_totals([0.25, 0.25])
        self.assertFalse(dropped)
        self.assertEqual([3, 4], [entry["token_id"] for entry in scored])

    def test_one_nonfinite_candidate_is_dropped(self):
        scored, dropped = self._score_with_totals([math.nan, 0.25])
        self.assertEqual([3], [entry["token_id"] for entry in scored])
        self.assertEqual([4], [entry["token_id"] for entry in dropped])

    def test_all_nonfinite_candidates_are_a_hard_failure(self):
        with self.assertRaisesRegex(
                v21.SuffixV211FatalError, "all_candidate_scores_nonfinite"):
            self._score_with_totals([math.nan, math.inf])

    def test_nonfinite_global_formal_loss_rolls_back_entry_and_is_fatal(self):
        model, embed_layer, tokens, entry, mask, targets = _fixture()
        with mock.patch.object(
                v21,
                "_vocab_metric_with_anchors",
                return_value=torch.tensor(math.nan),
        ):
            final_embedding, result = v21.run_suffix_reoptimization_v2_1_1(
                model,
                embed_layer,
                entry,
                tokens,
                targets,
                mask,
                0,
                3,
                _Tokenizer(),
                _config(),
                eval_start_pos=1,
            )
        self.assertTrue(torch.equal(entry, final_embedding))
        self.assertEqual(tokens, result["final_tokens"])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["rollback"])
        self.assertTrue(result["fatal_failure"])
        self.assertEqual("nonfinite_global_loss", result["rollback_reason"])
        self.assertEqual("global_optimization", result["failed_stage"])

    def test_explicitly_invoked_but_disabled_is_a_hard_failure(self):
        result = _run_fixture(
            config=v21.SuffixReoptimizationV211Config(),
            total_input_ids=_GroundTruthBomb(),
        )[-1]
        self.assertTrue(result["rollback"])
        self.assertTrue(result["fatal_failure"])
        self.assertEqual("suffix_v2_1_1_disabled", result["rollback_reason"])

    def test_non_qwen_causal_lm_is_rejected(self):
        model, embed_layer, tokens, entry, mask, targets = _fixture()
        model.config.model_type = "llama"
        _, result = v21.run_suffix_reoptimization_v2_1_1(
            model, embed_layer, entry, tokens, targets, mask, 0, 3,
            _Tokenizer(), _config(), eval_start_pos=1,
        )
        self.assertTrue(result["rollback"])
        self.assertEqual("unsupported_model_family", result["rollback_reason"])

    def test_disabled_diagnostics_perform_zero_ground_truth_reads(self):
        result = _run_fixture(
            config=_config(accuracy_diagnostics_enabled=False),
            total_input_ids=_GroundTruthBomb(),
        )[-1]
        self.assertTrue(result["accepted"])
        self.assertEqual({}, result["diagnostics"])
        self.assertFalse(result["diagnostics_failed"])
        self.assertIsNone(result["pre_acc"])
        self.assertIsNone(result["post_acc"])

    def test_diagnostics_exception_cannot_change_formal_result(self):
        result = _run_fixture(
            config=_config(accuracy_diagnostics_enabled=True),
            total_input_ids=_GroundTruthBomb(),
        )[-1]
        self.assertTrue(result["accepted"])
        self.assertFalse(result["rollback"])
        self.assertTrue(result["diagnostics_failed"])
        self.assertEqual("completed_without_hard_failure", result["reason"])

    def test_valid_diagnostics_are_post_formal_only(self):
        target_ids = torch.tensor([[0, 3, 4, 1]], dtype=torch.long)
        result = _run_fixture(
            config=_config(accuracy_diagnostics_enabled=True),
            total_input_ids=target_ids,
        )[-1]
        self.assertTrue(result["accepted"])
        self.assertFalse(result["diagnostics_failed"])
        self.assertIsNotNone(result["pre_acc"])
        self.assertEqual(result["post_acc"], result["final_accuracy"])

    def test_sidecar_has_no_removed_state_machine_or_model_branch(self):
        source = inspect.getsource(v21).lower()
        for removed in ("classifier", "cusum", "replay", "cumulative"):
            self.assertNotIn(removed, source)


if __name__ == "__main__":
    unittest.main()
