import inspect
import math
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from suffix_optimization_methods.method_versions import (
    suffix_reoptimization_v2_2_1 as v221,
)


class _Tokenizer:
    all_special_ids = [0]

    def decode(self, token_ids):
        values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
        return " ".join("t{}".format(int(value)) for value in values)


class _Block(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 0.1 * hidden.cumsum(dim=1)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(12, 4)
        self.layers = torch.nn.ModuleList([_Block()])
        self.lm_head = torch.nn.Linear(4, 12, bias=False)
        with torch.no_grad():
            values = torch.arange(48, dtype=torch.float32).reshape(12, 4)
            self.embed_tokens.weight.copy_(values / 20.0 - 1.0)
            self.lm_head.weight.copy_(values / 40.0 - 0.5)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        del attention_mask
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return types.SimpleNamespace(logits=self.lm_head(hidden))


def _register_layer_hooks(model, layer_id, callback, up_to=False):
    del up_to
    handle = model.layers[int(layer_id)].register_forward_hook(callback)
    return [handle]


def _embedding_top_indices(embed, embed_layer, top_k, invert_method):
    del invert_method
    scores = F.cosine_similarity(
        embed.detach().float().unsqueeze(0),
        embed_layer.weight.detach().float(),
        dim=-1,
    )
    return torch.topk(scores, min(int(top_k), int(scores.numel()))).indices


def _select_candidate(top_indices, tokenizer, filter_nonascii=True):
    del filter_nonascii
    values = [int(value) for value in top_indices.detach().cpu().tolist()]
    for value in values:
        if value not in tokenizer.all_special_ids:
            return value, values
    return values[0], values


def _get_perplexity(input_ids, model, layer_id, top_k):
    del input_ids, model, layer_id
    values = torch.arange(3, 3 + int(top_k), dtype=torch.long)
    return torch.ones(len(values)), values


def _forward_tokens(model, input_ids, attention_mask, layer_id):
    del attention_mask
    values = torch.as_tensor(input_ids, dtype=torch.long)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    collected = []

    def hook(module, inputs, output):
        del module, inputs
        collected.append(output)

    handle = model.layers[int(layer_id)].register_forward_hook(hook)
    try:
        model(input_ids=values)
    finally:
        handle.remove()
    return collected[0]


def _fixture():
    model = _Model()
    embed_layer = model.embed_tokens
    tokens = [0, 3, 4, 5]
    entry = embed_layer(torch.tensor([tokens])).detach().clone()
    mask = torch.ones((1, len(tokens)), dtype=torch.long)
    target = _forward_tokens(model, tokens, mask, 0).detach().clone()
    return model, embed_layer, entry, mask, target


def _config(**overrides):
    values = {
        "enabled": True,
        "max_attempts": 2,
        "max_attempts_per_position": 1,
        "steps": 1,
        "range_top_k": 2,
    }
    values.update(overrides)
    return v221.SuffixReoptimizationV221Config(**values)


def _run(model, embed_layer, entry, mask, target, config, fixed_prefix=None):
    return v221.run_suffix_reoptimization_v2_2_1(
        model=model,
        embed_layer=embed_layer,
        optimized_embedding=entry,
        target_hidden_state=target,
        attention_mask=mask,
        layer_id=0,
        register_layer_hooks=_register_layer_hooks,
        tokenizer=_Tokenizer(),
        config=config,
        fixed_prefix_tokens=fixed_prefix or [],
        eval_start_pos=1 if fixed_prefix else 0,
        top_k_ppl=2,
        top_k_cos=2,
        embedding_top_indices=_embedding_top_indices,
        select_candidate_from_top_indices=_select_candidate,
        get_perplexity=_get_perplexity,
        forward_and_get_last_hidden_state=_forward_tokens,
    )


class ConfigAndBoundaryTests(unittest.TestCase):
    def test_defaults_match_v221_contract(self):
        config = v221.SuffixReoptimizationV221Config()
        self.assertEqual((2, 1, 50), (
            config.max_attempts,
            config.max_attempts_per_position,
            config.steps,
        ))
        self.assertEqual((0.03, 0.90, 0.20, 0.005, 0.001), (
            config.lr,
            config.hidden_weight_decay,
            config.hidden_weight_floor,
            config.prox_weight,
            config.range_weight,
        ))
        self.assertEqual("always", config.trigger_mode)
        self.assertEqual("hidden_loss", config.accept_mode)

    def test_oracle_acceptance_and_noncanonical_modes_are_rejected(self):
        with self.assertRaises(ValueError):
            v221.SuffixReoptimizationV221Config(accept_mode="oracle_accuracy")
        with self.assertRaises(ValueError):
            v221.SuffixReoptimizationV221Config(hidden_weight_mode="uniform")

    def test_sidecar_signature_has_no_ground_truth_token_argument(self):
        names = set(inspect.signature(
            v221.run_suffix_reoptimization_v2_2_1
        ).parameters)
        self.assertNotIn("total_input_ids", names)
        self.assertNotIn("target_token_ids", names)
        self.assertNotIn("oracle_accuracy", inspect.getsource(v221))

    def test_front_decay_and_anchored_prefix(self):
        config = _config()
        weights = v221._build_suffix_hidden_weights(
            4, config, torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.all(weights[0, :-1] >= weights[0, 1:]))
        model, embed_layer, entry, _, _ = _fixture()
        del model
        current = entry.clone()
        current[:, 0, :] += 0.75
        current[:, 1, :] += 0.75
        anchored = v221._build_anchored_base_embedding(
            current,
            [0, 7, 4, 5],
            2,
            1,
            [0],
            embed_layer,
        )
        self.assertTrue(torch.equal(anchored[:, 0], embed_layer.weight[0]))
        self.assertTrue(torch.equal(anchored[:, 1], embed_layer.weight[7]))
        self.assertTrue(torch.equal(anchored[:, 2], current[:, 2]))


class FlowAndBudgetTests(unittest.TestCase):
    def test_hidden_loss_gate_accepts_without_accuracy_or_gt(self):
        model, embed_layer, entry, mask, target = _fixture()
        fake_summary = {"stopped_reason": "mock"}
        with mock.patch.object(
                v221,
                "_optimize_suffix",
                return_value=(entry.clone(), 1.0, 0.5, fake_summary)):
            final_embedding, result = _run(
                model, embed_layer, entry, mask, target, _config()
            )
        self.assertTrue(result["accepted"])
        self.assertEqual(2, result["attempt_count"])
        self.assertEqual(2, result["accepted_round_count"])
        self.assertEqual([], result.get("pre_acc", []))
        self.assertTrue(result["formal_gt_blind"])
        self.assertFalse(result["gt_accessed"])
        self.assertTrue(torch.equal(final_embedding, entry))

    def test_rejected_trial_rolls_back_and_consumes_attempt(self):
        model, embed_layer, entry, mask, target = _fixture()
        trial = entry.clone()
        trial[:, 1:, :] += 1.0
        with mock.patch.object(
                v221,
                "_optimize_suffix",
                return_value=(trial, 1.0, 1.0, {})):
            final_embedding, result = _run(
                model,
                embed_layer,
                entry,
                mask,
                target,
                _config(max_attempts=1),
            )
        self.assertFalse(result["accepted"])
        self.assertEqual(1, result["attempt_count"])
        self.assertGreaterEqual(result["budget_exhausted_count"], 1)
        self.assertTrue(torch.equal(final_embedding, entry))
        self.assertTrue(all(
            not event["accepted"] for event in result["events"]
        ))

    def test_fixed_prefix_is_not_changed_and_threshold_can_skip(self):
        model, embed_layer, entry, mask, target = _fixture()
        with mock.patch.object(
                v221,
                "_position_similarity",
                return_value=1.0,
        ):
            final_embedding, result = _run(
                model,
                embed_layer,
                entry,
                mask,
                target,
                _config(trigger_mode="threshold", trigger_threshold=0.0),
                fixed_prefix=[0],
            )
        self.assertTrue(torch.equal(final_embedding[:, 0], entry[:, 0]))
        self.assertEqual(0, result["attempt_count"])
        self.assertTrue(all(not event["triggered"] for event in result["events"]))

    def test_config_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError):
            v221.SuffixReoptimizationV221Config(lr=math.inf)


if __name__ == "__main__":
    unittest.main()
