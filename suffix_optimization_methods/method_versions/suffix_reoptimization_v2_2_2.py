"""Target-token-blind Stage-1 + anchored R + transactional checkpoint repair.

The sidecar keeps the Original DEML candidate policy and adds bounded
five-token checkpoints to the v2.2.1 R loop. Private token ids are absent from
the public runner signature.  Accuracy, when needed by an experiment, is
attached by the caller after this sidecar has finished.
"""

from dataclasses import dataclass
import math
import numbers
import hashlib
import json
import time
import random
from collections import Counter
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F


METHOD_NAME = "suffix_reoptimization_v2.2.2"
VERSION = "v2.2.2"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192

__all__ = [
    "SuffixReoptimizationV222Config",
    "run_suffix_reoptimization_v2_2_2",
]


@dataclass
class SuffixReoptimizationV222Config:
    enabled: bool = False
    log_enabled: bool = True
    max_attempts: int = 2
    max_attempts_per_position: int = 1
    steps: int = 50
    lr: float = 0.03
    trigger_mode: str = "always"
    trigger_threshold: float = 0.0
    hidden_weight_mode: str = "front_decay"
    hidden_weight_decay: float = 0.90
    hidden_weight_floor: float = 0.20
    prox_weight: float = 0.005
    range_weight: float = 0.001
    range_top_k: int = 10
    accept_mode: str = "hidden_loss"
    filter_nonascii: bool = True

    checkpoint_enabled: bool = False
    checkpoint_size: int = 5
    checkpoint_stride: int = 5
    checkpoint_trigger_metric: str = "pointwise_logmeanexp"
    checkpoint_deviation_tau: float = 0.05
    checkpoint_trigger_deviation: float = 0.05
    checkpoint_diagnostic_tolerance: float = 0.02
    checkpoint_candidate_min_cosine: float = 0.90
    checkpoint_candidate_threshold_source: str = "user_fixed_2026_09_16"
    checkpoint_forward_mode: str = "full_prefix"
    checkpoint_candidate_failure_policy: str = "abort_checkpoint_keep_state"
    checkpoint_tail_policy: str = "skip_incomplete"
    checkpoint_max_repairs: int = 1
    checkpoint_recursive: bool = False
    checkpoint_numeric_norm_epsilon: float = 1e-8
    checkpoint_score_dtype: str = "float32"
    checkpoint_schema_version: int = 3
    checkpoint_downstream_policy: str = "sequential_rerank_to_checkpoint_end"

    def __post_init__(self):
        validate_checkpoint_config(self)
        for name in ("enabled", "log_enabled", "filter_nonascii"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_2_2_{} must be boolean".format(name))
        for name in (
            "max_attempts",
            "max_attempts_per_position",
            "steps",
            "range_top_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_2_2_{} must be an integer".format(name))
            setattr(self, name, int(value))
        for name in (
            "lr",
            "trigger_threshold",
            "hidden_weight_decay",
            "hidden_weight_floor",
            "prox_weight",
            "range_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError("suffix_v2_2_2_{} must be numeric".format(name))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("suffix_v2_2_2_{} must be finite".format(name))
            setattr(self, name, value)
        if self.max_attempts < 0:
            raise ValueError("suffix_v2_2_2_max_attempts must be non-negative")
        if self.max_attempts_per_position <= 0:
            raise ValueError(
                "suffix_v2_2_2_max_attempts_per_position must be positive"
            )
        if self.steps <= 0:
            raise ValueError("suffix_v2_2_2_steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("suffix_v2_2_2_lr must be positive")
        if not 0.0 <= self.hidden_weight_decay <= 1.0:
            raise ValueError(
                "suffix_v2_2_2_hidden_weight_decay must be in [0, 1]"
            )
        if self.hidden_weight_floor < 0.0:
            raise ValueError(
                "suffix_v2_2_2_hidden_weight_floor must be non-negative"
            )
        if self.prox_weight < 0.0 or self.range_weight < 0.0:
            raise ValueError(
                "suffix_v2_2_2 regularization weights must be non-negative"
            )
        if self.range_top_k <= 0:
            raise ValueError("suffix_v2_2_2_range_top_k must be positive")
        if self.trigger_mode not in {"always", "threshold"}:
            raise ValueError(
                "suffix_v2_2_2_trigger_mode must be always or threshold"
            )
        if self.hidden_weight_mode != "front_decay":
            raise ValueError(
                "suffix_v2_2_2_hidden_weight_mode must be front_decay"
            )
        if self.accept_mode != "hidden_loss":
            raise ValueError(
                "suffix_v2_2_2_accept_mode must be hidden_loss"
            )


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _decode(tokenizer, token_ids, eval_start_pos):
    return tokenizer.decode(
        [int(value) for value in token_ids[int(eval_start_pos):]]
    )


def _forward_embedding_hidden(
        model, input_embed, attention_mask, layer_id, register_layer_hooks):
    hidden_state_list = []

    def forward_hook(module, inputs, output):
        del module, inputs
        if isinstance(output, tuple):
            hidden_state_list.append(output[0])
        else:
            hidden_state_list.append(output)

    handles = register_layer_hooks(model, layer_id, forward_hook, up_to=False)
    try:
        model(inputs_embeds=input_embed, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    if not hidden_state_list:
        raise ValueError("no hidden states collected for layer {}".format(layer_id))
    return hidden_state_list[0]


def _candidate_token_ids(
        embed, embed_layer, top_k_cos, invert_method, tokenizer,
        filter_nonascii, embedding_top_indices,
    select_candidate_from_top_indices):
    if int(top_k_cos) == 0:
        return 0, [0]
    top_indices = embedding_top_indices(
        embed,
        embed_layer,
        int(top_k_cos),
        invert_method,
    )
    selected_token, top_ids = select_candidate_from_top_indices(
        top_indices,
        tokenizer,
        filter_nonascii,
    )
    return int(selected_token), [int(item) for item in top_ids]


def _rerank_positions(
        input_embed, current_tokens, rerank_start, fixed_prefix_tokens,
        tokenizer, model, embed_layer, target_hidden_state, layer_id,
        invert_method, filter_nonascii, add_perplexity, top_k_ppl, top_k_cos,
        eval_start_pos, embedding_top_indices,
        select_candidate_from_top_indices, get_perplexity,
        forward_and_get_last_hidden_state, rerank_end=None, forward_stats=None):
    """Copy Original DEML candidate order without consulting target ids."""
    sequence_length = int(input_embed.shape[1])
    if current_tokens is None:
        ret_list = [0 for _ in range(sequence_length)]
    else:
        ret_list = [int(item) for item in current_tokens]
        if len(ret_list) != sequence_length:
            raise ValueError("current token length does not match embedding length")

    fixed_prefix_tokens = [int(value) for value in (fixed_prefix_tokens or [])]
    if len(fixed_prefix_tokens) != int(eval_start_pos):
        raise ValueError("fixed prefix length does not match eval_start_pos")
    for position in range(min(int(eval_start_pos), sequence_length)):
        ret_list[position] = fixed_prefix_tokens[position]

    rerank_start = max(int(eval_start_pos), int(rerank_start))
    rerank_end = sequence_length if rerank_end is None else int(rerank_end)
    new_input_embed_squeeze = input_embed.squeeze(0)
    ret_top_k = {}
    for position in range(rerank_start, rerank_end):
        selected_token, top_list = _candidate_token_ids(
            new_input_embed_squeeze[position],
            embed_layer,
            top_k_cos,
            invert_method,
            tokenizer,
            filter_nonascii,
            embedding_top_indices,
            select_candidate_from_top_indices,
        )
        if not top_list:
            top_list = [ret_list[position]]
        ret_top_k[position] = list(top_list)
        ret_list[position] = int(selected_token)

    diagnostics = []
    for position in range(rerank_start, rerank_end):
        top_list = list(ret_top_k.get(position, [ret_list[position]]))
        if position > 0 and add_perplexity:
            if forward_stats is not None:
                forward_stats["downstream_forward_calls"] += 1
                forward_stats["forward_token_count"] += position
            _, topk_ids = get_perplexity(
                list(ret_list[:position]),
                model,
                layer_id=layer_id,
                top_k=top_k_ppl,
            )
            # Original DEML appends PPL candidates and does not deduplicate.
            top_list.extend(int(item) for item in topk_ids.tolist())

        replaced_sequences = []
        for token_id in top_list:
            replaced = list(ret_list)
            replaced[position] = int(token_id)
            replaced_sequences.append(replaced)
        if forward_stats is not None:
            forward_stats["downstream_forward_calls"] += 1
            forward_stats["forward_token_count"] += len(replaced_sequences) * sequence_length
        hidden_states = forward_and_get_last_hidden_state(
            model,
            replaced_sequences,
            None,
            layer_id=layer_id,
        )
        target_hidden = target_hidden_state.to(hidden_states.device)
        candidate_states = hidden_states[:, position, :].float()
        target_state = target_hidden[:, position, :].float()
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        cosine = F.cosine_similarity(candidate_states, target_state, dim=-1)
        best_index = int(torch.argmax(cosine).detach().cpu().item())
        ret_list[position] = int(top_list[best_index])
        diagnostics.append({
            "prefix_fingerprint": prefix_fingerprint(ret_list[:position]),
            "position": int(position),
            "candidate_token_ids": [int(item) for item in top_list],
            "candidate_hidden_cosine": [
                float(item) for item in cosine.detach().cpu().tolist()
            ],
            "selected_token_id": int(ret_list[position]),
        })
    return ret_list, _decode(tokenizer, ret_list, eval_start_pos), diagnostics


def _build_suffix_hidden_weights(length, config, device, dtype):
    length = max(0, int(length))
    if length == 0:
        return torch.empty((1, 0), device=device, dtype=dtype)
    positions = torch.arange(length, device=device, dtype=dtype)
    weights = torch.pow(
        torch.tensor(config.hidden_weight_decay, device=device, dtype=dtype),
        positions,
    )
    weights = torch.maximum(
        weights,
        torch.tensor(config.hidden_weight_floor, device=device, dtype=dtype),
    )
    return (weights / weights.mean().clamp_min(1e-12)).view(1, length)


def _embedding_range_bound(embed_layer, device, dtype, top_k):
    weight = embed_layer.weight.detach()
    vocab_size = int(weight.shape[0])
    keep_k = max(1, min(int(top_k), vocab_size))
    best = None
    with torch.no_grad():
        for start in range(0, vocab_size, EMBEDDING_SEARCH_CHUNK_SIZE):
            end = min(start + EMBEDDING_SEARCH_CHUNK_SIZE, vocab_size)
            chunk = weight[start:end].abs().to(device=device, dtype=torch.float32)
            combined = chunk if best is None else torch.cat((best, chunk), dim=0)
            best = torch.topk(
                combined,
                min(keep_k, int(combined.shape[0])),
                dim=0,
            ).values
    return best[-1].to(dtype=dtype).view(1, 1, -1)


def _build_anchored_base_embedding(
        current_embedding, current_tokens, suffix_start, eval_start_pos,
        fixed_prefix_tokens, embed_layer):
    sequence_length = int(current_embedding.shape[1])
    if len(current_tokens) != sequence_length:
        raise ValueError("current token length must equal embedding sequence length")
    suffix_start = int(suffix_start)
    eval_start_pos = int(eval_start_pos)
    if suffix_start < eval_start_pos or suffix_start > sequence_length:
        raise ValueError("suffix_start is outside the sequence")
    fixed_prefix_tokens = [int(value) for value in (fixed_prefix_tokens or [])]
    if len(fixed_prefix_tokens) != eval_start_pos:
        raise ValueError("fixed prefix length does not match eval_start_pos")

    fixed_prefix = current_embedding[:, :eval_start_pos, :].detach()
    if eval_start_pos:
        token_device = embed_layer.weight.device
        prefix_ids = torch.tensor(
            fixed_prefix_tokens,
            dtype=torch.long,
            device=token_device,
        ).unsqueeze(0)
        with torch.no_grad():
            fixed_prefix = embed_layer(prefix_ids).detach()
        fixed_prefix = fixed_prefix.to(
            device=current_embedding.device,
            dtype=current_embedding.dtype,
        )

    if suffix_start > eval_start_pos:
        token_device = embed_layer.weight.device
        token_ids = torch.tensor(
            current_tokens[eval_start_pos:suffix_start],
            dtype=torch.long,
            device=token_device,
        ).unsqueeze(0)
        with torch.no_grad():
            anchored_prefix = embed_layer(token_ids).detach()
        anchored_prefix = anchored_prefix.to(
            device=current_embedding.device,
            dtype=current_embedding.dtype,
        )
    else:
        anchored_prefix = current_embedding[:, eval_start_pos:suffix_start, :].detach()
    continuous_suffix = current_embedding[:, suffix_start:, :].detach()
    anchored = torch.cat((fixed_prefix, anchored_prefix, continuous_suffix), dim=1)
    if int(anchored.shape[1]) != sequence_length:
        raise ValueError("anchored embedding sequence length changed")
    return anchored.detach()


def _hidden_cosine_loss(
        hidden_state, target_hidden_state, suffix_start, config):
    target_hidden = target_hidden_state.to(hidden_state.device)
    current = hidden_state[:, int(suffix_start):, :].float()
    target = target_hidden[:, int(suffix_start):, :].float()
    if current.shape != target.shape:
        raise ValueError("hidden state shape does not match target hidden state")
    if current.shape[1] == 0:
        return torch.zeros((), device=current.device, dtype=torch.float32)
    weights = _build_suffix_hidden_weights(
        current.shape[1],
        config,
        current.device,
        current.dtype,
    )
    cosine = F.cosine_similarity(current, target, dim=-1)
    weighted_similarity = (cosine * weights).sum() / weights.sum().clamp_min(1e-12)
    return 1.0 - weighted_similarity


def _hidden_loss_value(
        model, embedding, target_hidden_state, attention_mask, suffix_start,
        layer_id, register_layer_hooks, config):
    with torch.no_grad():
        hidden = _forward_embedding_hidden(
            model,
            embedding,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        loss = _hidden_cosine_loss(
            hidden,
            target_hidden_state,
            suffix_start,
            config,
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise ValueError("nonfinite hidden loss")
        return _as_float(loss)


def _position_similarity(
        model, embedding, target_hidden_state, attention_mask, position,
        layer_id, register_layer_hooks):
    with torch.no_grad():
        hidden = _forward_embedding_hidden(
            model,
            embedding,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target = target_hidden_state.to(hidden.device)
        score = F.cosine_similarity(
            hidden[:, int(position), :].float(),
            target[:, int(position), :].float(),
            dim=-1,
        )
        if not bool(torch.isfinite(score).all().detach().cpu()):
            raise ValueError("nonfinite trigger similarity")
        return _as_float(score.mean())


def _merge_suffix(base_embedding, suffix_start, suffix_embedding):
    prefix = base_embedding[:, :int(suffix_start), :].detach()
    return torch.cat(
        (prefix, suffix_embedding.to(dtype=base_embedding.dtype)),
        dim=1,
    )


def _optimize_suffix(
        model, current_embedding, current_tokens, fixed_prefix_tokens,
        target_hidden_state, attention_mask, suffix_start, eval_start_pos,
        layer_id, register_layer_hooks, embed_layer, config):
    anchored_base = _build_anchored_base_embedding(
        current_embedding,
        current_tokens,
        suffix_start,
        eval_start_pos,
        fixed_prefix_tokens,
        embed_layer,
    )
    original_suffix = anchored_base[:, int(suffix_start):, :].detach().clone()
    suffix_param = original_suffix.to(torch.float32).detach().clone()
    suffix_param.requires_grad_(True)
    optimizer = torch.optim.Adam([suffix_param], lr=float(config.lr))
    range_bound = _embedding_range_bound(
        embed_layer,
        suffix_param.device,
        suffix_param.dtype,
        config.range_top_k,
    ) if config.range_weight > 0.0 else None
    history = []
    stopped_reason = "completed"
    pre_loss = _hidden_loss_value(
        model,
        anchored_base,
        target_hidden_state,
        attention_mask,
        suffix_start,
        layer_id,
        register_layer_hooks,
        config,
    )
    completed_steps = 0
    for step in range(int(config.steps)):
        optimizer.zero_grad()
        trial_embedding = _merge_suffix(
            anchored_base,
            suffix_start,
            suffix_param,
        )
        hidden = _forward_embedding_hidden(
            model,
            trial_embedding,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        hidden_loss = _hidden_cosine_loss(
            hidden,
            target_hidden_state,
            suffix_start,
            config,
        )
        prox_loss = F.mse_loss(suffix_param, original_suffix.to(torch.float32))
        range_loss = torch.zeros((), device=suffix_param.device, dtype=torch.float32)
        if range_bound is not None:
            range_loss = F.relu(
                torch.abs(suffix_param) - range_bound.to(suffix_param.device)
            ).mean()
        total_loss = (
            hidden_loss
            + float(config.prox_weight) * prox_loss
            + float(config.range_weight) * range_loss
        )
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            stopped_reason = "nonfinite_loss"
            break
        total_loss.backward()
        if suffix_param.grad is None or not bool(
                torch.isfinite(suffix_param.grad).all().detach().cpu()):
            stopped_reason = "nonfinite_gradient"
            break
        optimizer.step()
        completed_steps += 1
        history.append(_as_float(total_loss))

    candidate_embedding = _merge_suffix(
        anchored_base,
        suffix_start,
        suffix_param.detach(),
    ).detach()
    post_loss = _hidden_loss_value(
        model,
        candidate_embedding,
        target_hidden_state,
        attention_mask,
        suffix_start,
        layer_id,
        register_layer_hooks,
        config,
    )
    return candidate_embedding, pre_loss, post_loss, {
        "optimizer": "Adam",
        "steps": int(config.steps),
        "completed_steps": completed_steps,
        "lr": float(config.lr),
        "suffix_start": int(suffix_start),
        "suffix_length": int(original_suffix.shape[1]),
        "suffix_dtype": "float32",
        "front_decay": float(config.hidden_weight_decay),
        "front_decay_floor": float(config.hidden_weight_floor),
        "prox_weight": float(config.prox_weight),
        "range_weight": float(config.range_weight),
        "range_top_k": int(config.range_top_k),
        "hidden_loss_start": pre_loss,
        "hidden_loss_end": post_loss,
        "objective_loss_start": history[0] if history else None,
        "objective_loss_end": history[-1] if history else None,
        "objective_loss_min": min(history) if history else None,
        "stopped_reason": stopped_reason,
    }


def _changed_positions(before_tokens, after_tokens, start_position):
    return [
        int(position)
        for position in range(
            max(0, int(start_position)),
            min(len(before_tokens), len(after_tokens)),
        )
        if int(before_tokens[position]) != int(after_tokens[position])
    ]


def _disabled_result(config, embedding, tokenizer, fixed_prefix_tokens, eval_start_pos):
    return embedding.detach().clone(), {
        "name": METHOD_NAME,
        "method": METHOD_NAME,
        "version": VERSION,
        "enabled": bool(config.enabled),
        "skipped": True,
        "reason": "disabled" if not config.enabled else "max_attempts <= 0",
        "formal_gt_blind": True,
        "gt_accessed": False,
        "accept_mode": config.accept_mode,
        "trigger_mode": config.trigger_mode,
        "fixed_prefix_length": len(fixed_prefix_tokens or []),
        "eval_start_pos": int(eval_start_pos),
        "events": [],
    }


def run_suffix_reoptimization_v2_2_2(
        model, embed_layer, optimized_embedding, target_hidden_state,
        attention_mask, layer_id, register_layer_hooks, tokenizer, config,
        fixed_prefix_tokens=None, eval_start_pos=0, filter_nonascii=True,
        add_perplexity=True, top_k_ppl=10, top_k_cos=10,
        invert_method="cosine", embedding_top_indices=None,
        select_candidate_from_top_indices=None, get_perplexity=None,
        forward_and_get_last_hidden_state=None, log_file=None):
    """Run v2.2.2 without exposing target token ids to the method."""
    del log_file
    embedding = torch.as_tensor(optimized_embedding).detach().clone()
    fixed_prefix_tokens = [int(value) for value in (fixed_prefix_tokens or [])]
    validate_inputs(model, embedding, target_hidden_state, attention_mask,
                    fixed_prefix_tokens, eval_start_pos, tokenizer)
    if not config.enabled:
        return _disabled_result(
            config,
            embedding,
            tokenizer,
            fixed_prefix_tokens,
            eval_start_pos,
        )

    required_helpers = {
        "embedding_top_indices": embedding_top_indices,
        "select_candidate_from_top_indices": select_candidate_from_top_indices,
        "get_perplexity": get_perplexity,
        "forward_and_get_last_hidden_state": forward_and_get_last_hidden_state,
    }
    missing_helpers = [name for name, value in required_helpers.items() if value is None]
    if missing_helpers:
        raise ValueError(
            "missing {} helpers: {}".format(METHOD_NAME, ", ".join(missing_helpers))
        )

    sequence_length = int(embedding.shape[1])
    if len(fixed_prefix_tokens) != int(eval_start_pos):
        raise ValueError("fixed prefix length does not match eval_start_pos")
    if int(eval_start_pos) < 0 or int(eval_start_pos) > sequence_length:
        raise ValueError("eval_start_pos is outside the sequence")

    current_tokens, current_text, initial_rerank = _rerank_positions(
        embedding,
        None,
        eval_start_pos,
        fixed_prefix_tokens,
        tokenizer,
        model,
        embed_layer,
        target_hidden_state,
        layer_id,
        invert_method,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        eval_start_pos,
        embedding_top_indices,
        select_candidate_from_top_indices,
        get_perplexity,
        forward_and_get_last_hidden_state,
    )
    current_embedding = embedding.detach().clone()
    if sequence_length == int(eval_start_pos):
        return current_embedding, empty_result(current_tokens, config, eval_start_pos)
    initial_hidden_loss = _hidden_loss_value(
        model,
        current_embedding,
        target_hidden_state,
        attention_mask,
        int(eval_start_pos),
        layer_id,
        register_layer_hooks,
        config,
    )
    before_tokens = list(current_tokens)
    events = []
    attempts = 0
    accepted_count = 0
    rejected_count = 0
    triggered_count = 0
    budget_exhausted_count = 0
    per_position_attempts = {}

    cp_events = []
    candidate_tables = {row["position"]: dict(row, generation=0) for row in initial_rerank}
    generation = 0
    dirty_positions = set()
    refresh_count = 0
    refresh_seconds = 0.0

    def finish_position(position):
        nonlocal current_text
        if not config.checkpoint_enabled or (position - eval_start_pos + 1) % 5:
            return
        event = run_checkpoint(
            model, embed_layer, current_tokens, current_embedding,
            target_hidden_state, layer_id, register_layer_hooks, tokenizer,
            config, candidate_tables, position - 4, position, eval_start_pos,
            filter_nonascii,
            downstream_rerank=lambda trial, begin, end, stats: _rerank_positions(
                current_embedding[:, :end].detach().clone(), trial, begin,
                fixed_prefix_tokens, tokenizer, model, embed_layer,
                target_hidden_state[:, :end], layer_id, invert_method,
                filter_nonascii, add_perplexity, top_k_ppl, top_k_cos,
                eval_start_pos, embedding_top_indices,
                select_candidate_from_top_indices, get_perplexity,
                forward_and_get_last_hidden_state, rerank_end=end, forward_stats=stats))
        cp_events.append(event)
        if event["accepted"]:
            dirty_positions.update(range(position + 1, sequence_length))
            current_text = _decode(tokenizer, current_tokens, eval_start_pos)

    for position in range(int(eval_start_pos), sequence_length):
        if position in dirty_positions:
            refresh_start = time.perf_counter()
            current_tokens, current_text, refreshed = _rerank_positions(
                current_embedding, current_tokens, position, fixed_prefix_tokens,
                tokenizer, model, embed_layer, target_hidden_state, layer_id,
                invert_method, filter_nonascii, add_perplexity, top_k_ppl,
                top_k_cos, eval_start_pos, embedding_top_indices,
                select_candidate_from_top_indices, get_perplexity,
                forward_and_get_last_hidden_state, rerank_end=position + 1)
            generation += 1
            candidate_tables.update({r["position"]: dict(r, generation=generation) for r in refreshed})
            dirty_positions.remove(position)
            refresh_count += 1
            refresh_seconds += time.perf_counter() - refresh_start
        anchored_before = _build_anchored_base_embedding(
            current_embedding,
            current_tokens,
            position,
            eval_start_pos,
            fixed_prefix_tokens,
            embed_layer,
        )
        pre_loss = _hidden_loss_value(
            model,
            anchored_before,
            target_hidden_state,
            attention_mask,
            position,
            layer_id,
            register_layer_hooks,
            config,
        )
        score = _position_similarity(
            model,
            anchored_before,
            target_hidden_state,
            attention_mask,
            position,
            layer_id,
            register_layer_hooks,
        )
        triggered = (
            config.trigger_mode == "always"
            or score < float(config.trigger_threshold)
        )
        if not triggered:
            events.append({
                "position": int(position),
                "triggered": False,
                "attempted": False,
                "accepted": False,
                "s_i_pre": score,
                "pre_hidden_loss": pre_loss,
                "post_hidden_loss": None,
                "reason": "above_trigger_threshold",
            })
            finish_position(position)
            continue
        triggered_count += 1
        position_attempts = per_position_attempts.get(position, 0)
        if (
            attempts >= int(config.max_attempts)
            or position_attempts >= int(config.max_attempts_per_position)
        ):
            budget_exhausted_count += 1
            events.append({
                "position": int(position),
                "triggered": True,
                "attempted": False,
                "accepted": False,
                "s_i_pre": score,
                "pre_hidden_loss": pre_loss,
                "post_hidden_loss": None,
                "reason": "attempt_budget_exhausted",
            })
            finish_position(position)
            continue

        attempts += 1
        per_position_attempts[position] = position_attempts + 1
        try:
            candidate_embedding, optimized_pre_loss, optimized_post_loss, summary = (
                _optimize_suffix(
                    model,
                    current_embedding,
                    current_tokens,
                    fixed_prefix_tokens,
                    target_hidden_state,
                    attention_mask,
                    position,
                    eval_start_pos,
                    layer_id,
                    register_layer_hooks,
                    embed_layer,
                    config,
                )
            )
            candidate_tokens, candidate_text, candidate_rerank = _rerank_positions(
                candidate_embedding,
                current_tokens,
                position,
                fixed_prefix_tokens,
                tokenizer,
                model,
                embed_layer,
                target_hidden_state,
                layer_id,
                invert_method,
                filter_nonascii,
                add_perplexity,
                top_k_ppl,
                top_k_cos,
                eval_start_pos,
                embedding_top_indices,
                select_candidate_from_top_indices,
                get_perplexity,
                forward_and_get_last_hidden_state,
            )
            accepted = bool(
                math.isfinite(float(optimized_pre_loss))
                and math.isfinite(float(optimized_post_loss))
                and float(optimized_post_loss) < float(optimized_pre_loss)
            )
            reason = "hidden_loss_decreased" if accepted else "hidden_loss_not_decreased"
            changed_positions = _changed_positions(
                current_tokens,
                candidate_tokens,
                position,
            )
            event = {
                "position": int(position),
                "triggered": True,
                "attempted": True,
                "accepted": accepted,
                "s_i_pre": score,
                "pre_hidden_loss": float(optimized_pre_loss),
                "post_hidden_loss": float(optimized_post_loss),
                "changed_positions": changed_positions,
                "reason": reason,
                "anchor": summary,
                "candidate_rerank": candidate_rerank,
            }
            events.append(event)
            if accepted:
                current_embedding = candidate_embedding.detach().clone()
                current_tokens = [int(item) for item in candidate_tokens]
                current_text = candidate_text
                generation += 1
                candidate_tables.update({r["position"]: dict(r, generation=generation) for r in candidate_rerank})
                dirty_positions.difference_update(range(position, sequence_length))
                accepted_count += 1
            else:
                rejected_count += 1
        except Exception as error:
            raise_if_fatal(error)
            rejected_count += 1
            events.append({
                "position": int(position),
                "triggered": True,
                "attempted": True,
                "accepted": False,
                "s_i_pre": score,
                "pre_hidden_loss": pre_loss,
                "post_hidden_loss": None,
                "changed_positions": [],
                "reason": "trial_failed:{}".format(type(error).__name__),
            })

        finish_position(position)

    final_hidden_loss = _hidden_loss_value(
        model,
        current_embedding,
        target_hidden_state,
        attention_mask,
        int(eval_start_pos),
        layer_id,
        register_layer_hooks,
        config,
    )
    return current_embedding.detach().clone(), {
        "name": METHOD_NAME,
        "method": METHOD_NAME,
        "version": VERSION,
        "enabled": True,
        "skipped": False,
        "formal_gt_blind": True,
        "gt_accessed": False,
        "accept_mode": config.accept_mode,
        "trigger_mode": config.trigger_mode,
        "max_attempts": int(config.max_attempts),
        "max_attempts_per_position": int(config.max_attempts_per_position),
        "steps": int(config.steps),
        "fixed_prefix_length": len(fixed_prefix_tokens),
        "eval_start_pos": int(eval_start_pos),
        "pre_tokens": before_tokens,
        "pre_text": _decode(tokenizer, before_tokens, eval_start_pos),
        "final_tokens": [int(item) for item in current_tokens],
        "final_text": _decode(tokenizer, current_tokens, eval_start_pos),
        "pre_hidden_loss": initial_hidden_loss,
        "final_hidden_loss": final_hidden_loss,
        "triggered": bool(triggered_count),
        "trigger_count": int(triggered_count),
        "attempt_count": int(attempts),
        "accepted_round_count": int(accepted_count),
        "rejected_round_count": int(rejected_count),
        "budget_exhausted_count": int(budget_exhausted_count),
        "accepted": bool(accepted_count),
        "reason": (
            "accepted {} suffix trial(s)".format(accepted_count)
            if accepted_count
            else "no suffix trial accepted"
            if attempts
            else "no suffix trial attempted"
        ),
        "checkpoint": {
            "enabled": config.checkpoint_enabled,
            "events": cp_events,
            "complete_segment_count": (sequence_length - eval_start_pos) // 5,
            "tail_skipped_token_count": (sequence_length - eval_start_pos) % 5,
            "reason_counts": dict(Counter(e["reason"] for e in cp_events)),
            "trigger_count": sum(e["triggered"] is True for e in cp_events),
            "localized_count": sum(e["selected_position"] is not None for e in cp_events),
            "attempt_count": sum(e["repair_attempt_count"] for e in cp_events),
            "accepted_count": sum(e["accepted"] for e in cp_events),
            "future_refresh_count": refresh_count,
            "future_refresh_seconds": refresh_seconds,
        },
        "initial_candidate_rerank": initial_rerank,
        "events": events,
    }


SuffixReoptimizationConfig = SuffixReoptimizationV222Config
run_suffix_reoptimization = run_suffix_reoptimization_v2_2_2


def validate_checkpoint_config(config):
    """Frozen first-release CP contract; no silent defaults at config boundary."""
    expected = {
        "checkpoint_size": 5, "checkpoint_stride": 5,
        "checkpoint_trigger_metric": "pointwise_logmeanexp",
        "checkpoint_deviation_tau": 0.05,
        "checkpoint_trigger_deviation": 0.05,
        "checkpoint_diagnostic_tolerance": 0.02,
        "checkpoint_candidate_min_cosine": 0.90,
        "checkpoint_candidate_threshold_source": "user_fixed_2026_09_16",
        "checkpoint_forward_mode": "full_prefix",
        "checkpoint_candidate_failure_policy": "abort_checkpoint_keep_state",
        "checkpoint_tail_policy": "skip_incomplete",
        "checkpoint_max_repairs": 1, "checkpoint_recursive": False,
        "checkpoint_numeric_norm_epsilon": 1e-8,
        "checkpoint_score_dtype": "float32", "checkpoint_schema_version": 3,
        "checkpoint_downstream_policy": "sequential_rerank_to_checkpoint_end",
    }
    if type(config.checkpoint_enabled) is not bool:
        raise TypeError("checkpoint_enabled must be boolean")
    for key, expected_value in expected.items():
        actual = getattr(config, key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError("{} must be {!r}".format(key, expected_value))


def config_from_mapping(values, require_explicit=True):
    from dataclasses import fields
    parsed = {}
    if "suffix_v2_2_2_checkpoint_trigger_cosine" in values:
        raise ValueError("obsolete checkpoint_trigger_cosine; migrate to explicit schema 3 deviation settings")
    for field in fields(SuffixReoptimizationV222Config):
        key = {"enabled": "suffix_reoptimization_v2_2_2",
               "log_enabled": "suffix_reoptimization_v2_2_2_log"}.get(
                   field.name, "suffix_v2_2_2_" + field.name)
        if key in values:
            parsed[field.name] = values[key]
        elif require_explicit:
            raise ValueError("missing explicit configuration: " + key)
    return SuffixReoptimizationV222Config(**parsed)


def prefix_fingerprint(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


class CheckpointContractError(ValueError):
    """A broken adapter contract cannot be treated as a recoverable candidate."""


def raise_if_fatal(error):
    text = str(error).lower()
    if isinstance(error, (CheckpointContractError, MemoryError, torch.cuda.OutOfMemoryError)) or any(
        word in text for word in ("out of memory", "device-side assert", "illegal memory", "cuda error")
    ):
        raise error


def validate_inputs(model, embedding, target, mask, prefix, start, tokenizer):
    if embedding.ndim != 3 or embedding.shape[0] != 1 or target.shape != embedding.shape:
        raise ValueError("expected matching [1,T,D] embeddings and target")
    if not torch.isfinite(target).all():
        raise ValueError("nonfinite target observation")
    if mask is not None and (tuple(mask.shape) != tuple(embedding.shape[:2]) or not bool((mask == 1).all())):
        raise ValueError("only unpadded, fully valid single sequences are supported")
    if len(prefix) != start or not 0 <= start <= embedding.shape[1] or start > 1:
        raise ValueError("invalid public prefix length")
    if prefix and prefix != [getattr(tokenizer, "bos_token_id", None)]:
        raise ValueError("only the tokenizer's public BOS may be fixed")
    if model.training:
        raise ValueError("v2.2.2 requires model.eval()")


def forward_discrete(model, token_ids, layer_id, register_layer_hooks):
    """Same full-prefix, batch-one, cache-free forward for old and trial scores."""
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=next(model.parameters()).device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise CheckpointContractError("checkpoint forward expects batch one")
    collected = []
    def hook(module, inputs, output):
        collected.append(output[0] if isinstance(output, tuple) else output)
    handles = register_layer_hooks(model, layer_id, hook, up_to=False)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if len(collected) != 1 or collected[0].ndim != 3 or collected[0].shape[:2] != ids.shape:
        raise CheckpointContractError("exact block output shape mismatch")
    return collected[0].detach()


def segment_cosine(current, target, epsilon=1e-8):
    current = current.float().sum(dim=-2).flatten()
    target = target.float().sum(dim=-2).flatten().to(current.device)
    if not bool(torch.isfinite(current).all() and torch.isfinite(target).all()):
        return None
    nc, nt = current.norm(), target.norm()
    if nc <= epsilon or nt <= epsilon:
        return None
    score = float((torch.dot(current, target) / (nc * nt)).detach().cpu())
    return score if math.isfinite(score) else None


def window_observation(current, target, tau=0.05, epsilon=1e-8):
    """Pointwise direction errors; scalar aggregation cannot cancel vectors."""
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("checkpoint tau must be positive and finite")
    if current.shape != target.shape or current.ndim != 3 or current.shape[0] != 1 or current.shape[1] == 0:
        raise CheckpointContractError("expected matching nonempty [1,m,D] checkpoint hidden states")
    current, target = current.float(), target.to(current.device).float()
    nc, nt = current.norm(dim=-1), target.norm(dim=-1)
    def safe_list(value):
        return [float(v) if math.isfinite(float(v)) else None for v in value.detach().flatten().cpu()]
    result = dict(pointwise_cosine=None, pointwise_deviation=None,
                  current_norms=safe_list(nc), target_norms=safe_list(nt),
                  D_win=None, invalid_reason=None)
    if not bool(torch.isfinite(current).all() and torch.isfinite(target).all()
                and torch.isfinite(nc).all() and torch.isfinite(nt).all()):
        result["invalid_reason"] = "nonfinite_hidden_or_norm"
        return result
    if bool((nc <= epsilon).any() or (nt <= epsilon).any()):
        result["invalid_reason"] = "zero_or_small_pointwise_norm"
        return result
    cosine = ((current / nc.unsqueeze(-1)) * (target / nt.unsqueeze(-1))).sum(dim=-1)
    if not bool(torch.isfinite(cosine).all()):
        result["invalid_reason"] = "nonfinite_pointwise_cosine"
        return result
    cosine = cosine.clamp(-1, 1)
    deviation = (1 - cosine) / 2
    score = tau * (torch.logsumexp(deviation / tau, dim=-1) - math.log(current.shape[1]))
    result.update(pointwise_cosine=safe_list(cosine), pointwise_deviation=safe_list(deviation),
                  D_win=float(score.item()))
    return result


def diagnose_segment(cumulative, eta=0.05, start=0):
    """Pure trajectory rule. No fallback to a low individual token score."""
    deltas = [None] + [cumulative[i] - cumulative[i-1] for i in range(1, 5)]
    isolated, sustained = [], []
    for i in range(1, 5):
        if deltas[i] >= -eta:
            continue
        recovery = next((j for j in range(i+1, 5) if cumulative[j] >= cumulative[i-1]-eta), None)
        event = dict(selected_position=start+i, drop_baseline=cumulative[i-1],
                     recovery_position=None if recovery is None else start+recovery,
                     followup_count=4-i, drop=deltas[i])
        if recovery is not None and recovery-i <= 2:
            isolated.append(event)
        elif recovery is None and i < 4:
            sustained.append(event)
    selected = sustained[0] if sustained else min(isolated, key=lambda e: (e["drop"], e["selected_position"]), default=None)
    result = dict(deltas=deltas, selected_position=None, diagnosis_type=None)
    if selected:
        result.update(selected)
        result["diagnosis_type"] = "sustained_drop" if sustained else "isolated_drop"
    return result


def filter_existing_candidates(table, current_id, tokenizer, vocab_size, filter_nonascii, threshold):
    ids, scores = table["candidate_token_ids"], table["candidate_hidden_cosine"]
    if len(ids) != len(scores):
        raise ValueError("candidate rows do not align")
    eligible, excluded, mapping = [], [], []
    for row, (token, score) in enumerate(zip(ids, scores)):
        reason = None
        if not isinstance(token, int) or not 0 <= token < vocab_size:
            reason = "invalid_id"
        elif token in tokenizer.all_special_ids:
            reason = "special_token"
        elif filter_nonascii and not tokenizer.decode([token]).isascii():
            reason = "nonascii"
        elif not isinstance(score, numbers.Real) or not math.isfinite(score):
            reason = "nonfinite_old_score"
        elif score < threshold:
            reason = "below_threshold"
        elif token == current_id:
            reason = "current_token"
        if reason:
            excluded.append({"row": row, "token_id": token, "reason": reason})
            mapping.append(None)
        else:
            if token not in eligible:
                eligible.append(token)
            mapping.append(eligible.index(token))
    return eligible, excluded, mapping


def run_checkpoint(model, embed_layer, tokens, embedding, target, layer_id,
                   register_layer_hooks, tokenizer, config, tables, a, b, start,
                   filter_nonascii=True, *, downstream_rerank=None):
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
    begun = time.perf_counter()
    event = dict(checkpoint_id=(a-start)//5, a=a, b=b, eval_start_pos=start,
                 position_index_base=0, target_block_index=layer_id,
                 group_cosine_before=None, group_cosine_after=None, triggered=None,
                 cumulative_cosine=None, deltas=None, eta=config.checkpoint_diagnostic_tolerance,
                 selected_position=None, diagnosis_type=None, drop_baseline=None,
                 recovery_position=None, followup_count=None, candidate_generation=None,
                 prefix_fingerprint=None, original_candidate_ids=[], original_candidate_scores=[],
                 eligible_ids=[], excluded_reasons=[], duplicate_mapping=[], evaluated_ids=[],
                 candidate_group_cosines=[], failed_id=None, unevaluated_ids=[],
                 all_candidates_scored=False, old_token_id=None, new_token_id=None,
                 best_group_cosine=None, accepted=False, reason=None, repair_attempt_count=0,
                 committed_end_before=b, committed_end_after=b,
                 future_context_invalidated_from=None, forward_mode="full_prefix",
                 observation_forward_calls=0, candidate_forward_calls=0, forward_token_count=0,
                 candidate_min_cosine=config.checkpoint_candidate_min_cosine,
                 candidate_threshold_source=config.checkpoint_candidate_threshold_source,
                 segment_tokens_before=list(tokens[a:b+1]))
    event.update(downstream_policy=config.checkpoint_downstream_policy,
                 schema_version=config.checkpoint_schema_version,
                 trigger_metric=config.checkpoint_trigger_metric,
                 deviation_tau=config.checkpoint_deviation_tau,
                 trigger_deviation=config.checkpoint_trigger_deviation,
                 is_first_window=a == start, observation_valid=False,
                 D_win_before=None, D_win_after=None,
                 candidate_paths=[], segment_tokens_after=list(tokens[a:b+1]),
                 changed_positions=[], downstream_forward_calls=0, downstream_rerank_positions=0,
                 downstream_candidate_sequences=0)
    def finish(reason):
        if embedding.is_cuda:
            torch.cuda.synchronize(embedding.device)
        event["reason"] = reason
        event["elapsed_ms"] = (time.perf_counter()-begun)*1000
        return event
    event["observation_forward_calls"] = 1
    event["forward_token_count"] += b+1
    try:
        hidden = forward_discrete(model, tokens[:b+1], layer_id, register_layer_hooks)
    except Exception as error:
        raise_if_fatal(error)
        event["error_type"] = type(error).__name__
        return finish("invalid_segment_observation")
    target_segment = target[:, a:b+1]
    observation = window_observation(hidden[:, a:b+1], target_segment,
                                     config.checkpoint_deviation_tau, config.checkpoint_numeric_norm_epsilon)
    event.update(observation)
    event["D_win_before"] = event["D_win_after"] = observation["D_win"]
    old_score = segment_cosine(hidden[:, a:b+1], target_segment)
    event["group_cosine_before"] = event["group_cosine_after"] = old_score
    if observation["invalid_reason"] is not None:
        return finish("invalid_segment_observation")
    event["observation_valid"] = True
    # Compare in the score's float32 representation; no extra acceptance margin.
    event["triggered"] = observation["D_win"] > float(torch.tensor(config.checkpoint_trigger_deviation, dtype=torch.float32))
    if not event["triggered"]:
        return finish("passed")
    if old_score is None:
        return finish("invalid_acceptance_observation")
    cumulative = [segment_cosine(hidden[:, a:k+1], target[:, a:k+1]) for k in range(a,b+1)]
    event["cumulative_cosine"] = cumulative
    if any(value is None for value in cumulative):
        return finish("invalid_cumulative_observation")
    event.update(diagnose_segment(cumulative, config.checkpoint_diagnostic_tolerance, a))
    p = event["selected_position"]
    if p is None:
        return finish("no_localizable_drop")
    table = tables.get(p)
    if table is None or table.get("prefix_fingerprint") != prefix_fingerprint(tokens[:p]):
        return finish("missing_or_stale_candidate_table")
    event.update(candidate_generation=table.get("generation"), prefix_fingerprint=table["prefix_fingerprint"],
                 original_candidate_ids=list(table["candidate_token_ids"]),
                 original_candidate_scores=[float(v) if math.isfinite(float(v)) else None for v in table["candidate_hidden_cosine"]],
                 old_token_id=tokens[p], new_token_id=tokens[p])
    eligible, excluded, mapping = filter_existing_candidates(
        table, tokens[p], tokenizer, embed_layer.num_embeddings, filter_nonascii,
        config.checkpoint_candidate_min_cosine)
    event.update(eligible_ids=eligible, excluded_reasons=excluded, duplicate_mapping=mapping)
    if not eligible:
        return finish("no_eligible_alternative")
    event["repair_attempt_count"] = 1
    best_id, best_score, best_trial, best_tables = None, -math.inf, None, []
    for index, token in enumerate(eligible):
        trial = list(tokens[:b+1])
        trial[p] = token
        try:
            trial_tables = []
            if p < b:
                if downstream_rerank is None:
                    raise CheckpointContractError("downstream rerank callback is required")
                fixed_trial_prefix = list(trial[:p+1])
                trial, _, trial_tables = downstream_rerank(trial, p+1, b+1, event)
                if len(trial) != b+1 or trial[:p+1] != fixed_trial_prefix:
                    raise CheckpointContractError("downstream rerank changed the fixed prefix or length")
                if [row["position"] for row in trial_tables] != list(range(p+1, b+1)):
                    raise CheckpointContractError("downstream candidate tables do not cover the suffix")
                for row in trial_tables:
                    j = row["position"]
                    if row["prefix_fingerprint"] != prefix_fingerprint(trial[:j]) or row["selected_token_id"] != trial[j]:
                        raise CheckpointContractError("downstream candidate context mismatch")
                    scores = row["candidate_hidden_cosine"]
                    if not scores or not all(math.isfinite(float(value)) for value in scores):
                        raise ValueError("invalid downstream candidate scores")
                event["downstream_rerank_positions"] += len(trial_tables)
                event["downstream_candidate_sequences"] += sum(len(row["candidate_token_ids"]) for row in trial_tables)
            event["candidate_forward_calls"] += 1
            event["forward_token_count"] += b+1
            h = forward_discrete(model, trial, layer_id, register_layer_hooks)
            score = segment_cosine(h[:, a:b+1], target_segment)
            trial_observation = window_observation(h[:, a:b+1], target_segment,
                config.checkpoint_deviation_tau, config.checkpoint_numeric_norm_epsilon)
        except Exception as error:
            raise_if_fatal(error)
            event.update(failed_id=token, unevaluated_ids=eligible[index+1:], error_type=type(error).__name__,
                         partial_best_group_cosine=best_score if best_id is not None else None)
            return finish("candidate_forward_failed")
        if score is None or trial_observation["invalid_reason"] is not None:
            event.update(failed_id=token, unevaluated_ids=eligible[index+1:],
                         partial_best_group_cosine=best_score if best_id is not None else None)
            return finish("invalid_candidate_score")
        event["evaluated_ids"].append(token)
        event["candidate_group_cosines"].append(score)
        event["candidate_paths"].append(dict(seed_token_id=token,
            segment_tokens=list(trial[a:b+1]), group_cosine=score,
            **trial_observation,
            downstream_candidate_tables=trial_tables))
        if score > best_score:
            best_id, best_score = token, score
            best_trial, best_tables = list(trial), trial_tables
    event.update(all_candidates_scored=True, best_group_cosine=best_score)
    if best_score <= old_score:
        return finish("no_improvement")
    # Prepare replacement BEFORE mutating either piece of formal state.
    replacement_ids = torch.tensor(best_trial[p:b+1], device=embed_layer.weight.device, dtype=torch.long)
    replacement = embed_layer.weight[replacement_ids].detach().to(embedding.device, embedding.dtype).clone()
    updated_tables = {row["position"]: dict(row, generation="checkpoint_{}".format(event["checkpoint_id"]))
                      for row in best_tables}
    updated_tables[p] = dict(tables[p], selected_token_id=best_id)
    changed = [j for j in range(p,b+1) if tokens[j] != best_trial[j]]
    with torch.no_grad():
        embedding[0, p:b+1].copy_(replacement)
    tokens[p:b+1] = best_trial[p:b+1]
    tables.update(updated_tables)
    event.update(accepted=True, new_token_id=best_id, group_cosine_after=best_score,
                 D_win_after=next(path["D_win"] for path in event["candidate_paths"] if path["seed_token_id"] == best_id),
                 future_context_invalidated_from=b+1, changed_positions=changed,
                 segment_tokens_after=list(tokens[a:b+1]))
    return finish("accepted")


def empty_result(tokens, config, start):
    return dict(name=METHOD_NAME, method=METHOD_NAME, version=VERSION, enabled=True,
                formal_gt_blind=True, gt_accessed=False, pre_tokens=list(tokens),
                final_tokens=list(tokens), final_text="", eval_start_pos=start,
                events=[], initial_candidate_rerank=[], attempt_count=0,
                checkpoint=dict(enabled=config.checkpoint_enabled, events=[],
                                complete_segment_count=0, tail_skipped_token_count=0))


def optimize_stage1(model, initial_embedding, prefix_embedding, target_hidden_state,
                    attention_mask, layer_id, register_layer_hooks, weight_mask,
                    right_range, lr, epoch, alpha, clip=True, optim_method="cosine"):
    """Legacy v2.2.1 Stage-1, excluding its benchmark-only token decoding."""
    variable = initial_embedding.detach().clone().requires_grad_(True)
    begun = time.perf_counter()
    completed = 0
    for step in range(epoch if variable.shape[1] else 0):
        if clip:
            with torch.no_grad():
                variable = torch.clip(variable, -0.2, 0.2)
        variable = variable.requires_grad_(True)
        optimizer = torch.optim.SGD([variable], lr=lr)
        full = torch.cat((prefix_embedding, variable), dim=1) if prefix_embedding is not None else variable
        hidden = _forward_embedding_hidden(model, full, attention_mask, layer_id, register_layer_hooks)
        target = target_hidden_state.to(hidden.device)
        cosine = F.cosine_similarity(hidden.float(), target.float(), dim=-1)
        mse = F.mse_loss(hidden.float(), target.float())
        range_loss = F.relu(full.abs() - right_range).sum()
        cosine_loss = (-cosine.to(weight_mask.device)*weight_mask).sum()
        loss = mse if optim_method == "MSELoss" else cosine_loss + alpha*range_loss
        if optim_method not in ("MSELoss", "cosine"):
            raise ValueError("unsupported Stage-1 objective")
        optimizer.zero_grad()
        loss.backward(inputs=[variable])
        if torch.isnan(cosine).any():
            break
        optimizer.step()
        completed += 1
    full = torch.cat((prefix_embedding, variable), dim=1) if prefix_embedding is not None else variable
    return full.detach().clone(), dict(epoch=completed-1, completed_steps=completed,
                                       elapsed_seconds=time.perf_counter()-begun)


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_two_stage(*, stage1_kwargs, stage2_kwargs, snapshot_path=None, snapshot_mode=None,
                  pair_id=None, snapshot_contract=None):
    """Full entry with a GT-free persisted Stage-1 snapshot for the paired runner."""
    target = stage2_kwargs["target_hidden_state"].detach()
    if snapshot_mode not in (None, "write", "read"):
        raise ValueError("snapshot_mode must be write/read or absent")
    if snapshot_mode and not snapshot_path:
        raise ValueError("snapshot_path required")
    path = Path(snapshot_path) if snapshot_path else None
    if snapshot_mode == "read":
        # Only local snapshots just produced by this runner are accepted.
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot["pair_id"] != pair_id or snapshot["contract"] != snapshot_contract:
            raise ValueError("Stage-1 snapshot identity/config mismatch")
        if not torch.equal(target.cpu(), snapshot["target"]):
            raise ValueError("paired target observation differs")
        embedding = snapshot["embedding"].to(stage1_kwargs["initial_embedding"].device)
        summary = snapshot["stage1"]
        restore_rng(snapshot["rng"])
    else:
        embedding, summary = optimize_stage1(**stage1_kwargs)
        snapshot = dict(pair_id=pair_id, contract=snapshot_contract, embedding=embedding.cpu(),
                        target=target.cpu(), stage1=summary, rng=rng_state())
        if snapshot_mode == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            torch.save(snapshot, path)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest() if path else hashlib.sha256(
        embedding.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    stage2_kwargs = dict(stage2_kwargs, optimized_embedding=embedding.detach().clone(),
                         target_hidden_state=target.clone())
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
        torch.cuda.reset_peak_memory_stats(embedding.device)
    started = time.perf_counter()
    final, result = run_suffix_reoptimization_v2_2_2(**stage2_kwargs)
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
    elapsed = time.perf_counter()-started
    result["triggered"] = bool(result.get("triggered") or result["checkpoint"].get("trigger_count"))
    result["accepted"] = bool(result.get("accepted") or result["checkpoint"].get("accepted_count"))
    reoptimization = dict(result)
    result.update(stage1=summary, reoptimization=reoptimization,
                  pair_id=pair_id, stage1_snapshot_sha256=fingerprint,
                  second_stage_seconds=elapsed, stage1_reused=snapshot_mode == "read",
                  comparable_end_to_end_seconds=elapsed+summary.get("elapsed_seconds", 0.),
                  second_stage_peak_memory_bytes=torch.cuda.max_memory_allocated(embedding.device) if embedding.is_cuda else None)
    return final, finite_json(result)


def finite_json(value):
    """Diagnostic nonfinite values must not escape into non-standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value
