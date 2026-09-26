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
    checkpoint_candidate_top_k: int = 9
    checkpoint_candidate_policy: str = "multi_source_3_2_4"
    checkpoint_acceptance_metric: str = "window_sum_delta"
    checkpoint_acceptance_epsilon_source: str = "independent_synthetic_calibration"
    checkpoint_acceptance_calibration_repeats: int = 3
    checkpoint_forward_mode: str = "full_prefix"
    checkpoint_candidate_failure_policy: str = "skip_invalid_candidate_abort_hard_failure"
    checkpoint_tail_policy: str = "skip_incomplete"
    checkpoint_max_repairs: int = 1
    checkpoint_recursive: bool = False
    checkpoint_numeric_norm_epsilon: float = 1e-8
    checkpoint_score_dtype: str = "float32"
    checkpoint_schema_version: int = 5
    checkpoint_downstream_policy: str = "fixed_tokens_single_point_scan"
    checkpoint_stage_top_k: int = 10
    checkpoint_acceptance_epsilon: float = None
    checkpoint_consistency_tolerance: float = None
    checkpoint_calibration_id: str = None

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
        layer_id, register_layer_hooks, embed_layer, config, trajectory_callback=None):
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
        if trajectory_callback is not None:
            trajectory_callback(completed_steps, _merge_suffix(
                anchored_base, suffix_start, suffix_param.detach()))

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
        forward_and_get_last_hidden_state=None, log_file=None, stage_candidates=None):
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
    stage_candidates = dict(stage_candidates or {})
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
            filter_nonascii, stage_candidates=stage_candidates,
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
            trajectory = (StageCandidateCollector(
                config.steps, position, embed_layer, invert_method,
                "R_{}".format(attempts), current_tokens[:position])
                if config.checkpoint_enabled and config.checkpoint_schema_version == 5 else None)
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
                    trajectory_callback=trajectory,
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
            trial_stage_candidates = (trajectory.finalize(summary.get("completed_steps", 0), sequence_length)
                                      if trajectory is not None else {})
            if trajectory is not None:
                summary["checkpoint_trajectory"] = trajectory.metadata()
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
                if trajectory is not None:
                    stage_candidates.update(trial_stage_candidates)
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


def stage_neighbours(embedding, embed_layer, top_k=10, metric="cosine"):
    """Chunked FP32 CPU search, deterministic token-ID order for distance ties."""
    points = embedding.detach().reshape(-1, embedding.shape[-1]).float().cpu()
    if not torch.isfinite(points).all():
        raise ValueError("nonfinite trajectory embedding")
    best_scores = torch.empty((len(points), 0))
    best_ids = torch.empty((len(points), 0), dtype=torch.long)
    with torch.no_grad():
        for begin in range(0, embed_layer.num_embeddings, EMBEDDING_SEARCH_CHUNK_SIZE):
            weight = embed_layer.weight[begin:begin+EMBEDDING_SEARCH_CHUNK_SIZE].detach().float().cpu()
            if metric == "cosine":
                scores = F.normalize(points, dim=-1, eps=1e-8) @ F.normalize(weight, dim=-1, eps=1e-8).T
            elif metric == "L2":
                scores = -torch.cdist(points, weight, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            else:
                raise ValueError("unsupported trajectory metric")
            scores = torch.where(torch.isfinite(scores), scores, -torch.inf)
            ids = torch.arange(begin, begin+len(weight)).expand(len(points), -1)
            scores, ids = torch.cat((best_scores, scores), 1), torch.cat((best_ids, ids), 1)
            by_id = torch.argsort(ids, dim=1, stable=True)
            scores, ids = scores.gather(1, by_id), ids.gather(1, by_id)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
            best_scores, best_ids = scores.gather(1, order), ids.gather(1, order)
    return [[int(t) for t, s in zip(row, values) if math.isfinite(float(s))]
            for row, values in zip(best_ids, best_scores)]


class StageCandidateCollector:
    """Retain only two Top10 snapshots, never the full optimization trajectory."""
    def __init__(self, steps, start, embed_layer, metric, call_id, prefix):
        self.steps, self.start = int(steps), int(start)
        self.embed_layer, self.metric = embed_layer, metric
        self.call_id, self.prefix = call_id, prefix_fingerprint(prefix)
        self.points = list(dict.fromkeys((math.ceil(steps/3), math.ceil(2*steps/3))))
        self.snapshots, self.elapsed_seconds, self.retrievals = {}, 0., 0

    def __call__(self, step, embedding):
        if step not in self.points or step >= self.steps:
            return
        begun = time.perf_counter()
        rows = stage_neighbours(embedding[:, self.start:], self.embed_layer, metric=self.metric)
        self.snapshots[step] = rows
        self.retrievals += len(rows)
        self.elapsed_seconds += time.perf_counter()-begun

    def finalize(self, completed_steps, sequence_length=None):
        self.completed_steps = int(completed_steps)
        kept = {step: rows for step, rows in self.snapshots.items() if step < completed_steps}
        length = (sequence_length-self.start if sequence_length is not None else
                  max((len(rows) for rows in self.snapshots.values()), default=0))
        # Missing stages overwrite prior calls too: never borrow an older trajectory.
        return {self.start+i: dict(call_id=self.call_id, prefix_fingerprint=self.prefix,
                    planned_steps=self.steps, completed_steps=completed_steps,
                    planned_sampling_steps=self.points,
                    stages=[dict(step=step, token_ids=rows[i]) for step, rows in sorted(kept.items())])
                for i in range(length)}

    def metadata(self):
        return dict(call_id=self.call_id, planned_steps=self.steps,
                    completed_steps=getattr(self, "completed_steps", None),
                    planned_sampling_steps=self.points, captured_steps=sorted(self.snapshots),
                    retrievals=self.retrievals, elapsed_seconds=self.elapsed_seconds)


def build_checkpoint_candidates(table, current_id, tokenizer, vocab_size,
                                filter_nonascii=True, trajectory=None):
    """4-1: stable 3+1+1+1+1+1+1 quotas; no model forwards or private labels."""
    selected, by_id, seen = [], {}, []
    counts = Counter()
    specials = set(tokenizer.all_special_ids)

    def valid(token):
        return (type(token) is int and 0 <= token < vocab_size and token not in specials
                and (not filter_nonascii or tokenizer.decode([token]).isascii()))

    def offer(token, source, **detail):
        counts[source+"_visited"] += 1
        label = dict(source=source, **detail)
        if not valid(token):
            seen.append(dict(token_id=token, reason="invalid_or_filtered", **label))
            return False
        if token == current_id:
            seen.append(dict(token_id=token, reason="current_token", **label))
            return False
        if token in by_id:
            by_id[token]["sources"].append(label)
            seen.append(dict(token_id=token, reason="duplicate", **label))
            return False
        row = dict(token_id=token, order=len(selected), quota_source=source, sources=[label])
        selected.append(row)
        by_id[token] = row
        return True

    old_ids, old_excluded, _ = filter_existing_candidates(
        table, current_id, tokenizer, vocab_size, filter_nonascii, vocab_size)
    for rank, token in enumerate(old_ids):
        if offer(token, "old", rank=rank+1) and sum(r["quota_source"] == "old" for r in selected) == 3:
            break
    stages = sorted((trajectory or {}).get("stages", []), key=lambda row: row["step"])
    for stage in stages[:2]:
        source = "stage_{}".format(stage["step"])
        for rank, token in enumerate(stage["token_ids"][:10]):
            if offer(token, source, step=stage["step"], rank=rank+1,
                     call_id=trajectory.get("call_id")):
                break
    stage_selected = [r["token_id"] for r in selected if r["quota_source"].startswith("stage_")]
    seeds = list(dict.fromkeys(old_ids+stage_selected+[current_id]))
    seed_texts = []

    def encode_single(text):
        counts["tokenizer_queries"] += 1
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or not valid(int(ids[0])):
            return None
        token = int(ids[0])
        if tokenizer.decode([token], clean_up_tokenization_spaces=False) != text:
            return None
        return token

    for seed in seeds:
        if not valid(seed):
            continue
        text = tokenizer.decode([seed], clean_up_tokenization_spaces=False)
        if text and encode_single(text) == seed:
            seed_texts.append((seed, text))

    def variants(source, text):
        if source == "case":
            leading = len(text)-len(text.lstrip())
            capitalized = text[:leading]+text[leading:].capitalize()
            yield from dict.fromkeys((text.lower(), text.upper(), capitalized))
        elif source == "space":
            yield text[1:] if text.startswith(" ") else " "+text
        else:
            for size in range(len(text)-1, 0, -1):
                yield text[:size] if source == "prefix" else text[-size:]

    for source in ("case", "space", "prefix", "suffix"):
        found = False
        for seed, text in seed_texts:
            for value in variants(source, text):
                if not value or value == text:
                    continue
                counts[source+"_generated"] += 1
                token = encode_single(value)
                if token is None:
                    counts[source+"_invalid"] += 1
                    continue
                if offer(token, source, seed_token_id=seed, text=value):
                    found = True
                    break
            if found:
                break
    return dict(candidate_ids=[r["token_id"] for r in selected], candidates=selected,
                baseline_token_id=current_id, original_candidate_ids=table["candidate_token_ids"],
                original_excluded=old_excluded, visited_exclusions=seen, counts=dict(counts),
                trajectory=trajectory, generation=table.get("generation"),
                source_prefix_fingerprint=table.get("prefix_fingerprint"))


def validate_checkpoint_config(config):
    """Explicit in-place checkpoint revision; old schemas cannot run silently."""
    expected = {
        "checkpoint_size": 5, "checkpoint_stride": 5,
        "checkpoint_trigger_metric": "pointwise_logmeanexp",
        "checkpoint_deviation_tau": 0.05,
        "checkpoint_trigger_deviation": 0.05,
        "checkpoint_diagnostic_tolerance": 0.02,
        "checkpoint_candidate_top_k": 9,
        "checkpoint_candidate_policy": "multi_source_3_2_4",
        "checkpoint_acceptance_metric": "window_sum_delta",
        "checkpoint_acceptance_epsilon_source": "independent_synthetic_calibration",
        "checkpoint_acceptance_calibration_repeats": 3,
        "checkpoint_forward_mode": "full_prefix",
        "checkpoint_candidate_failure_policy": "skip_invalid_candidate_abort_hard_failure",
        "checkpoint_tail_policy": "skip_incomplete",
        "checkpoint_max_repairs": 1, "checkpoint_recursive": False,
        "checkpoint_numeric_norm_epsilon": 1e-8,
        "checkpoint_score_dtype": "float32", "checkpoint_schema_version": 5,
        "checkpoint_downstream_policy": "fixed_tokens_single_point_scan",
        "checkpoint_stage_top_k": 10,
    }
    if config.checkpoint_schema_version == 4:
        expected.update(checkpoint_schema_version=4, checkpoint_candidate_top_k=3,
            checkpoint_candidate_policy="stable_top_k_alternatives",
            checkpoint_acceptance_metric="pointwise_logmeanexp",
            checkpoint_acceptance_epsilon_source="per_window_repeated_forward_range",
            checkpoint_candidate_failure_policy="abort_checkpoint_keep_state",
            checkpoint_downstream_policy="sequential_rerank_to_checkpoint_end")
    if type(config.checkpoint_enabled) is not bool:
        raise TypeError("checkpoint_enabled must be boolean")
    for key, expected_value in expected.items():
        actual = getattr(config, key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError("{} must be {!r}".format(key, expected_value))
    for name in ("checkpoint_acceptance_epsilon", "checkpoint_consistency_tolerance"):
        value = getattr(config, name)
        if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
            raise ValueError(name+" must be finite and nonnegative, or null before calibration")


def config_from_mapping(values, require_explicit=True):
    from dataclasses import fields
    parsed = {}
    for obsolete in ("checkpoint_trigger_cosine", "checkpoint_candidate_min_cosine",
                     "checkpoint_candidate_threshold_source"):
        if "suffix_v2_2_2_" + obsolete in values:
            raise ValueError("obsolete " + obsolete + "; migrate to explicit checkpoint settings")
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


def diagnose_segment(deviation, D_win, eta=0.02, start=0):
    """Earliest position above the relative deviation cutoff, not a token oracle."""
    if (not deviation or not all(math.isfinite(v) for v in deviation)
            or not math.isfinite(D_win) or not math.isfinite(eta) or eta < 0):
        raise ValueError("localization requires finite observations and nonnegative tolerance")
    cutoff = D_win - eta
    position = next((start+i for i, value in enumerate(deviation) if value >= cutoff), None)
    return dict(selected_position=position, localization_threshold=cutoff,
                diagnosis_type="earliest_relative_deviation" if position is not None else None)


def filter_existing_candidates(table, current_id, tokenizer, vocab_size, filter_nonascii, top_k):
    ids, scores = table["candidate_token_ids"], table["candidate_hidden_cosine"]
    if len(ids) != len(scores):
        raise ValueError("candidate rows do not align")
    eligible, excluded, mapping = [], [], [None] * len(ids)
    valid_rows, unique = [], {}
    for row, (token, score) in enumerate(zip(ids, scores)):
        reason = None
        if type(token) is not int or not 0 <= token < vocab_size:
            reason = "invalid_id"
        elif token in tokenizer.all_special_ids:
            reason = "special_token"
        elif filter_nonascii and not tokenizer.decode([token]).isascii():
            reason = "nonascii"
        elif isinstance(score, bool) or not isinstance(score, numbers.Real) or not math.isfinite(score):
            reason = "nonfinite_old_score"
        elif token == current_id:
            reason = "current_token"
        if reason:
            excluded.append({"row": row, "token_id": token, "reason": reason})
        else:
            valid_rows.append((row, token))
            # First valid occurrence defines both the deduplicated score and tie order.
            unique.setdefault(token, float(score))
    eligible = sorted(unique, key=lambda token: -unique[token])[:top_k]
    selected = {token: index for index, token in enumerate(eligible)}
    for row, token in valid_rows:
        mapping[row] = selected.get(token)
        if token not in selected:
            excluded.append(dict(row=row, token_id=token, reason="outside_top_k"))
    excluded.sort(key=lambda item: item["row"])
    return eligible, excluded, mapping


def joint_window_delta(before, after):
    """4-3 shared score: total hidden deviation change, without a local gate."""
    if before.get("invalid_reason") or after.get("invalid_reason"):
        raise ValueError("invalid window observation")
    left, right = before["pointwise_deviation"], after["pointwise_deviation"]
    if len(left) != len(right) or not left or not all(math.isfinite(v) for v in left+right):
        raise CheckpointContractError("window score alignment mismatch")
    return math.fsum(y-x for x, y in zip(left, right))


def calibrate_checkpoint(model, embed_layer, tokenizer, layer_id, register_layer_hooks, config):
    """Independent fixed synthetic inputs, before experiment samples; never GT."""
    if not config.checkpoint_enabled or config.checkpoint_schema_version != 5:
        return None
    if model.training:
        raise ValueError("calibration requires model.eval()")
    if any(getattr(config, key) is not None for key in (
            "checkpoint_acceptance_epsilon", "checkpoint_consistency_tolerance", "checkpoint_calibration_id")):
        raise ValueError("fresh runtime calibration required; configured thresholds must be null")
    begun, saved_rng = time.perf_counter(), rng_state()
    specials = set(tokenizer.all_special_ids)
    vocab = embed_layer.num_embeddings
    # Seven deterministic vocabulary positions; no benchmark text/target is read.
    pool = list(dict.fromkeys(min(vocab-1, i*vocab//8) for i in range(1, 8)))
    pool = [t for t in pool if t not in specials]
    if len(pool) < 2:
        pool = [t for t in range(vocab) if t not in specials][:7]
    if len(pool) < 2:
        raise ValueError("calibration requires at least two ordinary tokens")
    limit = int(getattr(getattr(model, "config", None), "max_position_embeddings", 512))
    lengths = [n for n in (5, 10, 32, 128, 512) if n <= limit]
    if not lengths:
        raise ValueError("model context too short for five-token checkpoints")
    rows, forward_calls, token_count = [], 0, 0
    def observe(ids, target=None):
        nonlocal forward_calls, token_count
        forward_calls += 1
        token_count += len(ids)
        hidden = forward_discrete(model, ids, layer_id, register_layer_hooks)
        if target is None:
            return hidden[:, -5:].clone()
        result = window_observation(hidden[:, -5:], target)
        if result["invalid_reason"]:
            raise ValueError("invalid independent calibration observation")
        return result
    try:
        for length in lengths:
            initial = [pool[i % len(pool)] for i in range(length)]
            reference = [pool[(i+1) % len(pool)] for i in range(length)]
            target = observe(reference)
            states = [list(initial)]
            for position in range(length-5, length):
                trial = list(states[-1])
                trial[position] = pool[(position+1) % len(pool)]
                states.append(trial)
            observations = [[observe(ids, target) for _ in range(config.checkpoint_acceptance_calibration_repeats)]
                            for ids in states]
            totals = [[math.fsum(o["pointwise_deviation"]) for o in repeats] for repeats in observations]
            deltas = [[joint_window_delta(observations[0][r], observations[j][r])
                       for r in range(config.checkpoint_acceptance_calibration_repeats)]
                      for j in range(1, len(states))]
            affected_errors = []
            for j in range(1, len(states)):
                before, after = observations[j-1][0], observations[j][0]
                affected = math.fsum(y-x for x, y in zip(
                    before["pointwise_deviation"][j-1:], after["pointwise_deviation"][j-1:]))
                affected_errors.append(abs(joint_window_delta(before, after)-affected))
            telescoping = abs(math.fsum(joint_window_delta(observations[j-1][0], observations[j][0])
                            for j in range(1, len(states)))-joint_window_delta(observations[0][-1], observations[-1][-1]))
            rows.append(dict(prefix_length=length, state_F_repeats=totals, S_repeats=deltas,
                             affected_interval_error=max(affected_errors), telescoping_error=telescoping))
    finally:
        restore_rng(saved_rng)
    observed = max([max(values)-min(values) for row in rows
                    for values in row["state_F_repeats"]+row["S_repeats"]]
                   + [row["affected_interval_error"] for row in rows])
    # FP32 cosine, Python float64 fsum. A positive reduction floor is explicit,
    # not a claim that repeated deterministic outputs establish zero error.
    floor = 8 * torch.finfo(torch.float32).eps * config.checkpoint_size
    epsilon = max(2*observed, floor)
    consistency = max(6*epsilon, 2*max(row["telescoping_error"] for row in rows))
    report = dict(protocol="synthetic_fixed_vocabulary_v1", calibration_kind="independent_no_gt",
                  model_class=type(model).__name__, layer_id=layer_id,
                  model_name=getattr(getattr(model, "config", None), "_name_or_path", None),
                  dtype=str(next(model.parameters()).dtype), device=str(next(model.parameters()).device),
                  torch_version=str(torch.__version__), forward_mode="full_prefix_batch1_no_cache",
                  score_precision="fp32_cosine_float64_fsum", lengths=lengths, seed_token_ids=pool,
                  repeats=config.checkpoint_acceptance_calibration_repeats, rows=rows,
                  max_observed_error=observed, numerical_floor=floor, safety_multiplier=2,
                  epsilon=epsilon, consistency_tolerance=consistency,
                  forward_calls=forward_calls, forward_token_count=token_count)
    identity = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode()).hexdigest()
    report.update(calibration_id=identity, elapsed_seconds=time.perf_counter()-begun)
    config.checkpoint_acceptance_epsilon = epsilon
    config.checkpoint_consistency_tolerance = consistency
    config.checkpoint_calibration_id = identity
    config._checkpoint_calibration_report = report
    return report


def run_checkpoint(model, embed_layer, tokens, embedding, target, layer_id,
                   register_layer_hooks, tokenizer, config, tables, a, b, start,
                   filter_nonascii=True, *, downstream_rerank=None, stage_candidates=None):
    if config.checkpoint_schema_version == 4:
        return _run_checkpoint_schema4(model, embed_layer, tokens, embedding, target, layer_id,
            register_layer_hooks, tokenizer, config, tables, a, b, start,
            filter_nonascii, downstream_rerank=downstream_rerank)
    epsilon, tolerance = config.checkpoint_acceptance_epsilon, config.checkpoint_consistency_tolerance
    if (epsilon is None or tolerance is None or not config.checkpoint_calibration_id
            or not math.isfinite(epsilon) or not math.isfinite(tolerance) or epsilon < 0 or tolerance < 0):
        raise CheckpointContractError("schema 5 requires frozen independent calibration")
    report = getattr(config, "_checkpoint_calibration_report", None)
    if report and b+1 > max(report["lengths"]):
        raise CheckpointContractError("prefix exceeds independently calibrated length range")
    if b-a+1 != 5 or a < start or b >= len(tokens) or model.training:
        raise CheckpointContractError("invalid checkpoint boundary or model state")
    begun = time.perf_counter()
    original, current = list(tokens[:b+1]), list(tokens[:b+1])
    event = dict(checkpoint_id=(a-start)//5, a=a, b=b, eval_start_pos=start,
        schema_version=5, candidate_top_k=9, candidate_policy=config.checkpoint_candidate_policy,
        search_policy="left_to_right_single_point", downstream_policy=config.checkpoint_downstream_policy,
        acceptance_metric=config.checkpoint_acceptance_metric,
        acceptance_epsilon_source=config.checkpoint_acceptance_epsilon_source,
        acceptance_calibration_repeats=config.checkpoint_acceptance_calibration_repeats,
        acceptance_epsilon=epsilon, consistency_tolerance=tolerance,
        calibration_id=config.checkpoint_calibration_id, acceptance_threshold=-epsilon,
        score_precision="fp32_cosine_float64_fsum", forward_mode="full_prefix",
        trigger_metric=config.checkpoint_trigger_metric, trigger_deviation=config.checkpoint_trigger_deviation,
        deviation_tau=config.checkpoint_deviation_tau, target_block_index=layer_id, position_index_base=0,
        accepted=False, triggered=False, observation_valid=False, selected_position=None,
        repair_attempt_count=0, all_candidates_scored=False, changed_positions=[],
        segment_tokens_before=list(original[a:]), segment_tokens_after=list(original[a:]),
        observation_forward_calls=0, baseline_forward_calls=0, candidate_forward_calls=0,
        verification_forward_calls=0, calibration_forward_calls=0, downstream_forward_calls=0,
        forward_token_count=0, candidate_state_count=0, valid_candidate_count=0,
        candidate_generation_seconds=0., positions=[], candidate_pools=[], local_deltas=[],
        D_win_before=None, D_win_after=None, S_final=0., committed_end_before=b, committed_end_after=b,
        future_context_invalidated_from=None)

    def finish(reason):
        if embedding.is_cuda:
            torch.cuda.synchronize(embedding.device)
        event.update(reason=reason, elapsed_ms=(time.perf_counter()-begun)*1000)
        return event

    def observe(ids, counter):
        event[counter] += 1
        event["forward_token_count"] += len(ids)
        hidden = forward_discrete(model, ids, layer_id, register_layer_hooks)
        result = window_observation(hidden[:, a:b+1], target[:, a:b+1],
            config.checkpoint_deviation_tau, config.checkpoint_numeric_norm_epsilon)
        if result["invalid_reason"]:
            raise ValueError(result["invalid_reason"])
        return result

    try:
        entry = observe(original, "observation_forward_calls")
    except Exception as error:
        raise_if_fatal(error)
        event["error_type"] = type(error).__name__
        return finish("invalid_segment_observation")
    event.update(observation_valid=True, D_win_before=entry["D_win"], D_win_after=entry["D_win"],
                 pointwise_deviation=entry["pointwise_deviation"],
                 pointwise_deviation_after=entry["pointwise_deviation"],
                 pointwise_deviation_change=[0.]*5)
    event["triggered"] = entry["D_win"] > float(torch.tensor(config.checkpoint_trigger_deviation))
    if not event["triggered"]:
        return finish("passed")
    event.update(diagnose_segment(entry["pointwise_deviation"], entry["D_win"],
                                 config.checkpoint_diagnostic_tolerance, a))
    # Localization is diagnostic only. Freeze every searchable pool at entry.
    try:
        for position in range(a, b+1):
            if original[position] in tokenizer.all_special_ids:
                event["candidate_pools"].append(dict(position=position, candidate_ids=[], reason="frozen_special_token"))
                continue
            if position not in tables:
                return finish("missing_candidate_table")
            tick = time.perf_counter()
            pool = build_checkpoint_candidates(tables[position], original[position], tokenizer,
                embed_layer.num_embeddings, filter_nonascii, (stage_candidates or {}).get(position))
            event["candidate_generation_seconds"] += time.perf_counter()-tick
            event["candidate_pools"].append(dict(position=position, **pool))
    except Exception as error:
        raise_if_fatal(error)
        event["error_type"] = type(error).__name__
        return finish("candidate_generation_failed")
    if not any(pool["candidate_ids"] for pool in event["candidate_pools"]):
        return finish("no_eligible_alternative")
    event["repair_attempt_count"] = 1
    for pool in event["candidate_pools"]:
        position = pool["position"]
        if not pool["candidate_ids"]:
            continue
        try:
            baseline = observe(current, "baseline_forward_calls")
        except Exception as error:
            raise_if_fatal(error)
            event["error_type"] = type(error).__name__
            return finish("invalid_position_baseline")
        row = dict(position=position, baseline_tokens=list(current[a:]),
                   prefix_fingerprint=prefix_fingerprint(current[:position]),
                   d_before=baseline["pointwise_deviation"], trials=[], accepted=False)
        event["positions"].append(row)
        best, best_delta = None, 0.
        for token in pool["candidate_ids"]:
            trial = list(current)
            trial[position] = token
            item = dict(token_id=token, segment_tokens=list(trial[a:]), valid=False, S=None)
            row["trials"].append(item)
            event["candidate_state_count"] += 1
            try:
                observed = observe(trial, "candidate_forward_calls")
                delta = joint_window_delta(baseline, observed)
            except Exception as error:
                raise_if_fatal(error)
                item["reason"] = "invalid_candidate_score"
                item["error_type"] = type(error).__name__
                continue
            item.update(valid=True, S=delta, d_after=observed["pointwise_deviation"], D_win=observed["D_win"])
            changes = [y-x for x, y in zip(baseline["pointwise_deviation"], observed["pointwise_deviation"])]
            item.update(pointwise_deviation_change=changes,
                        S_down=math.fsum(changes[position-a+1:]), M=max(changes[position-a:]))
            event["valid_candidate_count"] += 1
            if delta < best_delta:
                best, best_delta = trial, delta
        if best is not None and best_delta < -epsilon:
            current = best
            event["local_deltas"].append(best_delta)
            row.update(accepted=True, selected_token_id=current[position], S=best_delta)
        else:
            row.update(selected_token_id=current[position], S=0.)
    event["all_candidates_scored"] = event["valid_candidate_count"] == event["candidate_state_count"]
    if current == original:
        return finish("no_improvement")
    try:
        fresh_entry = observe(original, "verification_forward_calls")
        final = observe(current, "verification_forward_calls")
        delta = joint_window_delta(fresh_entry, final)
        mismatch = abs(delta-math.fsum(event["local_deltas"]))
        event.update(S_proposed=delta, local_sum=math.fsum(event["local_deltas"]), consistency_error=mismatch)
        if abs(joint_window_delta(entry, fresh_entry)) > tolerance or mismatch > tolerance:
            return finish("inconsistent_final_verification")
        if delta >= -epsilon:
            return finish("no_final_improvement")
    except Exception as error:
        raise_if_fatal(error)
        event["error_type"] = type(error).__name__
        return finish("final_verification_failed")
    changed = [j for j in range(a, b+1) if current[j] != original[j]]
    ids = torch.tensor([current[j] for j in changed], device=embed_layer.weight.device)
    replacement = embed_layer.weight[ids].detach().to(embedding.device, embedding.dtype).clone()
    indices = torch.tensor(changed, device=embedding.device)
    previous = embedding[0, indices].detach().clone()
    try:
        with torch.no_grad():
            embedding[0, indices] = replacement
    except Exception:
        with torch.no_grad():
            embedding[0, indices] = previous
        raise
    tokens[a:b+1] = current[a:b+1]
    # Retain original candidate provenance; scores are not relabelled as fresh.
    event.update(accepted=True, changed_positions=changed, S_final=delta,
                 D_win_after=final["D_win"], segment_tokens_after=list(current[a:]),
                 pointwise_deviation_after=final["pointwise_deviation"],
                 pointwise_deviation_change=[y-x for x, y in zip(
                     entry["pointwise_deviation"], final["pointwise_deviation"])],
                 future_context_invalidated_from=b+1)
    return finish("accepted")


def _run_checkpoint_schema4(model, embed_layer, tokens, embedding, target, layer_id,
                   register_layer_hooks, tokenizer, config, tables, a, b, start,
                   filter_nonascii=True, *, downstream_rerank=None):
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
    begun = time.perf_counter()
    event = dict(checkpoint_id=(a-start)//5, a=a, b=b, eval_start_pos=start,
                 position_index_base=0, target_block_index=layer_id,
                 group_cosine_before=None, group_cosine_after=None, triggered=None,
                 eta=config.checkpoint_diagnostic_tolerance,
                 selected_position=None, diagnosis_type=None, localization_threshold=None,
                 candidate_generation=None,
                 prefix_fingerprint=None, original_candidate_ids=[], original_candidate_scores=[],
                 eligible_ids=[], excluded_reasons=[], duplicate_mapping=[], evaluated_ids=[],
                 candidate_group_cosines=[], failed_id=None, unevaluated_ids=[],
                 all_candidates_scored=False, old_token_id=None, new_token_id=None,
                 best_group_cosine=None, accepted=False, reason=None, repair_attempt_count=0,
                 committed_end_before=b, committed_end_after=b,
                 future_context_invalidated_from=None, forward_mode="full_prefix",
                 observation_forward_calls=0, candidate_forward_calls=0, forward_token_count=0,
                 candidate_top_k=config.checkpoint_candidate_top_k,
                 candidate_policy=config.checkpoint_candidate_policy,
                 segment_tokens_before=list(tokens[a:b+1]))
    event.update(downstream_policy=config.checkpoint_downstream_policy,
                 schema_version=config.checkpoint_schema_version,
                 trigger_metric=config.checkpoint_trigger_metric,
                 deviation_tau=config.checkpoint_deviation_tau,
                 trigger_deviation=config.checkpoint_trigger_deviation,
                 is_first_window=a == start, observation_valid=False,
                 D_win_before=None, D_win_after=None,
                 acceptance_metric=config.checkpoint_acceptance_metric,
                 acceptance_epsilon_source=config.checkpoint_acceptance_epsilon_source,
                 acceptance_calibration_repeats=config.checkpoint_acceptance_calibration_repeats,
                 acceptance_calibration_scores=[], acceptance_epsilon=None, acceptance_threshold=None,
                 effective_acceptance_config=None,
                 calibration_forward_calls=0, best_D_win=None,
                 pointwise_deviation_after=None, pointwise_deviation_change=None,
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
    event["pointwise_deviation_after"] = observation.get("pointwise_deviation")
    if observation.get("pointwise_deviation") is not None:
        event["pointwise_deviation_change"] = [0.] * len(observation["pointwise_deviation"])
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
    event.update(diagnose_segment(observation["pointwise_deviation"], observation["D_win"],
                                  config.checkpoint_diagnostic_tolerance, a))
    p = event["selected_position"]
    if p is None:
        return finish("no_localizable_deviation")
    table = tables.get(p)
    if table is None or table.get("prefix_fingerprint") != prefix_fingerprint(tokens[:p]):
        return finish("missing_or_stale_candidate_table")
    event.update(candidate_generation=table.get("generation"), prefix_fingerprint=table["prefix_fingerprint"],
                 original_candidate_ids=list(table["candidate_token_ids"]),
                 original_candidate_scores=[float(v) if math.isfinite(float(v)) else None for v in table["candidate_hidden_cosine"]],
                 old_token_id=tokens[p], new_token_id=tokens[p])
    eligible, excluded, mapping = filter_existing_candidates(
        table, tokens[p], tokenizer, embed_layer.num_embeddings, filter_nonascii,
        config.checkpoint_candidate_top_k)
    event.update(eligible_ids=eligible, excluded_reasons=excluded, duplicate_mapping=mapping)
    if not eligible:
        return finish("no_eligible_alternative")
    # Calibrate on this unchanged discrete path and actual runtime, never on labels.
    # The initial observation counts as the first of the bounded repeated forwards.
    calibration = event["acceptance_calibration_scores"]
    calibration.append(observation["D_win"])
    for _ in range(config.checkpoint_acceptance_calibration_repeats - 1):
        event["calibration_forward_calls"] += 1
        event["forward_token_count"] += b+1
        try:
            repeated = forward_discrete(model, tokens[:b+1], layer_id, register_layer_hooks)
            repeated_observation = window_observation(repeated[:, a:b+1], target_segment,
                config.checkpoint_deviation_tau, config.checkpoint_numeric_norm_epsilon)
        except Exception as error:
            raise_if_fatal(error)
            event["error_type"] = type(error).__name__
            return finish("acceptance_calibration_failed")
        if repeated_observation["invalid_reason"] is not None:
            return finish("invalid_acceptance_calibration")
        calibration.append(repeated_observation["D_win"])
    epsilon = max(calibration) - min(calibration)
    event["acceptance_epsilon"] = epsilon
    event["acceptance_threshold"] = observation["D_win"] - epsilon
    event["effective_acceptance_config"] = dict(
        metric=config.checkpoint_acceptance_metric, epsilon=epsilon,
        epsilon_source=config.checkpoint_acceptance_epsilon_source,
        calibration_repeats=config.checkpoint_acceptance_calibration_repeats)
    event["repair_attempt_count"] = 1
    best_id, best_score, best_trial, best_tables, best_observation = None, math.inf, None, [], None
    best_group_cosine = None
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
                         partial_best_D_win=best_score if best_id is not None else None)
            return finish("candidate_forward_failed")
        if trial_observation["invalid_reason"] is not None:
            event.update(failed_id=token, unevaluated_ids=eligible[index+1:],
                         partial_best_D_win=best_score if best_id is not None else None)
            return finish("invalid_candidate_score")
        event["evaluated_ids"].append(token)
        event["candidate_group_cosines"].append(score)
        event["candidate_paths"].append(dict(seed_token_id=token,
            segment_tokens=list(trial[a:b+1]), group_cosine=score,
            pointwise_deviation_change=[after-before for before, after in zip(
                observation["pointwise_deviation"], trial_observation["pointwise_deviation"])],
            **trial_observation,
            downstream_candidate_tables=trial_tables))
        if trial_observation["D_win"] < best_score:
            best_id, best_score = token, trial_observation["D_win"]
            best_group_cosine, best_observation = score, trial_observation
            best_trial, best_tables = list(trial), trial_tables
    event.update(all_candidates_scored=True, best_group_cosine=best_group_cosine, best_D_win=best_score)
    if best_score >= event["acceptance_threshold"]:
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
    event.update(accepted=True, new_token_id=best_id, group_cosine_after=best_group_cosine,
                 D_win_after=best_score,
                 pointwise_deviation_after=best_observation["pointwise_deviation"],
                 pointwise_deviation_change=[after-before for before, after in zip(
                     observation["pointwise_deviation"], best_observation["pointwise_deviation"])],
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
                    right_range, lr, epoch, alpha, clip=True, optim_method="cosine", trajectory_callback=None):
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
        if trajectory_callback is not None:
            observed = torch.cat((prefix_embedding, variable), dim=1) if prefix_embedding is not None else variable
            trajectory_callback(completed, observed)
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
    config = stage2_kwargs["config"]
    collect_stages = config.checkpoint_enabled and config.checkpoint_schema_version == 5
    if collect_stages and not config.checkpoint_calibration_id:
        raise CheckpointContractError("calibrate schema 5 before any experiment sample")
    if snapshot_mode == "read":
        # Only local snapshots just produced by this runner are accepted.
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot["pair_id"] != pair_id or snapshot["contract"] != snapshot_contract:
            raise ValueError("Stage-1 snapshot identity/config mismatch")
        if not torch.equal(target.cpu(), snapshot["target"]):
            raise ValueError("paired target observation differs")
        embedding = snapshot["embedding"].to(stage1_kwargs["initial_embedding"].device)
        summary = snapshot["stage1"]
        if collect_stages and "stage_candidates" not in snapshot:
            raise ValueError("snapshot lacks schema-5 trajectory candidates; rerun Stage-1")
        stage_candidates = snapshot.get("stage_candidates", {})
        restore_rng(snapshot["rng"])
    else:
        collector = (StageCandidateCollector(stage1_kwargs["epoch"], stage2_kwargs.get("eval_start_pos", 0),
            stage2_kwargs["embed_layer"], stage2_kwargs.get("invert_method", "cosine"),
            "stage1", stage2_kwargs.get("fixed_prefix_tokens") or []) if collect_stages else None)
        arguments = dict(stage1_kwargs)
        if collector is not None:
            arguments["trajectory_callback"] = collector
        embedding, summary = optimize_stage1(**arguments)
        stage_candidates = collector.finalize(summary["completed_steps"], embedding.shape[1]) if collector else {}
        if collector:
            summary["checkpoint_trajectory"] = collector.metadata()
        snapshot = dict(pair_id=pair_id, contract=snapshot_contract, embedding=embedding.cpu(),
                        target=target.cpu(), stage1=summary, rng=rng_state(), stage_candidates=stage_candidates)
        if snapshot_mode == "write":
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            torch.save(snapshot, path)
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest() if path else hashlib.sha256(
        embedding.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    stage2_kwargs = dict(stage2_kwargs, optimized_embedding=embedding.detach().clone(),
                         target_hidden_state=target.clone(), stage_candidates=stage_candidates)
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
