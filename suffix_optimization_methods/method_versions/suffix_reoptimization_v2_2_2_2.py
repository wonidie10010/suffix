"""v2.2.2(2): expand every discretization's candidates, without checkpoints.

Stage-1 and R are copied from v2.2.2. Private labels remain outside the sidecar.
Snapshots preserve the original v2.2.2 identity contract and trajectory format.
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

METHOD_NAME = "suffix_reoptimization_v2.2.2(2)"
VERSION = "v2.2.2(2)"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192


@dataclass
class SuffixReoptimizationV222_2Config:
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
    expansion_policy: str = "checkpoint_sources_2_plus_4"

    def __post_init__(self):
        if self.checkpoint_enabled is not False:
            raise ValueError("v2.2.2(2) has no checkpoints")
        if self.expansion_policy != "checkpoint_sources_2_plus_4":
            raise ValueError("unsupported expansion policy")
        for name in ("enabled", "log_enabled", "filter_nonascii"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_2_2_2_{} must be boolean".format(name))
        for name in (
            "max_attempts",
            "max_attempts_per_position",
            "steps",
            "range_top_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_2_2_2_{} must be an integer".format(name))
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
                raise TypeError("suffix_v2_2_2_2_{} must be numeric".format(name))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("suffix_v2_2_2_2_{} must be finite".format(name))
            setattr(self, name, value)
        if self.max_attempts < 0:
            raise ValueError("suffix_v2_2_2_2_max_attempts must be non-negative")
        if self.max_attempts_per_position <= 0:
            raise ValueError(
                "suffix_v2_2_2_2_max_attempts_per_position must be positive"
            )
        if self.steps <= 0:
            raise ValueError("suffix_v2_2_2_2_steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("suffix_v2_2_2_2_lr must be positive")
        if not 0.0 <= self.hidden_weight_decay <= 1.0:
            raise ValueError(
                "suffix_v2_2_2_2_hidden_weight_decay must be in [0, 1]"
            )
        if self.hidden_weight_floor < 0.0:
            raise ValueError(
                "suffix_v2_2_2_2_hidden_weight_floor must be non-negative"
            )
        if self.prox_weight < 0.0 or self.range_weight < 0.0:
            raise ValueError(
                "suffix_v2_2_2 regularization weights must be non-negative"
            )
        if self.range_top_k <= 0:
            raise ValueError("suffix_v2_2_2_2_range_top_k must be positive")
        if self.trigger_mode not in {"always", "threshold"}:
            raise ValueError(
                "suffix_v2_2_2_2_trigger_mode must be always or threshold"
            )
        if self.hidden_weight_mode != "front_decay":
            raise ValueError(
                "suffix_v2_2_2_2_hidden_weight_mode must be front_decay"
            )
        if self.accept_mode != "hidden_loss":
            raise ValueError(
                "suffix_v2_2_2_2_accept_mode must be hidden_loss"
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
        forward_and_get_last_hidden_state, rerank_end=None, forward_stats=None,
        stage_candidates=None, expansion_events=None, expansion_phase="initial"):
    """Preserve original candidates, then score up to six new IDs without labels."""
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
        original_table = dict(candidate_token_ids=list(top_list),
            candidate_hidden_cosine=[float(v) for v in cosine.detach().cpu().tolist()],
            selected_token_id=ret_list[position], prefix_fingerprint=prefix_fingerprint(ret_list[:position]))
        if not torch.isfinite(cosine).all():
            raise CandidateExpansionContractError("nonfinite original candidate score")
        pool = build_checkpoint_candidates(original_table, ret_list[position], tokenizer,
            embed_layer.num_embeddings, filter_nonascii, (stage_candidates or {}).get(position))
        old_ids = set(top_list)
        additions = [row for row in pool["candidates"]
                     if row["quota_source"] != "old" and row["token_id"] not in old_ids]
        extra_ids = [row["token_id"] for row in additions]
        if len(extra_ids) > 6 or len(extra_ids) != len(set(extra_ids)):
            raise CandidateExpansionContractError("invalid expansion quota")
        original_winner = ret_list[position]
        best_score = float(cosine[best_index].detach())
        extra_scores = []
        if extra_ids:
            trials = []
            for token in extra_ids:
                trial = list(ret_list)
                trial[position] = token
                trials.append(trial)
            extra_hidden = forward_and_get_last_hidden_state(model, trials, None, layer_id=layer_id)
            extra_target = target_hidden_state[:, position, :].to(extra_hidden.device).float()
            scores = F.cosine_similarity(extra_hidden[:, position, :].float(), extra_target, dim=-1)
            extra_scores = [float(v) for v in scores.detach().cpu().tolist()]
            # Strict greater-than preserves the original winner on ties.
            for token, score in zip(extra_ids, extra_scores):
                if math.isfinite(score) and score > best_score:
                    ret_list[position], best_score = token, score
        event = dict(phase=expansion_phase, position=position,
            prefix_fingerprint=prefix_fingerprint(ret_list[:position]),
            original_candidate_ids=list(top_list), original_selected_token_id=original_winner,
            added_candidate_ids=extra_ids, added_candidates=additions,
            added_hidden_cosine=extra_scores, selected_token_id=ret_list[position],
            candidate_pool=pool, retained_in_formal_state=expansion_phase == "initial",
            added_forward_calls=int(bool(extra_ids)), added_forward_token_count=len(extra_ids)*sequence_length)
        if expansion_events is not None:
            expansion_events.append(event)
        top_list.extend(extra_ids)
        cosine_values = original_table["candidate_hidden_cosine"] + extra_scores
        diagnostics.append({
            "prefix_fingerprint": prefix_fingerprint(ret_list[:position]),
            "position": int(position),
            "candidate_token_ids": [int(item) for item in top_list],
            "candidate_hidden_cosine": [
                float(item) for item in cosine_values
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


def run_suffix_reoptimization_v2_2_2_2(
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

    expansion_events = []
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
        stage_candidates=stage_candidates, expansion_events=expansion_events,
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

    for position in range(int(eval_start_pos), sequence_length):
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
            continue

        attempts += 1
        per_position_attempts[position] = position_attempts + 1
        try:
            trajectory = (StageCandidateCollector(
                config.steps, position, embed_layer, invert_method,
                "R_{}".format(attempts), current_tokens[:position])
                )
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
            trial_stage_candidates = (trajectory.finalize(summary.get("completed_steps", 0), sequence_length)
                                      if trajectory is not None else {})
            if trajectory is not None:
                summary["checkpoint_trajectory"] = trajectory.metadata()
            expansion_begin = len(expansion_events)
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
                stage_candidates=trial_stage_candidates, expansion_events=expansion_events,
                expansion_phase="R_{}".format(attempts),
            )
            accepted = bool(
                math.isfinite(float(optimized_pre_loss))
                and math.isfinite(float(optimized_post_loss))
                and float(optimized_post_loss) < float(optimized_pre_loss)
            )
            for expansion_event in expansion_events[expansion_begin:]:
                expansion_event["retained_in_formal_state"] = accepted
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
        "candidate_expansion": {
            "enabled": True, "policy": config.expansion_policy, "events": expansion_events,
            "added_forward_calls": sum(e["added_forward_calls"] for e in expansion_events),
            "added_forward_token_count": sum(e["added_forward_token_count"] for e in expansion_events),
        },
        "anomaly_reasons": [e["reason"] for e in events if e["reason"].startswith("trial_failed:")],
        "initial_candidate_rerank": initial_rerank,
        "events": events,
    }


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


class CandidateExpansionContractError(ValueError):
    """A broken adapter contract cannot be treated as a recoverable candidate."""


def raise_if_fatal(error):
    text = str(error).lower()
    if isinstance(error, (CandidateExpansionContractError, MemoryError, torch.cuda.OutOfMemoryError)) or any(
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


def prefix_fingerprint(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


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


def empty_result(tokens, config, start):
    return dict(name=METHOD_NAME, method=METHOD_NAME, version=VERSION, enabled=True,
                formal_gt_blind=True, gt_accessed=False, pre_tokens=list(tokens),
                final_tokens=list(tokens), final_text="", eval_start_pos=start,
                events=[], initial_candidate_rerank=[], attempt_count=0,
                candidate_expansion=dict(enabled=True, events=[], added_forward_calls=0,
                                         added_forward_token_count=0), anomaly_reasons=[])



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
    collect_stages = True
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
    final, result = run_suffix_reoptimization_v2_2_2_2(**stage2_kwargs)
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
    elapsed = time.perf_counter()-started
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

def config_from_mapping(values):
    prefixes = {"enabled": "suffix_reoptimization_v2_2_2_2",
                "log_enabled": "suffix_reoptimization_v2_2_2_2_log"}
    kwargs = {}
    for field in SuffixReoptimizationV222_2Config.__dataclass_fields__:
        key = prefixes.get(field, "suffix_v2_2_2_2_" + field)
        if key not in values:
            raise ValueError("missing explicit v2.2.2(2) configuration: " + key)
        kwargs[field] = values[key]
    return SuffixReoptimizationV222_2Config(**kwargs)
