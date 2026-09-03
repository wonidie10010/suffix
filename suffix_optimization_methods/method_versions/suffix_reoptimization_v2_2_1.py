"""Minimal, target-token-blind suffix reoptimization v2.2.1.

The sidecar keeps the Original DEML candidate policy and adds only an
anchored suffix trial.  Ground-truth token ids are intentionally absent from
the public runner signature.  Accuracy, when needed by an experiment, is
attached by the caller after this sidecar has finished.
"""

from dataclasses import dataclass
import math
import numbers

import torch
import torch.nn.functional as F


METHOD_NAME = "suffix_reoptimization_v2.2.1"
VERSION = "v2.2.1"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192

__all__ = [
    "SuffixReoptimizationV221Config",
    "run_suffix_reoptimization_v2_2_1",
]


@dataclass
class SuffixReoptimizationV221Config:
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

    def __post_init__(self):
        for name in ("enabled", "log_enabled", "filter_nonascii"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_2_1_{} must be boolean".format(name))
        for name in (
            "max_attempts",
            "max_attempts_per_position",
            "steps",
            "range_top_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_2_1_{} must be an integer".format(name))
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
                raise TypeError("suffix_v2_2_1_{} must be numeric".format(name))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("suffix_v2_2_1_{} must be finite".format(name))
            setattr(self, name, value)
        if self.max_attempts < 0:
            raise ValueError("suffix_v2_2_1_max_attempts must be non-negative")
        if self.max_attempts_per_position <= 0:
            raise ValueError(
                "suffix_v2_2_1_max_attempts_per_position must be positive"
            )
        if self.steps <= 0:
            raise ValueError("suffix_v2_2_1_steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("suffix_v2_2_1_lr must be positive")
        if not 0.0 <= self.hidden_weight_decay <= 1.0:
            raise ValueError(
                "suffix_v2_2_1_hidden_weight_decay must be in [0, 1]"
            )
        if self.hidden_weight_floor < 0.0:
            raise ValueError(
                "suffix_v2_2_1_hidden_weight_floor must be non-negative"
            )
        if self.prox_weight < 0.0 or self.range_weight < 0.0:
            raise ValueError(
                "suffix_v2_2_1 regularization weights must be non-negative"
            )
        if self.range_top_k <= 0:
            raise ValueError("suffix_v2_2_1_range_top_k must be positive")
        if self.trigger_mode not in {"always", "threshold"}:
            raise ValueError(
                "suffix_v2_2_1_trigger_mode must be always or threshold"
            )
        if self.hidden_weight_mode != "front_decay":
            raise ValueError(
                "suffix_v2_2_1_hidden_weight_mode must be front_decay"
            )
        if self.accept_mode != "hidden_loss":
            raise ValueError(
                "suffix_v2_2_1_accept_mode must be hidden_loss"
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
            hidden_state_list.extend(output)
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
        forward_and_get_last_hidden_state):
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
    new_input_embed_squeeze = input_embed.squeeze(0)
    ret_top_k = {}
    for position in range(rerank_start, sequence_length):
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
    for position in range(rerank_start, sequence_length):
        top_list = list(ret_top_k.get(position, [ret_list[position]]))
        if position > 0 and add_perplexity:
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


def run_suffix_reoptimization_v2_2_1(
        model, embed_layer, optimized_embedding, target_hidden_state,
        attention_mask, layer_id, register_layer_hooks, tokenizer, config,
        fixed_prefix_tokens=None, eval_start_pos=0, filter_nonascii=True,
        add_perplexity=True, top_k_ppl=10, top_k_cos=10,
        invert_method="cosine", embedding_top_indices=None,
        select_candidate_from_top_indices=None, get_perplexity=None,
        forward_and_get_last_hidden_state=None, log_file=None):
    """Run v2.2.1 without exposing target token ids to the method."""
    del log_file
    embedding = torch.as_tensor(optimized_embedding).detach().clone()
    fixed_prefix_tokens = [int(value) for value in (fixed_prefix_tokens or [])]
    if not config.enabled or config.max_attempts <= 0:
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
                accepted_count += 1
            else:
                rejected_count += 1
        except Exception as error:
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
        "initial_candidate_rerank": initial_rerank,
        "events": events,
    }


SuffixReoptimizationConfig = SuffixReoptimizationV221Config
run_suffix_reoptimization = run_suffix_reoptimization_v2_2_1
