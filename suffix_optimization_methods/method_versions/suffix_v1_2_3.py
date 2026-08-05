from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F

from suffix_optimization_methods.method_versions import (
    suffix_reoptimization_v1_2_2 as v122,
)


METHOD_NAME = "suffix_v1.2.3"
VERSION = "v1.2.3"
REOPTIMIZATION_SOURCE = "suffix_reoptimization_v1.2.2_unchanged"
RELATIVE_MSE_EPSILON = v122.RELATIVE_MSE_EPSILON
EMBEDDING_SEARCH_CHUNK_SIZE = 8192
METRIC_DIRECTION_NOTE = (
    "embedding MSE, relative MSE, joint loss, and total loss are "
    "lower-is-better; accuracy is higher-is-better; MSE values from "
    "different definitions or spaces must not be compared directly"
)


@dataclass
class SuffixV123Config:
    enabled: bool = False
    log_enabled: bool = True
    max_rounds: int = 2
    epoch: int = 50
    lr: float = 0.03
    embedding_relative_mse_high_threshold: float = 1.0
    relative_mse_rise_threshold: float = 0.30
    token_relative_mse_high_threshold: float = 1.0
    min_anomaly_reasons: int = 1
    min_relative_mse_improvement: float = 0.01
    accuracy_tolerance: float = 0.0
    accept_mode: str = "oracle_accuracy"
    anomaly_detection_mode: str = "adaptive"
    adaptive_z_threshold: float = 1.5
    adaptive_rise_z_threshold: float = 1.5
    adaptive_min_std: float = 1e-6
    adaptive_min_points: int = 4
    hidden_weight_mode: str = "front_decay"
    hidden_weight_decay: float = 0.90
    hidden_weight_floor: float = 0.20
    cosine_loss_weight: float = 0.1
    relative_mse_loss_weight: float = 0.9
    prox_weight: float = 0.005
    range_weight: float = 0.001
    gradient_trend_stats_enabled: bool = True


_relative_mse = v122._relative_mse
_find_anomalies = v122._find_anomalies
_accept_candidate = v122._accept_candidate
_optimize_suffix = v122._optimize_suffix
_evaluate_state = v122._evaluate_state
_public_metrics = v122._public_metrics


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _safe_mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_max(values):
    if not values:
        return None
    return float(max(values))


def _dedupe(token_ids):
    seen = set()
    result = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        result.append(token_id)
    return result


def _target_ids(total_input_ids):
    return [int(item) for item in total_input_ids[0].detach().cpu().tolist()]


def _accuracy(token_ids, total_input_ids, eval_start_pos):
    targets = _target_ids(total_input_ids)
    start = min(max(int(eval_start_pos), 0), len(targets))
    if start >= len(targets):
        return 0.0
    correct = sum(
        int(token_ids[position]) == targets[position]
        for position in range(start, len(targets))
    )
    return correct / (len(targets) - start)


def _decode(tokenizer, token_ids, eval_start_pos):
    return tokenizer.decode(torch.tensor(token_ids[eval_start_pos:]))


def _valid_position_mask(
        sequence_length,
        eval_start_pos,
        attention_mask,
        device):
    mask = torch.ones(
        int(sequence_length),
        dtype=torch.bool,
        device=device,
    )
    start = min(max(int(eval_start_pos), 0), int(sequence_length))
    mask[:start] = False
    if attention_mask is not None:
        attention = attention_mask.detach().to(device=device, dtype=torch.bool)
        if attention.ndim == 2:
            attention = attention[0]
        if attention.ndim != 1 or int(attention.shape[0]) != int(sequence_length):
            raise ValueError(
                "attention mask must match the stage1 sequence length"
            )
        mask &= attention
    return mask


def _masked_values(values, valid_mask):
    mask = valid_mask.detach().to("cpu", dtype=torch.bool).tolist()
    return [
        value
        for position, value in enumerate(values)
        if position < len(mask) and mask[position]
    ]


def _weighted_relative_mse(
        current_hidden,
        target_hidden,
        valid_mask=None,
        position_weights=None):
    values = _relative_mse(current_hidden, target_hidden)
    if values.ndim == 2:
        values = values[0]
    if valid_mask is None:
        valid_mask = torch.ones_like(values, dtype=torch.bool)
    valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
    selected = values[valid_mask]
    if selected.numel() == 0:
        raise ValueError("stage1 has no valid optimization positions")
    if position_weights is None:
        weights = torch.ones_like(selected, dtype=torch.float32)
    else:
        weights = position_weights.to(
            device=values.device,
            dtype=torch.float32,
        )
        if weights.ndim == 2:
            weights = weights[0]
        weights = weights[valid_mask]
    return (selected.float() * weights).sum() / weights.sum().clamp_min(1e-12)


def _forward_embedding_hidden(model, input_embed, attention_mask, layer_id,
                              register_layer_hooks):
    return v122._forward_embedding_hidden(
        model,
        input_embed,
        attention_mask,
        layer_id,
        register_layer_hooks,
    )


def _embedding_mse_candidates(
        embed,
        embed_layer,
        top_k,
        tokenizer,
        filter_nonascii=True,
        chunk_size=EMBEDDING_SEARCH_CHUNK_SIZE):
    """Return the nearest legal vocabulary tokens under raw float32 MSE."""
    weight = embed_layer.weight.detach()
    vocab_size = int(weight.shape[0])
    requested_k = max(0, int(top_k))
    if vocab_size <= 0:
        raise ValueError("empty embedding vocabulary")
    if requested_k == 0:
        return [], {
            "top1_mse": None,
            "top_k_token_ids": [],
            "top_k_mse": [],
            "valid_candidate_count": 0,
            "requested_top_k": 0,
        }

    embed_cpu = embed.detach().to("cpu", dtype=torch.float32)
    distance_chunks = []
    with torch.no_grad():
        for start in range(0, vocab_size, max(1, int(chunk_size))):
            end = min(start + max(1, int(chunk_size)), vocab_size)
            chunk = weight[start:end].detach().to(
                "cpu",
                dtype=torch.float32,
            )
            distance_chunks.append(
                (chunk - embed_cpu).pow(2).mean(dim=-1)
            )
    distances = torch.cat(distance_chunks)
    try:
        ordered_ids = torch.argsort(distances, stable=True)
    except TypeError:
        ordered_ids = torch.argsort(distances)

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    keep_k = min(requested_k, vocab_size)
    valid_ids = []
    valid_distances = []
    for token_tensor in ordered_ids:
        token_id = int(token_tensor)
        if token_id in special_ids:
            continue
        if (
            filter_nonascii
            and not tokenizer.decode([token_id]).isascii()
        ):
            continue
        valid_ids.append(token_id)
        valid_distances.append(float(distances[token_id]))
        if len(valid_ids) >= keep_k:
            break

    if not valid_ids:
        fallback = int(ordered_ids[0])
        valid_ids = [fallback]
        valid_distances = [float(distances[fallback])]
    diagnostics = {
        "top1_mse": valid_distances[0],
        "top_k_token_ids": list(valid_ids),
        "top_k_mse": list(valid_distances),
        "valid_candidate_count": len(valid_ids),
        "requested_top_k": requested_k,
    }
    return valid_ids, diagnostics


def _stage1_rerank_positions(
        input_embed,
        tokenizer,
        model,
        embed_layer,
        target_hidden_state,
        total_input_ids,
        layer_id,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_embedding,
        eval_start_pos,
        get_perplexity,
        forward_and_get_last_hidden_state,
        attention_mask=None):
    sequence_length = int(input_embed.shape[1])
    target_ids = _target_ids(total_input_ids)
    valid_mask = _valid_position_mask(
        sequence_length,
        eval_start_pos,
        attention_mask,
        input_embed.device,
    ).detach().to("cpu")
    ret_list = [0 for _ in range(sequence_length)]
    for position in range(min(int(eval_start_pos), sequence_length)):
        ret_list[position] = target_ids[position]

    embedding_candidates = {}
    embedding_diagnostics = {}
    provisional_tokens = {}
    squeezed = input_embed.squeeze(0)
    for position in range(max(0, int(eval_start_pos)), sequence_length):
        candidates, diagnostics = _embedding_mse_candidates(
            squeezed[position],
            embed_layer,
            top_k_embedding,
            tokenizer,
            filter_nonascii,
        )
        if not candidates:
            candidates = [ret_list[position]]
        embedding_candidates[position] = candidates
        embedding_diagnostics[position] = diagnostics
        ret_list[position] = int(candidates[0])
        provisional_tokens[position] = ret_list[position]

    candidate_diagnostics = []
    changed_positions = []
    for position in range(max(0, int(eval_start_pos)), sequence_length):
        if not bool(valid_mask[position]):
            continue
        embedding_ids = list(embedding_candidates[position])
        current_token = int(ret_list[position])
        joint_candidates = _dedupe(embedding_ids + [current_token])
        if position > 0 and add_perplexity:
            _, ppl_ids = get_perplexity(
                ret_list[:position],
                model,
                layer_id=layer_id,
                top_k=top_k_ppl,
            )
            joint_candidates = _dedupe(
                joint_candidates
                + [int(item) for item in ppl_ids.detach().cpu().tolist()]
            )
        if not joint_candidates:
            joint_candidates = [current_token]

        replaced_sequences = []
        for token_id in joint_candidates:
            replaced = copy.deepcopy(ret_list)
            replaced[position] = int(token_id)
            replaced_sequences.append(replaced)
        hidden_states = forward_and_get_last_hidden_state(
            model,
            replaced_sequences,
            None,
            layer_id=layer_id,
        )
        target_hidden = target_hidden_state.to(hidden_states.device)
        candidate_states = hidden_states[:, position, :]
        target_state = target_hidden[:, position, :]
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        hidden_relative_mse = _relative_mse(
            candidate_states,
            target_state,
        )
        best_index = int(torch.argmin(hidden_relative_mse).detach().cpu())
        order = torch.argsort(hidden_relative_mse)
        best_mse = _as_float(hidden_relative_mse[order[0]])
        second_best_mse = (
            _as_float(hidden_relative_mse[order[1]])
            if int(order.numel()) > 1 else None
        )
        selected_token = int(joint_candidates[best_index])
        ret_list[position] = selected_token
        if selected_token != provisional_tokens[position]:
            changed_positions.append(position)

        target_token = target_ids[position]
        joint_rank = (
            joint_candidates.index(target_token) + 1
            if target_token in joint_candidates else None
        )
        embedding_info = embedding_diagnostics[position]
        candidate_diagnostics.append({
            "position": position,
            "embedding_mse_top1_distance": embedding_info["top1_mse"],
            "embedding_mse_top_k_token_ids": list(embedding_ids),
            "embedding_mse_top_k_distances": list(
                embedding_info["top_k_mse"]
            ),
            "embedding_mse_valid_candidate_count": embedding_info[
                "valid_candidate_count"
            ],
            "candidate_token_ids": list(joint_candidates),
            "candidate_count": len(joint_candidates),
            "candidate_hidden_relative_mse": [
                float(item)
                for item in hidden_relative_mse.detach().cpu().tolist()
            ],
            "selected_token_id": selected_token,
            "selected_hidden_relative_mse": _as_float(
                hidden_relative_mse[best_index]
            ),
            "best_mse": best_mse,
            "second_best_mse": second_best_mse,
            "mse_margin": (
                None
                if second_best_mse is None
                else second_best_mse - best_mse
            ),
            "oracle_stage1_embedding_candidate_hit": (
                target_token in embedding_ids
            ),
            "oracle_stage1_joint_candidate_hit": (
                target_token in joint_candidates
            ),
            "oracle_stage1_selected_token_correct": (
                selected_token == target_token
            ),
            "oracle_stage1_ground_truth_candidate_rank": joint_rank,
        })
    return (
        _accuracy(ret_list, total_input_ids, eval_start_pos),
        _decode(tokenizer, ret_list, eval_start_pos),
        ret_list,
        candidate_diagnostics,
        changed_positions,
    )


def _stage1_optimize(
        model,
        initial_optimizable_embedding,
        prefix_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        right_range,
        lr,
        epoch,
        range_weight,
        clip,
        eval_start_pos,
        gradient_trend_stats_enabled):
    optimizable = initial_optimizable_embedding
    sequence_length = int(target_hidden_state.shape[1])
    valid_mask = _valid_position_mask(
        sequence_length,
        eval_start_pos,
        attention_mask,
        optimizable.device,
    )
    valid_position_count = int(valid_mask.sum().detach().cpu())
    tracker = v122.BaselineGradientTrendTracker(
        enabled=gradient_trend_stats_enabled,
        position_offset=eval_start_pos,
    )
    relative_mse_history = []
    range_history = []
    total_history = []
    stopped_reason = "completed"
    nan_detected = False
    completed_steps = 0

    for _ in range(max(0, int(epoch))):
        if clip:
            with torch.no_grad():
                optimizable = torch.clip(optimizable, -0.2, 0.2)
        optimizable = optimizable.requires_grad_(True)
        optimizer = torch.optim.SGD([optimizable], lr=float(lr))
        current_embedding = (
            torch.cat((prefix_embedding, optimizable), dim=1)
            if prefix_embedding is not None
            else optimizable
        )
        hidden_state = _forward_embedding_hidden(
            model,
            current_embedding,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        hidden_loss = _weighted_relative_mse(
            hidden_state,
            target_hidden,
            valid_mask=valid_mask.to(hidden_state.device),
        )
        range_loss = F.relu(
            torch.abs(current_embedding) - right_range
        ).sum()
        total_loss = hidden_loss + float(range_weight) * range_loss
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            nan_detected = True
            stopped_reason = "nonfinite_loss"
            break
        optimizer.zero_grad()
        total_loss.backward(inputs=[optimizable])
        tracker.observe(optimizable.grad)
        optimizer.step()
        completed_steps += 1
        relative_mse_history.append(_as_float(hidden_loss))
        range_history.append(_as_float(range_loss))
        total_history.append(_as_float(total_loss))

    final_embedding = (
        torch.cat((prefix_embedding, optimizable), dim=1)
        if prefix_embedding is not None
        else optimizable
    ).detach()
    summary = {
        "metric": "relative_mse",
        "relative_mse_epsilon": RELATIVE_MSE_EPSILON,
        "relative_mse_loss_start": (
            relative_mse_history[0] if relative_mse_history else None
        ),
        "relative_mse_loss_end": (
            relative_mse_history[-1] if relative_mse_history else None
        ),
        "relative_mse_loss_min": (
            min(relative_mse_history) if relative_mse_history else None
        ),
        "range_loss_start": range_history[0] if range_history else None,
        "range_loss_end": range_history[-1] if range_history else None,
        "total_loss_start": total_history[0] if total_history else None,
        "total_loss_end": total_history[-1] if total_history else None,
        "total_loss_min": min(total_history) if total_history else None,
        "valid_position_count": valid_position_count,
        "optimization_steps": completed_steps,
        "configured_epoch": int(epoch),
        "optimizer": "SGD",
        "optimizer_recreated_each_epoch": True,
        "lr": float(lr),
        "range_weight": float(range_weight),
        "clip": bool(clip),
        "clip_range": 0.2 if clip else None,
        "stopped_reason": stopped_reason,
        "nan_detected": nan_detected,
    }
    return final_embedding, summary, tracker.summary()


def _v122_config(config):
    return v122.SuffixReoptimizationV122Config(
        enabled=config.enabled,
        log_enabled=config.log_enabled,
        max_rounds=config.max_rounds,
        epoch=config.epoch,
        lr=config.lr,
        embedding_relative_mse_high_threshold=(
            config.embedding_relative_mse_high_threshold
        ),
        relative_mse_rise_threshold=config.relative_mse_rise_threshold,
        token_relative_mse_high_threshold=(
            config.token_relative_mse_high_threshold
        ),
        min_anomaly_reasons=config.min_anomaly_reasons,
        min_relative_mse_improvement=config.min_relative_mse_improvement,
        accuracy_tolerance=config.accuracy_tolerance,
        accept_mode=config.accept_mode,
        anomaly_detection_mode=config.anomaly_detection_mode,
        adaptive_z_threshold=config.adaptive_z_threshold,
        adaptive_rise_z_threshold=config.adaptive_rise_z_threshold,
        adaptive_min_std=config.adaptive_min_std,
        adaptive_min_points=config.adaptive_min_points,
        hidden_weight_mode=config.hidden_weight_mode,
        hidden_weight_decay=config.hidden_weight_decay,
        hidden_weight_floor=config.hidden_weight_floor,
        cosine_loss_weight=config.cosine_loss_weight,
        relative_mse_loss_weight=config.relative_mse_loss_weight,
        prox_weight=config.prox_weight,
        range_weight=config.range_weight,
        gradient_trend_stats_enabled=config.gradient_trend_stats_enabled,
    )


def _run_reoptimization_v1_2_2(
        model,
        embed_layer,
        stage1_embedding,
        stage1_tokens,
        stage1_text,
        stage1_accuracy,
        stage1_candidate_diagnostics,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        tokenizer,
        total_input_ids,
        config,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        invert_method,
        eval_start_pos,
        embedding_top_indices,
        select_candidate_from_top_indices,
        get_perplexity,
        forward_and_get_last_hidden_state,
        stage1_gradient_trend_stats):
    scan_start = max(int(eval_start_pos), 1)
    current_embedding = stage1_embedding.detach().clone()
    current_tokens = [int(item) for item in stage1_tokens]
    current_metrics = v122._evaluate_state(
        model,
        current_embedding,
        current_tokens,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        total_input_ids,
        tokenizer,
        eval_start_pos,
        scan_start,
        config,
        forward_and_get_last_hidden_state,
    )
    current_metrics["accuracy"] = stage1_accuracy
    current_metrics["text"] = stage1_text
    before_metrics = current_metrics
    events = []
    all_changed_positions = set()
    scan_pos = scan_start
    attempts = 0
    max_rounds = max(0, int(config.max_rounds))
    accept_mode = v122._accept_mode(config)

    while attempts < max_rounds:
        metrics_for_scan = v122._evaluate_state(
            model,
            current_embedding,
            current_tokens,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            total_input_ids,
            tokenizer,
            eval_start_pos,
            scan_pos,
            config,
            forward_and_get_last_hidden_state,
        )
        metrics_for_scan["accuracy"] = current_metrics["accuracy"]
        metrics_for_scan["text"] = current_metrics["text"]
        anomalies = metrics_for_scan["_anomalies"]
        if not anomalies:
            if attempts == 0:
                events.append({
                    "round": 1,
                    "max_rounds": max_rounds,
                    "triggered": False,
                    "accept_mode": accept_mode,
                    "reason": "no anomaly found",
                })
            break

        anomaly = anomalies[0]
        suffix_start = int(anomaly["position"])
        attempts += 1
        candidate_embedding, loss_summary = v122._optimize_suffix(
            model,
            current_embedding,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            suffix_start,
            config,
            embed_layer,
            invert_method,
        )
        (
            candidate_accuracy,
            candidate_text,
            candidate_tokens,
            candidate_rerank,
        ) = v122._rerank_positions(
            candidate_embedding,
            current_tokens,
            suffix_start,
            tokenizer,
            model,
            embed_layer,
            target_hidden_state,
            total_input_ids,
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
        candidate_metrics = v122._evaluate_state(
            model,
            candidate_embedding,
            candidate_tokens,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            total_input_ids,
            tokenizer,
            eval_start_pos,
            scan_start,
            config,
            forward_and_get_last_hidden_state,
        )
        candidate_metrics["accuracy"] = candidate_accuracy
        candidate_metrics["text"] = candidate_text
        changed_positions = v122._changed_positions(
            current_tokens,
            candidate_tokens,
            suffix_start,
        )
        accepted, accept_reason, relative_mse_improved = (
            v122._accept_candidate(
                current_metrics,
                candidate_metrics,
                changed_positions,
                suffix_start,
                config,
            )
        )
        event = {
            "round": attempts,
            "max_rounds": max_rounds,
            "triggered": True,
            "anomaly_position": suffix_start,
            "anomaly_reasons": anomaly.get("reasons"),
            "anomaly_detection_mode": anomaly.get(
                "anomaly_detection_mode"
            ),
            "anomaly_score": anomaly.get("anomaly_score"),
            "anomaly_embedding_forward_relative_mse": anomaly.get(
                "embedding_forward_relative_mse"
            ),
            "anomaly_token_forward_relative_mse": anomaly.get(
                "token_forward_relative_mse"
            ),
            "relative_mse_rise": anomaly.get("relative_mse_rise"),
            "before_accuracy": current_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "before_token_forward_relative_mse_mean": current_metrics[
                "token_forward_relative_mse_mean"
            ],
            "candidate_token_forward_relative_mse_mean": candidate_metrics[
                "token_forward_relative_mse_mean"
            ],
            "before_token_forward_relative_mse_max": current_metrics[
                "token_forward_relative_mse_max"
            ],
            "candidate_token_forward_relative_mse_max": candidate_metrics[
                "token_forward_relative_mse_max"
            ],
            "before_suffix_token_forward_relative_mse_mean": (
                v122._suffix_token_forward_relative_mse_mean(
                    current_metrics,
                    suffix_start,
                )
            ),
            "candidate_suffix_token_forward_relative_mse_mean": (
                v122._suffix_token_forward_relative_mse_mean(
                    candidate_metrics,
                    suffix_start,
                )
            ),
            "before_relative_mse_anomaly_count": current_metrics[
                "relative_mse_anomaly_count"
            ],
            "candidate_relative_mse_anomaly_count": candidate_metrics[
                "relative_mse_anomaly_count"
            ],
            "relative_mse_improved": relative_mse_improved,
            "candidate_rerank": candidate_rerank,
            "changed_positions": changed_positions,
            "accepted": accepted,
            "accept_mode": accept_mode,
            "accept_reason": accept_reason,
            "loss_summary": loss_summary,
        }
        events.append(event)
        if accepted:
            current_embedding = candidate_embedding
            current_tokens = candidate_tokens
            current_metrics = candidate_metrics
            all_changed_positions.update(changed_positions)
            scan_pos = scan_start
        else:
            scan_pos = suffix_start + 1

    after_metrics = current_metrics
    triggered = any(event.get("triggered") for event in events)
    accepted_count = sum(bool(event.get("accepted")) for event in events)
    rejected_count = sum(
        bool(event.get("triggered")) and not bool(event.get("accepted"))
        for event in events
    )
    if not triggered:
        reason = "no anomaly found"
    elif accepted_count:
        reason = "accepted {} suffix round(s)".format(accepted_count)
    else:
        reason = "no suffix round accepted"
    return current_embedding, {
        "accept_mode": accept_mode,
        "pre_acc": before_metrics["accuracy"],
        "post_acc": after_metrics["accuracy"],
        "accuracy_gain": (
            after_metrics["accuracy"] - before_metrics["accuracy"]
        ),
        "triggered": triggered,
        "accepted": accepted_count > 0,
        "accepted_round_count": accepted_count,
        "rejected_round_count": rejected_count,
        "reason": reason,
        "before": v122._public_metrics(before_metrics),
        "after": v122._public_metrics(after_metrics),
        "initial_candidate_rerank": stage1_candidate_diagnostics,
        "events": events,
        "changed_positions": sorted(all_changed_positions),
        "final_tokens": [int(item) for item in current_tokens],
        "final_text": after_metrics["text"],
        "final_accuracy": after_metrics["accuracy"],
        "stage1_gradient_trend_stats": stage1_gradient_trend_stats,
    }


def run_suffix_v1_2_3(
        model,
        embed_layer,
        initial_optimizable_embedding,
        prefix_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        tokenizer,
        total_input_ids,
        right_range,
        config,
        stage1_lr,
        stage1_epoch,
        stage1_range_weight,
        stage1_clip,
        stage1_init_method,
        stage1_init_param,
        filter_nonascii=True,
        add_perplexity=True,
        top_k_ppl=10,
        top_k_cos=10,
        invert_method="cosine",
        eval_start_pos=0,
        embedding_top_indices=None,
        select_candidate_from_top_indices=None,
        get_perplexity=None,
        forward_and_get_last_hidden_state=None,
        log_file=None):
    del log_file
    if not config.enabled:
        return None, None, {
            "name": METHOD_NAME,
            "method": METHOD_NAME,
            "version": VERSION,
            "enabled": False,
            "skipped": True,
            "reason": "disabled",
            "events": [],
        }
    required_helpers = {
        "embedding_top_indices": embedding_top_indices,
        "select_candidate_from_top_indices": (
            select_candidate_from_top_indices
        ),
        "get_perplexity": get_perplexity,
        "forward_and_get_last_hidden_state": (
            forward_and_get_last_hidden_state
        ),
    }
    missing = [name for name, value in required_helpers.items() if value is None]
    if missing:
        raise ValueError(
            "missing {} helpers: {}".format(METHOD_NAME, ", ".join(missing))
        )

    (
        stage1_embedding,
        optimization_summary,
        gradient_trend_stats,
    ) = _stage1_optimize(
        model,
        initial_optimizable_embedding,
        prefix_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        right_range,
        stage1_lr,
        stage1_epoch,
        stage1_range_weight,
        stage1_clip,
        eval_start_pos,
        config.gradient_trend_stats_enabled,
    )
    embedding_values = v122._embedding_forward_relative_mse(
        model,
        stage1_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
    )
    (
        stage1_accuracy,
        stage1_text,
        stage1_tokens,
        candidate_diagnostics,
        stage1_changed_positions,
    ) = _stage1_rerank_positions(
        stage1_embedding,
        tokenizer,
        model,
        embed_layer,
        target_hidden_state,
        total_input_ids,
        layer_id,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        eval_start_pos,
        get_perplexity,
        forward_and_get_last_hidden_state,
        attention_mask,
    )
    token_values = v122._token_forward_relative_mse(
        model,
        stage1_tokens,
        target_hidden_state,
        layer_id,
        forward_and_get_last_hidden_state,
    )
    stage1_valid_mask = _valid_position_mask(
        int(stage1_embedding.shape[1]),
        eval_start_pos,
        attention_mask,
        stage1_embedding.device,
    )
    valid_embedding_values = _masked_values(
        embedding_values,
        stage1_valid_mask,
    )
    valid_token_values = _masked_values(
        token_values,
        stage1_valid_mask,
    )
    embedding_top1_values = [
        item["embedding_mse_top1_distance"]
        for item in candidate_diagnostics
        if item["embedding_mse_top1_distance"] is not None
    ]
    selected_hidden_values = [
        item["selected_hidden_relative_mse"]
        for item in candidate_diagnostics
    ]

    final_embedding, reoptimization_result = _run_reoptimization_v1_2_2(
        model,
        embed_layer,
        stage1_embedding,
        stage1_tokens,
        stage1_text,
        stage1_accuracy,
        candidate_diagnostics,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        tokenizer,
        total_input_ids,
        _v122_config(config),
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        invert_method,
        eval_start_pos,
        embedding_top_indices,
        select_candidate_from_top_indices,
        get_perplexity,
        forward_and_get_last_hidden_state,
        gradient_trend_stats,
    )
    stage1_result = {
        "metric": "relative_mse",
        "vocab_metric": "embedding_mse",
        "candidate_rerank_metric": "hidden_relative_mse",
        "accuracy": stage1_accuracy,
        "tokens": [int(item) for item in stage1_tokens],
        "text": stage1_text,
        "embedding_forward_relative_mse_mean": _safe_mean(
            valid_embedding_values
        ),
        "embedding_forward_relative_mse_max": _safe_max(
            valid_embedding_values
        ),
        "token_forward_relative_mse_mean": _safe_mean(
            valid_token_values
        ),
        "token_forward_relative_mse_max": _safe_max(
            valid_token_values
        ),
        "optimization": {
            **optimization_summary,
            "init_method": str(stage1_init_method),
            "init_param": float(stage1_init_param),
            "embedding_forward_relative_mse_mean": _safe_mean(
                valid_embedding_values
            ),
            "embedding_forward_relative_mse_max": _safe_max(
                valid_embedding_values
            ),
        },
        "candidate_diagnostics": candidate_diagnostics,
        "embedding_mse_top1_mean": _safe_mean(
            embedding_top1_values
        ),
        "embedding_mse_top1_max": _safe_max(
            embedding_top1_values
        ),
        "selected_hidden_relative_mse_mean": _safe_mean(
            selected_hidden_values
        ),
        "selected_hidden_relative_mse_max": _safe_max(
            selected_hidden_values
        ),
        "changed_positions": stage1_changed_positions,
    }
    result = {
        "name": METHOD_NAME,
        "method": METHOD_NAME,
        "version": VERSION,
        "enabled": True,
        "skipped": False,
        "stage1": stage1_result,
        "reoptimization": {
            "source": REOPTIMIZATION_SOURCE,
            **reoptimization_result,
        },
        "metric_direction_note": METRIC_DIRECTION_NOTE,
        "relative_mse_epsilon": RELATIVE_MSE_EPSILON,
        "pre_acc": stage1_accuracy,
        "post_acc": reoptimization_result["post_acc"],
        "final_accuracy": reoptimization_result["final_accuracy"],
        "final_tokens": reoptimization_result["final_tokens"],
        "final_text": reoptimization_result["final_text"],
        "triggered": reoptimization_result["triggered"],
        "accepted": reoptimization_result["accepted"],
        "accepted_round_count": reoptimization_result["accepted_round_count"],
        "rejected_round_count": reoptimization_result["rejected_round_count"],
        "reason": reoptimization_result["reason"],
        "events": reoptimization_result["events"],
        "changed_positions": reoptimization_result["changed_positions"],
        "suffix_gain": (
            reoptimization_result["post_acc"] - stage1_accuracy
        ),
    }
    return final_embedding, stage1_embedding, result


SuffixConfig = SuffixV123Config
run_suffix = run_suffix_v1_2_3
