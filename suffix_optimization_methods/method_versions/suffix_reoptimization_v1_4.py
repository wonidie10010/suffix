from dataclasses import dataclass
import copy
import math

import torch
import torch.nn.functional as F

from . import suffix_reoptimization_v1_2_1 as v121


MIN_ACCEPTANCE_EPS = v121.MIN_ACCEPTANCE_EPS
METHOD_NAME = "suffix_reoptimization_v1.4"
DEFAULT_ACCEPT_MODE = v121.DEFAULT_ACCEPT_MODE
ACCEPT_MODES = v121.ACCEPT_MODES
SCHEDULE_MODES = {"cosine", "constant"}
CONFIDENCE_MODES = {"hybrid", "fixed"}


@dataclass
class SuffixReoptimizationV14Config:
    enabled: bool = False
    log_enabled: bool = True
    coarse_lr_max: float = 0.10
    coarse_lr_min: float = 0.03
    coarse_schedule: str = "cosine"
    fine_epoch: int = 50
    fine_lr_max: float = 0.01
    fine_lr_min: float = 0.001
    fine_schedule: str = "cosine"
    confidence_mode: str = "hybrid"
    confidence_continuous_min: float = 0.80
    confidence_token_min: float = 0.80
    confidence_margin_min: float = 0.02
    confidence_gap_max: float = 0.10
    confidence_percentile_min: float = 0.60
    confidence_min_points: int = 4
    require_candidate_agreement: bool = True
    adaptive_z_threshold: float = 1.5
    adaptive_drop_z_threshold: float = 1.5
    adaptive_min_std: float = 1e-6
    adaptive_min_points: int = 4
    fine_window: int = 2
    fine_window_decay: float = 0.50
    prox_weight: float = 0.005
    range_weight: float = 0.001
    min_hidden_delta: float = 0.005
    accuracy_tolerance: float = 0.0
    accept_mode: str = DEFAULT_ACCEPT_MODE


_as_float = v121._as_float
_safe_mean = v121._safe_mean
_safe_min = v121._safe_min
_target_ids = v121._target_ids
_accuracy = v121._accuracy
_decode = v121._decode
_dedupe = v121._dedupe
_forward_embedding_hidden = v121._forward_embedding_hidden
_hidden_scores_from_embedding = v121._hidden_scores_from_embedding
_hidden_scores_from_tokens = v121._hidden_scores_from_tokens
_candidate_token_ids = v121._candidate_token_ids
_embedding_range_bound = v121._embedding_range_bound


def _normalized_mode(value, valid_modes, label):
    mode = str(value or "").strip().lower()
    if mode not in valid_modes:
        raise ValueError("{} must be one of: {}".format(label, ", ".join(sorted(valid_modes))))
    return mode


def validate_suffix_reoptimization_v1_4_config(config):
    _normalized_mode(config.coarse_schedule, SCHEDULE_MODES, "suffix v1.4 coarse_schedule")
    _normalized_mode(config.fine_schedule, SCHEDULE_MODES, "suffix v1.4 fine_schedule")
    _normalized_mode(config.confidence_mode, CONFIDENCE_MODES, "suffix v1.4 confidence_mode")
    _normalized_mode(config.accept_mode, ACCEPT_MODES, "suffix v1.4 accept_mode")
    if int(config.fine_epoch) <= 0:
        raise ValueError("suffix_v1_4_fine_epoch must be greater than 0")
    if float(config.coarse_lr_max) <= 0.0 or float(config.coarse_lr_min) < 0.0:
        raise ValueError("suffix v1.4 coarse learning rates must be non-negative and max must be positive")
    if float(config.coarse_lr_min) > float(config.coarse_lr_max):
        raise ValueError("suffix_v1_4_coarse_lr_min must not exceed coarse_lr_max")
    if float(config.fine_lr_max) <= 0.0 or float(config.fine_lr_min) < 0.0:
        raise ValueError("suffix v1.4 fine learning rates must be non-negative and max must be positive")
    if float(config.fine_lr_min) > float(config.fine_lr_max):
        raise ValueError("suffix_v1_4_fine_lr_min must not exceed fine_lr_max")
    if int(config.fine_window) < 0:
        raise ValueError("suffix_v1_4_fine_window must be non-negative")
    if not 0.0 <= float(config.fine_window_decay) <= 1.0:
        raise ValueError("suffix_v1_4_fine_window_decay must be in [0, 1]")
    if not 0.0 <= float(config.confidence_percentile_min) <= 1.0:
        raise ValueError("suffix_v1_4_confidence_percentile_min must be in [0, 1]")
    return config


def scheduled_learning_rate(step, total_steps, lr_max, lr_min, schedule="cosine"):
    schedule = _normalized_mode(schedule, SCHEDULE_MODES, "learning-rate schedule")
    total_steps = max(1, int(total_steps))
    step = min(max(0, int(step)), total_steps - 1)
    lr_max = float(lr_max)
    lr_min = float(lr_min)
    if schedule == "constant" or total_steps == 1:
        return lr_max
    progress = step / float(total_steps - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


def _valid_positions(attention_mask, seq_len):
    seq_len = max(0, int(seq_len))
    if attention_mask is None:
        return list(range(seq_len))
    mask = torch.as_tensor(attention_mask).detach()
    if mask.dim() > 1:
        mask = mask[0]
    values = mask.reshape(-1).to(device="cpu").tolist()
    return [pos for pos in range(min(seq_len, len(values))) if bool(values[pos])]


def _rerank_positions_with_diagnostics(input_embed, current_tokens, positions, tokenizer,
                                       model, embed_layer, target_hidden_state, layer_id,
                                       invert_method, filter_nonascii, add_perplexity,
                                       top_k_ppl, top_k_cos, embedding_top_indices,
                                       select_candidate_from_top_indices, get_perplexity,
                                       forward_and_get_last_hidden_state,
                                       fixed_token_ids=None):
    """Rerank only ``positions`` and return model-observable diagnostics.

    Ground Truth is deliberately absent from this interface. ``fixed_token_ids`` is
    limited to structural prefix tokens established by the common inversion path.
    """
    seq_len = int(input_embed.shape[1])
    if current_tokens is None:
        ret_list = [0 for _ in range(seq_len)]
    else:
        ret_list = [int(item) for item in current_tokens]
    if fixed_token_ids:
        for pos, token_id in fixed_token_ids.items():
            if 0 <= int(pos) < seq_len:
                ret_list[int(pos)] = int(token_id)

    requested_positions = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})
    embedding = input_embed.squeeze(0)
    candidate_map = {}
    embedding_top1 = {}
    for pos in requested_positions:
        candidates = _candidate_token_ids(
            embedding[pos],
            embed_layer,
            top_k_cos,
            invert_method,
            tokenizer,
            filter_nonascii,
            embedding_top_indices,
            select_candidate_from_top_indices,
        )
        if candidates:
            embedding_top1[pos] = int(candidates[0])
        if current_tokens is not None:
            candidates.append(int(current_tokens[pos]))
        if not candidates:
            candidates = [int(ret_list[pos])]
        candidates = _dedupe(candidates)
        candidate_map[pos] = candidates
        ret_list[pos] = int(candidates[0])

    diagnostics = {}
    for pos in requested_positions:
        candidates = list(candidate_map[pos])
        if pos > 0 and add_perplexity:
            _, topk_ids = get_perplexity(
                ret_list[:pos], model, layer_id=layer_id, top_k=top_k_ppl
            )
            candidates.extend(int(item) for item in topk_ids.tolist())
            candidates = _dedupe(candidates)

        replaced_sequences = []
        for token_id in candidates:
            replaced = list(ret_list)
            replaced[pos] = int(token_id)
            replaced_sequences.append(replaced)
        candidate_hidden = forward_and_get_last_hidden_state(
            model, replaced_sequences, None, layer_id=layer_id
        )
        target = target_hidden_state.to(candidate_hidden.device)
        candidate_states = candidate_hidden[:, pos, :].type(torch.float32)
        target_state = target[:, pos, :].type(torch.float32)
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        scores = F.cosine_similarity(candidate_states, target_state, dim=-1)
        order = torch.argsort(scores, descending=True).detach().cpu().tolist()
        ordered_ids = [int(candidates[index]) for index in order]
        ordered_scores = [_as_float(scores[index]) for index in order]
        top1_token = ordered_ids[0]
        top2_token = ordered_ids[1] if len(ordered_ids) > 1 else None
        top1_score = ordered_scores[0]
        top2_score = ordered_scores[1] if len(ordered_scores) > 1 else None
        margin = None if top2_score is None else float(top1_score - top2_score)
        ret_list[pos] = top1_token
        diagnostics[pos] = {
            "position": pos,
            "embedding_top1_token_id": embedding_top1.get(pos),
            "top1_token_id": top1_token,
            "top2_token_id": top2_token,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "margin": margin,
            "candidate_count": len(ordered_ids),
            "candidate_token_ids": ordered_ids,
            "candidate_scores": ordered_scores,
            "candidate_agreement": (
                embedding_top1.get(pos) is not None
                and int(embedding_top1[pos]) == int(top1_token)
            ),
        }
    return ret_list, diagnostics


def _percentile_ranks(values, higher_is_better=True, min_std=1e-6):
    if len(values) < 2:
        return None
    numeric = [float(value) for value in values]
    if any(not math.isfinite(value) for value in numeric):
        return None
    if max(numeric) - min(numeric) <= float(min_std):
        return None
    denominator = float(len(numeric) - 1)
    ranks = []
    for value in numeric:
        less = sum(other < value for other in numeric)
        equal = sum(other == value for other in numeric)
        rank = (less + 0.5 * (equal - 1)) / denominator
        ranks.append(rank if higher_is_better else 1.0 - rank)
    return ranks


def _build_confidence_mask(continuous_scores, token_scores, candidate_diagnostics,
                           valid_positions, anomaly_reasons, config):
    """Build the confidence mask exclusively from model-observable signals."""
    mode = _normalized_mode(config.confidence_mode, CONFIDENCE_MODES, "suffix v1.4 confidence_mode")
    entries = []
    finite_entry_indices = []
    for pos in [int(item) for item in valid_positions]:
        candidate = candidate_diagnostics.get(pos, {})
        continuous = continuous_scores[pos] if pos < len(continuous_scores) else None
        token = token_scores[pos] if pos < len(token_scores) else None
        margin = candidate.get("margin")
        gap = None
        if continuous is not None and token is not None:
            gap = max(0.0, float(continuous) - float(token))
        values = (continuous, token, margin, gap)
        finite = all(value is not None and math.isfinite(float(value)) for value in values)
        entry = {
            "position": pos,
            "continuous_similarity": None if continuous is None else float(continuous),
            "token_forward_similarity": None if token is None else float(token),
            "margin": None if margin is None else float(margin),
            "discretization_gap": gap,
            "embedding_top1_token_id": candidate.get("embedding_top1_token_id"),
            "selected_token_id": candidate.get("top1_token_id"),
            "candidate_count": int(candidate.get("candidate_count") or 0),
            "candidate_token_ids": list(candidate.get("candidate_token_ids") or []),
            "candidate_agreement": bool(candidate.get("candidate_agreement")),
            "anomaly_reasons": list(anomaly_reasons.get(pos) or []),
            "percentile_confidence": None,
            "adaptive_gate_applied": False,
            "high_confidence": False,
            "gate_failures": [],
        }
        if finite:
            finite_entry_indices.append(len(entries))
        entries.append(entry)

    adaptive_available = False
    if mode == "hybrid" and len(finite_entry_indices) >= max(2, int(config.confidence_min_points)):
        finite_entries = [entries[index] for index in finite_entry_indices]
        rank_groups = [
            _percentile_ranks(
                [entry["continuous_similarity"] for entry in finite_entries],
                True,
                config.adaptive_min_std,
            ),
            _percentile_ranks(
                [entry["token_forward_similarity"] for entry in finite_entries],
                True,
                config.adaptive_min_std,
            ),
            _percentile_ranks(
                [entry["margin"] for entry in finite_entries],
                True,
                config.adaptive_min_std,
            ),
            _percentile_ranks(
                [entry["discretization_gap"] for entry in finite_entries],
                False,
                config.adaptive_min_std,
            ),
        ]
        adaptive_available = all(ranks is not None for ranks in rank_groups)
        if adaptive_available:
            for local_index, entry_index in enumerate(finite_entry_indices):
                score = sum(ranks[local_index] for ranks in rank_groups) / len(rank_groups)
                entries[entry_index]["percentile_confidence"] = float(score)
                entries[entry_index]["adaptive_gate_applied"] = True

    high_positions = []
    low_positions = []
    for entry in entries:
        failures = []
        continuous = entry["continuous_similarity"]
        token = entry["token_forward_similarity"]
        margin = entry["margin"]
        gap = entry["discretization_gap"]
        if any(value is None or not math.isfinite(float(value)) for value in (continuous, token, margin, gap)):
            failures.append("nonfinite_or_missing_signal")
        else:
            if continuous < float(config.confidence_continuous_min):
                failures.append("continuous_similarity_below_min")
            if token < float(config.confidence_token_min):
                failures.append("token_forward_similarity_below_min")
            if margin < float(config.confidence_margin_min):
                failures.append("margin_below_min")
            if gap > float(config.confidence_gap_max):
                failures.append("discretization_gap_above_max")
        if entry["candidate_count"] < 2:
            failures.append("fewer_than_two_candidates")
        if bool(config.require_candidate_agreement) and not entry["candidate_agreement"]:
            failures.append("candidate_disagreement")
        if entry["anomaly_reasons"]:
            failures.append("adaptive_anomaly")
        if (
            adaptive_available
            and entry["percentile_confidence"] is not None
            and entry["percentile_confidence"] < float(config.confidence_percentile_min)
        ):
            failures.append("percentile_confidence_below_min")
        entry["gate_failures"] = failures
        entry["high_confidence"] = not failures
        if entry["high_confidence"]:
            high_positions.append(entry["position"])
        else:
            low_positions.append(entry["position"])

    return {
        "mode": mode,
        "adaptive_gate_applied": adaptive_available,
        "thresholds": {
            "continuous_min": float(config.confidence_continuous_min),
            "token_min": float(config.confidence_token_min),
            "margin_min": float(config.confidence_margin_min),
            "gap_max": float(config.confidence_gap_max),
            "percentile_min": float(config.confidence_percentile_min),
            "min_points": int(config.confidence_min_points),
            "require_candidate_agreement": bool(config.require_candidate_agreement),
        },
        "valid_positions": [int(item) for item in valid_positions],
        "high_confidence_positions": high_positions,
        "low_confidence_positions": low_positions,
        "high_confidence_count": len(high_positions),
        "low_confidence_count": len(low_positions),
        "per_position": entries,
    }


def _build_anchored_baseline(coarse_embedding, token_ids, embed_layer,
                             high_confidence_positions, structural_frozen_positions):
    anchored = coarse_embedding.detach().clone()
    positions = sorted({
        int(pos)
        for pos in list(high_confidence_positions) + list(structural_frozen_positions)
        if 0 <= int(pos) < int(anchored.shape[1])
    })
    if not positions:
        return anchored
    position_index = torch.tensor(positions, device=anchored.device, dtype=torch.long)
    token_index = torch.tensor(
        [int(token_ids[pos]) for pos in positions],
        device=embed_layer.weight.device,
        dtype=torch.long,
    )
    discrete = embed_layer.weight.detach().index_select(0, token_index).to(
        device=anchored.device, dtype=anchored.dtype
    )
    anchored[:, position_index, :] = discrete.unsqueeze(0)
    return anchored.detach()


def _build_fine_observation_weights(seq_len, low_confidence_positions, valid_positions,
                                    window, decay, device=None, dtype=torch.float32):
    seq_len = max(0, int(seq_len))
    weights = torch.zeros(seq_len, device=device, dtype=dtype)
    valid_set = {int(pos) for pos in valid_positions}
    window = max(0, int(window))
    decay = float(decay)
    for pos in low_confidence_positions:
        pos = int(pos)
        for distance in range(window + 1):
            observed = pos + distance
            if observed not in valid_set or observed >= seq_len:
                continue
            value = decay ** distance
            weights[observed] = torch.maximum(
                weights[observed],
                torch.tensor(value, device=weights.device, dtype=weights.dtype),
            )
    return weights.view(1, seq_len)


def _assemble_masked_embedding(frozen_base, low_position_index, low_parameter):
    if low_position_index.numel() == 0:
        return frozen_base.detach().clone()
    values = low_parameter.to(device=frozen_base.device, dtype=frozen_base.dtype)
    return torch.index_copy(frozen_base, 1, low_position_index, values)


def _tail_std(values, length=10):
    if not values:
        return None
    tail = values[-max(1, int(length)):]
    if len(tail) == 1:
        return 0.0
    mean_value = sum(tail) / len(tail)
    variance = sum((value - mean_value) ** 2 for value in tail) / len(tail)
    return variance ** 0.5


def _adaptive_anomalies_for_positions(embedding_scores, token_scores, valid_positions, config):
    positions = [int(pos) for pos in valid_positions]
    compact_embedding_scores = [embedding_scores[pos] for pos in positions]
    compact_token_scores = [token_scores[pos] for pos in positions]
    compact_anomalies = v121._find_adaptive_anomalies(
        compact_embedding_scores,
        compact_token_scores,
        0,
        config,
    )
    anomalies = []
    for compact_item in compact_anomalies:
        compact_position = int(compact_item["position"])
        if not 0 <= compact_position < len(positions):
            continue
        item = copy.deepcopy(compact_item)
        item["position"] = positions[compact_position]
        anomalies.append(item)
    return anomalies


def _accuracy_on_positions(token_ids, total_input_ids, valid_positions):
    targets = _target_ids(total_input_ids)
    comparable = [
        int(pos)
        for pos in valid_positions
        if int(pos) < len(token_ids) and int(pos) < len(targets)
    ]
    if not comparable:
        return 0.0
    correct = sum(int(token_ids[pos]) == int(targets[pos]) for pos in comparable)
    return correct / len(comparable)


def _optimize_masked_positions(model, anchored_embedding, coarse_embedding,
                               target_hidden_state, attention_mask, layer_id,
                               register_layer_hooks, low_confidence_positions,
                               valid_positions, config, embed_layer):
    low_positions = [int(pos) for pos in low_confidence_positions]
    if not low_positions:
        return anchored_embedding.detach().clone(), {
            "version": "v1.4",
            "optimizer": "Adam",
            "skipped": True,
            "stopped_reason": "no_low_confidence_positions",
            "updated_positions": [],
            "loss_start": None,
            "loss_end": None,
            "loss_min": None,
            "loss_tail_std": None,
        }

    base = anchored_embedding.detach()
    position_index = torch.tensor(low_positions, device=base.device, dtype=torch.long)
    reference = coarse_embedding.detach().index_select(1, position_index).to(torch.float32)
    low_parameter = reference.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([low_parameter], lr=float(config.fine_lr_max))
    observation_weights = _build_fine_observation_weights(
        int(base.shape[1]),
        low_positions,
        valid_positions,
        config.fine_window,
        config.fine_window_decay,
        device=base.device,
        dtype=torch.float32,
    )
    if _as_float(observation_weights.sum()) <= 0.0:
        return base.clone(), {
            "version": "v1.4",
            "optimizer": "Adam",
            "skipped": True,
            "stopped_reason": "no_valid_observation_positions",
            "updated_positions": low_positions,
            "loss_start": None,
            "loss_end": None,
            "loss_min": None,
            "loss_tail_std": None,
        }

    range_bound = None
    if float(config.range_weight) > 0.0:
        range_bound = _embedding_range_bound(
            embed_layer, low_parameter.device, low_parameter.dtype
        )
    histories = {"loss": [], "hidden_loss": [], "prox_loss": [], "range_loss": [], "lr": []}
    stopped_reason = "completed"
    total_steps = max(0, int(config.fine_epoch))
    for epoch_index in range(total_steps):
        lr = scheduled_learning_rate(
            epoch_index,
            total_steps,
            config.fine_lr_max,
            config.fine_lr_min,
            config.fine_schedule,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        current_embedding = _assemble_masked_embedding(base, position_index, low_parameter)
        hidden = _forward_embedding_hidden(
            model,
            current_embedding,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target = target_hidden_state.to(hidden.device)
        per_position = 1.0 - F.cosine_similarity(
            hidden.type(torch.float32), target.type(torch.float32), dim=-1
        )
        weights = observation_weights.to(per_position.device)
        hidden_loss = (per_position * weights).sum() / weights.sum().clamp_min(1e-12)
        prox_loss = F.mse_loss(low_parameter, reference.to(low_parameter.device))
        range_loss = torch.zeros((), device=low_parameter.device, dtype=low_parameter.dtype)
        if range_bound is not None:
            range_loss = F.relu(torch.abs(low_parameter) - range_bound).mean()
        loss = (
            hidden_loss
            + float(config.prox_weight) * prox_loss
            + float(config.range_weight) * range_loss
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            stopped_reason = "nonfinite_loss"
            break
        histories["loss"].append(_as_float(loss))
        histories["hidden_loss"].append(_as_float(hidden_loss))
        histories["prox_loss"].append(_as_float(prox_loss))
        histories["range_loss"].append(_as_float(range_loss))
        histories["lr"].append(float(lr))
        loss.backward()
        optimizer.step()

    final_embedding = _assemble_masked_embedding(base, position_index, low_parameter).detach()
    if not bool(torch.isfinite(final_embedding).all().detach().cpu()):
        stopped_reason = "nonfinite_embedding"
    elif not histories["loss"]:
        stopped_reason = "no_valid_loss"
    summary = {
        "version": "v1.4",
        "optimizer": "Adam",
        "optimizer_scope": "single sparse fine-stage optimizer",
        "skipped": False,
        "schedule": str(config.fine_schedule),
        "epoch": int(config.fine_epoch),
        "executed_steps": len(histories["loss"]),
        "lr_start": histories["lr"][0] if histories["lr"] else None,
        "lr_end": histories["lr"][-1] if histories["lr"] else None,
        "fine_window": int(config.fine_window),
        "fine_window_decay": float(config.fine_window_decay),
        "prox_weight": float(config.prox_weight),
        "range_weight": float(config.range_weight),
        "updated_positions": low_positions,
        "observation_positions": [
            int(index)
            for index, value in enumerate(observation_weights.squeeze(0).detach().cpu().tolist())
            if value > 0.0
        ],
        "loss_start": histories["loss"][0] if histories["loss"] else None,
        "loss_end": histories["loss"][-1] if histories["loss"] else None,
        "loss_min": min(histories["loss"]) if histories["loss"] else None,
        "loss_tail_std": _tail_std(histories["loss"]),
        "hidden_loss_start": histories["hidden_loss"][0] if histories["hidden_loss"] else None,
        "hidden_loss_end": histories["hidden_loss"][-1] if histories["hidden_loss"] else None,
        "prox_loss_start": histories["prox_loss"][0] if histories["prox_loss"] else None,
        "prox_loss_end": histories["prox_loss"][-1] if histories["prox_loss"] else None,
        "range_loss_start": histories["range_loss"][0] if histories["range_loss"] else None,
        "range_loss_end": histories["range_loss"][-1] if histories["range_loss"] else None,
        "stopped_reason": stopped_reason,
    }
    return final_embedding, summary


def _evaluate_state(model, embedding, tokens, target_hidden_state, attention_mask,
                    layer_id, register_layer_hooks, total_input_ids, tokenizer,
                    eval_start_pos, valid_positions, config,
                    forward_and_get_last_hidden_state):
    embedding_scores = _hidden_scores_from_embedding(
        model,
        embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
    )
    token_scores = _hidden_scores_from_tokens(
        model,
        tokens,
        target_hidden_state,
        layer_id,
        forward_and_get_last_hidden_state,
    )
    anomalies = _adaptive_anomalies_for_positions(
        embedding_scores, token_scores, valid_positions, config
    )
    valid_embedding_scores = [embedding_scores[pos] for pos in valid_positions]
    valid_token_scores = [token_scores[pos] for pos in valid_positions]
    first_anomaly = anomalies[0] if anomalies else None
    return {
        "accuracy": _accuracy_on_positions(tokens, total_input_ids, valid_positions),
        "text": _decode(tokenizer, tokens, eval_start_pos),
        "hidden_mean": _safe_mean(valid_token_scores),
        "hidden_min": _safe_min(valid_token_scores),
        "embedding_hidden_mean": _safe_mean(valid_embedding_scores),
        "embedding_hidden_min": _safe_min(valid_embedding_scores),
        "anomaly_count": len(anomalies),
        "first_anomaly_position": first_anomaly.get("position") if first_anomaly else None,
        "first_anomaly_reasons": first_anomaly.get("reasons") if first_anomaly else [],
        "_embedding_scores": embedding_scores,
        "_token_scores": token_scores,
        "_anomalies": anomalies,
    }


def _public_metrics(metrics):
    return {
        "evaluation_only": True,
        "oracle_accuracy": metrics.get("accuracy"),
        "hidden_mean": metrics.get("hidden_mean"),
        "hidden_min": metrics.get("hidden_min"),
        "embedding_hidden_mean": metrics.get("embedding_hidden_mean"),
        "embedding_hidden_min": metrics.get("embedding_hidden_min"),
        "anomaly_count": metrics.get("anomaly_count"),
        "first_anomaly_position": metrics.get("first_anomaly_position"),
        "first_anomaly_reasons": metrics.get("first_anomaly_reasons"),
    }


def _changed_positions(before_tokens, after_tokens, allowed_positions):
    allowed = {int(pos) for pos in allowed_positions}
    return [
        pos
        for pos in sorted(allowed)
        if pos < len(before_tokens)
        and pos < len(after_tokens)
        and int(before_tokens[pos]) != int(after_tokens[pos])
    ]


def _accept_candidate(baseline_metrics, candidate_metrics, changed_positions, config):
    mode = _normalized_mode(config.accept_mode, ACCEPT_MODES, "suffix v1.4 accept_mode")
    if not changed_positions:
        return False, "reconstruction_unchanged"
    required_numeric = (
        candidate_metrics.get("accuracy"),
        candidate_metrics.get("hidden_mean"),
        candidate_metrics.get("embedding_hidden_mean"),
    )
    if any(value is None or not math.isfinite(float(value)) for value in required_numeric):
        return False, "nonfinite_candidate_metrics"
    if mode == "always":
        return True, "always_accept"

    if mode == "oracle_accuracy":
        baseline_accuracy = float(baseline_metrics["accuracy"])
        candidate_accuracy = float(candidate_metrics["accuracy"])
        tolerance = max(0.0, float(config.accuracy_tolerance))
        if candidate_accuracy > baseline_accuracy + tolerance + MIN_ACCEPTANCE_EPS:
            return True, "accuracy_improved"
        if candidate_accuracy + tolerance + MIN_ACCEPTANCE_EPS < baseline_accuracy:
            return False, "accuracy_decreased"

    anomaly_improved = candidate_metrics["anomaly_count"] < baseline_metrics["anomaly_count"]
    baseline_hidden = baseline_metrics.get("hidden_mean")
    candidate_hidden = candidate_metrics.get("hidden_mean")
    hidden_improved = (
        baseline_hidden is not None
        and candidate_hidden is not None
        and candidate_hidden >= baseline_hidden + float(config.min_hidden_delta)
    )
    if anomaly_improved and hidden_improved:
        return True, "anomaly_count_and_hidden_improved"
    if anomaly_improved:
        return True, "anomaly_count_improved"
    if hidden_improved:
        return True, "hidden_improved"
    if mode == "hidden_anomaly":
        return False, "no_hidden_anomaly_improvement"
    return False, "no_global_improvement"


def _oracle_mask_diagnostics(mask, coarse_tokens, candidate_diagnostics, total_input_ids):
    targets = _target_ids(total_input_ids)
    high_positions = mask["high_confidence_positions"]
    low_positions = mask["low_confidence_positions"]
    wrong_positions = [
        pos
        for pos in mask["valid_positions"]
        if pos < len(targets) and int(coarse_tokens[pos]) != int(targets[pos])
    ]
    correct_high = sum(
        int(coarse_tokens[pos]) == int(targets[pos])
        for pos in high_positions
        if pos < len(targets)
    )
    uncertain_wrong = sum(pos in set(low_positions) for pos in wrong_positions)
    candidate_hits = 0
    low_candidate_hits = 0
    for pos in mask["valid_positions"]:
        candidates = candidate_diagnostics.get(pos, {}).get("candidate_token_ids") or []
        if pos < len(targets) and int(targets[pos]) in {int(item) for item in candidates}:
            candidate_hits += 1
            if pos in set(low_positions):
                low_candidate_hits += 1
    return {
        "evaluation_only": True,
        "oracle_anchor_precision": (
            correct_high / len(high_positions) if high_positions else None
        ),
        "oracle_uncertain_recall": (
            uncertain_wrong / len(wrong_positions) if wrong_positions else None
        ),
        "oracle_candidate_recall_at_k": (
            candidate_hits / len(mask["valid_positions"])
            if mask["valid_positions"] else None
        ),
        "oracle_low_confidence_candidate_recall_at_k": (
            low_candidate_hits / len(low_positions) if low_positions else None
        ),
        "oracle_wrong_position_count": len(wrong_positions),
        "oracle_correct_anchor_count": int(correct_high),
    }


def run_suffix_reoptimization_v1_4(model, embed_layer, optimized_embedding,
                                   target_hidden_state, attention_mask, layer_id,
                                   register_layer_hooks, tokenizer, total_input_ids,
                                   config, filter_nonascii=True, add_perplexity=True,
                                   top_k_ppl=10, top_k_cos=10,
                                   invert_method="cosine", eval_start_pos=0,
                                   embedding_top_indices=None,
                                   select_candidate_from_top_indices=None,
                                   get_perplexity=None,
                                   forward_and_get_last_hidden_state=None,
                                   coarse_stage_summary=None, log_file=None):
    del log_file  # Detailed sidecar output belongs in reconstructions.jsonl.
    if not config.enabled:
        return optimized_embedding, {
            "name": METHOD_NAME,
            "version": "v1.4",
            "enabled": False,
            "skipped": True,
            "reason": "disabled",
            "events": [],
            "manifold_enabled": False,
            "manifold_weight": 0.0,
            "manifold_updates": 0,
        }
    validate_suffix_reoptimization_v1_4_config(config)
    required_helpers = {
        "embedding_top_indices": embedding_top_indices,
        "select_candidate_from_top_indices": select_candidate_from_top_indices,
        "get_perplexity": get_perplexity,
        "forward_and_get_last_hidden_state": forward_and_get_last_hidden_state,
    }
    missing = [name for name, value in required_helpers.items() if value is None]
    if missing:
        raise ValueError("missing {} helpers: {}".format(METHOD_NAME, ", ".join(missing)))

    seq_len = int(optimized_embedding.shape[1])
    all_valid_positions = _valid_positions(attention_mask, seq_len)
    all_valid_position_set = set(all_valid_positions)
    padding_positions = [pos for pos in range(seq_len) if pos not in all_valid_position_set]
    valid_positions = [pos for pos in all_valid_positions if pos >= int(eval_start_pos)]
    structural_positions = [pos for pos in all_valid_positions if pos < int(eval_start_pos)]
    fixed_positions = structural_positions + padding_positions
    target_ids = _target_ids(total_input_ids)
    fixed_token_ids = {
        pos: target_ids[pos]
        for pos in fixed_positions
        if pos < len(target_ids)
    }

    coarse_embedding = optimized_embedding.detach().clone()
    coarse_tokens, candidate_diagnostics = _rerank_positions_with_diagnostics(
        coarse_embedding,
        None,
        valid_positions,
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
        embedding_top_indices,
        select_candidate_from_top_indices,
        get_perplexity,
        forward_and_get_last_hidden_state,
        fixed_token_ids=fixed_token_ids,
    )
    coarse_metrics = _evaluate_state(
        model,
        coarse_embedding,
        coarse_tokens,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        total_input_ids,
        tokenizer,
        eval_start_pos,
        valid_positions,
        config,
        forward_and_get_last_hidden_state,
    )
    anomaly_reasons = {
        int(item["position"]): list(item.get("reasons") or [])
        for item in coarse_metrics["_anomalies"]
    }
    mask = _build_confidence_mask(
        coarse_metrics["_embedding_scores"],
        coarse_metrics["_token_scores"],
        candidate_diagnostics,
        valid_positions,
        anomaly_reasons,
        config,
    )
    mask["structural_frozen_positions"] = structural_positions
    mask["structural_frozen_count"] = len(structural_positions)
    mask["padding_frozen_positions"] = padding_positions
    mask["padding_frozen_count"] = len(padding_positions)
    high_positions = mask["high_confidence_positions"]
    low_positions = mask["low_confidence_positions"]
    anchored_embedding = _build_anchored_baseline(
        coarse_embedding,
        coarse_tokens,
        embed_layer,
        high_positions,
        fixed_positions,
    )
    anchored_metrics = _evaluate_state(
        model,
        anchored_embedding,
        coarse_tokens,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        total_input_ids,
        tokenizer,
        eval_start_pos,
        valid_positions,
        config,
        forward_and_get_last_hidden_state,
    )
    oracle_diagnostics = _oracle_mask_diagnostics(
        mask, coarse_tokens, candidate_diagnostics, total_input_ids
    )

    candidate_embedding, fine_summary = _optimize_masked_positions(
        model,
        anchored_embedding,
        coarse_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        low_positions,
        all_valid_positions,
        config,
        embed_layer,
    )
    candidate_tokens = list(coarse_tokens)
    candidate_metrics = anchored_metrics
    changed_positions = []
    accepted = False
    if not low_positions:
        accept_reason = "all_positions_confident"
    elif fine_summary.get("stopped_reason") != "completed":
        accept_reason = "fine_stage_{}".format(fine_summary.get("stopped_reason"))
    else:
        candidate_tokens, _ = _rerank_positions_with_diagnostics(
            candidate_embedding,
            coarse_tokens,
            low_positions,
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
            embedding_top_indices,
            select_candidate_from_top_indices,
            get_perplexity,
            forward_and_get_last_hidden_state,
            fixed_token_ids=fixed_token_ids,
        )
        candidate_metrics = _evaluate_state(
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
            valid_positions,
            config,
            forward_and_get_last_hidden_state,
        )
        changed_positions = _changed_positions(coarse_tokens, candidate_tokens, low_positions)
        accepted, accept_reason = _accept_candidate(
            anchored_metrics, candidate_metrics, changed_positions, config
        )

    frozen_positions = sorted(set(fixed_positions + high_positions))
    frozen_position_mutations = [
        pos
        for pos in frozen_positions
        if pos < len(coarse_tokens)
        and pos < len(candidate_tokens)
        and int(coarse_tokens[pos]) != int(candidate_tokens[pos])
    ]
    if frozen_position_mutations:
        accepted = False
        accept_reason = "frozen_token_mutation_detected"
    frozen_embedding_mutations = [
        pos
        for pos in frozen_positions
        if not torch.equal(candidate_embedding[:, pos, :], anchored_embedding[:, pos, :])
    ]
    if frozen_embedding_mutations:
        accepted = False
        accept_reason = "frozen_embedding_mutation_detected"
    final_embedding = candidate_embedding if accepted else anchored_embedding
    final_tokens = candidate_tokens if accepted else coarse_tokens
    final_metrics = candidate_metrics if accepted else anchored_metrics

    triggered = bool(low_positions)
    event = {
        "round": 1,
        "triggered": triggered,
        "anomaly_position": low_positions[0] if low_positions else None,
        "anomaly_reasons": ["low_confidence_mask"] if low_positions else [],
        "accept_mode": str(config.accept_mode),
        "accepted": accepted,
        "accept_reason": accept_reason,
        "evaluation_only": True,
        "oracle_before_accuracy": anchored_metrics["accuracy"],
        "oracle_candidate_accuracy": candidate_metrics["accuracy"],
        "before_hidden_mean": anchored_metrics["hidden_mean"],
        "candidate_hidden_mean": candidate_metrics["hidden_mean"],
        "before_anomaly_count": anchored_metrics["anomaly_count"],
        "candidate_anomaly_count": candidate_metrics["anomaly_count"],
        "frozen_positions": frozen_positions,
        "optimized_positions": low_positions,
        "changed_positions": changed_positions,
        "frozen_position_mutations": frozen_position_mutations,
        "frozen_embedding_mutations": frozen_embedding_mutations,
        "optimization": fine_summary,
    }
    if not triggered:
        reason = "all positions passed the confidence gate"
    elif accepted:
        reason = "accepted masked fine-stage refinement"
    else:
        reason = "masked fine-stage refinement rejected: {}".format(accept_reason)
    coarse_stage = copy.deepcopy(coarse_stage_summary or {})
    if "acc" in coarse_stage:
        coarse_stage["oracle_accuracy"] = coarse_stage.pop("acc")
        coarse_stage["evaluation_only"] = True
    coarse_stage.setdefault("version", "v1.4")
    coarse_stage.setdefault("optimizer", "SGD")
    coarse_stage.setdefault("schedule", str(config.coarse_schedule))
    coarse_stage.setdefault("lr_start", float(config.coarse_lr_max))
    coarse_stage.setdefault("lr_end", float(config.coarse_lr_min))
    result = {
        "name": METHOD_NAME,
        "version": "v1.4",
        "enabled": True,
        "skipped": False,
        "accept_mode": str(config.accept_mode),
        "manifold_enabled": False,
        "manifold_weight": 0.0,
        "manifold_updates": 0,
        "coarse_stage": coarse_stage,
        "coarse_state": _public_metrics(coarse_metrics),
        "confidence_mask": mask,
        "anchored_baseline": _public_metrics(anchored_metrics),
        "fine_stage": fine_summary,
        "oracle_diagnostics": oracle_diagnostics,
        **oracle_diagnostics,
        "pre_acc": coarse_metrics["accuracy"],
        "post_acc": final_metrics["accuracy"],
        "oracle_pre_acc": coarse_metrics["accuracy"],
        "oracle_post_acc": final_metrics["accuracy"],
        "triggered": triggered,
        "accepted": accepted,
        "reason": reason,
        "before": _public_metrics(coarse_metrics),
        "after": _public_metrics(final_metrics),
        "events": [event],
        "frozen_positions": frozen_positions,
        "optimized_positions": low_positions,
        "changed_positions": changed_positions if accepted else [],
        "frozen_position_mutation_count": len(frozen_position_mutations),
        "frozen_embedding_mutation_count": len(frozen_embedding_mutations),
        "final_tokens": [int(item) for item in final_tokens],
        "final_text": final_metrics["text"],
        "final_accuracy": final_metrics["accuracy"],
        "oracle_final_accuracy": final_metrics["accuracy"],
    }
    return final_embedding.detach(), result


SuffixReoptimizationConfig = SuffixReoptimizationV14Config
run_suffix_reoptimization = run_suffix_reoptimization_v1_4
