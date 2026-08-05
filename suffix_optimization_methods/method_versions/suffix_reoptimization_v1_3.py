from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F

from experiment_outputs import format_metric, log_kv, log_line, log_section
from . import suffix_reoptimization_v1_2 as v12


MIN_ACCEPTANCE_EPS = v12.MIN_ACCEPTANCE_EPS
VOCAB_SEARCH_CHUNK_SIZE = v12.VOCAB_SEARCH_CHUNK_SIZE
METHOD_NAME = "suffix_reoptimization_v1.3"
DEFAULT_ACCEPT_MODE = v12.DEFAULT_ACCEPT_MODE
ACCEPT_MODES = v12.ACCEPT_MODES
DEFAULT_ANOMALY_DETECTION_MODE = v12.DEFAULT_ANOMALY_DETECTION_MODE
ANOMALY_DETECTION_MODES = v12.ANOMALY_DETECTION_MODES
DEFAULT_ANCHOR_MODE = "anchor_stable_prefix"
ANCHOR_MODES = {"anchor_off", "anchor_full_prefix", "anchor_stable_prefix"}


@dataclass
class SuffixReoptimizationV13Config:
    enabled: bool = False
    log_enabled: bool = True
    max_rounds: int = 2
    epoch: int = 50
    lr: float = 0.03
    hidden_low_threshold: float = 0.50
    hidden_drop_threshold: float = 0.15
    token_forward_low_threshold: float = 0.50
    min_anomaly_reasons: int = 2
    min_hidden_delta: float = 0.005
    accuracy_tolerance: float = 0.0
    accept_mode: str = DEFAULT_ACCEPT_MODE
    anomaly_detection_mode: str = DEFAULT_ANOMALY_DETECTION_MODE
    adaptive_z_threshold: float = 1.5
    adaptive_drop_z_threshold: float = 1.5
    adaptive_min_std: float = 1e-6
    adaptive_min_points: int = 4
    hidden_weight_mode: str = "front_decay"
    hidden_weight_decay: float = 0.90
    hidden_weight_floor: float = 0.20
    prox_weight: float = 0.005
    manifold_weight: float = 0.02
    manifold_update_every: int = 10
    manifold_warmup_epoch: int = 10
    range_weight: float = 0.001
    anchor_mode: str = DEFAULT_ANCHOR_MODE


_as_float = v12._as_float
_safe_mean = v12._safe_mean
_safe_min = v12._safe_min
_target_ids = v12._target_ids
_accuracy = v12._accuracy
_decode = v12._decode
_dedupe = v12._dedupe
_forward_embedding_hidden = v12._forward_embedding_hidden
_hidden_scores_from_embedding = v12._hidden_scores_from_embedding
_hidden_scores_from_tokens = v12._hidden_scores_from_tokens
_candidate_token_ids = v12._candidate_token_ids
_rerank_positions = v12._rerank_positions
_stats = v12._stats
_z_below = v12._z_below
_z_above = v12._z_above
_find_threshold_anomalies = v12._find_threshold_anomalies
_find_adaptive_anomalies = v12._find_adaptive_anomalies
_find_anomalies = v12._find_anomalies
_public_metrics = v12._public_metrics
_evaluate_state = v12._evaluate_state
_merge_suffix = v12._merge_suffix
_build_suffix_hidden_weights = v12._build_suffix_hidden_weights
_nearest_vocab_embeddings = v12._nearest_vocab_embeddings
_embedding_range_bound = v12._embedding_range_bound
_loss_summary_value = v12._loss_summary_value
_changed_positions = v12._changed_positions
_suffix_mean = v12._suffix_mean
_accept_mode = v12._accept_mode
_accept_candidate = v12._accept_candidate


def _anchor_mode(config):
    mode = str(getattr(config, "anchor_mode", DEFAULT_ANCHOR_MODE) or DEFAULT_ANCHOR_MODE)
    mode = mode.lower()
    if mode not in ANCHOR_MODES:
        raise ValueError("suffix anchor_mode must be one of: {}".format(
            ", ".join(sorted(ANCHOR_MODES))
        ))
    return mode


def _valid_positions(attention_mask, seq_len):
    seq_len = max(0, int(seq_len))
    if attention_mask is None:
        return list(range(seq_len))
    mask = torch.as_tensor(attention_mask).detach()
    if mask.dim() > 1:
        mask = mask[0]
    mask_values = mask.reshape(-1).to(device="cpu").tolist()
    return [
        pos
        for pos in range(min(seq_len, len(mask_values)))
        if bool(mask_values[pos])
    ]


def _resolve_effective_suffix_start(detected_anomaly_start, anchor_mode, eval_start_pos,
                                    full_anomalies, rejected_boundaries,
                                    accepted_changed_positions, valid_positions):
    detected_anomaly_start = int(detected_anomaly_start)
    eval_start_pos = max(0, int(eval_start_pos))
    if anchor_mode != "anchor_stable_prefix":
        return detected_anomaly_start, [], False

    valid_set = {int(pos) for pos in valid_positions}
    unstable_positions = set()
    for anomaly in full_anomalies or []:
        pos = int(anomaly.get("position", -1))
        if eval_start_pos <= pos < detected_anomaly_start and pos in valid_set:
            unstable_positions.add(pos)
    for collection in (rejected_boundaries, accepted_changed_positions):
        for value in collection or []:
            pos = int(value)
            if eval_start_pos <= pos < detected_anomaly_start and pos in valid_set:
                unstable_positions.add(pos)

    effective_suffix_start = min(
        [detected_anomaly_start] + sorted(unstable_positions)
    )
    return (
        effective_suffix_start,
        sorted(unstable_positions),
        effective_suffix_start < detected_anomaly_start,
    )


def _build_anchored_candidate(current_embedding, current_tokens, embed_layer,
                              effective_suffix_start, attention_mask, eval_start_pos,
                              anchor_mode):
    """Build anchoring only from the accepted token sequence.

    The helper accepts no target-token or label argument. Fixed special positions
    and the predicted prefix are both materialized from current_tokens.
    """
    mode = str(anchor_mode).lower()
    if mode not in ANCHOR_MODES:
        raise ValueError("suffix anchor_mode must be one of: {}".format(
            ", ".join(sorted(ANCHOR_MODES))
        ))
    anchored_embedding = current_embedding.detach().clone()
    if mode == "anchor_off":
        return anchored_embedding, [], []

    seq_len = int(anchored_embedding.shape[1])
    if len(current_tokens) < seq_len:
        raise ValueError("current_tokens is shorter than the embedding sequence")
    valid_set = set(_valid_positions(attention_mask, seq_len))
    eval_start_pos = max(0, min(int(eval_start_pos), seq_len))
    effective_suffix_start = max(
        eval_start_pos,
        min(int(effective_suffix_start), seq_len),
    )
    fixed_positions = [pos for pos in range(eval_start_pos) if pos in valid_set]
    anchored_positions = [
        pos
        for pos in range(eval_start_pos, effective_suffix_start)
        if pos in valid_set
    ]
    positions = fixed_positions + anchored_positions
    if not positions:
        return anchored_embedding, anchored_positions, fixed_positions

    weight = getattr(embed_layer, "weight", None)
    token_device = weight.device if weight is not None else current_embedding.device
    token_ids = torch.tensor(
        [int(current_tokens[pos]) for pos in positions],
        dtype=torch.long,
        device=token_device,
    )
    with torch.no_grad():
        discrete_embeddings = embed_layer(token_ids).detach().to(
            device=anchored_embedding.device,
            dtype=anchored_embedding.dtype,
        )
        position_index = torch.tensor(
            positions,
            dtype=torch.long,
            device=anchored_embedding.device,
        )
        anchored_embedding[:, position_index, :] = discrete_embeddings.unsqueeze(0)
    return anchored_embedding.detach(), anchored_positions, fixed_positions


def _initial_hidden_loss_from_metrics(metrics, suffix_start, config):
    scores = list(metrics.get("_embedding_scores") or [])[int(suffix_start):]
    if not scores:
        return None
    per_position = torch.tensor(
        [1.0 - float(value) for value in scores],
        dtype=torch.float32,
    ).view(1, -1)
    weights = _build_suffix_hidden_weights(
        len(scores),
        config.hidden_weight_mode,
        config.hidden_weight_decay,
        config.hidden_weight_floor,
        per_position.device,
        per_position.dtype,
    )
    return _as_float(
        (per_position * weights).sum() / weights.sum().clamp_min(1e-12)
    )


def _optimize_suffix(model, input_embed, target_hidden_state, attention_mask, layer_id,
                     register_layer_hooks, suffix_start, config, embed_layer,
                     invert_method="cosine", prox_reference_suffix=None):
    base_embed = input_embed.detach()
    initial_suffix = base_embed[:, suffix_start:, :].detach().clone().to(torch.float32)
    if prox_reference_suffix is None:
        original_suffix = initial_suffix.clone()
        prox_reference_source = "optimization_input_suffix"
    else:
        original_suffix = prox_reference_suffix.detach().clone().to(
            device=initial_suffix.device,
            dtype=torch.float32,
        )
        if tuple(original_suffix.shape) != tuple(initial_suffix.shape):
            raise ValueError("prox_reference_suffix shape must match the optimized suffix")
        prox_reference_source = "original_state_suffix"
    suffix_len = int(initial_suffix.shape[1])
    suffix_param = initial_suffix.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([suffix_param], lr=config.lr)
    hidden_weights = _build_suffix_hidden_weights(
        suffix_len,
        config.hidden_weight_mode,
        config.hidden_weight_decay,
        config.hidden_weight_floor,
        suffix_param.device,
        suffix_param.dtype,
    )
    nearest_vocab_embedding = None
    manifold_updates = 0
    manifold_update_every = max(1, int(config.manifold_update_every))
    manifold_warmup_epoch = max(0, int(config.manifold_warmup_epoch))
    range_bound = None
    if float(config.range_weight) > 0.0:
        range_bound = _embedding_range_bound(
            embed_layer,
            suffix_param.device,
            suffix_param.dtype,
        )

    histories = {
        "loss": [],
        "hidden_loss": [],
        "prox_loss": [],
        "manifold_loss": [],
        "range_loss": [],
    }
    stopped_reason = "completed"

    for epoch_idx in range(max(0, int(config.epoch))):
        optimizer.zero_grad()
        current_embed = _merge_suffix(base_embed, suffix_start, suffix_param)
        hidden_state = _forward_embedding_hidden(
            model,
            current_embed,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        per_pos_hidden_loss = 1.0 - F.cosine_similarity(
            hidden_state[:, suffix_start:, :].type(torch.float32),
            target_hidden[:, suffix_start:, :].type(torch.float32),
            dim=-1,
        )
        weighted_hidden_loss = (
            (per_pos_hidden_loss * hidden_weights).sum()
            / hidden_weights.sum().clamp_min(1e-12)
        )
        prox_loss = F.mse_loss(suffix_param, original_suffix)
        manifold_loss = torch.zeros(
            (), device=suffix_param.device, dtype=suffix_param.dtype
        )
        if float(config.manifold_weight) > 0.0 and epoch_idx >= manifold_warmup_epoch:
            if (
                nearest_vocab_embedding is None
                or (epoch_idx - manifold_warmup_epoch) % manifold_update_every == 0
            ):
                nearest_vocab_embedding = _nearest_vocab_embeddings(
                    suffix_param,
                    embed_layer,
                    invert_method=invert_method,
                )
                manifold_updates += 1
            manifold_loss = F.mse_loss(
                suffix_param,
                nearest_vocab_embedding.detach(),
            )
        range_loss = torch.zeros(
            (), device=suffix_param.device, dtype=suffix_param.dtype
        )
        if range_bound is not None:
            range_loss = F.relu(torch.abs(suffix_param) - range_bound).mean()
        loss = (
            weighted_hidden_loss
            + float(config.prox_weight) * prox_loss
            + float(config.manifold_weight) * manifold_loss
            + float(config.range_weight) * range_loss
        )
        if torch.isnan(loss):
            stopped_reason = "nan_loss"
            break
        histories["loss"].append(_as_float(loss))
        histories["hidden_loss"].append(_as_float(weighted_hidden_loss))
        histories["prox_loss"].append(_as_float(prox_loss))
        histories["manifold_loss"].append(_as_float(manifold_loss))
        histories["range_loss"].append(_as_float(range_loss))
        loss.backward()
        optimizer.step()

    optimization_summary = {
        "version": "v1.3",
        "optimizer": "Adam",
        "optimizer_recreated_for_round": True,
        "prox_reference_source": prox_reference_source,
        "hidden_weight_mode": config.hidden_weight_mode,
        "hidden_weight_decay": config.hidden_weight_decay,
        "hidden_weight_floor": config.hidden_weight_floor,
        "prox_weight": config.prox_weight,
        "manifold_weight": config.manifold_weight,
        "manifold_update_every": config.manifold_update_every,
        "manifold_warmup_epoch": config.manifold_warmup_epoch,
        "range_weight": config.range_weight,
        "loss_start": _loss_summary_value(histories["loss"], 0),
        "loss_end": _loss_summary_value(histories["loss"], -1),
        "loss_min": min(histories["loss"]) if histories["loss"] else None,
        "hidden_loss_start": _loss_summary_value(histories["hidden_loss"], 0),
        "hidden_loss_end": _loss_summary_value(histories["hidden_loss"], -1),
        "prox_loss_start": _loss_summary_value(histories["prox_loss"], 0),
        "prox_loss_end": _loss_summary_value(histories["prox_loss"], -1),
        "manifold_loss_start": _loss_summary_value(histories["manifold_loss"], 0),
        "manifold_loss_end": _loss_summary_value(histories["manifold_loss"], -1),
        "range_loss_start": _loss_summary_value(histories["range_loss"], 0),
        "range_loss_end": _loss_summary_value(histories["range_loss"], -1),
        "manifold_updates": manifold_updates,
        "stopped_reason": stopped_reason,
    }
    return (
        _merge_suffix(base_embed, suffix_start, suffix_param).detach(),
        optimization_summary,
    )


def _state_snapshot(metrics):
    return {
        "hidden_mean": metrics.get("embedding_hidden_mean"),
        "hidden_min": metrics.get("embedding_hidden_min"),
        "token_forward_mean": metrics.get("hidden_mean"),
        "token_forward_min": metrics.get("hidden_min"),
        "anomaly_count": metrics.get("anomaly_count"),
        "first_anomaly_position": metrics.get("first_anomaly_position"),
        "first_anomaly_reasons": metrics.get("first_anomaly_reasons"),
    }


def _select_next_state(accepted, original_embedding, original_tokens, original_metrics,
                       candidate_embedding, candidate_tokens, candidate_metrics):
    if accepted:
        return (
            candidate_embedding.detach().clone(),
            [int(item) for item in candidate_tokens],
            candidate_metrics,
        )
    return (
        original_embedding.detach().clone(),
        [int(item) for item in original_tokens],
        original_metrics,
    )


def _metric_increased(after, before):
    if after is None or before is None:
        return False
    return float(after) > float(before) + MIN_ACCEPTANCE_EPS


def _augment_anchor_off_result(result):
    result = copy.deepcopy(result)
    result["name"] = METHOD_NAME
    result["version"] = "v1.3"
    result["anchor_mode"] = "anchor_off"
    result["anchor_count"] = 0
    result["boundary_rewind_count"] = 0
    result["anchor_accepted_count"] = 0
    result["anchor_rejected_count"] = 0
    result["manifold_enabled"] = True
    result["manifold_weight"] = None
    result["manifold_updates"] = sum(
        int((event.get("optimization") or {}).get("manifold_updates") or 0)
        for event in result.get("events") or []
    )
    before = result.get("before") or {}
    after = result.get("after") or {}
    for event_index, event in enumerate(result.get("events") or []):
        if not event.get("triggered"):
            event["anchor_mode"] = "anchor_off"
            continue
        original = {
            "hidden_mean": before.get("embedding_hidden_mean") if event_index == 0 else None,
            "hidden_min": before.get("embedding_hidden_min") if event_index == 0 else None,
            "token_forward_mean": event.get("before_hidden_mean"),
            "token_forward_min": event.get("before_hidden_min"),
            "anomaly_count": event.get("before_anomaly_count"),
        }
        optimized = {
            "hidden_mean": after.get("embedding_hidden_mean") if event_index == len(result.get("events") or []) - 1 else None,
            "hidden_min": after.get("embedding_hidden_min") if event_index == len(result.get("events") or []) - 1 else None,
            "token_forward_mean": event.get("candidate_hidden_mean"),
            "token_forward_min": event.get("candidate_hidden_min"),
            "anomaly_count": event.get("candidate_anomaly_count"),
        }
        event.update({
            "anchor_mode": "anchor_off",
            "detected_anomaly_start": event.get("anomaly_position"),
            "effective_suffix_start": event.get("anomaly_position"),
            "boundary_rewound": False,
            "unstable_prefix_positions": [],
            "anchored_positions": [],
            "anchor_count": 0,
            "fixed_discrete_positions": [],
            "original_state": original,
            "anchored_baseline": copy.deepcopy(original),
            "optimized_candidate": optimized,
            "original_hidden_mean": original.get("hidden_mean"),
            "original_hidden_min": original.get("hidden_min"),
            "anchored_hidden_mean": original.get("hidden_mean"),
            "anchored_hidden_min": original.get("hidden_min"),
            "optimized_hidden_mean": optimized.get("hidden_mean"),
            "optimized_hidden_min": optimized.get("hidden_min"),
            "original_token_forward_mean": original.get("token_forward_mean"),
            "original_token_forward_min": original.get("token_forward_min"),
            "anchored_token_forward_mean": original.get("token_forward_mean"),
            "anchored_token_forward_min": original.get("token_forward_min"),
            "optimized_token_forward_mean": optimized.get("token_forward_mean"),
            "optimized_token_forward_min": optimized.get("token_forward_min"),
            "original_initial_hidden_loss": None,
            "anchored_initial_hidden_loss": None,
        })
    return result


def _log_result(f, result):
    log_section(f, METHOD_NAME)
    before = result.get("before", {})
    after = result.get("after", {})
    log_kv(f, "enabled", result.get("enabled"))
    log_kv(f, "anchor_mode", result.get("anchor_mode"))
    log_kv(f, "accept_mode", result.get("accept_mode"))
    log_kv(f, "before_accuracy", format_metric(before.get("accuracy")))
    log_kv(f, "before_hidden_mean", format_metric(before.get("hidden_mean")))
    log_kv(f, "before_embedding_hidden_mean", format_metric(before.get("embedding_hidden_mean")))
    for event in result.get("events", []):
        if not event.get("triggered"):
            log_line(f, "  suffix_round {}/{}: triggered=false, reason={}".format(
                event.get("round"), event.get("max_rounds"), event.get("reason")
            ), console=False)
            continue
        optimization = event.get("optimization") or {}
        log_line(f, (
            "  suffix_round {}/{}: detected_start={}, effective_start={}, rewound={}, "
            "anchor_mode={}, anchor_count={}, accepted={}, accept_reason={}, "
            "original_hidden_mean={}, anchored_hidden_mean={}, optimized_hidden_mean={}, "
            "original_token_forward_mean={}, optimized_token_forward_mean={}, "
            "anomalies={}->{}, initial_hidden_loss={}->{}, "
            "hidden_loss={}->{}, prox_loss={}->{}, manifold_loss={}->{}, "
            "range_loss={}->{}, loss={}->{}, changed_positions={}, stopped_reason={}"
        ).format(
            event.get("round"),
            event.get("max_rounds"),
            event.get("detected_anomaly_start"),
            event.get("effective_suffix_start"),
            event.get("boundary_rewound"),
            event.get("anchor_mode"),
            event.get("anchor_count"),
            event.get("accepted"),
            event.get("accept_reason"),
            format_metric(event.get("original_hidden_mean")),
            format_metric(event.get("anchored_hidden_mean")),
            format_metric(event.get("optimized_hidden_mean")),
            format_metric(event.get("original_token_forward_mean")),
            format_metric(event.get("optimized_token_forward_mean")),
            event.get("before_anomaly_count"),
            event.get("candidate_anomaly_count"),
            format_metric(event.get("original_initial_hidden_loss")),
            format_metric(event.get("anchored_initial_hidden_loss")),
            format_metric(optimization.get("hidden_loss_start")),
            format_metric(optimization.get("hidden_loss_end")),
            format_metric(optimization.get("prox_loss_start")),
            format_metric(optimization.get("prox_loss_end")),
            format_metric(optimization.get("manifold_loss_start")),
            format_metric(optimization.get("manifold_loss_end")),
            format_metric(optimization.get("range_loss_start")),
            format_metric(optimization.get("range_loss_end")),
            format_metric(optimization.get("loss_start")),
            format_metric(optimization.get("loss_end")),
            event.get("changed_positions"),
            event.get("stopped_reason") or optimization.get("stopped_reason"),
        ), console=False)
    log_kv(f, "after_accuracy", format_metric(after.get("accuracy")))
    log_kv(f, "after_hidden_mean", format_metric(after.get("hidden_mean")))
    log_kv(f, "after_embedding_hidden_mean", format_metric(after.get("embedding_hidden_mean")))
    log_kv(f, "boundary_rewind_count", result.get("boundary_rewind_count"))
    log_kv(f, "anchor_accepted_count", result.get("anchor_accepted_count"))
    log_kv(f, "anchor_rejected_count", result.get("anchor_rejected_count"))
    log_kv(f, "changed_positions", result.get("changed_positions"))
    log_kv(f, "reason", result.get("reason"))


def _run_anchor_off(model, embed_layer, optimized_embedding, target_hidden_state,
                    attention_mask, layer_id, register_layer_hooks, tokenizer,
                    total_input_ids, config, filter_nonascii, add_perplexity,
                    top_k_ppl, top_k_cos, invert_method, eval_start_pos,
                    embedding_top_indices, select_candidate_from_top_indices,
                    get_perplexity, forward_and_get_last_hidden_state, log_file):
    v12_config = copy.copy(config)
    v12_config.log_enabled = False
    final_embedding, v12_result = v12.run_suffix_reoptimization_v1_2(
        model=model,
        embed_layer=embed_layer,
        optimized_embedding=optimized_embedding,
        target_hidden_state=target_hidden_state,
        attention_mask=attention_mask,
        layer_id=layer_id,
        register_layer_hooks=register_layer_hooks,
        tokenizer=tokenizer,
        total_input_ids=total_input_ids,
        config=v12_config,
        filter_nonascii=filter_nonascii,
        add_perplexity=add_perplexity,
        top_k_ppl=top_k_ppl,
        top_k_cos=top_k_cos,
        invert_method=invert_method,
        eval_start_pos=eval_start_pos,
        embedding_top_indices=embedding_top_indices,
        select_candidate_from_top_indices=select_candidate_from_top_indices,
        get_perplexity=get_perplexity,
        forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
        log_file=None,
    )
    result = _augment_anchor_off_result(v12_result)
    result["manifold_enabled"] = float(config.manifold_weight) > 0.0
    result["manifold_weight"] = config.manifold_weight
    if config.log_enabled and log_file is not None:
        _log_result(log_file, result)
    return final_embedding, result


def run_suffix_reoptimization_v1_3(model, embed_layer, optimized_embedding, target_hidden_state,
                                   attention_mask, layer_id, register_layer_hooks, tokenizer,
                                   total_input_ids, config, filter_nonascii=True,
                                   add_perplexity=True, top_k_ppl=10, top_k_cos=10,
                                   invert_method="cosine", eval_start_pos=0,
                                   embedding_top_indices=None,
                                   select_candidate_from_top_indices=None,
                                   get_perplexity=None,
                                   forward_and_get_last_hidden_state=None,
                                   log_file=None):
    anchor_mode = _anchor_mode(config)
    if anchor_mode == "anchor_off":
        return _run_anchor_off(
            model, embed_layer, optimized_embedding, target_hidden_state,
            attention_mask, layer_id, register_layer_hooks, tokenizer,
            total_input_ids, config, filter_nonascii, add_perplexity,
            top_k_ppl, top_k_cos, invert_method, eval_start_pos,
            embedding_top_indices, select_candidate_from_top_indices,
            get_perplexity, forward_and_get_last_hidden_state, log_file,
        )

    if not config.enabled or config.max_rounds <= 0:
        result = {
            "name": METHOD_NAME,
            "version": "v1.3",
            "enabled": bool(config.enabled),
            "skipped": True,
            "anchor_mode": anchor_mode,
            "anchor_count": 0,
            "boundary_rewind_count": 0,
            "anchor_accepted_count": 0,
            "anchor_rejected_count": 0,
            "reason": "disabled" if not config.enabled else "max_rounds <= 0",
            "events": [],
        }
        return optimized_embedding, result

    required_helpers = {
        "embedding_top_indices": embedding_top_indices,
        "select_candidate_from_top_indices": select_candidate_from_top_indices,
        "get_perplexity": get_perplexity,
        "forward_and_get_last_hidden_state": forward_and_get_last_hidden_state,
    }
    missing_helpers = [name for name, value in required_helpers.items() if value is None]
    if missing_helpers:
        raise ValueError("missing {} helpers: {}".format(
            METHOD_NAME, ", ".join(missing_helpers)
        ))

    scan_start = max(int(eval_start_pos), 1)
    current_embedding = optimized_embedding.detach().clone()
    before_acc, before_text, before_tokens = _rerank_positions(
        current_embedding,
        None,
        eval_start_pos,
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
    current_tokens = [int(item) for item in before_tokens]
    current_metrics = _evaluate_state(
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
    current_metrics["accuracy"] = before_acc
    current_metrics["text"] = before_text
    before_metrics = current_metrics
    events = []
    all_changed_positions = set()
    rejected_boundaries = set()
    accepted_changed_positions = set()
    scan_pos = scan_start
    attempts = 0
    max_rounds = max(0, int(config.max_rounds))
    accept_mode = _accept_mode(config)
    last_failed_effective_start = None

    while attempts < max_rounds:
        metrics_for_scan = _evaluate_state(
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
        seq_len = int(current_embedding.shape[1])
        valid_positions = _valid_positions(attention_mask, seq_len)
        valid_set = set(valid_positions)
        anomalies = [
            anomaly
            for anomaly in metrics_for_scan["_anomalies"]
            if int(anomaly.get("position", -1)) in valid_set
        ]
        if not anomalies:
            if attempts == 0:
                events.append({
                    "round": 1,
                    "max_rounds": max_rounds,
                    "triggered": False,
                    "anchor_mode": anchor_mode,
                    "accept_mode": accept_mode,
                    "reason": "no anomaly found",
                })
            break

        anomaly = anomalies[0]
        detected_anomaly_start = int(anomaly["position"])
        effective_suffix_start, unstable_positions, boundary_rewound = (
            _resolve_effective_suffix_start(
                detected_anomaly_start,
                anchor_mode,
                eval_start_pos,
                current_metrics.get("_anomalies") or [],
                rejected_boundaries,
                accepted_changed_positions,
                valid_positions,
            )
        )
        attempts += 1

        original_embedding = current_embedding.detach().clone()
        original_tokens = [int(item) for item in current_tokens]
        original_metrics = current_metrics
        anchored_embedding, anchored_positions, fixed_positions = (
            _build_anchored_candidate(
                original_embedding,
                original_tokens,
                embed_layer,
                effective_suffix_start,
                attention_mask,
                eval_start_pos,
                anchor_mode,
            )
        )
        anchored_metrics = _evaluate_state(
            model,
            anchored_embedding,
            original_tokens,
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
        anchored_metrics["accuracy"] = original_metrics["accuracy"]
        anchored_metrics["text"] = original_metrics["text"]
        original_initial_hidden_loss = _initial_hidden_loss_from_metrics(
            original_metrics,
            effective_suffix_start,
            config,
        )
        anchored_initial_hidden_loss = _initial_hidden_loss_from_metrics(
            anchored_metrics,
            effective_suffix_start,
            config,
        )
        prox_reference_suffix = original_embedding[
            :, effective_suffix_start:, :
        ].detach().clone()
        candidate_embedding, optimization_summary = _optimize_suffix(
            model,
            anchored_embedding,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            effective_suffix_start,
            config,
            embed_layer,
            invert_method,
            prox_reference_suffix=prox_reference_suffix,
        )
        candidate_acc, candidate_text, candidate_tokens = _rerank_positions(
            candidate_embedding,
            original_tokens,
            effective_suffix_start,
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
            scan_start,
            config,
            forward_and_get_last_hidden_state,
        )
        candidate_metrics["accuracy"] = candidate_acc
        candidate_metrics["text"] = candidate_text
        changed_positions = _changed_positions(
            original_tokens,
            candidate_tokens,
            effective_suffix_start,
        )
        accepted, accept_reason = _accept_candidate(
            original_metrics,
            candidate_metrics,
            changed_positions,
            effective_suffix_start,
            config,
        )
        original_snapshot = _state_snapshot(original_metrics)
        anchored_snapshot = _state_snapshot(anchored_metrics)
        optimized_snapshot = _state_snapshot(candidate_metrics)
        token_forward_improved = (
            _metric_increased(
                optimized_snapshot["token_forward_mean"],
                original_snapshot["token_forward_mean"],
            )
            or _metric_increased(
                optimized_snapshot["token_forward_min"],
                original_snapshot["token_forward_min"],
            )
        )
        anomaly_count_improved = (
            optimized_snapshot["anomaly_count"] is not None
            and original_snapshot["anomaly_count"] is not None
            and int(optimized_snapshot["anomaly_count"])
            < int(original_snapshot["anomaly_count"])
        )
        repeated_failed_boundary = (
            not accepted
            and last_failed_effective_start == effective_suffix_start
            and not token_forward_improved
            and not anomaly_count_improved
        )
        stopped_reason = None
        if repeated_failed_boundary:
            stopped_reason = (
                "repeated_boundary_no_token_forward_or_anomaly_improvement"
            )

        event = {
            "round": attempts,
            "max_rounds": max_rounds,
            "triggered": True,
            "anchor_mode": anchor_mode,
            "anomaly_position": detected_anomaly_start,
            "detected_anomaly_start": detected_anomaly_start,
            "effective_suffix_start": effective_suffix_start,
            "boundary_rewound": boundary_rewound,
            "unstable_prefix_positions": unstable_positions,
            "anchored_positions": anchored_positions,
            "anchor_count": len(anchored_positions),
            "fixed_discrete_positions": fixed_positions,
            "anomaly_reasons": anomaly.get("reasons"),
            "anomaly_detection_mode": anomaly.get("anomaly_detection_mode"),
            "anomaly_score": anomaly.get("anomaly_score"),
            "anomaly_hidden_similarity": anomaly.get("hidden_similarity"),
            "anomaly_previous_hidden_similarity": anomaly.get("previous_hidden_similarity"),
            "anomaly_hidden_drop": anomaly.get("hidden_drop"),
            "anomaly_hidden_adaptive_mean": anomaly.get("hidden_adaptive_mean"),
            "anomaly_hidden_adaptive_std": anomaly.get("hidden_adaptive_std"),
            "anomaly_hidden_adaptive_cutoff": anomaly.get("hidden_adaptive_cutoff"),
            "anomaly_hidden_adaptive_z": anomaly.get("hidden_adaptive_z"),
            "anomaly_token_forward_similarity": anomaly.get("token_forward_similarity"),
            "anomaly_token_forward_mean": anomaly.get("token_forward_mean"),
            "anomaly_token_forward_std": anomaly.get("token_forward_std"),
            "anomaly_token_forward_cutoff": anomaly.get("token_forward_cutoff"),
            "anomaly_token_forward_z": anomaly.get("token_forward_z"),
            "anomaly_token_forward_drop": anomaly.get("token_forward_drop"),
            "anomaly_token_forward_drop_mean": anomaly.get("token_forward_drop_mean"),
            "anomaly_token_forward_drop_std": anomaly.get("token_forward_drop_std"),
            "anomaly_token_forward_drop_cutoff": anomaly.get("token_forward_drop_cutoff"),
            "anomaly_token_forward_drop_z": anomaly.get("token_forward_drop_z"),
            "original_state": original_snapshot,
            "anchored_baseline": anchored_snapshot,
            "optimized_candidate": optimized_snapshot,
            "original_hidden_mean": original_snapshot["hidden_mean"],
            "original_hidden_min": original_snapshot["hidden_min"],
            "anchored_hidden_mean": anchored_snapshot["hidden_mean"],
            "anchored_hidden_min": anchored_snapshot["hidden_min"],
            "optimized_hidden_mean": optimized_snapshot["hidden_mean"],
            "optimized_hidden_min": optimized_snapshot["hidden_min"],
            "original_token_forward_mean": original_snapshot["token_forward_mean"],
            "original_token_forward_min": original_snapshot["token_forward_min"],
            "anchored_token_forward_mean": anchored_snapshot["token_forward_mean"],
            "anchored_token_forward_min": anchored_snapshot["token_forward_min"],
            "optimized_token_forward_mean": optimized_snapshot["token_forward_mean"],
            "optimized_token_forward_min": optimized_snapshot["token_forward_min"],
            "original_initial_hidden_loss": original_initial_hidden_loss,
            "anchored_initial_hidden_loss": anchored_initial_hidden_loss,
            "before_accuracy": original_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "before_hidden_mean": original_metrics["hidden_mean"],
            "candidate_hidden_mean": candidate_metrics["hidden_mean"],
            "before_hidden_min": original_metrics["hidden_min"],
            "candidate_hidden_min": candidate_metrics["hidden_min"],
            "before_suffix_hidden_mean": _suffix_mean(
                original_metrics, effective_suffix_start
            ),
            "candidate_suffix_hidden_mean": _suffix_mean(
                candidate_metrics, effective_suffix_start
            ),
            "before_anomaly_count": original_metrics["anomaly_count"],
            "anchored_anomaly_count": anchored_metrics["anomaly_count"],
            "candidate_anomaly_count": candidate_metrics["anomaly_count"],
            "token_forward_improved": token_forward_improved,
            "anomaly_count_improved": anomaly_count_improved,
            "changed_positions": changed_positions,
            "accepted": accepted,
            "accept_mode": accept_mode,
            "accept_reason": accept_reason,
            "optimization": optimization_summary,
            "stopped_reason": stopped_reason,
        }
        events.append(event)

        current_embedding, current_tokens, current_metrics = _select_next_state(
            accepted,
            original_embedding,
            original_tokens,
            original_metrics,
            candidate_embedding,
            candidate_tokens,
            candidate_metrics,
        )
        if accepted:
            last_failed_effective_start = None
            for pos in changed_positions:
                all_changed_positions.add(pos)
                accepted_changed_positions.add(pos)
            scan_pos = scan_start
        else:
            rejected_boundaries.add(effective_suffix_start)
            last_failed_effective_start = effective_suffix_start
            scan_pos = detected_anomaly_start + 1
            if repeated_failed_boundary:
                break

    after_metrics = current_metrics
    triggered = any(event.get("triggered") for event in events)
    accepted_count = sum(bool(event.get("accepted")) for event in events)
    rejected_count = sum(
        bool(event.get("triggered")) and not bool(event.get("accepted"))
        for event in events
    )
    rewind_count = sum(bool(event.get("boundary_rewound")) for event in events)
    anchor_count = sum(int(event.get("anchor_count") or 0) for event in events)
    if not triggered:
        reason = "no anomaly found"
    elif accepted_count:
        reason = "accepted {} suffix round(s)".format(accepted_count)
    else:
        reason = "no suffix round accepted"
    result = {
        "name": METHOD_NAME,
        "version": "v1.3",
        "enabled": True,
        "skipped": False,
        "anchor_mode": anchor_mode,
        "accept_mode": accept_mode,
        "pre_acc": before_metrics["accuracy"],
        "post_acc": after_metrics["accuracy"],
        "triggered": triggered,
        "accepted": accepted_count > 0,
        "reason": reason,
        "before": _public_metrics(before_metrics),
        "after": _public_metrics(after_metrics),
        "events": events,
        "changed_positions": sorted(all_changed_positions),
        "anchor_count": anchor_count,
        "boundary_rewind_count": rewind_count,
        "anchor_accepted_count": accepted_count,
        "anchor_rejected_count": rejected_count,
        "manifold_enabled": float(config.manifold_weight) > 0.0,
        "manifold_weight": config.manifold_weight,
        "manifold_updates": sum(
            int((event.get("optimization") or {}).get("manifold_updates") or 0)
            for event in events
        ),
        "rejected_boundaries": sorted(rejected_boundaries),
        "accepted_changed_positions": sorted(accepted_changed_positions),
        "final_tokens": [int(item) for item in current_tokens],
        "final_text": after_metrics["text"],
        "final_accuracy": after_metrics["accuracy"],
    }
    if config.log_enabled and log_file is not None:
        _log_result(log_file, result)
    return current_embedding, result


SuffixReoptimizationConfig = SuffixReoptimizationV13Config
run_suffix_reoptimization = run_suffix_reoptimization_v1_3
