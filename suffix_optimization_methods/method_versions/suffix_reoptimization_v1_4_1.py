from dataclasses import dataclass
import copy
import math

import torch

from . import suffix_reoptimization_v1_4 as v14


MIN_ACCEPTANCE_EPS = v14.MIN_ACCEPTANCE_EPS
METHOD_NAME = "suffix_reoptimization_v1.4.1"
DEFAULT_ACCEPT_MODE = v14.DEFAULT_ACCEPT_MODE
ACCEPT_MODES = v14.ACCEPT_MODES
SCHEDULE_MODES = v14.SCHEDULE_MODES
CONFIDENCE_MODE_ALIASES = {
    "absolute": "absolute",
    "hybrid": "absolute",
    "fixed": "absolute",
}


@dataclass
class SuffixReoptimizationV141Config:
    enabled: bool = False
    log_enabled: bool = True
    coarse_lr_max: float = 0.10
    coarse_lr_min: float = 0.03
    coarse_schedule: str = "cosine"
    fine_epoch: int = 50
    fine_lr_max: float = 0.01
    fine_lr_min: float = 0.001
    fine_schedule: str = "cosine"
    confidence_mode: str = "absolute"
    confidence_continuous_min: float = 0.80
    confidence_token_min: float = 0.80
    confidence_margin_min: float = 0.02
    confidence_gap_max: float = 0.10
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


_normalized_mode = v14._normalized_mode
scheduled_learning_rate = v14.scheduled_learning_rate
_valid_positions = v14._valid_positions
_rerank_positions_with_diagnostics = v14._rerank_positions_with_diagnostics
_build_anchored_baseline = v14._build_anchored_baseline
_build_fine_observation_weights = v14._build_fine_observation_weights
_base_optimize_masked_positions = v14._optimize_masked_positions
_evaluate_state = v14._evaluate_state
_public_metrics = v14._public_metrics
_changed_positions = v14._changed_positions
_accept_candidate = v14._accept_candidate
_oracle_mask_diagnostics = v14._oracle_mask_diagnostics


def _normalize_confidence_mode(value):
    mode = str(value or "").strip().lower()
    if mode not in CONFIDENCE_MODE_ALIASES:
        raise ValueError(
            "suffix v1.4.1 confidence_mode must be one of: absolute, fixed, hybrid"
        )
    return CONFIDENCE_MODE_ALIASES[mode]


def validate_suffix_reoptimization_v1_4_1_config(config):
    _normalized_mode(config.coarse_schedule, SCHEDULE_MODES, "suffix v1.4.1 coarse_schedule")
    _normalized_mode(config.fine_schedule, SCHEDULE_MODES, "suffix v1.4.1 fine_schedule")
    config.confidence_mode = _normalize_confidence_mode(config.confidence_mode)
    _normalized_mode(config.accept_mode, ACCEPT_MODES, "suffix v1.4.1 accept_mode")
    if int(config.fine_epoch) <= 0:
        raise ValueError("suffix_v1_4_1_fine_epoch must be greater than 0")
    if float(config.coarse_lr_max) <= 0.0 or float(config.coarse_lr_min) < 0.0:
        raise ValueError(
            "suffix v1.4.1 coarse learning rates must be non-negative and max must be positive"
        )
    if float(config.coarse_lr_min) > float(config.coarse_lr_max):
        raise ValueError("suffix_v1_4_1_coarse_lr_min must not exceed coarse_lr_max")
    if float(config.fine_lr_max) <= 0.0 or float(config.fine_lr_min) < 0.0:
        raise ValueError(
            "suffix v1.4.1 fine learning rates must be non-negative and max must be positive"
        )
    if float(config.fine_lr_min) > float(config.fine_lr_max):
        raise ValueError("suffix_v1_4_1_fine_lr_min must not exceed fine_lr_max")
    if int(config.fine_window) < 0:
        raise ValueError("suffix_v1_4_1_fine_window must be non-negative")
    if not 0.0 <= float(config.fine_window_decay) <= 1.0:
        raise ValueError("suffix_v1_4_1_fine_window_decay must be in [0, 1]")
    return config


def _optimize_masked_positions(*args, **kwargs):
    embedding, summary = _base_optimize_masked_positions(*args, **kwargs)
    summary = copy.deepcopy(summary)
    summary["version"] = "v1.4.1"
    return embedding, summary


def _finite_float(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _valid_token_id(value):
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer() or numeric < 0:
        return None
    return int(numeric)


def _build_confidence_mask(continuous_scores, token_scores, candidate_diagnostics,
                           valid_positions, anomaly_reasons, config):
    """Build an independent absolute confidence decision for every position."""
    mode = _normalize_confidence_mode(config.confidence_mode)
    entries = []
    high_positions = []
    low_positions = []

    for pos in [int(item) for item in valid_positions]:
        candidate = candidate_diagnostics.get(pos, {})
        continuous_raw = continuous_scores[pos] if pos < len(continuous_scores) else None
        token_raw = token_scores[pos] if pos < len(token_scores) else None
        margin_raw = candidate.get("margin")
        continuous = _finite_float(continuous_raw)
        token = _finite_float(token_raw)
        margin = _finite_float(margin_raw)
        gap = (
            max(0.0, continuous - token)
            if continuous is not None and token is not None
            else None
        )

        embedding_top1_token_id = _valid_token_id(
            candidate.get("embedding_top1_token_id")
        )
        selected_token_id = _valid_token_id(candidate.get("top1_token_id"))
        candidate_token_ids = [
            token_id
            for token_id in (
                _valid_token_id(value)
                for value in (candidate.get("candidate_token_ids") or [])
            )
            if token_id is not None
        ]
        try:
            candidate_count = int(candidate.get("candidate_count") or 0)
        except (TypeError, ValueError):
            candidate_count = 0
        candidate_agreement = (
            embedding_top1_token_id is not None
            and selected_token_id is not None
            and embedding_top1_token_id == selected_token_id
        )
        reasons = list(anomaly_reasons.get(pos) or [])

        failures = []
        if any(value is None for value in (continuous, token, margin, gap)):
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
        if candidate_count < 2 or len(candidate_token_ids) < 2:
            failures.append("fewer_than_two_candidates")
        if embedding_top1_token_id is None or selected_token_id is None:
            failures.append("missing_candidate_token_id")
        if bool(config.require_candidate_agreement) and not candidate_agreement:
            failures.append("candidate_disagreement")
        if reasons:
            failures.append("adaptive_anomaly")

        high_confidence = not failures
        entry = {
            "position": pos,
            "continuous_similarity": continuous,
            "token_forward_similarity": token,
            "margin": margin,
            "discretization_gap": gap,
            "embedding_top1_token_id": embedding_top1_token_id,
            "selected_token_id": selected_token_id,
            "candidate_count": candidate_count,
            "candidate_token_ids": candidate_token_ids,
            "candidate_agreement": candidate_agreement,
            "anomaly_reasons": reasons,
            "high_confidence": high_confidence,
            "gate_failures": failures,
        }
        entries.append(entry)
        if high_confidence:
            high_positions.append(pos)
        else:
            low_positions.append(pos)

    return {
        "mode": mode,
        "thresholds": {
            "continuous_min": float(config.confidence_continuous_min),
            "token_min": float(config.confidence_token_min),
            "margin_min": float(config.confidence_margin_min),
            "gap_max": float(config.confidence_gap_max),
            "require_candidate_agreement": bool(config.require_candidate_agreement),
        },
        "valid_positions": [int(item) for item in valid_positions],
        "high_confidence_positions": high_positions,
        "low_confidence_positions": low_positions,
        "high_confidence_count": len(high_positions),
        "low_confidence_count": len(low_positions),
        "per_position": entries,
    }


def run_suffix_reoptimization_v1_4_1(model, embed_layer, optimized_embedding,
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
            "version": "v1.4.1",
            "enabled": False,
            "skipped": True,
            "reason": "disabled",
            "events": [],
            "manifold_enabled": False,
            "manifold_weight": 0.0,
            "manifold_updates": 0,
        }
    validate_suffix_reoptimization_v1_4_1_config(config)
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
    target_ids = v14._target_ids(total_input_ids)
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
    fine_summary = copy.deepcopy(fine_summary)
    fine_summary["version"] = "v1.4.1"
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
    coarse_stage.setdefault("version", "v1.4.1")
    coarse_stage.setdefault("optimizer", "SGD")
    coarse_stage.setdefault("schedule", str(config.coarse_schedule))
    coarse_stage.setdefault("lr_start", float(config.coarse_lr_max))
    coarse_stage.setdefault("lr_end", float(config.coarse_lr_min))
    result = {
        "name": METHOD_NAME,
        "version": "v1.4.1",
        "enabled": True,
        "skipped": False,
        "accept_mode": str(config.accept_mode),
        "confidence_mode": "absolute",
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


SuffixReoptimizationConfig = SuffixReoptimizationV141Config
run_suffix_reoptimization = run_suffix_reoptimization_v1_4_1
