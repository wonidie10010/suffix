import time
from dataclasses import dataclass

import torch

from .CGMR_v1_0 import (
    _as_float,
    _build_candidate_pool,
    _candidate_score_records,
    _normalize_and_aggregate,
    _score_candidates_by_layers,
    resolve_effective_layers,
)


METHOD_NAME = "CGMR_v1.1"


@dataclass
class CGMRV11Config:
    enabled: bool = False
    log_enabled: bool = True
    layer_offsets: tuple = (0, 1, 2)
    layer_weights: tuple = (0.5, 0.3, 0.2)
    normalization: str = "zscore"
    consistency_weight: float = 0.0
    relative_margin_epsilon: float = 1e-6
    relative_margin_risk_weight: float = 0.7
    low_score_risk_weight: float = 0.3
    score_drop_risk_weight: float = 0.0
    low_score_threshold: float = 0.80
    min_risk_score: float = 0.20
    risk_top_k: int = 20
    max_accepted_repairs: int = 10
    max_candidates: int = 32
    candidate_batch_size: int = 16
    min_enhanced_gain: float = 0.05
    min_enhanced_margin: float = 0.05
    max_layer_l_drop: float = 0.02

    def __post_init__(self):
        self.layer_offsets = tuple(int(value) for value in self.layer_offsets)
        self.layer_weights = tuple(float(value) for value in self.layer_weights)
        if not self.layer_offsets or 0 not in self.layer_offsets:
            raise ValueError("cgmr_v1_1_layer_offsets must contain 0")
        if any(offset < 0 for offset in self.layer_offsets):
            raise ValueError("cgmr_v1_1_layer_offsets must be non-negative")
        if len(set(self.layer_offsets)) != len(self.layer_offsets):
            raise ValueError("cgmr_v1_1_layer_offsets must not contain duplicates")
        if len(self.layer_offsets) != len(self.layer_weights):
            raise ValueError("cgmr_v1_1_layer_offsets and layer_weights must have equal length")
        if any(weight < 0 for weight in self.layer_weights) or sum(self.layer_weights) <= 0:
            raise ValueError("cgmr_v1_1_layer_weights must be non-negative with a positive sum")
        if self.normalization != "zscore":
            raise ValueError("cgmr_v1_1_normalization must be zscore")
        if self.consistency_weight < 0:
            raise ValueError("cgmr_v1_1_consistency_weight must be non-negative")
        if self.relative_margin_epsilon <= 0:
            raise ValueError("cgmr_v1_1_relative_margin_epsilon must be positive")
        risk_weights = (
            self.relative_margin_risk_weight,
            self.low_score_risk_weight,
            self.score_drop_risk_weight,
        )
        if any(weight < 0 for weight in risk_weights) or sum(risk_weights) <= 0:
            raise ValueError("CGMR v1.1 risk weights must be non-negative with a positive sum")
        if not -1.0 <= self.low_score_threshold <= 1.0:
            raise ValueError("cgmr_v1_1_low_score_threshold must be in [-1, 1]")
        if not 0.0 <= self.min_risk_score <= 1.0:
            raise ValueError("cgmr_v1_1_min_risk_score must be in [0, 1]")
        if self.risk_top_k <= 0 or self.max_accepted_repairs <= 0:
            raise ValueError("CGMR v1.1 risk_top_k and max_accepted_repairs must be positive")
        if self.max_candidates <= 1 or self.candidate_batch_size <= 0:
            raise ValueError("CGMR candidate limits must be positive and allow at least two candidates")
        if self.min_enhanced_gain < 0 or self.min_enhanced_margin < 0:
            raise ValueError("CGMR enhanced gain and margin thresholds must be non-negative")
        if self.max_layer_l_drop < 0:
            raise ValueError("cgmr_v1_1_max_layer_l_drop must be non-negative")


def _clamp_unit(value):
    return max(0.0, min(1.0, float(value)))


def _relative_margin(top1_score, top2_score, epsilon):
    if top2_score is None:
        return None
    absolute_margin = max(0.0, float(top1_score) - float(top2_score))
    denominator = max(1.0 - float(top2_score), float(epsilon))
    return absolute_margin / denominator


def _risk_score(top1_score, relative_margin, previous_top1_score, config):
    relative_margin_risk = 1.0 - _clamp_unit(relative_margin)
    low_score_scale = max(abs(float(config.low_score_threshold)), config.relative_margin_epsilon)
    low_score_risk = _clamp_unit(
        (float(config.low_score_threshold) - float(top1_score)) / low_score_scale
    )
    score_drop_risk = 0.0
    if previous_top1_score is not None:
        score_drop_risk = _clamp_unit(float(previous_top1_score) - float(top1_score))
    weight_sum = (
        config.relative_margin_risk_weight
        + config.low_score_risk_weight
        + config.score_drop_risk_weight
    )
    risk_score = (
        config.relative_margin_risk_weight * relative_margin_risk
        + config.low_score_risk_weight * low_score_risk
        + config.score_drop_risk_weight * score_drop_risk
    ) / weight_sum
    return _clamp_unit(risk_score), {
        "relative_margin_risk": relative_margin_risk,
        "low_score_risk": low_score_risk,
        "score_drop_risk": score_drop_risk,
    }


def _rank_and_select_risk_records(risk_records, min_risk_score, risk_top_k):
    scored = [record for record in risk_records if record.get("risk_score") is not None]
    ranked = sorted(scored, key=lambda record: (-record["risk_score"], record["position"]))
    for rank, record in enumerate(ranked, start=1):
        record["risk_rank"] = rank
        record["above_min_risk"] = record["risk_score"] >= min_risk_score
        record["selected_for_rerank"] = False
    eligible = [record for record in ranked if record["above_min_risk"]]
    selected = eligible[:risk_top_k]
    for record in selected:
        record["selected_for_rerank"] = True
    for record in risk_records:
        if record.get("risk_score") is None:
            record["risk_rank"] = None
            record["above_min_risk"] = False
            record["selected_for_rerank"] = False
    return selected, ranked


def _empty_result(config, reason, configured_layers, effective_layers=None,
                  effective_weights=None, filtered_layers=None):
    return {
        "name": METHOD_NAME,
        "enabled": bool(config.enabled),
        "skipped": True,
        "reason": reason,
        "configured_layers": configured_layers,
        "effective_layers": effective_layers or [],
        "effective_weights": effective_weights or [],
        "filtered_layers": filtered_layers or [],
        "risk_top_k": config.risk_top_k,
        "min_risk_score": config.min_risk_score,
        "max_accepted_repairs": config.max_accepted_repairs,
        "triggered": False,
        "accepted": False,
        "trigger_count": 0,
        "risk_scan_position_count": 0,
        "risk_position_count": 0,
        "selected_risk_positions": [],
        "processing_order": [],
        "processed_positions": [],
        "evaluated_position_count": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "changed_positions": [],
        "elapsed_seconds": 0.0,
        "risk_records": [],
        "events": [],
    }


def _build_and_score_layer_l(
        position, tokens, model, embed_layer, optimized_embedding,
        target_hidden_states, attention_mask, target_layer, model_device,
        config, tokenizer, filter_nonascii, add_perplexity, top_k_ppl,
        top_k_cos, invert_method, embedding_top_indices,
        select_candidate_from_top_indices, get_perplexity):
    candidates, candidate_sources, embedding_top1 = _build_candidate_pool(
        position,
        tokens,
        optimized_embedding,
        embed_layer,
        tokenizer,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        invert_method,
        config.max_candidates,
        embedding_top_indices,
        select_candidate_from_top_indices,
        get_perplexity,
        model,
        target_layer,
    )
    layer_l_raw = None
    if len(candidates) >= 2:
        layer_l_raw = _score_candidates_by_layers(
            model,
            tokens,
            position,
            candidates,
            target_hidden_states,
            [target_layer],
            attention_mask,
            config.candidate_batch_size,
            model_device,
        )[target_layer]
    return candidates, candidate_sources, embedding_top1, layer_l_raw


def _layer_l_summary(candidate_ids, layer_l_scores, epsilon):
    ranked = sorted(
        range(len(candidate_ids)),
        key=lambda idx: layer_l_scores[idx],
        reverse=True,
    )
    top1_idx = ranked[0]
    top2_idx = ranked[1]
    top1_score = float(layer_l_scores[top1_idx])
    top2_score = float(layer_l_scores[top2_idx])
    absolute_margin = top1_score - top2_score
    relative_margin = _relative_margin(top1_score, top2_score, epsilon)
    return {
        "single_layer_top1": int(candidate_ids[top1_idx]),
        "single_layer_top2": int(candidate_ids[top2_idx]),
        "top1_score": top1_score,
        "top2_score": top2_score,
        "absolute_margin": absolute_margin,
        "relative_margin": relative_margin,
    }


def run_cgmr_v1_1(
        model, embed_layer, optimized_embedding, current_tokens,
        target_hidden_states, attention_mask, target_layer, model_layer_count,
        model_device, config, tokenizer, filter_nonascii, add_perplexity,
        top_k_ppl, top_k_cos, invert_method, eval_start_pos,
        embedding_top_indices, select_candidate_from_top_indices,
        get_perplexity):
    configured_layers = [int(target_layer) + offset for offset in config.layer_offsets]
    if not config.enabled:
        return list(current_tokens), _empty_result(config, "disabled", configured_layers)

    effective_layers, effective_weights, filtered_layers = resolve_effective_layers(
        target_layer,
        model_layer_count,
        config.layer_offsets,
        config.layer_weights,
    )
    if len(effective_layers) < 2:
        return list(current_tokens), _empty_result(
            config,
            "insufficient_valid_layers",
            configured_layers,
            effective_layers,
            effective_weights,
            filtered_layers,
        )
    missing_targets = [layer_id for layer_id in effective_layers if layer_id not in target_hidden_states]
    if missing_targets:
        raise ValueError("missing CGMR target hidden states for layers {}".format(missing_targets))

    started = time.time()
    tokens = [int(item) for item in current_tokens]
    risk_records = []
    previous_top1_score = None
    scan_start = max(0, int(eval_start_pos))

    for position in range(scan_start, len(tokens)):
        candidates, candidate_sources, embedding_top1, layer_l_raw = _build_and_score_layer_l(
            position,
            tokens,
            model,
            embed_layer,
            optimized_embedding,
            target_hidden_states,
            attention_mask,
            target_layer,
            model_device,
            config,
            tokenizer,
            filter_nonascii,
            add_perplexity,
            top_k_ppl,
            top_k_cos,
            invert_method,
            embedding_top_indices,
            select_candidate_from_top_indices,
            get_perplexity,
        )
        record = {
            "position": position,
            "score_source": "initial_full_sequence_scan",
            "candidate_count": len(candidates),
            "candidate_ids": [int(token_id) for token_id in candidates],
            "candidate_sources": {
                str(token_id): candidate_sources[int(token_id)]
                for token_id in candidates
            },
            "embedding_top1": embedding_top1,
        }
        if layer_l_raw is None:
            record.update({
                "eligible_for_risk_scoring": False,
                "reason": "insufficient_candidates",
                "risk_score": None,
            })
            risk_records.append(record)
            previous_top1_score = None
            continue

        summary = _layer_l_summary(
            candidates,
            layer_l_raw,
            config.relative_margin_epsilon,
        )
        risk_score, risk_components = _risk_score(
            summary["top1_score"],
            summary["relative_margin"],
            previous_top1_score,
            config,
        )
        previous_top1_score = summary["top1_score"]
        record.update(summary)
        record.update({
            "eligible_for_risk_scoring": True,
            "risk_score": risk_score,
            "risk_components": risk_components,
            "layer_l_scores": [float(value) for value in layer_l_raw],
        })
        risk_records.append(record)

    selected_records, ranked_records = _rank_and_select_risk_records(
        risk_records,
        config.min_risk_score,
        config.risk_top_k,
    )
    selected_risk_positions = [record["position"] for record in selected_records]
    processing_order = sorted(selected_risk_positions)
    selected_by_position = {record["position"]: record for record in selected_records}

    events = []
    processed_positions = []
    changed_positions = []
    accepted_count = 0

    for position in processing_order:
        if accepted_count >= config.max_accepted_repairs:
            break
        processed_positions.append(position)
        selection_record = selected_by_position[position]
        candidates, candidate_sources, embedding_top1, layer_l_raw = _build_and_score_layer_l(
            position,
            tokens,
            model,
            embed_layer,
            optimized_embedding,
            target_hidden_states,
            attention_mask,
            target_layer,
            model_device,
            config,
            tokenizer,
            filter_nonascii,
            add_perplexity,
            top_k_ppl,
            top_k_cos,
            invert_method,
            embedding_top_indices,
            select_candidate_from_top_indices,
            get_perplexity,
        )
        current_token = int(tokens[position])
        if layer_l_raw is None:
            events.append({
                "position": position,
                "score_source": "recomputed_from_current_tokens",
                "selection_risk_rank": selection_record["risk_rank"],
                "risk_score": selection_record["risk_score"],
                "relative_margin": selection_record["relative_margin"],
                "candidate_count": len(candidates),
                "old_token": current_token,
                "new_token": current_token,
                "accepted": False,
                "accept_reason": "insufficient_candidates",
                "candidate_scores": [],
            })
            continue

        recomputed_summary = _layer_l_summary(
            candidates,
            layer_l_raw,
            config.relative_margin_epsilon,
        )
        remaining_layers = [layer_id for layer_id in effective_layers if layer_id != target_layer]
        raw_scores_by_layer = {target_layer: layer_l_raw}
        if remaining_layers:
            raw_scores_by_layer.update(_score_candidates_by_layers(
                model,
                tokens,
                position,
                candidates,
                target_hidden_states,
                remaining_layers,
                attention_mask,
                config.candidate_batch_size,
                model_device,
            ))
        normalized_by_layer, enhanced_scores, consistency_penalty = _normalize_and_aggregate(
            raw_scores_by_layer,
            effective_layers,
            effective_weights,
            config.consistency_weight,
        )
        ranked = torch.argsort(enhanced_scores, descending=True)
        best_idx = int(ranked[0].item())
        second_idx = int(ranked[1].item())
        current_idx = candidates.index(current_token)
        new_token = int(candidates[best_idx])
        gain = _as_float(enhanced_scores[best_idx] - enhanced_scores[current_idx])
        enhanced_margin = _as_float(enhanced_scores[best_idx] - enhanced_scores[second_idx])
        layer_l_delta = float(layer_l_raw[best_idx] - layer_l_raw[current_idx])

        accepted = False
        if new_token == current_token:
            accept_reason = "current_token_already_best"
        elif gain < config.min_enhanced_gain:
            accept_reason = "enhanced_gain_below_threshold"
        elif enhanced_margin < config.min_enhanced_margin:
            accept_reason = "enhanced_margin_below_threshold"
        elif layer_l_delta < -config.max_layer_l_drop:
            accept_reason = "layer_l_drop_exceeds_tolerance"
        else:
            accepted = True
            accept_reason = "accepted"
            tokens[position] = new_token
            changed_positions.append(position)
            accepted_count += 1

        events.append({
            "position": position,
            "score_source": "recomputed_from_current_tokens",
            "selection_risk_rank": selection_record["risk_rank"],
            "risk_score": selection_record["risk_score"],
            "relative_margin": selection_record["relative_margin"],
            "selection_absolute_margin": selection_record["absolute_margin"],
            "recomputed_absolute_margin": recomputed_summary["absolute_margin"],
            "recomputed_relative_margin": recomputed_summary["relative_margin"],
            "candidate_count": len(candidates),
            "embedding_top1": embedding_top1,
            "single_layer_top1": recomputed_summary["single_layer_top1"],
            "single_layer_top2": recomputed_summary["single_layer_top2"],
            "single_layer_top1_score": recomputed_summary["top1_score"],
            "single_layer_top2_score": recomputed_summary["top2_score"],
            "old_token": current_token,
            "new_token": new_token,
            "enhanced_gain": gain,
            "enhanced_margin": enhanced_margin,
            "layer_l_delta": layer_l_delta,
            "accepted": accepted,
            "accept_reason": accept_reason,
            "effective_layers": effective_layers,
            "effective_weights": effective_weights,
            "candidate_scores": _candidate_score_records(
                candidates,
                candidate_sources,
                raw_scores_by_layer,
                normalized_by_layer,
                enhanced_scores,
                consistency_penalty,
                effective_layers,
            ),
        })

    evaluated_count = len(events)
    risk_position_count = sum(
        bool(record.get("above_min_risk")) for record in ranked_records
    )
    result = {
        "name": METHOD_NAME,
        "enabled": True,
        "skipped": False,
        "reason": "completed" if selected_records else "no_risk_positions_above_threshold",
        "configured_layers": configured_layers,
        "effective_layers": effective_layers,
        "effective_weights": effective_weights,
        "filtered_layers": filtered_layers,
        "risk_top_k": config.risk_top_k,
        "min_risk_score": config.min_risk_score,
        "max_accepted_repairs": config.max_accepted_repairs,
        "triggered": bool(selected_records),
        "accepted": accepted_count > 0,
        "trigger_count": evaluated_count,
        "risk_scan_position_count": len(risk_records),
        "risk_position_count": risk_position_count,
        "selected_risk_positions": selected_risk_positions,
        "processing_order": processing_order,
        "processed_positions": processed_positions,
        "evaluated_position_count": evaluated_count,
        "accepted_count": accepted_count,
        "rejected_count": evaluated_count - accepted_count,
        "changed_positions": changed_positions,
        "elapsed_seconds": time.time() - started,
        "risk_records": risk_records,
        "events": events,
    }
    return tokens, result


run_cgmr = run_cgmr_v1_1
