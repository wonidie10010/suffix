import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


METHOD_NAME = "CGMR_v1.0"
NORMALIZATION_EPS = 1e-6


@dataclass
class CGMRV10Config:
    enabled: bool = False
    log_enabled: bool = True
    layer_offsets: tuple = (0, 1, 2)
    layer_weights: tuple = (0.5, 0.3, 0.2)
    normalization: str = "zscore"
    consistency_weight: float = 0.0
    strong_margin_threshold: float = 0.01
    weak_margin_threshold: float = 0.02
    low_score_threshold: float = 0.50
    weak_signals_required: int = 2
    max_candidates: int = 32
    candidate_batch_size: int = 16
    min_enhanced_gain: float = 0.05
    min_enhanced_margin: float = 0.05
    max_layer_l_drop: float = 0.02
    max_repair_steps: int = 5

    def __post_init__(self):
        self.layer_offsets = tuple(int(value) for value in self.layer_offsets)
        self.layer_weights = tuple(float(value) for value in self.layer_weights)
        if not self.layer_offsets or 0 not in self.layer_offsets:
            raise ValueError("cgmr_v1_0_layer_offsets must contain 0")
        if any(offset < 0 for offset in self.layer_offsets):
            raise ValueError("cgmr_v1_0_layer_offsets must be non-negative")
        if len(set(self.layer_offsets)) != len(self.layer_offsets):
            raise ValueError("cgmr_v1_0_layer_offsets must not contain duplicates")
        if len(self.layer_offsets) != len(self.layer_weights):
            raise ValueError("cgmr_v1_0_layer_offsets and layer_weights must have equal length")
        if any(weight < 0 for weight in self.layer_weights) or sum(self.layer_weights) <= 0:
            raise ValueError("cgmr_v1_0_layer_weights must be non-negative with a positive sum")
        if self.normalization != "zscore":
            raise ValueError("cgmr_v1_0_normalization must be zscore")
        if self.strong_margin_threshold < 0 or self.weak_margin_threshold < 0:
            raise ValueError("CGMR margin thresholds must be non-negative")
        if self.strong_margin_threshold > self.weak_margin_threshold:
            raise ValueError("CGMR strong margin threshold must not exceed weak margin threshold")
        if self.consistency_weight < 0:
            raise ValueError("cgmr_v1_0_consistency_weight must be non-negative")
        if self.weak_signals_required <= 0 or self.weak_signals_required > 3:
            raise ValueError("cgmr_v1_0_weak_signals_required must be in [1, 3]")
        if self.max_candidates <= 1 or self.candidate_batch_size <= 0:
            raise ValueError("CGMR candidate limits must be positive and allow at least two candidates")
        if self.min_enhanced_gain < 0 or self.min_enhanced_margin < 0:
            raise ValueError("CGMR enhanced gain and margin thresholds must be non-negative")
        if self.max_layer_l_drop < 0 or self.max_repair_steps <= 0:
            raise ValueError("CGMR max_layer_l_drop must be non-negative and max_repair_steps positive")


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _layer_id_from_name(name):
    parts = name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and parts[idx + 1].isdigit() and idx + 2 == len(parts):
            return int(parts[idx + 1])
    return None


def resolve_effective_layers(target_layer, model_layer_count, layer_offsets, layer_weights):
    configured = []
    for offset, weight in zip(layer_offsets, layer_weights):
        configured.append((int(target_layer) + int(offset), float(weight)))
    valid = [(layer_id, weight) for layer_id, weight in configured if 0 <= layer_id < model_layer_count]
    filtered = [layer_id for layer_id, _ in configured if layer_id < 0 or layer_id >= model_layer_count]
    weight_sum = sum(weight for _, weight in valid)
    if weight_sum <= 0:
        return [], [], filtered
    layers = [layer_id for layer_id, _ in valid]
    weights = [weight / weight_sum for _, weight in valid]
    return layers, weights, filtered


def _register_exact_layer_hooks(model, layer_ids, destination):
    requested = set(int(layer_id) for layer_id in layer_ids)
    handles = []
    registered = set()

    def make_hook(layer_id):
        def forward_hook(module, inputs, output):
            del module, inputs
            hidden_state = output[0] if isinstance(output, tuple) else output
            destination[layer_id] = hidden_state
        return forward_hook

    for name, module in model.named_modules():
        layer_id = _layer_id_from_name(name)
        if layer_id in requested and layer_id not in registered:
            handles.append(module.register_forward_hook(make_hook(layer_id)))
            registered.add(layer_id)
    if registered != requested:
        for handle in handles:
            handle.remove()
        missing = sorted(requested - registered)
        raise ValueError("could not register hooks for CGMR layers {}".format(missing))
    return handles


def collect_hidden_states_by_layer(
        model, layer_ids, input_ids=None, attention_mask=None,
        use_cache=None):
    if input_ids is None:
        raise ValueError("input_ids are required to collect CGMR target hidden states")
    collected = {}
    handles = _register_exact_layer_hooks(model, layer_ids, collected)
    try:
        with torch.inference_mode():
            inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            if use_cache is not None:
                inputs["use_cache"] = bool(use_cache)
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer_id for layer_id in layer_ids if layer_id not in collected]
    if missing:
        raise ValueError("no hidden states collected for CGMR layers {}".format(missing))
    return {layer_id: collected[layer_id].detach() for layer_id in layer_ids}


def _dedupe_candidates(candidate_ids, candidate_sources, current_token, max_candidates):
    ordered = []
    sources = {}
    for token_id, source in zip(candidate_ids, candidate_sources):
        token_id = int(token_id)
        if token_id not in sources:
            ordered.append(token_id)
            sources[token_id] = []
        if source not in sources[token_id]:
            sources[token_id].append(source)
    current_token = int(current_token)
    if current_token not in sources:
        ordered.append(current_token)
        sources[current_token] = ["current"]
    elif "current" not in sources[current_token]:
        sources[current_token].append("current")

    if len(ordered) > max_candidates:
        kept = ordered[:max_candidates]
        if current_token not in kept:
            kept[-1] = current_token
        ordered = kept
    return ordered, {token_id: sources[token_id] for token_id in ordered}


def _build_candidate_pool(position, current_tokens, optimized_embedding, embed_layer,
                          tokenizer, filter_nonascii, add_perplexity, top_k_ppl,
                          top_k_cos, invert_method, max_candidates,
                          embedding_top_indices, select_candidate_from_top_indices,
                          get_perplexity, model, layer_id):
    candidate_ids = []
    candidate_sources = []
    embedding_top1 = None
    if top_k_cos > 0:
        embed = optimized_embedding.squeeze(0)[position]
        top_indices = embedding_top_indices(embed, embed_layer, top_k_cos, invert_method)
        _, top_ids = select_candidate_from_top_indices(top_indices, tokenizer, filter_nonascii)
        embedding_ids = [int(item) for item in top_ids]
        if embedding_ids:
            embedding_top1 = embedding_ids[0]
        candidate_ids.extend(embedding_ids)
        candidate_sources.extend(["embedding"] * len(embedding_ids))

    if position > 0 and add_perplexity and top_k_ppl > 0:
        _, topk_ids = get_perplexity(
            current_tokens[:position],
            model,
            layer_id=layer_id,
            top_k=top_k_ppl,
        )
        perplexity_ids = [int(item) for item in topk_ids.detach().cpu().tolist()]
        candidate_ids.extend(perplexity_ids)
        candidate_sources.extend(["perplexity"] * len(perplexity_ids))

    current_token = int(current_tokens[position])
    candidate_ids.append(current_token)
    candidate_sources.append("current")
    candidates, sources = _dedupe_candidates(
        candidate_ids,
        candidate_sources,
        current_token,
        max_candidates,
    )
    return candidates, sources, embedding_top1


def _expand_attention_mask(attention_mask, batch_size, device):
    if attention_mask is None:
        return None
    mask = attention_mask.to(device)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    return mask


def _score_candidates_by_layers(model, current_tokens, position, candidate_ids,
                                target_hidden_states, layer_ids, attention_mask,
                                batch_size, model_device):
    scores_by_layer = {layer_id: [] for layer_id in layer_ids}
    for start in range(0, len(candidate_ids), batch_size):
        chunk_ids = candidate_ids[start:start + batch_size]
        sequences = []
        for token_id in chunk_ids:
            sequence = list(current_tokens)
            sequence[position] = int(token_id)
            sequences.append(sequence)
        input_ids = torch.tensor(sequences, dtype=torch.long, device=model_device)
        collected = {}
        handles = _register_exact_layer_hooks(model, layer_ids, collected)
        try:
            with torch.inference_mode():
                inputs = {"input_ids": input_ids}
                mask = _expand_attention_mask(attention_mask, len(sequences), model_device)
                if mask is not None:
                    inputs["attention_mask"] = mask
                model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        for layer_id in layer_ids:
            candidate_state = collected[layer_id][:, position, :].type(torch.float32)
            target_state = target_hidden_states[layer_id][:, position, :]
            target_state = target_state.to(candidate_state.device, dtype=torch.float32)
            if target_state.shape[0] == 1 and candidate_state.shape[0] != 1:
                target_state = target_state.expand(candidate_state.shape[0], -1)
            layer_scores = F.cosine_similarity(candidate_state, target_state, dim=-1)
            scores_by_layer[layer_id].extend(
                [_as_float(value) for value in layer_scores]
            )
    return scores_by_layer


def _confidence_decision(layer_l_scores, candidate_ids, embedding_top1, config):
    ranked = sorted(range(len(candidate_ids)), key=lambda idx: layer_l_scores[idx], reverse=True)
    top1_idx = ranked[0]
    top2_idx = ranked[1] if len(ranked) > 1 else None
    top1_score = float(layer_l_scores[top1_idx])
    top2_score = float(layer_l_scores[top2_idx]) if top2_idx is not None else None
    margin = top1_score - top2_score if top2_score is not None else None
    single_layer_top1 = int(candidate_ids[top1_idx])

    strong = margin is not None and margin <= config.strong_margin_threshold
    weak_reasons = []
    if margin is not None and margin <= config.weak_margin_threshold:
        weak_reasons.append("weak_margin")
    if top1_score < config.low_score_threshold:
        weak_reasons.append("low_score")
    if embedding_top1 is not None and int(embedding_top1) != single_layer_top1:
        weak_reasons.append("embedding_single_layer_disagreement")
    if strong:
        reasons = ["strong_margin"] + [reason for reason in weak_reasons if reason != "weak_margin"]
        triggered = True
    else:
        reasons = weak_reasons
        triggered = len(weak_reasons) >= config.weak_signals_required
    return {
        "triggered": triggered,
        "reasons": reasons,
        "single_layer_top1": single_layer_top1,
        "single_layer_top2": int(candidate_ids[top2_idx]) if top2_idx is not None else None,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": margin,
    }


def _normalize_and_aggregate(raw_scores_by_layer, layer_ids, layer_weights,
                             consistency_weight):
    columns = []
    normalized_by_layer = {}
    for layer_id in layer_ids:
        values = torch.tensor(raw_scores_by_layer[layer_id], dtype=torch.float32)
        std = values.std(unbiased=False)
        if _as_float(std) <= NORMALIZATION_EPS:
            normalized = torch.zeros_like(values)
        else:
            normalized = (values - values.mean()) / std
        normalized_by_layer[layer_id] = normalized
        columns.append(normalized)
    matrix = torch.stack(columns, dim=1)
    weights = torch.tensor(layer_weights, dtype=torch.float32)
    weighted = matrix.matmul(weights)
    consistency_penalty = matrix.var(dim=1, unbiased=False) * float(consistency_weight)
    enhanced = weighted - consistency_penalty
    return normalized_by_layer, enhanced, consistency_penalty


def _candidate_score_records(candidate_ids, candidate_sources, raw_scores_by_layer,
                             normalized_by_layer, enhanced_scores, consistency_penalty,
                             layer_ids):
    records = []
    for idx, token_id in enumerate(candidate_ids):
        records.append({
            "token_id": int(token_id),
            "sources": candidate_sources[int(token_id)],
            "raw_layer_scores": {
                str(layer_id): float(raw_scores_by_layer[layer_id][idx])
                for layer_id in layer_ids
            },
            "normalized_layer_scores": {
                str(layer_id): _as_float(normalized_by_layer[layer_id][idx])
                for layer_id in layer_ids
            },
            "consistency_penalty": _as_float(consistency_penalty[idx]),
            "enhanced_score": _as_float(enhanced_scores[idx]),
        })
    return records


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
        "triggered": False,
        "accepted": False,
        "trigger_count": 0,
        "accepted_count": 0,
        "changed_positions": [],
        "elapsed_seconds": 0.0,
        "events": [],
    }


def run_cgmr_v1_0(
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
    events = []
    changed_positions = []
    trigger_count = 0
    accepted_count = 0

    for position in range(max(0, int(eval_start_pos)), len(tokens)):
        if trigger_count >= config.max_repair_steps:
            break
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
        if len(candidates) < 2:
            continue

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
        confidence = _confidence_decision(layer_l_raw, candidates, embedding_top1, config)
        if not confidence["triggered"]:
            continue

        trigger_count += 1
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
        current_token = int(tokens[position])
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
            "triggered": True,
            "trigger_reasons": confidence["reasons"],
            "candidate_count": len(candidates),
            "embedding_top1": embedding_top1,
            "single_layer_top1": confidence["single_layer_top1"],
            "single_layer_top2": confidence["single_layer_top2"],
            "single_layer_top1_score": confidence["top1_score"],
            "single_layer_top2_score": confidence["top2_score"],
            "single_layer_margin": confidence["margin"],
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

    result = {
        "name": METHOD_NAME,
        "enabled": True,
        "skipped": False,
        "reason": "completed" if trigger_count else "no_low_confidence_positions",
        "configured_layers": configured_layers,
        "effective_layers": effective_layers,
        "effective_weights": effective_weights,
        "filtered_layers": filtered_layers,
        "triggered": trigger_count > 0,
        "accepted": accepted_count > 0,
        "trigger_count": trigger_count,
        "accepted_count": accepted_count,
        "changed_positions": changed_positions,
        "elapsed_seconds": time.time() - started,
        "events": events,
    }
    return tokens, result


run_cgmr = run_cgmr_v1_0
