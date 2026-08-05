import math
import time
from dataclasses import dataclass

import torch

from .CGMR_v1_0 import (
    _as_float,
    _register_exact_layer_hooks,
    resolve_effective_layers,
)


METHOD_NAME = "CGMR_v1.2"


@dataclass
class CGMRV12Config:
    enabled: bool = False
    log_enabled: bool = True
    layer_offsets: tuple = (0, 1, 2)
    layer_weights: tuple = (0.5, 0.3, 0.2)
    entropy_temperature: float = 0.05
    effective_candidate_threshold: float = 1.5
    max_multilayer_candidates: int = 6
    lookahead_window: int = 1
    improvement_epsilon: float = 1e-6
    relative_mse_epsilon: float = 1e-8
    max_candidates: int = 32
    candidate_batch_size: int = 16

    def __post_init__(self):
        self.layer_offsets = tuple(int(value) for value in self.layer_offsets)
        self.layer_weights = tuple(float(value) for value in self.layer_weights)
        self.entropy_temperature = float(self.entropy_temperature)
        self.effective_candidate_threshold = float(
            self.effective_candidate_threshold
        )
        self.max_multilayer_candidates = int(self.max_multilayer_candidates)
        self.lookahead_window = int(self.lookahead_window)
        self.improvement_epsilon = float(self.improvement_epsilon)
        self.relative_mse_epsilon = float(self.relative_mse_epsilon)
        self.max_candidates = int(self.max_candidates)
        self.candidate_batch_size = int(self.candidate_batch_size)

        if not self.layer_offsets or 0 not in self.layer_offsets:
            raise ValueError("cgmr_v1_2_layer_offsets must contain 0")
        if any(offset < 0 for offset in self.layer_offsets):
            raise ValueError("cgmr_v1_2_layer_offsets must be non-negative")
        if len(set(self.layer_offsets)) != len(self.layer_offsets):
            raise ValueError("cgmr_v1_2_layer_offsets must not contain duplicates")
        if len(self.layer_offsets) != len(self.layer_weights):
            raise ValueError(
                "cgmr_v1_2_layer_offsets and layer_weights must have equal length"
            )
        if (
            any(not math.isfinite(weight) or weight < 0 for weight in self.layer_weights)
            or not math.isfinite(sum(self.layer_weights))
            or sum(self.layer_weights) <= 0
        ):
            raise ValueError(
                "cgmr_v1_2_layer_weights must be finite, non-negative, "
                "and have a positive sum"
            )
        if (
            not math.isfinite(self.entropy_temperature)
            or self.entropy_temperature <= 0
        ):
            raise ValueError("cgmr_v1_2_entropy_temperature must be positive")
        if (
            not math.isfinite(self.effective_candidate_threshold)
            or self.effective_candidate_threshold < 1
        ):
            raise ValueError(
                "cgmr_v1_2_effective_candidate_threshold must be at least 1"
            )
        if self.max_multilayer_candidates < 2:
            raise ValueError(
                "cgmr_v1_2_max_multilayer_candidates must be at least 2"
            )
        if self.lookahead_window < 1:
            raise ValueError("cgmr_v1_2_lookahead_window must be at least 1")
        if (
            not math.isfinite(self.improvement_epsilon)
            or self.improvement_epsilon <= 0
        ):
            raise ValueError("cgmr_v1_2_improvement_epsilon must be positive")
        if (
            not math.isfinite(self.relative_mse_epsilon)
            or self.relative_mse_epsilon <= 0
        ):
            raise ValueError("cgmr_v1_2_relative_mse_epsilon must be positive")
        if self.max_candidates < 2 or self.candidate_batch_size < 1:
            raise ValueError(
                "CGMR v1.2 candidate limits must allow at least two candidates "
                "and use a positive batch size"
            )


def _relative_mse(candidate_hidden, target_hidden, epsilon):
    candidate = torch.as_tensor(candidate_hidden).float()
    target = torch.as_tensor(target_hidden).to(candidate.device).float()
    return (
        (candidate - target).pow(2).mean(dim=-1)
        / (target.pow(2).mean(dim=-1) + float(epsilon))
    )


def _distance_entropy_and_effective_count(relative_mse_values, temperature):
    values = torch.as_tensor(relative_mse_values, dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError("relative MSE values must not be empty")
    logits = -(values - torch.min(values)) / float(temperature)
    log_probabilities = logits - torch.logsumexp(logits, dim=0)
    probabilities = torch.exp(log_probabilities)
    entropy = -torch.sum(probabilities * log_probabilities)
    effective_count = torch.exp(entropy)
    return (
        float(entropy.detach().cpu()),
        float(effective_count.detach().cpu()),
        [float(value) for value in probabilities.detach().cpu().tolist()],
    )


def _multilayer_candidate_count(
        effective_candidate_count,
        actual_candidate_count,
        max_multilayer_candidates):
    rounded = math.floor(float(effective_candidate_count) + 0.5)
    return min(
        max(rounded, 2),
        int(actual_candidate_count),
        int(max_multilayer_candidates),
    )


def _normalize_effective_weights(layer_weights):
    weights = [float(value) for value in layer_weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("effective layer weights must have a positive sum")
    return [value / weight_sum for value in weights]


def _raw_multilayer_costs(
        relative_mse_by_layer,
        effective_layers,
        effective_weights,
        candidate_indices):
    normalized_weights = _normalize_effective_weights(effective_weights)
    costs = {}
    for candidate_idx in candidate_indices:
        costs[int(candidate_idx)] = sum(
            normalized_weights[layer_idx]
            * float(relative_mse_by_layer[layer_id][candidate_idx])
            for layer_idx, layer_id in enumerate(effective_layers)
        )
    return costs


def _relative_improvement(reference_value, candidate_value, epsilon):
    return (
        (float(reference_value) - float(candidate_value))
        / max(float(reference_value), float(epsilon))
    )


def _filter_generated_candidates(candidate_ids, tokenizer, filter_nonascii):
    special_token_ids = set(tokenizer.all_special_ids)
    filtered = []
    for candidate in candidate_ids:
        token_id = int(candidate)
        if token_id in special_token_ids:
            continue
        if filter_nonascii and not tokenizer.decode([token_id]).isascii():
            continue
        filtered.append(token_id)
    return filtered


def _merge_candidate_sources(
        embedding_candidates,
        perplexity_candidates,
        current_token,
        max_candidates):
    ordered = []
    sources = {}
    for candidate_ids, source in (
        (embedding_candidates, "embedding"),
        (perplexity_candidates, "perplexity"),
    ):
        for candidate in candidate_ids:
            token_id = int(candidate)
            if token_id not in sources:
                ordered.append(token_id)
                sources[token_id] = []
            if source not in sources[token_id]:
                sources[token_id].append(source)

    ordered = ordered[:int(max_candidates)]
    sources = {token_id: sources[token_id] for token_id in ordered}
    current_token = int(current_token)
    if current_token in sources:
        if "current" not in sources[current_token]:
            sources[current_token].append("current")
    elif len(ordered) < int(max_candidates):
        ordered.append(current_token)
        sources[current_token] = ["current"]
    else:
        removed_token = ordered[-1]
        ordered[-1] = current_token
        sources.pop(removed_token)
        sources[current_token] = ["current"]
    return ordered, sources


def _build_candidate_pool_v1_2(
        position,
        current_tokens,
        upstream_optimized_embedding,
        embed_layer,
        tokenizer,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        invert_method,
        max_candidates,
        embedding_top_indices,
        get_perplexity,
        model,
        layer_id):
    embedding_candidates = []
    if top_k_cos > 0:
        source_embedding = upstream_optimized_embedding.squeeze(0)[position]
        top_indices = embedding_top_indices(
            source_embedding,
            embed_layer,
            top_k_cos,
            invert_method,
        )
        embedding_candidates = _filter_generated_candidates(
            [
                int(torch.as_tensor(item).detach().cpu())
                for item in top_indices
            ],
            tokenizer,
            filter_nonascii,
        )

    perplexity_candidates = []
    if position > 0 and add_perplexity and top_k_ppl > 0:
        _, top_ids = get_perplexity(
            current_tokens[:position],
            model,
            layer_id=layer_id,
            top_k=top_k_ppl,
        )
        perplexity_candidates = _filter_generated_candidates(
            [int(item) for item in top_ids.detach().cpu().tolist()],
            tokenizer,
            filter_nonascii,
        )

    candidates, sources = _merge_candidate_sources(
        embedding_candidates,
        perplexity_candidates,
        current_tokens[position],
        max_candidates,
    )
    embedding_top1 = (
        int(embedding_candidates[0]) if embedding_candidates else None
    )
    return candidates, sources, embedding_top1


def _build_hybrid_candidate_embeddings(
        upstream_optimized_embedding,
        committed_tokens,
        position,
        candidate_ids,
        embed_layer,
        device):
    baseline = upstream_optimized_embedding.detach().to(device)
    if baseline.ndim != 3 or baseline.shape[0] != 1:
        raise ValueError(
            "upstream_optimized_embedding must have shape [1, sequence, hidden]"
        )
    if baseline.shape[1] != len(committed_tokens):
        raise ValueError(
            "upstream optimized embedding sequence length must match token sequence"
        )
    candidate_count = len(candidate_ids)
    hybrid = baseline.expand(candidate_count, -1, -1).clone()
    with torch.inference_mode():
        if position > 0:
            prefix_ids = torch.tensor(
                committed_tokens[:position],
                dtype=torch.long,
                device=device,
            )
            prefix_embeddings = embed_layer(prefix_ids).to(
                device=device,
                dtype=hybrid.dtype,
            )
            hybrid[:, :position, :] = prefix_embeddings.unsqueeze(0)
        candidate_tensor = torch.tensor(
            candidate_ids,
            dtype=torch.long,
            device=device,
        )
        candidate_embeddings = embed_layer(candidate_tensor).to(
            device=device,
            dtype=hybrid.dtype,
        )
        hybrid[:, position, :] = candidate_embeddings
    return hybrid


def _expand_attention_mask_for_embeddings(
        attention_mask,
        batch_size,
        sequence_length,
        device):
    if attention_mask is None:
        return None
    mask = attention_mask.to(device)
    if mask.ndim != 2:
        raise ValueError("CGMR v1.2 attention mask must be two-dimensional")
    if mask.shape[1] != int(sequence_length):
        raise ValueError(
            "CGMR v1.2 attention mask sequence length must match inputs_embeds"
        )
    if mask.shape[0] == 1 and int(batch_size) > 1:
        mask = mask.expand(int(batch_size), -1)
    elif mask.shape[0] != int(batch_size):
        raise ValueError(
            "CGMR v1.2 attention mask batch size must be 1 or match candidates"
        )
    return mask


def _score_hybrid_candidates(
        model,
        embed_layer,
        upstream_optimized_embedding,
        committed_tokens,
        position,
        candidate_ids,
        target_hidden_states,
        effective_layers,
        attention_mask,
        candidate_batch_size,
        model_device,
        target_layer,
        lookahead_positions,
        relative_mse_epsilon):
    relative_mse_by_layer = {
        int(layer_id): [] for layer_id in effective_layers
    }
    lookahead_by_position = {
        int(lookahead_position): [] for lookahead_position in lookahead_positions
    }
    for start in range(0, len(candidate_ids), candidate_batch_size):
        chunk_ids = candidate_ids[start:start + candidate_batch_size]
        inputs_embeds = _build_hybrid_candidate_embeddings(
            upstream_optimized_embedding,
            committed_tokens,
            position,
            chunk_ids,
            embed_layer,
            model_device,
        )
        mask = _expand_attention_mask_for_embeddings(
            attention_mask,
            len(chunk_ids),
            inputs_embeds.shape[1],
            inputs_embeds.device,
        )
        collected = {}
        handles = _register_exact_layer_hooks(model, effective_layers, collected)
        try:
            with torch.inference_mode():
                inputs = {"inputs_embeds": inputs_embeds}
                if mask is not None:
                    inputs["attention_mask"] = mask
                model(**inputs)
        finally:
            for handle in handles:
                handle.remove()

        for layer_id in effective_layers:
            candidate_hidden = collected[layer_id][:, position, :]
            target_hidden = target_hidden_states[layer_id][:, position, :]
            values = _relative_mse(
                candidate_hidden,
                target_hidden,
                relative_mse_epsilon,
            )
            relative_mse_by_layer[layer_id].extend(
                [_as_float(value) for value in values]
            )

        for lookahead_position in lookahead_positions:
            candidate_hidden = collected[target_layer][
                :, lookahead_position, :
            ]
            target_hidden = target_hidden_states[target_layer][
                :, lookahead_position, :
            ]
            values = _relative_mse(
                candidate_hidden,
                target_hidden,
                relative_mse_epsilon,
            )
            lookahead_by_position[lookahead_position].extend(
                [_as_float(value) for value in values]
            )
    return relative_mse_by_layer, lookahead_by_position


def _value_map(candidate_ids, values):
    return {
        str(int(token_id)): float(values[index])
        for index, token_id in enumerate(candidate_ids)
    }


def _layer_value_map(candidate_ids, values_by_layer):
    return {
        str(int(layer_id)): _value_map(candidate_ids, values)
        for layer_id, values in values_by_layer.items()
    }


def _empty_result(
        config,
        reason,
        configured_layers,
        input_token_source,
        effective_layers=None,
        effective_weights=None,
        filtered_layers=None):
    return {
        "name": METHOD_NAME,
        "enabled": bool(config.enabled),
        "log_enabled": bool(config.log_enabled),
        "skipped": True,
        "reason": reason,
        "input_token_source": input_token_source,
        "configured_layers": configured_layers,
        "effective_layers": effective_layers or [],
        "effective_weights": effective_weights or [],
        "filtered_layers": filtered_layers or [],
        "entropy_temperature": config.entropy_temperature,
        "effective_candidate_threshold": config.effective_candidate_threshold,
        "triggered": False,
        "accepted": False,
        "multilayer_accepted": False,
        "evaluated_position_count": 0,
        "high_entropy_position_count": 0,
        "high_entropy_positions": [],
        "multilayer_positions": [],
        "multilayer_accepted_positions": [],
        "multilayer_changed_positions": [],
        "accepted_count": 0,
        "multilayer_accepted_count": 0,
        "rejected_count": 0,
        "changed_positions": [],
        "elapsed_seconds": 0.0,
        "events": [],
    }


def run_cgmr_v1_2(
        model,
        embed_layer,
        upstream_optimized_embedding,
        current_tokens,
        target_hidden_states,
        attention_mask,
        target_layer,
        model_layer_count,
        model_device,
        config,
        tokenizer,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        invert_method,
        eval_start_pos,
        fixed_prefix_tokens,
        input_token_source,
        embedding_top_indices,
        select_candidate_from_top_indices,
        get_perplexity):
    del select_candidate_from_top_indices
    configured_layers = [
        int(target_layer) + offset for offset in config.layer_offsets
    ]
    if not config.enabled:
        return list(current_tokens), _empty_result(
            config,
            "disabled",
            configured_layers,
            input_token_source,
        )

    effective_layers, effective_weights, filtered_layers = resolve_effective_layers(
        target_layer,
        model_layer_count,
        config.layer_offsets,
        config.layer_weights,
    )
    if target_layer not in effective_layers:
        return list(current_tokens), _empty_result(
            config,
            "target_layer_unavailable",
            configured_layers,
            input_token_source,
            effective_layers,
            effective_weights,
            filtered_layers,
        )
    missing_targets = [
        layer_id
        for layer_id in effective_layers
        if layer_id not in target_hidden_states
    ]
    if missing_targets:
        raise ValueError(
            "missing CGMR v1.2 target hidden states for layers {}".format(
                missing_targets
            )
        )

    started = time.time()
    tokens = [int(item) for item in current_tokens]
    scan_start = max(0, int(eval_start_pos))
    if scan_start > len(tokens):
        scan_start = len(tokens)
    if scan_start:
        if fixed_prefix_tokens is None or len(fixed_prefix_tokens) < scan_start:
            raise ValueError(
                "fixed_prefix_tokens must provide every fixed prefix position"
            )
        tokens[:scan_start] = [
            int(token_id) for token_id in fixed_prefix_tokens[:scan_start]
        ]
    entry_tokens = list(tokens)

    events = []
    high_entropy_positions = []
    multilayer_positions = []
    multilayer_accepted_positions = []
    changed_positions = []

    for position in range(scan_start, len(tokens)):
        lookahead_positions = list(range(
            position + 1,
            min(len(tokens), position + 1 + config.lookahead_window),
        ))
        candidates, candidate_sources, embedding_top1 = (
            _build_candidate_pool_v1_2(
                position,
                tokens,
                upstream_optimized_embedding,
                embed_layer,
                tokenizer,
                filter_nonascii,
                add_perplexity,
                top_k_ppl,
                top_k_cos,
                invert_method,
                config.max_candidates,
                embedding_top_indices,
                get_perplexity,
                model,
                target_layer,
            )
        )
        relative_mse_by_layer, lookahead_by_position = _score_hybrid_candidates(
            model,
            embed_layer,
            upstream_optimized_embedding,
            tokens,
            position,
            candidates,
            target_hidden_states,
            effective_layers,
            attention_mask,
            config.candidate_batch_size,
            model_device,
            target_layer,
            lookahead_positions,
            config.relative_mse_epsilon,
        )
        layer_l_values = relative_mse_by_layer[target_layer]
        layer_l_best_idx = min(
            range(len(candidates)),
            key=lambda index: layer_l_values[index],
        )
        layer_l_best_token = int(candidates[layer_l_best_idx])
        distance_entropy, effective_candidate_count, _ = (
            _distance_entropy_and_effective_count(
                layer_l_values,
                config.entropy_temperature,
            )
        )
        high_entropy = (
            len(candidates) >= 2
            and effective_candidate_count
            > config.effective_candidate_threshold
        )
        if high_entropy:
            high_entropy_positions.append(position)

        window_values = None
        window_improvements = None
        lookahead_available = bool(lookahead_positions)
        if lookahead_available:
            window_values = [
                sum(
                    lookahead_by_position[lookahead_position][candidate_idx]
                    for lookahead_position in lookahead_positions
                ) / len(lookahead_positions)
                for candidate_idx in range(len(candidates))
            ]
            window_improvements = [
                _relative_improvement(
                    window_values[layer_l_best_idx],
                    window_values[candidate_idx],
                    config.relative_mse_epsilon,
                )
                for candidate_idx in range(len(candidates))
            ]

        selected_idx = layer_l_best_idx
        selection_source = "layer_l_low_entropy"
        selection_reason = (
            "insufficient_candidates"
            if len(candidates) < 2
            else "effective_candidate_count_not_above_threshold"
        )
        multilayer_candidate_indices = []
        multilayer_costs = {}
        multilayer_best_idx = None
        multilayer_improvement = None
        multilayer_accepted = False

        if high_entropy and len(effective_layers) < 2:
            selection_source = "layer_l_multilayer_rejected"
            selection_reason = "insufficient_effective_layers"
        elif high_entropy:
            multilayer_count = _multilayer_candidate_count(
                effective_candidate_count,
                len(candidates),
                config.max_multilayer_candidates,
            )
            multilayer_candidate_indices = sorted(
                range(len(candidates)),
                key=lambda index: layer_l_values[index],
            )[:multilayer_count]
            multilayer_positions.append(position)
            multilayer_costs = _raw_multilayer_costs(
                relative_mse_by_layer,
                effective_layers,
                effective_weights,
                multilayer_candidate_indices,
            )
            multilayer_best_idx = min(
                multilayer_candidate_indices,
                key=lambda index: multilayer_costs[index],
            )
            if multilayer_best_idx == layer_l_best_idx:
                selection_source = "layer_l_multilayer_agreement"
                selection_reason = "multilayer_agrees_with_layer_l"
            else:
                multilayer_improvement = _relative_improvement(
                    multilayer_costs[layer_l_best_idx],
                    multilayer_costs[multilayer_best_idx],
                    config.relative_mse_epsilon,
                )
                multilayer_improved = (
                    multilayer_improvement > config.improvement_epsilon
                )
                lookahead_improved = (
                    not lookahead_available
                    or window_improvements[multilayer_best_idx]
                    > config.improvement_epsilon
                )
                if multilayer_improved and lookahead_improved:
                    selected_idx = multilayer_best_idx
                    multilayer_accepted = True
                    multilayer_accepted_positions.append(position)
                    if lookahead_available:
                        selection_source = (
                            "multilayer_and_lookahead_accepted"
                        )
                        selection_reason = (
                            "multilayer_and_lookahead_improved"
                        )
                    else:
                        selection_source = (
                            "multilayer_without_lookahead_accepted"
                        )
                        selection_reason = (
                            "multilayer_improved_without_lookahead"
                        )
                else:
                    selection_source = "layer_l_multilayer_rejected"
                    selection_reason = (
                        "multilayer_improvement_not_above_epsilon"
                        if not multilayer_improved
                        else "lookahead_improvement_not_above_epsilon"
                    )

        selected_token = int(candidates[selected_idx])
        old_token = int(entry_tokens[position])
        accepted = selected_token != old_token
        tokens[position] = selected_token
        if accepted:
            changed_positions.append(position)

        event = {
            "position": position,
            "input_token_source": input_token_source,
            "candidate_ids": [int(token_id) for token_id in candidates],
            "candidate_sources": {
                str(int(token_id)): list(candidate_sources[int(token_id)])
                for token_id in candidates
            },
            "candidate_count": len(candidates),
            "embedding_top1": embedding_top1,
            "layer_l_relative_mse": _value_map(
                candidates,
                layer_l_values,
            ),
            "layer_l_best_token": layer_l_best_token,
            "distance_entropy": distance_entropy,
            "effective_candidate_count": effective_candidate_count,
            "high_entropy": high_entropy,
            "multilayer_candidate_count": len(
                multilayer_candidate_indices
            ),
            "multilayer_candidate_ids": [
                int(candidates[index])
                for index in multilayer_candidate_indices
            ],
            "relative_mse_by_layer": _layer_value_map(
                candidates,
                relative_mse_by_layer,
            ),
            "multilayer_costs": {
                str(int(candidates[index])): float(cost)
                for index, cost in multilayer_costs.items()
            },
            "multilayer_best_token": (
                int(candidates[multilayer_best_idx])
                if multilayer_best_idx is not None
                else None
            ),
            "multilayer_improvement": multilayer_improvement,
            "lookahead_window": config.lookahead_window,
            "lookahead_available": lookahead_available,
            "lookahead_relative_mse": (
                _value_map(candidates, window_values)
                if window_values is not None
                else {}
            ),
            "lookahead_relative_improvement": (
                _value_map(candidates, window_improvements)
                if window_improvements is not None
                else {}
            ),
            "old_token": old_token,
            "new_token": selected_token,
            "selected_token": selected_token,
            "accepted": accepted,
            "multilayer_accepted": multilayer_accepted,
            "selection_source": selection_source,
            "selection_reason": selection_reason,
        }
        events.append(event)

    elapsed_seconds = time.time() - started
    return tokens, {
        "name": METHOD_NAME,
        "enabled": True,
        "log_enabled": bool(config.log_enabled),
        "skipped": False,
        "reason": "completed",
        "input_token_source": input_token_source,
        "configured_layers": configured_layers,
        "effective_layers": effective_layers,
        "effective_weights": effective_weights,
        "filtered_layers": filtered_layers,
        "entropy_temperature": config.entropy_temperature,
        "effective_candidate_threshold": config.effective_candidate_threshold,
        "triggered": bool(multilayer_positions),
        "accepted": bool(changed_positions),
        "multilayer_accepted": bool(multilayer_accepted_positions),
        "evaluated_position_count": len(events),
        "high_entropy_position_count": len(high_entropy_positions),
        "high_entropy_positions": high_entropy_positions,
        "multilayer_positions": multilayer_positions,
        "multilayer_accepted_positions": multilayer_accepted_positions,
        "multilayer_changed_positions": multilayer_accepted_positions,
        "accepted_count": len(changed_positions),
        "multilayer_accepted_count": len(multilayer_accepted_positions),
        "rejected_count": len(events) - len(changed_positions),
        "changed_positions": changed_positions,
        "elapsed_seconds": elapsed_seconds,
        "events": events,
    }


run_cgmr = run_cgmr_v1_2
