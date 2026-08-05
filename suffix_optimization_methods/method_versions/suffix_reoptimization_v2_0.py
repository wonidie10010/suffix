"""Suffix reoptimization v2.0.

This sidecar owns the complete v2.0 method.  Ground-truth token ids are only
read by the diagnostic pass after every method decision has finished.
"""

from dataclasses import dataclass
from typing import Protocol, Sequence
import copy
import math
import numbers

import torch
import torch.nn.functional as F


METHOD_NAME = "suffix_reoptimization_v2.0"
VERSION = "v2.0"
HEURISTIC_CALIBRATION_NOTE = (
    "尚需实验校准的初始启发式值"
)
EMBEDDING_SEARCH_CHUNK_SIZE = 8192


class SuffixV20FatalError(RuntimeError):
    """A sample-level hard failure which requires entry-snapshot rollback."""

    def __init__(self, reason, stage, position=None, segment=None):
        super().__init__(str(reason))
        self.reason = str(reason)
        self.stage = str(stage)
        self.position = None if position is None else int(position)
        self.segment = (
            None if segment is None else [int(segment[0]), int(segment[1])]
        )


@dataclass(frozen=True)
class ClassifierCandidate:
    token_id: int
    score: float
    rank: int


class ClassifierCandidateProvider(Protocol):
    def top_candidates(
            self, *, position: int, committed_tokens: Sequence[int],
            continuous_embedding: torch.Tensor,
            top_k: int) -> Sequence[ClassifierCandidate]:
        """Return ranked classifier candidates for one position."""


@dataclass
class SuffixReoptimizationV20Config:
    enabled: bool = False
    log_enabled: bool = True
    layer_offsets: tuple = (0, 1, 2)
    layer_weights: tuple = (1.0, 0.5, 0.25)
    epsilon: float = 1e-8
    phase1_epoch: int = 1000
    phase1_lr: float = 1e-2
    phase1_direction_weight: float = 0.9
    phase1_magnitude_weight: float = 0.1
    phase2_epoch: int = 50
    phase2_lr: float = 1e-3
    phase2_direction_weight: float = 0.1
    phase2_magnitude_weight: float = 0.9
    score_direction_weight: float = 0.5
    score_magnitude_weight: float = 0.5
    prox_weight: float = 0.005
    range_weight: float = 0.001
    continuous_mad_multiplier: float = 3.0
    local_discrete_mad_multiplier: float = 3.0
    local_gap_jump_mad_multiplier: float = 3.0
    mad_epsilon: float = 1e-8
    local_min_points: int = 4
    normal_embedding_top_k: int = 10
    expanded_embedding_top_k: int = 20
    ppl_top_k: int = 10
    classifier_top_k: int = 10
    cumulative_min_points: int = 4
    cumulative_kappa: float = 0.5
    cumulative_threshold: float = 5.0
    replace_epsilon: float = 1e-8
    cumulative_max_repairs_per_trigger: int = 1
    accuracy_diagnostics_enabled: bool = True
    classifier_enabled: bool = False

    def __post_init__(self):
        for name in (
            "enabled", "log_enabled", "accuracy_diagnostics_enabled",
            "classifier_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_0_{} must be boolean".format(name))
        if not isinstance(self.layer_offsets, (list, tuple)) or not all(
                isinstance(value, numbers.Integral) and not isinstance(value, bool)
                for value in self.layer_offsets):
            raise TypeError("suffix_v2_0_layer_offsets must contain integers")
        if not isinstance(self.layer_weights, (list, tuple)) or not all(
                isinstance(value, numbers.Real) and not isinstance(value, bool)
                for value in self.layer_weights):
            raise TypeError("suffix_v2_0_layer_weights must contain numbers")
        self.layer_offsets = tuple(int(value) for value in self.layer_offsets)
        self.layer_weights = tuple(float(value) for value in self.layer_weights)
        integer_fields = (
            "phase1_epoch", "phase2_epoch", "local_min_points",
            "normal_embedding_top_k", "expanded_embedding_top_k",
            "ppl_top_k", "classifier_top_k", "cumulative_min_points",
            "cumulative_max_repairs_per_trigger",
        )
        float_fields = (
            "epsilon", "phase1_lr", "phase1_direction_weight",
            "phase1_magnitude_weight", "phase2_lr",
            "phase2_direction_weight", "phase2_magnitude_weight",
            "score_direction_weight", "score_magnitude_weight",
            "prox_weight", "range_weight", "continuous_mad_multiplier",
            "local_discrete_mad_multiplier",
            "local_gap_jump_mad_multiplier", "mad_epsilon",
            "cumulative_kappa", "cumulative_threshold", "replace_epsilon",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_0_{} must be an integer".format(name))
        for name in float_fields:
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError("suffix_v2_0_{} must be numeric".format(name))
        for name in integer_fields:
            value = getattr(self, name)
            setattr(self, name, int(value))
        for name in float_fields:
            setattr(self, name, float(getattr(self, name)))
        self.enabled = bool(self.enabled)
        self.log_enabled = bool(self.log_enabled)
        self.accuracy_diagnostics_enabled = bool(
            self.accuracy_diagnostics_enabled
        )
        self.classifier_enabled = bool(self.classifier_enabled)
        validate_suffix_reoptimization_v2_0_config(self)


def _require_finite_nonnegative(name, value, positive=False):
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("suffix_v2_0_{} must be finite and non-negative".format(name))
    if positive and float(value) <= 0:
        raise ValueError("suffix_v2_0_{} must be positive".format(name))


def validate_suffix_reoptimization_v2_0_config(config):
    if not config.layer_offsets or 0 not in config.layer_offsets:
        raise ValueError("suffix_v2_0_layer_offsets must contain 0")
    if any(offset < 0 for offset in config.layer_offsets):
        raise ValueError("suffix_v2_0_layer_offsets must be non-negative")
    if len(set(config.layer_offsets)) != len(config.layer_offsets):
        raise ValueError("suffix_v2_0_layer_offsets must not contain duplicates")
    if len(config.layer_offsets) != len(config.layer_weights):
        raise ValueError("suffix_v2_0 layer offsets and weights must have equal length")
    if any(offset < config.layer_offsets[index - 1]
           for index, offset in enumerate(config.layer_offsets) if index):
        raise ValueError("suffix_v2_0_layer_offsets must be ordered")
    for name in (
        "layer_weights", "phase1_direction_weight",
        "phase1_magnitude_weight", "phase2_direction_weight",
        "phase2_magnitude_weight", "score_direction_weight",
        "score_magnitude_weight", "prox_weight", "range_weight",
        "continuous_mad_multiplier", "local_discrete_mad_multiplier",
        "local_gap_jump_mad_multiplier", "cumulative_kappa",
        "cumulative_threshold", "replace_epsilon",
    ):
        values = getattr(config, name)
        values = values if isinstance(values, tuple) else (values,)
        for value in values:
            _require_finite_nonnegative(name, value)
    for name in ("epsilon", "mad_epsilon", "phase1_lr", "phase2_lr"):
        _require_finite_nonnegative(name, getattr(config, name), positive=True)
    for prefix in ("phase1", "phase2", "score"):
        if (
            getattr(config, "{}_direction_weight".format(prefix))
            + getattr(config, "{}_magnitude_weight".format(prefix))
            <= 0
        ):
            raise ValueError("suffix_v2_0_{} weights must have a positive sum".format(prefix))
    if sum(config.layer_weights) <= 0:
        raise ValueError("suffix_v2_0_layer_weights must have a positive sum")
    for name in (
        "phase1_epoch", "phase2_epoch", "local_min_points",
        "normal_embedding_top_k", "expanded_embedding_top_k", "ppl_top_k",
        "classifier_top_k", "cumulative_min_points",
        "cumulative_max_repairs_per_trigger",
    ):
        if getattr(config, name) <= 0:
            raise ValueError("suffix_v2_0_{} must be positive".format(name))
    if config.expanded_embedding_top_k < config.normal_embedding_top_k:
        raise ValueError("expanded embedding top-k must not be smaller than normal top-k")
    return config


def _normalized_pair(direction_weight, magnitude_weight):
    direction = float(direction_weight)
    magnitude = float(magnitude_weight)
    total = direction + magnitude
    if not math.isfinite(total) or total <= 0:
        raise ValueError("direction and magnitude weights must have a positive finite sum")
    return direction / total, magnitude / total


def direction_magnitude_joint_error(
        current_hidden, target_hidden, direction_weight=0.5,
        magnitude_weight=0.5, epsilon=1e-8):
    """Return (joint, direction, magnitude), reduced only over hidden dim."""
    current = torch.as_tensor(current_hidden).float()
    target = torch.as_tensor(target_hidden).to(current.device).float()
    if current.shape != target.shape:
        raise ValueError("current and target hidden states must have equal shapes")
    if not torch.isfinite(current).all() or not torch.isfinite(target).all():
        raise SuffixV20FatalError("nonfinite_hidden_state", "joint_error")
    current_norm = torch.linalg.vector_norm(current, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    denominator = (current_norm * target_norm).clamp_min(float(epsilon))
    cosine = (current * target).sum(dim=-1) / denominator
    cosine = cosine.clamp(-1.0, 1.0)
    direction = (1.0 - cosine) / 2.0
    magnitude = (
        (current_norm - target_norm)
        / (current_norm + target_norm + float(epsilon))
    ).pow(2)
    direction_weight, magnitude_weight = _normalized_pair(
        direction_weight, magnitude_weight
    )
    joint = direction_weight * direction + magnitude_weight * magnitude
    if not torch.isfinite(joint).all():
        raise SuffixV20FatalError("nonfinite_joint_error", "joint_error")
    return joint, direction, magnitude


def resolve_effective_layers(target_layer, model_layer_count, offsets, weights):
    effective = []
    filtered = []
    for offset, weight in zip(offsets, weights):
        layer = int(target_layer) + int(offset)
        if layer < int(model_layer_count):
            effective.append((layer, float(weight)))
        else:
            filtered.append(layer)
    if not effective:
        raise ValueError("suffix v2.0 has no effective layers")
    total = sum(weight for _, weight in effective)
    if total <= 0:
        raise ValueError("suffix v2.0 effective layer weights sum to zero")
    return (
        [layer for layer, _ in effective],
        [weight / total for _, weight in effective],
        filtered,
    )


def _median(values):
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64)
    return float(torch.median(tensor))


def _mad(values):
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def robust_upper_threshold(values, multiplier, mad_epsilon):
    if not values:
        raise ValueError("robust threshold requires at least one value")
    return _median(values) + float(multiplier) * max(_mad(values), float(mad_epsilon))


def local_anomaly_decision(d_value, delta_g, d_history, delta_history,
                           tau_c, config):
    if not math.isfinite(float(d_value)) or not math.isfinite(float(delta_g)):
        return True, {"mode": "nonfinite", "tau_d": None, "tau_delta_g": None}
    if len(d_history) < config.local_min_points:
        return float(d_value) > float(tau_c), {
            "mode": "continuous_threshold_warmup",
            "tau_d": float(tau_c),
            "tau_delta_g": None,
        }
    tau_d = robust_upper_threshold(
        d_history, config.local_discrete_mad_multiplier, config.mad_epsilon
    )
    tau_delta = robust_upper_threshold(
        delta_history, config.local_gap_jump_mad_multiplier, config.mad_epsilon
    )
    return (float(d_value) > tau_d or float(delta_g) > tau_delta), {
        "mode": "prefix_median_mad",
        "tau_d": tau_d,
        "tau_delta_g": tau_delta,
    }


def update_cumulative_state(g_value, g_history, previous_s, config,
                            current_position, active_segment_start=None):
    if len(g_history) < config.cumulative_min_points:
        return {
            "z": None,
            "S": 0.0,
            "triggered": False,
            "segment_start": None,
            "baseline_median": None,
            "baseline_scale": None,
        }
    baseline = _median(g_history)
    scale = 1.4826 * _mad(g_history) + float(config.epsilon)
    z_value = (float(g_value) - baseline) / scale
    s_value = max(0.0, float(previous_s) + z_value - config.cumulative_kappa)
    if s_value == 0.0:
        segment_start = None
    elif float(previous_s) == 0.0 or active_segment_start is None:
        segment_start = int(current_position)
    else:
        segment_start = int(active_segment_start)
    return {
        "z": z_value,
        "S": s_value,
        "triggered": s_value > config.cumulative_threshold,
        "segment_start": segment_start,
        "baseline_median": baseline,
        "baseline_scale": scale,
    }


def should_replace(current_score, candidate_score, replace_epsilon):
    return float(candidate_score) < float(current_score) - float(replace_epsilon)


def candidate_tie_break_key(entry):
    """Canonical deterministic Stage-4/repair ordering key."""
    infinity = float("inf")
    return (
        entry["score"],
        entry["target_layer_score"],
        entry["source_ranks"].get("embedding", infinity),
        entry["source_ranks"].get("perplexity", infinity),
        entry["source_ranks"].get("classifier", infinity),
        entry["token_id"],
    )


def _is_legal_token(token_id, tokenizer, filter_nonascii):
    token_id = int(token_id)
    if token_id < 0 or token_id >= int(tokenizer.vocab_size):
        return False
    if token_id in set(getattr(tokenizer, "all_special_ids", [])):
        return False
    return not filter_nonascii or tokenizer.decode([token_id]).isascii()


def build_entry_snapshot_from_embedding(
        embedding, embed_layer, tokenizer, invert_method, filter_nonascii,
        eval_start_pos, fixed_prefix_tokens, embedding_top_indices,
        select_candidate_from_top_indices):
    """Create the sidecar-entry token snapshot without target token ids."""
    tokens = [int(value) for value in fixed_prefix_tokens]
    squeezed = embedding.detach().squeeze(0)
    for position in range(int(eval_start_pos), int(squeezed.shape[0])):
        top_indices = embedding_top_indices(
            squeezed[position], embed_layer, 10, invert_method
        )
        selected_token, _ = select_candidate_from_top_indices(
            top_indices, tokenizer, filter_nonascii
        )
        if (
            selected_token is None
            or not _is_legal_token(
                int(selected_token), tokenizer, filter_nonascii
            )
        ):
            raise SuffixV20FatalError(
                "legal_entry_snapshot_generation_failed",
                "entry_snapshot",
                position,
            )
        tokens.append(int(selected_token))
    if len(tokens) != int(squeezed.shape[0]):
        raise SuffixV20FatalError(
            "token_sequence_length_changed", "entry_snapshot"
        )
    return tokens


def _candidate_entry(token_id):
    return {
        "token_id": int(token_id),
        "sources": [],
        "source_ranks": {},
        "classifier_score": None,
    }


def merge_candidate_sources(source_candidates, tokenizer, filter_nonascii,
                            fail_on_illegal_classifier=True):
    merged = {}
    ordered = []
    for source, candidates in source_candidates:
        for rank, candidate in enumerate(candidates, start=1):
            classifier_score = None
            supplied_rank = rank
            if source == "classifier":
                if not isinstance(candidate, ClassifierCandidate):
                    raise SuffixV20FatalError(
                        "illegal_classifier_result", "candidate_generation"
                    )
                if isinstance(candidate.token_id, bool) or not isinstance(candidate.token_id, int):
                    raise SuffixV20FatalError(
                        "illegal_classifier_token_id", "candidate_generation"
                    )
                if (
                    isinstance(candidate.rank, bool)
                    or not isinstance(candidate.rank, int)
                    or candidate.rank <= 0
                    or not math.isfinite(float(candidate.score))
                ):
                    raise SuffixV20FatalError(
                        "illegal_classifier_result", "candidate_generation"
                    )
                classifier_score = float(candidate.score)
                supplied_rank = int(candidate.rank)
                token_id = candidate.token_id
            else:
                if isinstance(candidate, bool) or not isinstance(candidate, int):
                    raise SuffixV20FatalError(
                        "illegal_{}_candidate".format(source),
                        "candidate_generation",
                    )
                token_id = candidate
            legal = _is_legal_token(token_id, tokenizer, filter_nonascii)
            if not legal:
                if source == "classifier" and fail_on_illegal_classifier:
                    raise SuffixV20FatalError(
                        "illegal_classifier_token", "candidate_generation"
                    )
                continue
            token_id = int(token_id)
            if token_id not in merged:
                merged[token_id] = _candidate_entry(token_id)
                ordered.append(token_id)
            entry = merged[token_id]
            if source not in entry["sources"]:
                entry["sources"].append(source)
                entry["source_ranks"][source] = supplied_rank
            if source == "classifier":
                entry["classifier_score"] = classifier_score
    return [merged[token_id] for token_id in ordered]


def validate_classifier_candidates(provider, position, committed_tokens,
                                   continuous_embedding, top_k):
    if provider is None or not hasattr(provider, "top_candidates"):
        raise SuffixV20FatalError(
            "classifier_provider_unavailable", "candidate_generation", position
        )
    returned = provider.top_candidates(
        position=int(position),
        committed_tokens=tuple(int(value) for value in committed_tokens),
        continuous_embedding=continuous_embedding.detach(),
        top_k=int(top_k),
    )
    if returned is None or isinstance(returned, (str, bytes)):
        raise SuffixV20FatalError(
            "illegal_classifier_result", "candidate_generation", position
        )
    candidates = list(returned)
    if len(candidates) != int(top_k):
        raise SuffixV20FatalError(
            "classifier_candidate_count_mismatch", "candidate_generation", position
        )
    ranks = []
    token_ids = []
    for candidate in candidates:
        if not isinstance(candidate, ClassifierCandidate):
            raise SuffixV20FatalError(
                "illegal_classifier_result", "candidate_generation", position
            )
        ranks.append(candidate.rank)
        token_ids.append(candidate.token_id)
    if (
        len(set(ranks)) != len(ranks)
        or sorted(ranks) != list(range(1, len(candidates) + 1))
        or len(set(token_ids)) != len(token_ids)
    ):
        raise SuffixV20FatalError(
            "duplicate_classifier_result", "candidate_generation", position
        )
    return candidates


def _register_exact_layer_hooks(model, layer_ids, destination):
    requested = set(int(layer_id) for layer_id in layer_ids)
    registered = set()
    handles = []

    def layer_id_from_name(name):
        parts = str(name).split(".")
        for index in range(len(parts) - 1):
            if parts[index] == "layers" and parts[index + 1].isdigit():
                return int(parts[index + 1])
        return None

    def make_hook(layer_id):
        def hook(module, inputs, output):
            del module, inputs
            destination[layer_id] = output[0] if isinstance(output, tuple) else output
        return hook

    for name, module in model.named_modules():
        layer_id = layer_id_from_name(name)
        if layer_id in requested and layer_id not in registered:
            handles.append(module.register_forward_hook(make_hook(layer_id)))
            registered.add(layer_id)
    if registered != requested:
        for handle in handles:
            handle.remove()
        raise SuffixV20FatalError(
            "missing_layer_hooks", "forward", segment=[min(requested), max(requested)]
        )
    return handles


def _collect_hidden(model, layer_ids, attention_mask, input_ids=None,
                    inputs_embeds=None, inference=False):
    collected = {}
    handles = _register_exact_layer_hooks(model, layer_ids, collected)
    inputs = {}
    if input_ids is not None:
        inputs["input_ids"] = input_ids
    elif inputs_embeds is not None:
        inputs["inputs_embeds"] = inputs_embeds
    else:
        raise ValueError("input_ids or inputs_embeds are required")
    if attention_mask is not None:
        mask = attention_mask.to(
            input_ids.device if input_ids is not None else inputs_embeds.device
        )
        batch_size = int(
            input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        )
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        inputs["attention_mask"] = mask
    try:
        if inference:
            with torch.inference_mode():
                model(**inputs)
        else:
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    for layer_id in layer_ids:
        if layer_id not in collected:
            raise SuffixV20FatalError("missing_hidden_state", "forward")
        if not torch.isfinite(collected[layer_id]).all():
            raise SuffixV20FatalError("nonfinite_hidden_state", "forward")
    return collected


def _valid_mask(sequence_length, eval_start_pos, attention_mask, device):
    mask = torch.ones(int(sequence_length), dtype=torch.bool, device=device)
    mask[:max(0, min(int(eval_start_pos), int(sequence_length)))] = False
    if attention_mask is not None:
        attention = attention_mask[0].to(device=device, dtype=torch.bool)
        if attention.numel() != int(sequence_length):
            raise SuffixV20FatalError("attention_mask_length_changed", "invariant")
        mask &= attention
    if not bool(mask.any().detach().cpu()):
        raise SuffixV20FatalError("no_valid_positions", "invariant")
    return mask


def _multilayer_loss(collected, targets, layers, weights, valid_mask,
                     direction_weight, magnitude_weight, epsilon):
    total = None
    direction_total = None
    magnitude_total = None
    for layer_id, layer_weight in zip(layers, weights):
        target = targets[layer_id].to(collected[layer_id].device)
        joint, direction, magnitude = direction_magnitude_joint_error(
            collected[layer_id], target, direction_weight, magnitude_weight, epsilon
        )
        mask = valid_mask.to(joint.device)
        if joint.ndim == 2:
            mask = mask.unsqueeze(0).expand(joint.shape[0], -1)
        layer_joint = joint[mask].mean()
        layer_direction = direction[mask].mean()
        layer_magnitude = magnitude[mask].mean()
        total = layer_weight * layer_joint if total is None else total + layer_weight * layer_joint
        direction_total = layer_weight * layer_direction if direction_total is None else direction_total + layer_weight * layer_direction
        magnitude_total = layer_weight * layer_magnitude if magnitude_total is None else magnitude_total + layer_weight * layer_magnitude
    return total, direction_total, magnitude_total


def _optimize_phase(model, optimizable, prefix_embedding, targets, attention_mask,
                    layers, weights, valid_mask, epoch, lr,
                    direction_weight, magnitude_weight, range_bound,
                    range_weight, prox_weight=0.0, prox_reference=None,
                    optimizer_name="SGD", clip=False, epsilon=1e-8):
    variable = optimizable.detach().clone().requires_grad_(True)
    frozen_reference = optimizable.detach().clone()
    prefix_length = 0 if prefix_embedding is None else int(prefix_embedding.shape[1])
    variable_valid_mask = valid_mask[prefix_length:].to(variable.device)
    if int(variable_valid_mask.numel()) != int(variable.shape[1]):
        raise SuffixV20FatalError(
            "frozen_position_mask_length_mismatch", "continuous_optimization"
        )
    persistent_optimizer = None
    if optimizer_name == "Adam":
        persistent_optimizer = torch.optim.Adam([variable], lr=float(lr))
    history = []
    for step in range(int(epoch)):
        if clip:
            with torch.no_grad():
                variable.clamp_(-0.2, 0.2)
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD([variable], lr=float(lr))
        else:
            optimizer = persistent_optimizer
        full_embedding = (
            torch.cat((prefix_embedding, variable), dim=1)
            if prefix_embedding is not None else variable
        )
        if not torch.isfinite(full_embedding).all():
            raise SuffixV20FatalError("nonfinite_embedding", "continuous_optimization")
        collected = _collect_hidden(
            model, layers, attention_mask, inputs_embeds=full_embedding
        )
        joint_loss, direction_loss, magnitude_loss = _multilayer_loss(
            collected, targets, layers, weights, valid_mask,
            direction_weight, magnitude_weight, epsilon,
        )
        range_values = F.relu(torch.abs(variable) - range_bound)
        range_loss = range_values[:, variable_valid_mask, :].sum()
        prox_loss = torch.zeros((), device=variable.device, dtype=torch.float32)
        if prox_reference is not None and float(prox_weight) > 0:
            prox_loss = (variable.float() - prox_reference.float()).pow(2).mean()
        total_loss = joint_loss + float(range_weight) * range_loss + float(prox_weight) * prox_loss
        if not torch.isfinite(total_loss):
            raise SuffixV20FatalError("nonfinite_loss", "continuous_optimization")
        optimizer.zero_grad()
        total_loss.backward(inputs=[variable])
        if variable.grad is None or not torch.isfinite(variable.grad).all():
            raise SuffixV20FatalError("nonfinite_gradient", "continuous_optimization")
        variable.grad[:, ~variable_valid_mask, :] = 0
        optimizer.step()
        with torch.no_grad():
            variable[:, ~variable_valid_mask, :] = frozen_reference[
                :, ~variable_valid_mask, :
            ]
            if not torch.equal(
                variable[:, ~variable_valid_mask, :],
                frozen_reference[:, ~variable_valid_mask, :],
            ):
                raise SuffixV20FatalError(
                    "frozen_position_optimizer_pollution",
                    "continuous_optimization",
                )
            if not torch.isfinite(variable).all():
                raise SuffixV20FatalError(
                    "nonfinite_embedding", "continuous_optimization"
                )
        history.append({
            "step": step + 1,
            "joint_loss": float(joint_loss.detach().cpu()),
            "direction_loss": float(direction_loss.detach().cpu()),
            "magnitude_loss": float(magnitude_loss.detach().cpu()),
            "prox_loss": float(prox_loss.detach().cpu()),
            "range_loss": float(range_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        })
    full = (
        torch.cat((prefix_embedding, variable), dim=1)
        if prefix_embedding is not None else variable
    ).detach()
    return full, variable.detach(), {
        "optimizer": optimizer_name,
        "optimizer_recreated_each_step": optimizer_name == "SGD",
        "configured_epoch": int(epoch),
        "completed_steps": len(history),
        "lr": float(lr),
        "start": history[0] if history else None,
        "end": history[-1] if history else None,
    }


class _V20Engine:
    def __init__(self, *, model, embed_layer, tokenizer, targets,
                 attention_mask, continuous_embedding, effective_layers,
                 effective_weights, target_layer, config, filter_nonascii,
                 embedding_top_indices, select_candidate_from_top_indices,
                 get_perplexity, classifier_provider, eval_start_pos):
        self.model = model
        self.embed_layer = embed_layer
        self.tokenizer = tokenizer
        self.targets = targets
        self.attention_mask = attention_mask
        self.continuous = continuous_embedding
        self.layers = effective_layers
        self.weights = effective_weights
        self.target_layer = int(target_layer)
        self.config = config
        self.filter_nonascii = bool(filter_nonascii)
        self.embedding_top_indices = embedding_top_indices
        self.select_candidate_from_top_indices = select_candidate_from_top_indices
        self.get_perplexity = get_perplexity
        self.classifier_provider = classifier_provider
        self.eval_start_pos = int(eval_start_pos)
        self.device = continuous_embedding.device
        self.classifier_candidate_count = 0

    def embedding_candidates(self, position, top_k):
        top_indices = self.embedding_top_indices(
            self.continuous[0, position], self.embed_layer, int(top_k), "cosine"
        )
        _, token_ids = self.select_candidate_from_top_indices(
            top_indices, self.tokenizer, self.filter_nonascii
        )
        return [int(value) for value in token_ids][:int(top_k)]

    def ppl_candidates(self, position, committed_tokens):
        if position <= 0 or not committed_tokens[:position]:
            raise SuffixV20FatalError(
                "missing_committed_prefix_for_ppl", "candidate_generation", position
            )
        _, token_ids = self.get_perplexity(
            committed_tokens[:position], self.model,
            layer_id=self.target_layer, top_k=self.config.ppl_top_k,
        )
        return [int(value) for value in token_ids.detach().cpu().tolist()]

    def candidate_pool(self, position, committed_tokens, embedding_top_k,
                       include_current=None, include_classifier=False):
        embedding = self.embedding_candidates(position, embedding_top_k)
        ppl = self.ppl_candidates(position, committed_tokens)
        sources = [("embedding", embedding), ("perplexity", ppl)]
        if self.config.classifier_enabled and include_classifier:
            classifier = validate_classifier_candidates(
                self.classifier_provider, position, committed_tokens,
                self.continuous[:, position, :], self.config.classifier_top_k,
            )
            self.classifier_candidate_count += len(classifier)
            sources.append(("classifier", classifier))
        if include_current is not None:
            sources.append(("current", [int(include_current)]))
        candidates = merge_candidate_sources(
            sources, self.tokenizer, self.filter_nonascii
        )
        candidate_generation_failed = False
        if not candidates:
            candidate_generation_failed = True
            fallback = self.embedding_candidates(
                position, int(self.tokenizer.vocab_size)
            )
            candidates = merge_candidate_sources(
                [("embedding_fallback", fallback[:1])],
                self.tokenizer, self.filter_nonascii,
            )
        if not candidates:
            raise SuffixV20FatalError(
                "legal_candidate_pool_generation_failed",
                "candidate_generation", position,
            )
        return candidates, candidate_generation_failed

    def _hybrid_embeddings(self, tokens, position, candidate_ids):
        batch = self.continuous.expand(len(candidate_ids), -1, -1).clone()
        with torch.inference_mode():
            prefix_ids = torch.tensor(
                tokens[:position], dtype=torch.long, device=self.device
            )
            if position:
                batch[:, :position, :] = self.embed_layer(prefix_ids).to(
                    device=self.device, dtype=batch.dtype
                ).unsqueeze(0)
            ids = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
            batch[:, position, :] = self.embed_layer(ids).to(
                device=self.device, dtype=batch.dtype
            )
        return batch

    def score_candidates(self, position, tokens, candidates):
        candidate_ids = [entry["token_id"] for entry in candidates]
        batch = self._hybrid_embeddings(tokens, position, candidate_ids)
        collected = _collect_hidden(
            self.model, self.layers, self.attention_mask,
            inputs_embeds=batch, inference=True,
        )
        total_scores = torch.zeros(len(candidates), dtype=torch.float32, device=self.device)
        layer_scores = {}
        for layer_id, weight in zip(self.layers, self.weights):
            target = self.targets[layer_id][:, position, :].to(self.device)
            target = target.expand(len(candidates), -1)
            values, _, _ = direction_magnitude_joint_error(
                collected[layer_id][:, position, :], target,
                self.config.score_direction_weight,
                self.config.score_magnitude_weight,
                self.config.epsilon,
            )
            if not torch.isfinite(values).all():
                raise SuffixV20FatalError(
                    "nonfinite_candidate_score", "candidate_scoring", position
                )
            layer_scores[layer_id] = [float(value) for value in values.detach().cpu()]
            total_scores += float(weight) * values.to(self.device)
        total_values = [float(value) for value in total_scores.detach().cpu()]
        for index, entry in enumerate(candidates):
            entry["score"] = total_values[index]
            entry["target_layer_score"] = layer_scores[self.target_layer][index]
            entry["layer_scores"] = {
                str(layer_id): layer_scores[layer_id][index]
                for layer_id in self.layers
            }
        candidates.sort(key=candidate_tie_break_key)
        return candidates

    def score_current(self, position, tokens):
        entry = _candidate_entry(tokens[position])
        entry["sources"] = ["current"]
        entry["source_ranks"] = {"current": 1}
        return self.score_candidates(position, tokens, [entry])[0]["score"]

    def continuous_diagnostics(self, valid_positions):
        collected = _collect_hidden(
            self.model, self.layers, self.attention_mask,
            inputs_embeds=self.continuous, inference=True,
        )
        values = {}
        for position in valid_positions:
            total = 0.0
            for layer_id, weight in zip(self.layers, self.weights):
                joint, _, _ = direction_magnitude_joint_error(
                    collected[layer_id][:, position, :],
                    self.targets[layer_id][:, position, :].to(self.device),
                    self.config.score_direction_weight,
                    self.config.score_magnitude_weight,
                    self.config.epsilon,
                )
                total += float(weight) * float(joint.detach().cpu().reshape(-1)[0])
            values[position] = total
        return values


def _diagnostic_accuracy(tokens, targets, valid_positions):
    if not valid_positions:
        return 0.0
    return sum(int(tokens[pos]) == int(targets[pos]) for pos in valid_positions) / len(valid_positions)


def _prefix_accuracy(tokens, targets, valid_positions, end_position):
    positions = [position for position in valid_positions if position <= end_position]
    return _diagnostic_accuracy(tokens, targets, positions)


def _global_joint_error(engine, tokens, valid_positions):
    input_ids = torch.tensor([tokens], dtype=torch.long, device=engine.device)
    collected = _collect_hidden(
        engine.model, engine.layers, engine.attention_mask,
        input_ids=input_ids, inference=True,
    )
    mask = torch.zeros(len(tokens), dtype=torch.bool, device=engine.device)
    mask[valid_positions] = True
    loss, _, _ = _multilayer_loss(
        collected, engine.targets, engine.layers, engine.weights, mask,
        engine.config.score_direction_weight,
        engine.config.score_magnitude_weight,
        engine.config.epsilon,
    )
    return float(loss.detach().cpu())


def _entry_snapshot_tokens(entry_tokens):
    if entry_tokens is None:
        raise ValueError("suffix v2.0 requires an explicit non-oracle entry snapshot")
    return [int(value) for value in entry_tokens]


def _invariant_snapshot(tokens, attention_mask, eval_start_pos):
    attention = (
        [1] * len(tokens) if attention_mask is None
        else [int(value) for value in attention_mask[0].detach().cpu().tolist()]
    )
    return {
        "length": len(tokens),
        "prefix": list(tokens[:int(eval_start_pos)]),
        "padding": {
            str(position): int(tokens[position])
            for position, active in enumerate(attention) if not active
        },
        "attention": attention,
    }


def _assert_invariants(tokens, snapshot, stage):
    if len(tokens) != snapshot["length"]:
        raise SuffixV20FatalError("token_sequence_length_changed", stage)
    if list(tokens[:len(snapshot["prefix"])]) != snapshot["prefix"]:
        raise SuffixV20FatalError("special_prefix_changed", stage)
    for position, value in snapshot["padding"].items():
        if int(tokens[int(position)]) != int(value):
            raise SuffixV20FatalError("padding_changed", stage, int(position))


def _run_discrete(engine, entry_tokens, valid_positions, continuous_values,
                  enable_repairs):
    config = engine.config
    tokens = list(entry_tokens)
    raw_stage4_tokens = list(entry_tokens)
    metrics = []
    events = []
    prefix_snapshots = []
    d_history = []
    delta_history = []
    g_history = []
    previous_g = 0.0
    cumulative_g = 0.0
    previous_s = 0.0
    segment_start = None

    def calculate_metric(position, d_value):
        g_value = max(0.0, float(d_value) - float(continuous_values[position]))
        delta_g = 0.0 if not g_history else max(0.0, g_value - previous_g)
        suspicious, local_info = local_anomaly_decision(
            d_value, delta_g, d_history, delta_history,
            robust_upper_threshold(
                list(continuous_values.values()),
                config.continuous_mad_multiplier, config.mad_epsilon,
            ),
            config,
        )
        cumulative = update_cumulative_state(
            g_value, g_history, previous_s, config, position, segment_start
        )
        return {
            "position": int(position),
            "c": float(continuous_values[position]),
            "d": float(d_value),
            "g": g_value,
            "delta_g": delta_g,
            "G": cumulative_g + g_value,
            "z": cumulative["z"],
            "S": cumulative["S"],
            "local_suspicious": suspicious,
            "local_threshold": local_info,
            "cumulative_triggered": cumulative["triggered"],
            "cumulative_segment_start": cumulative["segment_start"],
        }

    def commit_metric(metric):
        nonlocal previous_g, cumulative_g, previous_s, segment_start
        previous_g = metric["g"]
        cumulative_g = metric["G"]
        previous_s = metric["S"]
        segment_start = metric["cumulative_segment_start"]
        metrics.append(metric)
        d_history.append(metric["d"])
        delta_history.append(metric["delta_g"])
        g_history.append(metric["g"])

    def rebuild_metrics_through(end_position):
        nonlocal metrics, d_history, delta_history, g_history
        nonlocal previous_g, cumulative_g, previous_s, segment_start
        metrics = []
        d_history = []
        delta_history = []
        g_history = []
        previous_g = 0.0
        cumulative_g = 0.0
        previous_s = 0.0
        segment_start = None
        expected_positions = [
            p for p in valid_positions if p <= end_position
        ]
        for position in expected_positions:
            d_value = engine.score_current(position, tokens)
            metric = calculate_metric(position, d_value)
            commit_metric(metric)
        if (
            [item["position"] for item in metrics] != expected_positions
            or not (
                len(metrics) == len(d_history) == len(delta_history)
                == len(g_history)
            )
        ):
            raise SuffixV20FatalError(
                "cumulative_recompute_history_mismatch",
                "cumulative_repair",
                end_position,
                segment=[expected_positions[0], end_position],
            )

    tau_c = robust_upper_threshold(
        list(continuous_values.values()),
        config.continuous_mad_multiplier, config.mad_epsilon,
    )
    for position in valid_positions:
        embedding_top_k = (
            config.normal_embedding_top_k
            if continuous_values[position] <= tau_c
            else config.expanded_embedding_top_k
        )
        candidates, generation_failed = engine.candidate_pool(
            position, tokens, embedding_top_k,
            include_classifier=(
                continuous_values[position] > tau_c
            ),
        )
        scored = engine.score_candidates(position, tokens, candidates)
        chosen = scored[0]
        tokens[position] = chosen["token_id"]
        raw_stage4_tokens[position] = chosen["token_id"]
        metric = calculate_metric(position, chosen["score"])
        event = {
            "position": position,
            "stage4_embedding_top_k": embedding_top_k,
            "candidate_generation_failed": generation_failed,
            "stage4_candidates": copy.deepcopy(scored),
            "stage4_selected_token": chosen["token_id"],
            "pre_repair": copy.deepcopy(metric),
            "local_repair": None,
            "cumulative_repair": None,
        }

        if enable_repairs and metric["local_suspicious"]:
            before_tokens = list(tokens)
            local_pre_repair_metric = copy.deepcopy(metric)
            repair_candidates, repair_generation_failed = engine.candidate_pool(
                position, tokens, config.expanded_embedding_top_k,
                include_current=tokens[position],
                include_classifier=True,
            )
            repair_scored = engine.score_candidates(
                position, tokens, repair_candidates
            )
            best = repair_scored[0]
            before_score = engine.score_current(position, tokens)
            changed = should_replace(
                before_score, best["score"], config.replace_epsilon
            )
            if changed:
                tokens[position] = best["token_id"]
            after_score = engine.score_current(position, tokens)
            metric = calculate_metric(position, after_score)
            event["local_repair"] = {
                "candidate_generation_failed": repair_generation_failed,
                "candidates": copy.deepcopy(repair_scored),
                "before_token": before_tokens[position],
                "after_token": tokens[position],
                "before_score": before_score,
                "after_score": after_score,
                "changed_positions": [position] if changed else [],
                "pre_repair_metric": local_pre_repair_metric,
                "post_repair_metric": copy.deepcopy(metric),
                "before_tokens_diagnostic_snapshot": before_tokens,
                "after_tokens_diagnostic_snapshot": list(tokens),
            }

        if enable_repairs and metric["cumulative_triggered"]:
            cumulative_pre_repair_metric = copy.deepcopy(metric)
            trigger_start = metric["cumulative_segment_start"]
            if trigger_start is None:
                trigger_start = position
            before_segment_tokens = list(tokens)
            changed_positions = []
            segment_steps = []
            for repair_position in [
                    p for p in valid_positions if trigger_start <= p <= position
            ]:
                before_token = tokens[repair_position]
                try:
                    repair_candidates, failed = engine.candidate_pool(
                        repair_position, tokens,
                        config.expanded_embedding_top_k,
                        include_current=before_token,
                        include_classifier=True,
                    )
                    repair_scored = engine.score_candidates(
                        repair_position, tokens, repair_candidates
                    )
                    best = repair_scored[0]
                    current_score = engine.score_current(
                        repair_position, tokens
                    )
                except SuffixV20FatalError as error:
                    if error.position is None:
                        error.position = int(repair_position)
                    if error.segment is None:
                        error.segment = [int(trigger_start), int(position)]
                    raise
                changed = should_replace(
                    current_score, best["score"], config.replace_epsilon
                )
                if changed:
                    tokens[repair_position] = best["token_id"]
                    changed_positions.append(repair_position)
                try:
                    after_score = engine.score_current(
                        repair_position, tokens
                    )
                except SuffixV20FatalError as error:
                    if error.position is None:
                        error.position = int(repair_position)
                    if error.segment is None:
                        error.segment = [int(trigger_start), int(position)]
                    raise
                segment_steps.append({
                    "position": repair_position,
                    "candidate_generation_failed": failed,
                    "before_token": before_token,
                    "after_token": tokens[repair_position],
                    "before_score": current_score,
                    "after_score": after_score,
                    "changed": changed,
                    "candidate_sources": {
                        str(item["token_id"]): item["sources"]
                        for item in repair_scored
                    },
                })
            pre_repair_s = metric["S"]
            pre_repair_g = metric["G"]
            try:
                rebuild_metrics_through(position)
            except SuffixV20FatalError as error:
                if error.position is None:
                    error.position = int(position)
                if error.segment is None:
                    error.segment = [int(trigger_start), int(position)]
                raise
            metric = metrics[-1]
            event["cumulative_repair"] = {
                "cumulative_segment_start": trigger_start,
                "cumulative_segment_end": position,
                "trigger_position": position,
                "pre_repair_S": pre_repair_s,
                "pre_repair_G": pre_repair_g,
                "post_repair_S": metric["S"],
                "post_repair_G": metric["G"],
                "pre_repair_metric": cumulative_pre_repair_metric,
                "post_repair_metric": copy.deepcopy(metric),
                "cumulative_unresolved": metric["S"] > config.cumulative_threshold,
                "changed_positions": changed_positions,
                "steps": segment_steps,
                "before_tokens_diagnostic_snapshot": before_segment_tokens,
                "after_tokens_diagnostic_snapshot": list(tokens),
                "max_repairs_per_trigger": config.cumulative_max_repairs_per_trigger,
            }

        if not event["cumulative_repair"]:
            commit_metric(metric)
        event["post_repair"] = copy.deepcopy(metric)
        events.append(event)
        prefix_snapshots.append((position, list(tokens)))
    return tokens, raw_stage4_tokens, metrics, events, prefix_snapshots, tau_c


def run_suffix_reoptimization_v2_0(
        model, embed_layer, initial_optimizable_embedding, prefix_embedding,
        target_hidden_states, attention_mask, layer_id, model_layer_count,
        tokenizer, total_input_ids, right_range, config,
        embedding_top_indices, select_candidate_from_top_indices,
        get_perplexity, entry_tokens, filter_nonascii=True,
        classifier_provider=None, eval_start_pos=0, log_file=None):
    del log_file
    if not config.enabled:
        return None, {
            "name": METHOD_NAME, "method": METHOD_NAME, "version": VERSION,
            "enabled": False, "skipped": True, "reason": "disabled",
            "classifier_enabled": bool(config.classifier_enabled),
            "classifier_provider_available": classifier_provider is not None,
            "classifier_candidate_count": 0,
        }
    entry_snapshot = _entry_snapshot_tokens(entry_tokens)
    initial_full_embedding = (
        torch.cat((prefix_embedding, initial_optimizable_embedding), dim=1)
        if prefix_embedding is not None else initial_optimizable_embedding
    )
    invariant = {
        "length": len(entry_snapshot),
        "prefix": list(entry_snapshot[:int(eval_start_pos)]),
        "padding": {},
        "attention": [1] * len(entry_snapshot),
    }
    phase1_full = None
    phase2_full = None
    try:
        if int(initial_full_embedding.shape[1]) != len(entry_snapshot):
            raise SuffixV20FatalError(
                "token_sequence_length_changed", "preflight"
            )
        if (
            attention_mask is not None
            and int(attention_mask[0].numel()) != len(entry_snapshot)
        ):
            raise SuffixV20FatalError(
                "attention_mask_length_changed", "preflight"
            )
        invariant = _invariant_snapshot(
            entry_snapshot, attention_mask, eval_start_pos
        )
        if config.classifier_enabled and classifier_provider is None:
            raise SuffixV20FatalError(
                "classifier_provider_unavailable", "preflight"
            )
        required_helpers = {
            "embedding_top_indices": embedding_top_indices,
            "select_candidate_from_top_indices": select_candidate_from_top_indices,
            "get_perplexity": get_perplexity,
        }
        missing = [name for name, value in required_helpers.items() if value is None]
        if missing:
            raise SuffixV20FatalError(
                "missing_helpers:{}".format(",".join(missing)), "preflight"
            )
        layers, weights, filtered_layers = resolve_effective_layers(
            layer_id, model_layer_count, config.layer_offsets, config.layer_weights
        )
        if set(layers) != set(int(key) for key in target_hidden_states):
            raise SuffixV20FatalError("target_hidden_layers_mismatch", "preflight")
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        if (
            int(eval_start_pos) <= 0
            or not entry_snapshot[:int(eval_start_pos)]
            or int(entry_snapshot[0]) not in special_ids
        ):
            raise SuffixV20FatalError(
                "missing_committed_prefix_for_ppl", "preflight", 0
            )
        valid_mask = _valid_mask(
            initial_full_embedding.shape[1], eval_start_pos,
            attention_mask, initial_full_embedding.device,
        )
        valid_positions = [
            index for index, active in enumerate(valid_mask.detach().cpu().tolist())
            if active
        ]
        if valid_positions[0] == 0:
            raise SuffixV20FatalError(
                "missing_committed_prefix_for_ppl", "preflight", 0
            )
        phase1_full, phase1_optimizable, phase1_summary = _optimize_phase(
            model, initial_optimizable_embedding, prefix_embedding,
            target_hidden_states, attention_mask, layers, weights, valid_mask,
            config.phase1_epoch, config.phase1_lr,
            config.phase1_direction_weight, config.phase1_magnitude_weight,
            right_range, config.range_weight, optimizer_name="SGD", clip=True,
            epsilon=config.epsilon,
        )
        phase2_full, _, phase2_summary = _optimize_phase(
            model, phase1_optimizable, prefix_embedding,
            target_hidden_states, attention_mask, layers, weights, valid_mask,
            config.phase2_epoch, config.phase2_lr,
            config.phase2_direction_weight, config.phase2_magnitude_weight,
            right_range, config.range_weight, config.prox_weight,
            phase1_optimizable, optimizer_name="Adam", clip=False,
            epsilon=config.epsilon,
        )
        engine = _V20Engine(
            model=model, embed_layer=embed_layer, tokenizer=tokenizer,
            targets=target_hidden_states, attention_mask=attention_mask,
            continuous_embedding=phase2_full, effective_layers=layers,
            effective_weights=weights, target_layer=layer_id, config=config,
            filter_nonascii=filter_nonascii,
            embedding_top_indices=embedding_top_indices,
            select_candidate_from_top_indices=select_candidate_from_top_indices,
            get_perplexity=get_perplexity,
            classifier_provider=classifier_provider, eval_start_pos=eval_start_pos,
        )
        continuous_values = engine.continuous_diagnostics(valid_positions)
        (
            final_tokens, raw_stage4_tokens, metrics, events,
            prefix_snapshots, tau_c,
        ) = _run_discrete(
            engine, entry_snapshot, valid_positions, continuous_values, True
        )
        _assert_invariants(final_tokens, invariant, "final_acceptance")
        formal_classifier_candidate_count = engine.classifier_candidate_count
        phase1_probe_tokens = None
        phase2_probe_tokens = None
        diagnostic_classifier_candidate_count = 0
        if config.accuracy_diagnostics_enabled:
            engine1 = _V20Engine(
                model=model, embed_layer=embed_layer, tokenizer=tokenizer,
                targets=target_hidden_states, attention_mask=attention_mask,
                continuous_embedding=phase1_full, effective_layers=layers,
                effective_weights=weights, target_layer=layer_id, config=config,
                filter_nonascii=filter_nonascii,
                embedding_top_indices=embedding_top_indices,
                select_candidate_from_top_indices=(
                    select_candidate_from_top_indices
                ),
                get_perplexity=get_perplexity,
                classifier_provider=classifier_provider,
                eval_start_pos=eval_start_pos,
            )
            phase1_c = engine1.continuous_diagnostics(valid_positions)
            phase1_probe_tokens, _, _, _, _, _ = _run_discrete(
                engine1, entry_snapshot, valid_positions, phase1_c, False
            )
            phase2_probe_tokens, _, _, _, _, _ = _run_discrete(
                engine, entry_snapshot, valid_positions, continuous_values, False
            )
            diagnostic_classifier_candidate_count = (
                engine1.classifier_candidate_count
                + engine.classifier_candidate_count
                - formal_classifier_candidate_count
            )
        target_tokens = [
            int(value) for value in total_input_ids[0].detach().cpu().tolist()
        ]
        if len(target_tokens) != len(entry_snapshot):
            raise SuffixV20FatalError(
                "token_sequence_length_changed", "diagnostics"
            )
        diagnostics = {}
        if config.accuracy_diagnostics_enabled:
            diagnostics = {
                "pre_v2_accuracy": _diagnostic_accuracy(
                    entry_snapshot, target_tokens, valid_positions
                ),
                "phase1_probe_accuracy": _diagnostic_accuracy(
                    phase1_probe_tokens, target_tokens, valid_positions
                ),
                "phase2_probe_accuracy": _diagnostic_accuracy(
                    phase2_probe_tokens, target_tokens, valid_positions
                ),
                "stage4_online_prefix_accuracy_trace": [
                    {
                        "position": position,
                        "oracle_prefix_accuracy": _prefix_accuracy(
                            tokens, target_tokens, valid_positions, position
                        ),
                        "evaluated_position_count": len([
                            p for p in valid_positions if p <= position
                        ]),
                    }
                    for position, tokens in prefix_snapshots
                ],
                "stage4_final_accuracy": _diagnostic_accuracy(
                    final_tokens, target_tokens, valid_positions
                ),
                "final_accuracy": _diagnostic_accuracy(
                    final_tokens, target_tokens, valid_positions
                ),
            }
            for event in events:
                local = event.get("local_repair")
                if local:
                    position = event["position"]
                    before_snapshot = local.pop(
                        "before_tokens_diagnostic_snapshot"
                    )
                    local["oracle_before_repair_prefix_accuracy"] = _prefix_accuracy(
                        before_snapshot,
                        target_tokens, valid_positions, position,
                    )
                    after_snapshot = local.pop("after_tokens_diagnostic_snapshot")
                    local["oracle_after_repair_prefix_accuracy"] = _prefix_accuracy(
                        after_snapshot, target_tokens, valid_positions, position,
                    )
                    local["oracle_before_repair_full_accuracy"] = _diagnostic_accuracy(
                        before_snapshot, target_tokens, valid_positions,
                    )
                    local["oracle_after_repair_full_accuracy"] = _diagnostic_accuracy(
                        after_snapshot, target_tokens, valid_positions,
                    )
                cumulative = event.get("cumulative_repair")
                if cumulative:
                    before = cumulative.pop("before_tokens_diagnostic_snapshot")
                    after = cumulative.pop("after_tokens_diagnostic_snapshot")
                    cumulative["oracle_before_segment_accuracy"] = _diagnostic_accuracy(
                        before, target_tokens, valid_positions
                    )
                    cumulative["oracle_after_segment_accuracy"] = _diagnostic_accuracy(
                        after, target_tokens, valid_positions
                    )
                    start = cumulative["cumulative_segment_start"]
                    end = cumulative["cumulative_segment_end"]
                    segment_positions = [p for p in valid_positions if start <= p <= end]
                    cumulative["oracle_segment_correct_before"] = sum(
                        before[p] == target_tokens[p] for p in segment_positions
                    )
                    cumulative["oracle_segment_correct_after"] = sum(
                        after[p] == target_tokens[p] for p in segment_positions
                    )
        else:
            for event in events:
                if event.get("local_repair"):
                    event["local_repair"].pop("before_tokens_diagnostic_snapshot", None)
                    event["local_repair"].pop("after_tokens_diagnostic_snapshot", None)
                if event.get("cumulative_repair"):
                    event["cumulative_repair"].pop("before_tokens_diagnostic_snapshot", None)
                    event["cumulative_repair"].pop("after_tokens_diagnostic_snapshot", None)
        stage4_global = _global_joint_error(engine, raw_stage4_tokens, valid_positions)
        final_global = _global_joint_error(engine, final_tokens, valid_positions)
        return phase2_full, {
            "name": METHOD_NAME,
            "method": METHOD_NAME,
            "version": VERSION,
            "enabled": True,
            "skipped": False,
            "accepted": True,
            "rollback": False,
            "rollback_reason": None,
            "failed_stage": None,
            "failed_position": None,
            "failed_segment": None,
            "fatal_failure": False,
            "reason": "completed_without_hard_failure",
            "pre_acc": diagnostics.get("pre_v2_accuracy"),
            "post_acc": diagnostics.get("final_accuracy"),
            "final_accuracy": diagnostics.get("final_accuracy"),
            "final_tokens": final_tokens,
            "final_text": tokenizer.decode(torch.tensor(final_tokens[eval_start_pos:])),
            "effective_layers": layers,
            "effective_layer_weights": weights,
            "filtered_layers": filtered_layers,
            "phase1": phase1_summary,
            "phase2": phase2_summary,
            "stage1": phase1_summary,
            "reoptimization": phase2_summary,
            "stage3": {
                "continuous_joint_error_by_position": {
                    str(position): value for position, value in continuous_values.items()
                },
                "tau_c": tau_c,
            },
            "stage4": {
                "metrics": metrics,
                "events": events,
                "raw_stage4_tokens": raw_stage4_tokens,
            },
            "events": events,
            "triggered": any(
                event.get("local_repair") or event.get("cumulative_repair")
                for event in events
            ),
            "diagnostics": diagnostics,
            "stage4_unrepaired_global_joint_error": stage4_global,
            "final_repaired_global_joint_error": final_global,
            "global_joint_error_delta": final_global - stage4_global,
            "classifier_enabled": bool(config.classifier_enabled),
            "classifier_provider_available": classifier_provider is not None,
            "classifier_candidate_count": formal_classifier_candidate_count,
            "classifier_diagnostic_candidate_count": (
                diagnostic_classifier_candidate_count
            ),
            "cumulative_heuristic_calibration_note": HEURISTIC_CALIBRATION_NOTE,
        }
    except Exception as error:
        if isinstance(error, SuffixV20FatalError):
            fatal = error
        else:
            fatal = SuffixV20FatalError(
                "fatal_runtime_failure:{}".format(type(error).__name__), "runtime"
            )
        rollback_embedding = initial_full_embedding.detach().clone()
        _assert_invariants(entry_snapshot, invariant, "rollback")
        valid_positions = [
            position for position, active in enumerate(invariant["attention"])
            if active and position >= int(eval_start_pos)
        ]
        final_accuracy = None
        if config.accuracy_diagnostics_enabled:
            try:
                target_tokens = [
                    int(value)
                    for value in total_input_ids[0].detach().cpu().tolist()
                ]
                if len(target_tokens) == len(entry_snapshot):
                    final_accuracy = _diagnostic_accuracy(
                        entry_snapshot, target_tokens, valid_positions
                    )
            except Exception:
                final_accuracy = None
        classifier_candidate_count = 0
        if "engine" in locals():
            classifier_candidate_count += int(engine.classifier_candidate_count)
        if "engine1" in locals():
            classifier_candidate_count += int(engine1.classifier_candidate_count)
        return rollback_embedding, {
            "name": METHOD_NAME,
            "method": METHOD_NAME,
            "version": VERSION,
            "enabled": True,
            "skipped": False,
            "accepted": False,
            "rollback": True,
            "rollback_reason": fatal.reason,
            "failed_stage": fatal.stage,
            "failed_position": fatal.position,
            "failed_segment": fatal.segment,
            "fatal_failure": True,
            "reason": fatal.reason,
            "pre_acc": final_accuracy,
            "post_acc": final_accuracy,
            "final_accuracy": final_accuracy,
            "final_tokens": entry_snapshot,
            "final_text": tokenizer.decode(torch.tensor(entry_snapshot[eval_start_pos:])),
            "classifier_enabled": bool(config.classifier_enabled),
            "classifier_provider_available": classifier_provider is not None,
            "classifier_candidate_count": classifier_candidate_count,
            "stage4_unrepaired_global_joint_error": None,
            "final_repaired_global_joint_error": None,
            "global_joint_error_delta": None,
        }


SuffixConfig = SuffixReoptimizationV20Config
run_suffix = run_suffix_reoptimization_v2_0
