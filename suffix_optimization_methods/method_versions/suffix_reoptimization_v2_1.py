"""Canonical suffix reoptimization v2.1 sidecar.

The formal method is target-token blind.  Optional oracle accuracy is computed
only after the formal token and embedding result has been frozen.
"""

from dataclasses import dataclass
import copy
import math
import numbers

import torch


METHOD_NAME = "suffix_reoptimization_v2.1"
VERSION = "v2.1"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192

__all__ = [
    "SuffixReoptimizationV21Config",
    "SuffixV21FatalError",
    "resolve_effective_layers",
    "build_entry_snapshot_from_embedding",
    "run_suffix_reoptimization_v2_1",
]


class SuffixV21FatalError(RuntimeError):
    """A formal-domain failure that requires entry-snapshot rollback."""

    def __init__(self, reason, stage, position=None):
        super().__init__(str(reason))
        self.reason = str(reason)
        self.stage = str(stage)
        self.position = None if position is None else int(position)


@dataclass
class SuffixReoptimizationV21Config:
    enabled: bool = False
    log_enabled: bool = True
    layer_offsets: tuple = (0, 1, 2)
    layer_weights: tuple = (1.0, 0.5, 0.25)
    alpha_dir: float = 0.5
    alpha_mag: float = 0.5
    vocab_weight: float = 0.005
    vocab_temperature: float = 0.01
    vocab_anchor_top_k: int = 10
    vocab_anchor_refresh_interval: int = 10
    global_optimizer: str = "adam"
    global_steps: int = 1000
    global_lr: float = 1e-3
    local_optimizer: str = "adam"
    local_steps: int = 50
    local_lr: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay_enabled: bool = False
    scheduler_mode: str = "none"
    tau_J: float = 0.15
    delta_c_max: float = 0.01
    tau_r: float = 0.05
    embedding_top_k_normal: int = 10
    embedding_top_k_expanded: int = 20
    ppl_top_k: int = 10
    vocab_distance_mode: str = "mean_squared_l2"
    vocab_softmin_mode: str = "normalized_stable_logsumexp"
    candidate_tie_break_mode: str = "hidden_error_token_id"
    hidden_epsilon: float = 1e-8
    epsilon_J: float = 1e-8
    epsilon_d: float = 1e-8
    accuracy_diagnostics_enabled: bool = False
    filter_nonascii: bool = True

    def __post_init__(self):
        boolean_fields = (
            "enabled",
            "log_enabled",
            "weight_decay_enabled",
            "accuracy_diagnostics_enabled",
            "filter_nonascii",
        )
        integer_fields = (
            "vocab_anchor_top_k",
            "vocab_anchor_refresh_interval",
            "global_steps",
            "local_steps",
            "embedding_top_k_normal",
            "embedding_top_k_expanded",
            "ppl_top_k",
        )
        float_fields = (
            "alpha_dir",
            "alpha_mag",
            "vocab_weight",
            "vocab_temperature",
            "global_lr",
            "local_lr",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "tau_J",
            "delta_c_max",
            "tau_r",
            "hidden_epsilon",
            "epsilon_J",
            "epsilon_d",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_1_{} must be boolean".format(name))
        if not isinstance(self.layer_offsets, (list, tuple)) or not all(
                isinstance(value, numbers.Integral)
                and not isinstance(value, bool)
                for value in self.layer_offsets):
            raise TypeError("suffix_v2_1_layer_offsets must contain integers")
        if not isinstance(self.layer_weights, (list, tuple)) or not all(
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                for value in self.layer_weights):
            raise TypeError("suffix_v2_1_layer_weights must contain numbers")
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_1_{} must be an integer".format(name))
            setattr(self, name, int(value))
        for name in float_fields:
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError("suffix_v2_1_{} must be numeric".format(name))
            setattr(self, name, float(value))
        self.layer_offsets = tuple(int(value) for value in self.layer_offsets)
        self.layer_weights = tuple(float(value) for value in self.layer_weights)
        _validate_config(self)


def _require_finite(name, value, *, positive=False, upper=None):
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "suffix_v2_1_{} must be finite and non-negative".format(name)
        )
    if positive and value <= 0.0:
        raise ValueError("suffix_v2_1_{} must be positive".format(name))
    if upper is not None and value > float(upper):
        raise ValueError(
            "suffix_v2_1_{} must be at most {}".format(name, upper)
        )


def _validate_config(config):
    if config.layer_offsets != (0, 1, 2):
        raise ValueError("suffix_v2_1_layer_offsets must be exactly [0, 1, 2]")
    if len(config.layer_weights) != 3:
        raise ValueError("suffix_v2_1_layer_weights must contain three values")
    for value in config.layer_weights:
        _require_finite("layer_weights", value)
    if sum(config.layer_weights) <= 0.0:
        raise ValueError("suffix_v2_1_layer_weights must have a positive sum")
    for name in (
        "alpha_dir",
        "alpha_mag",
        "vocab_weight",
        "tau_J",
        "delta_c_max",
    ):
        _require_finite(name, getattr(config, name))
    if config.alpha_dir + config.alpha_mag <= 0.0:
        raise ValueError("suffix_v2_1 alpha weights must have a positive sum")
    for name in (
        "vocab_temperature",
        "global_lr",
        "local_lr",
        "adam_epsilon",
        "hidden_epsilon",
        "epsilon_J",
        "epsilon_d",
    ):
        _require_finite(name, getattr(config, name), positive=True)
    for name in ("adam_beta1", "adam_beta2"):
        _require_finite(name, getattr(config, name))
        if getattr(config, name) >= 1.0:
            raise ValueError("suffix_v2_1_{} must be below 1".format(name))
    _require_finite("tau_r", config.tau_r, upper=1.0)
    _require_finite("delta_c_max", config.delta_c_max, upper=1.0)
    for name in (
        "vocab_anchor_top_k",
        "vocab_anchor_refresh_interval",
        "global_steps",
        "local_steps",
        "embedding_top_k_normal",
        "embedding_top_k_expanded",
        "ppl_top_k",
    ):
        if getattr(config, name) <= 0:
            raise ValueError("suffix_v2_1_{} must be positive".format(name))
    if config.embedding_top_k_expanded <= config.embedding_top_k_normal:
        raise ValueError(
            "suffix_v2_1 expanded embedding top-k must exceed normal top-k"
        )
    fixed_values = {
        "global_optimizer": "adam",
        "local_optimizer": "adam",
        "scheduler_mode": "none",
        "vocab_distance_mode": "mean_squared_l2",
        "vocab_softmin_mode": "normalized_stable_logsumexp",
        "candidate_tie_break_mode": "hidden_error_token_id",
    }
    for name, expected in fixed_values.items():
        if getattr(config, name) != expected:
            raise ValueError(
                "suffix_v2_1_{} must be {!r}".format(name, expected)
            )
    if config.weight_decay_enabled:
        raise ValueError("suffix_v2_1_weight_decay_enabled must be false")
    return config


def _normalized_pair(first, second):
    total = float(first) + float(second)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("joint-error weights must have a positive finite sum")
    return float(first) / total, float(second) / total


def resolve_effective_layers(target_layer, model_layer_count, offsets, weights):
    """Resolve external hidden-collection indices and normalize their weights."""
    if isinstance(target_layer, bool) or not isinstance(target_layer, numbers.Integral):
        raise TypeError("target_layer must be an integer")
    if (
        isinstance(model_layer_count, bool)
        or not isinstance(model_layer_count, numbers.Integral)
        or int(model_layer_count) <= 0
    ):
        raise ValueError("model_layer_count must be a positive integer")
    if int(target_layer) < 0:
        raise ValueError("target_layer must be non-negative")
    if len(offsets) != len(weights):
        raise ValueError("layer offsets and weights must have equal length")
    effective = []
    filtered = []
    for offset, weight in zip(offsets, weights):
        if not isinstance(offset, numbers.Integral) or isinstance(offset, bool):
            raise TypeError("layer offsets must be integers")
        if (
            not isinstance(weight, numbers.Real)
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError("layer weights must be finite and non-negative")
        layer = int(target_layer) + int(offset)
        if 0 <= layer < int(model_layer_count):
            effective.append((layer, float(weight)))
        else:
            filtered.append(layer)
    if not effective:
        raise ValueError("suffix v2.1 has no effective layers")
    total = sum(weight for _, weight in effective)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("suffix v2.1 effective layer weights sum to zero")
    return (
        [layer for layer, _ in effective],
        [weight / total for _, weight in effective],
        filtered,
    )


def _joint_error(current_hidden, target_hidden, config):
    current = torch.as_tensor(current_hidden).float()
    target = torch.as_tensor(target_hidden, device=current.device).float()
    if current.shape != target.shape:
        raise SuffixV21FatalError("hidden_shape_mismatch", "hidden_error")
    current_norm = torch.linalg.vector_norm(current, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    denominator = (current_norm * target_norm).clamp_min(
        float(config.hidden_epsilon)
    )
    cosine = ((current * target).sum(dim=-1) / denominator).clamp(-1.0, 1.0)
    direction = (1.0 - cosine) / 2.0
    magnitude = (
        (current_norm - target_norm)
        / (current_norm + target_norm + float(config.hidden_epsilon))
    ).pow(2)
    alpha_dir, alpha_mag = _normalized_pair(
        config.alpha_dir, config.alpha_mag
    )
    return (alpha_dir * direction + alpha_mag * magnitude).clamp(0.0, 1.0)


def _tokenizer_vocab_size(tokenizer, embed_layer):
    tokenizer_size = getattr(tokenizer, "vocab_size", None)
    embedding_size = int(embed_layer.weight.shape[0])
    if tokenizer_size is None:
        return embedding_size
    if int(tokenizer_size) != embedding_size:
        raise SuffixV21FatalError("vocab_size_mismatch", "preflight")
    return embedding_size


def _is_legal_token(token_id, tokenizer, vocab_size, filter_nonascii):
    token_id = int(token_id)
    if token_id < 0 or token_id >= int(vocab_size):
        return False
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id in special_ids or (
        pad_token_id is not None and token_id == int(pad_token_id)
    ):
        return False
    if filter_nonascii:
        try:
            if not tokenizer.decode([token_id]).isascii():
                return False
        except Exception:
            return False
    return True


class _LegalVocabulary:
    """Legal-token policy plus deterministic chunked embedding retrieval."""

    def __init__(self, embed_layer, tokenizer, filter_nonascii, chunk_size):
        self.embed_layer = embed_layer
        self.tokenizer = tokenizer
        self.vocab_size = _tokenizer_vocab_size(tokenizer, embed_layer)
        self.filter_nonascii = bool(filter_nonascii)
        self.chunk_size = int(chunk_size)
        self.ids = tuple(
            token_id for token_id in range(self.vocab_size)
            if _is_legal_token(
                token_id, tokenizer, self.vocab_size, self.filter_nonascii
            )
        )

    def is_legal(self, token_id):
        return _is_legal_token(
            token_id, self.tokenizer, self.vocab_size, self.filter_nonascii
        )

    def nearest(self, embedding, top_k):
        top_k = int(top_k)
        if top_k <= 0 or len(self.ids) < top_k:
            raise SuffixV21FatalError(
                "legal_vocab_top_k_contract_failed", "legal_vocab"
            )
        query = embedding.detach().reshape(-1)
        weight = self.embed_layer.weight.detach()
        if int(weight.shape[1]) != int(query.numel()):
            raise SuffixV21FatalError(
                "embedding_dimension_mismatch", "legal_vocab"
            )
        best = []
        for start in range(0, len(self.ids), self.chunk_size):
            chunk_ids = self.ids[start:start + self.chunk_size]
            index = torch.tensor(chunk_ids, dtype=torch.long, device=weight.device)
            anchors = weight.index_select(0, index)
            distances = (
                anchors.float() - query.to(weight.device).float().unsqueeze(0)
            ).pow(2).mean(dim=-1)
            for token_id, distance in zip(
                    chunk_ids, distances.detach().cpu().tolist()):
                if math.isfinite(float(distance)):
                    best.append((float(distance), int(token_id)))
            best.sort(key=lambda item: (item[0], item[1]))
            del best[top_k:]
        if len(best) < top_k:
            raise SuffixV21FatalError(
                "legal_vocab_nonfinite_distance", "legal_vocab"
            )
        ids = [token_id for _, token_id in best]
        index = torch.tensor(ids, dtype=torch.long, device=weight.device)
        anchors = weight.index_select(0, index).detach().clone()
        return ids, anchors, [distance for distance, _ in best]


def build_entry_snapshot_from_embedding(
        embedding, embed_layer, tokenizer, eval_start_pos,
        fixed_prefix_tokens, attention_mask=None, filter_nonascii=True,
        chunk_size=EMBEDDING_SEARCH_CHUNK_SIZE):
    """Discretize an entry embedding without consulting target token ids."""
    snapshot = torch.as_tensor(embedding)
    if snapshot.ndim != 3 or int(snapshot.shape[0]) != 1:
        raise SuffixV21FatalError("invalid_entry_embedding_shape", "entry_snapshot")
    sequence_length = int(snapshot.shape[1])
    eval_start_pos = int(eval_start_pos)
    prefix = [int(value) for value in fixed_prefix_tokens]
    if eval_start_pos < 0 or eval_start_pos > sequence_length:
        raise SuffixV21FatalError("invalid_eval_start_pos", "entry_snapshot")
    if len(prefix) != eval_start_pos:
        raise SuffixV21FatalError("fixed_prefix_length_mismatch", "entry_snapshot")
    if attention_mask is None:
        active = [1] * sequence_length
    else:
        mask = torch.as_tensor(attention_mask)
        if mask.shape != (1, sequence_length):
            raise SuffixV21FatalError("attention_mask_shape_mismatch", "entry_snapshot")
        active = [int(value) for value in mask[0].detach().cpu().tolist()]
        if any(value not in (0, 1) for value in active):
            raise SuffixV21FatalError("attention_mask_not_binary", "entry_snapshot")
    legal = _LegalVocabulary(embed_layer, tokenizer, filter_nonascii, chunk_size)
    tokens = list(prefix)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    for position in range(eval_start_pos, sequence_length):
        if not active[position]:
            if pad_token_id is None:
                raise SuffixV21FatalError(
                    "padding_token_unavailable", "entry_snapshot", position
                )
            tokens.append(int(pad_token_id))
            continue
        ids, _, _ = legal.nearest(snapshot[0, position], 1)
        tokens.append(ids[0])
    return tokens


def _validate_model_contract(model):
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [
        str(value).lower()
        for value in (getattr(config, "architectures", None) or [])
    ]
    type_supported = model_type in {"qwen2", "qwen2_5", "qwen2.5"}
    architecture_supported = any(
        "qwen2" in value and "causallm" in value
        for value in architectures
    )
    class_name = type(model).__name__.lower()
    architecture_supported = architecture_supported or (
        "qwen2" in class_name and "causallm" in class_name
    )
    if not type_supported or not architecture_supported:
        raise SuffixV21FatalError("unsupported_model_family", "preflight")
    if not hasattr(model, "forward"):
        raise SuffixV21FatalError("causal_lm_forward_unavailable", "preflight")


def _register_layer_hooks(model, layer_ids, destination):
    requested = {int(layer_id) for layer_id in layer_ids}
    registered = set()
    handles = []

    def layer_index(name):
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
        current = layer_index(name)
        if current in requested and current not in registered:
            handles.append(module.register_forward_hook(make_hook(current)))
            registered.add(current)
    if registered != requested:
        for handle in handles:
            handle.remove()
        raise SuffixV21FatalError("missing_layer_hooks", "preflight")
    return handles


def _expanded_attention_mask(attention_mask, batch_size, device):
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=device)
    if int(mask.shape[0]) == 1 and int(batch_size) > 1:
        mask = mask.expand(int(batch_size), -1)
    return mask


def _forward_hidden(
        model, layer_ids, attention_mask, *, input_ids=None,
        inputs_embeds=None, inference=False, require_finite=True,
        return_output=False):
    collected = {}
    handles = _register_layer_hooks(model, layer_ids, collected)
    kwargs = {"use_cache": False}
    if input_ids is not None:
        kwargs["input_ids"] = input_ids
        device = input_ids.device
        batch_size = int(input_ids.shape[0])
    elif inputs_embeds is not None:
        kwargs["inputs_embeds"] = inputs_embeds
        device = inputs_embeds.device
        batch_size = int(inputs_embeds.shape[0])
    else:
        for handle in handles:
            handle.remove()
        raise SuffixV21FatalError("missing_forward_input", "forward")
    mask = _expanded_attention_mask(attention_mask, batch_size, device)
    if mask is not None:
        kwargs["attention_mask"] = mask
    try:
        if inference:
            with torch.inference_mode():
                output = model(**kwargs)
        else:
            output = model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    if set(collected) != {int(value) for value in layer_ids}:
        raise SuffixV21FatalError("missing_hidden_state", "forward")
    if require_finite and any(
            not torch.isfinite(value).all() for value in collected.values()):
        raise SuffixV21FatalError("nonfinite_hidden_state", "forward")
    if return_output:
        return collected, output
    return collected


def _multi_hidden_error(collected, targets, layers, weights, positions, config):
    if not positions:
        raise SuffixV21FatalError("empty_effective_positions", "hidden_error")
    position_index = torch.tensor(
        positions, dtype=torch.long, device=collected[layers[0]].device
    )
    total = torch.zeros(
        (int(collected[layers[0]].shape[0]), len(positions)),
        dtype=torch.float32,
        device=collected[layers[0]].device,
    )
    for layer_id, weight in zip(layers, weights):
        current = collected[layer_id].index_select(1, position_index)
        target = targets[layer_id].detach().to(current.device).index_select(
            1, position_index
        )
        if int(target.shape[0]) == 1 and int(current.shape[0]) > 1:
            target = target.expand(int(current.shape[0]), -1, -1)
        total = total + float(weight) * _joint_error(current, target, config)
    return total.clamp(0.0, 1.0)


def _fresh_vocab_metric(embedding, legal_vocab, config):
    _, anchors, _ = legal_vocab.nearest(
        embedding, config.vocab_anchor_top_k
    )
    return _vocab_metric_with_anchors(embedding, anchors, config)


def _vocab_metric_with_anchors(embedding, anchors, config):
    current = embedding.reshape(1, -1).float()
    anchors = anchors.to(device=current.device).detach().float()
    distances = (current - anchors).pow(2).mean(dim=-1)
    value = -float(config.vocab_temperature) * (
        torch.logsumexp(
            -distances / float(config.vocab_temperature), dim=0
        )
        - math.log(int(anchors.shape[0]))
    )
    return value.clamp_min(0.0)


def _assemble_full_embedding(base, positions, trainable):
    index = torch.tensor(positions, dtype=torch.long, device=base.device)
    return base.index_copy(1, index, trainable)


def _anchor_cache(trainable, legal_vocab, config):
    anchors = []
    for index in range(int(trainable.shape[1])):
        _, current_anchors, _ = legal_vocab.nearest(
            trainable[0, index], config.vocab_anchor_top_k
        )
        anchors.append(
            current_anchors.to(
                device=trainable.device, dtype=trainable.dtype
            ).detach()
        )
    return anchors


def _global_optimize(
        model, entry_embedding, targets, attention_mask, positions, layers,
        weights, legal_vocab, config):
    base = entry_embedding.detach().clone()
    position_index = torch.tensor(
        positions, dtype=torch.long, device=base.device
    )
    trainable = torch.nn.Parameter(
        base.index_select(1, position_index).detach().clone()
    )
    optimizer = torch.optim.Adam(
        [trainable],
        lr=float(config.global_lr),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
        weight_decay=0.0,
    )
    anchors = None
    anchor_refresh_count = 0
    history_start = None
    history_end = None
    for step in range(int(config.global_steps)):
        if anchors is None or step % int(config.vocab_anchor_refresh_interval) == 0:
            anchors = _anchor_cache(trainable, legal_vocab, config)
            anchor_refresh_count += 1
        full = _assemble_full_embedding(base, positions, trainable)
        collected = _forward_hidden(
            model,
            layers,
            attention_mask,
            inputs_embeds=full,
            inference=False,
            require_finite=True,
        )
        hidden_loss = _multi_hidden_error(
            collected, targets, layers, weights, positions, config
        ).mean()
        vocab_values = [
            _vocab_metric_with_anchors(trainable[0, index], anchors[index], config)
            for index in range(len(positions))
        ]
        vocab_loss = torch.stack(vocab_values).mean()
        total_loss = hidden_loss + float(config.vocab_weight) * vocab_loss
        if not torch.isfinite(total_loss):
            raise SuffixV21FatalError(
                "nonfinite_global_loss", "global_optimization"
            )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward(inputs=[trainable])
        if trainable.grad is None or not torch.isfinite(trainable.grad).all():
            raise SuffixV21FatalError(
                "nonfinite_global_gradient", "global_optimization"
            )
        optimizer.step()
        if not torch.isfinite(trainable).all():
            raise SuffixV21FatalError(
                "nonfinite_global_update", "global_optimization"
            )
        record = {
            "step": step + 1,
            "hidden_loss": float(hidden_loss.detach().cpu()),
            "vocab_loss": float(vocab_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }
        if history_start is None:
            history_start = record
        history_end = record
    work = _assemble_full_embedding(base, positions, trainable).detach()
    for position in range(int(base.shape[1])):
        if position not in positions and not torch.equal(
                work[:, position], base[:, position]):
            raise SuffixV21FatalError(
                "frozen_embedding_changed", "global_optimization", position
            )
    return work, {
        "optimizer": "Adam",
        "optimizer_instances": 1,
        "optimizer_persistent": True,
        "weight_decay": 0.0,
        "scheduler": None,
        "configured_steps": int(config.global_steps),
        "completed_steps": int(config.global_steps),
        "lr": float(config.global_lr),
        "anchor_refresh_count": anchor_refresh_count,
        "anchor_refresh_recreated_optimizer": False,
        "start": history_start,
        "end": history_end,
    }


def _mixed_context(work, committed_tokens, committed_positions, embed_layer,
                   current_position=None, current_embedding=None):
    mixed = work.detach().clone()
    if committed_positions:
        ids = torch.tensor(
            [int(committed_tokens[position]) for position in committed_positions],
            dtype=torch.long,
            device=work.device,
        )
        token_embeddings = embed_layer(ids).to(
            device=work.device, dtype=work.dtype
        ).detach()
        index = torch.tensor(
            committed_positions, dtype=torch.long, device=work.device
        )
        mixed = mixed.index_copy(1, index, token_embeddings.unsqueeze(0))
    if current_position is not None and current_embedding is not None:
        index = torch.tensor(
            [int(current_position)], dtype=torch.long, device=work.device
        )
        mixed = mixed.index_copy(
            1, index, current_embedding.reshape(1, 1, -1)
        )
    return mixed


def _continuous_metrics(
        model, mixed, position, targets, attention_mask, layers, weights,
        legal_vocab, config):
    collected = _forward_hidden(
        model,
        layers,
        attention_mask,
        inputs_embeds=mixed,
        inference=True,
        require_finite=True,
    )
    hidden = _multi_hidden_error(
        collected, targets, layers, weights, [position], config
    )[0, 0]
    vocab = _fresh_vocab_metric(mixed[0, position], legal_vocab, config)
    objective = hidden + float(config.vocab_weight) * vocab
    if not all(torch.isfinite(value) for value in (hidden, vocab, objective)):
        raise SuffixV21FatalError(
            "nonfinite_continuous_metric", "continuous_gate", position
        )
    return (
        float(hidden.detach().cpu()),
        float(vocab.detach().cpu()),
        float(objective.detach().cpu()),
    )


def _try_vector_repair(
        model, work, committed_tokens, committed_positions, position,
        embed_layer, targets, attention_mask, layers, weights, legal_vocab,
        config, old_metrics):
    variable = torch.nn.Parameter(
        work[:, position, :].detach().clone()
    )
    optimizer = torch.optim.Adam(
        [variable],
        lr=float(config.local_lr),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
        weight_decay=0.0,
    )
    anchors = None
    refresh_count = 0
    start = None
    end = None
    try:
        for step in range(int(config.local_steps)):
            if (
                anchors is None
                or step % int(config.vocab_anchor_refresh_interval) == 0
            ):
                _, anchors, _ = legal_vocab.nearest(
                    variable[0], config.vocab_anchor_top_k
                )
                anchors = anchors.to(
                    device=variable.device, dtype=variable.dtype
                ).detach()
                refresh_count += 1
            mixed = _mixed_context(
                work,
                committed_tokens,
                committed_positions,
                embed_layer,
                current_position=position,
                current_embedding=variable,
            )
            collected = _forward_hidden(
                model,
                layers,
                attention_mask,
                inputs_embeds=mixed,
                inference=False,
                require_finite=True,
            )
            hidden_loss = _multi_hidden_error(
                collected, targets, layers, weights, [position], config
            ).mean()
            vocab_loss = _vocab_metric_with_anchors(
                variable[0], anchors, config
            )
            total_loss = hidden_loss + float(config.vocab_weight) * vocab_loss
            if not torch.isfinite(total_loss):
                raise SuffixV21FatalError(
                    "nonfinite_local_loss", "vector_repair", position
                )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward(inputs=[variable])
            if variable.grad is None or not torch.isfinite(variable.grad).all():
                raise SuffixV21FatalError(
                    "nonfinite_local_gradient", "vector_repair", position
                )
            optimizer.step()
            if not torch.isfinite(variable).all():
                raise SuffixV21FatalError(
                    "nonfinite_local_update", "vector_repair", position
                )
            record = {
                "step": step + 1,
                "hidden_loss": float(hidden_loss.detach().cpu()),
                "vocab_loss": float(vocab_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
            }
            if start is None:
                start = record
            end = record
        proposed = _mixed_context(
            work,
            committed_tokens,
            committed_positions,
            embed_layer,
            current_position=position,
            current_embedding=variable.detach(),
        )
        new_metrics = _continuous_metrics(
            model,
            proposed,
            position,
            targets,
            attention_mask,
            layers,
            weights,
            legal_vocab,
            config,
        )
    except Exception as error:
        reason = (
            error.reason
            if isinstance(error, SuffixV21FatalError)
            else "trial_exception:{}".format(type(error).__name__)
        )
        return work, {
            "triggered": True,
            "trial_safe": False,
            "accepted": False,
            "fallback_reason": reason,
            "optimizer": "Adam",
            "optimizer_instances": 1,
            "optimizer_persistent_within_trial": True,
            "configured_steps": int(config.local_steps),
            "completed_steps": 0 if end is None else int(end["step"]),
            "anchor_refresh_count": refresh_count,
            "old_c": old_metrics[0],
            "old_R": old_metrics[1],
            "old_J": old_metrics[2],
            "new_c": None,
            "new_R": None,
            "new_J": None,
            "strict_J_improvement": False,
            "hidden_cap_passed": False,
            "start": start,
            "end": end,
        }
    strict_improvement = (
        new_metrics[2] < old_metrics[2] - float(config.epsilon_J)
    )
    hidden_cap_passed = (
        new_metrics[0] <= old_metrics[0] + float(config.delta_c_max)
    )
    accepted = strict_improvement and hidden_cap_passed
    updated = work
    if accepted:
        index = torch.tensor([position], dtype=torch.long, device=work.device)
        updated = work.index_copy(
            1, index, variable.detach().reshape(1, 1, -1)
        ).detach()
    return updated, {
        "triggered": True,
        "trial_safe": True,
        "accepted": bool(accepted),
        "fallback_reason": None if accepted else (
            "hidden_cap_failed" if not hidden_cap_passed
            else "no_strict_J_improvement"
        ),
        "optimizer": "Adam",
        "optimizer_instances": 1,
        "optimizer_persistent_within_trial": True,
        "configured_steps": int(config.local_steps),
        "completed_steps": int(config.local_steps),
        "anchor_refresh_count": refresh_count,
        "old_c": old_metrics[0],
        "old_R": old_metrics[1],
        "old_J": old_metrics[2],
        "new_c": new_metrics[0],
        "new_R": new_metrics[1],
        "new_J": new_metrics[2],
        "strict_J_improvement": bool(strict_improvement),
        "hidden_cap_passed": bool(hidden_cap_passed),
        "start": start,
        "end": end,
    }


def _next_token_candidates(model, committed_prefix, legal_vocab, top_k,
                           attention_mask):
    if not committed_prefix:
        raise SuffixV21FatalError(
            "missing_committed_prefix_for_ppl", "candidate_generation"
        )
    device = legal_vocab.embed_layer.weight.device
    input_ids = torch.tensor(
        [committed_prefix], dtype=torch.long, device=device
    )
    prefix_mask = None
    if attention_mask is not None:
        prefix_mask = attention_mask[:, :len(committed_prefix)].to(device)
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if prefix_mask is not None:
        kwargs["attention_mask"] = prefix_mask
    with torch.inference_mode():
        output = model(**kwargs)
    logits = getattr(output, "logits", None)
    if (
        logits is None
        or logits.ndim != 3
        or int(logits.shape[0]) != 1
        or int(logits.shape[1]) != len(committed_prefix)
        or int(logits.shape[2]) != legal_vocab.vocab_size
    ):
        raise SuffixV21FatalError(
            "causal_lm_logits_contract_failed", "candidate_generation"
        )
    final_logits = logits[0, -1].detach().float().cpu()
    ranked = []
    for token_id in legal_vocab.ids:
        score = float(final_logits[token_id])
        if math.isfinite(score):
            ranked.append((-score, int(token_id)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [token_id for _, token_id in ranked[:int(top_k)]]


def _merge_candidate_sources(source_candidates, legal_vocab):
    merged = {}
    for source, candidates in source_candidates:
        for rank, raw_token_id in enumerate(candidates, start=1):
            if isinstance(raw_token_id, bool) or not isinstance(
                    raw_token_id, numbers.Integral):
                continue
            token_id = int(raw_token_id)
            if not legal_vocab.is_legal(token_id):
                continue
            if token_id not in merged:
                merged[token_id] = {
                    "token_id": token_id,
                    "sources": [],
                    "source_ranks": {},
                }
            entry = merged[token_id]
            if source not in entry["sources"]:
                entry["sources"].append(source)
                entry["source_ranks"][source] = rank
    return [merged[token_id] for token_id in sorted(merged)]


def _candidate_sources(
        model, work, position, committed_tokens, legal_vocab, attention_mask,
        config):
    normal_ids, _, _ = legal_vocab.nearest(
        work[0, position], config.embedding_top_k_normal
    )
    expanded_ids, _, _ = legal_vocab.nearest(
        work[0, position], config.embedding_top_k_expanded
    )
    ppl_ids = _next_token_candidates(
        model,
        committed_tokens[:position],
        legal_vocab,
        config.ppl_top_k,
        attention_mask,
    )
    current = [int(committed_tokens[position])]
    return {
        "embedding_normal": normal_ids,
        "embedding_expanded": expanded_ids,
        "perplexity": ppl_ids,
        "current": current,
    }


def _pool_from_sources(sources, mode, legal_vocab):
    embedding_source = (
        "embedding_expanded" if mode == "expanded" else "embedding_normal"
    )
    pool = _merge_candidate_sources(
        [
            (embedding_source, sources[embedding_source]),
            ("perplexity", sources["perplexity"]),
            ("current", sources["current"]),
        ],
        legal_vocab,
    )
    if not pool:
        raise SuffixV21FatalError(
            "formal_candidate_pool_empty", "candidate_generation"
        )
    return pool


def _candidate_batch(
        work, committed_tokens, committed_positions, position,
        candidate_ids, embed_layer):
    base = _mixed_context(
        work, committed_tokens, committed_positions, embed_layer
    )
    batch = base.expand(len(candidate_ids), -1, -1).clone()
    ids = torch.tensor(candidate_ids, dtype=torch.long, device=work.device)
    values = embed_layer(ids).to(device=work.device, dtype=work.dtype).detach()
    batch[:, position, :] = values
    return batch.detach()


def _score_candidates(
        model, work, committed_tokens, committed_positions, position,
        candidates, embed_layer, targets, attention_mask, layers, weights,
        config):
    candidate_ids = [entry["token_id"] for entry in candidates]
    batch = _candidate_batch(
        work,
        committed_tokens,
        committed_positions,
        position,
        candidate_ids,
        embed_layer,
    )
    collected = _forward_hidden(
        model,
        layers,
        attention_mask,
        inputs_embeds=batch,
        inference=True,
        require_finite=False,
    )
    totals = _multi_hidden_error(
        collected, targets, layers, weights, [position], config
    )[:, 0]
    per_layer = {}
    for layer_id in layers:
        current = collected[layer_id][:, position, :]
        target = targets[layer_id][:, position, :].detach().to(current.device)
        if int(target.shape[0]) == 1 and len(candidates) > 1:
            target = target.expand(len(candidates), -1)
        per_layer[layer_id] = _joint_error(current, target, config)
    scored = []
    dropped = []
    for index, entry in enumerate(candidates):
        score = float(totals[index].detach().cpu())
        layer_values = {
            str(layer_id): float(per_layer[layer_id][index].detach().cpu())
            for layer_id in layers
        }
        if not math.isfinite(score) or not all(
                math.isfinite(value) for value in layer_values.values()):
            dropped.append({
                "token_id": int(entry["token_id"]),
                "sources": list(entry["sources"]),
                "reason": "nonfinite_candidate_score",
            })
            continue
        current_entry = copy.deepcopy(entry)
        current_entry["hidden_error"] = score
        current_entry["layer_hidden_errors"] = layer_values
        scored.append(current_entry)
    if not scored:
        raise SuffixV21FatalError(
            "all_candidate_scores_nonfinite", "candidate_scoring", position
        )
    scored.sort(key=lambda entry: (entry["hidden_error"], entry["token_id"]))
    return scored, dropped


def _snapshot_tokens(entry_token_snapshot):
    if entry_token_snapshot is None or isinstance(
            entry_token_snapshot, (str, bytes)):
        raise SuffixV21FatalError("missing_entry_token_snapshot", "preflight")
    values = list(entry_token_snapshot)
    if not values:
        raise SuffixV21FatalError("empty_entry_token_snapshot", "preflight")
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Integral)
        for value in values
    ):
        raise SuffixV21FatalError("invalid_entry_token_snapshot", "preflight")
    return [int(value) for value in values]


def _preflight(
        model, embed_layer, entry_embedding_snapshot, entry_token_snapshot,
        target_hidden_states, attention_mask, layer_id, model_layer_count,
        tokenizer, config, eval_start_pos):
    if not config.enabled:
        raise SuffixV21FatalError("suffix_v2_1_disabled", "preflight")
    _validate_model_contract(model)
    if embed_layer is None or not hasattr(embed_layer, "weight"):
        raise SuffixV21FatalError("embedding_layer_unavailable", "preflight")
    entry_embedding = torch.as_tensor(entry_embedding_snapshot).detach().clone()
    if entry_embedding.ndim != 3 or int(entry_embedding.shape[0]) != 1:
        raise SuffixV21FatalError("invalid_entry_embedding_shape", "preflight")
    if not torch.isfinite(entry_embedding).all():
        raise SuffixV21FatalError("nonfinite_entry_embedding", "preflight")
    entry_tokens = _snapshot_tokens(entry_token_snapshot)
    sequence_length = int(entry_embedding.shape[1])
    if len(entry_tokens) != sequence_length:
        raise SuffixV21FatalError("token_sequence_length_changed", "preflight")
    if int(entry_embedding.shape[2]) != int(embed_layer.weight.shape[1]):
        raise SuffixV21FatalError("embedding_dimension_mismatch", "preflight")
    if entry_embedding.device != embed_layer.weight.device:
        raise SuffixV21FatalError("embedding_device_mismatch", "preflight")
    eval_start_pos = int(eval_start_pos)
    if eval_start_pos <= 0 or eval_start_pos >= sequence_length:
        raise SuffixV21FatalError("invalid_eval_start_pos", "preflight")
    special_ids = {
        int(value) for value in getattr(tokenizer, "all_special_ids", [])
    }
    if not special_ids or entry_tokens[0] not in special_ids:
        raise SuffixV21FatalError(
            "missing_committed_special_prefix", "preflight", 0
        )
    if attention_mask is None:
        mask = torch.ones(
            (1, sequence_length), dtype=torch.long, device=entry_embedding.device
        )
    else:
        mask = torch.as_tensor(attention_mask, device=entry_embedding.device)
        if mask.shape != (1, sequence_length):
            raise SuffixV21FatalError(
                "attention_mask_shape_mismatch", "preflight"
            )
        if any(
            int(value) not in (0, 1)
            for value in mask[0].detach().cpu().tolist()
        ):
            raise SuffixV21FatalError("attention_mask_not_binary", "preflight")
        mask = mask.detach().clone()
    positions = [
        position
        for position in range(eval_start_pos, sequence_length)
        if int(mask[0, position]) == 1
    ]
    if not positions:
        raise SuffixV21FatalError("empty_effective_positions", "preflight")
    layers, weights, filtered_layers = resolve_effective_layers(
        layer_id,
        model_layer_count,
        config.layer_offsets,
        config.layer_weights,
    )
    decoder = getattr(model, "model", None)
    decoder_layers = getattr(decoder, "layers", None)
    if decoder_layers is None or len(decoder_layers) != int(model_layer_count):
        raise SuffixV21FatalError("model_layer_count_mismatch", "preflight")
    if not isinstance(target_hidden_states, dict) or set(
            target_hidden_states
    ) != set(layers):
        raise SuffixV21FatalError("target_hidden_layers_mismatch", "preflight")
    for layer_id_value in layers:
        target = torch.as_tensor(target_hidden_states[layer_id_value])
        if target.ndim != 3 or int(target.shape[0]) != 1:
            raise SuffixV21FatalError("target_hidden_shape_mismatch", "preflight")
        if int(target.shape[1]) != sequence_length:
            raise SuffixV21FatalError("target_hidden_length_mismatch", "preflight")
        if not torch.isfinite(target).all():
            raise SuffixV21FatalError("nonfinite_target_hidden", "preflight")
    legal_vocab = _LegalVocabulary(
        embed_layer,
        tokenizer,
        config.filter_nonascii,
        EMBEDDING_SEARCH_CHUNK_SIZE,
    )
    required_legal_count = max(
        int(config.vocab_anchor_top_k),
        int(config.embedding_top_k_expanded),
    )
    if len(legal_vocab.ids) < required_legal_count:
        raise SuffixV21FatalError("legal_vocab_too_small", "preflight")
    legal_vocab.nearest(entry_embedding[0, positions[0]], required_legal_count)
    for position in positions:
        if not legal_vocab.is_legal(entry_tokens[position]):
            raise SuffixV21FatalError(
                "illegal_working_token", "preflight", position
            )
    collected, output = _forward_hidden(
        model,
        layers,
        mask,
        inputs_embeds=entry_embedding,
        inference=True,
        require_finite=True,
        return_output=True,
    )
    logits = getattr(output, "logits", None)
    if (
        logits is None
        or logits.ndim != 3
        or tuple(logits.shape[:2]) != (1, sequence_length)
        or int(logits.shape[2]) != legal_vocab.vocab_size
    ):
        raise SuffixV21FatalError(
            "causal_lm_logits_contract_failed", "preflight"
        )
    for layer_id_value in layers:
        if collected[layer_id_value].shape != target_hidden_states[layer_id_value].shape:
            raise SuffixV21FatalError("target_hidden_shape_mismatch", "preflight")
    return {
        "entry_embedding": entry_embedding,
        "entry_tokens": entry_tokens,
        "attention_mask": mask,
        "positions": positions,
        "layers": layers,
        "weights": weights,
        "filtered_layers": filtered_layers,
        "legal_vocab": legal_vocab,
        "eval_start_pos": eval_start_pos,
    }


def _position_event(
        position, baseline, repair, initial_mode, initial_scored,
        initial_dropped, selected_first, r_first, expanded_attempt,
        expanded_added_count, expanded_scored, expanded_dropped,
        selected_final, r_final, unresolved_continuous,
        unresolved_local):
    return {
        "position": int(position),
        "c_i": baseline[0],
        "R_i": baseline[1],
        "J_i": baseline[2],
        "vector_repair": repair,
        "initial_pool_mode": initial_mode,
        "initial_candidates": initial_scored,
        "initial_nonfinite_candidates_dropped": initial_dropped,
        "selected_initial_token": int(selected_first["token_id"]),
        "d_old": float(selected_first["hidden_error"]),
        "r_old": float(r_first),
        "expanded_attempt": bool(expanded_attempt),
        "expanded_added_count": expanded_added_count,
        "expanded_candidates": expanded_scored,
        "expanded_nonfinite_candidates_dropped": expanded_dropped,
        "selected_final_token": int(selected_final["token_id"]),
        "d_final": float(selected_final["hidden_error"]),
        "r_final": float(r_final),
        "unresolved_continuous_quality": bool(unresolved_continuous),
        "unresolved_local_degradation": bool(unresolved_local),
    }


def _assert_final_invariants(
        entry_embedding, entry_tokens, final_embedding, final_tokens,
        positions, embed_layer):
    if len(final_tokens) != len(entry_tokens):
        raise SuffixV21FatalError(
            "token_sequence_length_changed", "final_invariant"
        )
    if final_embedding.shape != entry_embedding.shape:
        raise SuffixV21FatalError(
            "embedding_shape_changed", "final_invariant"
        )
    position_set = set(positions)
    for position in range(len(entry_tokens)):
        if position not in position_set:
            if int(final_tokens[position]) != int(entry_tokens[position]):
                raise SuffixV21FatalError(
                    "fixed_token_changed", "final_invariant", position
                )
            if not torch.equal(
                    final_embedding[:, position], entry_embedding[:, position]):
                raise SuffixV21FatalError(
                    "fixed_embedding_changed", "final_invariant", position
                )
    ids = torch.tensor(
        [int(final_tokens[position]) for position in positions],
        dtype=torch.long,
        device=entry_embedding.device,
    )
    expected = embed_layer(ids).to(
        device=entry_embedding.device, dtype=entry_embedding.dtype
    ).detach()
    actual = final_embedding[:, positions, :][0]
    if not torch.equal(actual, expected):
        raise SuffixV21FatalError(
            "final_embedding_token_mismatch", "final_invariant"
        )
    if not torch.isfinite(final_embedding).all():
        raise SuffixV21FatalError(
            "nonfinite_final_embedding", "final_invariant"
        )


def _formal_method(model, embed_layer, targets, config, state):
    entry_embedding = state["entry_embedding"]
    entry_tokens = state["entry_tokens"]
    mask = state["attention_mask"]
    positions = state["positions"]
    layers = state["layers"]
    weights = state["weights"]
    legal_vocab = state["legal_vocab"]
    work, global_summary = _global_optimize(
        model,
        entry_embedding,
        targets,
        mask,
        positions,
        layers,
        weights,
        legal_vocab,
        config,
    )
    committed_tokens = list(entry_tokens)
    committed_positions = []
    events = []
    anomaly_reasons = []
    for position in positions:
        old_mixed = _mixed_context(
            work, committed_tokens, committed_positions, embed_layer
        )
        old_metrics = _continuous_metrics(
            model,
            old_mixed,
            position,
            targets,
            mask,
            layers,
            weights,
            legal_vocab,
            config,
        )
        repair = {
            "triggered": False,
            "trial_safe": None,
            "accepted": False,
            "fallback_reason": None,
        }
        if old_metrics[2] > float(config.tau_J):
            work, repair = _try_vector_repair(
                model,
                work,
                committed_tokens,
                committed_positions,
                position,
                embed_layer,
                targets,
                mask,
                layers,
                weights,
                legal_vocab,
                config,
                old_metrics,
            )
        frozen_mixed = _mixed_context(
            work, committed_tokens, committed_positions, embed_layer
        )
        baseline = _continuous_metrics(
            model,
            frozen_mixed,
            position,
            targets,
            mask,
            layers,
            weights,
            legal_vocab,
            config,
        )
        unresolved_continuous = baseline[2] > float(config.tau_J)
        initial_mode = "expanded" if unresolved_continuous else "normal"
        sources = _candidate_sources(
            model,
            work,
            position,
            committed_tokens,
            legal_vocab,
            mask,
            config,
        )
        initial_pool = _pool_from_sources(sources, initial_mode, legal_vocab)
        initial_scored, initial_dropped = _score_candidates(
            model,
            work,
            committed_tokens,
            committed_positions,
            position,
            initial_pool,
            embed_layer,
            targets,
            mask,
            layers,
            weights,
            config,
        )
        selected_first = initial_scored[0]
        r_first = max(0.0, selected_first["hidden_error"] - baseline[0])
        expanded_attempt = initial_mode == "expanded"
        normal_token_ids = {
            entry["token_id"]
            for entry in _pool_from_sources(sources, "normal", legal_vocab)
        }
        expanded_pool = None
        expanded_added_count = None
        expanded_scored = []
        expanded_dropped = []
        selected_final = selected_first
        if initial_mode == "expanded":
            expanded_added_count = len(
                {entry["token_id"] for entry in initial_pool} - normal_token_ids
            )
        elif r_first > float(config.tau_r):
            expanded_attempt = True
            expanded_pool = _pool_from_sources(sources, "expanded", legal_vocab)
            expanded_added_count = len(
                {entry["token_id"] for entry in expanded_pool} - normal_token_ids
            )
            expanded_scored, expanded_dropped = _score_candidates(
                model,
                work,
                committed_tokens,
                committed_positions,
                position,
                expanded_pool,
                embed_layer,
                targets,
                mask,
                layers,
                weights,
                config,
            )
            proposed = expanded_scored[0]
            if (
                proposed["hidden_error"]
                < selected_first["hidden_error"] - float(config.epsilon_d)
            ):
                selected_final = proposed
        r_final = max(0.0, selected_final["hidden_error"] - baseline[0])
        unresolved_local = r_final > float(config.tau_r)
        committed_tokens[position] = int(selected_final["token_id"])
        committed_positions.append(position)
        if unresolved_continuous:
            anomaly_reasons.append(
                {"position": int(position), "reason": "unresolved_continuous_quality"}
            )
        if unresolved_local:
            anomaly_reasons.append(
                {"position": int(position), "reason": "unresolved_local_degradation"}
            )
        events.append(_position_event(
            position,
            baseline,
            repair,
            initial_mode,
            initial_scored,
            initial_dropped,
            selected_first,
            r_first,
            expanded_attempt,
            expanded_added_count,
            expanded_scored,
            expanded_dropped,
            selected_final,
            r_final,
            unresolved_continuous,
            unresolved_local,
        ))
    final_embedding = entry_embedding.detach().clone()
    final_ids = torch.tensor(
        [committed_tokens[position] for position in positions],
        dtype=torch.long,
        device=entry_embedding.device,
    )
    final_values = embed_layer(final_ids).to(
        device=entry_embedding.device, dtype=entry_embedding.dtype
    ).detach()
    position_index = torch.tensor(
        positions, dtype=torch.long, device=entry_embedding.device
    )
    final_embedding = final_embedding.index_copy(
        1, position_index, final_values.unsqueeze(0)
    ).detach()
    _assert_final_invariants(
        entry_embedding,
        entry_tokens,
        final_embedding,
        committed_tokens,
        positions,
        embed_layer,
    )
    return final_embedding, list(committed_tokens), global_summary, events, anomaly_reasons


def _accuracy(tokens, targets, positions):
    if not positions:
        return 0.0
    return sum(
        int(tokens[position]) == int(targets[position])
        for position in positions
    ) / len(positions)


def _run_offline_diagnostics(
        result, entry_tokens, final_tokens, positions, total_input_ids):
    diagnostics = {}
    diagnostics_failed = False
    diagnostics_error = None
    try:
        target_tensor = total_input_ids.detach()
        if target_tensor.ndim != 2 or int(target_tensor.shape[0]) != 1:
            raise ValueError("target token tensor must have shape [1, sequence]")
        target_tokens = [
            int(value) for value in target_tensor[0].cpu().tolist()
        ]
        if len(target_tokens) != len(entry_tokens):
            raise ValueError("target token length mismatch")
        pre_accuracy = _accuracy(entry_tokens, target_tokens, positions)
        post_accuracy = _accuracy(final_tokens, target_tokens, positions)
        diagnostics = {
            "pre_accuracy": pre_accuracy,
            "post_accuracy": post_accuracy,
            "final_accuracy": post_accuracy,
            "evaluated_position_count": len(positions),
        }
        result["pre_acc"] = pre_accuracy
        result["post_acc"] = post_accuracy
        result["final_accuracy"] = post_accuracy
    except Exception as error:
        diagnostics_failed = True
        diagnostics_error = "{}:{}".format(type(error).__name__, str(error))
    result["diagnostics"] = diagnostics
    result["diagnostics_failed"] = diagnostics_failed
    result["diagnostics_error"] = diagnostics_error


def _decode_final(tokenizer, tokens, eval_start_pos):
    try:
        return tokenizer.decode([int(value) for value in tokens[eval_start_pos:]])
    except Exception as error:
        raise SuffixV21FatalError(
            "final_text_decode_failed:{}".format(type(error).__name__),
            "finalization",
        ) from error


def _success_result(
        config, state, final_tokens, global_summary, events,
        anomaly_reasons, tokenizer):
    return {
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
        "fatal_failure": False,
        "reason": "completed_without_hard_failure",
        "pre_acc": None,
        "post_acc": None,
        "final_accuracy": None,
        "final_tokens": list(final_tokens),
        "final_text": _decode_final(
            tokenizer, final_tokens, state["eval_start_pos"]
        ),
        "effective_layers": list(state["layers"]),
        "effective_layer_weights": list(state["weights"]),
        "filtered_layers": list(state["filtered_layers"]),
        "stage1": {
            "entry_embedding_snapshot_frozen": True,
            "entry_token_snapshot_frozen": True,
        },
        "reoptimization": {
            "global_optimization": global_summary,
            "causal_positions": events,
        },
        "global_optimization": global_summary,
        "events": events,
        "anomaly_reasons": anomaly_reasons,
        "triggered": any(
            event["vector_repair"]["triggered"]
            or event["expanded_attempt"]
            for event in events
        ),
        "accuracy_diagnostics_enabled": bool(
            config.accuracy_diagnostics_enabled
        ),
        "diagnostics": {},
        "diagnostics_failed": False,
        "diagnostics_error": None,
    }


def _rollback_result(
        config, entry_tokens, tokenizer, eval_start_pos, fatal):
    try:
        final_text = tokenizer.decode(
            [int(value) for value in entry_tokens[int(eval_start_pos):]]
        )
    except Exception:
        final_text = None
    return {
        "name": METHOD_NAME,
        "method": METHOD_NAME,
        "version": VERSION,
        "enabled": bool(config.enabled),
        "skipped": False,
        "accepted": False,
        "rollback": True,
        "rollback_reason": fatal.reason,
        "failed_stage": fatal.stage,
        "failed_position": fatal.position,
        "fatal_failure": True,
        "reason": fatal.reason,
        "pre_acc": None,
        "post_acc": None,
        "final_accuracy": None,
        "final_tokens": list(entry_tokens),
        "final_text": final_text,
        "events": [],
        "anomaly_reasons": [],
        "triggered": False,
        "accuracy_diagnostics_enabled": bool(
            config.accuracy_diagnostics_enabled
        ),
        "diagnostics": {},
        "diagnostics_failed": False,
        "diagnostics_error": None,
    }


def run_suffix_reoptimization_v2_1(
        model, embed_layer, entry_embedding_snapshot, entry_token_snapshot,
        target_hidden_states, attention_mask, layer_id, model_layer_count,
        tokenizer, config, total_input_ids=None, eval_start_pos=0,
        log_file=None):
    """Run v2.1 and return ``(token_consistent_embedding, result)``.

    ``total_input_ids`` belongs exclusively to the optional post-formal
    diagnostics domain.  It is not inspected when diagnostics are disabled.
    """
    del log_file
    try:
        entry_embedding = torch.as_tensor(
            entry_embedding_snapshot
        ).detach().clone()
    except Exception as error:
        raise SuffixV21FatalError(
            "invalid_entry_embedding_snapshot:{}".format(type(error).__name__),
            "preflight",
        ) from error
    try:
        entry_tokens = _snapshot_tokens(entry_token_snapshot)
    except SuffixV21FatalError:
        raise
    if not config.enabled:
        fatal = SuffixV21FatalError("suffix_v2_1_disabled", "preflight")
        return entry_embedding, _rollback_result(
            config, entry_tokens, tokenizer, eval_start_pos, fatal
        )
    state = None
    training_state = bool(getattr(model, "training", False))
    try:
        if not hasattr(model, "eval") or not hasattr(model, "train"):
            raise SuffixV21FatalError("causal_lm_unavailable", "preflight")
        model.eval()
        state = _preflight(
            model,
            embed_layer,
            entry_embedding,
            entry_tokens,
            target_hidden_states,
            attention_mask,
            layer_id,
            model_layer_count,
            tokenizer,
            config,
            eval_start_pos,
        )
        (
            formal_embedding,
            formal_tokens,
            global_summary,
            events,
            anomaly_reasons,
        ) = _formal_method(
            model,
            embed_layer,
            target_hidden_states,
            config,
            state,
        )
        result = _success_result(
            config,
            state,
            formal_tokens,
            global_summary,
            events,
            anomaly_reasons,
            tokenizer,
        )
    except Exception as error:
        fatal = error if isinstance(error, SuffixV21FatalError) else (
            SuffixV21FatalError(
                "fatal_runtime_failure:{}".format(type(error).__name__),
                "runtime",
            )
        )
        formal_embedding = entry_embedding.detach().clone()
        formal_tokens = list(entry_tokens)
        result = _rollback_result(
            config, entry_tokens, tokenizer, eval_start_pos, fatal
        )
    finally:
        if hasattr(model, "train"):
            model.train(training_state)

    frozen_embedding = formal_embedding.detach().clone()
    frozen_tokens = list(formal_tokens)
    frozen_acceptance = (
        bool(result["accepted"]), bool(result["rollback"]), result["reason"]
    )
    if config.accuracy_diagnostics_enabled:
        diagnostic_positions = (
            list(state["positions"])
            if state is not None
            else list(range(int(eval_start_pos), len(entry_tokens)))
        )
        _run_offline_diagnostics(
            result,
            entry_tokens,
            frozen_tokens,
            diagnostic_positions,
            total_input_ids,
        )
    if (
        frozen_tokens != list(result["final_tokens"])
        or frozen_acceptance
        != (bool(result["accepted"]), bool(result["rollback"]), result["reason"])
    ):
        raise RuntimeError("offline diagnostics changed the formal result")
    return frozen_embedding, result
