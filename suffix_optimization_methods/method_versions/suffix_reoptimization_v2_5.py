"""v2.5: complete R optimization, freeze, then low-fit bounded backtracking.

Continuous Stage-1/R kernels are copied from v2.2.2(2). Candidate scores are
local to one position and prefix; private labels enter only offline evaluation.
"""
from dataclasses import dataclass, field
import copy
from contextlib import contextmanager
import math
import numbers
import hashlib
import json
import time
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

METHOD_NAME = "suffix_reoptimization_v2.5"
VERSION = "v2.5"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192



@dataclass
class SuffixReoptimizationV25Config:
    enabled: bool = False
    log_enabled: bool = True
    reoptimization_enabled: bool = True
    expansion_enabled: bool = True
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

    trajectory_max_new: int = 5
    seed_top_k: int = 5
    case_quota: int = 3
    space_quota: int = 3
    prefix_quota: int = 1
    suffix_quota: int = 3
    stage1_sampling_steps: tuple = (500, 600, 700, 800, 900)
    r_sampling_steps: tuple = (30, 40)
    candidate_batch_size: int = 35

    backtracking_enabled: bool = True
    backtracking_gap_threshold: float = 1e-3
    backtracking_fit_threshold: float = None
    backtracking_first_fit_threshold: float = None
    backtracking_acceptance_epsilon: float = None
    backtracking_consistency_tolerance: float = None
    backtracking_max_distance: int = 4
    backtracking_alternatives: int = 4
    backtracking_max_trials: int = 16
    backtracking_lookahead: int = 2
    backtracking_max_events: int = 2
    backtracking_forward_budget: int = 640
    backtracking_prune: bool = True
    backtracking_calibration: dict = field(default_factory=dict)

    def __post_init__(self):
        _validate_backtracking_config(self)
        for name, ceiling in (("trajectory_max_new", 5), ("seed_top_k", 5),
                              ("case_quota", 3), ("space_quota", 3),
                              ("prefix_quota", 1), ("suffix_quota", 3)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError("suffix_v2_5_{} must be in [0, {}]".format(name, ceiling))
        if type(self.candidate_batch_size) is not int or self.candidate_batch_size <= 0:
            raise ValueError("suffix_v2_5_candidate_batch_size must be positive")
        for name, expected in (("stage1_sampling_steps", (500, 600, 700, 800, 900)),
                               ("r_sampling_steps", (30, 40))):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or any(type(s) is not int for s in value):
                raise TypeError("suffix_v2_5_{} must be integer steps".format(name))
            if tuple(value) != expected:
                raise ValueError("v2.5 uses fixed absolute sampling steps: {}".format(expected))
            setattr(self, name, tuple(value))
        for name in ("enabled", "log_enabled", "reoptimization_enabled", "expansion_enabled", "filter_nonascii"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_5_{} must be boolean".format(name))
        for name in (
            "max_attempts",
            "max_attempts_per_position",
            "steps",
            "range_top_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_5_{} must be an integer".format(name))
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
                raise TypeError("suffix_v2_5_{} must be numeric".format(name))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("suffix_v2_5_{} must be finite".format(name))
            setattr(self, name, value)
        if self.max_attempts < 0:
            raise ValueError("suffix_v2_5_max_attempts must be non-negative")
        if self.max_attempts_per_position <= 0:
            raise ValueError(
                "suffix_v2_5_max_attempts_per_position must be positive"
            )
        if self.steps <= 0:
            raise ValueError("suffix_v2_5_steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("suffix_v2_5_lr must be positive")
        if not 0.0 <= self.hidden_weight_decay <= 1.0:
            raise ValueError(
                "suffix_v2_5_hidden_weight_decay must be in [0, 1]"
            )
        if self.hidden_weight_floor < 0.0:
            raise ValueError(
                "suffix_v2_5_hidden_weight_floor must be non-negative"
            )
        if self.prox_weight < 0.0 or self.range_weight < 0.0:
            raise ValueError(
                "suffix_v2_5 regularization weights must be non-negative"
            )
        if self.range_top_k <= 0:
            raise ValueError("suffix_v2_5_range_top_k must be positive")
        if self.trigger_mode not in {"always", "threshold"}:
            raise ValueError(
                "suffix_v2_5_trigger_mode must be always or threshold"
            )
        if self.hidden_weight_mode != "front_decay":
            raise ValueError(
                "suffix_v2_5_hidden_weight_mode must be front_decay"
            )
        if self.accept_mode != "hidden_loss":
            raise ValueError(
                "suffix_v2_5_accept_mode must be hidden_loss"
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
            hidden_state_list.append(output[0])
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
        layer_id, register_layer_hooks, embed_layer, config, trajectory_callback=None):
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
        if trajectory_callback is not None:
            trajectory_callback(completed_steps, _merge_suffix(
                anchored_base, suffix_start, suffix_param.detach()))

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


def stage_neighbours(embedding, embed_layer, top_k=10, metric="cosine"):
    """Chunked FP32 CPU search, deterministic token-ID order for distance ties."""
    points = embedding.detach().reshape(-1, embedding.shape[-1]).float().cpu()
    if not torch.isfinite(points).all():
        raise ValueError("nonfinite trajectory embedding")
    best_scores = torch.empty((len(points), 0))
    best_ids = torch.empty((len(points), 0), dtype=torch.long)
    with torch.no_grad():
        for begin in range(0, embed_layer.num_embeddings, EMBEDDING_SEARCH_CHUNK_SIZE):
            weight = embed_layer.weight[begin:begin+EMBEDDING_SEARCH_CHUNK_SIZE].detach().float().cpu()
            if metric == "cosine":
                scores = F.normalize(points, dim=-1, eps=1e-8) @ F.normalize(weight, dim=-1, eps=1e-8).T
            elif metric == "L2":
                scores = -torch.cdist(points, weight, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            else:
                raise ValueError("unsupported trajectory metric")
            scores = torch.where(torch.isfinite(scores), scores, -torch.inf)
            ids = torch.arange(begin, begin+len(weight)).expand(len(points), -1)
            scores, ids = torch.cat((best_scores, scores), 1), torch.cat((best_ids, ids), 1)
            by_id = torch.argsort(ids, dim=1, stable=True)
            scores, ids = scores.gather(1, by_id), ids.gather(1, by_id)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
            best_scores, best_ids = scores.gather(1, order), ids.gather(1, order)
    return [[int(t) for t, s in zip(row, values) if math.isfinite(float(s))]
            for row, values in zip(best_ids, best_scores)]


def prefix_fingerprint(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def optimize_stage1(model, initial_embedding, prefix_embedding, target_hidden_state,
                    attention_mask, layer_id, register_layer_hooks, weight_mask,
                    right_range, lr, epoch, alpha, clip=True, optim_method="cosine", trajectory_callback=None):
    """Legacy v2.2.1 Stage-1, excluding its benchmark-only token decoding."""
    variable = initial_embedding.detach().clone().requires_grad_(True)
    begun = time.perf_counter()
    completed = 0
    for step in range(epoch if variable.shape[1] else 0):
        if clip:
            with torch.no_grad():
                variable = torch.clip(variable, -0.2, 0.2)
        variable = variable.requires_grad_(True)
        optimizer = torch.optim.SGD([variable], lr=lr)
        full = torch.cat((prefix_embedding, variable), dim=1) if prefix_embedding is not None else variable
        hidden = _forward_embedding_hidden(model, full, attention_mask, layer_id, register_layer_hooks)
        target = target_hidden_state.to(hidden.device)
        cosine = F.cosine_similarity(hidden.float(), target.float(), dim=-1)
        mse = F.mse_loss(hidden.float(), target.float())
        range_loss = F.relu(full.abs() - right_range).sum()
        cosine_loss = (-cosine.to(weight_mask.device)*weight_mask).sum()
        loss = mse if optim_method == "MSELoss" else cosine_loss + alpha*range_loss
        if optim_method not in ("MSELoss", "cosine"):
            raise ValueError("unsupported Stage-1 objective")
        optimizer.zero_grad()
        loss.backward(inputs=[variable])
        if torch.isnan(cosine).any():
            break
        optimizer.step()
        completed += 1
        if trajectory_callback is not None:
            observed = torch.cat((prefix_embedding, variable), dim=1) if prefix_embedding is not None else variable
            trajectory_callback(completed, observed)
    full = torch.cat((prefix_embedding, variable), dim=1) if prefix_embedding is not None else variable
    return full.detach().clone(), dict(epoch=completed-1, completed_steps=completed,
                                       elapsed_seconds=time.perf_counter()-begun)


def rng_state():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def finite_json(value):
    """Diagnostic nonfinite values must not escape into non-standard JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    return value


def config_from_mapping(values):
    prefixes = {"enabled": "suffix_reoptimization_v2_5",
                "log_enabled": "suffix_reoptimization_v2_5_log"}
    kwargs = {}
    for field in SuffixReoptimizationV25Config.__dataclass_fields__:
        key = prefixes.get(field, "suffix_v2_5_" + field)
        if key not in values:
            if field in ("reoptimization_enabled", "expansion_enabled"):
                kwargs[field] = True  # Preserve existing v2.5 configs captured before component switches.
                continue
            raise ValueError("missing explicit v2.5 configuration: " + key)
        kwargs[field] = values[key]
    return SuffixReoptimizationV25Config(**kwargs)


class DiscretizationContractError(ValueError):
    """A broken snapshot/scoring contract fails the run, rather than shrinking B."""


def raise_if_fatal(error):
    message = str(error).lower()
    if isinstance(error, (DiscretizationContractError, MemoryError, torch.cuda.OutOfMemoryError)) or any(
        word in message for word in ("out of memory", "device-side assert", "illegal memory", "cuda error")
    ):
        raise error


def validate_inputs(model, embedding, target, mask, prefix, start, tokenizer):
    if embedding.ndim != 3 or embedding.shape[0] != 1 or target.shape != embedding.shape:
        raise DiscretizationContractError("expected matching [1,T,D] embeddings and target")
    if not torch.isfinite(target).all() or not torch.isfinite(embedding).all():
        raise DiscretizationContractError("nonfinite target/entry embedding")
    if mask is not None and (tuple(mask.shape) != tuple(embedding.shape[:2]) or not bool((mask == 1).all())):
        raise DiscretizationContractError("only unpadded, fully valid single sequences are supported")
    if len(prefix) != start or not 0 <= start <= embedding.shape[1] or start > 1:
        raise DiscretizationContractError("invalid public prefix length")
    if prefix and prefix != [getattr(tokenizer, "bos_token_id", None)]:
        raise DiscretizationContractError("only the tokenizer's public BOS may be fixed")
    if model.training:
        raise DiscretizationContractError("v2.5 requires model.eval()")


def _token_text(tokenizer, token):
    return tokenizer.decode([token], clean_up_tokenization_spaces=False)


def _token_exclusion(token, tokenizer, vocab_size, filter_nonascii):
    if type(token) is not int or not 0 <= token < vocab_size:
        return "invalid_id"
    if token in tokenizer.all_special_ids:
        return "special_token"
    if filter_nonascii and not _token_text(tokenizer, token).isascii():
        return "nonascii"
    return None


def trajectory_identity(model, embed_layer, tokenizer):
    """Persist vocabulary/model identity once per sample and reuse it across R."""
    weight_hash = hashlib.sha256()
    for begin in range(0, embed_layer.num_embeddings, EMBEDDING_SEARCH_CHUNK_SIZE):
        chunk = embed_layer.weight[begin:begin+EMBEDDING_SEARCH_CHUNK_SIZE].detach().cpu().contiguous()
        weight_hash.update(chunk.view(torch.uint8).numpy().tobytes())
    if hasattr(tokenizer, "get_vocab"):
        vocabulary = sorted(tokenizer.get_vocab().items())
    else:
        vocabulary = [_token_text(tokenizer, i) for i in range(embed_layer.num_embeddings)]
    config = getattr(model, "config", None)
    return dict(model_class=type(model).__module__+"."+type(model).__qualname__,
                model_name=getattr(config, "_name_or_path", None),
                model_type=getattr(config, "model_type", None),
                vocabulary_sha256=hashlib.sha256(json.dumps(vocabulary, ensure_ascii=False).encode()).hexdigest(),
                embedding_sha256=weight_hash.hexdigest(), embedding_shape=list(embed_layer.weight.shape),
                embedding_dtype=str(embed_layer.weight.dtype), special_ids=sorted(tokenizer.all_special_ids))


class DiscretizationTrajectoryCollector:
    """Completed-step Top1 samples, with absolute position and entrance identity."""
    def __init__(self, steps, start, embed_layer, metric, call_id, prefix, identity, sampling_steps):
        self.steps, self.start = int(steps), int(start)
        if len(prefix) != self.start:
            raise DiscretizationContractError("trajectory entrance prefix is misaligned")
        self.embed_layer, self.metric = embed_layer, metric
        self.call_id, self.prefix = call_id, prefix_fingerprint(prefix)
        self.identity, self.points = identity, tuple(sampling_steps)
        self.snapshots, self.elapsed_seconds, self.retrievals = {}, 0., 0
        self.last_completed_step = 0

    def __call__(self, step, embedding):
        self.last_completed_step = int(step)
        if step not in self.points or step >= self.steps:
            return
        begun = time.perf_counter()
        try:
            rows = stage_neighbours(embedding[:, self.start:], self.embed_layer, top_k=1, metric=self.metric)
        except ValueError as error:
            raise DiscretizationContractError(str(error)) from error
        if any(len(row) != 1 for row in rows):
            raise DiscretizationContractError("trajectory Top1 is unavailable")
        self.snapshots[step] = rows
        self.retrievals += len(rows)
        self.elapsed_seconds += time.perf_counter()-begun

    def finalize(self, completed_steps, sequence_length):
        self.completed_steps = int(completed_steps)
        if not 0 <= self.completed_steps <= self.steps or not self.start <= sequence_length:
            raise DiscretizationContractError("invalid completed trajectory length")
        length = sequence_length-self.start
        if any(len(rows) != length for rows in self.snapshots.values()):
            raise DiscretizationContractError("trajectory position mapping changed")
        kept = {step: rows for step, rows in self.snapshots.items() if step < completed_steps}
        return {self.start+i: dict(call_id=self.call_id, position=self.start+i,
                    source_start=self.start, prefix_fingerprint=self.prefix, identity=self.identity,
                    metric=self.metric, planned_steps=self.steps, completed_steps=self.completed_steps,
                    planned_sampling_steps=list(self.points),
                    stages=[dict(step=step, token_ids=list(rows[i])) for step, rows in sorted(kept.items())])
                for i in range(length)}

    def metadata(self):
        completed = getattr(self, "completed_steps", self.last_completed_step)
        return dict(call_id=self.call_id, source_start=self.start, planned_steps=self.steps,
                    completed_steps=completed, planned_sampling_steps=list(self.points), top_k=1,
                    captured_steps=sorted(self.snapshots),
                    retained_steps=sorted(step for step in self.snapshots if step < completed),
                    retrievals=self.retrievals, elapsed_seconds=self.elapsed_seconds,
                    prefix_fingerprint=self.prefix, identity=self.identity, metric=self.metric)


def validate_trajectory(trajectory, *, position, start, call_id, prefix, identity, sampling_steps, metric):
    if trajectory is None:
        return
    expected = dict(position=position, source_start=start, call_id=call_id,
                    prefix_fingerprint=prefix_fingerprint(prefix), identity=identity, metric=metric)
    if any(trajectory.get(key) != value for key, value in expected.items()):
        raise DiscretizationContractError("trajectory identity/position/entrance mismatch")
    if tuple(trajectory.get("planned_sampling_steps", ())) != tuple(sampling_steps):
        raise DiscretizationContractError("trajectory has different absolute sampling steps")
    completed, planned = trajectory.get("completed_steps"), trajectory.get("planned_steps")
    if type(completed) is not int or type(planned) is not int or not 0 <= completed <= planned:
        raise DiscretizationContractError("invalid trajectory completed_steps")
    seen = set()
    for row in trajectory.get("stages", []):
        step = row.get("step")
        if type(step) is not int or step not in sampling_steps or step >= completed or step in seen:
            raise DiscretizationContractError("trajectory contains an invalid/terminal/duplicate step")
        if not isinstance(row.get("token_ids"), (list, tuple)):
            raise DiscretizationContractError("trajectory token_ids must be a sequence")
        seen.add(step)


def build_discretization_base(top_ids, ppl_ids, tokenizer, vocab_size, filter_nonascii=True,
                              trajectory=None, max_new=5, expansion_enabled=True):
    """Keep B0 intact; inject Stage1 Top1 only, retaining R samples for diagnostics."""
    rows, by_id, exclusions = [], {}, []
    for source, tokens in (("embedding", top_ids), ("ppl", ppl_ids)):
        for rank, token in enumerate(tokens):
            if type(token) is not int or not 0 <= token < vocab_size:
                raise DiscretizationContractError("invalid original candidate ID")
            label = dict(source=source, rank=rank+1)
            if token in by_id:
                by_id[token]["sources"].append(label)
            else:
                row = dict(token_id=token, order=len(rows), quota_source="original", sources=[label])
                rows.append(row)
                by_id[token] = row
    if not rows:
        raise DiscretizationContractError("empty original candidate table")
    original_ids = list(by_id)
    count = 0
    stages = (trajectory or {}).get("stages", []) if expansion_enabled else []
    for stage in sorted(stages, key=lambda row: row["step"]):
        tokens = stage["token_ids"]
        if not tokens:
            exclusions.append(dict(source="trajectory", step=stage["step"], reason="missing_top1"))
            continue
        token = tokens[0]  # Never backfill from Top2, including legacy Top10 metadata.
        label = dict(source="trajectory", step=stage["step"], call_id=trajectory["call_id"], rank=1)
        if trajectory["call_id"].startswith("R_"):
            exclusions.append(dict(token_id=token, reason="r_trajectory_candidates_disabled", **label))
            continue
        reason = _token_exclusion(token, tokenizer, vocab_size, filter_nonascii)
        if not reason and token in by_id:
            by_id[token]["sources"].append(label)
            reason = "duplicate_original" if token in original_ids else "duplicate_trajectory"
        if not reason and count >= max_new:
            reason = "trajectory_quota_exhausted"
        if reason:
            exclusions.append(dict(token_id=token, reason=reason, **label))
            continue
        row = dict(token_id=token, order=len(rows), quota_source="trajectory", sources=[label])
        rows.append(row)
        by_id[token] = row
        count += 1
    return dict(candidates=rows, original_candidate_ids=original_ids,
                trajectory_candidate_ids=[row["token_id"] for row in rows if row["quota_source"] == "trajectory"],
                trajectory=trajectory, exclusions=exclusions,
                trajectory_quota_gap=max_new-count if expansion_enabled else 0)


def _variant_texts(category, text):
    if category == "case":
        leading = len(text)-len(text.lstrip())
        whitespace, body = text[:leading], text[leading:]
        yield from (whitespace+body.lower(), whitespace+body.upper(), whitespace+body.capitalize())
    elif category == "space":
        yield text[1:] if text.startswith(" ") else " "+text
    else:
        for size in range(len(text)-1, 0, -1):
            yield text[:size] if category == "prefix" else text[-size:]


def build_discretization_variants(base_rows, tokenizer, vocab_size, filter_nonascii, config):
    seeds, excluded, variants, counts = [], [], [], Counter()
    quotas = {category: getattr(config, category+"_quota") if config.expansion_enabled else 0
              for category in ("case", "space", "prefix", "suffix")}
    if not config.expansion_enabled:
        return dict(seeds=[], candidates=[], exclusions=[], counts={}, quotas=quotas,
                    quota_gaps=dict(quotas))
    # Python's stable sorting preserves B's first-occurrence order on score ties.
    ranked = sorted(base_rows, key=lambda row: -row["score"] if math.isfinite(row["score"]) else math.inf)
    for rank, row in enumerate(ranked):
        token, score = row["token_id"], row["score"]
        reason = "nonfinite_seed_score" if not math.isfinite(score) else _token_exclusion(
            token, tokenizer, vocab_size, filter_nonascii)
        text = None if reason else _token_text(tokenizer, token)
        if not reason and not text:
            reason = "empty_seed_text"
        if not reason and list(tokenizer.encode(text, add_special_tokens=False)) != [token]:
            reason = "seed_roundtrip"
        if reason:
            excluded.append(dict(token_id=token, reason=reason, source="seed", base_rank=rank+1))
        elif len(seeds) < config.seed_top_k:
            seeds.append(dict(token_id=token, text=text, score=score, base_rank=rank+1, seed_rank=len(seeds)+1))
    by_id = {row["token_id"]: row for row in base_rows}
    for category, quota in quotas.items():
        for seed in seeds:
            if counts[category] >= quota:
                break
            for text in _variant_texts(category, seed["text"]):
                label = dict(source=category, seed_token_id=seed["token_id"], seed_rank=seed["seed_rank"], text=text)
                counts[category+"_attempted"] += 1
                reason, token = None, None
                if not text or text == seed["text"]:
                    reason = "empty_or_unchanged_text"
                else:
                    ids = list(tokenizer.encode(text, add_special_tokens=False))
                    if len(ids) != 1:
                        reason = "not_single_token"
                    else:
                        token = int(ids[0])
                        reason = _token_exclusion(token, tokenizer, vocab_size, filter_nonascii)
                        if not reason and _token_text(tokenizer, token) != text:
                            reason = "variant_roundtrip"
                        if not reason and token in by_id:
                            by_id[token]["sources"].append(label)
                            reason = "duplicate_base" if token in {r["token_id"] for r in base_rows} else "duplicate_variant"
                if reason:
                    excluded.append(dict(token_id=token, reason=reason, **label))
                    counts[category+"_excluded"] += 1
                    continue
                row = dict(token_id=token, order=len(variants), quota_source=category, sources=[label])
                variants.append(row)
                by_id[token] = row
                counts[category] += 1
                break  # This seed contributes at most one new ID in this category.
    return dict(seeds=seeds, candidates=variants, exclusions=excluded, counts=dict(counts), quotas=quotas,
                quota_gaps={category: quota-counts[category] for category, quota in quotas.items()})


def _new_compute_stats():
    return {category: dict(forward_calls=0, candidate_count=0, token_count=0, elapsed_seconds=0.)
            for category in ("candidate", "ppl", "r_optimization", "observation", "stage1")}


@contextmanager
def _preserve_forward_hooks(model):
    hooks = {module: set(module._forward_hooks) for module in model.modules()}
    try:
        yield
    finally:
        for module, original in hooks.items():
            for key in set(module._forward_hooks)-original:
                module._forward_hooks.pop(key, None)
                module._forward_hooks_with_kwargs.pop(key, None)
                module._forward_hooks_always_called.pop(key, None)


@contextmanager
def _measure_model_forwards(model, category, stats):
    """Count actual full forwards, including loss/trigger observations and failures."""
    begun = time.perf_counter()
    def count(module, args, kwargs):
        del module
        value = kwargs.get("inputs_embeds")
        embed = value is not None
        if value is None:
            value = kwargs.get("input_ids", args[0] if args else None)
        stats[category]["forward_calls"] += 1
        if value is not None:
            shape = value.shape[:-1] if embed else value.shape
            stats[category]["token_count"] += math.prod(shape)
    handle = model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        stats[category]["elapsed_seconds"] += time.perf_counter()-begun


class BacktrackingBudgetExceeded(RuntimeError):
    """Raised before a forward that would exceed the sample's extra budget."""


class BacktrackingTrialError(ValueError):
    """An invalid speculative path leaves the valid greedy path intact."""


def backtracking_inference_signature(model, identity, layer_id):
    model_config = getattr(model, "config", None)
    config_values = (model_config.to_dict() if hasattr(model_config, "to_dict") else
                     vars(model_config) if model_config is not None else {})
    parameter = next(model.parameters())
    return dict(model_identity=identity, target_layer=str(layer_id),
        architectural_config_sha256=hashlib.sha256(json.dumps(
            config_values, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest(),
        model_revision=getattr(model_config, "_commit_hash", None),
        dtype=str(parameter.dtype), device_type=parameter.device.type,
        attention_implementation=getattr(model_config, "_attn_implementation", None),
        use_cache=getattr(model_config, "use_cache", None), torch_version=str(torch.__version__),
        tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32),
        tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
        position_protocol="zero_based_public_BOS_unpadded", score="hidden_cosine_fp32",
        window_error="sum_half_clamped_cosine_distance", model_training=bool(model.training))


def build_backtracking_calibration(score_observations, consistency_observations,
                                  sample_ids, inference_signature, quantile=0.05):
    """Freeze independent score quantiles and repeated-window numerical drift.

    Observations contain sample IDs, a has_prefix flag and top1_score, or repeated
    window_errors. This interface never takes reference token IDs or correctness.
    Collect them with backtracking disabled under the intended inference settings.
    """
    ids = list(sample_ids)
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("calibration requires distinct independent sample IDs")
    if isinstance(quantile, bool) or not isinstance(quantile, numbers.Real) or not 0 <= quantile <= 1:
        raise ValueError("invalid calibration quantile")
    if not isinstance(inference_signature, dict) or not inference_signature:
        raise ValueError("calibration requires an inference signature")
    groups = {False: [], True: []}
    for row in score_observations:
        if row.get("sample_id") not in ids or type(row.get("has_prefix")) is not bool:
            raise ValueError("calibration score identity/group mismatch")
        value = row.get("top1_score")
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("calibration scores must be finite cosines")
        groups[row["has_prefix"]].append(float(value))
    if not any(groups.values()):
        raise ValueError("calibration has no score observations")
    drift, scale, repeat_count = 0., 1., 0
    for row in consistency_observations:
        values = row.get("window_errors", [])
        if row.get("sample_id") not in ids or len(values) < 2 or any(
            isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("calibration requires repeated finite window errors")
        drift = max(drift, max(values)-min(values))
        scale = max(scale, max(values))
        repeat_count += len(values)
    if not repeat_count:
        raise ValueError("calibration lacks numerical consistency observations")
    tolerance = drift + float(np.finfo(np.float32).eps)*scale
    payload = dict(schema_version=1, status="frozen", split="independent_calibration",
        fitted_without_reference_tokens=True, sample_ids=ids,
        grouping_strategy="empty_prefix_separate", quantile=float(quantile),
        inference_signature=copy.deepcopy(inference_signature),
        fit_threshold=float(np.quantile(groups[True], quantile)) if groups[True] else None,
        first_fit_threshold=float(np.quantile(groups[False], quantile)) if groups[False] else None,
        acceptance_epsilon=tolerance, consistency_tolerance=tolerance,
        score_counts={"with_prefix": len(groups[True]), "empty_prefix": len(groups[False])},
        numerical_repeat_count=repeat_count, observed_maximum_drift=drift)
    payload["calibration_id"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def collect_backtracking_calibration(sample_runs, model, embed_layer, tokenizer,
                                    layer_id, forward_and_get_last_hidden_state,
                                    repeats=3, quantile=0.05):
    """Calibrate from separate greedy runs and repeated discrete window forwards.

    Each item supplies sample_id, result, target_hidden_state and attention_mask.
    Run the independent samples with backtracking_enabled=False first. No reference
    tokens or offline correctness enter the quantiles or numerical tolerance.
    """
    if type(repeats) is not int or repeats < 2:
        raise ValueError("calibration needs at least two repeated forwards")
    identity = trajectory_identity(model, embed_layer, tokenizer)
    signature = backtracking_inference_signature(model, identity, layer_id)
    observations, consistency, ids = [], [], []
    stats = {"window": dict(forward_calls=0, candidate_count=0, token_count=0, elapsed_seconds=0.)}
    for sample in sample_runs:
        sample_id, result = sample["sample_id"], sample["result"]
        if result.get("version") not in ("v2.3", "v2.5") or result.get("gt_accessed") is not False:
            raise ValueError("calibration requires GT-blind v2.3/v2.5 greedy results")
        if result.get("backtracking", {}).get("event_count", 0):
            raise ValueError("collect calibration with backtracking disabled")
        if result.get("stage1", {}).get("discretization_trajectory", {}).get("identity", identity) != identity:
            raise DiscretizationContractError("calibration run model/vocabulary mismatch")
        ids.append(sample_id)
        target, mask = sample["target_hidden_state"], sample["attention_mask"]
        tokens, start = result["final_tokens"], result["eval_start_pos"]
        embedding = embed_layer(torch.tensor([tokens], device=embed_layer.weight.device)).detach()
        validate_inputs(model, embedding, target, mask, tokens[:start], start, tokenizer)
        for decision in result.get("discretization", {}).get("events", []):
            if not decision.get("selected_in_call", True):
                continue
            if result.get("version") == VERSION and not decision.get("final_retained"):
                continue
            scores = [row["score"] for row in decision["base_candidates"]+decision["variant_candidates"]
                      if row.get("score") is not None and math.isfinite(row["score"])]
            if scores:
                observations.append(dict(sample_id=sample_id, has_prefix=decision["position"] > 0,
                                         top1_score=max(scores)))
        for position in range(start, len(tokens)):
            distance = result.get("backtracking", {}).get("max_distance", 4 if result.get("version") == VERSION else 2)
            lookahead = result.get("backtracking", {}).get("lookahead", 2)
            left, right = max(start, position-distance), min(len(tokens)-1, position+lookahead)
            errors = [_window_error(tokens, left, right, target, model, layer_id,
                forward_and_get_last_hidden_state, mask, stats, None)[0] for _ in range(repeats)]
            consistency.append(dict(sample_id=sample_id, window_errors=errors))
    return build_backtracking_calibration(observations, consistency, ids, signature, quantile)


def _validate_backtracking_config(config):
    for name in ("backtracking_enabled", "backtracking_prune"):
        if type(getattr(config, name)) is not bool:
            raise TypeError("suffix_v2_5_{} must be boolean".format(name))
    for name, ceiling in (("max_distance", 4), ("alternatives", 4), ("max_trials", 16),
                          ("lookahead", 2), ("max_events", 2), ("forward_budget", 640)):
        value = getattr(config, "backtracking_"+name)
        if type(value) is not int or value < 0 or (ceiling is not None and value > ceiling):
            raise ValueError("invalid suffix_v2_5_backtracking_"+name)
    gap = config.backtracking_gap_threshold
    if isinstance(gap, bool) or not isinstance(gap, numbers.Real) or not math.isfinite(gap) or gap < 0:
        raise ValueError("invalid backtracking gap threshold")
    config.backtracking_gap_threshold = float(gap)
    payload = config.backtracking_calibration
    if not isinstance(payload, dict):
        raise TypeError("backtracking_calibration must be a mapping")
    fields = ("fit_threshold", "first_fit_threshold", "acceptance_epsilon", "consistency_tolerance")
    if payload:
        frozen = {key: value for key, value in payload.items() if key != "calibration_id"}
        checksum = hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if (payload.get("schema_version") != 1 or payload.get("status") != "frozen"
                or payload.get("split") != "independent_calibration"
                or payload.get("fitted_without_reference_tokens") is not True
                or payload.get("grouping_strategy") != "empty_prefix_separate"
                or not payload.get("sample_ids") or not payload.get("inference_signature")
                or payload.get("calibration_id") != checksum):
            raise ValueError("invalid or modified frozen backtracking calibration")
        for name in fields:
            if name not in payload:
                raise ValueError("calibration lacks "+name)
            configured = getattr(config, "backtracking_"+name)
            if configured is not None and configured != payload[name]:
                raise ValueError("configured threshold differs from frozen calibration: "+name)
            setattr(config, "backtracking_"+name, payload[name])
        if payload["acceptance_epsilon"] is None or payload["consistency_tolerance"] is None:
            raise ValueError("calibration lacks numerical acceptance tolerance")
    elif any(getattr(config, "backtracking_"+name) is not None for name in fields):
        raise ValueError("backtracking thresholds require independent frozen calibration")
    for name in fields:
        value = getattr(config, "backtracking_"+name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError("nonfinite/non-numeric backtracking "+name)
        if name.endswith("threshold") and not -1 <= value <= 1:
            raise ValueError("fit threshold outside cosine range")
        if not name.endswith("threshold") and value < 0:
            raise ValueError("negative numerical tolerance")
        setattr(config, "backtracking_"+name, float(value))


def _new_backtracking_state(config, model, identity, layer_id, sample_id=None):
    _validate_backtracking_config(config)
    signature = backtracking_inference_signature(model, identity, layer_id)
    payload = config.backtracking_calibration
    if payload and payload["inference_signature"] != signature:
        raise DiscretizationContractError("backtracking calibration inference settings mismatch")
    if payload and sample_id in payload["sample_ids"]:
        raise DiscretizationContractError("evaluation sample belongs to the calibration split")
    return dict(config=config, identity=identity, sample_id=sample_id, signature=signature,
        history=[], clock=0, events=[], events_used=0,
        compute=dict(forward_calls=0, candidate_count=0, token_count=0, elapsed_seconds=0.,
                     by_category={}), calibration_ready=bool(payload))


@contextmanager
def _discrete_forward_scope(model, category, stats, extra_state=None, candidates=False):
    """Count actual batched model calls; guard every extra call before execution."""
    begun = time.perf_counter()
    counters = (stats[category] if extra_state is None else
        extra_state["compute"]["by_category"].setdefault(category,
            dict(forward_calls=0, candidate_count=0, token_count=0, elapsed_seconds=0.)))
    def count(module, args, kwargs):
        del module
        if extra_state is not None and extra_state["compute"]["forward_calls"] >= extra_state["config"].backtracking_forward_budget:
            raise BacktrackingBudgetExceeded("extra model forward budget exhausted")
        value = kwargs.get("inputs_embeds")
        embedded = value is not None
        if value is None:
            value = kwargs.get("input_ids", args[0] if args else None)
        shape = value.shape[:-1] if embedded else value.shape if value is not None else ()
        increments = dict(forward_calls=1, token_count=math.prod(shape) if shape else 0,
                          candidate_count=int(shape[0]) if candidates and len(shape) > 1 else 0)
        for key, amount in increments.items():
            counters[key] += amount
            if extra_state is not None:
                extra_state["compute"][key] += amount
    handle = model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        elapsed = time.perf_counter()-begun
        counters["elapsed_seconds"] += elapsed
        if extra_state is not None:
            extra_state["compute"]["elapsed_seconds"] += elapsed


def _score_candidates(ids, tokens, position, target, model, layer_id, forward, batch_size,
                      stats, attention_mask=None, extra_state=None):
    scores = []
    for begin in range(0, len(ids), batch_size):
        subset = ids[begin:begin+batch_size]
        trials = [list(tokens) for _ in subset]
        for trial, token in zip(trials, subset):
            trial[position] = token
        mask = attention_mask.expand(len(subset), -1) if attention_mask is not None else None
        with torch.no_grad(), _preserve_forward_hooks(model), _discrete_forward_scope(
                model, "candidate", stats, extra_state, candidates=True):
            hidden = forward(model, trials, mask, layer_id=layer_id)
            if hidden.ndim != 3 or hidden.shape[:2] != (len(subset), len(tokens)) or hidden.shape[2] != target.shape[2]:
                raise DiscretizationContractError("candidate hidden shape mismatch")
            similarity = F.cosine_similarity(hidden[:, position, :].float(),
                target[:, position, :].to(hidden.device).float(), dim=-1)
        scores.extend(float(value) for value in similarity.detach().cpu().tolist())
    return scores


def _ranked_rows(decision, tokenizer, vocab_size, filter_nonascii):
    rows, seen = [], set()
    for row in decision["base_candidates"]+decision["variant_candidates"]:
        token, score = row["token_id"], row.get("score")
        if token in seen or score is None or not math.isfinite(score):
            continue
        seen.add(token)
        if _token_exclusion(token, tokenizer, vocab_size, filter_nonascii) is None:
            rows.append(row)
    return sorted(rows, key=lambda row: -row["score"])


def _backtracking_signal(decision, position, config):
    by_id = {}
    for row in decision["base_candidates"]+decision["variant_candidates"]:
        score = row.get("score")
        if score is not None and math.isfinite(score):
            by_id.setdefault(row["token_id"], score)
    scores = sorted(by_id.values(), reverse=True)
    if not scores:
        raise DiscretizationContractError("no finite candidate maximum")
    s1, s2 = scores[0], scores[1] if len(scores) > 1 else None
    fit = config.backtracking_first_fit_threshold if position == 0 else config.backtracking_fit_threshold
    low_fit = fit is not None and s1 < fit
    small_gap = s2 is not None and s1-s2 < config.backtracking_gap_threshold
    return dict(type="low_fit" if low_fit else None,
        s1=s1, s2=s2, gap=s1-s2 if s2 is not None else None,
        low_fit=low_fit, small_gap=small_gap, fit_threshold=fit,
        gap_threshold=config.backtracking_gap_threshold)


def _backtracking_alternatives(decision, tokens, position, state, cutoff, tokenizer, vocab_size):
    config = state["config"]
    if decision["prefix_fingerprint"] != prefix_fingerprint(tokens[:position]):
        raise DiscretizationContractError("backtracking proposal table has a stale prefix")
    original = tokens[position]
    ranked = [row for row in _ranked_rows(decision, tokenizer, vocab_size, config.filter_nonascii)
              if row["token_id"] != original]
    # Alternatives come only from the valid table under this exact prefix.
    # Historical winners never displace the current top four.
    return [dict(token_id=row["token_id"], source="current_ranking", alternative_rank=rank+1)
            for rank, row in enumerate(ranked[:config.backtracking_alternatives])]


def _window_error(tokens, left, right, target, model, layer_id, forward, attention_mask, stats, state):
    with torch.no_grad(), _preserve_forward_hooks(model), _discrete_forward_scope(model, "window", stats, state):
        hidden = forward(model, [tokens], attention_mask, layer_id=layer_id)
        if hidden.shape != target.shape:
            raise DiscretizationContractError("window hidden shape mismatch")
        cosine = F.cosine_similarity(hidden[:, left:right+1].float(),
            target[:, left:right+1].to(hidden.device).float(), dim=-1)
        errors = ((1-cosine.clamp(-1, 1))/2).detach().cpu().reshape(-1).tolist()
    if not all(math.isfinite(value) for value in errors):
        raise BacktrackingTrialError("nonfinite window error")
    return sum(errors), errors


def _rerank_positions(input_embed, current_tokens, rerank_start, fixed_prefix_tokens,
                      tokenizer, model, embed_layer, target_hidden_state, layer_id,
                      invert_method, filter_nonascii, add_perplexity, top_k_ppl, top_k_cos,
                      eval_start_pos, embedding_top_indices, select_candidate_from_top_indices,
                      get_perplexity, forward_and_get_last_hidden_state, *, config,
                      stage_candidates=None, discretization_events=None, call_id="stage1",
                      identity=None, compute_stats=None, attention_mask=None, backtracking_state=None,
                      trajectory_contracts=None):
    # Continuous input_embed is read-only throughout this scan and all its branches.
    n = int(input_embed.shape[1])
    tokens = [0]*n if current_tokens is None else list(current_tokens)
    if len(tokens) != n or len(fixed_prefix_tokens) != eval_start_pos:
        raise DiscretizationContractError("discretization token/prefix alignment mismatch")
    tokens[:eval_start_pos] = fixed_prefix_tokens
    start = max(eval_start_pos, rerank_start)
    entry_prefix = list(tokens[:start])
    identity = identity or trajectory_identity(model, embed_layer, tokenizer)
    stats = compute_stats if compute_stats is not None else _new_compute_stats()
    state = backtracking_state or _new_backtracking_state(config, model, identity, layer_id)
    decisions = discretization_events if discretization_events is not None else []
    sampling = config.stage1_sampling_steps if call_id == "stage1" else config.r_sampling_steps
    initial_top_ids, active = {}, {}
    for position in range(start, n):
        selected, ids = _candidate_token_ids(input_embed[0, position], embed_layer, top_k_cos,
            invert_method, tokenizer, filter_nonascii, embedding_top_indices, select_candidate_from_top_indices)
        initial_top_ids[position] = ids or [tokens[position]]
        tokens[position] = selected

    def evaluate(path, position, *, extra=False, path_kind="normal", event_id=None, trial_index=None, forced=None):
        begun = time.perf_counter()
        extra_state = state if extra else None
        candidate_counters = stats["candidate"] if extra_state is None else state["compute"]["by_category"].get("candidate", {})
        candidate_calls_before = candidate_counters.get("forward_calls", 0)
        trajectory = (stage_candidates or {}).get(position)
        if call_id == "final_discretization" and trajectory is not None:
            source = (trajectory_contracts or {}).get(trajectory.get("call_id"))
            if source is None:
                raise DiscretizationContractError("final trajectory has no accepted optimization source")
            validate_trajectory(trajectory, position=position, start=source["start"],
                call_id=source["call_id"], prefix=source["prefix"], identity=identity,
                sampling_steps=source["sampling_steps"], metric=invert_method)
        else:
            validate_trajectory(trajectory, position=position, start=start, call_id=call_id,
                prefix=entry_prefix, identity=identity, sampling_steps=sampling, metric=invert_method)
        ppl_ids = []
        if add_perplexity and position > 0 and top_k_ppl > 0:
            with torch.no_grad(), _preserve_forward_hooks(model), _discrete_forward_scope(model, "ppl", stats, extra_state):
                _, ids = get_perplexity(path[:position], model, layer_id=layer_id, top_k=top_k_ppl)
            ppl_ids = [int(token) for token in ids]
        base = build_discretization_base(initial_top_ids[position], ppl_ids, tokenizer,
            embed_layer.num_embeddings, config.filter_nonascii, trajectory, config.trajectory_max_new,
            expansion_enabled=config.expansion_enabled)
        base_rows = base["candidates"]
        def score(rows):
            values = _score_candidates([row["token_id"] for row in rows], path, position,
                target_hidden_state, model, layer_id, forward_and_get_last_hidden_state,
                config.candidate_batch_size, stats, attention_mask, extra_state)
            for row, value in zip(rows, values):
                row.update(score=value, validly_scored=math.isfinite(value))
                if not math.isfinite(value) and row["quota_source"] == "original":
                    error_type = BacktrackingTrialError if extra else DiscretizationContractError
                    raise error_type("nonfinite original candidate score")
            return values
        base_scores = score(base_rows)
        eligible = [row for row in base_rows if row["validly_scored"]]
        if not eligible:
            raise DiscretizationContractError("no finite candidate maximum")
        best_base = max(eligible, key=lambda row: row["score"])
        variants = build_discretization_variants(base_rows, tokenizer, embed_layer.num_embeddings, config.filter_nonascii, config)
        variant_rows = variants["candidates"]
        variant_scores = score(variant_rows)
        valid = [row for row in variant_rows if row["validly_scored"]]
        best = max([best_base]+valid, key=lambda row: row["score"])
        greedy = best
        if forced is not None:
            token = forced["token_id"]
            if _token_exclusion(token, tokenizer, embed_layer.num_embeddings, config.filter_nonascii):
                raise BacktrackingTrialError("illegal forced candidate")
            matching = [row for row in base_rows+variant_rows if row["token_id"] == token]
            if not matching:
                raise BacktrackingTrialError("forced candidate absent from rebuilt current table")
            best = matching[0]
            if not best["validly_scored"]:
                raise BacktrackingTrialError("nonfinite forced candidate")
        prefix_hash = prefix_fingerprint(path[:position])
        path[position] = best["token_id"]
        best_variant = max(valid, key=lambda row: row["score"]) if valid else None
        selected = path_kind != "trial"
        row = dict(phase=("final" if call_id == "final_discretization" else
                          "initial" if call_id == "stage1" else "R"), call_id=call_id, position=position,
            expansion_enabled=config.expansion_enabled,
            prefix_fingerprint=prefix_hash, entrance_prefix_fingerprint=prefix_fingerprint(entry_prefix),
            original_candidate_ids=base["original_candidate_ids"], original_top_ids=initial_top_ids[position], ppl_ids=ppl_ids,
            trajectory_candidate_ids=base["trajectory_candidate_ids"], trajectory=trajectory,
            base_candidates=base_rows, base_scores=base_scores, base_best_token_id=best_base["token_id"],
            base_best_score=best_base["score"], seeds=variants["seeds"],
            variant_candidates=variant_rows, variant_scores=variant_scores,
            variant_best_token_id=best_variant["token_id"] if best_variant else None,
            variant_best_score=best_variant["score"] if best_variant else None,
            selected_token_id=best["token_id"], selected_score=best["score"], greedy_token_id=greedy["token_id"],
            forced_selection=forced is not None, alternative_source=forced["source"] if forced else None,
            tie_kept_base=any(item["score"] == best_base["score"] for item in valid) and best is best_base,
            exclusions=base["exclusions"]+variants["exclusions"]+[
                dict(token_id=item["token_id"], source=item["quota_source"], reason="nonfinite_score")
                for item in base_rows+variant_rows if not item["validly_scored"]],
            quotas=variants["quotas"], quota_gaps=variants["quota_gaps"], trajectory_quota_gap=base["trajectory_quota_gap"],
            generation_counts=variants["counts"], generated_variant_count=len(variant_rows),
            validly_scored_candidate_count=sum(item["validly_scored"] for item in base_rows+variant_rows),
            scored_candidate_count=len(base_rows)+len(variant_rows),
            candidate_forward_calls=(stats["candidate"] if extra_state is None else state["compute"]["by_category"].get("candidate", {})).get("forward_calls", 0)-candidate_calls_before,
            forward_token_count=(len(base_rows)+len(variant_rows))*n, elapsed_seconds=time.perf_counter()-begun,
            path_kind=path_kind, backtracking_event_id=event_id, trial_index=trial_index,
            stage_produced=True, selected_in_call=selected, accepted_into_formal_state=selected and call_id in ("stage1", "final_discretization"),
            retained_in_formal_state=selected and call_id in ("stage1", "final_discretization"), final_retained=False, rolled_back=False,
            anomaly_trigger_suppressed=path_kind != "normal")
        row["candidate_signal"] = _backtracking_signal(row, position, config)
        index = len(decisions)
        decisions.append(row)
        state["history"].append(dict(token_id=greedy["token_id"], position=position, identity=identity,
            sample_id=state["sample_id"], call_id=call_id, clock=state["clock"], path_kind=path_kind))
        state["clock"] += 1
        return index

    def keep(path, selected_indices):
        for position, index in selected_indices.items():
            if position in active and active[position] != index:
                old = decisions[active[position]]
                old.update(selected_in_call=False, retained_in_formal_state=False, accepted_into_formal_state=False)
            active[position] = index
            decisions[index].update(selected_in_call=True, accepted_into_formal_state=call_id in ("stage1", "final_discretization"),
                retained_in_formal_state=call_id in ("stage1", "final_discretization"))
        tokens[:] = path

    position = start
    while position < n:
        index = evaluate(tokens, position)
        active[position] = index
        signal = decisions[index]["candidate_signal"]
        if not config.backtracking_enabled or signal["type"] is None:
            position += 1
            continue
        left = max(start, position-config.backtracking_max_distance)
        right = min(n-1, position+config.backtracking_lookahead)
        splits = list(range(position, left-1, -1))
        if signal["type"] == "low_fit":
            splits = splits[1:] or [position]
        cutoff = state["clock"]
        proposals = [(split, _backtracking_alternatives(decisions[active[split]], tokens, split,
                     state, cutoff, tokenizer, embed_layer.num_embeddings)) for split in splits]
        event = dict(event_id=len(state["events"]), call_id=call_id, phase=decisions[index]["phase"],
            position=position, **signal, window=[left, right], left_boundary=start,
            entrance_tokens=list(tokens), entrance_text=_decode(tokenizer, tokens, eval_start_pos),
            prefix_fingerprint=prefix_fingerprint(tokens[:position]), split_order=splits,
            proposals=[dict(position=split, alternatives=items) for split, items in proposals],
            launched=False, accepted=False, accepted_into_formal_state=False, r_accepted=None,
            final_retained=False, rolled_back=False, trials=[], next_scan_position=position+1,
            E_base=None, E_selected=None, epsilon=config.backtracking_acceptance_epsilon,
            consistency_tolerance=config.backtracking_consistency_tolerance,
            calibration_id=config.backtracking_calibration.get("calibration_id"),
            budget_before=copy.deepcopy(state["compute"]), budget_exhausted=False)
        state["events"].append(event)
        reason = ("calibration_missing" if not state["calibration_ready"] else
            "no_legal_alternative" if not any(items for _, items in proposals) or config.backtracking_max_trials == 0 else
            "event_budget_exhausted" if state["events_used"] >= config.backtracking_max_events else
            "forward_budget_exhausted" if state["compute"]["forward_calls"] >= config.backtracking_forward_budget else None)
        if reason:
            event.update(reason=reason, budget_after=copy.deepcopy(state["compute"]))
            position += 1
            continue
        state["events_used"] += 1
        event["launched"] = True
        baseline, baseline_indices = list(tokens), dict(active)
        # Baseline lookahead is charged as extra work. If interrupted, complete the
        # ordinary greedy path using the normal budget before committing any path.
        exhausted, baseline_failure = False, None
        for next_position in range(position+1, right+1):
            try:
                next_index = evaluate(baseline, next_position, extra=not exhausted and baseline_failure is None,
                    path_kind="baseline_lookahead", event_id=event["event_id"])
            except BacktrackingBudgetExceeded:
                exhausted = True
                next_index = evaluate(baseline, next_position, path_kind="baseline_lookahead", event_id=event["event_id"])
            except Exception as error:
                raise_if_fatal(error)
                baseline_failure = "baseline_failed:"+type(error).__name__
                next_index = evaluate(baseline, next_position, path_kind="baseline_lookahead", event_id=event["event_id"])
            baseline_indices[next_position] = next_index
        keep(baseline, {p: baseline_indices[p] for p in range(position+1, right+1)})
        event.update(base_tokens=list(baseline), base_text=_decode(tokenizer, baseline, eval_start_pos))
        selected_path, selected_indices = baseline, baseline_indices
        try:
            if baseline_failure:
                raise BacktrackingTrialError(baseline_failure)
            if exhausted:
                raise BacktrackingBudgetExceeded("baseline lookahead interrupted")
            base_error, base_errors = _window_error(baseline, left, right, target_hidden_state, model,
                layer_id, forward_and_get_last_hidden_state, attention_mask, stats, state)
            event.update(E_base=base_error, base_errors=base_errors)
            seen_paths = {tuple(baseline[:right+1])}
            finished = False
            for split, alternatives in proposals:
                for alternative in alternatives:
                    if len(event["trials"]) >= config.backtracking_max_trials:
                        finished = True
                        break
                    branch = list(event["entrance_tokens"])
                    indices = {p: baseline_indices[p] for p in range(left, split)}
                    trial = dict(trial_index=len(event["trials"]), split_position=split,
                        retreat_depth=position-split, alternative=alternative,
                        prefix_fingerprint=prefix_fingerprint(branch[:split]),
                        E_trial=None, partial_error=sum(base_errors[:split-left]),
                        accepted=False, completed=False, window=[left, right],
                        active_changes=1, budget_before=copy.deepcopy(state["compute"]))
                    event["trials"].append(trial)
                    try:
                        for next_position in range(split, right+1):
                            next_index = evaluate(branch, next_position, extra=True, path_kind="trial",
                                event_id=event["event_id"], trial_index=trial["trial_index"],
                                forced=alternative if next_position == split else None)
                            indices[next_position] = next_index
                            trial["partial_error"] += (1-max(-1., min(1., decisions[next_index]["selected_score"])))/2
                            if config.backtracking_prune and trial["partial_error"] > base_error+config.backtracking_consistency_tolerance:
                                trial.update(reason="nonnegative_error_lower_bound", pruned_at=next_position)
                                break
                        else:
                            key = tuple(branch[:right+1])
                            if key in seen_paths:
                                trial["reason"] = "duplicate_path"
                            else:
                                seen_paths.add(key)
                                error, errors = _window_error(branch, left, right, target_hidden_state, model,
                                    layer_id, forward_and_get_last_hidden_state, attention_mask, stats, state)
                                trial.update(completed=True, E_trial=error, errors=errors,
                                    improves_baseline=error < base_error-config.backtracking_acceptance_epsilon,
                                    reason="window_error_decreased" if error < base_error-config.backtracking_acceptance_epsilon else "window_error_not_decreased")
                                if trial["improves_baseline"] and (not event["accepted"] or error < event["E_selected"]):
                                    selected_path, selected_indices = list(branch), dict(baseline_indices)
                                    selected_indices.update(indices)
                                    event.update(accepted=True, accepted_trial_index=trial["trial_index"],
                                        split_position=split, retreat_depth=position-split, E_selected=error)
                    except BacktrackingBudgetExceeded:
                        trial["reason"] = "forward_budget_exhausted"
                        raise
                    except Exception as error:
                        raise_if_fatal(error)
                        trial.update(reason="trial_failed:"+type(error).__name__, error=str(error))
                    finally:
                        trial.update(tokens=list(branch), text=_decode(tokenizer, branch, eval_start_pos),
                            decision_indices=[indices[p] for p in sorted(indices) if p >= split],
                            budget_after=copy.deepcopy(state["compute"]))
                    if finished:
                        break
                if finished:
                    break
            event["reason"] = "window_error_decreased" if event["accepted"] else "all_alternatives_rejected"
        except BacktrackingBudgetExceeded:
            selected_path, selected_indices = baseline, baseline_indices
            event.update(reason="forward_budget_exhausted", budget_exhausted=True,
                         accepted=False, accepted_trial_index=None, split_position=None,
                         retreat_depth=None, E_selected=event["E_base"])
        except Exception as error:
            raise_if_fatal(error)
            event.update(reason="baseline_failed:"+type(error).__name__, error=str(error))
        # Commit tokens, text and selected decision tables together. There is no
        # speculative cache: every changed prefix rebuilds PPL and rescores.
        for trial in event["trials"]:
            trial["accepted"] = bool(event["accepted"] and
                                     trial["trial_index"] == event.get("accepted_trial_index"))
        keep(selected_path, {p: selected_indices[p] for p in range(left, right+1)})
        event.update(selected_tokens=list(tokens), selected_text=_decode(tokenizer, tokens, eval_start_pos),
            committed_decision_indices=[active[p] for p in range(left, right+1)],
            accepted_into_formal_state=event["accepted"] and call_id in ("stage1", "final_discretization"),
            E_selected=event["E_selected"] if event["accepted"] else event["E_base"],
            next_scan_position=right+1, budget_after=copy.deepcopy(state["compute"]))
        position = right+1
    tables = [dict(position=p, call_id=call_id, prefix_fingerprint=decisions[active[p]]["prefix_fingerprint"],
        candidate_token_ids=[row["token_id"] for row in decisions[active[p]]["base_candidates"]+decisions[active[p]]["variant_candidates"]],
        candidate_hidden_cosine=[row["score"] for row in decisions[active[p]]["base_candidates"]+decisions[active[p]]["variant_candidates"]],
        selected_token_id=tokens[p]) for p in range(start, n)]
    return tokens, _decode(tokenizer, tokens, eval_start_pos), tables


def _mark_final_decisions(decisions):
    latest = {}
    for event in decisions:
        event["final_retained"] = False
        if event["accepted_into_formal_state"] and event.get("selected_in_call", True):
            latest[event["position"]] = event
    for event in decisions:
        event["retained_in_formal_state"] = False
    for event in latest.values():
        event["final_retained"] = True
        event["retained_in_formal_state"] = True


def tensor_fingerprint(value):
    """Hash shape, dtype and exact bytes at the continuous/discrete boundary."""
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _frozen_discretization_scope(model, embedding, target, checks):
    """Enforce inference-only finalization and observe all actual model calls."""
    checks.update(embedding_sha256_before=tensor_fingerprint(embedding),
                  target_sha256_before=tensor_fingerprint(target),
                  forward_calls=0, grad_enabled_forward_calls=0,
                  requires_grad_embedding_forward_calls=0)
    def observe(module, args, kwargs):
        del module, args
        checks["forward_calls"] += 1
        checks["grad_enabled_forward_calls"] += int(torch.is_grad_enabled())
        embedded = kwargs.get("inputs_embeds")
        checks["requires_grad_embedding_forward_calls"] += int(
            embedded is not None and embedded.requires_grad)
    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        with torch.no_grad():
            yield
    finally:
        handle.remove()
        checks.update(embedding_sha256_after=tensor_fingerprint(embedding),
                      target_sha256_after=tensor_fingerprint(target))
        checks["embedding_unchanged"] = checks["embedding_sha256_before"] == checks["embedding_sha256_after"]
        checks["target_unchanged"] = checks["target_sha256_before"] == checks["target_sha256_after"]
        checks["passed"] = (checks["embedding_unchanged"] and checks["target_unchanged"]
            and checks["grad_enabled_forward_calls"] == 0
            and checks["requires_grad_embedding_forward_calls"] == 0)
        if not checks["passed"]:
            raise DiscretizationContractError("final discretization violated the freeze boundary")


def run_suffix_reoptimization_v2_5(model, embed_layer, optimized_embedding, target_hidden_state,
        attention_mask, layer_id, register_layer_hooks, tokenizer, config,
        fixed_prefix_tokens=None, eval_start_pos=0, filter_nonascii=True,
        add_perplexity=True, top_k_ppl=10, top_k_cos=10, invert_method="cosine",
        embedding_top_indices=None, select_candidate_from_top_indices=None, get_perplexity=None,
        forward_and_get_last_hidden_state=None, log_file=None, stage_candidates=None, identity=None, sample_id=None):
    del log_file  # Method details are returned only; experiment.log belongs to the main flow.
    embedding = optimized_embedding.detach().clone()
    prefix = [int(token) for token in (fixed_prefix_tokens or [])]
    validate_inputs(model, embedding, target_hidden_state, attention_mask, prefix, eval_start_pos, tokenizer)
    if not config.enabled:
        return embedding, dict(name=METHOD_NAME, method=METHOD_NAME, version=VERSION, enabled=False,
                               skipped=True, reason="disabled", events=[], anomaly_reasons=[])
    if any(helper is None for helper in (embedding_top_indices, select_candidate_from_top_indices,
                                        forward_and_get_last_hidden_state)) or (add_perplexity and get_perplexity is None):
        raise DiscretizationContractError("missing discretization helper")
    identity = identity or trajectory_identity(model, embed_layer, tokenizer)
    stats, decisions, events = _new_compute_stats(), [], []
    optimization_config = copy.copy(config)
    optimization_config.backtracking_enabled = False
    optimization_state = _new_backtracking_state(optimization_config, model, identity, layer_id, sample_id)
    final_sources = copy.deepcopy(stage_candidates or {})
    source_contracts = {"stage1": dict(call_id="stage1", start=eval_start_pos,
        prefix=list(prefix), sampling_steps=list(config.stage1_sampling_steps))}
    begun = time.perf_counter()
    common = dict(tokenizer=tokenizer, model=model, embed_layer=embed_layer, target_hidden_state=target_hidden_state,
        layer_id=layer_id, invert_method=invert_method, filter_nonascii=filter_nonascii,
        add_perplexity=add_perplexity, top_k_ppl=top_k_ppl, top_k_cos=top_k_cos, eval_start_pos=eval_start_pos,
        embedding_top_indices=embedding_top_indices, select_candidate_from_top_indices=select_candidate_from_top_indices,
        get_perplexity=get_perplexity, forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
        config=optimization_config, discretization_events=decisions, identity=identity, compute_stats=stats,
        attention_mask=attention_mask, backtracking_state=optimization_state)
    tokens, text, initial_tables = _rerank_positions(embedding, None, eval_start_pos, prefix,
                                                    stage_candidates=stage_candidates, **common)
    before_tokens = list(tokens)
    with _measure_model_forwards(model, "observation", stats):
        initial_loss = (_hidden_loss_value(model, embedding, target_hidden_state, attention_mask, eval_start_pos,
                        layer_id, register_layer_hooks, config) if embedding.shape[1] > eval_start_pos else None)
    attempts = accepted_count = rejected_count = triggered_count = budget_exhausted = 0
    r_optimizer_calls = 0
    position_attempts, trajectory_stats = {}, []
    r_positions = range(eval_start_pos, embedding.shape[1]) if config.reoptimization_enabled else ()
    for position in r_positions:
        with _measure_model_forwards(model, "observation", stats):
            anchored = _build_anchored_base_embedding(embedding, tokens, position, eval_start_pos, prefix, embed_layer)
            pre_loss = _hidden_loss_value(model, anchored, target_hidden_state, attention_mask, position,
                                         layer_id, register_layer_hooks, config)
            similarity = _position_similarity(model, anchored, target_hidden_state, attention_mask,
                                              position, layer_id, register_layer_hooks)
        triggered = config.trigger_mode == "always" or similarity < config.trigger_threshold
        if triggered:
            triggered_count += 1
        event = dict(position=position, triggered=triggered, attempted=False, accepted=False,
                     s_i_pre=similarity, pre_hidden_loss=pre_loss, post_hidden_loss=None, changed_positions=[])
        if not triggered:
            event["reason"] = "above_trigger_threshold"
            events.append(event)
            continue
        if attempts >= config.max_attempts or position_attempts.get(position, 0) >= config.max_attempts_per_position:
            budget_exhausted += 1
            event["reason"] = "attempt_budget_exhausted"
            events.append(event)
            continue
        attempts += 1
        position_attempts[position] = position_attempts.get(position, 0)+1
        call_id, decision_begin = "R_{}".format(attempts), len(decisions)
        event.update(attempted=True, call_id=call_id)
        collector = DiscretizationTrajectoryCollector(config.steps, position, embed_layer, invert_method,
                                                      call_id, tokens[:position], identity, config.r_sampling_steps)
        try:
            r_optimizer_calls += 1
            with _measure_model_forwards(model, "r_optimization", stats):
                candidate_embedding, optimized_pre, optimized_post, summary = _optimize_suffix(
                    model, embedding, tokens, prefix, target_hidden_state, attention_mask, position,
                    eval_start_pos, layer_id, register_layer_hooks, embed_layer, optimization_config, trajectory_callback=collector)
            trajectories = collector.finalize(summary["completed_steps"], embedding.shape[1])
            summary["discretization_trajectory"] = collector.metadata()
            candidate_tokens, candidate_text, tables = _rerank_positions(candidate_embedding, tokens, position,
                prefix, stage_candidates=trajectories, call_id=call_id, **common)
            accepted = math.isfinite(optimized_pre) and math.isfinite(optimized_post) and optimized_post < optimized_pre
            event.update(accepted=accepted, pre_hidden_loss=optimized_pre, post_hidden_loss=optimized_post,
                changed_positions=_changed_positions(tokens, candidate_tokens, position), candidate_rerank=tables,
                reason="hidden_loss_decreased" if accepted else "hidden_loss_not_decreased", anchor=summary,
                trial_tokens=list(candidate_tokens), entrance_tokens=list(tokens), rolled_back=not accepted)
            if accepted:
                embedding, tokens, text = candidate_embedding.detach().clone(), list(candidate_tokens), candidate_text
                final_sources.update(copy.deepcopy(trajectories))
                source_contracts[call_id] = dict(call_id=call_id, start=position,
                    prefix=list(event["entrance_tokens"][:position]), sampling_steps=list(config.r_sampling_steps))
                accepted_count += 1
            else:
                rejected_count += 1
        except Exception as error:
            for decision in decisions[decision_begin:]:
                decision.update(accepted_into_formal_state=False, retained_in_formal_state=False, rolled_back=True)
            raise_if_fatal(error)
            rejected_count += 1
            accepted = False
            event.update(reason="trial_failed:{}".format(type(error).__name__), error=str(error), rolled_back=True)
        finally:
            trajectory_stats.append(collector.metadata())
        for decision in decisions[decision_begin:]:
            decision.update(accepted_into_formal_state=accepted and decision.get("selected_in_call", True),
                            retained_in_formal_state=accepted and decision.get("selected_in_call", True), rolled_back=not accepted)
        events.append(event)
    with _measure_model_forwards(model, "observation", stats):
        final_loss = (_hidden_loss_value(model, embedding, target_hidden_state, attention_mask, eval_start_pos,
                      layer_id, register_layer_hooks, config) if embedding.shape[1] > eval_start_pos else None)

    # All R acceptance/rollback and continuous-loss observations are now complete.
    # The final scan owns a fresh event budget, history and decision selection.
    optimization_tokens = list(tokens)
    for decision in decisions:
        decision.update(accepted_into_optimization_state=decision["accepted_into_formal_state"],
                        retained_in_optimization_state=False,
                        accepted_into_formal_state=False, retained_in_formal_state=False)
    latest_optimization = {}
    for decision in decisions:
        if decision["accepted_into_optimization_state"] and decision["selected_in_call"]:
            latest_optimization[decision["position"]] = decision
    for decision in latest_optimization.values():
        decision["retained_in_optimization_state"] = True
    optimization_decision_count = len(decisions)
    frozen_embedding = embedding.detach().clone()
    frozen_target = target_hidden_state.detach().clone()
    backtracking = _new_backtracking_state(config, model, identity, layer_id, sample_id)
    r_calls_at_freeze = r_optimizer_calls
    freeze_checks = dict(r_attempts_completed_before_freeze=attempts,
                        r_optimizer_calls_before_freeze=r_calls_at_freeze,
                        optimization_backtracking_events=len(optimization_state["events"]),
                        optimization_backtracking_forward_calls=optimization_state["compute"]["forward_calls"])
    final_common = dict(common, config=config, target_hidden_state=frozen_target,
                        backtracking_state=backtracking, trajectory_contracts=source_contracts)
    with _frozen_discretization_scope(model, frozen_embedding, frozen_target, freeze_checks):
        tokens, text, final_tables = _rerank_positions(frozen_embedding, None, eval_start_pos, prefix,
            stage_candidates=final_sources, call_id="final_discretization", **final_common)
    freeze_checks["r_optimization_calls_after_freeze"] = r_optimizer_calls - r_calls_at_freeze
    _mark_final_decisions(decisions)
    final_decisions = {row["position"]: row for row in decisions if row["final_retained"]}
    if (set(final_decisions) != set(range(eval_start_pos, len(tokens))) or
            any(row["selected_token_id"] != tokens[position]
                or row["prefix_fingerprint"] != prefix_fingerprint(tokens[:position])
                or row["call_id"] != "final_discretization" for position, row in final_decisions.items())):
        raise DiscretizationContractError("final token/decision/prefix mismatch")
    freeze_checks["final_decisions_match_tokens"] = True
    for search in backtracking["events"]:
        search["final_retained"] = bool(search["accepted_into_formal_state"] and
            all(decisions[index]["final_retained"] for index in search.get("committed_decision_indices", [])))
    stats["trajectory"] = dict(retrievals=sum(row["retrievals"] for row in trajectory_stats),
                               elapsed_seconds=sum(row["elapsed_seconds"] for row in trajectory_stats))
    stats["elapsed_seconds"] = time.perf_counter()-begun
    stats["backtracking"] = backtracking["compute"]
    backtracking_result = dict(events=backtracking["events"], event_count=backtracking["events_used"],
        triggered_count=len(backtracking["events"]),
        accepted_count=sum(event["accepted"] for event in backtracking["events"]),
        formal_accepted_count=sum(event["accepted_into_formal_state"] for event in backtracking["events"]),
        final_retained_count=sum(event["final_retained"] for event in backtracking["events"]),
        budget_exhausted_count=sum(event["budget_exhausted"] or event.get("reason") == "forward_budget_exhausted" for event in backtracking["events"]),
        calibration_ready=backtracking["calibration_ready"], calibration=copy.deepcopy(config.backtracking_calibration),
        inference_signature=backtracking["signature"], extra_forward_budget=config.backtracking_forward_budget,
        max_events=config.backtracking_max_events, max_distance=config.backtracking_max_distance,
        alternatives=config.backtracking_alternatives, max_trials=config.backtracking_max_trials,
        lookahead=config.backtracking_lookahead, cache_policy="no_speculative_cache", compute=backtracking["compute"])
    result = dict(name=METHOD_NAME, method=METHOD_NAME, version=VERSION, enabled=True, skipped=False,
        reoptimization_enabled=config.reoptimization_enabled, expansion_enabled=config.expansion_enabled,
        component_switches=dict(R=config.reoptimization_enabled, E=config.expansion_enabled, B=config.backtracking_enabled),
        formal_gt_blind=True, gt_accessed=False, eval_start_pos=eval_start_pos, fixed_prefix_length=len(prefix),
        accept_mode=config.accept_mode, trigger_mode=config.trigger_mode, steps=config.steps,
        max_attempts=config.max_attempts, max_attempts_per_position=config.max_attempts_per_position,
        pre_tokens=before_tokens, pre_text=_decode(tokenizer, before_tokens, eval_start_pos),
        final_tokens=list(tokens), final_text=text, pre_hidden_loss=initial_loss, final_hidden_loss=final_loss,
        triggered=bool(triggered_count or backtracking_result["event_count"]), trigger_count=triggered_count, attempt_count=attempts,
        accepted=bool(accepted_count or backtracking_result["formal_accepted_count"]), accepted_round_count=accepted_count, rejected_round_count=rejected_count,
        budget_exhausted_count=budget_exhausted, reason="accepted {} suffix trial(s), {} backtracking path(s)".format(
            accepted_count, backtracking_result["formal_accepted_count"]),
        initial_candidate_rerank=initial_tables, events=events, backtracking=backtracking_result, discretization=dict(events=decisions,
        expansion_enabled=config.expansion_enabled,
        stage1_sampling_steps=list(config.stage1_sampling_steps), r_sampling_steps=list(config.r_sampling_steps),
        stage1_trajectory_candidates_enabled=config.expansion_enabled and config.trajectory_max_new > 0,
        r_trajectory_candidates_enabled=False,
        trajectory_max_new=config.trajectory_max_new, seed_top_k=config.seed_top_k), compute_stats=stats,
        anomaly_reasons=[event["reason"] for event in events if event["reason"].startswith("trial_failed:")])
    result.update(phase_order=["stage1", "all_R_with_greedy_anchors", "freeze", "final_discretization"],
        freeze_checks=freeze_checks, optimization_final_tokens=optimization_tokens,
        optimization_final_text=_decode(tokenizer, optimization_tokens, eval_start_pos),
        final_candidate_rerank=final_tables, optimization_decision_count=optimization_decision_count,
        final_trajectory_contracts=copy.deepcopy(source_contracts),
        backtracking_trigger="low_fit_required_small_gap_diagnostic_only",
        backtracking_selection="minimum_complete_window_error_then_strict_improvement")
    return frozen_embedding, finite_json(result)


def run_two_stage(*, stage1_kwargs, stage2_kwargs, snapshot_path=None, snapshot_mode=None,
                  pair_id=None, snapshot_contract=None):
    """Grouped shared snapshots keep the old Top10 sampling separate from v2.5."""
    total_begun = time.perf_counter()
    if snapshot_mode not in (None, "write", "read") or (snapshot_mode and not snapshot_path):
        raise DiscretizationContractError("snapshot mode/path mismatch")
    path = Path(snapshot_path) if snapshot_path else None
    config, target = stage2_kwargs["config"], stage2_kwargs["target_hidden_state"].detach()
    prefix, start = stage2_kwargs.get("fixed_prefix_tokens") or [], stage2_kwargs.get("eval_start_pos", 0)
    initial = stage1_kwargs["initial_embedding"]
    prefix_embedding = stage1_kwargs.get("prefix_embedding")
    entry = torch.cat((prefix_embedding, initial), dim=1) if prefix_embedding is not None else initial
    validate_inputs(stage2_kwargs["model"], entry, target, stage2_kwargs["attention_mask"],
                    prefix, start, stage2_kwargs["tokenizer"])
    identity = trajectory_identity(stage2_kwargs["model"], stage2_kwargs["embed_layer"], stage2_kwargs["tokenizer"])
    collector = DiscretizationTrajectoryCollector(stage1_kwargs["epoch"], start, stage2_kwargs["embed_layer"],
        stage2_kwargs.get("invert_method", "cosine"), "stage1", prefix, identity, config.stage1_sampling_steps)
    stats = _new_compute_stats()
    if snapshot_mode == "read":
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot.get("pair_id") != pair_id or snapshot.get("contract") != snapshot_contract:
            raise DiscretizationContractError("Stage-1 snapshot identity/config mismatch")
        if snapshot.get("trajectory_identity") != identity or not torch.equal(target.cpu(), snapshot["target"]):
            raise DiscretizationContractError("paired target/model/vocabulary differs")
        group = (snapshot.get("trajectory_groups", {}).get(VERSION) or
                 snapshot.get("trajectory_groups", {}).get("v2.4") or
                 snapshot.get("trajectory_groups", {}).get("v2.3"))
        if group is None or tuple(group.get("sampling_steps", ())) != config.stage1_sampling_steps:
            raise DiscretizationContractError("snapshot lacks v2.5 Top1 group; recapture Stage-1")
        embedding = snapshot["embedding"].to(stage1_kwargs["initial_embedding"].device)
        summary = dict(snapshot["stage1"])
        summary.pop("checkpoint_trajectory", None)  # Old name belongs only to the compatibility reader.
        summary["discretization_trajectory"] = group["metadata"]
        trajectories = group["candidates"]
        if set(trajectories) != set(range(start, embedding.shape[1])):
            raise DiscretizationContractError("snapshot trajectory positions are incomplete/misaligned")
        metadata = group["metadata"]
        if (metadata.get("identity") != identity or metadata.get("source_start") != start
                or metadata.get("prefix_fingerprint") != prefix_fingerprint(prefix)
                or metadata.get("planned_steps") != stage1_kwargs["epoch"]
                or metadata.get("completed_steps") != summary["completed_steps"]
                or any(row.get("completed_steps") != summary["completed_steps"]
                       or row.get("planned_steps") != stage1_kwargs["epoch"] for row in trajectories.values())):
            raise DiscretizationContractError("snapshot trajectory metadata differs from actual Stage-1")
        stats["stage1"] = snapshot["stage1_compute_stats"]
        restore_rng(snapshot["rng"])
    else:
        legacy = None
        if snapshot_mode == "write":
            # Read-only adapter: the old sidecar/config stays byte-for-byte unchanged.
            from .suffix_reoptimization_v2_2_2_2 import StageCandidateCollector as LegacyCollector
            legacy = LegacyCollector(stage1_kwargs["epoch"], start, stage2_kwargs["embed_layer"],
                stage2_kwargs.get("invert_method", "cosine"), "stage1", prefix)
        def collect(step, embedding):
            collector(step, embedding)
            if legacy is not None:
                legacy(step, embedding)
        with _measure_model_forwards(stage2_kwargs["model"], "stage1", stats):
            embedding, summary = optimize_stage1(**dict(stage1_kwargs, trajectory_callback=collect))
        trajectories = collector.finalize(summary["completed_steps"], embedding.shape[1])
        summary["discretization_trajectory"] = collector.metadata()
        if snapshot_mode == "write":
            legacy_candidates = legacy.finalize(summary["completed_steps"], embedding.shape[1])
            old_summary = dict(summary, checkpoint_trajectory=legacy.metadata())
            old_summary.pop("discretization_trajectory", None)
            snapshot = dict(pair_id=pair_id, contract=snapshot_contract, embedding=embedding.cpu(), target=target.cpu(),
                stage1=old_summary, rng=rng_state(), stage_candidates=legacy_candidates,
                trajectory_identity=identity, stage1_compute_stats=stats["stage1"],
                trajectory_groups={VERSION: dict(sampling_steps=list(config.stage1_sampling_steps),
                    candidates=trajectories, metadata=collector.metadata()),
                    "v2.4": dict(sampling_steps=list(config.stage1_sampling_steps),
                        candidates=trajectories, metadata=collector.metadata()),
                    "v2.3": dict(sampling_steps=list(config.stage1_sampling_steps),
                        candidates=trajectories, metadata=collector.metadata()),
                    "v2.2.2(2)": dict(candidates=legacy_candidates, metadata=legacy.metadata())})
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            torch.save(snapshot, path)
    validate_inputs(stage2_kwargs["model"], embedding, target, stage2_kwargs["attention_mask"],
                    prefix, start, stage2_kwargs["tokenizer"])
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest() if path else hashlib.sha256(
        embedding.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
        torch.cuda.reset_peak_memory_stats(embedding.device)
    begun = time.perf_counter()
    final, result = run_suffix_reoptimization_v2_5(**dict(stage2_kwargs, optimized_embedding=embedding,
        target_hidden_state=target.clone(), stage_candidates=trajectories, identity=identity,
        sample_id=stage2_kwargs.get("sample_id", pair_id)))
    if embedding.is_cuda:
        torch.cuda.synchronize(embedding.device)
    elapsed = time.perf_counter()-begun
    result["compute_stats"]["stage1"] = dict(stats["stage1"], executed_in_this_run=snapshot_mode != "read")
    result["compute_stats"]["stage1_trajectory"] = summary["discretization_trajectory"]
    if snapshot_mode:
        legacy_metadata = (legacy.metadata() if snapshot_mode == "write" else
                           snapshot.get("trajectory_groups", {}).get("v2.2.2(2)", {}).get("metadata"))
        result["compute_stats"]["shared_snapshot_legacy_trajectory"] = legacy_metadata
    result["compute_stats"]["total_elapsed_seconds"] = time.perf_counter()-total_begun
    reoptimization = dict(result)
    result.update(stage1=summary, reoptimization=reoptimization, pair_id=pair_id,
        stage1_snapshot_sha256=fingerprint, stage1_reused=snapshot_mode == "read", second_stage_seconds=elapsed,
        comparable_end_to_end_seconds=elapsed+summary.get("elapsed_seconds", 0.),
        second_stage_peak_memory_bytes=torch.cuda.max_memory_allocated(embedding.device) if embedding.is_cuda else None)
    return final, finite_json(result)
