"""v2.3: Top1 trajectories, scored seeds, and local discretization variants.

Continuous Stage-1/R kernels are copied from v2.2.2(2). Candidate scores are
local to one position and prefix; private labels enter only offline evaluation.
"""
from dataclasses import dataclass
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

METHOD_NAME = "suffix_reoptimization_v2.3"
VERSION = "v2.3"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192



@dataclass
class SuffixReoptimizationV23Config:
    enabled: bool = False
    log_enabled: bool = True
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

    def __post_init__(self):
        for name, ceiling in (("trajectory_max_new", 5), ("seed_top_k", 5),
                              ("case_quota", 3), ("space_quota", 3),
                              ("prefix_quota", 1), ("suffix_quota", 3)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= ceiling:
                raise ValueError("suffix_v2_3_{} must be in [0, {}]".format(name, ceiling))
        if type(self.candidate_batch_size) is not int or self.candidate_batch_size <= 0:
            raise ValueError("suffix_v2_3_candidate_batch_size must be positive")
        for name, expected in (("stage1_sampling_steps", (500, 600, 700, 800, 900)),
                               ("r_sampling_steps", (30, 40))):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)) or any(type(s) is not int for s in value):
                raise TypeError("suffix_v2_3_{} must be integer steps".format(name))
            if tuple(value) != expected:
                raise ValueError("v2.3 uses fixed absolute sampling steps: {}".format(expected))
            setattr(self, name, tuple(value))
        for name in ("enabled", "log_enabled", "filter_nonascii"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError("suffix_v2_3_{} must be boolean".format(name))
        for name in (
            "max_attempts",
            "max_attempts_per_position",
            "steps",
            "range_top_k",
        ):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise TypeError("suffix_v2_3_{} must be an integer".format(name))
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
                raise TypeError("suffix_v2_3_{} must be numeric".format(name))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("suffix_v2_3_{} must be finite".format(name))
            setattr(self, name, value)
        if self.max_attempts < 0:
            raise ValueError("suffix_v2_3_max_attempts must be non-negative")
        if self.max_attempts_per_position <= 0:
            raise ValueError(
                "suffix_v2_3_max_attempts_per_position must be positive"
            )
        if self.steps <= 0:
            raise ValueError("suffix_v2_3_steps must be positive")
        if self.lr <= 0.0:
            raise ValueError("suffix_v2_3_lr must be positive")
        if not 0.0 <= self.hidden_weight_decay <= 1.0:
            raise ValueError(
                "suffix_v2_3_hidden_weight_decay must be in [0, 1]"
            )
        if self.hidden_weight_floor < 0.0:
            raise ValueError(
                "suffix_v2_3_hidden_weight_floor must be non-negative"
            )
        if self.prox_weight < 0.0 or self.range_weight < 0.0:
            raise ValueError(
                "suffix_v2_3 regularization weights must be non-negative"
            )
        if self.range_top_k <= 0:
            raise ValueError("suffix_v2_3_range_top_k must be positive")
        if self.trigger_mode not in {"always", "threshold"}:
            raise ValueError(
                "suffix_v2_3_trigger_mode must be always or threshold"
            )
        if self.hidden_weight_mode != "front_decay":
            raise ValueError(
                "suffix_v2_3_hidden_weight_mode must be front_decay"
            )
        if self.accept_mode != "hidden_loss":
            raise ValueError(
                "suffix_v2_3_accept_mode must be hidden_loss"
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
    prefixes = {"enabled": "suffix_reoptimization_v2_3",
                "log_enabled": "suffix_reoptimization_v2_3_log"}
    kwargs = {}
    for field in SuffixReoptimizationV23Config.__dataclass_fields__:
        key = prefixes.get(field, "suffix_v2_3_" + field)
        if key not in values:
            raise ValueError("missing explicit v2.3 configuration: " + key)
        kwargs[field] = values[key]
    return SuffixReoptimizationV23Config(**kwargs)


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
        raise DiscretizationContractError("v2.3 requires model.eval()")


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
                              trajectory=None, max_new=5):
    """Keep B0 intact; offer only the first ID from each chronological sample."""
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
    for stage in sorted((trajectory or {}).get("stages", []), key=lambda row: row["step"]):
        tokens = stage["token_ids"]
        if not tokens:
            exclusions.append(dict(source="trajectory", step=stage["step"], reason="missing_top1"))
            continue
        token = tokens[0]  # Never backfill from Top2, including legacy Top10 metadata.
        label = dict(source="trajectory", step=stage["step"], call_id=trajectory["call_id"], rank=1)
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
                trajectory=trajectory, exclusions=exclusions, trajectory_quota_gap=max_new-count)


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
    quotas = {category: getattr(config, category+"_quota") for category in ("case", "space", "prefix", "suffix")}
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


def _score_candidates(ids, tokens, position, target, model, layer_id, forward, batch_size, stats):
    scores = []
    for begin in range(0, len(ids), batch_size):
        subset = ids[begin:begin+batch_size]
        trials = [list(tokens) for _ in subset]
        for trial, token in zip(trials, subset):
            trial[position] = token
        begun = time.perf_counter()
        stats["candidate"]["forward_calls"] += 1
        stats["candidate"]["candidate_count"] += len(subset)
        stats["candidate"]["token_count"] += len(subset)*len(tokens)
        with torch.no_grad(), _preserve_forward_hooks(model):
            hidden = forward(model, trials, None, layer_id=layer_id)
            if hidden.ndim != 3 or hidden.shape[:2] != (len(subset), len(tokens)) or hidden.shape[2] != target.shape[2]:
                raise DiscretizationContractError("candidate hidden shape mismatch")
            similarity = F.cosine_similarity(hidden[:, position, :].float(),
                                             target[:, position, :].to(hidden.device).float(), dim=-1)
        scores.extend(float(value) for value in similarity.detach().cpu().tolist())
        stats["candidate"]["elapsed_seconds"] += time.perf_counter()-begun
    return scores


def _rerank_positions(input_embed, current_tokens, rerank_start, fixed_prefix_tokens,
                      tokenizer, model, embed_layer, target_hidden_state, layer_id,
                      invert_method, filter_nonascii, add_perplexity, top_k_ppl, top_k_cos,
                      eval_start_pos, embedding_top_indices, select_candidate_from_top_indices,
                      get_perplexity, forward_and_get_last_hidden_state, *, config,
                      stage_candidates=None, discretization_events=None, call_id="stage1",
                      identity=None, compute_stats=None):
    sequence_length = input_embed.shape[1]
    tokens = [0]*sequence_length if current_tokens is None else list(current_tokens)
    if len(tokens) != sequence_length or len(fixed_prefix_tokens) != eval_start_pos:
        raise DiscretizationContractError("discretization token/prefix alignment mismatch")
    tokens[:eval_start_pos] = fixed_prefix_tokens
    start = max(eval_start_pos, rerank_start)
    entry_prefix = list(tokens[:start])
    identity = identity or trajectory_identity(model, embed_layer, tokenizer)
    sampling_steps = config.stage1_sampling_steps if call_id == "stage1" else config.r_sampling_steps
    stats = compute_stats if compute_stats is not None else _new_compute_stats()
    tables, initial_top_ids = [], {}
    for position in range(start, sequence_length):
        selected, ids = _candidate_token_ids(input_embed[0, position], embed_layer, top_k_cos,
            invert_method, tokenizer, filter_nonascii, embedding_top_indices, select_candidate_from_top_indices)
        initial_top_ids[position] = ids or [tokens[position]]
        tokens[position] = selected
    for position in range(start, sequence_length):
        begun = time.perf_counter()
        prefix_hash = prefix_fingerprint(tokens[:position])
        trajectory = (stage_candidates or {}).get(position)
        validate_trajectory(trajectory, position=position, start=start, call_id=call_id,
                            prefix=entry_prefix, identity=identity, sampling_steps=sampling_steps, metric=invert_method)
        ppl_ids = []
        if add_perplexity and position > 0 and top_k_ppl > 0:
            ppl_begun = time.perf_counter()
            stats["ppl"]["forward_calls"] += 1
            stats["ppl"]["token_count"] += position
            with torch.no_grad(), _preserve_forward_hooks(model):
                _, ids = get_perplexity(tokens[:position], model, layer_id=layer_id, top_k=top_k_ppl)
            ppl_ids = [int(token) for token in ids]
            stats["ppl"]["elapsed_seconds"] += time.perf_counter()-ppl_begun
        base = build_discretization_base(initial_top_ids[position], ppl_ids, tokenizer,
            embed_layer.num_embeddings, config.filter_nonascii, trajectory, config.trajectory_max_new)
        base_rows = base["candidates"]
        base_ids = [row["token_id"] for row in base_rows]
        before_calls, before_count = stats["candidate"]["forward_calls"], stats["candidate"]["candidate_count"]
        base_scores = _score_candidates(base_ids, tokens, position, target_hidden_state, model, layer_id,
            forward_and_get_last_hidden_state, config.candidate_batch_size, stats)
        for row, score in zip(base_rows, base_scores):
            row["score"] = score
            row["validly_scored"] = math.isfinite(score)
            if not math.isfinite(score):
                if row["quota_source"] == "original":
                    raise DiscretizationContractError("nonfinite original candidate score")
                base["exclusions"].append(dict(token_id=row["token_id"], source="trajectory", reason="nonfinite_score"))
        eligible = [row for row in base_rows if row["validly_scored"]]
        best_base = max(eligible, key=lambda row: row["score"])
        variants = build_discretization_variants(base_rows, tokenizer, embed_layer.num_embeddings, config.filter_nonascii, config)
        variant_rows = variants["candidates"]
        variant_ids = [row["token_id"] for row in variant_rows]
        variant_scores = _score_candidates(variant_ids, tokens, position, target_hidden_state, model, layer_id,
            forward_and_get_last_hidden_state, config.candidate_batch_size, stats)
        best = best_base
        for row, score in zip(variant_rows, variant_scores):
            row["score"], row["validly_scored"] = score, math.isfinite(score)
            if not math.isfinite(score):
                variants["exclusions"].append(dict(token_id=row["token_id"], source=row["quota_source"], reason="nonfinite_score"))
            elif score > best["score"]:
                best = row
        tokens[position] = best["token_id"]
        valid_variants = [row for row in variant_rows if row["validly_scored"]]
        best_variant = max(valid_variants, key=lambda row: row["score"]) if valid_variants else None
        event = dict(phase="initial" if call_id == "stage1" else "R", call_id=call_id, position=position,
            prefix_fingerprint=prefix_hash, entrance_prefix_fingerprint=prefix_fingerprint(entry_prefix),
            original_candidate_ids=base["original_candidate_ids"], original_top_ids=initial_top_ids[position], ppl_ids=ppl_ids,
            trajectory_candidate_ids=base["trajectory_candidate_ids"], trajectory=trajectory,
            base_candidates=base_rows, base_scores=base_scores, base_best_token_id=best_base["token_id"],
            base_best_score=best_base["score"], seeds=variants["seeds"],
            variant_candidates=variant_rows, variant_scores=variant_scores,
            variant_best_token_id=best_variant["token_id"] if best_variant else None,
            variant_best_score=best_variant["score"] if best_variant else None,
            selected_token_id=best["token_id"], selected_score=best["score"],
            tie_kept_base=any(row["score"] == best_base["score"] for row in valid_variants) and best is best_base,
            exclusions=base["exclusions"]+variants["exclusions"], quotas=variants["quotas"],
            quota_gaps=variants["quota_gaps"], trajectory_quota_gap=base["trajectory_quota_gap"],
            generation_counts=variants["counts"], generated_variant_count=len(variant_ids),
            validly_scored_candidate_count=len(eligible)+len(valid_variants),
            candidate_forward_calls=stats["candidate"]["forward_calls"]-before_calls,
            scored_candidate_count=stats["candidate"]["candidate_count"]-before_count,
            forward_token_count=(len(base_ids)+len(variant_ids))*sequence_length,
            elapsed_seconds=time.perf_counter()-begun,
            stage_produced=True, accepted_into_formal_state=call_id == "stage1",
            retained_in_formal_state=call_id == "stage1", final_retained=False, rolled_back=False)
        if discretization_events is not None:
            discretization_events.append(event)
        tables.append(dict(position=position, call_id=call_id, prefix_fingerprint=prefix_hash,
            candidate_token_ids=base_ids+variant_ids, candidate_hidden_cosine=base_scores+variant_scores,
            selected_token_id=tokens[position]))
    return tokens, _decode(tokenizer, tokens, eval_start_pos), tables


def _mark_final_decisions(decisions):
    latest = {}
    for event in decisions:
        event["final_retained"] = False
        if event["accepted_into_formal_state"]:
            latest[event["position"]] = event
    for event in latest.values():
        event["final_retained"] = True


def run_suffix_reoptimization_v2_3(model, embed_layer, optimized_embedding, target_hidden_state,
        attention_mask, layer_id, register_layer_hooks, tokenizer, config,
        fixed_prefix_tokens=None, eval_start_pos=0, filter_nonascii=True,
        add_perplexity=True, top_k_ppl=10, top_k_cos=10, invert_method="cosine",
        embedding_top_indices=None, select_candidate_from_top_indices=None, get_perplexity=None,
        forward_and_get_last_hidden_state=None, log_file=None, stage_candidates=None, identity=None):
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
    begun = time.perf_counter()
    common = dict(tokenizer=tokenizer, model=model, embed_layer=embed_layer, target_hidden_state=target_hidden_state,
        layer_id=layer_id, invert_method=invert_method, filter_nonascii=filter_nonascii,
        add_perplexity=add_perplexity, top_k_ppl=top_k_ppl, top_k_cos=top_k_cos, eval_start_pos=eval_start_pos,
        embedding_top_indices=embedding_top_indices, select_candidate_from_top_indices=select_candidate_from_top_indices,
        get_perplexity=get_perplexity, forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
        config=config, discretization_events=decisions, identity=identity, compute_stats=stats)
    tokens, text, initial_tables = _rerank_positions(embedding, None, eval_start_pos, prefix,
                                                    stage_candidates=stage_candidates, **common)
    before_tokens = list(tokens)
    with _measure_model_forwards(model, "observation", stats):
        initial_loss = (_hidden_loss_value(model, embedding, target_hidden_state, attention_mask, eval_start_pos,
                        layer_id, register_layer_hooks, config) if embedding.shape[1] > eval_start_pos else None)
    attempts = accepted_count = rejected_count = triggered_count = budget_exhausted = 0
    position_attempts, trajectory_stats = {}, []
    for position in range(eval_start_pos, embedding.shape[1]):
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
            with _measure_model_forwards(model, "r_optimization", stats):
                candidate_embedding, optimized_pre, optimized_post, summary = _optimize_suffix(
                    model, embedding, tokens, prefix, target_hidden_state, attention_mask, position,
                    eval_start_pos, layer_id, register_layer_hooks, embed_layer, config, trajectory_callback=collector)
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
            decision.update(accepted_into_formal_state=accepted, retained_in_formal_state=accepted, rolled_back=not accepted)
        events.append(event)
    _mark_final_decisions(decisions)
    with _measure_model_forwards(model, "observation", stats):
        final_loss = (_hidden_loss_value(model, embedding, target_hidden_state, attention_mask, eval_start_pos,
                      layer_id, register_layer_hooks, config) if embedding.shape[1] > eval_start_pos else None)
    stats["trajectory"] = dict(retrievals=sum(row["retrievals"] for row in trajectory_stats),
                               elapsed_seconds=sum(row["elapsed_seconds"] for row in trajectory_stats))
    stats["elapsed_seconds"] = time.perf_counter()-begun
    result = dict(name=METHOD_NAME, method=METHOD_NAME, version=VERSION, enabled=True, skipped=False,
        formal_gt_blind=True, gt_accessed=False, eval_start_pos=eval_start_pos, fixed_prefix_length=len(prefix),
        accept_mode=config.accept_mode, trigger_mode=config.trigger_mode, steps=config.steps,
        max_attempts=config.max_attempts, max_attempts_per_position=config.max_attempts_per_position,
        pre_tokens=before_tokens, pre_text=_decode(tokenizer, before_tokens, eval_start_pos),
        final_tokens=list(tokens), final_text=text, pre_hidden_loss=initial_loss, final_hidden_loss=final_loss,
        triggered=bool(triggered_count), trigger_count=triggered_count, attempt_count=attempts,
        accepted=bool(accepted_count), accepted_round_count=accepted_count, rejected_round_count=rejected_count,
        budget_exhausted_count=budget_exhausted, reason="accepted {} suffix trial(s)".format(accepted_count),
        initial_candidate_rerank=initial_tables, events=events, discretization=dict(events=decisions,
        stage1_sampling_steps=list(config.stage1_sampling_steps), r_sampling_steps=list(config.r_sampling_steps),
        trajectory_max_new=config.trajectory_max_new, seed_top_k=config.seed_top_k), compute_stats=stats,
        anomaly_reasons=[event["reason"] for event in events if event["reason"].startswith("trial_failed:")])
    return embedding, finite_json(result)


def run_two_stage(*, stage1_kwargs, stage2_kwargs, snapshot_path=None, snapshot_mode=None,
                  pair_id=None, snapshot_contract=None):
    """Grouped shared snapshots keep the old Top10 sampling separate from v2.3."""
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
        group = snapshot.get("trajectory_groups", {}).get(VERSION)
        if group is None or tuple(group.get("sampling_steps", ())) != config.stage1_sampling_steps:
            raise DiscretizationContractError("snapshot lacks v2.3 Top1 group; recapture Stage-1")
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
    final, result = run_suffix_reoptimization_v2_3(**dict(stage2_kwargs, optimized_embedding=embedding,
        target_hidden_state=target.clone(), stage_candidates=trajectories, identity=identity))
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
