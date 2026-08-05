from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F


MIN_ACCEPTANCE_EPS = 1e-12
RELATIVE_MSE_EPSILON = 1e-8
GRADIENT_TREND_EPSILON = 1e-12
GRADIENT_EMA_BETA = 0.9
VOCAB_SEARCH_CHUNK_SIZE = 8192
METHOD_NAME = "suffix_reoptimization_v1.2.2"
VERSION = "v1.2.2"
DEFAULT_ACCEPT_MODE = "oracle_accuracy"
ACCEPT_MODES = {"oracle_accuracy", "hidden_anomaly", "always"}
DEFAULT_ANOMALY_DETECTION_MODE = "adaptive"
ANOMALY_DETECTION_MODES = {"adaptive", "threshold"}
METRIC_DIRECTION_NOTE = (
    "cosine similarity is higher-is-better; cosine loss, relative MSE, "
    "joint hidden loss, and total loss are lower-is-better; metrics with "
    "different definitions must not be compared directly"
)


@dataclass
class SuffixReoptimizationV122Config:
    enabled: bool = False
    log_enabled: bool = True
    max_rounds: int = 2
    epoch: int = 50
    lr: float = 0.03
    embedding_relative_mse_high_threshold: float = 1.0
    relative_mse_rise_threshold: float = 0.30
    token_relative_mse_high_threshold: float = 1.0
    min_anomaly_reasons: int = 2
    min_relative_mse_improvement: float = 0.01
    accuracy_tolerance: float = 0.0
    accept_mode: str = DEFAULT_ACCEPT_MODE
    anomaly_detection_mode: str = DEFAULT_ANOMALY_DETECTION_MODE
    adaptive_z_threshold: float = 1.5
    adaptive_rise_z_threshold: float = 1.5
    adaptive_min_std: float = 1e-6
    adaptive_min_points: int = 4
    hidden_weight_mode: str = "front_decay"
    hidden_weight_decay: float = 0.90
    hidden_weight_floor: float = 0.20
    cosine_loss_weight: float = 0.1
    relative_mse_loss_weight: float = 0.9
    prox_weight: float = 0.005
    range_weight: float = 0.001
    gradient_trend_stats_enabled: bool = True


class BaselineGradientTrendTracker:
    """Read-only per-position baseline gradient trend statistics."""

    def __init__(self, enabled=True, position_offset=0, beta=GRADIENT_EMA_BETA):
        self.enabled = bool(enabled)
        self.position_offset = int(position_offset)
        self.beta = float(beta)
        self.observed_step_count = 0
        self._gradient_ema = None
        self._gradient_norm_ema = None
        self._cosine_sum = None
        self._valid_step_count = None

    def observe(self, gradient):
        if not self.enabled or gradient is None:
            return
        current = gradient.detach().to(dtype=torch.float32)
        if current.ndim == 3 and int(current.shape[0]) == 1:
            current = current.squeeze(0)
        if current.ndim != 2:
            raise ValueError("baseline gradient must have shape [positions, hidden]")

        current_norm = torch.linalg.vector_norm(current, dim=-1)
        if self._gradient_ema is None:
            self._gradient_ema = current.clone()
            self._gradient_norm_ema = current_norm.clone()
            self._cosine_sum = torch.zeros_like(current_norm)
            self._valid_step_count = torch.zeros_like(current_norm, dtype=torch.long)
        else:
            history_norm = torch.linalg.vector_norm(self._gradient_ema, dim=-1)
            valid = (
                (current_norm > GRADIENT_TREND_EPSILON)
                & (history_norm > GRADIENT_TREND_EPSILON)
            )
            if bool(valid.any()):
                cosine = F.cosine_similarity(
                    current[valid],
                    self._gradient_ema[valid],
                    dim=-1,
                )
                self._cosine_sum[valid] += cosine
                self._valid_step_count[valid] += 1
            one_minus_beta = 1.0 - self.beta
            self._gradient_ema.mul_(self.beta).add_(current, alpha=one_minus_beta)
            self._gradient_norm_ema.mul_(self.beta).add_(
                current_norm,
                alpha=one_minus_beta,
            )
        self.observed_step_count += 1

    @staticmethod
    def _mean_present(values):
        values = [value for value in values if value is not None]
        if not values:
            return None
        return float(sum(values) / len(values))

    def summary(self):
        if not self.enabled:
            return {
                "enabled": False,
                "position_offset": self.position_offset,
                "ema_beta": self.beta,
                "observed_step_count": 0,
                "positions": [],
            }
        if self._gradient_ema is None:
            return {
                "enabled": True,
                "position_offset": self.position_offset,
                "ema_beta": self.beta,
                "observed_step_count": 0,
                "positions": [],
                "summary_mean": {
                    "recent_gradient_mean_norm": None,
                    "historical_direction_consistency": None,
                    "current_vs_history_cosine": None,
                },
            }

        history_norm = torch.linalg.vector_norm(self._gradient_ema, dim=-1)
        positions = []
        recent_values = []
        consistency_values = []
        cosine_values = []
        for local_position in range(int(history_norm.shape[0])):
            recent_norm = float(
                self._gradient_norm_ema[local_position].detach().cpu()
            )
            consistency = None
            if recent_norm > GRADIENT_TREND_EPSILON:
                consistency = float(
                    (
                        history_norm[local_position]
                        / self._gradient_norm_ema[local_position]
                    )
                    .detach()
                    .cpu()
                )
            valid_step_count = int(
                self._valid_step_count[local_position].detach().cpu()
            )
            current_vs_history_cosine = None
            if valid_step_count > 0:
                current_vs_history_cosine = float(
                    (
                        self._cosine_sum[local_position]
                        / self._valid_step_count[local_position]
                    )
                    .detach()
                    .cpu()
                )
            positions.append({
                "position": self.position_offset + local_position,
                "recent_gradient_mean_norm": recent_norm,
                "historical_direction_consistency": consistency,
                "current_vs_history_cosine": current_vs_history_cosine,
                "valid_step_count": valid_step_count,
            })
            recent_values.append(recent_norm)
            consistency_values.append(consistency)
            cosine_values.append(current_vs_history_cosine)

        return {
            "enabled": True,
            "position_offset": self.position_offset,
            "ema_beta": self.beta,
            "observed_step_count": self.observed_step_count,
            "positions": positions,
            "summary_mean": {
                "recent_gradient_mean_norm": self._mean_present(recent_values),
                "historical_direction_consistency": self._mean_present(
                    consistency_values
                ),
                "current_vs_history_cosine": self._mean_present(cosine_values),
            },
        }


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _safe_mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_max(values):
    if not values:
        return None
    return float(max(values))


def _target_ids(total_input_ids):
    return [int(item.data.cpu()) for item in total_input_ids[0]]


def _accuracy(token_ids, total_input_ids, eval_start_pos):
    target_ids = _target_ids(total_input_ids)
    eval_length = max(len(target_ids) - eval_start_pos, 0)
    if eval_length == 0:
        return 0.0
    correct = 0
    for pos in range(eval_start_pos, len(target_ids)):
        if int(token_ids[pos]) == int(target_ids[pos]):
            correct += 1
    return correct / eval_length


def _decode(tokenizer, token_ids, eval_start_pos):
    return tokenizer.decode(torch.tensor(token_ids[eval_start_pos:]))


def _dedupe(token_ids):
    seen = set()
    ret = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        ret.append(token_id)
    return ret


def _relative_mse(current_hidden, target_hidden):
    current_hidden = current_hidden.float()
    target_hidden = target_hidden.float()
    return (
        (current_hidden - target_hidden).pow(2).mean(dim=-1)
        / (
            target_hidden.pow(2).mean(dim=-1)
            + RELATIVE_MSE_EPSILON
        )
    )


def _forward_embedding_hidden(model, input_embed, attention_mask, layer_id,
                              register_layer_hooks):
    hidden_state_list = []

    def forward_hook(module, inputs, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    hook_handles = register_layer_hooks(model, layer_id, forward_hook, up_to=False)
    try:
        model(inputs_embeds=input_embed, attention_mask=attention_mask)
    finally:
        for handle in hook_handles:
            handle.remove()
    if not hidden_state_list:
        raise ValueError("no hidden states collected for layer {}".format(layer_id))
    return hidden_state_list[0]


def _embedding_forward_relative_mse(model, input_embed, target_hidden_state,
                                    attention_mask, layer_id,
                                    register_layer_hooks):
    with torch.no_grad():
        hidden_state = _forward_embedding_hidden(
            model,
            input_embed,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        values = _relative_mse(hidden_state, target_hidden).squeeze(0)
    return [_as_float(item) for item in values]


def _token_forward_relative_mse(model, token_ids, target_hidden_state, layer_id,
                                forward_and_get_last_hidden_state):
    with torch.no_grad():
        hidden_state = forward_and_get_last_hidden_state(
            model,
            token_ids,
            None,
            layer_id=layer_id,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        values = _relative_mse(hidden_state, target_hidden).squeeze(0)
    return [_as_float(item) for item in values]


def _candidate_token_ids(embed, embed_layer, top_k_cos, invert_method, tokenizer,
                         filter_nonascii, embedding_top_indices,
                         select_candidate_from_top_indices):
    if top_k_cos == 0:
        return []
    top_indices = embedding_top_indices(
        embed,
        embed_layer,
        top_k_cos,
        invert_method,
    )
    _, top_ids = select_candidate_from_top_indices(
        top_indices,
        tokenizer,
        filter_nonascii,
    )
    return [int(item) for item in top_ids]


def _rerank_positions(input_embed, current_tokens, rerank_start, tokenizer, model,
                      embed_layer, target_hidden_state, total_input_ids, layer_id,
                      invert_method, filter_nonascii, add_perplexity, top_k_ppl,
                      top_k_cos, eval_start_pos, embedding_top_indices,
                      select_candidate_from_top_indices, get_perplexity,
                      forward_and_get_last_hidden_state):
    seq_len = int(input_embed.shape[1])
    if current_tokens is None:
        ret_list = [0 for _ in range(seq_len)]
    else:
        ret_list = [int(item) for item in current_tokens]

    target_ids = _target_ids(total_input_ids)
    for pos in range(min(eval_start_pos, seq_len)):
        ret_list[pos] = target_ids[pos]

    new_input_embed_squeeze = input_embed.squeeze(0)
    ret_top_k = {}
    for pos in range(max(eval_start_pos, rerank_start), seq_len):
        top_list = _candidate_token_ids(
            new_input_embed_squeeze[pos],
            embed_layer,
            top_k_cos,
            invert_method,
            tokenizer,
            filter_nonascii,
            embedding_top_indices,
            select_candidate_from_top_indices,
        )
        if not top_list:
            top_list = [ret_list[pos]]
        if current_tokens is not None:
            top_list.append(int(current_tokens[pos]))
        top_list = _dedupe(top_list)
        ret_top_k[pos] = top_list
        ret_list[pos] = int(top_list[0])

    selected_candidate_diagnostics = []
    for pos in range(max(eval_start_pos, rerank_start), seq_len):
        top_list = list(ret_top_k.get(pos, [ret_list[pos]]))
        if pos > 0 and add_perplexity:
            _, topk_ids = get_perplexity(
                ret_list[:pos],
                model,
                layer_id=layer_id,
                top_k=top_k_ppl,
            )
            top_list.extend([int(item) for item in topk_ids.tolist()])
            top_list = _dedupe(top_list)

        replaced_ret_list = []
        for token_id in top_list:
            replaced_ret = copy.deepcopy(ret_list)
            replaced_ret[pos] = int(token_id)
            replaced_ret_list.append(replaced_ret)

        new_hidden_states = forward_and_get_last_hidden_state(
            model,
            replaced_ret_list,
            None,
            layer_id=layer_id,
        )
        target_hidden = target_hidden_state.to(new_hidden_states.device)
        candidate_states = new_hidden_states[:, pos, :]
        target_state = target_hidden[:, pos, :]
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        candidate_relative_mse = _relative_mse(candidate_states, target_state)
        best_idx = int(torch.argmin(candidate_relative_mse).detach().cpu().item())
        ret_list[pos] = int(top_list[best_idx])
        selected_candidate_diagnostics.append({
            "position": pos,
            "selected_token_id": ret_list[pos],
            "candidate_relative_mse": _as_float(
                candidate_relative_mse[best_idx]
            ),
            "candidate_count": len(top_list),
        })

    acc = _accuracy(ret_list, total_input_ids, eval_start_pos)
    text = _decode(tokenizer, ret_list, eval_start_pos)
    return acc, text, ret_list, selected_candidate_diagnostics


def _stats(values, min_points, min_std):
    values = [float(value) for value in values if value is not None]
    if len(values) < max(1, int(min_points)):
        return None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std_value = variance ** 0.5
    if std_value < float(min_std):
        return None
    return mean_value, std_value


def _z_above(value, stats):
    if value is None or stats is None:
        return None
    mean_value, std_value = stats
    return (float(value) - mean_value) / std_value


def _find_threshold_anomalies(embedding_values, token_values, scan_pos, config):
    anomalies = []
    min_reasons = max(1, int(config.min_anomaly_reasons))
    for pos in range(max(0, scan_pos), len(embedding_values)):
        reasons = []
        current_embedding = embedding_values[pos]
        previous_embedding = embedding_values[pos - 1] if pos > 0 else None
        token_forward = token_values[pos] if pos < len(token_values) else None
        relative_mse_rise = (
            None
            if previous_embedding is None
            else current_embedding - previous_embedding
        )
        if (
            config.embedding_relative_mse_high_threshold >= 0.0
            and current_embedding > config.embedding_relative_mse_high_threshold
        ):
            reasons.append("embedding_forward_relative_mse_high")
        if (
            config.relative_mse_rise_threshold > 0.0
            and relative_mse_rise is not None
            and relative_mse_rise >= config.relative_mse_rise_threshold
        ):
            reasons.append("embedding_forward_relative_mse_rise")
        if (
            config.token_relative_mse_high_threshold >= 0.0
            and token_forward is not None
            and token_forward > config.token_relative_mse_high_threshold
        ):
            reasons.append("token_forward_relative_mse_high")
        if len(reasons) >= min_reasons:
            anomalies.append({
                "position": pos,
                "reasons": reasons,
                "anomaly_detection_mode": "threshold",
                "embedding_forward_relative_mse": current_embedding,
                "previous_embedding_forward_relative_mse": previous_embedding,
                "relative_mse_rise": relative_mse_rise,
                "token_forward_relative_mse": token_forward,
            })
    return anomalies


def _find_adaptive_anomalies(embedding_values, token_values, scan_pos, config):
    anomalies = []
    start_pos = max(0, int(scan_pos))
    min_points = max(2, int(config.adaptive_min_points))
    min_std = max(0.0, float(config.adaptive_min_std))
    high_z_threshold = max(0.0, float(config.adaptive_z_threshold))
    rise_z_threshold = max(0.0, float(config.adaptive_rise_z_threshold))

    embedding_stats = _stats(
        embedding_values[start_pos:],
        min_points,
        min_std,
    )
    token_region = token_values[start_pos:] if start_pos < len(token_values) else []
    token_stats = _stats(token_region, min_points, min_std)
    token_rises = []
    for pos in range(max(1, start_pos), len(token_values)):
        previous_token = token_values[pos - 1]
        current_token = token_values[pos]
        if previous_token is None or current_token is None:
            continue
        token_rises.append(float(current_token) - float(previous_token))
    rise_stats = _stats(token_rises, min_points, min_std)

    for pos in range(start_pos, len(embedding_values)):
        reasons = []
        current_embedding = embedding_values[pos]
        previous_embedding = embedding_values[pos - 1] if pos > 0 else None
        token_forward = token_values[pos] if pos < len(token_values) else None

        embedding_z = _z_above(current_embedding, embedding_stats)
        token_z = _z_above(token_forward, token_stats)
        token_rise = None
        token_rise_z = None
        if pos > 0 and pos < len(token_values):
            previous_token = token_values[pos - 1]
            if previous_token is not None and token_forward is not None:
                token_rise = float(token_forward) - float(previous_token)
                token_rise_z = _z_above(token_rise, rise_stats)

        anomaly_scores = []
        if token_z is not None and token_z >= high_z_threshold:
            reasons.append("adaptive_token_forward_relative_mse_high")
            anomaly_scores.append(token_z)
        if (
            token_rise_z is not None
            and token_rise > 0.0
            and token_rise_z >= rise_z_threshold
        ):
            reasons.append("adaptive_token_forward_relative_mse_rise")
            anomaly_scores.append(token_rise_z)
        if embedding_z is not None and embedding_z >= high_z_threshold:
            reasons.append("adaptive_embedding_forward_relative_mse_high")
            anomaly_scores.append(embedding_z)

        if reasons:
            embedding_mean, embedding_std = (
                embedding_stats if embedding_stats is not None else (None, None)
            )
            token_mean, token_std = (
                token_stats if token_stats is not None else (None, None)
            )
            rise_mean, rise_std = (
                rise_stats if rise_stats is not None else (None, None)
            )
            anomalies.append({
                "position": pos,
                "reasons": reasons,
                "anomaly_detection_mode": "adaptive",
                "anomaly_score": max(anomaly_scores) if anomaly_scores else None,
                "embedding_forward_relative_mse": current_embedding,
                "previous_embedding_forward_relative_mse": previous_embedding,
                "embedding_forward_relative_mse_mean": embedding_mean,
                "embedding_forward_relative_mse_std": embedding_std,
                "embedding_forward_relative_mse_cutoff": (
                    None
                    if embedding_stats is None
                    else embedding_mean + high_z_threshold * embedding_std
                ),
                "embedding_forward_relative_mse_z": embedding_z,
                "token_forward_relative_mse": token_forward,
                "token_forward_relative_mse_mean": token_mean,
                "token_forward_relative_mse_std": token_std,
                "token_forward_relative_mse_cutoff": (
                    None
                    if token_stats is None
                    else token_mean + high_z_threshold * token_std
                ),
                "token_forward_relative_mse_z": token_z,
                "relative_mse_rise": token_rise,
                "relative_mse_rise_mean": rise_mean,
                "relative_mse_rise_std": rise_std,
                "relative_mse_rise_cutoff": (
                    None
                    if rise_stats is None
                    else rise_mean + rise_z_threshold * rise_std
                ),
                "relative_mse_rise_z": token_rise_z,
            })
    return anomalies


def _find_anomalies(embedding_values, token_values, scan_pos, config):
    mode = str(
        getattr(
            config,
            "anomaly_detection_mode",
            DEFAULT_ANOMALY_DETECTION_MODE,
        )
        or DEFAULT_ANOMALY_DETECTION_MODE
    ).lower()
    if mode not in ANOMALY_DETECTION_MODES:
        raise ValueError(
            "suffix anomaly_detection_mode must be one of: {}".format(
                ", ".join(sorted(ANOMALY_DETECTION_MODES))
            )
        )
    if mode == "threshold":
        return _find_threshold_anomalies(
            embedding_values,
            token_values,
            scan_pos,
            config,
        )
    return _find_adaptive_anomalies(
        embedding_values,
        token_values,
        scan_pos,
        config,
    )


def _public_metrics(metrics):
    return {
        "accuracy": metrics["accuracy"],
        "token_forward_relative_mse_mean": (
            metrics["token_forward_relative_mse_mean"]
        ),
        "token_forward_relative_mse_max": (
            metrics["token_forward_relative_mse_max"]
        ),
        "embedding_forward_relative_mse_mean": (
            metrics["embedding_forward_relative_mse_mean"]
        ),
        "embedding_forward_relative_mse_max": (
            metrics["embedding_forward_relative_mse_max"]
        ),
        "relative_mse_anomaly_count": metrics["relative_mse_anomaly_count"],
        "first_anomaly_position": metrics["first_anomaly_position"],
        "first_anomaly_reasons": metrics["first_anomaly_reasons"],
        "metric_direction": "relative MSE is lower-is-better; max is worst",
    }


def _evaluate_state(model, embedding, tokens, target_hidden_state, attention_mask,
                    layer_id, register_layer_hooks, total_input_ids, tokenizer,
                    eval_start_pos, scan_pos, config,
                    forward_and_get_last_hidden_state):
    embedding_values = _embedding_forward_relative_mse(
        model,
        embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
    )
    token_values = _token_forward_relative_mse(
        model,
        tokens,
        target_hidden_state,
        layer_id,
        forward_and_get_last_hidden_state,
    )
    anomalies = _find_anomalies(
        embedding_values,
        token_values,
        scan_pos,
        config,
    )
    token_region = token_values[eval_start_pos:]
    embedding_region = embedding_values[eval_start_pos:]
    first_anomaly = anomalies[0] if anomalies else None
    return {
        "accuracy": _accuracy(tokens, total_input_ids, eval_start_pos),
        "text": _decode(tokenizer, tokens, eval_start_pos),
        "token_forward_relative_mse_mean": _safe_mean(token_region),
        "token_forward_relative_mse_max": _safe_max(token_region),
        "embedding_forward_relative_mse_mean": _safe_mean(embedding_region),
        "embedding_forward_relative_mse_max": _safe_max(embedding_region),
        "relative_mse_anomaly_count": len(anomalies),
        "first_anomaly_position": (
            first_anomaly.get("position") if first_anomaly else None
        ),
        "first_anomaly_reasons": (
            first_anomaly.get("reasons") if first_anomaly else []
        ),
        "_embedding_forward_relative_mse": embedding_values,
        "_token_forward_relative_mse": token_values,
        "_anomalies": anomalies,
    }


def _merge_suffix(base_embed, suffix_start, suffix_embed):
    prefix = base_embed[:, :suffix_start, :].detach()
    return torch.cat(
        (prefix, suffix_embed.to(dtype=base_embed.dtype)),
        dim=1,
    )


def _build_suffix_hidden_weights(length, mode, decay, floor, device, dtype):
    length = max(0, int(length))
    if length == 0:
        return torch.empty((1, 0), device=device, dtype=dtype)
    mode = str(mode or "uniform")
    if mode == "uniform":
        weights = torch.ones(length, device=device, dtype=dtype)
    else:
        decay = max(0.0, float(decay))
        floor = max(0.0, float(floor))
        positions = torch.arange(length, device=device, dtype=dtype)
        if mode == "tail_decay":
            positions = torch.flip(positions, dims=[0])
        elif mode != "front_decay":
            positions = torch.zeros(length, device=device, dtype=dtype)
        weights = torch.pow(
            torch.tensor(decay, device=device, dtype=dtype),
            positions,
        )
        if floor > 0.0:
            floor_tensor = torch.tensor(floor, device=device, dtype=dtype)
            weights = torch.maximum(weights, floor_tensor)
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights.view(1, length)


def _embedding_range_bound(embed_layer, device, dtype, top_k=10,
                           chunk_size=VOCAB_SEARCH_CHUNK_SIZE):
    weight = embed_layer.weight.detach()
    vocab_size = int(weight.shape[0])
    if vocab_size <= 0:
        raise ValueError("empty embedding vocabulary")
    keep_k = max(1, min(int(top_k), vocab_size))
    best = None
    with torch.no_grad():
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            chunk = weight[start:end].detach().abs().to(
                device=device,
                dtype=torch.float32,
            )
            combined = chunk if best is None else torch.cat((best, chunk), dim=0)
            current_k = min(keep_k, int(combined.shape[0]))
            best = torch.topk(combined, current_k, dim=0).values
        bound = best[-1].to(dtype=dtype).view(1, 1, -1)
    return bound


def _history_value(history, index):
    if not history:
        return None
    return history[index]


def _history_min(history):
    return min(history) if history else None


def _history_max(history):
    return max(history) if history else None


def _optimize_suffix(model, input_embed, target_hidden_state, attention_mask,
                     layer_id, register_layer_hooks, suffix_start, config,
                     embed_layer, invert_method="cosine"):
    del invert_method
    base_embed = input_embed.detach()
    original_suffix = (
        base_embed[:, suffix_start:, :].detach().clone().to(torch.float32)
    )
    suffix_len = int(original_suffix.shape[1])
    suffix_param = original_suffix.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([suffix_param], lr=config.lr)
    hidden_weights = _build_suffix_hidden_weights(
        suffix_len,
        config.hidden_weight_mode,
        config.hidden_weight_decay,
        config.hidden_weight_floor,
        suffix_param.device,
        suffix_param.dtype,
    )
    range_bound = None
    if float(config.range_weight) > 0.0:
        range_bound = _embedding_range_bound(
            embed_layer,
            suffix_param.device,
            suffix_param.dtype,
        )

    history_names = (
        "weighted_cosine_similarity",
        "weighted_cosine_loss",
        "weighted_relative_mse_loss",
        "scaled_cosine_loss_term",
        "scaled_relative_mse_loss_term",
        "joint_hidden_loss",
        "prox_loss",
        "range_loss",
        "total_loss",
    )
    histories = {name: [] for name in history_names}
    stopped_reason = "completed"

    for _ in range(max(0, int(config.epoch))):
        optimizer.zero_grad()
        current_embed = _merge_suffix(
            base_embed,
            suffix_start,
            suffix_param,
        )
        hidden_state = _forward_embedding_hidden(
            model,
            current_embed,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        current_suffix_hidden = hidden_state[:, suffix_start:, :]
        target_suffix_hidden = target_hidden[:, suffix_start:, :]
        per_pos_cosine_similarity = F.cosine_similarity(
            current_suffix_hidden.float(),
            target_suffix_hidden.float(),
            dim=-1,
        )
        per_pos_cosine_loss = 1.0 - per_pos_cosine_similarity
        per_pos_relative_mse = _relative_mse(
            current_suffix_hidden,
            target_suffix_hidden,
        )
        weight_sum = hidden_weights.sum().clamp_min(1e-12)
        weighted_cosine_similarity = (
            per_pos_cosine_similarity * hidden_weights
        ).sum() / weight_sum
        weighted_cosine_loss = (
            per_pos_cosine_loss * hidden_weights
        ).sum() / weight_sum
        weighted_relative_mse_loss = (
            per_pos_relative_mse * hidden_weights
        ).sum() / weight_sum
        scaled_cosine_loss_term = (
            float(config.cosine_loss_weight) * weighted_cosine_loss
        )
        scaled_relative_mse_loss_term = (
            float(config.relative_mse_loss_weight)
            * weighted_relative_mse_loss
        )
        joint_hidden_loss = (
            scaled_cosine_loss_term + scaled_relative_mse_loss_term
        )
        prox_loss = F.mse_loss(suffix_param, original_suffix)
        range_loss = torch.zeros(
            (),
            device=suffix_param.device,
            dtype=suffix_param.dtype,
        )
        if range_bound is not None:
            range_loss = F.relu(
                torch.abs(suffix_param) - range_bound
            ).mean()
        total_loss = (
            joint_hidden_loss
            + float(config.prox_weight) * prox_loss
            + float(config.range_weight) * range_loss
        )
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            stopped_reason = "nonfinite_loss"
            break
        current_values = {
            "weighted_cosine_similarity": weighted_cosine_similarity,
            "weighted_cosine_loss": weighted_cosine_loss,
            "weighted_relative_mse_loss": weighted_relative_mse_loss,
            "scaled_cosine_loss_term": scaled_cosine_loss_term,
            "scaled_relative_mse_loss_term": scaled_relative_mse_loss_term,
            "joint_hidden_loss": joint_hidden_loss,
            "prox_loss": prox_loss,
            "range_loss": range_loss,
            "total_loss": total_loss,
        }
        for name, value in current_values.items():
            histories[name].append(_as_float(value))
        total_loss.backward()
        optimizer.step()

    loss_summary = {
        "version": VERSION,
        "metric_direction_note": METRIC_DIRECTION_NOTE,
        "relative_mse_epsilon": RELATIVE_MSE_EPSILON,
        "cosine_loss_weight": config.cosine_loss_weight,
        "relative_mse_loss_weight": config.relative_mse_loss_weight,
        "hidden_weight_mode": config.hidden_weight_mode,
        "hidden_weight_decay": config.hidden_weight_decay,
        "hidden_weight_floor": config.hidden_weight_floor,
        "prox_weight": config.prox_weight,
        "range_weight": config.range_weight,
        "manifold_enabled": False,
        "manifold_weight": 0.0,
        "manifold_updates": 0,
        "stopped_reason": stopped_reason,
    }
    for name, history in histories.items():
        loss_summary["{}_start".format(name)] = _history_value(history, 0)
        loss_summary["{}_end".format(name)] = _history_value(history, -1)
        if name == "weighted_cosine_similarity":
            loss_summary["{}_max".format(name)] = _history_max(history)
        else:
            loss_summary["{}_min".format(name)] = _history_min(history)
    return (
        _merge_suffix(base_embed, suffix_start, suffix_param).detach(),
        loss_summary,
    )


def _changed_positions(before_tokens, after_tokens, start_pos):
    changed = []
    for pos in range(
        max(0, start_pos),
        min(len(before_tokens), len(after_tokens)),
    ):
        if int(before_tokens[pos]) != int(after_tokens[pos]):
            changed.append(pos)
    return changed


def _suffix_token_forward_relative_mse_mean(metrics, suffix_start):
    return _safe_mean(
        metrics["_token_forward_relative_mse"][suffix_start:]
    )


def _accept_mode(config):
    mode = str(
        getattr(config, "accept_mode", DEFAULT_ACCEPT_MODE)
        or DEFAULT_ACCEPT_MODE
    )
    if mode not in ACCEPT_MODES:
        raise ValueError(
            "suffix accept_mode must be one of: {}".format(
                ", ".join(sorted(ACCEPT_MODES))
            )
        )
    return mode


def _accept_candidate(current_metrics, candidate_metrics, changed_positions,
                      suffix_start, config):
    mode = _accept_mode(config)
    if mode == "always":
        if not changed_positions:
            return False, "reconstruction_unchanged", False
        return True, "always_accept", False

    current_acc = current_metrics["accuracy"]
    candidate_acc = candidate_metrics["accuracy"]
    if mode == "oracle_accuracy":
        tolerance = max(0.0, float(config.accuracy_tolerance))
        if candidate_acc > current_acc + tolerance + MIN_ACCEPTANCE_EPS:
            return True, "accuracy_improved", False
        if candidate_acc + tolerance + MIN_ACCEPTANCE_EPS < current_acc:
            return False, "accuracy_decreased", False

    if not changed_positions:
        return False, "reconstruction_unchanged", False

    anomaly_improved = (
        candidate_metrics["relative_mse_anomaly_count"]
        < current_metrics["relative_mse_anomaly_count"]
    )
    current_suffix_mean = _suffix_token_forward_relative_mse_mean(
        current_metrics,
        suffix_start,
    )
    candidate_suffix_mean = _suffix_token_forward_relative_mse_mean(
        candidate_metrics,
        suffix_start,
    )
    relative_mse_improved = (
        current_suffix_mean is not None
        and candidate_suffix_mean is not None
        and candidate_suffix_mean
        <= current_suffix_mean - config.min_relative_mse_improvement
    )
    if anomaly_improved and relative_mse_improved:
        return (
            True,
            "anomaly_count_and_suffix_relative_mse_improved",
            True,
        )
    if anomaly_improved:
        return True, "relative_mse_anomaly_count_improved", False
    if relative_mse_improved:
        return True, "suffix_relative_mse_improved", True
    if mode == "hidden_anomaly":
        return False, "no_relative_mse_anomaly_improvement", False
    return False, "no_global_improvement", False


def run_suffix_reoptimization_v1_2_2(
        model, embed_layer, optimized_embedding, target_hidden_state,
        attention_mask, layer_id, register_layer_hooks, tokenizer,
        total_input_ids, config, filter_nonascii=True, add_perplexity=True,
        top_k_ppl=10, top_k_cos=10, invert_method="cosine",
        eval_start_pos=0, embedding_top_indices=None,
        select_candidate_from_top_indices=None, get_perplexity=None,
        forward_and_get_last_hidden_state=None, log_file=None,
        baseline_gradient_trend_stats=None):
    del log_file
    if not config.enabled or config.max_rounds <= 0:
        result = {
            "name": METHOD_NAME,
            "version": VERSION,
            "enabled": bool(config.enabled),
            "skipped": True,
            "reason": "disabled" if not config.enabled else "max_rounds <= 0",
            "events": [],
            "baseline_gradient_trend_stats": baseline_gradient_trend_stats,
        }
        return optimized_embedding, result

    required_helpers = {
        "embedding_top_indices": embedding_top_indices,
        "select_candidate_from_top_indices": (
            select_candidate_from_top_indices
        ),
        "get_perplexity": get_perplexity,
        "forward_and_get_last_hidden_state": (
            forward_and_get_last_hidden_state
        ),
    }
    missing_helpers = [
        name for name, value in required_helpers.items() if value is None
    ]
    if missing_helpers:
        raise ValueError(
            "missing {} helpers: {}".format(
                METHOD_NAME,
                ", ".join(missing_helpers),
            )
        )

    scan_start = max(int(eval_start_pos), 1)
    current_embedding = optimized_embedding.detach().clone()
    before_acc, before_text, before_tokens, before_rerank = _rerank_positions(
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
    current_tokens = before_tokens
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
    scan_pos = scan_start
    attempts = 0
    max_rounds = max(0, int(config.max_rounds))
    accept_mode = _accept_mode(config)

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
        anomalies = metrics_for_scan["_anomalies"]
        if not anomalies:
            if attempts == 0:
                events.append({
                    "round": 1,
                    "max_rounds": max_rounds,
                    "triggered": False,
                    "accept_mode": accept_mode,
                    "reason": "no anomaly found",
                })
            break

        anomaly = anomalies[0]
        suffix_start = int(anomaly["position"])
        attempts += 1
        candidate_embedding, loss_summary = _optimize_suffix(
            model,
            current_embedding,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            suffix_start,
            config,
            embed_layer,
            invert_method,
        )
        (
            candidate_acc,
            candidate_text,
            candidate_tokens,
            candidate_rerank,
        ) = _rerank_positions(
            candidate_embedding,
            current_tokens,
            suffix_start,
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
            current_tokens,
            candidate_tokens,
            suffix_start,
        )
        accepted, accept_reason, relative_mse_improved = _accept_candidate(
            current_metrics,
            candidate_metrics,
            changed_positions,
            suffix_start,
            config,
        )
        event = {
            "round": attempts,
            "max_rounds": max_rounds,
            "triggered": True,
            "anomaly_position": suffix_start,
            "anomaly_reasons": anomaly.get("reasons"),
            "anomaly_detection_mode": anomaly.get(
                "anomaly_detection_mode"
            ),
            "anomaly_score": anomaly.get("anomaly_score"),
            "anomaly_embedding_forward_relative_mse": anomaly.get(
                "embedding_forward_relative_mse"
            ),
            "anomaly_token_forward_relative_mse": anomaly.get(
                "token_forward_relative_mse"
            ),
            "relative_mse_rise": anomaly.get("relative_mse_rise"),
            "before_accuracy": current_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "before_token_forward_relative_mse_mean": current_metrics[
                "token_forward_relative_mse_mean"
            ],
            "candidate_token_forward_relative_mse_mean": candidate_metrics[
                "token_forward_relative_mse_mean"
            ],
            "before_token_forward_relative_mse_max": current_metrics[
                "token_forward_relative_mse_max"
            ],
            "candidate_token_forward_relative_mse_max": candidate_metrics[
                "token_forward_relative_mse_max"
            ],
            "before_suffix_token_forward_relative_mse_mean": (
                _suffix_token_forward_relative_mse_mean(
                    current_metrics,
                    suffix_start,
                )
            ),
            "candidate_suffix_token_forward_relative_mse_mean": (
                _suffix_token_forward_relative_mse_mean(
                    candidate_metrics,
                    suffix_start,
                )
            ),
            "before_relative_mse_anomaly_count": current_metrics[
                "relative_mse_anomaly_count"
            ],
            "candidate_relative_mse_anomaly_count": candidate_metrics[
                "relative_mse_anomaly_count"
            ],
            "relative_mse_improved": relative_mse_improved,
            "candidate_rerank": candidate_rerank,
            "changed_positions": changed_positions,
            "accepted": accepted,
            "accept_mode": accept_mode,
            "accept_reason": accept_reason,
            "loss_summary": loss_summary,
        }
        events.append(event)
        if accepted:
            current_embedding = candidate_embedding
            current_tokens = candidate_tokens
            current_metrics = candidate_metrics
            for pos in changed_positions:
                all_changed_positions.add(pos)
            scan_pos = scan_start
        else:
            scan_pos = suffix_start + 1

    after_metrics = current_metrics
    triggered = any(event.get("triggered") for event in events)
    accepted_count = sum(bool(event.get("accepted")) for event in events)
    rejected_count = sum(
        bool(event.get("triggered")) and not bool(event.get("accepted"))
        for event in events
    )
    if not triggered:
        reason = "no anomaly found"
    elif accepted_count:
        reason = "accepted {} suffix round(s)".format(accepted_count)
    else:
        reason = "no suffix round accepted"
    result = {
        "name": METHOD_NAME,
        "version": VERSION,
        "enabled": True,
        "skipped": False,
        "metric_direction_note": METRIC_DIRECTION_NOTE,
        "relative_mse_epsilon": RELATIVE_MSE_EPSILON,
        "accept_mode": accept_mode,
        "pre_acc": before_metrics["accuracy"],
        "post_acc": after_metrics["accuracy"],
        "accuracy_gain": after_metrics["accuracy"] - before_metrics["accuracy"],
        "triggered": triggered,
        "accepted": accepted_count > 0,
        "accepted_round_count": accepted_count,
        "rejected_round_count": rejected_count,
        "reason": reason,
        "before": _public_metrics(before_metrics),
        "after": _public_metrics(after_metrics),
        "initial_candidate_rerank": before_rerank,
        "events": events,
        "changed_positions": sorted(all_changed_positions),
        "final_tokens": [int(item) for item in current_tokens],
        "final_text": after_metrics["text"],
        "final_accuracy": after_metrics["accuracy"],
        "baseline_gradient_trend_stats": baseline_gradient_trend_stats,
    }
    return current_embedding, result


SuffixReoptimizationConfig = SuffixReoptimizationV122Config
run_suffix_reoptimization = run_suffix_reoptimization_v1_2_2
