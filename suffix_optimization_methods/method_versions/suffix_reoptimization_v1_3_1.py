from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F


MIN_ACCEPTANCE_EPS = 1e-12
VOCAB_SEARCH_CHUNK_SIZE = 8192
METHOD_NAME = "suffix_reoptimization_v1.3.1"
VERSION = "v1.3.1"
DEFAULT_ACCEPT_MODE = "oracle_accuracy"
ACCEPT_MODES = {"oracle_accuracy", "hidden_anomaly", "always"}
DEFAULT_ANOMALY_DETECTION_MODE = "adaptive"
ANOMALY_DETECTION_MODES = {"adaptive", "threshold"}


@dataclass
class SuffixReoptimizationV131Config:
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
    range_weight: float = 0.001


def _as_float(value):
    return float(torch.as_tensor(value).detach().cpu())


def _safe_mean(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_min(values):
    if not values:
        return None
    return float(min(values))


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


def _hidden_scores_from_embedding(model, input_embed, target_hidden_state,
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
        scores = F.cosine_similarity(
            hidden_state.float(),
            target_hidden.float(),
            dim=-1,
        ).squeeze(0)
    return [_as_float(item) for item in scores]


def _hidden_scores_from_tokens(model, token_ids, target_hidden_state, layer_id,
                               forward_and_get_last_hidden_state):
    with torch.no_grad():
        hidden_state = forward_and_get_last_hidden_state(
            model,
            token_ids,
            None,
            layer_id=layer_id,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        scores = F.cosine_similarity(
            hidden_state.float(),
            target_hidden.float(),
            dim=-1,
        ).squeeze(0)
    return [_as_float(item) for item in scores]


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
        candidate_states = new_hidden_states[:, pos, :].float()
        target_state = target_hidden[:, pos, :].float()
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        cosine_similarity = F.cosine_similarity(
            candidate_states,
            target_state,
            dim=-1,
        )
        best_idx = int(torch.argmax(cosine_similarity).detach().cpu().item())
        ret_list[pos] = int(top_list[best_idx])

    acc = _accuracy(ret_list, total_input_ids, eval_start_pos)
    text = _decode(tokenizer, ret_list, eval_start_pos)
    return acc, text, ret_list


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


def _z_below(value, stats):
    if value is None or stats is None:
        return None
    mean_value, std_value = stats
    return (mean_value - float(value)) / std_value


def _z_above(value, stats):
    if value is None or stats is None:
        return None
    mean_value, std_value = stats
    return (float(value) - mean_value) / std_value


def _find_threshold_anomalies(embedding_scores, token_scores, scan_pos, config):
    anomalies = []
    min_reasons = max(1, int(config.min_anomaly_reasons))
    for pos in range(max(0, scan_pos), len(embedding_scores)):
        reasons = []
        current_hidden = embedding_scores[pos]
        previous_hidden = embedding_scores[pos - 1] if pos > 0 else None
        token_forward = token_scores[pos] if pos < len(token_scores) else None
        if (
            config.hidden_low_threshold > -1.0
            and current_hidden < config.hidden_low_threshold
        ):
            reasons.append("hidden_low")
        if (
            config.hidden_drop_threshold > 0.0
            and previous_hidden is not None
            and previous_hidden - current_hidden >= config.hidden_drop_threshold
        ):
            reasons.append("hidden_drop")
        if (
            config.token_forward_low_threshold > -1.0
            and token_forward is not None
            and token_forward < config.token_forward_low_threshold
        ):
            reasons.append("token_forward_low")
        if len(reasons) >= min_reasons:
            anomalies.append({
                "position": pos,
                "reasons": reasons,
                "anomaly_detection_mode": "threshold",
                "hidden_similarity": current_hidden,
                "previous_hidden_similarity": previous_hidden,
                "hidden_drop": (
                    None
                    if previous_hidden is None
                    else previous_hidden - current_hidden
                ),
                "token_forward_similarity": token_forward,
            })
    return anomalies


def _find_adaptive_anomalies(embedding_scores, token_scores, scan_pos, config):
    anomalies = []
    start_pos = max(0, int(scan_pos))
    min_points = max(2, int(config.adaptive_min_points))
    min_std = max(0.0, float(config.adaptive_min_std))
    low_z_threshold = max(0.0, float(config.adaptive_z_threshold))
    drop_z_threshold = max(0.0, float(config.adaptive_drop_z_threshold))

    hidden_stats = _stats(
        embedding_scores[start_pos:],
        min_points,
        min_std,
    )
    token_region = token_scores[start_pos:] if start_pos < len(token_scores) else []
    token_stats = _stats(token_region, min_points, min_std)
    token_drops = []
    for pos in range(max(1, start_pos), len(token_scores)):
        previous_token = token_scores[pos - 1]
        current_token = token_scores[pos]
        if previous_token is None or current_token is None:
            continue
        token_drops.append(float(previous_token) - float(current_token))
    drop_stats = _stats(token_drops, min_points, min_std)

    for pos in range(start_pos, len(embedding_scores)):
        reasons = []
        current_hidden = embedding_scores[pos]
        previous_hidden = embedding_scores[pos - 1] if pos > 0 else None
        token_forward = token_scores[pos] if pos < len(token_scores) else None

        hidden_z = _z_below(current_hidden, hidden_stats)
        token_z = _z_below(token_forward, token_stats)
        token_drop = None
        token_drop_z = None
        if pos > 0 and pos < len(token_scores):
            previous_token = token_scores[pos - 1]
            if previous_token is not None and token_forward is not None:
                token_drop = float(previous_token) - float(token_forward)
                token_drop_z = _z_above(token_drop, drop_stats)

        anomaly_scores = []
        if token_z is not None and token_z >= low_z_threshold:
            reasons.append("token_forward_adaptive_low")
            anomaly_scores.append(token_z)
        if (
            token_drop_z is not None
            and token_drop > 0.0
            and token_drop_z >= drop_z_threshold
        ):
            reasons.append("token_forward_drop")
            anomaly_scores.append(token_drop_z)
        if hidden_z is not None and hidden_z >= low_z_threshold:
            reasons.append("hidden_adaptive_low")
            anomaly_scores.append(hidden_z)

        if reasons:
            hidden_mean, hidden_std = (
                hidden_stats if hidden_stats is not None else (None, None)
            )
            token_mean, token_std = (
                token_stats if token_stats is not None else (None, None)
            )
            drop_mean, drop_std = (
                drop_stats if drop_stats is not None else (None, None)
            )
            anomalies.append({
                "position": pos,
                "reasons": reasons,
                "anomaly_detection_mode": "adaptive",
                "anomaly_score": max(anomaly_scores) if anomaly_scores else None,
                "hidden_similarity": current_hidden,
                "previous_hidden_similarity": previous_hidden,
                "hidden_drop": (
                    None
                    if previous_hidden is None
                    else previous_hidden - current_hidden
                ),
                "hidden_adaptive_mean": hidden_mean,
                "hidden_adaptive_std": hidden_std,
                "hidden_adaptive_cutoff": (
                    None
                    if hidden_stats is None
                    else hidden_mean - low_z_threshold * hidden_std
                ),
                "hidden_adaptive_z": hidden_z,
                "token_forward_similarity": token_forward,
                "token_forward_mean": token_mean,
                "token_forward_std": token_std,
                "token_forward_cutoff": (
                    None
                    if token_stats is None
                    else token_mean - low_z_threshold * token_std
                ),
                "token_forward_z": token_z,
                "token_forward_drop": token_drop,
                "token_forward_drop_mean": drop_mean,
                "token_forward_drop_std": drop_std,
                "token_forward_drop_cutoff": (
                    None
                    if drop_stats is None
                    else drop_mean + drop_z_threshold * drop_std
                ),
                "token_forward_drop_z": token_drop_z,
            })
    return anomalies


def _find_anomalies(embedding_scores, token_scores, scan_pos, config):
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
            embedding_scores,
            token_scores,
            scan_pos,
            config,
        )
    return _find_adaptive_anomalies(
        embedding_scores,
        token_scores,
        scan_pos,
        config,
    )


def _public_metrics(metrics):
    return {
        "accuracy": metrics["accuracy"],
        "hidden_mean": metrics["hidden_mean"],
        "hidden_min": metrics["hidden_min"],
        "embedding_hidden_mean": metrics["embedding_hidden_mean"],
        "embedding_hidden_min": metrics["embedding_hidden_min"],
        "anomaly_count": metrics["anomaly_count"],
        "first_anomaly_position": metrics["first_anomaly_position"],
        "first_anomaly_reasons": metrics["first_anomaly_reasons"],
    }


def _evaluate_state(model, embedding, tokens, target_hidden_state, attention_mask,
                    layer_id, register_layer_hooks, total_input_ids, tokenizer,
                    eval_start_pos, scan_pos, config,
                    forward_and_get_last_hidden_state):
    embedding_scores = _hidden_scores_from_embedding(
        model,
        embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
    )
    token_scores = _hidden_scores_from_tokens(
        model,
        tokens,
        target_hidden_state,
        layer_id,
        forward_and_get_last_hidden_state,
    )
    anomalies = _find_anomalies(
        embedding_scores,
        token_scores,
        scan_pos,
        config,
    )
    scored_region = token_scores[eval_start_pos:]
    embedding_region = embedding_scores[eval_start_pos:]
    first_anomaly = anomalies[0] if anomalies else None
    return {
        "accuracy": _accuracy(tokens, total_input_ids, eval_start_pos),
        "text": _decode(tokenizer, tokens, eval_start_pos),
        "hidden_mean": _safe_mean(scored_region),
        "hidden_min": _safe_min(scored_region),
        "embedding_hidden_mean": _safe_mean(embedding_region),
        "embedding_hidden_min": _safe_min(embedding_region),
        "anomaly_count": len(anomalies),
        "first_anomaly_position": (
            first_anomaly.get("position") if first_anomaly else None
        ),
        "first_anomaly_reasons": (
            first_anomaly.get("reasons") if first_anomaly else []
        ),
        "_embedding_scores": embedding_scores,
        "_token_scores": token_scores,
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


def _build_anchored_base_embedding(current_embedding, current_tokens,
                                   suffix_start, eval_start_pos, embed_layer):
    sequence_length = int(current_embedding.shape[1])
    if len(current_tokens) != sequence_length:
        raise ValueError(
            "current token length must equal current embedding sequence length"
        )
    suffix_start = int(suffix_start)
    eval_start_pos = int(eval_start_pos)
    fixed_prefix = current_embedding[:, :eval_start_pos, :].detach()
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
        anchored_prefix = current_embedding[
            :, eval_start_pos:suffix_start, :
        ].detach()
    if int(anchored_prefix.shape[1]) != suffix_start - eval_start_pos:
        raise ValueError("anchored prefix length does not match suffix start")
    continuous_suffix = current_embedding[:, suffix_start:, :].detach()
    anchored_base = torch.cat(
        (fixed_prefix, anchored_prefix, continuous_suffix),
        dim=1,
    )
    if int(anchored_base.shape[1]) != sequence_length:
        raise ValueError("anchored embedding sequence length changed")
    return anchored_base.detach()


def _weighted_similarity_from_scores(scores, suffix_start, hidden_weights):
    suffix_scores = scores[suffix_start:]
    if not suffix_scores:
        return None
    score_tensor = torch.tensor(
        suffix_scores,
        device=hidden_weights.device,
        dtype=hidden_weights.dtype,
    ).view(1, -1)
    return _as_float(
        (score_tensor * hidden_weights).sum()
        / hidden_weights.sum().clamp_min(1e-12)
    )


def _loss_summary_value(history, index):
    if not history:
        return None
    return history[index]


def _optimize_suffix(model, input_embed, current_tokens, target_hidden_state,
                     attention_mask, layer_id, register_layer_hooks,
                     suffix_start, eval_start_pos, pre_anchor_embedding_scores,
                     config, embed_layer, invert_method="cosine"):
    del invert_method
    anchored_base = _build_anchored_base_embedding(
        input_embed,
        current_tokens,
        suffix_start,
        eval_start_pos,
        embed_layer,
    )
    original_suffix = (
        input_embed[:, suffix_start:, :].detach().clone().to(torch.float32)
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

    pre_anchor_similarity = _weighted_similarity_from_scores(
        pre_anchor_embedding_scores,
        suffix_start,
        hidden_weights,
    )
    post_anchor_similarity = None
    histories = {
        "loss": [],
        "cosine_loss": [],
        "prox_loss": [],
        "range_loss": [],
    }
    stopped_reason = "completed"

    for epoch_idx in range(max(0, int(config.epoch))):
        optimizer.zero_grad()
        current_embed = _merge_suffix(
            anchored_base,
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
        per_pos_cosine_similarity = F.cosine_similarity(
            hidden_state[:, suffix_start:, :].float(),
            target_hidden[:, suffix_start:, :].float(),
            dim=-1,
        )
        weight_sum = hidden_weights.sum().clamp_min(1e-12)
        weighted_cosine_similarity = (
            per_pos_cosine_similarity * hidden_weights
        ).sum() / weight_sum
        weighted_cosine_loss = 1.0 - weighted_cosine_similarity
        if epoch_idx == 0:
            post_anchor_similarity = _as_float(
                weighted_cosine_similarity
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
        loss = (
            weighted_cosine_loss
            + float(config.prox_weight) * prox_loss
            + float(config.range_weight) * range_loss
        )
        if not bool(torch.isfinite(loss).detach().cpu()):
            stopped_reason = "nonfinite_loss"
            break
        histories["loss"].append(_as_float(loss))
        histories["cosine_loss"].append(_as_float(weighted_cosine_loss))
        histories["prox_loss"].append(_as_float(prox_loss))
        histories["range_loss"].append(_as_float(range_loss))
        loss.backward()
        optimizer.step()

    pre_anchor_loss = (
        None
        if pre_anchor_similarity is None
        else 1.0 - pre_anchor_similarity
    )
    post_anchor_loss = (
        None
        if post_anchor_similarity is None
        else 1.0 - post_anchor_similarity
    )
    optimization_summary = {
        "version": VERSION,
        "hidden_weight_mode": config.hidden_weight_mode,
        "hidden_weight_decay": config.hidden_weight_decay,
        "hidden_weight_floor": config.hidden_weight_floor,
        "prox_weight": config.prox_weight,
        "range_weight": config.range_weight,
        "manifold_enabled": False,
        "manifold_weight": 0.0,
        "manifold_updates": 0,
        "anchor_count": max(0, suffix_start - eval_start_pos),
        "anchor_token_source": "current_accepted_reconstruction",
        "anchor_uses_model_input_embedding_layer": (
            suffix_start > eval_start_pos
        ),
        "anchor_uses_ground_truth_reconstruction": False,
        "pre_anchor_weighted_cosine_similarity": pre_anchor_similarity,
        "post_anchor_weighted_cosine_similarity": post_anchor_similarity,
        "anchor_weighted_cosine_similarity_delta": (
            None
            if pre_anchor_similarity is None
            or post_anchor_similarity is None
            else post_anchor_similarity - pre_anchor_similarity
        ),
        "pre_anchor_weighted_cosine_loss": pre_anchor_loss,
        "post_anchor_weighted_cosine_loss": post_anchor_loss,
        "anchor_weighted_cosine_loss_delta": (
            None
            if pre_anchor_loss is None or post_anchor_loss is None
            else post_anchor_loss - pre_anchor_loss
        ),
        "loss_start": _loss_summary_value(histories["loss"], 0),
        "loss_end": _loss_summary_value(histories["loss"], -1),
        "loss_min": min(histories["loss"]) if histories["loss"] else None,
        "cosine_loss_start": _loss_summary_value(
            histories["cosine_loss"],
            0,
        ),
        "cosine_loss_end": _loss_summary_value(
            histories["cosine_loss"],
            -1,
        ),
        "cosine_loss_min": (
            min(histories["cosine_loss"])
            if histories["cosine_loss"] else None
        ),
        "prox_loss_start": _loss_summary_value(histories["prox_loss"], 0),
        "prox_loss_end": _loss_summary_value(histories["prox_loss"], -1),
        "range_loss_start": _loss_summary_value(histories["range_loss"], 0),
        "range_loss_end": _loss_summary_value(histories["range_loss"], -1),
        "stopped_reason": stopped_reason,
    }
    return (
        _merge_suffix(
            anchored_base,
            suffix_start,
            suffix_param,
        ).detach(),
        optimization_summary,
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


def _suffix_mean(metrics, suffix_start):
    return _safe_mean(metrics["_token_scores"][suffix_start:])


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
            return False, "reconstruction_unchanged"
        return True, "always_accept"

    current_acc = current_metrics["accuracy"]
    candidate_acc = candidate_metrics["accuracy"]
    if mode == "oracle_accuracy":
        tolerance = max(0.0, float(config.accuracy_tolerance))
        if candidate_acc > current_acc + tolerance + MIN_ACCEPTANCE_EPS:
            return True, "accuracy_improved"
        if candidate_acc + tolerance + MIN_ACCEPTANCE_EPS < current_acc:
            return False, "accuracy_decreased"

    if not changed_positions:
        return False, "reconstruction_unchanged"

    anomaly_improved = (
        candidate_metrics["anomaly_count"] < current_metrics["anomaly_count"]
    )
    current_suffix_mean = _suffix_mean(current_metrics, suffix_start)
    candidate_suffix_mean = _suffix_mean(candidate_metrics, suffix_start)
    suffix_hidden_improved = (
        current_suffix_mean is not None
        and candidate_suffix_mean is not None
        and candidate_suffix_mean >= current_suffix_mean + config.min_hidden_delta
    )
    if anomaly_improved and suffix_hidden_improved:
        return True, "anomaly_count_and_suffix_hidden_improved"
    if anomaly_improved:
        return True, "anomaly_count_improved"
    if suffix_hidden_improved:
        return True, "suffix_hidden_improved"
    if mode == "hidden_anomaly":
        return False, "no_hidden_anomaly_improvement"
    return False, "no_global_improvement"


def run_suffix_reoptimization_v1_3_1(
        model, embed_layer, optimized_embedding, target_hidden_state,
        attention_mask, layer_id, register_layer_hooks, tokenizer,
        total_input_ids, config, filter_nonascii=True, add_perplexity=True,
        top_k_ppl=10, top_k_cos=10, invert_method="cosine",
        eval_start_pos=0, embedding_top_indices=None,
        select_candidate_from_top_indices=None, get_perplexity=None,
        forward_and_get_last_hidden_state=None, log_file=None):
    del log_file
    if not config.enabled or config.max_rounds <= 0:
        result = {
            "name": METHOD_NAME,
            "version": VERSION,
            "enabled": bool(config.enabled),
            "skipped": True,
            "manifold_enabled": False,
            "manifold_weight": 0.0,
            "manifold_updates": 0,
            "reason": "disabled" if not config.enabled else "max_rounds <= 0",
            "events": [],
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
        candidate_embedding, optimization_summary = _optimize_suffix(
            model,
            current_embedding,
            current_tokens,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            suffix_start,
            eval_start_pos,
            metrics_for_scan["_embedding_scores"],
            config,
            embed_layer,
            invert_method,
        )
        candidate_acc, candidate_text, candidate_tokens = _rerank_positions(
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
        accepted, accept_reason = _accept_candidate(
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
            "anomaly_hidden_similarity": anomaly.get("hidden_similarity"),
            "anomaly_previous_hidden_similarity": anomaly.get(
                "previous_hidden_similarity"
            ),
            "anomaly_hidden_drop": anomaly.get("hidden_drop"),
            "anomaly_token_forward_similarity": anomaly.get(
                "token_forward_similarity"
            ),
            "before_accuracy": current_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "before_hidden_mean": current_metrics["hidden_mean"],
            "candidate_hidden_mean": candidate_metrics["hidden_mean"],
            "before_hidden_min": current_metrics["hidden_min"],
            "candidate_hidden_min": candidate_metrics["hidden_min"],
            "before_suffix_hidden_mean": _suffix_mean(
                current_metrics,
                suffix_start,
            ),
            "candidate_suffix_hidden_mean": _suffix_mean(
                candidate_metrics,
                suffix_start,
            ),
            "before_anomaly_count": current_metrics["anomaly_count"],
            "candidate_anomaly_count": candidate_metrics["anomaly_count"],
            "changed_positions": changed_positions,
            "accepted": accepted,
            "accept_mode": accept_mode,
            "accept_reason": accept_reason,
            "anchor": optimization_summary,
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
        "manifold_enabled": False,
        "manifold_weight": 0.0,
        "manifold_updates": 0,
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
        "events": events,
        "changed_positions": sorted(all_changed_positions),
        "final_tokens": [int(item) for item in current_tokens],
        "final_text": after_metrics["text"],
        "final_accuracy": after_metrics["accuracy"],
    }
    return current_embedding, result


SuffixReoptimizationConfig = SuffixReoptimizationV131Config
run_suffix_reoptimization = run_suffix_reoptimization_v1_3_1
