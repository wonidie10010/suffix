from dataclasses import dataclass
import copy

import torch
import torch.nn.functional as F

from experiment_outputs import format_metric, log_kv, log_line, log_section


MIN_ACCEPTANCE_EPS = 1e-12
METHOD_NAME = "suffix_reoptimization_v1.0"
DEFAULT_ACCEPT_MODE = "oracle_accuracy"
ACCEPT_MODES = {"oracle_accuracy", "hidden_anomaly", "always"}


@dataclass
class SuffixReoptimizationV10Config:
    enabled: bool = False
    log_enabled: bool = True
    max_rounds: int = 2
    epoch: int = 50
    lr: float = 0.03
    reg_weight: float = 0.02
    hidden_low_threshold: float = 0.50
    hidden_drop_threshold: float = 0.15
    token_forward_low_threshold: float = 0.50
    min_anomaly_reasons: int = 2
    min_hidden_delta: float = 0.005
    accuracy_tolerance: float = 0.0
    accept_mode: str = DEFAULT_ACCEPT_MODE


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


def _forward_embedding_hidden(model, input_embed, attention_mask, layer_id, register_layer_hooks):
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


def _hidden_scores_from_embedding(model, input_embed, target_hidden_state, attention_mask,
                                  layer_id, register_layer_hooks):
    with torch.no_grad():
        hidden_state = _forward_embedding_hidden(
            model,
            input_embed,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden_state = target_hidden_state.to(hidden_state.device)
        scores = F.cosine_similarity(
            hidden_state.type(torch.float32),
            target_hidden_state.type(torch.float32),
            dim=-1,
        ).squeeze(0)
    return [_as_float(item) for item in scores]


def _hidden_scores_from_tokens(model, token_ids, target_hidden_state, layer_id,
                               forward_and_get_last_hidden_state):
    with torch.no_grad():
        hidden_state = forward_and_get_last_hidden_state(model, token_ids, None, layer_id=layer_id)
        target_hidden_state = target_hidden_state.to(hidden_state.device)
        scores = F.cosine_similarity(
            hidden_state.type(torch.float32),
            target_hidden_state.type(torch.float32),
            dim=-1,
        ).squeeze(0)
    return [_as_float(item) for item in scores]


def _candidate_token_ids(embed, embed_layer, top_k_cos, invert_method, tokenizer,
                         filter_nonascii, embedding_top_indices,
                         select_candidate_from_top_indices):
    if top_k_cos == 0:
        return []
    top_indices = embedding_top_indices(embed, embed_layer, top_k_cos, invert_method)
    _, top_ids = select_candidate_from_top_indices(top_indices, tokenizer, filter_nonascii)
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
            _, topk_ids = get_perplexity(ret_list[:pos], model, layer_id=layer_id, top_k=top_k_ppl)
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
        target_hidden_state = target_hidden_state.to(new_hidden_states.device)
        candidate_states = new_hidden_states[:, pos, :].type(torch.float32)
        target_state = target_hidden_state[:, pos, :].type(torch.float32)
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        cos_sim_list = F.cosine_similarity(candidate_states, target_state, dim=-1)
        best_idx = int(torch.argmax(cos_sim_list).detach().cpu().item())
        ret_list[pos] = int(top_list[best_idx])

    acc = _accuracy(ret_list, total_input_ids, eval_start_pos)
    text = _decode(tokenizer, ret_list, eval_start_pos)
    return acc, text, ret_list


def _find_anomalies(embedding_scores, token_scores, scan_pos, config):
    anomalies = []
    min_reasons = max(1, int(config.min_anomaly_reasons))
    for pos in range(max(0, scan_pos), len(embedding_scores)):
        reasons = []
        current_hidden = embedding_scores[pos]
        previous_hidden = embedding_scores[pos - 1] if pos > 0 else None
        token_forward = token_scores[pos] if pos < len(token_scores) else None
        if config.hidden_low_threshold > -1.0 and current_hidden < config.hidden_low_threshold:
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
                "hidden_similarity": current_hidden,
                "previous_hidden_similarity": previous_hidden,
                "hidden_drop": None if previous_hidden is None else previous_hidden - current_hidden,
                "token_forward_similarity": token_forward,
            })
    return anomalies


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


def _evaluate_state(model, embedding, tokens, target_hidden_state, attention_mask, layer_id,
                    register_layer_hooks, total_input_ids, tokenizer, eval_start_pos,
                    scan_pos, config, forward_and_get_last_hidden_state):
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
    anomalies = _find_anomalies(embedding_scores, token_scores, scan_pos, config)
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
        "first_anomaly_position": first_anomaly.get("position") if first_anomaly else None,
        "first_anomaly_reasons": first_anomaly.get("reasons") if first_anomaly else [],
        "_embedding_scores": embedding_scores,
        "_token_scores": token_scores,
        "_anomalies": anomalies,
    }


def _merge_suffix(base_embed, suffix_start, suffix_embed):
    prefix = base_embed[:, :suffix_start, :].detach()
    return torch.cat((prefix, suffix_embed.to(dtype=base_embed.dtype)), dim=1)


def _optimize_suffix(model, input_embed, target_hidden_state, attention_mask, layer_id,
                     register_layer_hooks, suffix_start, config):
    base_embed = input_embed.detach()
    original_suffix = base_embed[:, suffix_start:, :].detach().clone().to(torch.float32)
    suffix_param = original_suffix.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([suffix_param], lr=config.lr)

    for _ in range(max(0, int(config.epoch))):
        optimizer.zero_grad()
        current_embed = _merge_suffix(base_embed, suffix_start, suffix_param)
        hidden_state = _forward_embedding_hidden(
            model,
            current_embed,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden_state = target_hidden_state.to(hidden_state.device)
        hidden_loss = 1.0 - F.cosine_similarity(
            hidden_state[:, suffix_start:, :].type(torch.float32),
            target_hidden_state[:, suffix_start:, :].type(torch.float32),
            dim=-1,
        ).mean()
        reg_loss = F.mse_loss(suffix_param, original_suffix)
        loss = hidden_loss + config.reg_weight * reg_loss
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()

    return _merge_suffix(base_embed, suffix_start, suffix_param).detach()


def _changed_positions(before_tokens, after_tokens, start_pos):
    changed = []
    for pos in range(max(0, start_pos), min(len(before_tokens), len(after_tokens))):
        if int(before_tokens[pos]) != int(after_tokens[pos]):
            changed.append(pos)
    return changed


def _suffix_mean(metrics, suffix_start):
    return _safe_mean(metrics["_token_scores"][suffix_start:])


def _accept_mode(config):
    mode = str(getattr(config, "accept_mode", DEFAULT_ACCEPT_MODE) or DEFAULT_ACCEPT_MODE)
    if mode not in ACCEPT_MODES:
        raise ValueError("suffix accept_mode must be one of: {}".format(", ".join(sorted(ACCEPT_MODES))))
    return mode


def _accept_candidate(current_metrics, candidate_metrics, changed_positions, suffix_start, config):
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

    anomaly_improved = candidate_metrics["anomaly_count"] < current_metrics["anomaly_count"]
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


def _log_result(f, result):
    log_section(f, METHOD_NAME)
    before = result.get("before", {})
    after = result.get("after", {})
    log_kv(f, "enabled", result.get("enabled"))
    log_kv(f, "accept_mode", result.get("accept_mode"))
    log_kv(f, "before_accuracy", format_metric(before.get("accuracy")))
    log_kv(f, "before_hidden_mean", format_metric(before.get("hidden_mean")))
    log_kv(f, "before_hidden_min", format_metric(before.get("hidden_min")))
    log_kv(f, "before_anomaly_count", before.get("anomaly_count"))
    log_kv(f, "before_first_anomaly_position", before.get("first_anomaly_position"))
    log_kv(f, "before_first_anomaly_reasons", before.get("first_anomaly_reasons"))
    for event in result.get("events", []):
        if event.get("triggered"):
            log_line(f, (
                "  suffix_round {}/{}: anomaly_pos={}, reasons={}, accept_mode={}, accepted={}, "
                "changed_positions={}, before_acc={}, candidate_acc={}, "
                "before_hidden_mean={}, candidate_hidden_mean={}, "
                "before_hidden_min={}, candidate_hidden_min={}, "
                "before_anomalies={}, candidate_anomalies={}, accept_reason={}"
            ).format(
                event.get("round"),
                event.get("max_rounds"),
                event.get("anomaly_position"),
                event.get("anomaly_reasons"),
                event.get("accept_mode"),
                event.get("accepted"),
                event.get("changed_positions"),
                format_metric(event.get("before_accuracy")),
                format_metric(event.get("candidate_accuracy")),
                format_metric(event.get("before_hidden_mean")),
                format_metric(event.get("candidate_hidden_mean")),
                format_metric(event.get("before_hidden_min")),
                format_metric(event.get("candidate_hidden_min")),
                event.get("before_anomaly_count"),
                event.get("candidate_anomaly_count"),
                event.get("accept_reason"),
            ), console=False)
        else:
            log_line(f, (
                "  suffix_round {}/{}: triggered=false, reason={}"
            ).format(
                event.get("round"),
                event.get("max_rounds"),
                event.get("reason"),
            ), console=False)
    log_kv(f, "after_accuracy", format_metric(after.get("accuracy")))
    log_kv(f, "after_hidden_mean", format_metric(after.get("hidden_mean")))
    log_kv(f, "after_hidden_min", format_metric(after.get("hidden_min")))
    log_kv(f, "after_anomaly_count", after.get("anomaly_count"))
    log_kv(f, "after_first_anomaly_position", after.get("first_anomaly_position"))
    log_kv(f, "after_first_anomaly_reasons", after.get("first_anomaly_reasons"))
    log_kv(f, "changed_positions", result.get("changed_positions"))
    log_kv(f, "reason", result.get("reason"))


def run_suffix_reoptimization_v1_0(model, embed_layer, optimized_embedding, target_hidden_state,
                                   attention_mask, layer_id, register_layer_hooks, tokenizer,
                                   total_input_ids, config, filter_nonascii=True,
                                   add_perplexity=True, top_k_ppl=10, top_k_cos=10,
                                   invert_method="cosine", eval_start_pos=0,
                                   embedding_top_indices=None,
                                   select_candidate_from_top_indices=None,
                                   get_perplexity=None,
                                   forward_and_get_last_hidden_state=None,
                                   log_file=None):
    if not config.enabled or config.max_rounds <= 0:
        result = {
            "name": METHOD_NAME,
            "enabled": bool(config.enabled),
            "skipped": True,
            "reason": "disabled" if not config.enabled else "max_rounds <= 0",
            "events": [],
        }
        return optimized_embedding, result

    required_helpers = {
        "embedding_top_indices": embedding_top_indices,
        "select_candidate_from_top_indices": select_candidate_from_top_indices,
        "get_perplexity": get_perplexity,
        "forward_and_get_last_hidden_state": forward_and_get_last_hidden_state,
    }
    missing_helpers = [name for name, value in required_helpers.items() if value is None]
    if missing_helpers:
        raise ValueError("missing {} helpers: {}".format(METHOD_NAME, ", ".join(missing_helpers)))

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
        candidate_embedding = _optimize_suffix(
            model,
            current_embedding,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            suffix_start,
            config,
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
        changed_positions = _changed_positions(current_tokens, candidate_tokens, suffix_start)
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
            "anomaly_hidden_similarity": anomaly.get("hidden_similarity"),
            "anomaly_previous_hidden_similarity": anomaly.get("previous_hidden_similarity"),
            "anomaly_hidden_drop": anomaly.get("hidden_drop"),
            "anomaly_token_forward_similarity": anomaly.get("token_forward_similarity"),
            "before_accuracy": current_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "before_hidden_mean": current_metrics["hidden_mean"],
            "candidate_hidden_mean": candidate_metrics["hidden_mean"],
            "before_hidden_min": current_metrics["hidden_min"],
            "candidate_hidden_min": candidate_metrics["hidden_min"],
            "before_suffix_hidden_mean": _suffix_mean(current_metrics, suffix_start),
            "candidate_suffix_hidden_mean": _suffix_mean(candidate_metrics, suffix_start),
            "before_anomaly_count": current_metrics["anomaly_count"],
            "candidate_anomaly_count": candidate_metrics["anomaly_count"],
            "changed_positions": changed_positions,
            "accepted": accepted,
            "accept_mode": accept_mode,
            "accept_reason": accept_reason,
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
    accepted_count = len([event for event in events if event.get("accepted")])
    if not triggered:
        reason = "no anomaly found"
    elif accepted_count:
        reason = "accepted {} suffix round(s)".format(accepted_count)
    else:
        reason = "no suffix round accepted"
    result = {
        "name": METHOD_NAME,
        "enabled": True,
        "skipped": False,
        "accept_mode": accept_mode,
        "pre_acc": before_metrics["accuracy"],
        "post_acc": after_metrics["accuracy"],
        "triggered": triggered,
        "accepted": accepted_count > 0,
        "reason": reason,
        "before": _public_metrics(before_metrics),
        "after": _public_metrics(after_metrics),
        "events": events,
        "changed_positions": sorted(all_changed_positions),
        "final_tokens": [int(item) for item in current_tokens],
        "final_text": after_metrics["text"],
        "final_accuracy": after_metrics["accuracy"],
    }
    if config.log_enabled and log_file is not None:
        _log_result(log_file, result)
    return current_embedding, result


SuffixReoptimizationConfig = SuffixReoptimizationV10Config
run_suffix_reoptimization = run_suffix_reoptimization_v1_0
