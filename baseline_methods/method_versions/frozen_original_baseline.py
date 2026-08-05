import copy
import time

import torch
import torch.nn.functional as F


METHOD_NAME = "frozen_original_baseline"
VERSION = "frozen-original-v1"
EMBEDDING_SEARCH_CHUNK_SIZE = 8192

def _target_ids(total_input_ids):
    return [int(item) for item in total_input_ids[0].detach().cpu().tolist()]


def _accuracy(token_ids, total_input_ids, eval_start_pos):
    targets = _target_ids(total_input_ids)
    start = min(max(int(eval_start_pos), 0), len(targets))
    if start >= len(targets):
        return 0.0
    correct = sum(
        int(token_ids[position]) == targets[position]
        for position in range(start, len(targets))
    )
    return correct / (len(targets) - start)


def _decode(tokenizer, token_ids, eval_start_pos):
    return tokenizer.decode(torch.tensor(token_ids[eval_start_pos:]))


def _forward_embedding_hidden(model, input_embed, attention_mask, layer_id,
                              register_layer_hooks):
    hidden_state_list = []

    def forward_hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden_state_list.extend(output)
        else:
            hidden_state_list.append(output)

    handles = register_layer_hooks(
        model,
        layer_id,
        forward_hook,
        up_to=False,
    )
    try:
        model(inputs_embeds=input_embed, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()
    if not hidden_state_list:
        raise ValueError(
            "no hidden states collected for layer {}".format(layer_id)
        )
    return hidden_state_list[0]


def _embedding_top_indices_cosine(
        embed,
        embed_layer,
        top_k,
        chunk_size=EMBEDDING_SEARCH_CHUNK_SIZE):
    """Frozen copy of the original chunked cosine vocabulary lookup."""
    weight = embed_layer.weight.detach()
    vocab_size = int(weight.shape[0])
    keep_k = max(1, min(int(top_k), vocab_size))
    embed_cpu = embed.detach().to("cpu", dtype=torch.float32)
    best_scores = None
    best_indices = None

    with torch.no_grad():
        for start in range(0, vocab_size, int(chunk_size)):
            end = min(start + int(chunk_size), vocab_size)
            weight_chunk = weight[start:end].detach().to(
                "cpu",
                dtype=torch.float32,
            )
            scores = F.cosine_similarity(
                embed_cpu.unsqueeze(0),
                weight_chunk,
                dim=-1,
            )
            chunk_k = min(keep_k, int(scores.numel()))
            chunk_scores, chunk_indices = torch.topk(scores, chunk_k)
            chunk_indices = chunk_indices + start
            if best_scores is None:
                best_scores = chunk_scores
                best_indices = chunk_indices
                continue
            combined_scores = torch.cat((best_scores, chunk_scores))
            combined_indices = torch.cat((best_indices, chunk_indices))
            current_k = min(keep_k, int(combined_scores.numel()))
            best_scores, best_positions = torch.topk(
                combined_scores,
                current_k,
            )
            best_indices = combined_indices[best_positions]
    return best_indices


def _select_candidate_from_top_indices(
        top_indices,
        tokenizer,
        filter_nonascii=True):
    """Frozen copy of the original first-valid-token/fallback behavior."""
    special_token_ids = set(tokenizer.all_special_ids)
    top_ids = [int(item.detach().cpu()) for item in top_indices]
    for token_id in top_ids:
        if token_id in special_token_ids:
            continue
        if filter_nonascii and not tokenizer.decode([token_id]).isascii():
            continue
        return token_id, top_ids
    return top_ids[0], top_ids


def _invert_embedding_top1(
        embedding,
        tokenizer,
        embed_layer,
        total_input_ids,
        filter_nonascii,
        top_k,
        eval_start_pos):
    token_ids = []
    for embed in embedding.squeeze(0):
        top_indices = _embedding_top_indices_cosine(
            embed,
            embed_layer,
            top_k,
        )
        token_id, _ = _select_candidate_from_top_indices(
            top_indices,
            tokenizer,
            filter_nonascii,
        )
        token_ids.append(token_id)
    return (
        _accuracy(token_ids, total_input_ids, eval_start_pos),
        _decode(tokenizer, token_ids, eval_start_pos),
        token_ids,
    )


def _optimize_continuous_embedding(
        model,
        initial_optimizable_embedding,
        prefix_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        tokenizer,
        embed_layer,
        total_input_ids,
        right_range,
        lr,
        epoch,
        alpha,
        clip,
        filter_nonascii,
        top_k,
        eval_start_pos):
    """Frozen copy of the original continuous cosine optimization path."""
    optimizable = initial_optimizable_embedding
    target_hidden_state = target_hidden_state.detach()
    weight_mask = torch.ones(
        int(total_input_ids.shape[1]),
        device=optimizable.device,
        dtype=torch.float16,
    )
    optimization_result = {}
    loss_history = []
    cosine_objective_history = []
    nan_detected = False
    stopped_reason = "completed"
    start_time = time.time()
    completed_steps = 0

    for epoch_index in range(max(0, int(epoch))):
        if clip:
            with torch.no_grad():
                optimizable = torch.clip(optimizable, -0.2, 0.2)
        optimizable = optimizable.requires_grad_(True)
        optimizer = torch.optim.SGD([optimizable], lr=float(lr))
        current_embedding = (
            torch.cat((prefix_embedding, optimizable), dim=1)
            if prefix_embedding is not None
            else optimizable
        )
        hidden_state = _forward_embedding_hidden(
            model,
            current_embedding,
            attention_mask,
            layer_id,
            register_layer_hooks,
        )
        target_hidden = target_hidden_state.to(hidden_state.device)
        cosine_similarity = F.cosine_similarity(
            hidden_state.float(),
            target_hidden.float(),
            dim=-1,
        )
        optimizer.zero_grad()
        weighted_negative_cosine = (
            -cosine_similarity * weight_mask.to(cosine_similarity.device)
        ).sum()
        range_loss = F.relu(
            torch.abs(current_embedding) - right_range
        ).sum()
        total_loss = weighted_negative_cosine + float(alpha) * range_loss
        if not bool(torch.isfinite(total_loss).detach().cpu()):
            nan_detected = True
            stopped_reason = "nonfinite_loss"
            break
        total_loss.backward(inputs=[optimizable])
        if bool(torch.isnan(cosine_similarity).any().detach().cpu()):
            nan_detected = True
            stopped_reason = "nan_cosine_similarity"
            break
        optimizer.step()
        completed_steps += 1
        loss_history.append(float(range_loss.detach().cpu()))
        cosine_objective_history.append(
            float(weighted_negative_cosine.detach().cpu())
        )

        if epoch_index == int(epoch) - 1:
            final_embedding = (
                torch.cat((prefix_embedding, optimizable), dim=1)
                if prefix_embedding is not None
                else optimizable
            )
            top1_accuracy, top1_text, _ = _invert_embedding_top1(
                final_embedding,
                tokenizer,
                embed_layer,
                total_input_ids,
                filter_nonascii,
                top_k,
                eval_start_pos,
            )
            optimization_result = {
                "epoch": epoch_index,
                "acc": top1_accuracy,
                "cos_sim_mean": float(
                    cosine_similarity.mean().detach().cpu()
                ),
                "relu_loss": float(range_loss.detach().cpu()),
                "elapsed_seconds": time.time() - start_time,
                "tokens": top1_text,
            }

    final_embedding = (
        torch.cat((prefix_embedding, optimizable), dim=1)
        if prefix_embedding is not None
        else optimizable
    ).detach()
    summary = {
        "optimizer": "SGD",
        "optimizer_recreated_each_epoch": True,
        "lr": float(lr),
        "epoch": int(epoch),
        "completed_steps": completed_steps,
        "clip": bool(clip),
        "clip_range": 0.2 if clip else None,
        "range_weight": float(alpha),
        "stopped_reason": stopped_reason,
        "nan_detected": nan_detected,
        "weighted_negative_cosine_start": (
            cosine_objective_history[0]
            if cosine_objective_history else None
        ),
        "weighted_negative_cosine_end": (
            cosine_objective_history[-1]
            if cosine_objective_history else None
        ),
        "range_loss_start": loss_history[0] if loss_history else None,
        "range_loss_end": loss_history[-1] if loss_history else None,
    }
    return final_embedding, optimization_result, summary


def _rerank_hidden_cosine(
        optimized_embedding,
        tokenizer,
        model,
        embed_layer,
        target_hidden_state,
        total_input_ids,
        layer_id,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        eval_start_pos,
        get_perplexity,
        forward_and_get_last_hidden_state):
    """Frozen copy of the original embedding/PPL/hidden-cosine reranker."""
    ret_list = []
    ret_top_k = []
    if int(top_k_cos) == 0:
        for _ in optimized_embedding.squeeze(0):
            ret_top_k.append([0])
            ret_list.append(0)
    else:
        for embed in optimized_embedding.squeeze(0):
            top_indices = _embedding_top_indices_cosine(
                embed,
                embed_layer,
                top_k_cos,
            )
            token_id, top_ids = _select_candidate_from_top_indices(
                top_indices,
                tokenizer,
                filter_nonascii,
            )
            ret_top_k.append(top_ids)
            ret_list.append(token_id)

    target_ids = _target_ids(total_input_ids)
    for position in range(min(int(eval_start_pos), len(ret_list))):
        ret_list[position] = target_ids[position]
        ret_top_k[position] = [ret_list[position]]

    diagnostics = []
    for position, top_list in enumerate(ret_top_k):
        top_list = list(top_list)
        if position > 0 and add_perplexity:
            _, topk_ids = get_perplexity(
                copy.deepcopy(ret_list[:position]),
                model,
                layer_id=layer_id,
                top_k=top_k_ppl,
            )
            top_list += [int(item) for item in topk_ids.tolist()]
        replaced_sequences = []
        for token_id in top_list:
            replaced = copy.deepcopy(ret_list)
            replaced[position] = int(token_id)
            replaced_sequences.append(replaced)
        hidden_states = forward_and_get_last_hidden_state(
            model,
            replaced_sequences,
            None,
            layer_id=layer_id,
        )
        target_hidden = target_hidden_state.to(hidden_states.device)
        candidate_states = hidden_states[:, position, :].float()
        target_state = target_hidden[:, position, :].float()
        if target_state.shape[0] == 1 and candidate_states.shape[0] != 1:
            target_state = target_state.expand(candidate_states.shape[0], -1)
        cosine = F.cosine_similarity(
            candidate_states,
            target_state,
            dim=-1,
        )
        best_index = int(torch.argmax(cosine).detach().cpu())
        ret_list[position] = int(top_list[best_index])
        diagnostics.append({
            "position": position,
            "selected_token_id": ret_list[position],
            "candidate_token_ids": [int(item) for item in top_list],
            "candidate_hidden_cosine": [
                float(item) for item in cosine.detach().cpu().tolist()
            ],
        })
    return (
        _accuracy(ret_list, total_input_ids, eval_start_pos),
        _decode(tokenizer, ret_list, eval_start_pos),
        ret_list,
        diagnostics,
    )


def run_frozen_original_baseline(
        model,
        embed_layer,
        initial_optimizable_embedding,
        prefix_embedding,
        target_hidden_state,
        attention_mask,
        layer_id,
        register_layer_hooks,
        tokenizer,
        total_input_ids,
        right_range,
        lr,
        epoch,
        alpha,
        clip,
        init_method,
        init_param,
        filter_nonascii=True,
        add_perplexity=True,
        top_k_ppl=10,
        top_k_cos=10,
        eval_start_pos=0,
        get_perplexity=None,
        forward_and_get_last_hidden_state=None,
        log_file=None):
    del log_file
    if get_perplexity is None or forward_and_get_last_hidden_state is None:
        raise ValueError("frozen original baseline helpers are required")

    optimized_embedding, optimization_result, optimization_summary = (
        _optimize_continuous_embedding(
            model,
            initial_optimizable_embedding,
            prefix_embedding,
            target_hidden_state,
            attention_mask,
            layer_id,
            register_layer_hooks,
            tokenizer,
            embed_layer,
            total_input_ids,
            right_range,
            lr,
            epoch,
            alpha,
            clip,
            filter_nonascii,
            top_k_cos,
            eval_start_pos,
        )
    )
    accuracy, text, token_ids, rerank_diagnostics = _rerank_hidden_cosine(
        optimized_embedding,
        tokenizer,
        model,
        embed_layer,
        target_hidden_state,
        total_input_ids,
        layer_id,
        filter_nonascii,
        add_perplexity,
        top_k_ppl,
        top_k_cos,
        eval_start_pos,
        get_perplexity,
        forward_and_get_last_hidden_state,
    )
    result = {
        "method": METHOD_NAME,
        "name": METHOD_NAME,
        "version": VERSION,
        "enabled": True,
        "skipped": False,
        "objective": "weighted_negative_hidden_cosine_plus_range",
        "vocab_metric": "embedding_cosine",
        "candidate_rerank_metric": "hidden_cosine",
        "optimizer": "SGD",
        "lr": float(lr),
        "epoch": int(epoch),
        "init_method": str(init_method),
        "init_param": float(init_param),
        "clip": bool(clip),
        "accuracy": accuracy,
        "final_accuracy": accuracy,
        "final_text": text,
        "final_tokens": token_ids,
        "optimization_result": optimization_result,
        "optimization_summary": optimization_summary,
        "candidate_rerank": rerank_diagnostics,
    }
    return optimized_embedding, result
