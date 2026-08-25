import argparse
import os
import time
import pickle
import copy
import json
import sys
import hashlib

from experiment_outputs import (
    annotate_candidate_events_for_offline_evaluation,
    build_stage_accuracy,
    build_resolved_config,
    console_finish_progress,
    console_safe_text,
    console_update,
    disable_external_progress_bars,
    dump_json,
    experiment_method_directory_name,
    format_progress_bar,
    json_default,
    log_line,
    suppress_startup_noise,
    write_experiment_average_summary,
    write_experiment_sample_summary,
)


suppress_startup_noise()

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, top_k_top_p_filtering)
from accelerate import Accelerator, dispatch_model
from dataset import load_dataset_samples
from baseline_methods import (
    run_frozen_original_baseline,
)
from CGMR import (
    CGMRV10Config,
    CGMRV11Config,
    CGMRV12Config,
    collect_hidden_states_by_layer,
    resolve_effective_layers,
    run_cgmr_v1_0,
    run_cgmr_v1_1,
    run_cgmr_v1_2,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_0 import (
    SuffixReoptimizationV10Config,
    run_suffix_reoptimization_v1_0,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_1 import (
    SuffixReoptimizationV11Config,
    run_suffix_reoptimization_v1_1,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2 import (
    SuffixReoptimizationV12Config,
    run_suffix_reoptimization_v1_2,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2_1 import (
    SuffixReoptimizationV121Config,
    run_suffix_reoptimization_v1_2_1,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2_2 import (
    BaselineGradientTrendTracker,
    SuffixReoptimizationV122Config,
    run_suffix_reoptimization_v1_2_2,
)
from suffix_optimization_methods.method_versions.suffix_v1_2_3 import (
    SuffixV123Config,
    run_suffix_v1_2_3,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_3 import (
    SuffixReoptimizationV13Config,
    run_suffix_reoptimization_v1_3,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_3_1 import (
    SuffixReoptimizationV131Config,
    run_suffix_reoptimization_v1_3_1,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4 import (
    SuffixReoptimizationV14Config,
    run_suffix_reoptimization_v1_4,
    scheduled_learning_rate as suffix_v1_4_scheduled_learning_rate,
    validate_suffix_reoptimization_v1_4_config,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_4_1 import (
    SuffixReoptimizationV141Config,
    run_suffix_reoptimization_v1_4_1,
    scheduled_learning_rate as suffix_v1_4_1_scheduled_learning_rate,
    validate_suffix_reoptimization_v1_4_1_config,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v2_0 import (
    SuffixReoptimizationV20Config,
    build_entry_snapshot_from_embedding,
    resolve_effective_layers as resolve_suffix_v2_0_effective_layers,
    run_suffix_reoptimization_v2_0,
)
from suffix_optimization_methods.method_versions.suffix_reoptimization_v2_1 import (
    SuffixReoptimizationV21Config,
    build_entry_snapshot_from_embedding as build_suffix_v2_1_entry_snapshot,
    resolve_effective_layers as resolve_suffix_v2_1_effective_layers,
    run_suffix_reoptimization_v2_1,
)
from utils import *


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected a boolean value")


def json_arg(value):
    return json.loads(value)


def normalize_max_memory(max_memory):
    if not max_memory:
        return None
    normalized = {}
    for key, value in max_memory.items():
        if isinstance(key, str) and key.isdigit():
            normalized[int(key)] = value
        else:
            normalized[key] = value
    return normalized


def normalize_suffix_version(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    aliases = {
        "2.1": "v2.1",
        "v2.1": "v2.1",
        "suffix_v2_1": "v2.1",
        "suffix_reoptimization_v2.1": "v2.1",
        "suffix_reoptimization_v2_1": "v2.1",
        "2.0": "v2.0",
        "v2.0": "v2.0",
        "suffix_v2_0": "v2.0",
        "suffix_reoptimization_v2.0": "v2.0",
        "suffix_reoptimization_v2_0": "v2.0",
        "1.4.1": "v1.4.1",
        "v1.4.1": "v1.4.1",
        "suffix_reoptimization_v1.4.1": "v1.4.1",
        "suffix_reoptimization_v1_4_1": "v1.4.1",
        "1.4": "v1.4",
        "v1.4": "v1.4",
        "suffix_reoptimization_v1.4": "v1.4",
        "suffix_reoptimization_v1_4": "v1.4",
        "1.3": "v1.3",
        "v1.3": "v1.3",
        "suffix_reoptimization_v1.3": "v1.3",
        "suffix_reoptimization_v1_3": "v1.3",
        "1.2.1": "v1.2.1",
        "v1.2.1": "v1.2.1",
        "suffix_reoptimization_v1.2.1": "v1.2.1",
        "suffix_reoptimization_v1_2_1": "v1.2.1",
        "1.2.2": "v1.2.2",
        "v1.2.2": "v1.2.2",
        "suffix_reoptimization_v1.2.2": "v1.2.2",
        "suffix_reoptimization_v1_2_2": "v1.2.2",
        "1.2.3": "v1.2.3",
        "v1.2.3": "v1.2.3",
        "suffix_v1.2.3": "v1.2.3",
        "suffix_v1_2_3": "v1.2.3",
        "suffix_reoptimization_v1.2.3": "v1.2.3",
        "suffix_reoptimization_v1_2_3": "v1.2.3",
        "1.3.1": "v1.3.1",
        "v1.3.1": "v1.3.1",
        "suffix_reoptimization_v1.3.1": "v1.3.1",
        "suffix_reoptimization_v1_3_1": "v1.3.1",
        "1.0": "v1.0",
        "v1.0": "v1.0",
        "suffix_reoptimization_v1.0": "v1.0",
        "suffix_reoptimization_v1_0": "v1.0",
        "1.1": "v1.1",
        "v1.1": "v1.1",
        "suffix_reoptimization_v1.1": "v1.1",
        "suffix_reoptimization_v1_1": "v1.1",
        "1.2": "v1.2",
        "v1.2": "v1.2",
        "suffix_reoptimization_v1.2": "v1.2",
        "suffix_reoptimization_v1_2": "v1.2",
        "none": "none",
        "off": "none",
        "baseline": "none",
    }
    if value not in aliases:
        raise ValueError("suffix_version must be one of: v2.1, v2.0, v1.4.1, v1.4, v1.3.1, v1.3, v1.2.3, v1.2.2, v1.2.1, v1.2, v1.1, v1.0, none")
    return aliases[value]


def normalize_cgmr_version(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    aliases = {
        "1.0": "v1.0",
        "v1.0": "v1.0",
        "cgmr_v1.0": "v1.0",
        "cgmr_v1_0": "v1.0",
        "1.1": "v1.1",
        "v1.1": "v1.1",
        "cgmr_v1.1": "v1.1",
        "cgmr_v1_1": "v1.1",
        "1.2": "v1.2",
        "v1.2": "v1.2",
        "cgmr_v1.2": "v1.2",
        "cgmr_v1_2": "v1.2",
        "none": "none",
        "off": "none",
        "baseline": "none",
    }
    if value not in aliases:
        raise ValueError(
            "cgmr_version must be one of: v1.2, v1.1, v1.0, none"
        )
    return aliases[value]


def select_cgmr_method(
        version,
        cgmr_v1_0_config,
        cgmr_v1_1_config=None,
        cgmr_v1_2_config=None):
    version = normalize_cgmr_version(version)
    if version == "v1.2":
        if cgmr_v1_2_config is None or not cgmr_v1_2_config.enabled:
            raise ValueError(
                "cgmr_version is v1.2 but cgmr_v1_2 is disabled"
            )
        return "CGMR_v1.2"
    if version == "v1.1":
        if cgmr_v1_1_config is None or not cgmr_v1_1_config.enabled:
            raise ValueError(
                "cgmr_version is v1.1 but cgmr_v1_1 is disabled"
            )
        return "CGMR_v1.1"
    if version == "v1.0":
        if not cgmr_v1_0_config.enabled:
            raise ValueError(
                "cgmr_version is v1.0 but cgmr_v1_0 is disabled"
            )
        return "CGMR_v1.0"
    if version == "none":
        return "none"
    if cgmr_v1_2_config is not None and cgmr_v1_2_config.enabled:
        return "CGMR_v1.2"
    if cgmr_v1_1_config is not None and cgmr_v1_1_config.enabled:
        return "CGMR_v1.1"
    if cgmr_v1_0_config.enabled:
        return "CGMR_v1.0"
    return "none"


def validate_advanced_candidate_combination(
        selected_advanced_method, selected_candidate_reranking_method):
    """Reject combinations whose formal semantics are not defined."""
    if (
        selected_advanced_method == "suffix_reoptimization_v2.1"
        and selected_candidate_reranking_method != "none"
    ):
        raise ValueError("suffix v2.1 cannot be combined with any CGMR version")
    return True


def validate_suffix_v21_model_config(selected_advanced_method, model_config):
    if selected_advanced_method != "suffix_reoptimization_v2.1":
        return True
    model_type = str(getattr(model_config, "model_type", "")).lower()
    architectures = [
        str(value).lower()
        for value in (getattr(model_config, "architectures", None) or [])
    ]
    if (
        model_type not in {"qwen2", "qwen2_5", "qwen2.5"}
        or not any(
            "qwen2" in value and "causallm" in value
            for value in architectures
        )
    ):
        raise ValueError(
            "suffix v2.1 supports only Qwen2/Qwen2.5 causal language models"
        )
    return True


def _nonfixed_token_accuracy(reference_token_ids, candidate_tokens,
                             eval_start_pos):
    reference = reference_token_ids
    if hasattr(reference, "detach"):
        reference = reference.detach().cpu().tolist()
    if reference and isinstance(reference[0], (list, tuple)):
        reference = reference[0]
    reference = [int(token_id) for token_id in reference]
    candidate = [int(token_id) for token_id in candidate_tokens]
    if len(reference) != len(candidate):
        raise ValueError(
            "reference and candidate token sequences must have equal length"
        )
    start = min(max(int(eval_start_pos), 0), len(reference))
    evaluated_count = len(reference) - start
    if not evaluated_count:
        return 0.0
    correct_count = sum(
        reference[position] == candidate[position]
        for position in range(start, len(reference))
    )
    return correct_count / evaluated_count


def evaluate_frozen_suffix_v21_accuracy(
        formal_result, reference_token_ids, eval_start_pos):
    """Evaluate frozen v2.1 tokens without mutating the formal result.

    This is experiment-level benchmark evaluation after the sidecar returns.
    It stays outside the v2.1 formal/diagnostics result domain and therefore
    must never populate or alter formal diagnostics fields.
    """
    final_tokens = formal_result.get("final_tokens")
    if final_tokens is None:
        return None
    return _nonfixed_token_accuracy(
        reference_token_ids,
        tuple(int(token_id) for token_id in final_tokens),
        eval_start_pos,
    )


def select_advanced_method(suffix_version, suffix_reopt_v1_2_1_config, suffix_reopt_v1_2_config,
                           suffix_reopt_v1_1_config, suffix_reopt_v1_0_config,
                           suffix_reopt_v1_3_config=None,
                           suffix_reopt_v1_4_config=None,
                           suffix_reopt_v1_4_1_config=None,
                           suffix_reopt_v1_2_2_config=None,
                           suffix_reopt_v1_3_1_config=None,
                           suffix_v1_2_3_config=None,
                           suffix_reopt_v2_0_config=None,
                           suffix_reopt_v2_1_config=None):
    suffix_version = normalize_suffix_version(suffix_version)
    if suffix_version == "v2.1":
        if suffix_reopt_v2_1_config is None or not suffix_reopt_v2_1_config.enabled:
            raise ValueError(
                "suffix_version is v2.1 but suffix_reoptimization_v2_1 is disabled"
            )
        return "suffix_reoptimization_v2.1"
    if suffix_version == "v2.0":
        if suffix_reopt_v2_0_config is None or not suffix_reopt_v2_0_config.enabled:
            raise ValueError(
                "suffix_version is v2.0 but suffix_reoptimization_v2_0 is disabled"
            )
        return "suffix_reoptimization_v2.0"
    if suffix_version == "v1.2.3":
        if suffix_v1_2_3_config is None or not suffix_v1_2_3_config.enabled:
            raise ValueError(
                "suffix_version is v1.2.3 but suffix_v1_2_3 is disabled"
            )
        return "suffix_v1.2.3"
    if suffix_version == "v1.3.1":
        if suffix_reopt_v1_3_1_config is None or not suffix_reopt_v1_3_1_config.enabled:
            raise ValueError(
                "suffix_version is v1.3.1 but suffix_reoptimization_v1_3_1 is disabled"
            )
        return "suffix_reoptimization_v1.3.1"
    if suffix_version == "v1.2.2":
        if suffix_reopt_v1_2_2_config is None or not suffix_reopt_v1_2_2_config.enabled:
            raise ValueError(
                "suffix_version is v1.2.2 but suffix_reoptimization_v1_2_2 is disabled"
            )
        return "suffix_reoptimization_v1.2.2"
    if suffix_version == "v1.4.1":
        if suffix_reopt_v1_4_1_config is None or not suffix_reopt_v1_4_1_config.enabled:
            raise ValueError(
                "suffix_version is v1.4.1 but suffix_reoptimization_v1_4_1 is disabled"
            )
        return "suffix_reoptimization_v1.4.1"
    if suffix_version == "v1.4":
        if suffix_reopt_v1_4_config is None or not suffix_reopt_v1_4_config.enabled:
            raise ValueError(
                "suffix_version is v1.4 but suffix_reoptimization_v1_4 is disabled"
            )
        return "suffix_reoptimization_v1.4"
    if suffix_version == "v1.3":
        if suffix_reopt_v1_3_config is None or not suffix_reopt_v1_3_config.enabled:
            raise ValueError(
                "suffix_version is v1.3 but suffix_reoptimization_v1_3 is disabled"
            )
        return "suffix_reoptimization_v1.3"
    if suffix_version == "v1.2.1":
        if not suffix_reopt_v1_2_1_config.enabled:
            raise ValueError(
                "suffix_version is v1.2.1 but suffix_reoptimization_v1_2_1 is disabled"
            )
        return "suffix_reoptimization_v1.2.1"
    if suffix_version == "v1.2":
        if not suffix_reopt_v1_2_config.enabled:
            raise ValueError("suffix_version is v1.2 but suffix_reoptimization_v1_2 is disabled")
        return "suffix_reoptimization_v1.2"
    if suffix_version == "v1.1":
        if not suffix_reopt_v1_1_config.enabled:
            raise ValueError("suffix_version is v1.1 but suffix_reoptimization_v1_1 is disabled")
        return "suffix_reoptimization_v1.1"
    if suffix_version == "v1.0":
        if not suffix_reopt_v1_0_config.enabled:
            raise ValueError("suffix_version is v1.0 but suffix_reoptimization_v1_0 is disabled")
        return "suffix_reoptimization_v1.0"
    if suffix_version == "none":
        return "frozen_original_baseline"
    if suffix_reopt_v2_1_config is not None and suffix_reopt_v2_1_config.enabled:
        return "suffix_reoptimization_v2.1"
    if suffix_reopt_v1_4_1_config is not None and suffix_reopt_v1_4_1_config.enabled:
        return "suffix_reoptimization_v1.4.1"
    if suffix_reopt_v1_4_config is not None and suffix_reopt_v1_4_config.enabled:
        return "suffix_reoptimization_v1.4"
    if suffix_reopt_v1_3_config is not None and suffix_reopt_v1_3_config.enabled:
        return "suffix_reoptimization_v1.3"
    if suffix_reopt_v1_2_1_config.enabled:
        return "suffix_reoptimization_v1.2.1"
    if suffix_reopt_v1_2_config.enabled:
        return "suffix_reoptimization_v1.2"
    if suffix_reopt_v1_1_config.enabled:
        return "suffix_reoptimization_v1.1"
    if suffix_reopt_v1_0_config.enabled:
        return "suffix_reoptimization_v1.0"
    return "frozen_original_baseline"


def merge_config(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def canonicalize_config_aliases(config, source="<config>"):
    canonical = copy.deepcopy(config)
    alias_pairs = (
        ("suffix_version", "suffix_reoptimization_version"),
        ("suffix_v1_2_3", "suffix_reoptimization_v1_2_3"),
        ("suffix_v1_2_3_log", "suffix_reoptimization_v1_2_3_log"),
    )
    for canonical_key, legacy_key in alias_pairs:
        if canonical_key in canonical and legacy_key in canonical:
            if canonical[canonical_key] != canonical[legacy_key]:
                raise ValueError(
                    "{} defines conflicting {} and {}".format(
                        source,
                        canonical_key,
                        legacy_key,
                    )
                )
        elif legacy_key in canonical:
            canonical[canonical_key] = canonical[legacy_key]
        canonical.pop(legacy_key, None)
    return canonical


def resolve_cli_suffix_version(
        canonical_value,
        legacy_value,
        canonical_option_present=False):
    if legacy_value is None:
        return canonical_value
    if (
        canonical_option_present
        and canonical_value is not None
        and normalize_suffix_version(canonical_value)
        != normalize_suffix_version(legacy_value)
    ):
        raise ValueError(
            "--suffix-version conflicts with "
            "--suffix-reoptimization-version"
        )
    return legacy_value


def load_config(path, seen=None):
    if path is None:
        return {}
    path = os.path.abspath(path)
    if seen is None:
        seen = set()
    if path in seen:
        raise ValueError("recursive config include detected for {}".format(path))
    seen.add(path)
    if not os.path.exists(path):
        raise FileNotFoundError("config file {} does not exist".format(path))
    with open(path, "r", encoding="utf-8") as f:
        config = canonicalize_config_aliases(json.load(f), source=path)

    merged = {}
    include_configs = config.get("include_configs", [])
    if isinstance(include_configs, str):
        include_configs = [include_configs]
    for include_path in include_configs:
        if not os.path.isabs(include_path):
            include_path = os.path.join(os.path.dirname(path), include_path)
        merged = merge_config(merged, load_config(include_path, seen))
    current_config = copy.deepcopy(config)
    current_config.pop("include_configs", None)
    result = merge_config(merged, current_config)
    seen.remove(path)
    return result


def _utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_parallel_worker_spec(path, output_dir):
    if bool(path) != bool(output_dir):
        raise ValueError(
            "--parallel-worker-spec and --worker-output-dir must be used together"
        )
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as source:
        spec = json.load(source)
    required = (
        "worker_id", "physical_gpu_id", "assigned_global_indices",
        "assigned_samples",
    )
    missing = [name for name in required if name not in spec]
    if missing:
        raise ValueError("worker spec missing: {}".format(", ".join(missing)))
    indices = [int(value) for value in spec["assigned_global_indices"]]
    if len(indices) != len(set(indices)) or any(value < 0 or value > 9 for value in indices):
        raise ValueError("worker assigned_global_indices are invalid")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(spec["physical_gpu_id"]):
        raise ValueError("worker CUDA_VISIBLE_DEVICES does not match its physical GPU")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise ValueError("parallel worker must see exactly one available CUDA GPU")
    if args_device := spec.get("local_device"):
        if args_device != "cuda:0":
            raise ValueError("parallel worker local_device must be cuda:0")
    spec["assigned_global_indices"] = indices
    spec["output_dir"] = os.path.abspath(output_dir)
    spec["started_at"] = _utc_timestamp()
    return spec


def _atomic_dump_json(path, data):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False, default=json_default)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _write_parallel_worker_status(spec, completed_indices, record_count,
                                  exit_code, success, failure_reason,
                                  resolved_config=None):
    if spec is None:
        return
    status = {
        "worker_id": int(spec["worker_id"]),
        "physical_gpu_id": int(spec["physical_gpu_id"]),
        "assigned_global_indices": list(spec["assigned_global_indices"]),
        "completed_global_indices": [int(value) for value in completed_indices],
        "record_count": int(record_count),
        "exit_code": int(exit_code),
        "started_at": spec["started_at"],
        "finished_at": _utc_timestamp(),
        "success": bool(success),
        "failure_reason": failure_reason,
        "local_device": "cuda:0",
        "device_map": {"": 0},
    }
    if resolved_config is not None:
        status["resolved_config"] = resolved_config
        fingerprint_source = copy.deepcopy(resolved_config)
        fingerprint_source.get("artifacts", {}).clear()
        fingerprint_source.get("outputs", {}).clear()
        fingerprint_source.get("run", {}).pop("timestamp", None)
        fingerprint_bytes = json.dumps(
            fingerprint_source,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=json_default,
        ).encode("utf-8")
        status["runtime_config_fingerprint"] = hashlib.sha256(
            fingerprint_bytes
        ).hexdigest()
        model_metadata = resolved_config.get("model") or {}
        status["model_metadata"] = {
            key: model_metadata.get(key)
            for key in (
                "base_model_name", "model_type", "config_layers",
                "loaded_layers", "num_invert_layers",
            )
        }
    os.makedirs(spec["output_dir"], exist_ok=True)
    _atomic_dump_json(
        os.path.join(spec["output_dir"], "worker_status.json"), status
    )


def get_layer_id(name):
    parts = name.split(".")
    for i in range(len(parts) - 1):
        if parts[i] == "layers" and parts[i + 1].isdigit() and i + 2 == len(parts):
            return int(parts[i + 1])
    return None


def register_layer_hooks(model, layer_id, forward_hook, full_hook=None, up_to=True):
    hook_handles = []
    for name, module in model.named_modules():
        current_layer = get_layer_id(name)
        if current_layer is None:
            continue
        if up_to:
            if current_layer == 0 and full_hook is not None:
                hook_handles.append(module.register_forward_hook(full_hook))
            elif current_layer <= layer_id:
                hook_handles.append(module.register_forward_hook(forward_hook))
        elif current_layer == layer_id:
            hook_handles.append(module.register_forward_hook(forward_hook))
    if not hook_handles:
        raise ValueError("no transformer layer hooks registered for layer {}".format(layer_id))
    return hook_handles


def get_model_layers(model):
    roots = [
        getattr(model, "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for root in roots:
        layers = getattr(root, "layers", None)
        if layers is not None:
            return len(layers)
    raise ValueError("could not find transformer layers on model")


def get_input_embedding_layer(model):
    roots = [
        model,
        getattr(model, "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for root in roots:
        if root is None or not hasattr(root, "get_input_embeddings"):
            continue
        embed_layer = root.get_input_embeddings()
        if embed_layer is not None:
            return embed_layer
    raise ValueError("could not find input embedding layer on model")


def get_model_device(model):
    try:
        return get_input_embedding_layer(model).weight.device
    except Exception:
        return next(model.parameters()).device


def get_model(base_model_name,
              lora_model_name,
              device_map="manual",
              offload_folder=None,
              offload_state_dict=False,
              max_memory=None,
              quantization="none",
              model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {"low_cpu_mem_usage": True, "use_cache": False}
    if os.path.isabs(base_model_name) and not os.path.exists(base_model_name):
        raise FileNotFoundError("base model path {} does not exist".format(base_model_name))
    quantization = quantization.lower()
    if quantization not in ("4bit", "8bit", "none"):
        raise ValueError("quantization must be one of: 4bit, 8bit, none")
    if quantization == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    if device_map != "manual":
        model_kwargs["device_map"] = {"": 0} if device_map == "single_gpu" else device_map
        max_memory = normalize_max_memory(max_memory)
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        if offload_folder:
            os.makedirs(offload_folder, exist_ok=True)
            model_kwargs["offload_folder"] = offload_folder
        model_kwargs["offload_state_dict"] = offload_state_dict

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if device_map != "manual":
        if lora_model_name is None:
            model = base_model
        else:
            model = PeftModel.from_pretrained(base_model, lora_model_name, torch_dtype=torch.float16)
        model.gradient_checkpointing = True
        return tokenizer, model
    if torch.cuda.device_count() == 0:
        raise RuntimeError("manual device_map requires at least one CUDA device")
    device_map = {}
    model_layers = len(base_model.model.layers)
    if lora_model_name is None:
        for i in range(model_layers):
            layer = "model.layers." + str(i)
            device_map[layer] = int(i / (model_layers) * torch.cuda.device_count())
        device_map["model.embed_tokens"] = 0
        device_map["model.norm"] = torch.cuda.device_count() - 1
        device_map["lm_head"] = torch.cuda.device_count() - 1
        model = dispatch_model(base_model, device_map=device_map)
    else:
        for i in range(model_layers):
            layer = "base_model.model.model.layers." + str(i)
            device_map[layer] = int(i / (model_layers) * torch.cuda.device_count())
        device_map["base_model.model.model.embed_tokens"] = 0
        device_map["base_model.model.model.norm"] = torch.cuda.device_count() - 1
        device_map["base_model.model.lm_head"] = torch.cuda.device_count() - 1
        lora_model = PeftModel.from_pretrained(base_model, lora_model_name, torch_dtype=torch.float16)
        model = dispatch_model(lora_model, device_map=device_map)
    model.gradient_checkpointing = True
    return tokenizer, model


def ensure_suffix_v2_committed_prefix(
        input_ids,
        attention_mask,
        tokenizer,
        selected_advanced_method):
    """Give suffix v2.x the fixed committed prefix required by its PPL stage."""
    suffix_v2_methods = {
        "suffix_reoptimization_v2.0",
        "suffix_reoptimization_v2.1",
    }
    if selected_advanced_method not in suffix_v2_methods:
        return input_ids, attention_mask

    special_ids = {int(token_id) for token_id in tokenizer.all_special_ids}
    if int(input_ids[0, 0].item()) in special_ids:
        return input_ids, attention_mask

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None or int(eos_token_id) not in special_ids:
        version = "v2.1" if selected_advanced_method.endswith("v2.1") else "v2.0"
        raise ValueError(
            "suffix {} requires the tokenizer EOS special token as its "
            "committed prefix".format(version)
        )

    prefix_ids = torch.full(
        (input_ids.shape[0], 1),
        int(eos_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    prefix_mask = torch.ones(
        (attention_mask.shape[0], 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return (
        torch.cat((prefix_ids, input_ids), dim=1),
        torch.cat((prefix_mask, attention_mask), dim=1),
    )


def formal_method_result_for_record(record):
    """Return the selected canonical suffix result without version branching."""
    return (
        record.get("suffix_reoptimization_result")
        or record.get("suffix_reoptimization_v2_1_result")
        or record.get("suffix_reoptimization_v2_0_result")
        or {}
    )


def suffix_v2_classifier_record_fields(
        selected_advanced_method, suffix_reopt_v2_0_result):
    """Return the legacy classifier schema only for selected suffix v2.0."""
    if selected_advanced_method != "suffix_reoptimization_v2.0":
        return {}
    return {
        "classifier_enabled": bool(
            suffix_reopt_v2_0_result.get("classifier_enabled", False)
        ),
        "classifier_provider_available": bool(
            suffix_reopt_v2_0_result.get(
                "classifier_provider_available", False
            )
        ),
        "classifier_candidate_count": int(
            suffix_reopt_v2_0_result.get("classifier_candidate_count", 0)
        ),
    }


def record_has_fatal_formal_failure(record):
    result = formal_method_result_for_record(record)
    return bool(record.get("fatal_failure") or result.get("fatal_failure"))


def experiment_exit_code_for_records(run_records):
    """Return a failing exit code when any formal sample result is fatal."""
    return 2 if any(record_has_fatal_formal_failure(record)
                    for record in run_records) else 0


def get_hidden_state(
        tokenizer,
        model,
        layer_id,
        prompt=None,
        input_embed=None,
        target_attention_mask=None,
        up_to=True,
        selected_advanced_method=None):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    def full_hook(module, input, output):
        '''full hook'''
        if isinstance(input, tuple):
            hidden_state_list.append(input[0])
        else:
            hidden_state_list.append(input)
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids, target_attention_mask = (
                ensure_suffix_v2_committed_prefix(
                    target_token['input_ids'],
                    target_token['attention_mask'],
                    tokenizer,
                    selected_advanced_method,
                )
            )
            target_input_ids = target_input_ids.to(get_model_device(model))
            target_attention_mask = target_attention_mask.to(get_model_device(model))
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
            
            hook_handles = register_layer_hooks(
                model,
                layer_id,
                forward_hook,
                full_hook=full_hook if up_to else None,
                up_to=up_to)

            next_ = model(**inputs)
            embed_layer = get_input_embedding_layer(model)
            ori_input_embed = embed_layer(target_input_ids)

    elif input_embed != None:
        with torch.no_grad():
            input_embed = input_embed.to(get_model_device(model))
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask}

            hook_handles = register_layer_hooks(
                model,
                layer_id,
                forward_hook,
                full_hook=full_hook if up_to else None,
                up_to=up_to)

            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            target_input_ids = None
    else:    
        raise NotImplementedError
    for handle in hook_handles:
        handle.remove()
    return target_input_ids, target_attention_mask, ori_input_embed, hidden_state_list


def get_hidden_state_base(tokenizer, model, layer_id, prompt=None, input_embed=None, target_attention_mask=None, up_to=True):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    def full_hook(module, input, output):
        if isinstance(input, tuple):
            hidden_state_list.append(input[0])
        else:
            hidden_state_list.append(input)
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(get_model_device(model))
            target_attention_mask = target_token['attention_mask'].to(get_model_device(model))
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
            
            hook_handles = register_layer_hooks(
                model,
                layer_id,
                forward_hook,
                full_hook=full_hook if up_to else None,
                up_to=up_to)

            next_ = model(**inputs)
            embed_layer = get_input_embedding_layer(model)
            ori_input_embed = embed_layer(target_input_ids)

    elif input_embed != None:
        with torch.no_grad():
            input_embed = input_embed.to(get_model_device(model))
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask} 
            
            hook_handles = register_layer_hooks(
                model,
                layer_id,
                forward_hook,
                full_hook=full_hook if up_to else None,
                up_to=up_to)

            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            target_input_ids = None
    else:    
        raise NotImplementedError
    for handle in hook_handles:
        handle.remove()
    return target_input_ids, target_attention_mask, ori_input_embed, hidden_state_list


EMBEDDING_SEARCH_CHUNK_SIZE = 8192


def embedding_top_indices(embed, embed_layer, top_k, invert_method, chunk_size=EMBEDDING_SEARCH_CHUNK_SIZE):
    weight = embed_layer.weight.detach()
    vocab_size = weight.shape[0]
    top_k = max(1, min(top_k, vocab_size))
    embed_cpu = embed.detach().to("cpu", dtype=torch.float32)
    best_scores = None
    best_indices = None

    with torch.no_grad():
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            weight_chunk = weight[start:end].detach().to("cpu", dtype=torch.float32)
            if invert_method == 'L2':
                scores = -torch.norm(weight_chunk - embed_cpu, p=2, dim=1)
            elif invert_method == 'cosine':
                scores = F.cosine_similarity(embed_cpu.unsqueeze(0), weight_chunk, dim=-1)
            else:
                raise NotImplementedError

            chunk_top_k = min(top_k, scores.numel())
            chunk_scores, chunk_indices = torch.topk(scores, chunk_top_k)
            chunk_indices = chunk_indices + start

            if best_scores is None:
                best_scores = chunk_scores
                best_indices = chunk_indices
            else:
                combined_scores = torch.cat((best_scores, chunk_scores))
                combined_indices = torch.cat((best_indices, chunk_indices))
                best_top_k = min(top_k, combined_scores.numel())
                best_scores, best_positions = torch.topk(combined_scores, best_top_k)
                best_indices = combined_indices[best_positions]

    return best_indices


def select_candidate_from_top_indices(top_indices, tokenizer, filter_nonascii=True):
    special_token_ids = set(tokenizer.all_special_ids)
    for candidate in top_indices:
        token_id = int(candidate.data.cpu())
        if token_id in special_token_ids:
            continue
        if filter_nonascii and not tokenizer.decode([token_id]).isascii():
            continue
        return token_id, [int(item.data.cpu()) for item in top_indices]
    fallback = int(top_indices[0].data.cpu())
    return fallback, [int(item.data.cpu()) for item in top_indices]


def select_candidate_token(dist_ret, tokenizer, top_k, filter_nonascii=True):
    top_indices = torch.topk(dist_ret, top_k).indices
    return select_candidate_from_top_indices(top_indices, tokenizer, filter_nonascii)


def invert_embedding(hidden_state, tokenizer, embed_layer, total_input_ids, f=None, invert_method='cosine', show_position=False, filter_nonascii=True, top_k=10, console=True, eval_start_pos=0):
    def write(text):
        if f is not None:
            log_line(f, text, console=console)
        else:
            print(console_safe_text(text), flush=True)

    if len(hidden_state.shape) >= 3:
        new_input_embed_squeeze = hidden_state.squeeze(0)
    else:
        new_input_embed_squeeze = hidden_state

    ret_list = []
    for embed in new_input_embed_squeeze:
        top_indices = embedding_top_indices(embed, embed_layer, top_k, invert_method)
        token_id, _ = select_candidate_from_top_indices(top_indices, tokenizer, filter_nonascii)
        ret_list.append(token_id)

    '''show position accuracy'''
    acc_cnt = 0
    prompt_length = len(total_input_ids[0])
    eval_length = max(prompt_length - eval_start_pos, 0)
    for j in range(eval_start_pos, prompt_length):
        if int(total_input_ids[0][j].data.cpu()) == int(ret_list[j]):
            acc_cnt += 1
    acc = acc_cnt / eval_length if eval_length else 0.0
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[eval_start_pos:]))
    return acc, ret_tokens, ret_list


def invert_and_find_best(hidden_state, gt_hidden_state, tokenizer, model, 
                         total_input_ids, layer_id, f=None, invert_method='cosine', 
                         filter_nonascii=True, add_perplexity=True, top_k_ppl=10, top_k_cos=10, eval_start_pos=0):
    def write(text, console=True):
        if f is not None:
            log_line(f, text, console=console)
        elif console:
            print(console_safe_text(text), flush=True)

    embed_layer = get_input_embedding_layer(model)
    if len(hidden_state.shape) >= 3:
        new_input_embed_squeeze = hidden_state.squeeze(0)
    else:
        new_input_embed_squeeze = hidden_state

    ret_list = []
    ret_top_k = []
    if top_k_cos == 0:
        for embed in new_input_embed_squeeze:
            ret_top_k.append([0])
            ret_list.append(0)
    else:
        for embed in new_input_embed_squeeze:
            top_indices = embedding_top_indices(embed, embed_layer, top_k_cos, invert_method)
            token_id, top_ids = select_candidate_from_top_indices(top_indices, tokenizer, filter_nonascii)
            ret_top_k.append(top_ids)
            ret_list.append(token_id)

    for i in range(min(eval_start_pos, len(ret_list))):
        ret_list[i] = int(total_input_ids[0][i].data.cpu())
        ret_top_k[i] = [ret_list[i]]

    start = time.time()
    token_count = len(ret_top_k)
    for i, top_list in enumerate(ret_top_k):
        console_update("token processing[{}/{}]".format(i + 1, token_count))
        if i > 0 and add_perplexity:
            input_ids = copy.deepcopy(ret_list[:i])
            perplexity, topk_ids = get_perplexity(input_ids, model, layer_id=layer_id, top_k=top_k_ppl)
            top_list += topk_ids.tolist()
        
        replaced_ret_list = []
        for item in top_list:
            replaced_ret = copy.deepcopy(ret_list)
            replaced_ret[i] = item
            replaced_ret_list.append(replaced_ret)

        new_hidden_states = forward_and_get_last_hidden_state(model, replaced_ret_list, None, layer_id=layer_id)
        gt_hidden_state = gt_hidden_state.to(new_hidden_states.device)
        cos_sim_list = F.cosine_similarity(new_hidden_states.index_select(1, torch.tensor([i]).to(new_hidden_states.device)).permute(1,0,2).squeeze(0).type(torch.float32), 
                                           gt_hidden_state.index_select(1, torch.tensor([i]).to(new_hidden_states.device)).permute(1,0,2).squeeze(0).type(torch.float32), dim=-1)
        cos_sim_list = cos_sim_list.data.cpu().numpy()
        idx = np.argmax(cos_sim_list)
        ret_list[i] = top_list[idx]

    end = time.time()
    '''show accuracy'''
    acc_cnt = 0
    prompt_length = len(total_input_ids[0])
    eval_length = max(prompt_length - eval_start_pos, 0)
    for j in range(eval_start_pos, prompt_length):
        if int(total_input_ids[0][j].data.cpu()) == int(ret_list[j]):
            acc_cnt += 1
    acc = acc_cnt / eval_length if eval_length else 0.0
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[eval_start_pos:]))
    return acc, ret_tokens, ret_list


def forward_and_get_last_hidden_state(model, input_ids, attention_mask, layer_id):
    if len(torch.tensor(input_ids).shape) < 2:
        input_ids_squeeze = torch.tensor(input_ids).unsqueeze(0).to(get_model_device(model))
    else:
        input_ids_squeeze = torch.tensor(input_ids).to(get_model_device(model))
    new_inputs = {'input_ids': input_ids_squeeze, 'attention_mask': attention_mask} 

    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    hook_handles = register_layer_hooks(model, layer_id, forward_hook, up_to=False)
    phi_relaxed = model(**new_inputs)
    for handle in hook_handles:
        handle.remove()
    if not hidden_state_list:
        raise ValueError("no hidden states collected for layer {}".format(layer_id))
    last_hidden_state = hidden_state_list[0]
    return last_hidden_state


def get_perplexity(input_ids, model, layer_id, next_ids=None, top_k=None):
    hidden_state_list = []
    hook_handles = []
    if isinstance(input_ids, torch.Tensor):
        inputs = {'input_ids': input_ids.to(get_model_device(model)), 'attention_mask': None}
    else:
        inputs = {'input_ids': torch.tensor(input_ids).unsqueeze(0).to(get_model_device(model)), 'attention_mask': None}
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    hook_handles = register_layer_hooks(model, layer_id, forward_hook, up_to=False)
    next_token_logits = model(**inputs).logits[:, -1, :]
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    for handle in hook_handles:
        handle.remove()
    if next_ids != None:
        perplexity = probs[0][next_ids]
        top_ids = next_ids
    elif top_k != None:
        top_ids = torch.topk(probs[0], top_k).indices
        perplexity = probs[0][top_ids]
    else:
        raise NotImplementedError
    return perplexity, top_ids


def main(args):
    disable_external_progress_bars()
    np.random.seed(args.seed)
    worker_spec = _load_parallel_worker_spec(
        getattr(args, "parallel_worker_spec", None),
        getattr(args, "worker_output_dir", None),
    )
    args._parallel_worker_runtime_spec = worker_spec
    if worker_spec is not None:
        args.device_map = "single_gpu"
    if args.base_model_name is None:
        raise ValueError("--base-model-name is required unless provided by --config")
    if args.epoch <= 0:
        raise ValueError("--epoch must be greater than 0")
    if args.suffix_reoptimization_v1_0 and args.suffix_v1_0_epoch <= 0:
        raise ValueError("suffix_v1_0_epoch must be greater than 0 when suffix reoptimization v1.0 is enabled")
    if args.suffix_reoptimization_v1_0 and args.suffix_v1_0_max_rounds <= 0:
        raise ValueError("suffix_v1_0_max_rounds must be greater than 0 when suffix reoptimization v1.0 is enabled")
    if args.suffix_reoptimization_v1_1 and args.suffix_v1_1_epoch <= 0:
        raise ValueError("suffix_v1_1_epoch must be greater than 0 when suffix reoptimization v1.1 is enabled")
    if args.suffix_reoptimization_v1_1 and args.suffix_v1_1_max_rounds <= 0:
        raise ValueError("suffix_v1_1_max_rounds must be greater than 0 when suffix reoptimization v1.1 is enabled")
    if args.suffix_reoptimization_v1_2_1 and args.suffix_v1_2_1_epoch <= 0:
        raise ValueError("suffix_v1_2_1_epoch must be greater than 0 when suffix reoptimization v1.2.1 is enabled")
    if args.suffix_reoptimization_v1_2_1 and args.suffix_v1_2_1_max_rounds <= 0:
        raise ValueError("suffix_v1_2_1_max_rounds must be greater than 0 when suffix reoptimization v1.2.1 is enabled")
    if args.suffix_reoptimization_v1_2_2 and args.suffix_v1_2_2_epoch <= 0:
        raise ValueError("suffix_v1_2_2_epoch must be greater than 0 when suffix reoptimization v1.2.2 is enabled")
    if args.suffix_reoptimization_v1_2_2 and args.suffix_v1_2_2_max_rounds <= 0:
        raise ValueError("suffix_v1_2_2_max_rounds must be greater than 0 when suffix reoptimization v1.2.2 is enabled")
    if args.suffix_v1_2_3 and args.suffix_v1_2_3_epoch <= 0:
        raise ValueError("suffix_v1_2_3_epoch must be greater than 0 when suffix v1.2.3 is enabled")
    if args.suffix_v1_2_3 and args.suffix_v1_2_3_max_rounds <= 0:
        raise ValueError("suffix_v1_2_3_max_rounds must be greater than 0 when suffix v1.2.3 is enabled")
    if args.suffix_reoptimization_v1_3_1 and args.suffix_v1_3_1_epoch <= 0:
        raise ValueError("suffix_v1_3_1_epoch must be greater than 0 when suffix reoptimization v1.3.1 is enabled")
    if args.suffix_reoptimization_v1_3_1 and args.suffix_v1_3_1_max_rounds <= 0:
        raise ValueError("suffix_v1_3_1_max_rounds must be greater than 0 when suffix reoptimization v1.3.1 is enabled")
    if args.suffix_reoptimization_v1_3 and args.suffix_v1_3_epoch <= 0:
        raise ValueError("suffix_v1_3_epoch must be greater than 0 when suffix reoptimization v1.3 is enabled")
    if args.suffix_reoptimization_v1_3 and args.suffix_v1_3_max_rounds <= 0:
        raise ValueError("suffix_v1_3_max_rounds must be greater than 0 when suffix reoptimization v1.3 is enabled")
    if args.suffix_reoptimization_v1_4 and args.suffix_v1_4_fine_epoch <= 0:
        raise ValueError("suffix_v1_4_fine_epoch must be greater than 0 when suffix reoptimization v1.4 is enabled")
    if args.suffix_reoptimization_v1_4_1 and args.suffix_v1_4_1_fine_epoch <= 0:
        raise ValueError("suffix_v1_4_1_fine_epoch must be greater than 0 when suffix reoptimization v1.4.1 is enabled")
    if args.suffix_reoptimization_v1_2 and args.suffix_v1_2_epoch <= 0:
        raise ValueError("suffix_v1_2_epoch must be greater than 0 when suffix reoptimization v1.2 is enabled")
    if args.suffix_reoptimization_v1_2 and args.suffix_v1_2_max_rounds <= 0:
        raise ValueError("suffix_v1_2_max_rounds must be greater than 0 when suffix reoptimization v1.2 is enabled")
    dataset_specs = getattr(args, "datasets", None)
    if (
        not dataset_specs
        and args.dataset_type in ("local", "github")
        and not os.path.exists(args.dataset_path)
    ):
        raise FileNotFoundError("{} dataset does not exist".format(args.dataset_path))
    cgmr_v1_0_config = CGMRV10Config(
        enabled=args.cgmr_v1_0,
        log_enabled=args.cgmr_v1_0_log,
        layer_offsets=tuple(args.cgmr_v1_0_layer_offsets),
        layer_weights=tuple(args.cgmr_v1_0_layer_weights),
        normalization=args.cgmr_v1_0_normalization,
        consistency_weight=args.cgmr_v1_0_consistency_weight,
        strong_margin_threshold=args.cgmr_v1_0_strong_margin_threshold,
        weak_margin_threshold=args.cgmr_v1_0_weak_margin_threshold,
        low_score_threshold=args.cgmr_v1_0_low_score_threshold,
        weak_signals_required=args.cgmr_v1_0_weak_signals_required,
        max_candidates=args.cgmr_v1_0_max_candidates,
        candidate_batch_size=args.cgmr_v1_0_candidate_batch_size,
        min_enhanced_gain=args.cgmr_v1_0_min_enhanced_gain,
        min_enhanced_margin=args.cgmr_v1_0_min_enhanced_margin,
        max_layer_l_drop=args.cgmr_v1_0_max_layer_l_drop,
        max_repair_steps=args.cgmr_v1_0_max_repair_steps,
    )
    cgmr_v1_1_config = CGMRV11Config(
        enabled=args.cgmr_v1_1,
        log_enabled=args.cgmr_v1_1_log,
        layer_offsets=tuple(args.cgmr_v1_1_layer_offsets),
        layer_weights=tuple(args.cgmr_v1_1_layer_weights),
        normalization=args.cgmr_v1_1_normalization,
        consistency_weight=args.cgmr_v1_1_consistency_weight,
        relative_margin_epsilon=args.cgmr_v1_1_relative_margin_epsilon,
        relative_margin_risk_weight=args.cgmr_v1_1_relative_margin_risk_weight,
        low_score_risk_weight=args.cgmr_v1_1_low_score_risk_weight,
        score_drop_risk_weight=args.cgmr_v1_1_score_drop_risk_weight,
        low_score_threshold=args.cgmr_v1_1_low_score_threshold,
        min_risk_score=args.cgmr_v1_1_min_risk_score,
        risk_top_k=args.cgmr_v1_1_risk_top_k,
        max_accepted_repairs=args.cgmr_v1_1_max_accepted_repairs,
        max_candidates=args.cgmr_v1_1_max_candidates,
        candidate_batch_size=args.cgmr_v1_1_candidate_batch_size,
        min_enhanced_gain=args.cgmr_v1_1_min_enhanced_gain,
        min_enhanced_margin=args.cgmr_v1_1_min_enhanced_margin,
        max_layer_l_drop=args.cgmr_v1_1_max_layer_l_drop,
    )
    cgmr_v1_2_config = CGMRV12Config(
        enabled=args.cgmr_v1_2,
        log_enabled=args.cgmr_v1_2_log,
        layer_offsets=tuple(args.cgmr_v1_2_layer_offsets),
        layer_weights=tuple(args.cgmr_v1_2_layer_weights),
        entropy_temperature=args.cgmr_v1_2_entropy_temperature,
        effective_candidate_threshold=(
            args.cgmr_v1_2_effective_candidate_threshold
        ),
        max_multilayer_candidates=(
            args.cgmr_v1_2_max_multilayer_candidates
        ),
        lookahead_window=args.cgmr_v1_2_lookahead_window,
        improvement_epsilon=args.cgmr_v1_2_improvement_epsilon,
        relative_mse_epsilon=args.cgmr_v1_2_relative_mse_epsilon,
        max_candidates=args.cgmr_v1_2_max_candidates,
        candidate_batch_size=args.cgmr_v1_2_candidate_batch_size,
    )
    suffix_reopt_v1_0_config = SuffixReoptimizationV10Config(
        enabled=args.suffix_reoptimization_v1_0,
        log_enabled=args.suffix_reoptimization_v1_0_log,
        max_rounds=args.suffix_v1_0_max_rounds,
        epoch=args.suffix_v1_0_epoch,
        lr=args.suffix_v1_0_lr,
        reg_weight=args.suffix_v1_0_reg_weight,
        hidden_low_threshold=args.suffix_v1_0_hidden_low_threshold,
        hidden_drop_threshold=args.suffix_v1_0_hidden_drop_threshold,
        token_forward_low_threshold=args.suffix_v1_0_token_forward_low_threshold,
        min_anomaly_reasons=args.suffix_v1_0_min_anomaly_reasons,
        min_hidden_delta=args.suffix_v1_0_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_0_accuracy_tolerance,
        accept_mode=args.suffix_v1_0_accept_mode,
    )
    suffix_reopt_v1_1_config = SuffixReoptimizationV11Config(
        enabled=args.suffix_reoptimization_v1_1,
        log_enabled=args.suffix_reoptimization_v1_1_log,
        max_rounds=args.suffix_v1_1_max_rounds,
        epoch=args.suffix_v1_1_epoch,
        lr=args.suffix_v1_1_lr,
        hidden_low_threshold=args.suffix_v1_1_hidden_low_threshold,
        hidden_drop_threshold=args.suffix_v1_1_hidden_drop_threshold,
        token_forward_low_threshold=args.suffix_v1_1_token_forward_low_threshold,
        min_anomaly_reasons=args.suffix_v1_1_min_anomaly_reasons,
        min_hidden_delta=args.suffix_v1_1_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_1_accuracy_tolerance,
        accept_mode=args.suffix_v1_1_accept_mode,
        hidden_weight_mode=args.suffix_v1_1_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_1_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_1_hidden_weight_floor,
        prox_weight=args.suffix_v1_1_prox_weight,
        manifold_weight=args.suffix_v1_1_manifold_weight,
        manifold_update_every=args.suffix_v1_1_manifold_update_every,
        manifold_warmup_epoch=args.suffix_v1_1_manifold_warmup_epoch,
        range_weight=args.suffix_v1_1_range_weight,
    )
    suffix_reopt_v1_2_config = SuffixReoptimizationV12Config(
        enabled=args.suffix_reoptimization_v1_2,
        log_enabled=args.suffix_reoptimization_v1_2_log,
        max_rounds=args.suffix_v1_2_max_rounds,
        epoch=args.suffix_v1_2_epoch,
        lr=args.suffix_v1_2_lr,
        hidden_low_threshold=args.suffix_v1_2_hidden_low_threshold,
        hidden_drop_threshold=args.suffix_v1_2_hidden_drop_threshold,
        token_forward_low_threshold=args.suffix_v1_2_token_forward_low_threshold,
        min_anomaly_reasons=args.suffix_v1_2_min_anomaly_reasons,
        min_hidden_delta=args.suffix_v1_2_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_2_accuracy_tolerance,
        accept_mode=args.suffix_v1_2_accept_mode,
        anomaly_detection_mode=args.suffix_v1_2_anomaly_detection_mode,
        adaptive_z_threshold=args.suffix_v1_2_adaptive_z_threshold,
        adaptive_drop_z_threshold=args.suffix_v1_2_adaptive_drop_z_threshold,
        adaptive_min_std=args.suffix_v1_2_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_2_adaptive_min_points,
        hidden_weight_mode=args.suffix_v1_2_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_2_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_2_hidden_weight_floor,
        prox_weight=args.suffix_v1_2_prox_weight,
        manifold_weight=args.suffix_v1_2_manifold_weight,
        manifold_update_every=args.suffix_v1_2_manifold_update_every,
        manifold_warmup_epoch=args.suffix_v1_2_manifold_warmup_epoch,
        range_weight=args.suffix_v1_2_range_weight,
    )
    suffix_reopt_v1_2_1_config = SuffixReoptimizationV121Config(
        enabled=args.suffix_reoptimization_v1_2_1,
        log_enabled=args.suffix_reoptimization_v1_2_1_log,
        max_rounds=args.suffix_v1_2_1_max_rounds,
        epoch=args.suffix_v1_2_1_epoch,
        lr=args.suffix_v1_2_1_lr,
        hidden_low_threshold=args.suffix_v1_2_1_hidden_low_threshold,
        hidden_drop_threshold=args.suffix_v1_2_1_hidden_drop_threshold,
        token_forward_low_threshold=args.suffix_v1_2_1_token_forward_low_threshold,
        min_anomaly_reasons=args.suffix_v1_2_1_min_anomaly_reasons,
        min_hidden_delta=args.suffix_v1_2_1_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_2_1_accuracy_tolerance,
        accept_mode=args.suffix_v1_2_1_accept_mode,
        anomaly_detection_mode=args.suffix_v1_2_1_anomaly_detection_mode,
        adaptive_z_threshold=args.suffix_v1_2_1_adaptive_z_threshold,
        adaptive_drop_z_threshold=args.suffix_v1_2_1_adaptive_drop_z_threshold,
        adaptive_min_std=args.suffix_v1_2_1_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_2_1_adaptive_min_points,
        hidden_weight_mode=args.suffix_v1_2_1_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_2_1_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_2_1_hidden_weight_floor,
        prox_weight=args.suffix_v1_2_1_prox_weight,
        range_weight=args.suffix_v1_2_1_range_weight,
    )
    suffix_reopt_v1_2_2_config = SuffixReoptimizationV122Config(
        enabled=args.suffix_reoptimization_v1_2_2,
        log_enabled=args.suffix_reoptimization_v1_2_2_log,
        max_rounds=args.suffix_v1_2_2_max_rounds,
        epoch=args.suffix_v1_2_2_epoch,
        lr=args.suffix_v1_2_2_lr,
        embedding_relative_mse_high_threshold=(
            args.suffix_v1_2_2_embedding_relative_mse_high_threshold
        ),
        relative_mse_rise_threshold=args.suffix_v1_2_2_relative_mse_rise_threshold,
        token_relative_mse_high_threshold=(
            args.suffix_v1_2_2_token_relative_mse_high_threshold
        ),
        min_anomaly_reasons=args.suffix_v1_2_2_min_anomaly_reasons,
        min_relative_mse_improvement=(
            args.suffix_v1_2_2_min_relative_mse_improvement
        ),
        accuracy_tolerance=args.suffix_v1_2_2_accuracy_tolerance,
        accept_mode=args.suffix_v1_2_2_accept_mode,
        anomaly_detection_mode=args.suffix_v1_2_2_anomaly_detection_mode,
        adaptive_z_threshold=args.suffix_v1_2_2_adaptive_z_threshold,
        adaptive_rise_z_threshold=args.suffix_v1_2_2_adaptive_rise_z_threshold,
        adaptive_min_std=args.suffix_v1_2_2_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_2_2_adaptive_min_points,
        hidden_weight_mode=args.suffix_v1_2_2_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_2_2_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_2_2_hidden_weight_floor,
        cosine_loss_weight=args.suffix_v1_2_2_cosine_loss_weight,
        relative_mse_loss_weight=args.suffix_v1_2_2_relative_mse_loss_weight,
        prox_weight=args.suffix_v1_2_2_prox_weight,
        range_weight=args.suffix_v1_2_2_range_weight,
        gradient_trend_stats_enabled=(
            args.suffix_v1_2_2_gradient_trend_stats_enabled
        ),
    )
    suffix_v1_2_3_config = SuffixV123Config(
        enabled=args.suffix_v1_2_3,
        log_enabled=args.suffix_v1_2_3_log,
        max_rounds=args.suffix_v1_2_3_max_rounds,
        epoch=args.suffix_v1_2_3_epoch,
        lr=args.suffix_v1_2_3_lr,
        embedding_relative_mse_high_threshold=(
            args.suffix_v1_2_3_embedding_relative_mse_high_threshold
        ),
        relative_mse_rise_threshold=(
            args.suffix_v1_2_3_relative_mse_rise_threshold
        ),
        token_relative_mse_high_threshold=(
            args.suffix_v1_2_3_token_relative_mse_high_threshold
        ),
        min_anomaly_reasons=args.suffix_v1_2_3_min_anomaly_reasons,
        min_relative_mse_improvement=(
            args.suffix_v1_2_3_min_relative_mse_improvement
        ),
        accuracy_tolerance=args.suffix_v1_2_3_accuracy_tolerance,
        accept_mode=args.suffix_v1_2_3_accept_mode,
        anomaly_detection_mode=args.suffix_v1_2_3_anomaly_detection_mode,
        adaptive_z_threshold=args.suffix_v1_2_3_adaptive_z_threshold,
        adaptive_rise_z_threshold=args.suffix_v1_2_3_adaptive_rise_z_threshold,
        adaptive_min_std=args.suffix_v1_2_3_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_2_3_adaptive_min_points,
        hidden_weight_mode=args.suffix_v1_2_3_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_2_3_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_2_3_hidden_weight_floor,
        cosine_loss_weight=args.suffix_v1_2_3_cosine_loss_weight,
        relative_mse_loss_weight=(
            args.suffix_v1_2_3_relative_mse_loss_weight
        ),
        prox_weight=args.suffix_v1_2_3_prox_weight,
        range_weight=args.suffix_v1_2_3_range_weight,
        gradient_trend_stats_enabled=(
            args.suffix_v1_2_3_gradient_trend_stats_enabled
        ),
    )
    suffix_reopt_v1_3_config = SuffixReoptimizationV13Config(
        enabled=args.suffix_reoptimization_v1_3,
        log_enabled=args.suffix_reoptimization_v1_3_log,
        max_rounds=args.suffix_v1_3_max_rounds,
        epoch=args.suffix_v1_3_epoch,
        lr=args.suffix_v1_3_lr,
        hidden_low_threshold=args.suffix_v1_3_hidden_low_threshold,
        hidden_drop_threshold=args.suffix_v1_3_hidden_drop_threshold,
        token_forward_low_threshold=args.suffix_v1_3_token_forward_low_threshold,
        min_anomaly_reasons=args.suffix_v1_3_min_anomaly_reasons,
        min_hidden_delta=args.suffix_v1_3_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_3_accuracy_tolerance,
        accept_mode=args.suffix_v1_3_accept_mode,
        anomaly_detection_mode=args.suffix_v1_3_anomaly_detection_mode,
        adaptive_z_threshold=args.suffix_v1_3_adaptive_z_threshold,
        adaptive_drop_z_threshold=args.suffix_v1_3_adaptive_drop_z_threshold,
        adaptive_min_std=args.suffix_v1_3_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_3_adaptive_min_points,
        hidden_weight_mode=args.suffix_v1_3_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_3_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_3_hidden_weight_floor,
        prox_weight=args.suffix_v1_3_prox_weight,
        manifold_weight=args.suffix_v1_3_manifold_weight,
        manifold_update_every=args.suffix_v1_3_manifold_update_every,
        manifold_warmup_epoch=args.suffix_v1_3_manifold_warmup_epoch,
        range_weight=args.suffix_v1_3_range_weight,
        anchor_mode=args.suffix_v1_3_anchor_mode,
    )
    suffix_reopt_v1_3_1_config = SuffixReoptimizationV131Config(
        enabled=args.suffix_reoptimization_v1_3_1,
        log_enabled=args.suffix_reoptimization_v1_3_1_log,
        max_rounds=args.suffix_v1_3_1_max_rounds,
        epoch=args.suffix_v1_3_1_epoch,
        lr=args.suffix_v1_3_1_lr,
        hidden_low_threshold=args.suffix_v1_3_1_hidden_low_threshold,
        hidden_drop_threshold=args.suffix_v1_3_1_hidden_drop_threshold,
        token_forward_low_threshold=args.suffix_v1_3_1_token_forward_low_threshold,
        min_anomaly_reasons=args.suffix_v1_3_1_min_anomaly_reasons,
        min_hidden_delta=args.suffix_v1_3_1_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_3_1_accuracy_tolerance,
        accept_mode=args.suffix_v1_3_1_accept_mode,
        anomaly_detection_mode=args.suffix_v1_3_1_anomaly_detection_mode,
        adaptive_z_threshold=args.suffix_v1_3_1_adaptive_z_threshold,
        adaptive_drop_z_threshold=args.suffix_v1_3_1_adaptive_drop_z_threshold,
        adaptive_min_std=args.suffix_v1_3_1_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_3_1_adaptive_min_points,
        hidden_weight_mode=args.suffix_v1_3_1_hidden_weight_mode,
        hidden_weight_decay=args.suffix_v1_3_1_hidden_weight_decay,
        hidden_weight_floor=args.suffix_v1_3_1_hidden_weight_floor,
        prox_weight=args.suffix_v1_3_1_prox_weight,
        range_weight=args.suffix_v1_3_1_range_weight,
    )
    suffix_reopt_v1_4_config = SuffixReoptimizationV14Config(
        enabled=args.suffix_reoptimization_v1_4,
        log_enabled=args.suffix_reoptimization_v1_4_log,
        coarse_lr_max=args.suffix_v1_4_coarse_lr_max,
        coarse_lr_min=args.suffix_v1_4_coarse_lr_min,
        coarse_schedule=args.suffix_v1_4_coarse_schedule,
        fine_epoch=args.suffix_v1_4_fine_epoch,
        fine_lr_max=args.suffix_v1_4_fine_lr_max,
        fine_lr_min=args.suffix_v1_4_fine_lr_min,
        fine_schedule=args.suffix_v1_4_fine_schedule,
        confidence_mode=args.suffix_v1_4_confidence_mode,
        confidence_continuous_min=args.suffix_v1_4_confidence_continuous_min,
        confidence_token_min=args.suffix_v1_4_confidence_token_min,
        confidence_margin_min=args.suffix_v1_4_confidence_margin_min,
        confidence_gap_max=args.suffix_v1_4_confidence_gap_max,
        confidence_percentile_min=args.suffix_v1_4_confidence_percentile_min,
        confidence_min_points=args.suffix_v1_4_confidence_min_points,
        require_candidate_agreement=args.suffix_v1_4_require_candidate_agreement,
        adaptive_z_threshold=args.suffix_v1_4_adaptive_z_threshold,
        adaptive_drop_z_threshold=args.suffix_v1_4_adaptive_drop_z_threshold,
        adaptive_min_std=args.suffix_v1_4_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_4_adaptive_min_points,
        fine_window=args.suffix_v1_4_fine_window,
        fine_window_decay=args.suffix_v1_4_fine_window_decay,
        prox_weight=args.suffix_v1_4_prox_weight,
        range_weight=args.suffix_v1_4_range_weight,
        min_hidden_delta=args.suffix_v1_4_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_4_accuracy_tolerance,
        accept_mode=args.suffix_v1_4_accept_mode,
    )
    if suffix_reopt_v1_4_config.enabled:
        validate_suffix_reoptimization_v1_4_config(suffix_reopt_v1_4_config)
    suffix_reopt_v1_4_1_config = SuffixReoptimizationV141Config(
        enabled=args.suffix_reoptimization_v1_4_1,
        log_enabled=args.suffix_reoptimization_v1_4_1_log,
        coarse_lr_max=args.suffix_v1_4_1_coarse_lr_max,
        coarse_lr_min=args.suffix_v1_4_1_coarse_lr_min,
        coarse_schedule=args.suffix_v1_4_1_coarse_schedule,
        fine_epoch=args.suffix_v1_4_1_fine_epoch,
        fine_lr_max=args.suffix_v1_4_1_fine_lr_max,
        fine_lr_min=args.suffix_v1_4_1_fine_lr_min,
        fine_schedule=args.suffix_v1_4_1_fine_schedule,
        confidence_mode=args.suffix_v1_4_1_confidence_mode,
        confidence_continuous_min=args.suffix_v1_4_1_confidence_continuous_min,
        confidence_token_min=args.suffix_v1_4_1_confidence_token_min,
        confidence_margin_min=args.suffix_v1_4_1_confidence_margin_min,
        confidence_gap_max=args.suffix_v1_4_1_confidence_gap_max,
        require_candidate_agreement=args.suffix_v1_4_1_require_candidate_agreement,
        adaptive_z_threshold=args.suffix_v1_4_1_adaptive_z_threshold,
        adaptive_drop_z_threshold=args.suffix_v1_4_1_adaptive_drop_z_threshold,
        adaptive_min_std=args.suffix_v1_4_1_adaptive_min_std,
        adaptive_min_points=args.suffix_v1_4_1_adaptive_min_points,
        fine_window=args.suffix_v1_4_1_fine_window,
        fine_window_decay=args.suffix_v1_4_1_fine_window_decay,
        prox_weight=args.suffix_v1_4_1_prox_weight,
        range_weight=args.suffix_v1_4_1_range_weight,
        min_hidden_delta=args.suffix_v1_4_1_min_hidden_delta,
        accuracy_tolerance=args.suffix_v1_4_1_accuracy_tolerance,
        accept_mode=args.suffix_v1_4_1_accept_mode,
    )
    if suffix_reopt_v1_4_1_config.enabled:
        validate_suffix_reoptimization_v1_4_1_config(suffix_reopt_v1_4_1_config)
        args.suffix_v1_4_1_confidence_mode = suffix_reopt_v1_4_1_config.confidence_mode
    suffix_reopt_v2_0_config = SuffixReoptimizationV20Config(
        enabled=args.suffix_reoptimization_v2_0,
        log_enabled=args.suffix_reoptimization_v2_0_log,
        layer_offsets=tuple(args.suffix_v2_0_layer_offsets),
        layer_weights=tuple(args.suffix_v2_0_layer_weights),
        epsilon=args.suffix_v2_0_epsilon,
        phase1_epoch=args.suffix_v2_0_phase1_epoch,
        phase1_lr=args.suffix_v2_0_phase1_lr,
        phase1_direction_weight=args.suffix_v2_0_phase1_direction_weight,
        phase1_magnitude_weight=args.suffix_v2_0_phase1_magnitude_weight,
        phase2_epoch=args.suffix_v2_0_phase2_epoch,
        phase2_lr=args.suffix_v2_0_phase2_lr,
        phase2_direction_weight=args.suffix_v2_0_phase2_direction_weight,
        phase2_magnitude_weight=args.suffix_v2_0_phase2_magnitude_weight,
        score_direction_weight=args.suffix_v2_0_score_direction_weight,
        score_magnitude_weight=args.suffix_v2_0_score_magnitude_weight,
        prox_weight=args.suffix_v2_0_prox_weight,
        range_weight=args.suffix_v2_0_range_weight,
        continuous_mad_multiplier=args.suffix_v2_0_continuous_mad_multiplier,
        local_discrete_mad_multiplier=args.suffix_v2_0_local_discrete_mad_multiplier,
        local_gap_jump_mad_multiplier=args.suffix_v2_0_local_gap_jump_mad_multiplier,
        mad_epsilon=args.suffix_v2_0_mad_epsilon,
        local_min_points=args.suffix_v2_0_local_min_points,
        normal_embedding_top_k=args.suffix_v2_0_normal_embedding_top_k,
        expanded_embedding_top_k=args.suffix_v2_0_expanded_embedding_top_k,
        ppl_top_k=args.suffix_v2_0_ppl_top_k,
        classifier_top_k=args.suffix_v2_0_classifier_top_k,
        cumulative_min_points=args.suffix_v2_0_cumulative_min_points,
        cumulative_kappa=args.suffix_v2_0_cumulative_kappa,
        cumulative_threshold=args.suffix_v2_0_cumulative_threshold,
        replace_epsilon=args.suffix_v2_0_replace_epsilon,
        cumulative_max_repairs_per_trigger=(
            args.suffix_v2_0_cumulative_max_repairs_per_trigger
        ),
        accuracy_diagnostics_enabled=(
            args.suffix_v2_0_accuracy_diagnostics_enabled
        ),
        classifier_enabled=args.suffix_v2_0_classifier_enabled,
    )
    suffix_reopt_v2_1_config = SuffixReoptimizationV21Config(
        enabled=args.suffix_reoptimization_v2_1,
        log_enabled=args.suffix_reoptimization_v2_1_log,
        layer_offsets=tuple(args.suffix_v2_1_layer_offsets),
        layer_weights=tuple(args.suffix_v2_1_layer_weights),
        alpha_dir=args.suffix_v2_1_alpha_dir,
        alpha_mag=args.suffix_v2_1_alpha_mag,
        vocab_weight=args.suffix_v2_1_vocab_weight,
        vocab_temperature=args.suffix_v2_1_vocab_temperature,
        vocab_anchor_top_k=args.suffix_v2_1_vocab_anchor_top_k,
        vocab_anchor_refresh_interval=(
            args.suffix_v2_1_vocab_anchor_refresh_interval
        ),
        global_optimizer=args.suffix_v2_1_global_optimizer,
        global_steps=args.suffix_v2_1_global_steps,
        global_lr=args.suffix_v2_1_global_lr,
        local_optimizer=args.suffix_v2_1_local_optimizer,
        local_steps=args.suffix_v2_1_local_steps,
        local_lr=args.suffix_v2_1_local_lr,
        adam_beta1=args.suffix_v2_1_adam_beta1,
        adam_beta2=args.suffix_v2_1_adam_beta2,
        adam_epsilon=args.suffix_v2_1_adam_epsilon,
        weight_decay_enabled=args.suffix_v2_1_weight_decay_enabled,
        scheduler_mode=args.suffix_v2_1_scheduler_mode,
        tau_J=args.suffix_v2_1_tau_J,
        delta_c_max=args.suffix_v2_1_delta_c_max,
        tau_r=args.suffix_v2_1_tau_r,
        embedding_top_k_normal=args.suffix_v2_1_embedding_top_k_normal,
        embedding_top_k_expanded=args.suffix_v2_1_embedding_top_k_expanded,
        ppl_top_k=args.suffix_v2_1_ppl_top_k,
        vocab_distance_mode=args.suffix_v2_1_vocab_distance_mode,
        vocab_softmin_mode=args.suffix_v2_1_vocab_softmin_mode,
        candidate_tie_break_mode=(
            args.suffix_v2_1_candidate_tie_break_mode
        ),
        hidden_epsilon=args.suffix_v2_1_hidden_epsilon,
        epsilon_J=args.suffix_v2_1_epsilon_J,
        epsilon_d=args.suffix_v2_1_epsilon_d,
        accuracy_diagnostics_enabled=(
            args.suffix_v2_1_accuracy_diagnostics_enabled
        ),
        filter_nonascii=args.suffix_v2_1_filter_nonascii,
    )
    selected_advanced_method = select_advanced_method(
        args.suffix_version,
        suffix_reopt_v1_2_1_config,
        suffix_reopt_v1_2_config,
        suffix_reopt_v1_1_config,
        suffix_reopt_v1_0_config,
        suffix_reopt_v1_3_config=suffix_reopt_v1_3_config,
        suffix_reopt_v1_4_config=suffix_reopt_v1_4_config,
        suffix_reopt_v1_4_1_config=suffix_reopt_v1_4_1_config,
        suffix_reopt_v1_2_2_config=suffix_reopt_v1_2_2_config,
        suffix_reopt_v1_3_1_config=suffix_reopt_v1_3_1_config,
        suffix_v1_2_3_config=suffix_v1_2_3_config,
        suffix_reopt_v2_0_config=suffix_reopt_v2_0_config,
        suffix_reopt_v2_1_config=suffix_reopt_v2_1_config,
    )
    selected_candidate_reranking_method = select_cgmr_method(
        args.cgmr_version,
        cgmr_v1_0_config,
        cgmr_v1_1_config,
        cgmr_v1_2_config,
    )
    validate_advanced_candidate_combination(
        selected_advanced_method,
        selected_candidate_reranking_method,
    )
    args.resolved_cgmr_version = normalize_cgmr_version(args.cgmr_version)
    args.selected_advanced_method = selected_advanced_method
    args.selected_candidate_reranking_method = selected_candidate_reranking_method
    timestamp = (
        str(worker_spec.get("parent_timestamp"))
        if worker_spec is not None
        else time.strftime("%Y%m%d-%H%M%S")
    )
    method_directory_name = experiment_method_directory_name(
        selected_advanced_method,
        selected_candidate_reranking_method,
    )
    if worker_spec is None:
        run_dir = os.path.join(args.log_dir, method_directory_name, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        experiment_log_path = os.path.join(run_dir, "experiment.log")
        reconstruction_path = os.path.join(run_dir, "reconstructions.jsonl")
    else:
        run_dir = worker_spec["output_dir"]
        os.makedirs(run_dir, exist_ok=True)
        experiment_log_path = os.devnull
        reconstruction_path = os.path.join(
            run_dir, "shard_reconstructions.jsonl"
        )
    with open(experiment_log_path, "w", encoding="utf-8") as txt_file, \
            open(reconstruction_path, "w", encoding="utf-8") as recon_file:
        '''load prompt datasets'''
        prompt_samples, dataset_parameters = load_dataset_samples(
            dataset_specs=dataset_specs,
            dataset_path=args.dataset_path,
            dataset_type=args.dataset_type,
            dataset_len=args.dataset_len,
        )
        if worker_spec is not None:
            if len(prompt_samples) != 10:
                raise ValueError("parallel v2.0 worker requires ten canonical samples")
            assigned = set(worker_spec["assigned_global_indices"])
            selected_samples = []
            for global_index, prompt_sample in enumerate(prompt_samples):
                if global_index not in assigned:
                    continue
                copied_sample = copy.deepcopy(prompt_sample)
                copied_sample["parallel_assignment"] = {
                    "global_index": global_index,
                    "worker_id": int(worker_spec["worker_id"]),
                    "physical_gpu_id": int(worker_spec["physical_gpu_id"]),
                }
                selected_samples.append(copied_sample)
            if [
                sample["parallel_assignment"]["global_index"]
                for sample in selected_samples
            ] != worker_spec["assigned_global_indices"]:
                raise ValueError("worker shard order does not match canonical sample order")
            prompt_samples = selected_samples
        primary_dataset = dataset_parameters[0]
        args.dataset_path = primary_dataset["path"]
        args.dataset_type = primary_dataset["type"]
        args.dataset_len = primary_dataset["len"]
        args.datasets = dataset_parameters
        total_samples = len(prompt_samples)

        model_config = AutoConfig.from_pretrained(
            args.base_model_name,
            trust_remote_code=True,
            local_files_only=os.path.exists(args.base_model_name),
        )
        validate_suffix_v21_model_config(
            selected_advanced_method,
            model_config,
        )
        model_layers = getattr(model_config, "num_hidden_layers", None)
        if model_layers is not None and (args.num_invert_layers < 0 or args.num_invert_layers >= model_layers):
            raise ValueError("--num-invert-layers must be in [0, {}] for this model".format(model_layers - 1))
        if model_layers is not None:
            cgmr_v1_0_effective_layers, cgmr_v1_0_effective_weights, cgmr_v1_0_filtered_layers = resolve_effective_layers(
                args.num_invert_layers,
                model_layers,
                cgmr_v1_0_config.layer_offsets,
                cgmr_v1_0_config.layer_weights,
            )
            cgmr_v1_1_effective_layers, cgmr_v1_1_effective_weights, cgmr_v1_1_filtered_layers = resolve_effective_layers(
                args.num_invert_layers,
                model_layers,
                cgmr_v1_1_config.layer_offsets,
                cgmr_v1_1_config.layer_weights,
            )
            cgmr_v1_2_effective_layers, cgmr_v1_2_effective_weights, cgmr_v1_2_filtered_layers = resolve_effective_layers(
                args.num_invert_layers,
                model_layers,
                cgmr_v1_2_config.layer_offsets,
                cgmr_v1_2_config.layer_weights,
            )
        else:
            cgmr_v1_0_effective_layers, cgmr_v1_0_effective_weights, cgmr_v1_0_filtered_layers = [], [], []
            cgmr_v1_1_effective_layers, cgmr_v1_1_effective_weights, cgmr_v1_1_filtered_layers = [], [], []
            cgmr_v1_2_effective_layers, cgmr_v1_2_effective_weights, cgmr_v1_2_filtered_layers = [], [], []
        if selected_candidate_reranking_method == "CGMR_v1.2":
            cgmr_effective_layers = cgmr_v1_2_effective_layers
            cgmr_effective_weights = cgmr_v1_2_effective_weights
            cgmr_filtered_layers = cgmr_v1_2_filtered_layers
        elif selected_candidate_reranking_method == "CGMR_v1.1":
            cgmr_effective_layers = cgmr_v1_1_effective_layers
            cgmr_effective_weights = cgmr_v1_1_effective_weights
            cgmr_filtered_layers = cgmr_v1_1_filtered_layers
        else:
            cgmr_effective_layers = cgmr_v1_0_effective_layers
            cgmr_effective_weights = cgmr_v1_0_effective_weights
            cgmr_filtered_layers = cgmr_v1_0_filtered_layers
        args.cgmr_v1_0_effective_layers = cgmr_v1_0_effective_layers
        args.cgmr_v1_0_effective_weights = cgmr_v1_0_effective_weights
        args.cgmr_v1_0_filtered_layers = cgmr_v1_0_filtered_layers
        args.cgmr_v1_1_effective_layers = cgmr_v1_1_effective_layers
        args.cgmr_v1_1_effective_weights = cgmr_v1_1_effective_weights
        args.cgmr_v1_1_filtered_layers = cgmr_v1_1_filtered_layers
        args.cgmr_v1_2_effective_layers = cgmr_v1_2_effective_layers
        args.cgmr_v1_2_effective_weights = cgmr_v1_2_effective_weights
        args.cgmr_v1_2_filtered_layers = cgmr_v1_2_filtered_layers
        resolved_config_path = os.path.join(run_dir, "resolved_config.json")
        resolved_config = build_resolved_config(
            args=args,
            timestamp=timestamp,
            run_dir=run_dir,
            experiment_log_path=experiment_log_path,
            reconstruction_path=reconstruction_path,
            summary_excel_path=None,
            total_samples=total_samples,
            model_config_layers=model_layers,
            model_type=getattr(model_config, "model_type", None),
            dataset_parameters=dataset_parameters,
        )
        if worker_spec is None:
            dump_json(resolved_config_path, resolved_config)

        '''get model'''
        console_update("loading checkpoint: loading model and tokenizer...")
        tokenizer, model = get_model(base_model_name=args.base_model_name,
                                     lora_model_name=args.lora_model_name,
                                     device_map=args.device_map,
                                     offload_folder=args.offload_folder,
                                     offload_state_dict=args.offload_state_dict,
                                     max_memory=args.max_memory,
                                     quantization=args.quantization)
        console_update("loading checkpoint: done")
        console_finish_progress()
        loaded_model_layers = get_model_layers(model)
        if args.num_invert_layers < 0 or args.num_invert_layers >= loaded_model_layers:
            raise ValueError("--num-invert-layers must be in [0, {}] for this model".format(loaded_model_layers - 1))
        resolved_config["model"]["loaded_layers"] = loaded_model_layers
        cgmr_v1_0_effective_layers, cgmr_v1_0_effective_weights, cgmr_v1_0_filtered_layers = resolve_effective_layers(
            args.num_invert_layers,
            loaded_model_layers,
            cgmr_v1_0_config.layer_offsets,
            cgmr_v1_0_config.layer_weights,
        )
        cgmr_v1_1_effective_layers, cgmr_v1_1_effective_weights, cgmr_v1_1_filtered_layers = resolve_effective_layers(
            args.num_invert_layers,
            loaded_model_layers,
            cgmr_v1_1_config.layer_offsets,
            cgmr_v1_1_config.layer_weights,
        )
        cgmr_v1_2_effective_layers, cgmr_v1_2_effective_weights, cgmr_v1_2_filtered_layers = resolve_effective_layers(
            args.num_invert_layers,
            loaded_model_layers,
            cgmr_v1_2_config.layer_offsets,
            cgmr_v1_2_config.layer_weights,
        )
        if selected_candidate_reranking_method == "CGMR_v1.2":
            cgmr_effective_layers = cgmr_v1_2_effective_layers
            cgmr_effective_weights = cgmr_v1_2_effective_weights
            cgmr_filtered_layers = cgmr_v1_2_filtered_layers
        elif selected_candidate_reranking_method == "CGMR_v1.1":
            cgmr_effective_layers = cgmr_v1_1_effective_layers
            cgmr_effective_weights = cgmr_v1_1_effective_weights
            cgmr_filtered_layers = cgmr_v1_1_filtered_layers
        else:
            cgmr_effective_layers = cgmr_v1_0_effective_layers
            cgmr_effective_weights = cgmr_v1_0_effective_weights
            cgmr_filtered_layers = cgmr_v1_0_filtered_layers
        args.cgmr_v1_0_effective_layers = cgmr_v1_0_effective_layers
        args.cgmr_v1_0_effective_weights = cgmr_v1_0_effective_weights
        args.cgmr_v1_0_filtered_layers = cgmr_v1_0_filtered_layers
        args.cgmr_v1_1_effective_layers = cgmr_v1_1_effective_layers
        args.cgmr_v1_1_effective_weights = cgmr_v1_1_effective_weights
        args.cgmr_v1_1_filtered_layers = cgmr_v1_1_filtered_layers
        args.cgmr_v1_2_effective_layers = cgmr_v1_2_effective_layers
        args.cgmr_v1_2_effective_weights = cgmr_v1_2_effective_weights
        args.cgmr_v1_2_filtered_layers = cgmr_v1_2_filtered_layers
        (
            suffix_v2_0_effective_layers,
            suffix_v2_0_effective_weights,
            suffix_v2_0_filtered_layers,
        ) = resolve_suffix_v2_0_effective_layers(
            args.num_invert_layers,
            loaded_model_layers,
            suffix_reopt_v2_0_config.layer_offsets,
            suffix_reopt_v2_0_config.layer_weights,
        )
        args.suffix_v2_0_effective_layers = suffix_v2_0_effective_layers
        args.suffix_v2_0_effective_weights = suffix_v2_0_effective_weights
        args.suffix_v2_0_filtered_layers = suffix_v2_0_filtered_layers
        (
            suffix_v2_1_effective_layers,
            suffix_v2_1_effective_weights,
            suffix_v2_1_filtered_layers,
        ) = resolve_suffix_v2_1_effective_layers(
            args.num_invert_layers,
            loaded_model_layers,
            suffix_reopt_v2_1_config.layer_offsets,
            suffix_reopt_v2_1_config.layer_weights,
        )
        args.suffix_v2_1_effective_layers = suffix_v2_1_effective_layers
        args.suffix_v2_1_effective_weights = suffix_v2_1_effective_weights
        args.suffix_v2_1_filtered_layers = suffix_v2_1_filtered_layers
        v2_resolved = resolved_config.get("advanced_methods", {}).get(
            "suffix_reoptimization_v2_0"
        )
        if v2_resolved is not None:
            v2_resolved["effective_layers"] = suffix_v2_0_effective_layers
            v2_resolved["effective_layer_weights"] = suffix_v2_0_effective_weights
            v2_resolved["filtered_layers"] = suffix_v2_0_filtered_layers
        v21_resolved = resolved_config.get("advanced_methods", {}).get(
            "suffix_reoptimization_v2_1"
        )
        if v21_resolved is not None:
            v21_resolved["effective_layers"] = suffix_v2_1_effective_layers
            v21_resolved["effective_layer_weights"] = suffix_v2_1_effective_weights
            v21_resolved["filtered_layers"] = suffix_v2_1_filtered_layers
        resolved_candidate_configs = resolved_config[
            "candidate_reranking_methods"
        ]
        for config_key, layers, weights, filtered in (
            (
                "cgmr_v1_0",
                cgmr_v1_0_effective_layers,
                cgmr_v1_0_effective_weights,
                cgmr_v1_0_filtered_layers,
            ),
            (
                "cgmr_v1_1",
                cgmr_v1_1_effective_layers,
                cgmr_v1_1_effective_weights,
                cgmr_v1_1_filtered_layers,
            ),
            (
                "cgmr_v1_2",
                cgmr_v1_2_effective_layers,
                cgmr_v1_2_effective_weights,
                cgmr_v1_2_filtered_layers,
            ),
        ):
            if config_key not in resolved_candidate_configs:
                continue
            resolved_candidate_configs[config_key]["effective_layers"] = layers
            resolved_candidate_configs[config_key]["effective_weights"] = weights
            resolved_candidate_configs[config_key]["filtered_layers"] = filtered
        if worker_spec is None:
            dump_json(resolved_config_path, resolved_config)
        run_records = []

        '''freeze model parameter'''
        for param in model.parameters():
            param.requires_grad = False

        embed_layer = get_input_embedding_layer(model)
        model_device = get_model_device(model)
        embed_dim = embed_layer.weight.shape[-1]

        '''get range'''
        embed_matrix = np.array(embed_layer.weight.data.cpu())
        left_range = get_sorted_top_k(embed_matrix, top_k=10, axis=0, reverse=False)
        left_range = torch.FloatTensor(left_range[0][-1]).type(torch.float16).to(model_device)
        right_range = get_sorted_top_k(embed_matrix, top_k=10, axis=0, reverse=True)
        right_range = torch.FloatTensor(right_range[0][-1]).type(torch.float16).to(model_device)

        cached_target_states = None
        if args.lora_model_name is not None:
            cached_target_states = []
            for sample_idx, prompt_sample in enumerate(prompt_samples):
                prompt_ = prompt_sample["text"]
                total_input_ids, total_attention_mask, _, all_hidden_states = get_hidden_state(
                    tokenizer,
                    model,
                    layer_id=args.num_invert_layers,
                    prompt=prompt_,
                    up_to=False,
                    selected_advanced_method=selected_advanced_method,
                )
                cgmr_target_hidden_states = None
                suffix_v2_0_target_hidden_states = None
                suffix_v2_1_target_hidden_states = None
                if selected_advanced_method == "suffix_reoptimization_v2.0":
                    suffix_v2_0_target_hidden_states = collect_hidden_states_by_layer(
                        model,
                        suffix_v2_0_effective_layers,
                        input_ids=total_input_ids,
                        attention_mask=total_attention_mask,
                    )
                if selected_advanced_method == "suffix_reoptimization_v2.1":
                    suffix_v2_1_target_hidden_states = collect_hidden_states_by_layer(
                        model,
                        suffix_v2_1_effective_layers,
                        input_ids=total_input_ids,
                        attention_mask=total_attention_mask,
                        use_cache=False,
                    )
                if (
                    (
                        selected_candidate_reranking_method
                        in ("CGMR_v1.0", "CGMR_v1.1")
                        and len(cgmr_effective_layers) >= 2
                    )
                    or (
                        selected_candidate_reranking_method == "CGMR_v1.2"
                        and len(cgmr_effective_layers) >= 1
                    )
                ):
                    cgmr_target_hidden_states = collect_hidden_states_by_layer(
                        model,
                        cgmr_effective_layers,
                        input_ids=total_input_ids,
                        attention_mask=total_attention_mask,
                    )
                cached_target_states.append((
                    total_input_ids,
                    total_attention_mask,
                    all_hidden_states[-1].detach().requires_grad_(False),
                    cgmr_target_hidden_states,
                    suffix_v2_0_target_hidden_states,
                    suffix_v2_1_target_hidden_states,
                ))

        if args.lora_model_name is not None:
            '''disable lora'''
            # model = model.merge_and_unload(progressbar=True)
            # model.unmerge_adapter()
            # model.merge_adapter()
            model.unload()

        for sample_idx, prompt_sample in enumerate(prompt_samples):
            sample_start = time.time()
            prompt = prompt_sample["text"]
            dataset_metadata = copy.deepcopy(prompt_sample["dataset"])
            prompt_text = str(prompt)
            if cached_target_states is None:
                total_input_ids, total_attention_mask, _, all_hidden_states = get_hidden_state(
                    tokenizer,
                    model,
                    layer_id=args.num_invert_layers,
                    prompt=prompt,
                    up_to=False,
                    selected_advanced_method=selected_advanced_method,
                )
                next_hidden_states_last = all_hidden_states[-1].detach().requires_grad_(False)
                cgmr_target_hidden_states = None
                suffix_v2_0_target_hidden_states = None
                suffix_v2_1_target_hidden_states = None
                if selected_advanced_method == "suffix_reoptimization_v2.0":
                    suffix_v2_0_target_hidden_states = collect_hidden_states_by_layer(
                        model,
                        suffix_v2_0_effective_layers,
                        input_ids=total_input_ids,
                        attention_mask=total_attention_mask,
                    )
                if selected_advanced_method == "suffix_reoptimization_v2.1":
                    suffix_v2_1_target_hidden_states = collect_hidden_states_by_layer(
                        model,
                        suffix_v2_1_effective_layers,
                        input_ids=total_input_ids,
                        attention_mask=total_attention_mask,
                        use_cache=False,
                    )
                if (
                    (
                        selected_candidate_reranking_method
                        in ("CGMR_v1.0", "CGMR_v1.1")
                        and len(cgmr_effective_layers) >= 2
                    )
                    or (
                        selected_candidate_reranking_method == "CGMR_v1.2"
                        and len(cgmr_effective_layers) >= 1
                    )
                ):
                    cgmr_target_hidden_states = collect_hidden_states_by_layer(
                        model,
                        cgmr_effective_layers,
                        input_ids=total_input_ids,
                        attention_mask=total_attention_mask,
                    )
                del all_hidden_states
            else:
                (
                    total_input_ids,
                    total_attention_mask,
                    next_hidden_states_last,
                    cgmr_target_hidden_states,
                    suffix_v2_0_target_hidden_states,
                    suffix_v2_1_target_hidden_states,
                ) = cached_target_states[sample_idx]

            prompt_length = len(total_input_ids[0])
            recover_length = prompt_length
            target_input_ids = total_input_ids
            target_attention_mask = total_attention_mask
            first_token_id = int(target_input_ids[0, 0].item())
            special_token_ids = set(tokenizer.all_special_ids)
            fixed_prefix = first_token_id in special_token_ids
            eval_start_pos = 1 if fixed_prefix else 0
            prefix_embed = None
            if fixed_prefix:
                prefix_embed = embed_layer(target_input_ids[:, :1]).detach()
                prefix_embed = prefix_embed.to(model_device)
                prefix_embed.requires_grad_(False)
            baseline_gradient_tracker = BaselineGradientTrendTracker(
                enabled=(
                    selected_advanced_method
                    in (
                        "suffix_reoptimization_v1.2.2",
                        "suffix_v1.2.3",
                    )
                    and (
                        suffix_v1_2_3_config.gradient_trend_stats_enabled
                        if selected_advanced_method
                        == "suffix_v1.2.3"
                        else suffix_reopt_v1_2_2_config.gradient_trend_stats_enabled
                    )
                ),
                position_offset=eval_start_pos,
            )

            '''define hyper-params'''
            loss_func = torch.nn.MSELoss(reduction='mean')
            lr = args.lr
            total_epoch = args.epoch
            alpha = args.alpha
            
            '''init input embed'''
            size = (prompt_length - eval_start_pos, embed_dim)
            if args.init_method == "gaussian":
                means = torch.zeros(size)
                new_input_embed_0 = torch.normal(mean=means, std=args.init_param)
                new_input_embed_0 = new_input_embed_0.unsqueeze(0).type(torch.float16)
            elif args.init_method == "uniform":
                new_input_embed_np = np.random.uniform(low=-args.init_param, high=args.init_param, size=size)
                new_input_embed_0 = torch.FloatTensor(new_input_embed_np)
            else:
                raise NotImplementedError
                
            new_input_embed_0 = new_input_embed_0.unsqueeze(dim=0).type(torch.float16).to(model_device)
            new_input_embed_0.requires_grad_(True)

            epochs = []
            loss_lst = []
            cos_sim_lst = []
            optimization_result = {}

            '''weighted average loss'''
            weight_mask = init_weight_mask(0, recover_length, method="linear", devices=[model_device])
            use_external_stage1 = (
                selected_advanced_method
                in (
                    "suffix_reoptimization_v2.1",
                    "suffix_reoptimization_v2.0",
                    "suffix_v1.2.3",
                    "frozen_original_baseline",
                )
            )
            part_epoch = 0 if use_external_stage1 else total_epoch
            use_v1_4_coarse_stage = selected_advanced_method in (
                "suffix_reoptimization_v1.4.1",
                "suffix_reoptimization_v1.4",
            )
            active_v1_4_config = (
                suffix_reopt_v1_4_1_config
                if selected_advanced_method == "suffix_reoptimization_v1.4.1"
                else suffix_reopt_v1_4_config
            )
            active_v1_4_schedule = (
                suffix_v1_4_1_scheduled_learning_rate
                if selected_advanced_method == "suffix_reoptimization_v1.4.1"
                else suffix_v1_4_scheduled_learning_rate
            )
            v1_4_coarse_optimizer = None
            v1_4_coarse_objective_history = []
            v1_4_coarse_lr_history = []
            if use_v1_4_coarse_stage:
                v1_4_coarse_optimizer = torch.optim.SGD(
                    [new_input_embed_0],
                    lr=active_v1_4_config.coarse_lr_max,
                )
            
            '''start timer'''
            start = time.time()
            last_optimization_percent = 0
            console_update("sample {} | optimizing {} {:>3}%".format(
                sample_idx + 1,
                format_progress_bar(last_optimization_percent),
                last_optimization_percent))

            for epoch_idx in range(part_epoch):
                '''clip embedding'''
                if args.clip:
                    with torch.no_grad():
                        clip_range = 0.2
                        if use_v1_4_coarse_stage:
                            new_input_embed_0.clamp_(-clip_range, clip_range)
                        else:
                            new_input_embed_0 = torch.clip(new_input_embed_0, -clip_range, clip_range)
                new_input_embed_0 = new_input_embed_0.requires_grad_(True)
                if use_v1_4_coarse_stage:
                    coarse_lr = active_v1_4_schedule(
                        epoch_idx,
                        part_epoch,
                        active_v1_4_config.coarse_lr_max,
                        active_v1_4_config.coarse_lr_min,
                        active_v1_4_config.coarse_schedule,
                    )
                    for param_group in v1_4_coarse_optimizer.param_groups:
                        param_group["lr"] = coarse_lr
                    optim = v1_4_coarse_optimizer
                else:
                    coarse_lr = lr
                    optim = torch.optim.SGD([new_input_embed_0], lr=lr)

                if fixed_prefix:
                    new_input_embed_ = torch.cat((prefix_embed, new_input_embed_0), dim=1)
                else:
                    new_input_embed_ = new_input_embed_0

                '''||phi(relaxed(Z, T)) - phi(x*)||**2'''
                new_inputs = {'inputs_embeds': new_input_embed_, 'attention_mask': target_attention_mask} 

                hidden_state_list = []
                def forward_hook(module, input, output):
                    if isinstance(output, tuple):
                        for item in output:
                            hidden_state_list.append(item)
                    else:
                        hidden_state_list.append(output)
        
                '''get hidden states from target layer'''
                hook_handles = register_layer_hooks(model, args.num_invert_layers, forward_hook, up_to=False)
                phi_relaxed = model(**new_inputs)
                for handle in hook_handles:
                    handle.remove()
                if not hidden_state_list:
                    raise ValueError("no hidden states collected for layer {}".format(args.num_invert_layers))
                last_hidden_state = hidden_state_list[0]
                hidden_state_list = []

                '''compute mse loss'''
                next_hidden_states_last = next_hidden_states_last.to(last_hidden_state.device)
                loss_mse = loss_func(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32))

                '''compute similarity'''
                cos_sim = F.cosine_similarity(
                    last_hidden_state.type(torch.float32),
                    next_hidden_states_last.type(torch.float32),
                    dim=-1)

                '''backward'''
                optim.zero_grad()
                cos_sim = cos_sim.to(weight_mask.device)
                sum_cos_sim = ((-cos_sim) * weight_mask).sum()

                relu_loss = F.relu(torch.abs(new_input_embed_) - right_range).sum()
                loss = sum_cos_sim + alpha * relu_loss 
                coarse_objective = loss_mse if args.optim_method == "MSELoss" else loss
                if use_v1_4_coarse_stage and not bool(torch.isfinite(coarse_objective).detach().cpu()):
                    console_finish_progress()
                    print(console_safe_text("encounter non-finite coarse loss"), flush=True)
                    break
                if args.optim_method == "MSELoss":
                    loss_mse.backward(inputs=[new_input_embed_0])    # optimize by MSELoss
                elif args.optim_method == "cosine":
                    loss.backward(inputs=[new_input_embed_0])  # optimize by cosine sim
                if selected_advanced_method == "suffix_reoptimization_v1.2.2":
                    baseline_gradient_tracker.observe(new_input_embed_0.grad)
                
                if torch.any(torch.isnan(cos_sim)):
                    console_finish_progress()
                    print(console_safe_text("encounter NAN"), flush=True)
                    break
                optim.step()
                if use_v1_4_coarse_stage:
                    v1_4_coarse_objective_history.append(
                        float(coarse_objective.detach().cpu())
                    )
                    v1_4_coarse_lr_history.append(float(coarse_lr))
                epochs.append(epoch_idx)                              # for loss graph
                loss_lst.append(relu_loss.data.cpu())                 # for loss graph
                cos_sim_lst.append(sum_cos_sim.data.cpu())            # for loss graph
                optimization_percent = int((epoch_idx + 1) * 100 / part_epoch)
                if optimization_percent != last_optimization_percent:
                    last_optimization_percent = optimization_percent
                    console_update("sample {} | optimizing {} {:>3}%".format(
                        sample_idx + 1,
                        format_progress_bar(last_optimization_percent),
                        last_optimization_percent))

                if epoch_idx == part_epoch - 1:
                    end = time.time()
                    final_input_embed = torch.cat((prefix_embed, new_input_embed_0), dim=1) if fixed_prefix else new_input_embed_0
                    opt_acc, opt_tokens, opt_list = invert_embedding(
                        final_input_embed, 
                        tokenizer, 
                        embed_layer, 
                        total_input_ids, 
                        invert_method=args.invert_method,
                        filter_nonascii=args.filter_nonascii,
                        f=None,
                        console=False,
                        eval_start_pos=eval_start_pos)
                    optimization_result = {
                        "epoch": epoch_idx,
                        "acc": opt_acc,
                        "cos_sim_mean": float(cos_sim.mean().detach().cpu()),
                        "relu_loss": float(relu_loss.detach().cpu()),
                        "elapsed_seconds": end - start,
                        "tokens": opt_tokens,
                    }
                    
            '''best inversion policy'''
            final_input_embed = torch.cat((prefix_embed, new_input_embed_0), dim=1) if fixed_prefix else new_input_embed_0
            stage1_optimized_embedding = None
            frozen_original_baseline_result = {
                "name": "frozen_original_baseline",
                "version": "frozen-original-v1",
                "enabled": selected_advanced_method == "frozen_original_baseline",
                "skipped": selected_advanced_method != "frozen_original_baseline",
                "reason": "not selected",
            }
            suffix_reopt_v2_0_result = {
                "name": "suffix_reoptimization_v2.0",
                "method": "suffix_reoptimization_v2.0",
                "version": "v2.0",
                "enabled": args.suffix_reoptimization_v2_0,
                "skipped": selected_advanced_method != "suffix_reoptimization_v2.0",
                "reason": (
                    "disabled" if not args.suffix_reoptimization_v2_0
                    else "not selected"
                ),
                "classifier_enabled": args.suffix_v2_0_classifier_enabled,
                "classifier_provider_available": False,
                "classifier_candidate_count": 0,
            }
            suffix_reopt_v2_1_result = {
                "name": "suffix_reoptimization_v2.1",
                "method": "suffix_reoptimization_v2.1",
                "version": "v2.1",
                "enabled": args.suffix_reoptimization_v2_1,
                "skipped": selected_advanced_method != "suffix_reoptimization_v2.1",
                "reason": (
                    "disabled" if not args.suffix_reoptimization_v2_1
                    else "not selected"
                ),
                "events": [],
            }
            suffix_v1_2_3_result = {
                "name": "suffix_v1.2.3",
                "method": "suffix_v1.2.3",
                "version": "v1.2.3",
                "enabled": args.suffix_v1_2_3,
                "skipped": (
                    selected_advanced_method
                    != "suffix_v1.2.3"
                ),
                "reason": (
                    "disabled"
                    if not args.suffix_v1_2_3
                    else "not selected"
                ),
                "events": [],
            }
            if selected_advanced_method == "suffix_reoptimization_v2.1":
                fixed_prefix_tokens = [
                    int(value)
                    for value in total_input_ids[0, :eval_start_pos]
                    .detach().cpu().tolist()
                ]
                entry_snapshot_tokens = build_suffix_v2_1_entry_snapshot(
                    final_input_embed,
                    embed_layer,
                    tokenizer,
                    eval_start_pos=eval_start_pos,
                    fixed_prefix_tokens=fixed_prefix_tokens,
                    attention_mask=target_attention_mask,
                    filter_nonascii=suffix_reopt_v2_1_config.filter_nonascii,
                )
                (
                    final_input_embed,
                    suffix_reopt_v2_1_result,
                ) = run_suffix_reoptimization_v2_1(
                    model=model,
                    embed_layer=embed_layer,
                    entry_embedding_snapshot=final_input_embed,
                    entry_token_snapshot=entry_snapshot_tokens,
                    target_hidden_states=suffix_v2_1_target_hidden_states,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    model_layer_count=loaded_model_layers,
                    tokenizer=tokenizer,
                    config=suffix_reopt_v2_1_config,
                    total_input_ids=total_input_ids,
                    eval_start_pos=eval_start_pos,
                    log_file=None,
                )
                stage1_optimized_embedding = final_input_embed
                optimization_result = {
                    "epoch": suffix_reopt_v2_1_config.global_steps,
                    "acc": suffix_reopt_v2_1_result.get("pre_acc"),
                    "cos_sim_mean": None,
                    "relu_loss": None,
                    "elapsed_seconds": time.time() - start,
                    "tokens": suffix_reopt_v2_1_result.get("final_text"),
                    "stage": "suffix_v2_1_global_and_causal",
                    "version": "v2.1",
                }
            elif selected_advanced_method == "suffix_reoptimization_v2.0":
                fixed_prefix_tokens = [
                    int(value)
                    for value in total_input_ids[0, :eval_start_pos]
                    .detach().cpu().tolist()
                ]
                entry_snapshot_tokens = build_entry_snapshot_from_embedding(
                    final_input_embed,
                    embed_layer,
                    tokenizer,
                    args.invert_method,
                    args.filter_nonascii,
                    eval_start_pos,
                    fixed_prefix_tokens,
                    embedding_top_indices,
                    select_candidate_from_top_indices,
                )
                (
                    final_input_embed,
                    suffix_reopt_v2_0_result,
                ) = run_suffix_reoptimization_v2_0(
                    model=model,
                    embed_layer=embed_layer,
                    initial_optimizable_embedding=new_input_embed_0,
                    prefix_embedding=prefix_embed,
                    target_hidden_states=suffix_v2_0_target_hidden_states,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    model_layer_count=loaded_model_layers,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    right_range=right_range,
                    config=suffix_reopt_v2_0_config,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=(
                        select_candidate_from_top_indices
                    ),
                    get_perplexity=get_perplexity,
                    entry_tokens=entry_snapshot_tokens,
                    filter_nonascii=args.filter_nonascii,
                    classifier_provider=None,
                    eval_start_pos=eval_start_pos,
                    log_file=None,
                )
                stage1_optimized_embedding = final_input_embed
                optimization_result = {
                    "epoch": suffix_reopt_v2_0_config.phase1_epoch,
                    "acc": suffix_reopt_v2_0_result.get("pre_acc"),
                    "cos_sim_mean": None,
                    "relu_loss": None,
                    "elapsed_seconds": time.time() - start,
                    "tokens": suffix_reopt_v2_0_result.get("final_text"),
                    "stage": "suffix_v2_0_phase1_phase2",
                    "version": "v2.0",
                }
            elif selected_advanced_method == "frozen_original_baseline":
                (
                    final_input_embed,
                    frozen_original_baseline_result,
                ) = run_frozen_original_baseline(
                    model=model,
                    embed_layer=embed_layer,
                    initial_optimizable_embedding=new_input_embed_0,
                    prefix_embedding=prefix_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    right_range=right_range,
                    lr=args.lr,
                    epoch=args.epoch,
                    alpha=args.alpha,
                    clip=args.clip,
                    init_method=args.init_method,
                    init_param=args.init_param,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    eval_start_pos=eval_start_pos,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=(
                        forward_and_get_last_hidden_state
                    ),
                    log_file=None,
                )
                stage1_optimized_embedding = final_input_embed
                optimization_result = dict(
                    frozen_original_baseline_result.get(
                        "optimization_result",
                        {},
                    )
                )
                optimization_result.setdefault(
                    "acc",
                    frozen_original_baseline_result.get("accuracy"),
                )
            elif selected_advanced_method == "suffix_v1.2.3":
                (
                    final_input_embed,
                    stage1_optimized_embedding,
                    suffix_v1_2_3_result,
                ) = run_suffix_v1_2_3(
                    model=model,
                    embed_layer=embed_layer,
                    initial_optimizable_embedding=new_input_embed_0,
                    prefix_embedding=prefix_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    right_range=right_range,
                    config=suffix_v1_2_3_config,
                    stage1_lr=args.lr,
                    stage1_epoch=args.epoch,
                    stage1_range_weight=args.alpha,
                    stage1_clip=args.clip,
                    stage1_init_method=args.init_method,
                    stage1_init_param=args.init_param,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=(
                        select_candidate_from_top_indices
                    ),
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=(
                        forward_and_get_last_hidden_state
                    ),
                    log_file=None,
                )
                optimization_result = dict(
                    suffix_v1_2_3_result["stage1"]["optimization"]
                )
                optimization_result.update({
                    "stage": "stage1",
                    "version": "v1.2.3",
                    "acc": suffix_v1_2_3_result["stage1"]["accuracy"],
                    "tokens": suffix_v1_2_3_result["stage1"]["text"],
                })
            upstream_optimized_embedding = (
                (
                    stage1_optimized_embedding
                    if stage1_optimized_embedding is not None
                    else final_input_embed
                ).detach().clone()
            )
            stage1_gradient_trend_stats = (
                suffix_v1_2_3_result.get(
                    "reoptimization",
                    {},
                ).get(
                    "stage1_gradient_trend_stats"
                )
                if selected_advanced_method == "suffix_v1.2.3"
                else (
                    baseline_gradient_tracker.summary()
                    if selected_advanced_method
                    == "suffix_reoptimization_v1.2.2"
                    else None
                )
            )
            pre_advanced_acc = optimization_result.get("acc")
            if pre_advanced_acc is None:
                pre_advanced_acc, opt_tokens, opt_list = invert_embedding(
                    final_input_embed,
                    tokenizer,
                    embed_layer,
                    total_input_ids,
                    invert_method=args.invert_method,
                    filter_nonascii=args.filter_nonascii,
                    f=None,
                    console=False,
                    eval_start_pos=eval_start_pos)
                optimization_result = {
                    "epoch": None,
                    "acc": pre_advanced_acc,
                    "cos_sim_mean": None,
                    "relu_loss": None,
                    "elapsed_seconds": time.time() - start,
                    "tokens": opt_tokens,
                }
            if use_v1_4_coarse_stage:
                optimization_result["oracle_accuracy"] = optimization_result.pop(
                    "acc", pre_advanced_acc
                )
                coarse_tail = v1_4_coarse_objective_history[-10:]
                optimization_result.update({
                    "stage": "coarse",
                    "version": (
                        "v1.4.1"
                        if selected_advanced_method == "suffix_reoptimization_v1.4.1"
                        else "v1.4"
                    ),
                    "optimizer": "SGD",
                    "optimizer_persistent": True,
                    "schedule": active_v1_4_config.coarse_schedule,
                    "lr_start": (
                        v1_4_coarse_lr_history[0]
                        if v1_4_coarse_lr_history
                        else active_v1_4_config.coarse_lr_max
                    ),
                    "lr_end": (
                        v1_4_coarse_lr_history[-1]
                        if v1_4_coarse_lr_history
                        else active_v1_4_config.coarse_lr_min
                    ),
                    "objective_start": (
                        v1_4_coarse_objective_history[0]
                        if v1_4_coarse_objective_history else None
                    ),
                    "objective_end": (
                        v1_4_coarse_objective_history[-1]
                        if v1_4_coarse_objective_history else None
                    ),
                    "objective_min": (
                        min(v1_4_coarse_objective_history)
                        if v1_4_coarse_objective_history else None
                    ),
                    "objective_tail_std": (
                        float(np.std(coarse_tail)) if coarse_tail else None
                    ),
                    "loss_formula": (
                        "MSE(hidden, target_hidden)"
                        if args.optim_method == "MSELoss"
                        else "weighted_negative_cosine + alpha * range_loss"
                    ),
                })
            suffix_reopt_v1_4_1_result = {
                "name": "suffix_reoptimization_v1.4.1",
                "enabled": args.suffix_reoptimization_v1_4_1,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.4.1",
                "accept_mode": args.suffix_v1_4_1_accept_mode,
                "confidence_mode": args.suffix_v1_4_1_confidence_mode,
                "manifold_enabled": False,
                "manifold_weight": 0.0,
                "manifold_updates": 0,
                "reason": "disabled" if not args.suffix_reoptimization_v1_4_1 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_4_result = {
                "name": "suffix_reoptimization_v1.4",
                "enabled": args.suffix_reoptimization_v1_4,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.4",
                "accept_mode": args.suffix_v1_4_accept_mode,
                "confidence_mode": args.suffix_v1_4_confidence_mode,
                "manifold_enabled": False,
                "manifold_weight": 0.0,
                "manifold_updates": 0,
                "reason": "disabled" if not args.suffix_reoptimization_v1_4 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_3_result = {
                "name": "suffix_reoptimization_v1.3",
                "enabled": args.suffix_reoptimization_v1_3,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.3",
                "anchor_mode": args.suffix_v1_3_anchor_mode,
                "accept_mode": args.suffix_v1_3_accept_mode,
                "anomaly_detection_mode": args.suffix_v1_3_anomaly_detection_mode,
                "reason": "disabled" if not args.suffix_reoptimization_v1_3 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_3_1_result = {
                "name": "suffix_reoptimization_v1.3.1",
                "version": "v1.3.1",
                "enabled": args.suffix_reoptimization_v1_3_1,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.3.1",
                "accept_mode": args.suffix_v1_3_1_accept_mode,
                "anomaly_detection_mode": args.suffix_v1_3_1_anomaly_detection_mode,
                "reason": "disabled" if not args.suffix_reoptimization_v1_3_1 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_2_1_result = {
                "name": "suffix_reoptimization_v1.2.1",
                "enabled": args.suffix_reoptimization_v1_2_1,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.2.1",
                "accept_mode": args.suffix_v1_2_1_accept_mode,
                "anomaly_detection_mode": args.suffix_v1_2_1_anomaly_detection_mode,
                "manifold_enabled": False,
                "manifold_weight": 0.0,
                "manifold_updates": 0,
                "reason": "disabled" if not args.suffix_reoptimization_v1_2_1 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_2_2_result = {
                "name": "suffix_reoptimization_v1.2.2",
                "version": "v1.2.2",
                "enabled": args.suffix_reoptimization_v1_2_2,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.2.2",
                "accept_mode": args.suffix_v1_2_2_accept_mode,
                "anomaly_detection_mode": args.suffix_v1_2_2_anomaly_detection_mode,
                "reason": "disabled" if not args.suffix_reoptimization_v1_2_2 else "not selected",
                "events": [],
                "baseline_gradient_trend_stats": stage1_gradient_trend_stats,
            }
            suffix_reopt_v1_2_result = {
                "name": "suffix_reoptimization_v1.2",
                "enabled": args.suffix_reoptimization_v1_2,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.2",
                "accept_mode": args.suffix_v1_2_accept_mode,
                "anomaly_detection_mode": args.suffix_v1_2_anomaly_detection_mode,
                "reason": "disabled" if not args.suffix_reoptimization_v1_2 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_1_result = {
                "name": "suffix_reoptimization_v1.1",
                "enabled": args.suffix_reoptimization_v1_1,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.1",
                "accept_mode": args.suffix_v1_1_accept_mode,
                "reason": "disabled" if not args.suffix_reoptimization_v1_1 else "not selected",
                "events": [],
            }
            suffix_reopt_v1_0_result = {
                "name": "suffix_reoptimization_v1.0",
                "enabled": args.suffix_reoptimization_v1_0,
                "skipped": selected_advanced_method != "suffix_reoptimization_v1.0",
                "accept_mode": args.suffix_v1_0_accept_mode,
                "reason": "disabled" if not args.suffix_reoptimization_v1_0 else "not selected",
                "events": [],
            }
            cgmr_v1_0_result = {
                "name": "CGMR_v1.0",
                "enabled": args.cgmr_v1_0,
                "skipped": (
                    selected_candidate_reranking_method
                    != "CGMR_v1.0"
                ),
                "reason": (
                    "disabled"
                    if not args.cgmr_v1_0
                    else "not selected"
                ),
                "events": [],
            }
            cgmr_v1_1_result = {
                "name": "CGMR_v1.1",
                "enabled": args.cgmr_v1_1,
                "skipped": (
                    selected_candidate_reranking_method
                    != "CGMR_v1.1"
                ),
                "reason": (
                    "disabled"
                    if not args.cgmr_v1_1
                    else "not selected"
                ),
                "events": [],
            }
            cgmr_v1_2_result = {
                "name": "CGMR_v1.2",
                "enabled": args.cgmr_v1_2,
                "skipped": (
                    selected_candidate_reranking_method
                    != "CGMR_v1.2"
                ),
                "reason": (
                    "disabled"
                    if not args.cgmr_v1_2
                    else "not selected"
                ),
                "events": [],
            }
            if selected_advanced_method == "suffix_reoptimization_v2.0":
                selected_suffix_result = suffix_reopt_v2_0_result
                advanced_triggered = bool(
                    (selected_suffix_result.get("stage4") or {}).get("events")
                )
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get(
                    "post_acc", pre_advanced_acc
                )
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.4.1":
                if args.suffix_reoptimization_v1_4:
                    suffix_reopt_v1_4_result["reason"] = "not selected by suffix_version=v1.4.1"
                if args.suffix_reoptimization_v1_3:
                    suffix_reopt_v1_3_result["reason"] = "not selected by suffix_version=v1.4.1"
                if args.suffix_reoptimization_v1_2_1:
                    suffix_reopt_v1_2_1_result["reason"] = "not selected by suffix_version=v1.4.1"
                if args.suffix_reoptimization_v1_2:
                    suffix_reopt_v1_2_result["reason"] = "not selected by suffix_version=v1.4.1"
                if args.suffix_reoptimization_v1_1:
                    suffix_reopt_v1_1_result["reason"] = "not selected by suffix_version=v1.4.1"
                if args.suffix_reoptimization_v1_0:
                    suffix_reopt_v1_0_result["reason"] = "not selected by suffix_version=v1.4.1"
            elif selected_advanced_method == "suffix_reoptimization_v1.4":
                if args.suffix_reoptimization_v1_3:
                    suffix_reopt_v1_3_result["reason"] = "not selected by suffix_version=v1.4"
                if args.suffix_reoptimization_v1_2_1:
                    suffix_reopt_v1_2_1_result["reason"] = "not selected by suffix_version=v1.4"
                if args.suffix_reoptimization_v1_2:
                    suffix_reopt_v1_2_result["reason"] = "not selected by suffix_version=v1.4"
                if args.suffix_reoptimization_v1_1:
                    suffix_reopt_v1_1_result["reason"] = "not selected by suffix_version=v1.4"
                if args.suffix_reoptimization_v1_0:
                    suffix_reopt_v1_0_result["reason"] = "not selected by suffix_version=v1.4"
            elif selected_advanced_method == "suffix_reoptimization_v1.3.1":
                pass
            elif selected_advanced_method == "suffix_reoptimization_v1.3":
                if args.suffix_reoptimization_v1_2_1:
                    suffix_reopt_v1_2_1_result["reason"] = "not selected by suffix_version=v1.3"
                if args.suffix_reoptimization_v1_2:
                    suffix_reopt_v1_2_result["reason"] = "not selected by suffix_version=v1.3"
                if args.suffix_reoptimization_v1_1:
                    suffix_reopt_v1_1_result["reason"] = "not selected by suffix_version=v1.3"
                if args.suffix_reoptimization_v1_0:
                    suffix_reopt_v1_0_result["reason"] = "not selected by suffix_version=v1.3"
            elif selected_advanced_method == "suffix_v1.2.3":
                pass
            elif selected_advanced_method == "suffix_reoptimization_v1.2.2":
                pass
            elif selected_advanced_method == "suffix_reoptimization_v1.2.1":
                if args.suffix_reoptimization_v1_3:
                    suffix_reopt_v1_3_result["reason"] = "not selected by suffix_version=v1.2.1"
                if args.suffix_reoptimization_v1_2:
                    suffix_reopt_v1_2_result["reason"] = "not selected by suffix_version=v1.2.1"
                if args.suffix_reoptimization_v1_1:
                    suffix_reopt_v1_1_result["reason"] = "not selected by suffix_version=v1.2.1"
                if args.suffix_reoptimization_v1_0:
                    suffix_reopt_v1_0_result["reason"] = "not selected by suffix_version=v1.2.1"
            elif selected_advanced_method == "suffix_reoptimization_v1.2":
                if args.suffix_reoptimization_v1_3:
                    suffix_reopt_v1_3_result["reason"] = "not selected by suffix_version=v1.2"
                if args.suffix_reoptimization_v1_2_1:
                    suffix_reopt_v1_2_1_result["reason"] = "not selected by suffix_version=v1.2"
                if args.suffix_reoptimization_v1_1:
                    suffix_reopt_v1_1_result["reason"] = "not selected by suffix_version=v1.2"
                if args.suffix_reoptimization_v1_0:
                    suffix_reopt_v1_0_result["reason"] = "not selected by suffix_version=v1.2"
            elif selected_advanced_method == "suffix_reoptimization_v1.1":
                if args.suffix_reoptimization_v1_3:
                    suffix_reopt_v1_3_result["reason"] = "not selected by suffix_version=v1.1"
                if args.suffix_reoptimization_v1_2_1:
                    suffix_reopt_v1_2_1_result["reason"] = "not selected by suffix_version=v1.1"
                if args.suffix_reoptimization_v1_2:
                    suffix_reopt_v1_2_result["reason"] = "not selected by suffix_version=v1.1"
                if args.suffix_reoptimization_v1_0:
                    suffix_reopt_v1_0_result["reason"] = "not selected by suffix_version=v1.1"
            elif selected_advanced_method == "suffix_reoptimization_v1.0":
                if args.suffix_reoptimization_v1_3:
                    suffix_reopt_v1_3_result["reason"] = "not selected by suffix_version=v1.0"
                if args.suffix_reoptimization_v1_2_1:
                    suffix_reopt_v1_2_1_result["reason"] = "not selected by suffix_version=v1.0"
                if args.suffix_reoptimization_v1_2:
                    suffix_reopt_v1_2_result["reason"] = "not selected by suffix_version=v1.0"
            advanced_reason = "advanced methods disabled"
            advanced_triggered = False
            post_advanced_acc = pre_advanced_acc
            acc = None
            ret_tokens = None
            ret_list = None
            selected_suffix_result = None

            if selected_advanced_method == "suffix_reoptimization_v2.1":
                selected_suffix_result = suffix_reopt_v2_1_result
                advanced_triggered = bool(
                    selected_suffix_result.get("triggered", False)
                )
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get(
                    "post_acc", pre_advanced_acc
                )
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
                if acc is None:
                    acc = evaluate_frozen_suffix_v21_accuracy(
                        selected_suffix_result,
                        total_input_ids,
                        eval_start_pos,
                    )
                    post_advanced_acc = acc
            elif selected_advanced_method == "suffix_reoptimization_v1.4.1":
                final_input_embed, suffix_reopt_v1_4_1_result = run_suffix_reoptimization_v1_4_1(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_4_1_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    coarse_stage_summary=optimization_result,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_4_1_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.4":
                final_input_embed, suffix_reopt_v1_4_result = run_suffix_reoptimization_v1_4(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_4_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    coarse_stage_summary=optimization_result,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_4_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.3.1":
                final_input_embed, suffix_reopt_v1_3_1_result = run_suffix_reoptimization_v1_3_1(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_3_1_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_3_1_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.3":
                final_input_embed, suffix_reopt_v1_3_result = run_suffix_reoptimization_v1_3(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_3_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_3_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_v1.2.3":
                selected_suffix_result = suffix_v1_2_3_result
                advanced_triggered = selected_suffix_result.get(
                    "triggered",
                    False,
                )
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get(
                    "post_acc",
                    pre_advanced_acc,
                )
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.2.2":
                final_input_embed, suffix_reopt_v1_2_2_result = run_suffix_reoptimization_v1_2_2(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_2_2_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    baseline_gradient_trend_stats=stage1_gradient_trend_stats,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_2_2_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.2.1":
                final_input_embed, suffix_reopt_v1_2_1_result = run_suffix_reoptimization_v1_2_1(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_2_1_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_2_1_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.2":
                final_input_embed, suffix_reopt_v1_2_result = run_suffix_reoptimization_v1_2(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_2_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_2_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.1":
                final_input_embed, suffix_reopt_v1_1_result = run_suffix_reoptimization_v1_1(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_1_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_1_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            elif selected_advanced_method == "suffix_reoptimization_v1.0":
                final_input_embed, suffix_reopt_v1_0_result = run_suffix_reoptimization_v1_0(
                    model=model,
                    embed_layer=embed_layer,
                    optimized_embedding=final_input_embed,
                    target_hidden_state=next_hidden_states_last,
                    attention_mask=target_attention_mask,
                    layer_id=args.num_invert_layers,
                    register_layer_hooks=register_layer_hooks,
                    tokenizer=tokenizer,
                    total_input_ids=total_input_ids,
                    config=suffix_reopt_v1_0_config,
                    filter_nonascii=args.filter_nonascii,
                    add_perplexity=args.perplexity,
                    top_k_ppl=args.top_k_ppl,
                    top_k_cos=args.top_k_cos,
                    invert_method=args.invert_method,
                    eval_start_pos=eval_start_pos,
                    embedding_top_indices=embedding_top_indices,
                    select_candidate_from_top_indices=select_candidate_from_top_indices,
                    get_perplexity=get_perplexity,
                    forward_and_get_last_hidden_state=forward_and_get_last_hidden_state,
                    log_file=None,
                )
                selected_suffix_result = suffix_reopt_v1_0_result
                advanced_triggered = selected_suffix_result.get("triggered", False)
                advanced_reason = selected_suffix_result.get("reason")
                post_advanced_acc = selected_suffix_result.get("post_acc", pre_advanced_acc)
                acc = selected_suffix_result.get("final_accuracy")
                ret_tokens = selected_suffix_result.get("final_text")
                ret_list = selected_suffix_result.get("final_tokens")
            if selected_advanced_method not in (
                "suffix_reoptimization_v2.1",
                "suffix_reoptimization_v2.0",
                "suffix_reoptimization_v1.4.1",
                "suffix_reoptimization_v1.4",
                "suffix_reoptimization_v1.3.1",
                "suffix_reoptimization_v1.3",
                "suffix_v1.2.3",
                "suffix_reoptimization_v1.2.2",
                "suffix_reoptimization_v1.2.1",
                "suffix_reoptimization_v1.2",
                "suffix_reoptimization_v1.1",
                "suffix_reoptimization_v1.0",
            ):
                if selected_candidate_reranking_method == "CGMR_v1.2":
                    acc, ret_tokens, ret_list = invert_embedding(
                        upstream_optimized_embedding,
                        tokenizer,
                        embed_layer,
                        total_input_ids,
                        invert_method=args.invert_method,
                        filter_nonascii=args.filter_nonascii,
                        f=None,
                        console=False,
                        eval_start_pos=eval_start_pos,
                    )
                elif selected_advanced_method == "frozen_original_baseline":
                    acc = frozen_original_baseline_result["accuracy"]
                    ret_tokens = frozen_original_baseline_result["final_text"]
                    ret_list = frozen_original_baseline_result["final_tokens"]
                else:
                    acc, ret_tokens, ret_list = invert_and_find_best(
                        final_input_embed,
                        next_hidden_states_last,
                        tokenizer,
                        model,
                        total_input_ids,
                        layer_id=args.num_invert_layers,
                        f=None,
                        invert_method=args.invert_method,
                        filter_nonascii=args.filter_nonascii,
                        add_perplexity=args.perplexity,
                        top_k_ppl=args.top_k_ppl,
                        top_k_cos=args.top_k_cos,
                        eval_start_pos=eval_start_pos)
                post_advanced_acc = acc
            cgmr_v1_2_input_token_source = None
            if selected_candidate_reranking_method == "CGMR_v1.2":
                if str(selected_advanced_method).startswith(
                        ("suffix_reoptimization_v", "suffix_v")):
                    cgmr_v1_2_input_token_source = "suffix_output"
                else:
                    cgmr_v1_2_input_token_source = "embedding_top1"
                fixed_prefix_tokens = [
                    int(token_id)
                    for token_id in total_input_ids[
                        0, :eval_start_pos
                    ].detach().cpu().tolist()
                ]
                if eval_start_pos:
                    ret_list[:eval_start_pos] = fixed_prefix_tokens
                candidate_reranking_pre_acc = _nonfixed_token_accuracy(
                    total_input_ids,
                    ret_list,
                    eval_start_pos,
                )
            else:
                fixed_prefix_tokens = None
                candidate_reranking_pre_acc = acc
            selected_cgmr_result = None
            selected_cgmr_config = None
            selected_cgmr_runner = None
            if selected_candidate_reranking_method == "CGMR_v1.0":
                selected_cgmr_config = cgmr_v1_0_config
                selected_cgmr_runner = run_cgmr_v1_0
            elif selected_candidate_reranking_method == "CGMR_v1.1":
                selected_cgmr_config = cgmr_v1_1_config
                selected_cgmr_runner = run_cgmr_v1_1
            elif selected_candidate_reranking_method == "CGMR_v1.2":
                selected_cgmr_config = cgmr_v1_2_config
                selected_cgmr_runner = run_cgmr_v1_2
            if selected_cgmr_runner is not None:
                if selected_candidate_reranking_method == "CGMR_v1.2":
                    ret_list, selected_cgmr_result = selected_cgmr_runner(
                        model=model,
                        embed_layer=embed_layer,
                        upstream_optimized_embedding=(
                            upstream_optimized_embedding
                        ),
                        current_tokens=ret_list,
                        target_hidden_states=cgmr_target_hidden_states or {},
                        attention_mask=target_attention_mask,
                        target_layer=args.num_invert_layers,
                        model_layer_count=loaded_model_layers,
                        model_device=model_device,
                        config=selected_cgmr_config,
                        tokenizer=tokenizer,
                        filter_nonascii=args.filter_nonascii,
                        add_perplexity=args.perplexity,
                        top_k_ppl=args.top_k_ppl,
                        top_k_cos=args.top_k_cos,
                        invert_method=args.invert_method,
                        eval_start_pos=eval_start_pos,
                        fixed_prefix_tokens=fixed_prefix_tokens,
                        input_token_source=cgmr_v1_2_input_token_source,
                        embedding_top_indices=embedding_top_indices,
                        select_candidate_from_top_indices=(
                            select_candidate_from_top_indices
                        ),
                        get_perplexity=get_perplexity,
                    )
                else:
                    ret_list, selected_cgmr_result = selected_cgmr_runner(
                        model=model,
                        embed_layer=embed_layer,
                        optimized_embedding=final_input_embed,
                        current_tokens=ret_list,
                        target_hidden_states=cgmr_target_hidden_states or {},
                        attention_mask=target_attention_mask,
                        target_layer=args.num_invert_layers,
                        model_layer_count=loaded_model_layers,
                        model_device=model_device,
                        config=selected_cgmr_config,
                        tokenizer=tokenizer,
                        filter_nonascii=args.filter_nonascii,
                        add_perplexity=args.perplexity,
                        top_k_ppl=args.top_k_ppl,
                        top_k_cos=args.top_k_cos,
                        invert_method=args.invert_method,
                        eval_start_pos=eval_start_pos,
                        embedding_top_indices=embedding_top_indices,
                        select_candidate_from_top_indices=(
                            select_candidate_from_top_indices
                        ),
                        get_perplexity=get_perplexity,
                    )
                if selected_candidate_reranking_method == "CGMR_v1.2":
                    cgmr_v1_2_result = selected_cgmr_result
                elif selected_candidate_reranking_method == "CGMR_v1.1":
                    cgmr_v1_1_result = selected_cgmr_result
                else:
                    cgmr_v1_0_result = selected_cgmr_result
                acc = _nonfixed_token_accuracy(
                    total_input_ids,
                    ret_list,
                    eval_start_pos,
                )
                ret_tokens = tokenizer.decode(torch.tensor(ret_list[eval_start_pos:]))
                selected_cgmr_result["pre_acc"] = candidate_reranking_pre_acc
                if selected_candidate_reranking_method == "CGMR_v1.2":
                    selected_cgmr_result["before_accuracy"] = (
                        candidate_reranking_pre_acc
                    )
                selected_cgmr_result["post_acc"] = acc
                selected_cgmr_result["final_accuracy"] = acc
                selected_cgmr_result["final_text"] = ret_tokens
                selected_cgmr_result["final_tokens"] = ret_list
            console_finish_progress()
            sample_elapsed = time.time() - sample_start
            suffix_reopt_result = (
                selected_suffix_result
                or suffix_reopt_v1_4_1_result
                or suffix_reopt_v1_4_result
                or suffix_reopt_v1_3_result
                or suffix_reopt_v1_2_1_result
                or suffix_reopt_v1_2_result
            )
            if selected_candidate_reranking_method == "CGMR_v1.2":
                candidate_reranking_result = cgmr_v1_2_result
            elif selected_candidate_reranking_method == "CGMR_v1.1":
                candidate_reranking_result = cgmr_v1_1_result
            elif selected_candidate_reranking_method == "CGMR_v1.0":
                candidate_reranking_result = cgmr_v1_0_result
            else:
                candidate_reranking_result = {
                    "name": "none",
                    "enabled": False,
                    "skipped": True,
                    "reason": "not selected",
                    "events": [],
                }
            if selected_candidate_reranking_method != "none":
                annotate_candidate_events_for_offline_evaluation(
                    candidate_reranking_result,
                    total_input_ids[0].detach().cpu().tolist(),
                )
            record = {
                "sample_index": sample_idx,
                "dataset": dataset_metadata,
                "original": prompt_text,
                "reconstructed": ret_tokens,
                "accuracy": acc,
                "optimization_result": optimization_result,
                "fixed_prefix": fixed_prefix,
                "eval_start_pos": eval_start_pos,
                "token_length": prompt_length,
                "elapsed_seconds": sample_elapsed,
                "method": selected_advanced_method,
                "version": suffix_reopt_result.get("version"),
                "accepted": bool(suffix_reopt_result.get("accepted", True)),
                "rollback": bool(suffix_reopt_result.get("rollback", False)),
                "fatal_failure": bool(
                    suffix_reopt_result.get("fatal_failure", False)
                ),
                "num_invert_layers": args.num_invert_layers,
                "selected_advanced_method": selected_advanced_method,
                "selected_candidate_reranking_method": selected_candidate_reranking_method,
                "frozen_original_baseline_result": (
                    frozen_original_baseline_result
                ),
                "suffix_reoptimization_v1_0_result": suffix_reopt_v1_0_result,
                "suffix_reoptimization_v1_1_result": suffix_reopt_v1_1_result,
                "suffix_reoptimization_v1_2_result": suffix_reopt_v1_2_result,
                "suffix_reoptimization_v1_2_1_result": suffix_reopt_v1_2_1_result,
                "suffix_reoptimization_v1_3_result": suffix_reopt_v1_3_result,
                "suffix_reoptimization_v1_4_result": suffix_reopt_v1_4_result,
                "suffix_reoptimization_v1_4_1_result": suffix_reopt_v1_4_1_result,
                "suffix_reoptimization_v2_0_result": suffix_reopt_v2_0_result,
                "suffix_reoptimization_v2_1_result": suffix_reopt_v2_1_result,
                "suffix_v1_2_3_result": suffix_v1_2_3_result,
                "suffix_reoptimization_result": suffix_reopt_result,
                "cgmr_v1_0_result": cgmr_v1_0_result,
                "cgmr_v1_1_result": cgmr_v1_1_result,
                "cgmr_v1_2_result": cgmr_v1_2_result,
                "candidate_reranking_result": candidate_reranking_result,
                "candidate_reranking_method": {
                    "name": selected_candidate_reranking_method,
                    "enabled": selected_candidate_reranking_method != "none",
                    "pre_acc": candidate_reranking_result.get("pre_acc"),
                    "post_acc": candidate_reranking_result.get("post_acc"),
                    "triggered": candidate_reranking_result.get("triggered", False),
                    "accepted": candidate_reranking_result.get("accepted", False),
                    "reason": candidate_reranking_result.get("reason"),
                    "changed_positions": candidate_reranking_result.get("changed_positions", []),
                    "events": candidate_reranking_result.get("events", []),
                },
                "advanced_method": {
                    "name": selected_advanced_method,
                    "enabled": selected_advanced_method != "frozen_original_baseline",
                    "pre_acc": (
                        pre_advanced_acc
                        if selected_advanced_method
                        == "suffix_reoptimization_v2.1"
                        else suffix_reopt_result.get("pre_acc")
                        if selected_advanced_method in (
                            "suffix_reoptimization_v2.0",
                            "suffix_reoptimization_v1.4.1",
                            "suffix_reoptimization_v1.4",
                            "suffix_reoptimization_v1.3.1",
                            "suffix_reoptimization_v1.3",
                            "suffix_v1.2.3",
                            "suffix_reoptimization_v1.2.2",
                            "suffix_reoptimization_v1.2.1",
                            "suffix_reoptimization_v1.2",
                            "suffix_reoptimization_v1.1",
                            "suffix_reoptimization_v1.0",
                        )
                        else pre_advanced_acc
                    ),
                    "post_acc": post_advanced_acc,
                    "triggered": advanced_triggered,
                    "reason": advanced_reason,
                    "events": (
                        suffix_reopt_result.get("events", [])
                        if selected_advanced_method in (
                            "suffix_reoptimization_v2.1",
                            "suffix_reoptimization_v2.0",
                            "suffix_reoptimization_v1.4.1",
                            "suffix_reoptimization_v1.4",
                            "suffix_reoptimization_v1.3.1",
                            "suffix_reoptimization_v1.3",
                            "suffix_v1.2.3",
                            "suffix_reoptimization_v1.2.2",
                            "suffix_reoptimization_v1.2.1",
                            "suffix_reoptimization_v1.2",
                            "suffix_reoptimization_v1.1",
                            "suffix_reoptimization_v1.0",
                        )
                        else []
                    ),
                },
            }
            record.update(suffix_v2_classifier_record_fields(
                selected_advanced_method,
                suffix_reopt_v2_0_result,
            ))
            if worker_spec is not None:
                assignment = prompt_sample["parallel_assignment"]
                record.update({
                    "global_index": int(assignment["global_index"]),
                    "dataset_name": dataset_metadata["name"],
                    "dataset_sample_index": int(dataset_metadata["sample_index"]),
                    "assigned_worker_id": int(assignment["worker_id"]),
                    "assigned_physical_gpu_id": int(
                        assignment["physical_gpu_id"]
                    ),
                })
            if selected_advanced_method in (
                "suffix_reoptimization_v2.1",
                "suffix_reoptimization_v2.0",
                "suffix_v1.2.3",
                "suffix_reoptimization_v1.2.2",
                "suffix_reoptimization_v1.3.1",
            ):
                for legacy_result_key in (
                    "suffix_reoptimization_v1_0_result",
                    "suffix_reoptimization_v1_1_result",
                    "suffix_reoptimization_v1_2_result",
                    "suffix_reoptimization_v1_2_1_result",
                    "suffix_reoptimization_v1_3_result",
                    "suffix_reoptimization_v1_4_result",
                    "suffix_reoptimization_v1_4_1_result",
                ):
                    record.pop(legacy_result_key, None)
                if selected_advanced_method in (
                    "suffix_reoptimization_v2.1",
                    "suffix_reoptimization_v2.0",
                ):
                    record.pop("suffix_v1_2_3_result", None)
                if selected_advanced_method == "suffix_reoptimization_v2.1":
                    record.pop("suffix_reoptimization_v2_0_result", None)
                elif selected_advanced_method == "suffix_reoptimization_v2.0":
                    record.pop("suffix_reoptimization_v2_1_result", None)
                record["advanced_method"] = {
                    "name": selected_advanced_method,
                    "version": suffix_reopt_result.get("version"),
                    "enabled": True,
                }
            record["stage_accuracy"] = build_stage_accuracy(record)
            recon_file.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
            recon_file.flush()
            run_records.append(record)
            if worker_spec is None:
                write_experiment_sample_summary(
                    txt_file,
                    record,
                    sample_idx + 1,
                    total_samples,
                    prompt_length,
                )
            del (
                next_hidden_states_last,
                cgmr_target_hidden_states,
                suffix_v2_0_target_hidden_states,
                suffix_v2_1_target_hidden_states,
                new_input_embed_0,
                total_input_ids,
                total_attention_mask,
            )
            del target_input_ids, target_attention_mask, weight_mask, epochs, loss_lst, cos_sim_lst
            if prefix_embed is not None:
                del prefix_embed
            torch.cuda.empty_cache()

        if worker_spec is None:
            write_experiment_average_summary(txt_file, run_records)
            return experiment_exit_code_for_records(run_records)
        completed_indices = [
            int(record["global_index"]) for record in run_records
        ]
        fatal_records = [
            record for record in run_records
            if record_has_fatal_formal_failure(record)
        ]
        success = (
            completed_indices == worker_spec["assigned_global_indices"]
            and len(run_records) == len(worker_spec["assigned_global_indices"])
            and not fatal_records
        )
        failure_reason = None
        if not success:
            failure_reason = (
                "fatal sample failure"
                if fatal_records else "worker record count or order mismatch"
            )
        _write_parallel_worker_status(
            worker_spec,
            completed_indices,
            len(run_records),
            0 if success else 2,
            success,
            failure_reason,
            resolved_config=resolved_config,
        )
        return 0 if success else 2


if __name__ == "__main__":
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    if config_args.config:
        config = load_config(config_args.config)
    else:
        config = {}

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_args.config)
    parser.add_argument("--dataset-path", type=str, default=config.get("dataset_path", os.path.join(SCRIPT_DIR, "data", "financial.json")))
    parser.add_argument("--dataset-type", type=str, default=config.get("dataset_type", "local"),
                        choices=["local", "datasets", "github"])
    parser.add_argument("--dataset-len", type=int, default=config.get("dataset_len", 100))
    parser.set_defaults(datasets=config.get("datasets"))
    parser.add_argument("--base-model-name", type=str, default=config.get("base_model_name"))
    parser.add_argument("--lora-model-name", type=str, default=config.get("lora_model_name"))
    parser.add_argument("--lr", type=float, default=config.get("lr", 0.1))
    parser.add_argument("--epoch", type=int, default=config.get("epoch", 2000))
    parser.add_argument("--alpha", type=float, default=config.get("alpha", 1e-3))
    parser.add_argument("--clip", type=str2bool, default=config.get("clip", True))
    parser.add_argument("--num-invert-layers", type=int, default=config.get("num_invert_layers", 30))
    parser.add_argument("--init-method", type=str, default=config.get("init_method", "uniform"))
    parser.add_argument("--init-param", type=float, default=config.get("init_param", 0.1))
    parser.add_argument("--optim-method", type=str, default=config.get("optim_method", "cosine"),
                        choices=["cosine", "MSELoss"])
    parser.add_argument("--invert-method", type=str, default=config.get("invert_method", "cosine"),
                        choices=["cosine", "L2"])
    parser.add_argument("--show-low-confidence", type=str2bool, default=config.get("show_low_confidence", False))
    parser.add_argument("--fine-tune", type=str2bool, default=config.get("fine_tune", False))
    parser.add_argument("--filter-nonascii", type=str2bool, default=config.get("filter_nonascii", True))
    parser.add_argument("--perplexity", type=str2bool, default=config.get("perplexity", True))
    parser.add_argument("--top-k-ppl", type=int, default=config.get("top_k_ppl", 10))
    parser.add_argument("--top-k-cos", type=int, default=config.get("top_k_cos", 10))
    parser.add_argument("--suffix-version", type=str,
                        default=config.get("suffix_version"))
    parser.add_argument(
        "--suffix-reoptimization-version",
        dest="legacy_suffix_version",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cgmr-version", type=str,
                        default=config.get("cgmr_version"))
    parser.set_defaults(
        cgmr_v1_0=config.get("cgmr_v1_0", False),
        cgmr_v1_0_log=config.get("cgmr_v1_0_log", True),
        cgmr_v1_0_layer_offsets=config.get("cgmr_v1_0_layer_offsets", [0, 1, 2]),
        cgmr_v1_0_layer_weights=config.get("cgmr_v1_0_layer_weights", [0.5, 0.3, 0.2]),
        cgmr_v1_0_normalization=config.get("cgmr_v1_0_normalization", "zscore"),
        cgmr_v1_0_consistency_weight=config.get("cgmr_v1_0_consistency_weight", 0.0),
        cgmr_v1_0_strong_margin_threshold=config.get(
            "cgmr_v1_0_strong_margin_threshold", 0.01
        ),
        cgmr_v1_0_weak_margin_threshold=config.get("cgmr_v1_0_weak_margin_threshold", 0.02),
        cgmr_v1_0_low_score_threshold=config.get("cgmr_v1_0_low_score_threshold", 0.50),
        cgmr_v1_0_weak_signals_required=config.get("cgmr_v1_0_weak_signals_required", 2),
        cgmr_v1_0_max_candidates=config.get("cgmr_v1_0_max_candidates", 32),
        cgmr_v1_0_candidate_batch_size=config.get("cgmr_v1_0_candidate_batch_size", 16),
        cgmr_v1_0_min_enhanced_gain=config.get("cgmr_v1_0_min_enhanced_gain", 0.05),
        cgmr_v1_0_min_enhanced_margin=config.get("cgmr_v1_0_min_enhanced_margin", 0.05),
        cgmr_v1_0_max_layer_l_drop=config.get("cgmr_v1_0_max_layer_l_drop", 0.02),
        cgmr_v1_0_max_repair_steps=config.get("cgmr_v1_0_max_repair_steps", 5),
    )
    parser.set_defaults(
        cgmr_v1_1=config.get("cgmr_v1_1", False),
        cgmr_v1_1_log=config.get("cgmr_v1_1_log", True),
        cgmr_v1_1_layer_offsets=config.get("cgmr_v1_1_layer_offsets", [0, 1, 2]),
        cgmr_v1_1_layer_weights=config.get("cgmr_v1_1_layer_weights", [0.5, 0.3, 0.2]),
        cgmr_v1_1_normalization=config.get("cgmr_v1_1_normalization", "zscore"),
        cgmr_v1_1_consistency_weight=config.get("cgmr_v1_1_consistency_weight", 0.0),
        cgmr_v1_1_relative_margin_epsilon=config.get(
            "cgmr_v1_1_relative_margin_epsilon", 1e-6
        ),
        cgmr_v1_1_relative_margin_risk_weight=config.get(
            "cgmr_v1_1_relative_margin_risk_weight", 0.7
        ),
        cgmr_v1_1_low_score_risk_weight=config.get(
            "cgmr_v1_1_low_score_risk_weight", 0.3
        ),
        cgmr_v1_1_score_drop_risk_weight=config.get(
            "cgmr_v1_1_score_drop_risk_weight", 0.0
        ),
        cgmr_v1_1_low_score_threshold=config.get("cgmr_v1_1_low_score_threshold", 0.80),
        cgmr_v1_1_min_risk_score=config.get("cgmr_v1_1_min_risk_score", 0.20),
        cgmr_v1_1_risk_top_k=config.get("cgmr_v1_1_risk_top_k", 20),
        cgmr_v1_1_max_accepted_repairs=config.get(
            "cgmr_v1_1_max_accepted_repairs", 10
        ),
        cgmr_v1_1_max_candidates=config.get("cgmr_v1_1_max_candidates", 32),
        cgmr_v1_1_candidate_batch_size=config.get(
            "cgmr_v1_1_candidate_batch_size", 16
        ),
        cgmr_v1_1_min_enhanced_gain=config.get("cgmr_v1_1_min_enhanced_gain", 0.05),
        cgmr_v1_1_min_enhanced_margin=config.get(
            "cgmr_v1_1_min_enhanced_margin", 0.05
        ),
        cgmr_v1_1_max_layer_l_drop=config.get("cgmr_v1_1_max_layer_l_drop", 0.02),
    )
    parser.set_defaults(
        cgmr_v1_2=config.get("cgmr_v1_2", False),
        cgmr_v1_2_log=config.get("cgmr_v1_2_log", True),
        cgmr_v1_2_layer_offsets=config.get(
            "cgmr_v1_2_layer_offsets", [0, 1, 2]
        ),
        cgmr_v1_2_layer_weights=config.get(
            "cgmr_v1_2_layer_weights", [0.5, 0.3, 0.2]
        ),
        cgmr_v1_2_entropy_temperature=config.get(
            "cgmr_v1_2_entropy_temperature", 0.05
        ),
        cgmr_v1_2_effective_candidate_threshold=config.get(
            "cgmr_v1_2_effective_candidate_threshold", 1.5
        ),
        cgmr_v1_2_max_multilayer_candidates=config.get(
            "cgmr_v1_2_max_multilayer_candidates", 6
        ),
        cgmr_v1_2_lookahead_window=config.get(
            "cgmr_v1_2_lookahead_window", 1
        ),
        cgmr_v1_2_improvement_epsilon=config.get(
            "cgmr_v1_2_improvement_epsilon", 1e-6
        ),
        cgmr_v1_2_relative_mse_epsilon=config.get(
            "cgmr_v1_2_relative_mse_epsilon", 1e-8
        ),
        cgmr_v1_2_max_candidates=config.get(
            "cgmr_v1_2_max_candidates", 32
        ),
        cgmr_v1_2_candidate_batch_size=config.get(
            "cgmr_v1_2_candidate_batch_size", 16
        ),
    )
    parser.set_defaults(
        suffix_reoptimization_v1_0=config.get("suffix_reoptimization_v1_0", False),
        suffix_reoptimization_v1_0_log=config.get("suffix_reoptimization_v1_0_log", True),
        suffix_v1_0_max_rounds=config.get("suffix_v1_0_max_rounds", 2),
        suffix_v1_0_epoch=config.get("suffix_v1_0_epoch", 50),
        suffix_v1_0_lr=config.get("suffix_v1_0_lr", 0.03),
        suffix_v1_0_reg_weight=config.get("suffix_v1_0_reg_weight", 0.02),
        suffix_v1_0_hidden_low_threshold=config.get("suffix_v1_0_hidden_low_threshold", 0.50),
        suffix_v1_0_hidden_drop_threshold=config.get("suffix_v1_0_hidden_drop_threshold", 0.15),
        suffix_v1_0_token_forward_low_threshold=config.get("suffix_v1_0_token_forward_low_threshold", 0.50),
        suffix_v1_0_min_anomaly_reasons=config.get("suffix_v1_0_min_anomaly_reasons", 2),
        suffix_v1_0_min_hidden_delta=config.get("suffix_v1_0_min_hidden_delta", 0.005),
        suffix_v1_0_accuracy_tolerance=config.get("suffix_v1_0_accuracy_tolerance", 0.0),
        suffix_v1_0_accept_mode=config.get("suffix_v1_0_accept_mode", "oracle_accuracy"),
        suffix_reoptimization_v1_1=config.get("suffix_reoptimization_v1_1", False),
        suffix_reoptimization_v1_1_log=config.get("suffix_reoptimization_v1_1_log", True),
        suffix_v1_1_max_rounds=config.get("suffix_v1_1_max_rounds", 2),
        suffix_v1_1_epoch=config.get("suffix_v1_1_epoch", 50),
        suffix_v1_1_lr=config.get("suffix_v1_1_lr", 0.03),
        suffix_v1_1_hidden_low_threshold=config.get("suffix_v1_1_hidden_low_threshold", 0.50),
        suffix_v1_1_hidden_drop_threshold=config.get("suffix_v1_1_hidden_drop_threshold", 0.15),
        suffix_v1_1_token_forward_low_threshold=config.get("suffix_v1_1_token_forward_low_threshold", 0.50),
        suffix_v1_1_min_anomaly_reasons=config.get("suffix_v1_1_min_anomaly_reasons", 2),
        suffix_v1_1_min_hidden_delta=config.get("suffix_v1_1_min_hidden_delta", 0.005),
        suffix_v1_1_accuracy_tolerance=config.get("suffix_v1_1_accuracy_tolerance", 0.0),
        suffix_v1_1_accept_mode=config.get("suffix_v1_1_accept_mode", "oracle_accuracy"),
        suffix_v1_1_hidden_weight_mode=config.get("suffix_v1_1_hidden_weight_mode", "front_decay"),
        suffix_v1_1_hidden_weight_decay=config.get("suffix_v1_1_hidden_weight_decay", 0.90),
        suffix_v1_1_hidden_weight_floor=config.get("suffix_v1_1_hidden_weight_floor", 0.20),
        suffix_v1_1_prox_weight=config.get("suffix_v1_1_prox_weight", 0.005),
        suffix_v1_1_manifold_weight=config.get("suffix_v1_1_manifold_weight", 0.02),
        suffix_v1_1_manifold_update_every=config.get("suffix_v1_1_manifold_update_every", 10),
        suffix_v1_1_manifold_warmup_epoch=config.get("suffix_v1_1_manifold_warmup_epoch", 10),
        suffix_v1_1_range_weight=config.get("suffix_v1_1_range_weight", 0.001),
        suffix_reoptimization_v1_2_1=config.get("suffix_reoptimization_v1_2_1", False),
        suffix_reoptimization_v1_2_1_log=config.get("suffix_reoptimization_v1_2_1_log", True),
        suffix_v1_2_1_max_rounds=config.get("suffix_v1_2_1_max_rounds", 2),
        suffix_v1_2_1_epoch=config.get("suffix_v1_2_1_epoch", 50),
        suffix_v1_2_1_lr=config.get("suffix_v1_2_1_lr", 0.03),
        suffix_v1_2_1_hidden_low_threshold=config.get("suffix_v1_2_1_hidden_low_threshold", 0.50),
        suffix_v1_2_1_hidden_drop_threshold=config.get("suffix_v1_2_1_hidden_drop_threshold", 0.15),
        suffix_v1_2_1_token_forward_low_threshold=config.get("suffix_v1_2_1_token_forward_low_threshold", 0.50),
        suffix_v1_2_1_min_anomaly_reasons=config.get("suffix_v1_2_1_min_anomaly_reasons", 2),
        suffix_v1_2_1_min_hidden_delta=config.get("suffix_v1_2_1_min_hidden_delta", 0.005),
        suffix_v1_2_1_accuracy_tolerance=config.get("suffix_v1_2_1_accuracy_tolerance", 0.0),
        suffix_v1_2_1_accept_mode=config.get("suffix_v1_2_1_accept_mode", "oracle_accuracy"),
        suffix_v1_2_1_anomaly_detection_mode=config.get("suffix_v1_2_1_anomaly_detection_mode", "adaptive"),
        suffix_v1_2_1_adaptive_z_threshold=config.get("suffix_v1_2_1_adaptive_z_threshold", 1.5),
        suffix_v1_2_1_adaptive_drop_z_threshold=config.get("suffix_v1_2_1_adaptive_drop_z_threshold", 1.5),
        suffix_v1_2_1_adaptive_min_std=config.get("suffix_v1_2_1_adaptive_min_std", 1e-6),
        suffix_v1_2_1_adaptive_min_points=config.get("suffix_v1_2_1_adaptive_min_points", 4),
        suffix_v1_2_1_hidden_weight_mode=config.get("suffix_v1_2_1_hidden_weight_mode", "front_decay"),
        suffix_v1_2_1_hidden_weight_decay=config.get("suffix_v1_2_1_hidden_weight_decay", 0.90),
        suffix_v1_2_1_hidden_weight_floor=config.get("suffix_v1_2_1_hidden_weight_floor", 0.20),
        suffix_v1_2_1_prox_weight=config.get("suffix_v1_2_1_prox_weight", 0.005),
        suffix_v1_2_1_range_weight=config.get("suffix_v1_2_1_range_weight", 0.001),
        suffix_reoptimization_v1_2_2=config.get("suffix_reoptimization_v1_2_2", False),
        suffix_reoptimization_v1_2_2_log=config.get("suffix_reoptimization_v1_2_2_log", True),
        suffix_v1_2_2_max_rounds=config.get("suffix_v1_2_2_max_rounds", 2),
        suffix_v1_2_2_epoch=config.get("suffix_v1_2_2_epoch", 50),
        suffix_v1_2_2_lr=config.get("suffix_v1_2_2_lr", 0.03),
        suffix_v1_2_2_embedding_relative_mse_high_threshold=config.get(
            "suffix_v1_2_2_embedding_relative_mse_high_threshold", 1.0
        ),
        suffix_v1_2_2_relative_mse_rise_threshold=config.get(
            "suffix_v1_2_2_relative_mse_rise_threshold", 0.30
        ),
        suffix_v1_2_2_token_relative_mse_high_threshold=config.get(
            "suffix_v1_2_2_token_relative_mse_high_threshold", 1.0
        ),
        suffix_v1_2_2_min_anomaly_reasons=config.get(
            "suffix_v1_2_2_min_anomaly_reasons", 2
        ),
        suffix_v1_2_2_min_relative_mse_improvement=config.get(
            "suffix_v1_2_2_min_relative_mse_improvement", 0.01
        ),
        suffix_v1_2_2_accuracy_tolerance=config.get(
            "suffix_v1_2_2_accuracy_tolerance", 0.0
        ),
        suffix_v1_2_2_accept_mode=config.get(
            "suffix_v1_2_2_accept_mode", "oracle_accuracy"
        ),
        suffix_v1_2_2_anomaly_detection_mode=config.get(
            "suffix_v1_2_2_anomaly_detection_mode", "adaptive"
        ),
        suffix_v1_2_2_adaptive_z_threshold=config.get(
            "suffix_v1_2_2_adaptive_z_threshold", 1.5
        ),
        suffix_v1_2_2_adaptive_rise_z_threshold=config.get(
            "suffix_v1_2_2_adaptive_rise_z_threshold", 1.5
        ),
        suffix_v1_2_2_adaptive_min_std=config.get(
            "suffix_v1_2_2_adaptive_min_std", 1e-6
        ),
        suffix_v1_2_2_adaptive_min_points=config.get(
            "suffix_v1_2_2_adaptive_min_points", 4
        ),
        suffix_v1_2_2_hidden_weight_mode=config.get(
            "suffix_v1_2_2_hidden_weight_mode", "front_decay"
        ),
        suffix_v1_2_2_hidden_weight_decay=config.get(
            "suffix_v1_2_2_hidden_weight_decay", 0.90
        ),
        suffix_v1_2_2_hidden_weight_floor=config.get(
            "suffix_v1_2_2_hidden_weight_floor", 0.20
        ),
        suffix_v1_2_2_cosine_loss_weight=config.get(
            "suffix_v1_2_2_cosine_loss_weight", 0.1
        ),
        suffix_v1_2_2_relative_mse_loss_weight=config.get(
            "suffix_v1_2_2_relative_mse_loss_weight", 0.9
        ),
        suffix_v1_2_2_prox_weight=config.get(
            "suffix_v1_2_2_prox_weight", 0.005
        ),
        suffix_v1_2_2_range_weight=config.get(
            "suffix_v1_2_2_range_weight", 0.001
        ),
        suffix_v1_2_2_gradient_trend_stats_enabled=config.get(
            "suffix_v1_2_2_gradient_trend_stats_enabled", True
        ),
        suffix_v1_2_3=config.get(
            "suffix_v1_2_3", False
        ),
        suffix_v1_2_3_log=config.get(
            "suffix_v1_2_3_log", True
        ),
        suffix_v1_2_3_max_rounds=config.get(
            "suffix_v1_2_3_max_rounds", 2
        ),
        suffix_v1_2_3_epoch=config.get("suffix_v1_2_3_epoch", 50),
        suffix_v1_2_3_lr=config.get("suffix_v1_2_3_lr", 0.03),
        suffix_v1_2_3_embedding_relative_mse_high_threshold=config.get(
            "suffix_v1_2_3_embedding_relative_mse_high_threshold", 1.0
        ),
        suffix_v1_2_3_relative_mse_rise_threshold=config.get(
            "suffix_v1_2_3_relative_mse_rise_threshold", 0.30
        ),
        suffix_v1_2_3_token_relative_mse_high_threshold=config.get(
            "suffix_v1_2_3_token_relative_mse_high_threshold", 1.0
        ),
        suffix_v1_2_3_min_anomaly_reasons=config.get(
            "suffix_v1_2_3_min_anomaly_reasons", 1
        ),
        suffix_v1_2_3_min_relative_mse_improvement=config.get(
            "suffix_v1_2_3_min_relative_mse_improvement", 0.01
        ),
        suffix_v1_2_3_accuracy_tolerance=config.get(
            "suffix_v1_2_3_accuracy_tolerance", 0.0
        ),
        suffix_v1_2_3_accept_mode=config.get(
            "suffix_v1_2_3_accept_mode", "oracle_accuracy"
        ),
        suffix_v1_2_3_anomaly_detection_mode=config.get(
            "suffix_v1_2_3_anomaly_detection_mode", "adaptive"
        ),
        suffix_v1_2_3_adaptive_z_threshold=config.get(
            "suffix_v1_2_3_adaptive_z_threshold", 1.5
        ),
        suffix_v1_2_3_adaptive_rise_z_threshold=config.get(
            "suffix_v1_2_3_adaptive_rise_z_threshold", 1.5
        ),
        suffix_v1_2_3_adaptive_min_std=config.get(
            "suffix_v1_2_3_adaptive_min_std", 1e-6
        ),
        suffix_v1_2_3_adaptive_min_points=config.get(
            "suffix_v1_2_3_adaptive_min_points", 4
        ),
        suffix_v1_2_3_hidden_weight_mode=config.get(
            "suffix_v1_2_3_hidden_weight_mode", "front_decay"
        ),
        suffix_v1_2_3_hidden_weight_decay=config.get(
            "suffix_v1_2_3_hidden_weight_decay", 0.90
        ),
        suffix_v1_2_3_hidden_weight_floor=config.get(
            "suffix_v1_2_3_hidden_weight_floor", 0.20
        ),
        suffix_v1_2_3_cosine_loss_weight=config.get(
            "suffix_v1_2_3_cosine_loss_weight", 0.1
        ),
        suffix_v1_2_3_relative_mse_loss_weight=config.get(
            "suffix_v1_2_3_relative_mse_loss_weight", 0.9
        ),
        suffix_v1_2_3_prox_weight=config.get(
            "suffix_v1_2_3_prox_weight", 0.005
        ),
        suffix_v1_2_3_range_weight=config.get(
            "suffix_v1_2_3_range_weight", 0.001
        ),
        suffix_v1_2_3_gradient_trend_stats_enabled=config.get(
            "suffix_v1_2_3_gradient_trend_stats_enabled", True
        ),
        suffix_reoptimization_v1_2=config.get("suffix_reoptimization_v1_2", False),
        suffix_reoptimization_v1_2_log=config.get("suffix_reoptimization_v1_2_log", True),
        suffix_v1_2_max_rounds=config.get("suffix_v1_2_max_rounds", 2),
        suffix_v1_2_epoch=config.get("suffix_v1_2_epoch", 50),
        suffix_v1_2_lr=config.get("suffix_v1_2_lr", 0.03),
        suffix_v1_2_hidden_low_threshold=config.get("suffix_v1_2_hidden_low_threshold", 0.50),
        suffix_v1_2_hidden_drop_threshold=config.get("suffix_v1_2_hidden_drop_threshold", 0.15),
        suffix_v1_2_token_forward_low_threshold=config.get("suffix_v1_2_token_forward_low_threshold", 0.50),
        suffix_v1_2_min_anomaly_reasons=config.get("suffix_v1_2_min_anomaly_reasons", 2),
        suffix_v1_2_min_hidden_delta=config.get("suffix_v1_2_min_hidden_delta", 0.005),
        suffix_v1_2_accuracy_tolerance=config.get("suffix_v1_2_accuracy_tolerance", 0.0),
        suffix_v1_2_accept_mode=config.get("suffix_v1_2_accept_mode", "oracle_accuracy"),
        suffix_v1_2_anomaly_detection_mode=config.get("suffix_v1_2_anomaly_detection_mode", "adaptive"),
        suffix_v1_2_adaptive_z_threshold=config.get("suffix_v1_2_adaptive_z_threshold", 1.5),
        suffix_v1_2_adaptive_drop_z_threshold=config.get("suffix_v1_2_adaptive_drop_z_threshold", 1.5),
        suffix_v1_2_adaptive_min_std=config.get("suffix_v1_2_adaptive_min_std", 1e-6),
        suffix_v1_2_adaptive_min_points=config.get("suffix_v1_2_adaptive_min_points", 4),
        suffix_v1_2_hidden_weight_mode=config.get("suffix_v1_2_hidden_weight_mode", "front_decay"),
        suffix_v1_2_hidden_weight_decay=config.get("suffix_v1_2_hidden_weight_decay", 0.90),
        suffix_v1_2_hidden_weight_floor=config.get("suffix_v1_2_hidden_weight_floor", 0.20),
        suffix_v1_2_prox_weight=config.get("suffix_v1_2_prox_weight", 0.005),
        suffix_v1_2_manifold_weight=config.get("suffix_v1_2_manifold_weight", 0.02),
        suffix_v1_2_manifold_update_every=config.get("suffix_v1_2_manifold_update_every", 10),
        suffix_v1_2_manifold_warmup_epoch=config.get("suffix_v1_2_manifold_warmup_epoch", 10),
        suffix_v1_2_range_weight=config.get("suffix_v1_2_range_weight", 0.001),
        suffix_reoptimization_v1_3=config.get("suffix_reoptimization_v1_3", False),
        suffix_reoptimization_v1_3_log=config.get("suffix_reoptimization_v1_3_log", True),
        suffix_v1_3_max_rounds=config.get("suffix_v1_3_max_rounds", 2),
        suffix_v1_3_epoch=config.get("suffix_v1_3_epoch", 50),
        suffix_v1_3_lr=config.get("suffix_v1_3_lr", 0.03),
        suffix_v1_3_hidden_low_threshold=config.get("suffix_v1_3_hidden_low_threshold", 0.50),
        suffix_v1_3_hidden_drop_threshold=config.get("suffix_v1_3_hidden_drop_threshold", 0.15),
        suffix_v1_3_token_forward_low_threshold=config.get("suffix_v1_3_token_forward_low_threshold", 0.50),
        suffix_v1_3_min_anomaly_reasons=config.get("suffix_v1_3_min_anomaly_reasons", 2),
        suffix_v1_3_min_hidden_delta=config.get("suffix_v1_3_min_hidden_delta", 0.005),
        suffix_v1_3_accuracy_tolerance=config.get("suffix_v1_3_accuracy_tolerance", 0.0),
        suffix_v1_3_accept_mode=config.get("suffix_v1_3_accept_mode", "oracle_accuracy"),
        suffix_v1_3_anomaly_detection_mode=config.get("suffix_v1_3_anomaly_detection_mode", "adaptive"),
        suffix_v1_3_adaptive_z_threshold=config.get("suffix_v1_3_adaptive_z_threshold", 1.5),
        suffix_v1_3_adaptive_drop_z_threshold=config.get("suffix_v1_3_adaptive_drop_z_threshold", 1.5),
        suffix_v1_3_adaptive_min_std=config.get("suffix_v1_3_adaptive_min_std", 1e-6),
        suffix_v1_3_adaptive_min_points=config.get("suffix_v1_3_adaptive_min_points", 4),
        suffix_v1_3_hidden_weight_mode=config.get("suffix_v1_3_hidden_weight_mode", "front_decay"),
        suffix_v1_3_hidden_weight_decay=config.get("suffix_v1_3_hidden_weight_decay", 0.90),
        suffix_v1_3_hidden_weight_floor=config.get("suffix_v1_3_hidden_weight_floor", 0.20),
        suffix_v1_3_prox_weight=config.get("suffix_v1_3_prox_weight", 0.005),
        suffix_v1_3_manifold_weight=config.get("suffix_v1_3_manifold_weight", 0.02),
        suffix_v1_3_manifold_update_every=config.get("suffix_v1_3_manifold_update_every", 10),
        suffix_v1_3_manifold_warmup_epoch=config.get("suffix_v1_3_manifold_warmup_epoch", 10),
        suffix_v1_3_range_weight=config.get("suffix_v1_3_range_weight", 0.001),
        suffix_v1_3_anchor_mode=config.get("suffix_v1_3_anchor_mode", "anchor_stable_prefix"),
        suffix_reoptimization_v1_3_1=config.get("suffix_reoptimization_v1_3_1", False),
        suffix_reoptimization_v1_3_1_log=config.get("suffix_reoptimization_v1_3_1_log", True),
        suffix_v1_3_1_max_rounds=config.get("suffix_v1_3_1_max_rounds", 2),
        suffix_v1_3_1_epoch=config.get("suffix_v1_3_1_epoch", 50),
        suffix_v1_3_1_lr=config.get("suffix_v1_3_1_lr", 0.03),
        suffix_v1_3_1_hidden_low_threshold=config.get(
            "suffix_v1_3_1_hidden_low_threshold", 0.50
        ),
        suffix_v1_3_1_hidden_drop_threshold=config.get(
            "suffix_v1_3_1_hidden_drop_threshold", 0.15
        ),
        suffix_v1_3_1_token_forward_low_threshold=config.get(
            "suffix_v1_3_1_token_forward_low_threshold", 0.50
        ),
        suffix_v1_3_1_min_anomaly_reasons=config.get(
            "suffix_v1_3_1_min_anomaly_reasons", 2
        ),
        suffix_v1_3_1_min_hidden_delta=config.get(
            "suffix_v1_3_1_min_hidden_delta", 0.005
        ),
        suffix_v1_3_1_accuracy_tolerance=config.get(
            "suffix_v1_3_1_accuracy_tolerance", 0.0
        ),
        suffix_v1_3_1_accept_mode=config.get(
            "suffix_v1_3_1_accept_mode", "oracle_accuracy"
        ),
        suffix_v1_3_1_anomaly_detection_mode=config.get(
            "suffix_v1_3_1_anomaly_detection_mode", "adaptive"
        ),
        suffix_v1_3_1_adaptive_z_threshold=config.get(
            "suffix_v1_3_1_adaptive_z_threshold", 1.5
        ),
        suffix_v1_3_1_adaptive_drop_z_threshold=config.get(
            "suffix_v1_3_1_adaptive_drop_z_threshold", 1.5
        ),
        suffix_v1_3_1_adaptive_min_std=config.get(
            "suffix_v1_3_1_adaptive_min_std", 1e-6
        ),
        suffix_v1_3_1_adaptive_min_points=config.get(
            "suffix_v1_3_1_adaptive_min_points", 4
        ),
        suffix_v1_3_1_hidden_weight_mode=config.get(
            "suffix_v1_3_1_hidden_weight_mode", "front_decay"
        ),
        suffix_v1_3_1_hidden_weight_decay=config.get(
            "suffix_v1_3_1_hidden_weight_decay", 0.90
        ),
        suffix_v1_3_1_hidden_weight_floor=config.get(
            "suffix_v1_3_1_hidden_weight_floor", 0.20
        ),
        suffix_v1_3_1_prox_weight=config.get(
            "suffix_v1_3_1_prox_weight", 0.005
        ),
        suffix_v1_3_1_range_weight=config.get(
            "suffix_v1_3_1_range_weight", 0.001
        ),
        suffix_reoptimization_v1_4=config.get("suffix_reoptimization_v1_4", False),
        suffix_reoptimization_v1_4_log=config.get("suffix_reoptimization_v1_4_log", True),
        suffix_v1_4_coarse_lr_max=config.get("suffix_v1_4_coarse_lr_max", 0.10),
        suffix_v1_4_coarse_lr_min=config.get("suffix_v1_4_coarse_lr_min", 0.03),
        suffix_v1_4_coarse_schedule=config.get("suffix_v1_4_coarse_schedule", "cosine"),
        suffix_v1_4_fine_epoch=config.get("suffix_v1_4_fine_epoch", 50),
        suffix_v1_4_fine_lr_max=config.get("suffix_v1_4_fine_lr_max", 0.01),
        suffix_v1_4_fine_lr_min=config.get("suffix_v1_4_fine_lr_min", 0.001),
        suffix_v1_4_fine_schedule=config.get("suffix_v1_4_fine_schedule", "cosine"),
        suffix_v1_4_confidence_mode=config.get("suffix_v1_4_confidence_mode", "hybrid"),
        suffix_v1_4_confidence_continuous_min=config.get(
            "suffix_v1_4_confidence_continuous_min", 0.80
        ),
        suffix_v1_4_confidence_token_min=config.get(
            "suffix_v1_4_confidence_token_min", 0.80
        ),
        suffix_v1_4_confidence_margin_min=config.get(
            "suffix_v1_4_confidence_margin_min", 0.02
        ),
        suffix_v1_4_confidence_gap_max=config.get(
            "suffix_v1_4_confidence_gap_max", 0.10
        ),
        suffix_v1_4_confidence_percentile_min=config.get(
            "suffix_v1_4_confidence_percentile_min", 0.60
        ),
        suffix_v1_4_confidence_min_points=config.get(
            "suffix_v1_4_confidence_min_points", 4
        ),
        suffix_v1_4_require_candidate_agreement=config.get(
            "suffix_v1_4_require_candidate_agreement", True
        ),
        suffix_v1_4_adaptive_z_threshold=config.get(
            "suffix_v1_4_adaptive_z_threshold", 1.5
        ),
        suffix_v1_4_adaptive_drop_z_threshold=config.get(
            "suffix_v1_4_adaptive_drop_z_threshold", 1.5
        ),
        suffix_v1_4_adaptive_min_std=config.get(
            "suffix_v1_4_adaptive_min_std", 1e-6
        ),
        suffix_v1_4_adaptive_min_points=config.get(
            "suffix_v1_4_adaptive_min_points", 4
        ),
        suffix_v1_4_fine_window=config.get("suffix_v1_4_fine_window", 2),
        suffix_v1_4_fine_window_decay=config.get(
            "suffix_v1_4_fine_window_decay", 0.50
        ),
        suffix_v1_4_prox_weight=config.get("suffix_v1_4_prox_weight", 0.005),
        suffix_v1_4_range_weight=config.get("suffix_v1_4_range_weight", 0.001),
        suffix_v1_4_min_hidden_delta=config.get("suffix_v1_4_min_hidden_delta", 0.005),
        suffix_v1_4_accuracy_tolerance=config.get("suffix_v1_4_accuracy_tolerance", 0.0),
        suffix_v1_4_accept_mode=config.get("suffix_v1_4_accept_mode", "oracle_accuracy"),
        suffix_reoptimization_v1_4_1=config.get("suffix_reoptimization_v1_4_1", False),
        suffix_reoptimization_v1_4_1_log=config.get("suffix_reoptimization_v1_4_1_log", True),
        suffix_v1_4_1_coarse_lr_max=config.get("suffix_v1_4_1_coarse_lr_max", 0.10),
        suffix_v1_4_1_coarse_lr_min=config.get("suffix_v1_4_1_coarse_lr_min", 0.03),
        suffix_v1_4_1_coarse_schedule=config.get("suffix_v1_4_1_coarse_schedule", "cosine"),
        suffix_v1_4_1_fine_epoch=config.get("suffix_v1_4_1_fine_epoch", 50),
        suffix_v1_4_1_fine_lr_max=config.get("suffix_v1_4_1_fine_lr_max", 0.01),
        suffix_v1_4_1_fine_lr_min=config.get("suffix_v1_4_1_fine_lr_min", 0.001),
        suffix_v1_4_1_fine_schedule=config.get("suffix_v1_4_1_fine_schedule", "cosine"),
        suffix_v1_4_1_confidence_mode=config.get(
            "suffix_v1_4_1_confidence_mode", "absolute"
        ),
        suffix_v1_4_1_confidence_continuous_min=config.get(
            "suffix_v1_4_1_confidence_continuous_min", 0.80
        ),
        suffix_v1_4_1_confidence_token_min=config.get(
            "suffix_v1_4_1_confidence_token_min", 0.80
        ),
        suffix_v1_4_1_confidence_margin_min=config.get(
            "suffix_v1_4_1_confidence_margin_min", 0.02
        ),
        suffix_v1_4_1_confidence_gap_max=config.get(
            "suffix_v1_4_1_confidence_gap_max", 0.10
        ),
        suffix_v1_4_1_require_candidate_agreement=config.get(
            "suffix_v1_4_1_require_candidate_agreement", True
        ),
        suffix_v1_4_1_adaptive_z_threshold=config.get(
            "suffix_v1_4_1_adaptive_z_threshold", 1.5
        ),
        suffix_v1_4_1_adaptive_drop_z_threshold=config.get(
            "suffix_v1_4_1_adaptive_drop_z_threshold", 1.5
        ),
        suffix_v1_4_1_adaptive_min_std=config.get(
            "suffix_v1_4_1_adaptive_min_std", 1e-6
        ),
        suffix_v1_4_1_adaptive_min_points=config.get(
            "suffix_v1_4_1_adaptive_min_points", 4
        ),
        suffix_v1_4_1_fine_window=config.get("suffix_v1_4_1_fine_window", 2),
        suffix_v1_4_1_fine_window_decay=config.get(
            "suffix_v1_4_1_fine_window_decay", 0.50
        ),
        suffix_v1_4_1_prox_weight=config.get("suffix_v1_4_1_prox_weight", 0.005),
        suffix_v1_4_1_range_weight=config.get("suffix_v1_4_1_range_weight", 0.001),
        suffix_v1_4_1_min_hidden_delta=config.get(
            "suffix_v1_4_1_min_hidden_delta", 0.005
        ),
        suffix_v1_4_1_accuracy_tolerance=config.get(
            "suffix_v1_4_1_accuracy_tolerance", 0.0
        ),
        suffix_v1_4_1_accept_mode=config.get(
            "suffix_v1_4_1_accept_mode", "oracle_accuracy"
        ),
    )
    parser.set_defaults(
        suffix_reoptimization_v2_0=config.get(
            "suffix_reoptimization_v2_0", False
        ),
        suffix_reoptimization_v2_0_log=config.get(
            "suffix_reoptimization_v2_0_log", True
        ),
        suffix_v2_0_layer_offsets=config.get(
            "suffix_v2_0_layer_offsets", [0, 1, 2]
        ),
        suffix_v2_0_layer_weights=config.get(
            "suffix_v2_0_layer_weights", [1.0, 0.5, 0.25]
        ),
        suffix_v2_0_epsilon=config.get("suffix_v2_0_epsilon", 1e-8),
        suffix_v2_0_phase1_epoch=config.get("suffix_v2_0_phase1_epoch", 1000),
        suffix_v2_0_phase1_lr=config.get("suffix_v2_0_phase1_lr", 0.01),
        suffix_v2_0_phase1_direction_weight=config.get(
            "suffix_v2_0_phase1_direction_weight", 0.9
        ),
        suffix_v2_0_phase1_magnitude_weight=config.get(
            "suffix_v2_0_phase1_magnitude_weight", 0.1
        ),
        suffix_v2_0_phase2_epoch=config.get("suffix_v2_0_phase2_epoch", 50),
        suffix_v2_0_phase2_lr=config.get("suffix_v2_0_phase2_lr", 0.001),
        suffix_v2_0_phase2_direction_weight=config.get(
            "suffix_v2_0_phase2_direction_weight", 0.1
        ),
        suffix_v2_0_phase2_magnitude_weight=config.get(
            "suffix_v2_0_phase2_magnitude_weight", 0.9
        ),
        suffix_v2_0_score_direction_weight=config.get(
            "suffix_v2_0_score_direction_weight", 0.5
        ),
        suffix_v2_0_score_magnitude_weight=config.get(
            "suffix_v2_0_score_magnitude_weight", 0.5
        ),
        suffix_v2_0_prox_weight=config.get("suffix_v2_0_prox_weight", 0.005),
        suffix_v2_0_range_weight=config.get("suffix_v2_0_range_weight", 0.001),
        suffix_v2_0_continuous_mad_multiplier=config.get(
            "suffix_v2_0_continuous_mad_multiplier", 3.0
        ),
        suffix_v2_0_local_discrete_mad_multiplier=config.get(
            "suffix_v2_0_local_discrete_mad_multiplier", 3.0
        ),
        suffix_v2_0_local_gap_jump_mad_multiplier=config.get(
            "suffix_v2_0_local_gap_jump_mad_multiplier", 3.0
        ),
        suffix_v2_0_mad_epsilon=config.get("suffix_v2_0_mad_epsilon", 1e-8),
        suffix_v2_0_local_min_points=config.get("suffix_v2_0_local_min_points", 4),
        suffix_v2_0_normal_embedding_top_k=config.get(
            "suffix_v2_0_normal_embedding_top_k", 10
        ),
        suffix_v2_0_expanded_embedding_top_k=config.get(
            "suffix_v2_0_expanded_embedding_top_k", 20
        ),
        suffix_v2_0_ppl_top_k=config.get("suffix_v2_0_ppl_top_k", 10),
        suffix_v2_0_classifier_top_k=config.get(
            "suffix_v2_0_classifier_top_k", 10
        ),
        suffix_v2_0_cumulative_min_points=config.get(
            "suffix_v2_0_cumulative_min_points", 4
        ),
        suffix_v2_0_cumulative_kappa=config.get(
            "suffix_v2_0_cumulative_kappa", 0.5
        ),
        suffix_v2_0_cumulative_threshold=config.get(
            "suffix_v2_0_cumulative_threshold", 5.0
        ),
        suffix_v2_0_replace_epsilon=config.get(
            "suffix_v2_0_replace_epsilon", 1e-8
        ),
        suffix_v2_0_cumulative_max_repairs_per_trigger=config.get(
            "suffix_v2_0_cumulative_max_repairs_per_trigger", 1
        ),
        suffix_v2_0_accuracy_diagnostics_enabled=config.get(
            "suffix_v2_0_accuracy_diagnostics_enabled", True
        ),
        suffix_v2_0_classifier_enabled=config.get(
            "suffix_v2_0_classifier_enabled", False
        ),
    )
    v2_cli_fields = (
        ("suffix-reoptimization-v2-0", "suffix_reoptimization_v2_0", str2bool),
        ("suffix-reoptimization-v2-0-log", "suffix_reoptimization_v2_0_log", str2bool),
        ("suffix-v2-0-layer-offsets", "suffix_v2_0_layer_offsets", json_arg),
        ("suffix-v2-0-layer-weights", "suffix_v2_0_layer_weights", json_arg),
        ("suffix-v2-0-epsilon", "suffix_v2_0_epsilon", float),
        ("suffix-v2-0-phase1-epoch", "suffix_v2_0_phase1_epoch", int),
        ("suffix-v2-0-phase1-lr", "suffix_v2_0_phase1_lr", float),
        ("suffix-v2-0-phase1-direction-weight", "suffix_v2_0_phase1_direction_weight", float),
        ("suffix-v2-0-phase1-magnitude-weight", "suffix_v2_0_phase1_magnitude_weight", float),
        ("suffix-v2-0-phase2-epoch", "suffix_v2_0_phase2_epoch", int),
        ("suffix-v2-0-phase2-lr", "suffix_v2_0_phase2_lr", float),
        ("suffix-v2-0-phase2-direction-weight", "suffix_v2_0_phase2_direction_weight", float),
        ("suffix-v2-0-phase2-magnitude-weight", "suffix_v2_0_phase2_magnitude_weight", float),
        ("suffix-v2-0-score-direction-weight", "suffix_v2_0_score_direction_weight", float),
        ("suffix-v2-0-score-magnitude-weight", "suffix_v2_0_score_magnitude_weight", float),
        ("suffix-v2-0-prox-weight", "suffix_v2_0_prox_weight", float),
        ("suffix-v2-0-range-weight", "suffix_v2_0_range_weight", float),
        ("suffix-v2-0-continuous-mad-multiplier", "suffix_v2_0_continuous_mad_multiplier", float),
        ("suffix-v2-0-local-discrete-mad-multiplier", "suffix_v2_0_local_discrete_mad_multiplier", float),
        ("suffix-v2-0-local-gap-jump-mad-multiplier", "suffix_v2_0_local_gap_jump_mad_multiplier", float),
        ("suffix-v2-0-mad-epsilon", "suffix_v2_0_mad_epsilon", float),
        ("suffix-v2-0-local-min-points", "suffix_v2_0_local_min_points", int),
        ("suffix-v2-0-normal-embedding-top-k", "suffix_v2_0_normal_embedding_top_k", int),
        ("suffix-v2-0-expanded-embedding-top-k", "suffix_v2_0_expanded_embedding_top_k", int),
        ("suffix-v2-0-ppl-top-k", "suffix_v2_0_ppl_top_k", int),
        ("suffix-v2-0-classifier-top-k", "suffix_v2_0_classifier_top_k", int),
        ("suffix-v2-0-cumulative-min-points", "suffix_v2_0_cumulative_min_points", int),
        ("suffix-v2-0-cumulative-kappa", "suffix_v2_0_cumulative_kappa", float),
        ("suffix-v2-0-cumulative-threshold", "suffix_v2_0_cumulative_threshold", float),
        ("suffix-v2-0-replace-epsilon", "suffix_v2_0_replace_epsilon", float),
        ("suffix-v2-0-cumulative-max-repairs-per-trigger", "suffix_v2_0_cumulative_max_repairs_per_trigger", int),
        ("suffix-v2-0-accuracy-diagnostics-enabled", "suffix_v2_0_accuracy_diagnostics_enabled", str2bool),
        ("suffix-v2-0-classifier-enabled", "suffix_v2_0_classifier_enabled", str2bool),
    )
    for option, destination, value_type in v2_cli_fields:
        parser.add_argument(
            "--" + option,
            dest=destination,
            type=value_type,
            default=parser.get_default(destination),
        )
    parser.set_defaults(
        suffix_reoptimization_v2_1=config.get(
            "suffix_reoptimization_v2_1", True
        ),
        suffix_reoptimization_v2_1_log=config.get(
            "suffix_reoptimization_v2_1_log", True
        ),
        suffix_v2_1_layer_offsets=config.get(
            "suffix_v2_1_layer_offsets", [0, 1, 2]
        ),
        suffix_v2_1_layer_weights=config.get(
            "suffix_v2_1_layer_weights", [1.0, 0.5, 0.25]
        ),
        suffix_v2_1_alpha_dir=config.get("suffix_v2_1_alpha_dir", 0.5),
        suffix_v2_1_alpha_mag=config.get("suffix_v2_1_alpha_mag", 0.5),
        suffix_v2_1_vocab_weight=config.get("suffix_v2_1_vocab_weight", 0.005),
        suffix_v2_1_vocab_temperature=config.get(
            "suffix_v2_1_vocab_temperature", 0.01
        ),
        suffix_v2_1_vocab_anchor_top_k=config.get(
            "suffix_v2_1_vocab_anchor_top_k", 10
        ),
        suffix_v2_1_vocab_anchor_refresh_interval=config.get(
            "suffix_v2_1_vocab_anchor_refresh_interval", 10
        ),
        suffix_v2_1_global_optimizer=config.get(
            "suffix_v2_1_global_optimizer", "adam"
        ),
        suffix_v2_1_global_steps=config.get("suffix_v2_1_global_steps", 1000),
        suffix_v2_1_global_lr=config.get("suffix_v2_1_global_lr", 0.001),
        suffix_v2_1_local_optimizer=config.get(
            "suffix_v2_1_local_optimizer", "adam"
        ),
        suffix_v2_1_local_steps=config.get("suffix_v2_1_local_steps", 50),
        suffix_v2_1_local_lr=config.get("suffix_v2_1_local_lr", 0.001),
        suffix_v2_1_adam_beta1=config.get("suffix_v2_1_adam_beta1", 0.9),
        suffix_v2_1_adam_beta2=config.get("suffix_v2_1_adam_beta2", 0.999),
        suffix_v2_1_adam_epsilon=config.get("suffix_v2_1_adam_epsilon", 1e-8),
        suffix_v2_1_weight_decay_enabled=config.get(
            "suffix_v2_1_weight_decay_enabled", False
        ),
        suffix_v2_1_scheduler_mode=config.get(
            "suffix_v2_1_scheduler_mode", "none"
        ),
        suffix_v2_1_tau_J=config.get("suffix_v2_1_tau_J", 0.15),
        suffix_v2_1_delta_c_max=config.get("suffix_v2_1_delta_c_max", 0.01),
        suffix_v2_1_tau_r=config.get("suffix_v2_1_tau_r", 0.05),
        suffix_v2_1_embedding_top_k_normal=config.get(
            "suffix_v2_1_embedding_top_k_normal", 10
        ),
        suffix_v2_1_embedding_top_k_expanded=config.get(
            "suffix_v2_1_embedding_top_k_expanded", 20
        ),
        suffix_v2_1_ppl_top_k=config.get("suffix_v2_1_ppl_top_k", 10),
        suffix_v2_1_vocab_distance_mode=config.get(
            "suffix_v2_1_vocab_distance_mode", "mean_squared_l2"
        ),
        suffix_v2_1_vocab_softmin_mode=config.get(
            "suffix_v2_1_vocab_softmin_mode", "normalized_stable_logsumexp"
        ),
        suffix_v2_1_candidate_tie_break_mode=config.get(
            "suffix_v2_1_candidate_tie_break_mode", "hidden_error_token_id"
        ),
        suffix_v2_1_hidden_epsilon=config.get(
            "suffix_v2_1_hidden_epsilon", 1e-8
        ),
        suffix_v2_1_epsilon_J=config.get("suffix_v2_1_epsilon_J", 1e-8),
        suffix_v2_1_epsilon_d=config.get("suffix_v2_1_epsilon_d", 1e-8),
        suffix_v2_1_accuracy_diagnostics_enabled=config.get(
            "suffix_v2_1_accuracy_diagnostics_enabled", False
        ),
        suffix_v2_1_filter_nonascii=config.get(
            "suffix_v2_1_filter_nonascii", True
        ),
    )
    v21_cli_fields = (
        ("suffix-reoptimization-v2-1", "suffix_reoptimization_v2_1", str2bool),
        ("suffix-reoptimization-v2-1-log", "suffix_reoptimization_v2_1_log", str2bool),
        ("suffix-v2-1-layer-offsets", "suffix_v2_1_layer_offsets", json_arg),
        ("suffix-v2-1-layer-weights", "suffix_v2_1_layer_weights", json_arg),
        ("suffix-v2-1-alpha-dir", "suffix_v2_1_alpha_dir", float),
        ("suffix-v2-1-alpha-mag", "suffix_v2_1_alpha_mag", float),
        ("suffix-v2-1-vocab-weight", "suffix_v2_1_vocab_weight", float),
        ("suffix-v2-1-vocab-temperature", "suffix_v2_1_vocab_temperature", float),
        ("suffix-v2-1-vocab-anchor-top-k", "suffix_v2_1_vocab_anchor_top_k", int),
        ("suffix-v2-1-vocab-anchor-refresh-interval", "suffix_v2_1_vocab_anchor_refresh_interval", int),
        ("suffix-v2-1-global-optimizer", "suffix_v2_1_global_optimizer", str),
        ("suffix-v2-1-global-steps", "suffix_v2_1_global_steps", int),
        ("suffix-v2-1-global-lr", "suffix_v2_1_global_lr", float),
        ("suffix-v2-1-local-optimizer", "suffix_v2_1_local_optimizer", str),
        ("suffix-v2-1-local-steps", "suffix_v2_1_local_steps", int),
        ("suffix-v2-1-local-lr", "suffix_v2_1_local_lr", float),
        ("suffix-v2-1-adam-beta1", "suffix_v2_1_adam_beta1", float),
        ("suffix-v2-1-adam-beta2", "suffix_v2_1_adam_beta2", float),
        ("suffix-v2-1-adam-epsilon", "suffix_v2_1_adam_epsilon", float),
        ("suffix-v2-1-weight-decay-enabled", "suffix_v2_1_weight_decay_enabled", str2bool),
        ("suffix-v2-1-scheduler-mode", "suffix_v2_1_scheduler_mode", str),
        ("suffix-v2-1-tau-j", "suffix_v2_1_tau_J", float),
        ("suffix-v2-1-delta-c-max", "suffix_v2_1_delta_c_max", float),
        ("suffix-v2-1-tau-r", "suffix_v2_1_tau_r", float),
        ("suffix-v2-1-embedding-top-k-normal", "suffix_v2_1_embedding_top_k_normal", int),
        ("suffix-v2-1-embedding-top-k-expanded", "suffix_v2_1_embedding_top_k_expanded", int),
        ("suffix-v2-1-ppl-top-k", "suffix_v2_1_ppl_top_k", int),
        ("suffix-v2-1-vocab-distance-mode", "suffix_v2_1_vocab_distance_mode", str),
        ("suffix-v2-1-vocab-softmin-mode", "suffix_v2_1_vocab_softmin_mode", str),
        ("suffix-v2-1-candidate-tie-break-mode", "suffix_v2_1_candidate_tie_break_mode", str),
        ("suffix-v2-1-hidden-epsilon", "suffix_v2_1_hidden_epsilon", float),
        ("suffix-v2-1-epsilon-j", "suffix_v2_1_epsilon_J", float),
        ("suffix-v2-1-epsilon-d", "suffix_v2_1_epsilon_d", float),
        ("suffix-v2-1-accuracy-diagnostics-enabled", "suffix_v2_1_accuracy_diagnostics_enabled", str2bool),
        ("suffix-v2-1-filter-nonascii", "suffix_v2_1_filter_nonascii", str2bool),
    )
    for option, destination, value_type in v21_cli_fields:
        parser.add_argument(
            "--" + option,
            dest=destination,
            type=value_type,
            default=parser.get_default(destination),
        )
    parser.add_argument("--quantization", type=str, default=config.get("quantization", "none"),
                        choices=["4bit", "8bit", "none"])
    parser.add_argument("--device-map", type=str, default=config.get("device_map", "manual"))
    parser.add_argument("--offload-folder", type=str, default=config.get("offload_folder"))
    parser.add_argument("--offload-state-dict", type=str2bool, default=config.get("offload_state_dict", False))
    parser.add_argument("--max-memory", type=json_arg, default=config.get("max_memory"))
    parser.add_argument(
        "--log-dir",
        type=str,
        default=config.get("log_dir", os.path.join(SCRIPT_DIR, "results", "invert_timestamp_runs")),
    )
    parser.add_argument("--seed", type=int, default=config.get("seed", 0))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.get("output_dir", "experiment"),
    )
    parser.add_argument("--parallel-worker-spec", type=str, default=None)
    parser.add_argument("--worker-output-dir", type=str, default=None)
    
    args = parser.parse_args()
    canonical_suffix_option_present = any(
        item == "--suffix-version"
        or item.startswith("--suffix-version=")
        for item in sys.argv[1:]
    )
    args.suffix_version = resolve_cli_suffix_version(
        args.suffix_version,
        args.legacy_suffix_version,
        canonical_option_present=canonical_suffix_option_present,
    )
    del args.legacy_suffix_version
    try:
        exit_code = main(args)
    except Exception as error:
        runtime_spec = getattr(args, "_parallel_worker_runtime_spec", None)
        if runtime_spec is not None:
            completed = []
            shard_path = os.path.join(
                runtime_spec["output_dir"], "shard_reconstructions.jsonl"
            )
            if os.path.exists(shard_path):
                with open(shard_path, "r", encoding="utf-8") as shard:
                    for line in shard:
                        if line.strip():
                            completed.append(int(json.loads(line)["global_index"]))
            _write_parallel_worker_status(
                runtime_spec,
                completed,
                len(completed),
                1,
                False,
                "{}: {}".format(type(error).__name__, error),
            )
        raise
    raise SystemExit(exit_code or 0)
