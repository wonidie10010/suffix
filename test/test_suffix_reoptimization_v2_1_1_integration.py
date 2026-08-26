import copy
import inspect
import json
import os
import types
import unittest

import torch

import invert
from experiment_outputs import (
    _resolved_suffix_v211_config,
    _suffix_result,
    build_resolved_config,
)
from suffix_optimization_methods.method_versions import (
    suffix_reoptimization_v2_0 as v20,
)
from suffix_optimization_methods.method_versions import (
    suffix_reoptimization_v2_1_1 as v21,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVANCED_CONFIG = os.path.join(
    ROOT,
    "suffix_optimization_methods",
    "configs",
    "advanced_methods.json",
)
CANDIDATE_CONFIG = os.path.join(
    ROOT,
    "CGMR",
    "configs",
    "candidate_reranking_methods.json",
)


def _disabled():
    return types.SimpleNamespace(enabled=False)


class SelectorAndCompatibilityTests(unittest.TestCase):
    def test_v21_aliases_and_strict_disabled_failure(self):
        for alias in (
            "2.1.1",
            "v2.1.1",
            "suffix_v2_1_1",
            "suffix_reoptimization_v2.1.1",
            "suffix_reoptimization_v2_1_1",
        ):
            self.assertEqual("v2.1.1", invert.normalize_suffix_version(alias))

        selected = invert.select_advanced_method(
            "v2.1.1",
            _disabled(),
            _disabled(),
            _disabled(),
            _disabled(),
            suffix_reopt_v2_1_1_config=(
                v21.SuffixReoptimizationV211Config(enabled=True)
            ),
        )
        self.assertEqual(v21.METHOD_NAME, selected)
        with self.assertRaisesRegex(ValueError, "v2.1.1.*disabled"):
            invert.select_advanced_method(
                "v2.1.1",
                _disabled(),
                _disabled(),
                _disabled(),
                _disabled(),
                suffix_reopt_v2_1_1_config=(
                    v21.SuffixReoptimizationV211Config(enabled=False)
                ),
            )

    def test_v20_explicit_selector_semantics_are_unchanged(self):
        selected = invert.select_advanced_method(
            "v2.0",
            _disabled(),
            _disabled(),
            _disabled(),
            _disabled(),
            suffix_reopt_v2_0_config=(
                v20.SuffixReoptimizationV20Config(enabled=True)
            ),
            suffix_reopt_v2_1_1_config=(
                v21.SuffixReoptimizationV211Config(enabled=True)
            ),
        )
        self.assertEqual(v20.METHOD_NAME, selected)
        self.assertTrue(
            invert.validate_advanced_candidate_combination(
                v20.METHOD_NAME,
                "CGMR_v1.2",
            )
        )

    def test_v21_rejects_every_cgmr_version_and_accepts_none(self):
        self.assertTrue(
            invert.validate_advanced_candidate_combination(
                v21.METHOD_NAME,
                "none",
            )
        )
        for candidate_method in ("CGMR_v1.0", "CGMR_v1.1", "CGMR_v1.2"):
            with self.subTest(candidate_method=candidate_method):
                with self.assertRaisesRegex(ValueError, "v2.1.1.*CGMR"):
                    invert.validate_advanced_candidate_combination(
                        v21.METHOD_NAME,
                        candidate_method,
                    )

    def test_v21_model_contract_is_qwen2_causal_only(self):
        accepted = types.SimpleNamespace(
            model_type="qwen2",
            architectures=["Qwen2ForCausalLM"],
        )
        self.assertTrue(
            invert.validate_suffix_v21_model_config(v21.METHOD_NAME, accepted)
        )
        rejected = (
            types.SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
            ),
            types.SimpleNamespace(
                model_type="qwen2",
                architectures=["Qwen2ForSequenceClassification"],
            ),
        )
        for model_config in rejected:
            with self.subTest(model_type=model_config.model_type):
                with self.assertRaisesRegex(ValueError, "Qwen2/Qwen2.5"):
                    invert.validate_suffix_v21_model_config(
                        v21.METHOD_NAME,
                        model_config,
                    )

    def test_committed_prefix_helper_supports_v21_and_preserves_v20(self):
        tokenizer = types.SimpleNamespace(
            all_special_ids=[0, 1],
            eos_token_id=0,
        )
        input_ids = torch.tensor([[5, 6]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        outputs = []
        for method in (v21.METHOD_NAME, v20.METHOD_NAME):
            outputs.append(
                invert.ensure_suffix_v2_committed_prefix(
                    input_ids,
                    attention_mask,
                    tokenizer,
                    method,
                )
            )
        for prefixed_ids, prefixed_mask in outputs:
            self.assertEqual([[0, 5, 6]], prefixed_ids.tolist())
            self.assertEqual([[1, 1, 1]], prefixed_mask.tolist())

    def test_combined_defaults_select_v21_without_cgmr(self):
        advanced = invert.load_config(ADVANCED_CONFIG)
        candidate = invert.load_config(CANDIDATE_CONFIG)
        self.assertEqual("v2.1.1", advanced["suffix_version"])
        self.assertTrue(advanced["suffix_reoptimization_v2_1_1"])
        self.assertEqual("none", candidate["cgmr_version"])
        self.assertFalse(candidate["cgmr_v1_0"])
        self.assertFalse(candidate["cgmr_v1_1"])
        self.assertFalse(candidate["cgmr_v1_2"])

    def test_v211_keeps_legacy_stage1_in_front_of_sidecar(self):
        source = inspect.getsource(invert.main)
        start = source.index("use_external_stage1 =")
        end = source.index("part_epoch =", start)
        stage1_selector_block = source[start:end]
        self.assertIn('"suffix_reoptimization_v2.1"', stage1_selector_block)
        self.assertNotIn('"suffix_reoptimization_v2.1.1"', stage1_selector_block)
        self.assertIn(
            'part_epoch = 0 if use_external_stage1 else total_epoch',
            source,
        )
        self.assertIn(
            '"legacy_stage1_then_v2_1_global_causal"',
            source,
        )


class FormalResultAndOutputTests(unittest.TestCase):
    def test_experiment_evaluation_does_not_mutate_frozen_v21_result(self):
        formal_result = {
            "version": "v2.1.1",
            "final_tokens": [0, 2, 4],
            "final_accuracy": None,
            "pre_acc": None,
            "post_acc": None,
            "accepted": True,
            "rollback": False,
            "diagnostics": {},
            "diagnostics_failed": False,
        }
        frozen = copy.deepcopy(formal_result)
        accuracy = invert.evaluate_frozen_suffix_v21_accuracy(
            formal_result,
            torch.tensor([[0, 2, 3]], dtype=torch.long),
            1,
        )
        self.assertEqual(0.5, accuracy)
        self.assertEqual(frozen, formal_result)
        self.assertIsNone(formal_result["final_accuracy"])
        self.assertEqual({}, formal_result["diagnostics"])

    def test_generic_fatal_exit_code_covers_v21_and_v20_results(self):
        self.assertEqual(
            2,
            invert.experiment_exit_code_for_records([{
                "suffix_reoptimization_v2_1_1_result": {
                    "fatal_failure": True,
                },
            }]),
        )
        self.assertEqual(
            2,
            invert.experiment_exit_code_for_records([{
                "suffix_reoptimization_v2_0_result": {
                    "fatal_failure": True,
                },
            }]),
        )
        self.assertEqual(0, invert.experiment_exit_code_for_records([{}]))

    def test_jsonl_wiring_keeps_v21_specific_and_canonical_results(self):
        source = inspect.getsource(invert.main)
        self.assertIn('"suffix_reoptimization_v2_1_1_result"', source)
        self.assertIn('"suffix_reoptimization_result"', source)
        formal = {"version": "v2.1.1", "fatal_failure": False}
        record = {
            "suffix_reoptimization_v2_1_1_result": formal,
            "suffix_reoptimization_result": formal,
        }
        self.assertIs(
            formal,
            invert.formal_method_result_for_record(record),
        )
        self.assertIs(formal, _suffix_result(record))

    def test_v21_jsonl_schema_excludes_v20_classifier_fields(self):
        v20_result = {
            "classifier_enabled": True,
            "classifier_provider_available": True,
            "classifier_candidate_count": 7,
        }
        v21_record = {"selected_advanced_method": v21.METHOD_NAME}
        v21_record.update(invert.suffix_v2_classifier_record_fields(
            v21.METHOD_NAME,
            v20_result,
        ))
        encoded = json.loads(json.dumps(v21_record))
        for key in (
            "classifier_enabled",
            "classifier_provider_available",
            "classifier_candidate_count",
        ):
            self.assertNotIn(key, encoded)

        v20_fields = invert.suffix_v2_classifier_record_fields(
            v20.METHOD_NAME,
            v20_result,
        )
        self.assertEqual(v20_result, v20_fields)
        self.assertIn(
            "record.update(suffix_v2_classifier_record_fields(",
            inspect.getsource(invert.main),
        )

    def test_resolved_v21_defaults_match_frozen_config(self):
        resolved = _resolved_suffix_v211_config(types.SimpleNamespace())
        self.assertFalse(resolved["enabled"])
        self.assertEqual([0, 1, 2], resolved["layer_offsets"])
        self.assertEqual([1.0, 0.5, 0.25], resolved["layer_weights"])
        self.assertEqual(0.005, resolved["vocab_weight"])
        self.assertEqual(10, resolved["vocab_anchor_top_k"])
        self.assertEqual(10, resolved["vocab_anchor_refresh_interval"])
        self.assertEqual(1000, resolved["global_steps"])
        self.assertEqual(50, resolved["local_steps"])
        self.assertEqual(0.15, resolved["tau_J"])
        self.assertEqual(0.01, resolved["delta_c_max"])
        self.assertEqual(0.05, resolved["tau_r"])
        self.assertEqual(10, resolved["embedding_top_k_normal"])
        self.assertEqual(20, resolved["embedding_top_k_expanded"])
        self.assertEqual(10, resolved["ppl_top_k"])
        self.assertFalse(resolved["accuracy_diagnostics_enabled"])
        self.assertTrue(resolved["filter_nonascii"])
        self.assertEqual(
            "normalized_stable_logsumexp",
            resolved["vocab_softmin_mode"],
        )
        config_path = os.path.join(
            ROOT,
            "suffix_optimization_methods",
            "configs",
            "suffix_reoptimization_v2_1_1.json",
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            config_payload = json.load(handle)
        self.assertEqual(
            {
                "normalized_stable_logsumexp",
            },
            {
                resolved["vocab_softmin_mode"],
                v21.SuffixReoptimizationV211Config().vocab_softmin_mode,
                config_payload["suffix_v2_1_1_vocab_softmin_mode"],
            },
        )
        self.assertFalse(resolved["use_cache"])
        self.assertTrue(resolved["formal_diagnostics_isolation"])

    def test_resolved_config_keeps_only_selected_v21_advanced_method(self):
        merged = invert.load_config(ADVANCED_CONFIG)
        merged.update({
            "config": ADVANCED_CONFIG,
            "output_dir": "v21",
            "seed": 0,
            "dataset_path": "not_loaded.csv",
            "dataset_type": "local",
            "dataset_len": 1,
            "base_model_name": "not_loaded_model",
            "lora_model_name": None,
            "num_invert_layers": 24,
            "quantization": "none",
            "lr": 0.1,
            "epoch": 1000,
            "alpha": 0.001,
            "clip": True,
            "init_method": "uniform",
            "init_param": 0.1,
            "optim_method": "cosine",
            "invert_method": "cosine",
            "filter_nonascii": True,
            "top_k_cos": 10,
            "perplexity": True,
            "top_k_ppl": 10,
            "selected_advanced_method": v21.METHOD_NAME,
            "suffix_version": "v2.1.1",
            "suffix_reoptimization_v2_1_1": True,
            "selected_candidate_reranking_method": "none",
            "cgmr_version": "none",
            "device_map": "auto",
            "offload_folder": None,
            "offload_state_dict": True,
            "max_memory": None,
            "log_dir": "results/invert_timestamp_runs",
        })
        resolved = build_resolved_config(
            types.SimpleNamespace(**merged),
            "timestamp",
            "run_dir",
            "experiment.log",
            "reconstructions.jsonl",
            None,
            1,
            28,
            "qwen2",
            28,
        )
        self.assertEqual(
            ["suffix_reoptimization_v2_1_1"],
            list(resolved["advanced_methods"]),
        )
        self.assertEqual({}, resolved["candidate_reranking_methods"])
        selected = resolved["advanced_methods"]["suffix_reoptimization_v2_1_1"]
        self.assertTrue(selected["enabled"])
        self.assertEqual("v2.1.1", selected["version"])
        self.assertFalse(selected["accuracy_diagnostics_enabled"])


if __name__ == "__main__":
    unittest.main()
