import importlib
import os
import unittest

from CGMR import (
    CGMRV10Config,
    CGMRV11Config,
    CGMRV12Config,
    run_cgmr_v1_0,
    run_cgmr_v1_1,
    run_cgmr_v1_2,
)
from experiment_outputs import _resolved_candidate_config_view
from invert import (
    _nonfixed_token_accuracy,
    load_config,
    normalize_cgmr_version,
    select_cgmr_method,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVANCED_CONFIG = os.path.join(
    ROOT, "suffix_optimization_methods", "configs", "advanced_methods.json"
)


class MethodPackageLayoutTests(unittest.TestCase):
    def test_all_suffix_versions_import_from_method_versions(self):
        modules = {
            "v1_0": "suffix_reoptimization_v1_0",
            "v1_1": "suffix_reoptimization_v1_1",
            "v1_2": "suffix_reoptimization_v1_2",
            "v1_2_1": "suffix_reoptimization_v1_2_1",
            "v1_2_2": "suffix_reoptimization_v1_2_2",
            "v1_2_3": "suffix_v1_2_3",
            "v1_3": "suffix_reoptimization_v1_3",
            "v1_3_1": "suffix_reoptimization_v1_3_1",
            "v1_4": "suffix_reoptimization_v1_4",
            "v1_4_1": "suffix_reoptimization_v1_4_1",
        }
        for version, module_name in modules.items():
            module = importlib.import_module(
                "suffix_optimization_methods.method_versions."
                + module_name
            )
            self.assertTrue(hasattr(module, "METHOD_NAME"))
            if version == "v1_2_3":
                self.assertTrue(hasattr(module, "run_suffix_v1_2_3"))
            else:
                self.assertTrue(hasattr(module, "run_suffix_reoptimization"))

    def test_removed_local_embedding_repair_is_not_importable(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("CGMR.local_embedding_repair")

    def test_cgmr_v1_0_public_api(self):
        self.assertEqual("CGMR.method_versions.CGMR_v1_0", CGMRV10Config.__module__)
        self.assertEqual("CGMR.method_versions.CGMR_v1_0", run_cgmr_v1_0.__module__)

    def test_cgmr_v1_0_selector_uses_short_name(self):
        config = CGMRV10Config(enabled=True)

        self.assertEqual("v1.0", normalize_cgmr_version("CGMR_v1.0"))
        self.assertEqual("CGMR_v1.0", select_cgmr_method("v1.0", config))

    def test_cgmr_v1_1_public_api_and_selector(self):
        v1_0_config = CGMRV10Config(enabled=True)
        v1_1_config = CGMRV11Config(enabled=True)

        self.assertEqual("CGMR.method_versions.CGMR_v1_1", CGMRV11Config.__module__)
        self.assertEqual("CGMR.method_versions.CGMR_v1_1", run_cgmr_v1_1.__module__)
        self.assertEqual("v1.1", normalize_cgmr_version("CGMR_v1.1"))
        self.assertEqual(
            "CGMR_v1.1",
            select_cgmr_method("v1.1", v1_0_config, v1_1_config),
        )

    def test_cgmr_v1_2_public_api_aliases_and_selector_compatibility(self):
        v1_0_config = CGMRV10Config(enabled=True)
        v1_1_config = CGMRV11Config(enabled=True)
        v1_2_config = CGMRV12Config(enabled=True)

        self.assertEqual("CGMR.method_versions.CGMR_v1_2", CGMRV12Config.__module__)
        self.assertEqual("CGMR.method_versions.CGMR_v1_2", run_cgmr_v1_2.__module__)
        for alias in ("1.2", "v1.2", "cgmr_v1.2", "cgmr_v1_2"):
            with self.subTest(alias=alias):
                self.assertEqual("v1.2", normalize_cgmr_version(alias))
        self.assertEqual(
            "CGMR_v1.2",
            select_cgmr_method("v1.2", v1_0_config, v1_1_config, v1_2_config),
        )
        self.assertEqual(
            "CGMR_v1.1",
            select_cgmr_method(None, v1_0_config, v1_1_config),
        )
        self.assertEqual(
            "CGMR_v1.2",
            select_cgmr_method(
                None,
                v1_0_config,
                v1_1_config,
                v1_2_config,
            ),
        )
        with self.assertRaisesRegex(ValueError, "v1.2.*disabled"):
            select_cgmr_method(
                "v1.2",
                v1_0_config,
                v1_1_config,
                CGMRV12Config(enabled=False),
            )

    def test_resolved_candidate_config_is_multiversion_only_for_v1_2(self):
        configs = {
            "cgmr_v1_0": {"enabled": True},
            "cgmr_v1_1": {"enabled": True},
            "cgmr_v1_2": {"enabled": True},
        }

        self.assertEqual(
            configs,
            _resolved_candidate_config_view(configs, "CGMR_v1.2"),
        )
        self.assertEqual(
            {"cgmr_v1_1": {"enabled": True}},
            _resolved_candidate_config_view(configs, "CGMR_v1.1"),
        )
        self.assertEqual(
            {"cgmr_v1_0": {"enabled": True}},
            _resolved_candidate_config_view(configs, "CGMR_v1.0"),
        )

    def test_v1_2_accuracy_excludes_fixed_prefix(self):
        self.assertEqual(
            2 / 3,
            _nonfixed_token_accuracy(
                [[99, 1, 2, 3]],
                [0, 1, 7, 3],
                eval_start_pos=1,
            ),
        )
        self.assertEqual(
            0.0,
            _nonfixed_token_accuracy([[99]], [0], eval_start_pos=1),
        )

    def test_advanced_config_merges_versioned_suffix_configs(self):
        merged = load_config(ADVANCED_CONFIG)

        self.assertEqual("v2.1", merged["suffix_version"])
        self.assertTrue(merged["suffix_reoptimization_v2_1"])
        self.assertEqual("none", merged["cgmr_version"])
        self.assertFalse(merged["cgmr_v1_0"])
        self.assertFalse(merged["cgmr_v1_1"])
        self.assertFalse(merged["cgmr_v1_2"])
        self.assertNotIn("local_embedding_repair", merged)
        for version in (
            "v1_0",
            "v1_1",
            "v1_2",
            "v1_2_1",
            "v1_2_2",
            "v1_3",
            "v1_3_1",
            "v1_4",
            "v1_4_1",
            "v2_0",
            "v2_1",
        ):
            self.assertIn("suffix_reoptimization_{}".format(version), merged)
        self.assertIn("suffix_v1_2_3", merged)

    def test_legacy_root_configs_directory_is_removed(self):
        self.assertFalse(os.path.exists(os.path.join(ROOT, "configs")))


if __name__ == "__main__":
    unittest.main()
