import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE.parents[1] / "一键运行_suffix_v2_1.py"
SPEC = importlib.util.spec_from_file_location("one_click_suffix_v21", LAUNCHER)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class OneClickSuffixV21Tests(unittest.TestCase):
    def test_launcher_targets_v21_layout_and_config(self):
        self.assertEqual(HERE.parents[2], launcher.PROJECT_DIR)
        self.assertEqual(
            "run_experiment_suffix_v2_1.sh",
            launcher.BOOTSTRAP_SCRIPT.name,
        )
        self.assertEqual("runner_suffix_v2_1.py", launcher.SINGLE_GPU_RUNNER.name)
        self.assertEqual(
            "l24_airport_medical_suffix_v2_1_1_no_cgmr.json",
            launcher.EXPERIMENT_CONFIG.name,
        )

    def test_launcher_uses_only_v21_bootstrap(self):
        completed = types.SimpleNamespace(returncode=0)
        run = mock.Mock(return_value=completed)
        status = launcher.main(
            [], platform_name="linux", machine="x86_64",
            which=lambda name: "/bin/bash", run=run,
        )
        self.assertEqual(0, status)
        command = run.call_args.args[0]
        self.assertEqual("/bin/bash", command[0])
        self.assertEqual("run_experiment_suffix_v2_1.sh", Path(command[1]).name)

    def test_smoke_test_is_forwarded_explicitly(self):
        completed = types.SimpleNamespace(returncode=0)
        run = mock.Mock(return_value=completed)
        with mock.patch.dict(os.environ, {}, clear=True):
            status = launcher.main(
                ["--smoke-test"],
                platform_name="linux",
                machine="x86_64",
                which=lambda name: "/bin/bash",
                run=run,
            )

        self.assertEqual(0, status)
        self.assertEqual("1", run.call_args.kwargs["env"]["DEML_SMOKE_TEST"])

    def test_unsupported_platform_does_not_start(self):
        run = mock.Mock()
        status = launcher.main(
            [], platform_name="win32", machine="amd64",
            which=lambda name: "bash", run=run,
        )
        self.assertEqual(3, status)
        run.assert_not_called()

    def test_launcher_is_the_only_v21_top_level_entry(self):
        self.assertEqual(
            {"一键运行_suffix_v2_0.py", "一键运行_suffix_v2_1.py"},
            {path.name for path in HERE.parents[1].glob("*.py")},
        )

    def test_bootstrap_is_v21_only_and_uses_shared_runtime(self):
        source = (HERE / "run_experiment_suffix_v2_1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("suffix_reoptimization_v2.1.1", source)
        self.assertIn(
            "l24_airport_medical_suffix_v2_1_1_no_cgmr.json", source
        )
        self.assertIn('RUNTIME_DIR="${BUNDLE_DIR}/.runtime"', source)
        self.assertIn("flock", source)
        self.assertIn("--smoke-test", source)
        self.assertNotIn("suffix_reoptimization_v2.0", source)
        self.assertNotIn("suffix_v1_", source)


if __name__ == "__main__":
    unittest.main()
