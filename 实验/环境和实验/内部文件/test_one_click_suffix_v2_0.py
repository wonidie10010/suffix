import importlib.util
import os
from pathlib import Path
import types
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
LAUNCHER = HERE.parents[1] / "一键运行_suffix_v2_0.py"
SPEC = importlib.util.spec_from_file_location("one_click_suffix_v20", LAUNCHER)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class OneClickSuffixV20Tests(unittest.TestCase):
    def test_launcher_uses_repository_root_project(self):
        self.assertEqual(HERE.parents[2], launcher.PROJECT_DIR)

    def test_launcher_uses_only_v2_bootstrap(self):
        completed = types.SimpleNamespace(returncode=0)
        run = mock.Mock(return_value=completed)
        status = launcher.main(
            [], platform_name="linux", machine="x86_64",
            which=lambda name: "/bin/bash", run=run,
        )
        self.assertEqual(0, status)
        command = run.call_args.args[0]
        self.assertEqual("/bin/bash", command[0])
        self.assertEqual("run_experiment.sh", Path(command[1]).name)

    def test_smoke_test_is_forwarded_explicitly_without_changing_default(self):
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

    def test_launcher_is_the_only_python_file_at_experiment_top_level(self):
        self.assertEqual([LAUNCHER], list(HERE.parents[1].glob("*.py")))
        self.assertEqual([], list(HERE.parent.glob("*.py")))

    def test_bootstrap_prepares_environment_and_runs_only_v2(self):
        source = (HERE / "run_experiment.sh").read_text(encoding="utf-8")
        self.assertIn("download_miniconda", source)
        self.assertIn("-m pip install", source)
        self.assertIn("-m pip check", source)
        self.assertIn("torch.cuda.is_available()", source)
        self.assertIn('"${SCRIPT_DIR}/runner.py" check-env', source)
        self.assertIn('"${SCRIPT_DIR}/runner.py" "${RUNNER_ARGS[@]}"', source)
        self.assertIn("RUNNER_ARGS=(", source)
        self.assertIn("RUNNER_ARGS+=(--smoke-test)", source)
        self.assertIn("CASE_ENTRY_COUNT", source)
        self.assertIn('${HOME}/.cache/deml-one-click', source)
        self.assertGreaterEqual(source.count("--override-channels"), 2)
        self.assertGreaterEqual(source.count("--channel conda-forge"), 2)
        self.assertNotIn("suffix_v2_0_parallel_runner.py", source)
        self.assertNotIn("suffix_v1_", source)


if __name__ == "__main__":
    unittest.main()
