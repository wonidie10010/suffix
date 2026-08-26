import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import runner_suffix_v2_1 as runner


class RunnerSuffixV21Tests(unittest.TestCase):
    def test_runner_targets_suffix_v2_1_only(self):
        self.assertEqual(
            "experiment_configs/l24_airport_medical_suffix_v2_1_1_no_cgmr.json",
            runner.EXPERIMENT_CONFIG,
        )
        self.assertEqual("suffix_reoptimization_v2.1.1", runner.METHOD_DIRECTORY)
        self.assertEqual(
            (
                "resolved_config.json",
                "experiment.log",
                "reconstructions.jsonl",
            ),
            runner.REQUIRED_ARTIFACTS,
        )

    def test_progress_printer_is_monotonic_and_clamped(self):
        stream = io.StringIO()
        progress = runner.ProgressPrinter(stream=stream, initial=10)

        progress.update(9)
        progress.update(25)
        progress.update(25)
        progress.update(101)

        self.assertEqual(100, progress.last_percent)
        self.assertEqual(
            "\r实验总进度：10%\r实验总进度：25%\r实验总进度：100%",
            stream.getvalue(),
        )

    def test_smoke_config_is_v211_and_short(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            project = root / "project"
            official_config = (
                project
                / "experiment_configs"
                / "l24_airport_medical_suffix_v2_1_1_no_cgmr.json"
            )
            official_config.parent.mkdir(parents=True)
            official_config.write_text("{}", encoding="utf-8")

            smoke = runner.prepare_smoke_experiment(project, root / "runtime")
            config = json.loads(smoke["config"].read_text(encoding="utf-8"))
            data = json.loads(smoke["data"].read_text(encoding="utf-8"))

        self.assertEqual(["Test."], data)
        self.assertEqual([str(official_config.resolve())], config["include_configs"])
        self.assertEqual(1, config["dataset_len"])
        self.assertEqual(1, config["epoch"])
        self.assertEqual(1, config["suffix_v2_1_1_global_steps"])
        self.assertEqual(1, config["suffix_v2_1_1_local_steps"])
        self.assertFalse(config["suffix_v2_1_1_accuracy_diagnostics_enabled"])
        self.assertEqual(smoke["run_root"], Path(config["log_dir"]))

    def test_experiment_launches_one_v21_process_without_parallel_flags(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            project = root / "project"
            project.mkdir()
            log_file = root / "experiment.log"
            progress = runner.ProgressPrinter(
                stream=io.StringIO(), initial=35
            )
            process = mock.Mock()
            process.poll.side_effect = [None, 0]
            process.wait.return_value = 0
            with mock.patch.object(
                    runner.subprocess, "Popen", return_value=process) as popen:
                status, run_dir = runner.run_experiment(
                    project,
                    root / "runtime",
                    log_file,
                    progress,
                    {"CUDA_VISIBLE_DEVICES": "0"},
                    interval=0,
                )

        self.assertEqual(0, status)
        self.assertIsNone(run_dir)
        self.assertEqual(1, popen.call_count)
        command = popen.call_args.args[0]
        self.assertEqual(str(project / "invert.py"), command[1])
        self.assertEqual(
            Path(runner.EXPERIMENT_CONFIG).name,
            Path(command[-1]).name,
        )
        self.assertNotIn("--parallel-worker-spec", command)
        self.assertNotIn("--worker-output-dir", command)

    def test_runtime_environment_exposes_exactly_one_selected_gpu(self):
        with mock.patch.dict(
                runner.os.environ, {"DEML_GPU_ID": "2"}, clear=True):
            environment = runner.runtime_environment("runtime")

        self.assertEqual("2", environment["CUDA_VISIBLE_DEVICES"])

    def test_runtime_environment_rejects_multiple_gpu_ids(self):
        with (
            mock.patch.dict(
                runner.os.environ, {"DEML_GPU_ID": "0,1"}, clear=True
            ),
            self.assertRaisesRegex(ValueError, "DEML_GPU_ID"),
        ):
            runner.runtime_environment("runtime")

    def test_model_command_checks_cache_before_downloading(self):
        command = runner.model_download_command()

        self.assertIn("local_files_only=True", command[-1])
        self.assertIn(runner.MODEL_ID, command[-1])

    def _make_run(self, temporary_dir):
        project = Path(temporary_dir) / "project"
        run_dir = (
            project
            / "results"
            / "invert_timestamp_runs"
            / runner.METHOD_DIRECTORY
            / "20260824-120000"
        )
        run_dir.mkdir(parents=True)
        for index, artifact in enumerate(runner.REQUIRED_ARTIFACTS):
            (run_dir / artifact).write_text(
                "artifact-{}".format(index),
                encoding="utf-8",
            )
        return project, run_dir

    def test_result_copy_preserves_source_and_matches_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            project, run_dir = self._make_run(temporary_dir)
            result_root = Path(temporary_dir) / "shared-results"
            before = {
                name: runner.sha256_file(run_dir / name)
                for name in runner.REQUIRED_ARTIFACTS
            }

            destination = runner.copy_and_verify_results(
                project,
                result_root,
                run_dir,
            )

            after = {
                name: runner.sha256_file(run_dir / name)
                for name in runner.REQUIRED_ARTIFACTS
            }
            copied = {
                name: runner.sha256_file(destination / name)
                for name in runner.REQUIRED_ARTIFACTS
            }

        self.assertEqual(before, after)
        self.assertEqual(before, copied)
        self.assertEqual(runner.METHOD_DIRECTORY, destination.parent.name)

    def test_copy_rejects_run_outside_v21_result_root(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            outside = root / "outside"
            outside.mkdir()
            for artifact in runner.REQUIRED_ARTIFACTS:
                (outside / artifact).write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "outside"):
                runner.copy_and_verify_results(
                    root / "project",
                    root / "results",
                    outside,
                )

    def _args(self, temporary_dir):
        root = Path(temporary_dir)
        project = root / "project"
        project.mkdir()
        return types.SimpleNamespace(
            project=str(project),
            runtime=str(root / "runtime"),
            result_root=str(root / "results"),
            log_file=str(root / "runtime" / "logs" / "run.log"),
        )

    def test_model_failure_has_dedicated_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = self._args(temporary_dir)
            with (
                mock.patch.object(
                    runner,
                    "run_with_heartbeat",
                    return_value=1,
                ),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                status = runner.run_bundle(args)

        self.assertEqual(runner.EXIT_MODEL_FAILURE, status)

    def test_successful_bundle_copies_and_verifies_complete_result(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = self._args(temporary_dir)
            project = Path(args.project)
            run_dir = (
                project
                / "results"
                / "invert_timestamp_runs"
                / runner.METHOD_DIRECTORY
                / "flow-test"
            )
            run_dir.mkdir(parents=True)
            for index, artifact in enumerate(runner.REQUIRED_ARTIFACTS):
                (run_dir / artifact).write_text(
                    "artifact-{}".format(index), encoding="utf-8"
                )
            with (
                mock.patch.object(
                    runner, "run_with_heartbeat", return_value=0
                ),
                mock.patch.object(
                    runner, "run_experiment", return_value=(0, run_dir)
                ),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                status = runner.run_bundle(args)

            destination = (
                Path(args.result_root)
                / runner.METHOD_DIRECTORY
                / run_dir.name
            )
            self.assertEqual(0, status)
            self.assertTrue(run_dir.is_dir())
            self.assertTrue(destination.is_dir())
            for artifact in runner.REQUIRED_ARTIFACTS:
                self.assertEqual(
                    runner.sha256_file(run_dir / artifact),
                    runner.sha256_file(destination / artifact),
                )


if __name__ == "__main__":
    unittest.main()
