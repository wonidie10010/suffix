import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import runner


class RunnerProgressTests(unittest.TestCase):
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

    def test_experiment_percent_covers_samples_and_stage1(self):
        values = [
            runner.experiment_percent({
                "phase": "sample_optimization",
                "sample_index": 0,
                "total_samples": 10,
                "completed_steps": 0,
                "total_steps": 1000,
            }),
            runner.experiment_percent({
                "phase": "sample_optimization",
                "sample_index": 0,
                "total_samples": 10,
                "completed_steps": 1000,
                "total_steps": 1000,
            }),
            runner.experiment_percent({
                "phase": "sample_completed",
                "completed_samples": 1,
                "total_samples": 10,
            }),
            runner.experiment_percent({
                "phase": "experiment_completed",
                "completed_samples": 10,
                "total_samples": 10,
            }),
        ]

        self.assertEqual(sorted(values), values)
        self.assertEqual(50, values[0])
        self.assertEqual(99, values[-1])

    def test_subprocess_output_goes_to_log_not_console(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            log_file = Path(temporary_dir) / "run.log"
            stream = io.StringIO()
            progress = runner.ProgressPrinter(stream=stream, initial=0)

            status = runner.run_with_heartbeat(
                [sys.executable, "-c", "print('hidden child output')"],
                log_file,
                progress,
                1,
                2,
                interval=0.001,
            )

            self.assertEqual(0, status)
            self.assertNotIn("hidden child output", stream.getvalue())
            self.assertIn(
                "hidden child output",
                log_file.read_text(encoding="utf-8"),
            )


class RunnerEnvironmentTests(unittest.TestCase):
    def test_environment_checker_accepts_exact_versions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            project = Path(temporary_dir)
            (project / "requirements.txt").write_text(
                "--extra-index-url https://example.invalid\n"
                "Example_Package==1.2.3\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    runner.sys,
                    "version_info",
                    (3, 10, 20),
                ),
                mock.patch.object(
                    runner.importlib.metadata,
                    "version",
                    return_value="1.2.3",
                ),
            ):
                mismatches = runner.environment_mismatches(project)

        self.assertEqual([], mismatches)

    def test_model_command_checks_cache_before_downloading(self):
        command = runner.model_download_command()

        self.assertIn("local_files_only=True", command[-1])
        self.assertIn(runner.MODEL_ID, command[-1])


class RunnerResultTests(unittest.TestCase):
    def _make_run(self, temporary_dir):
        project = Path(temporary_dir) / "project"
        run_dir = (
            project
            / "results"
            / "invert_timestamp_runs"
            / runner.METHOD_DIRECTORY
            / "20260729-120000"
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

    def test_hash_mismatch_removes_partial_destination(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            project, run_dir = self._make_run(temporary_dir)
            result_root = Path(temporary_dir) / "shared-results"
            destination = (
                result_root
                / runner.METHOD_DIRECTORY
                / run_dir.name
            )

            with (
                mock.patch.object(
                    runner,
                    "sha256_file",
                    side_effect=["source", "different"],
                ),
                self.assertRaisesRegex(RuntimeError, "hash mismatch"),
            ):
                runner.copy_and_verify_results(
                    project,
                    result_root,
                    run_dir,
                )

            self.assertFalse(destination.exists())
            self.assertTrue(run_dir.exists())


class RunnerExitCodeTests(unittest.TestCase):
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
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                status = runner.run_bundle(args)

        self.assertEqual(runner.EXIT_MODEL_FAILURE, status)

    def test_experiment_failure_has_dedicated_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = self._args(temporary_dir)
            with (
                mock.patch.object(
                    runner,
                    "run_with_heartbeat",
                    return_value=0,
                ),
                mock.patch.object(
                    runner,
                    "run_experiment",
                    return_value=(1, None),
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                status = runner.run_bundle(args)

        self.assertEqual(runner.EXIT_EXPERIMENT_FAILURE, status)

    def test_result_failure_has_dedicated_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            args = self._args(temporary_dir)
            with (
                mock.patch.object(
                    runner,
                    "run_with_heartbeat",
                    return_value=0,
                ),
                mock.patch.object(
                    runner,
                    "run_experiment",
                    return_value=(0, Path(temporary_dir) / "run"),
                ),
                mock.patch.object(
                    runner,
                    "copy_and_verify_results",
                    side_effect=RuntimeError("copy failed"),
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                status = runner.run_bundle(args)

        self.assertEqual(runner.EXIT_RESULT_FAILURE, status)


if __name__ == "__main__":
    unittest.main()
