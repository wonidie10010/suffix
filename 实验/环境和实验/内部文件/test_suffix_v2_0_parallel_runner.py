import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
RUNNER_PATH = HERE / "suffix_v2_0_parallel_runner.py"
SPEC = importlib.util.spec_from_file_location("suffix_v20_parallel_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

OUTPUT_SPEC = importlib.util.spec_from_file_location(
    "suffix_v20_test_experiment_outputs", PROJECT / "experiment_outputs.py"
)
experiment_outputs = importlib.util.module_from_spec(OUTPUT_SPEC)
OUTPUT_SPEC.loader.exec_module(experiment_outputs)


def synthetic_resolved_config():
    return {
        "run": {"timestamp": "worker", "execution_mode": "worker"},
        "dataset": {"len_setting": 10},
        "model": {"num_invert_layers": 24, "device_map": "single_gpu"},
        "datasets": [
            {"name": "airport", "len": 5},
            {"name": "medical", "len": 5},
        ],
        "advanced_method": {
            "name": "suffix_reoptimization_v2.0", "enabled": True,
            "suffix_version": "v2.0",
        },
        "candidate_reranking_method": {"name": "none", "enabled": False},
        "selectors": {
            "suffix_version": "v2.0",
            "candidate_reranking": "none",
        },
        "advanced_methods": {
            "suffix_reoptimization_v2_0": {
                "enabled": True,
                "classifier_enabled": False,
                "classifier_provider_available": False,
                "classifier_candidate_count": 0,
            },
        },
        "artifacts": {"worker": True},
    }


def synthetic_fingerprint(config):
    clone = json.loads(json.dumps(config, ensure_ascii=False))
    clone.get("artifacts", {}).clear()
    clone.get("outputs", {}).clear()
    clone.get("run", {}).pop("timestamp", None)
    payload = json.dumps(
        clone, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def synthetic_record(spec, global_index):
    dataset_name, dataset_index = runner.SAMPLE_MAP[global_index]
    return {
        "global_index": global_index,
        "dataset_name": dataset_name,
        "dataset_sample_index": dataset_index,
        "assigned_worker_id": spec["worker_id"],
        "assigned_physical_gpu_id": spec["physical_gpu_id"],
        "sample_index": global_index,
        "dataset": {
            "name": dataset_name,
            "sample_number": dataset_index + 1,
            "sample_count": 5,
        },
        "selected_advanced_method": runner.METHOD_DIRECTORY,
        "selected_candidate_reranking_method": "none",
        "method": runner.METHOD_DIRECTORY,
        "version": runner.VERSION,
        "accepted": True,
        "rollback": False,
        "fatal_failure": False,
        "classifier_enabled": False,
        "classifier_provider_available": False,
        "classifier_candidate_count": 0,
        "num_invert_layers": 24,
        "token_length": 3,
        "elapsed_seconds": 0.01,
        "stage_accuracy": {"pre_suffix": 0.0, "suffix_v2_0": 1.0},
        "suffix_reoptimization_v2_0_result": {
            "enabled": True,
            "accepted": True,
            "rollback": False,
            "fatal_failure": False,
            "final_tokens": [1, 2, 3],
            "pre_acc": 0.0,
            "post_acc": 1.0,
            "classifier_enabled": False,
            "classifier_provider_available": False,
            "classifier_candidate_count": 0,
        },
        "suffix_reoptimization_result": {
            "enabled": True,
            "accepted": True,
            "final_tokens": [1, 2, 3],
            "pre_acc": 0.0,
            "post_acc": 1.0,
        },
        "candidate_reranking_result": {
            "name": "none", "enabled": False, "skipped": True,
        },
    }


class FakePopenFactory:
    def __init__(self, failing_worker=None):
        self.failing_worker = failing_worker
        self.calls = []
        self.wait_calls = []

    def __call__(self, command, cwd, env, stdout, stderr):
        spec_path = Path(command[command.index("--parallel-worker-spec") + 1])
        worker_dir = Path(command[command.index("--worker-output-dir") + 1])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        worker_id = spec["worker_id"]
        self.calls.append({
            "command": list(command),
            "cwd": cwd,
            "env": dict(env),
            "spec": spec,
            "worker_dir": worker_dir,
        })
        records = [synthetic_record(spec, index) for index in spec["assigned_global_indices"]]
        with (worker_dir / "shard_reconstructions.jsonl").open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        failed = worker_id == self.failing_worker
        resolved = synthetic_resolved_config()
        resolved["outputs"] = {
            "run_dir": str(worker_dir),
            "reconstructions": str(
                worker_dir / "shard_reconstructions.jsonl"
            ),
        }
        status = {
            "worker_id": worker_id,
            "physical_gpu_id": spec["physical_gpu_id"],
            "assigned_global_indices": spec["assigned_global_indices"],
            "completed_global_indices": (
                spec["assigned_global_indices"] if not failed
                else spec["assigned_global_indices"][:-1]
            ),
            "record_count": len(records),
            "exit_code": 1 if failed else 0,
            "started_at": "2026-08-04T00:00:00Z",
            "finished_at": "2026-08-04T00:00:01Z",
            "success": not failed,
            "failure_reason": "synthetic failure" if failed else None,
            "local_device": "cuda:0",
            "device_map": {"": 0},
            "resolved_config": resolved,
            "runtime_config_fingerprint": synthetic_fingerprint(resolved),
            "model_metadata": resolved["model"],
        }
        (worker_dir / "worker_status.json").write_text(
            json.dumps(status, ensure_ascii=False), encoding="utf-8"
        )
        stdout.write("worker {} stdout\n".format(worker_id))
        stderr.write("worker {} stderr\n".format(worker_id))
        stdout.flush()
        stderr.flush()
        factory = self

        class Process:
            def wait(self):
                if len(factory.calls) != 4:
                    raise AssertionError("workers were waited before all four launched")
                factory.wait_calls.append(worker_id)
                return 1 if failed else 0

        return Process()


class ParallelRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, factory, timestamp="20260804-000000"):
        return runner.run_parallel(
            PROJECT,
            self.repo,
            "python",
            timestamp=timestamp,
            gpu_probe=lambda _: [0, 1, 2, 3],
            popen=factory,
            prepare_model=False,
            experiment_outputs_loader=lambda _: experiment_outputs,
        )

    def test_fixed_shards_cover_global_indices_once(self):
        runner.validate_static_shards()
        self.assertEqual(
            [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]],
            list(runner.SHARD_MAP.values()),
        )
        flat = [index for shard in runner.SHARD_MAP.values() for index in shard]
        self.assertEqual(list(range(10)), sorted(flat))

    def test_preflight_rejects_less_than_four_gpus_without_fallback(self):
        with self.assertRaisesRegex(runner.ParallelRunError, "GPU probe"):
            runner.preflight(
                PROJECT, self.repo, "python", "less-than-four",
                gpu_probe=lambda _: [0, 1],
            )
        self.assertFalse((self.repo / "results").exists())

    def test_mock_dry_run_does_not_create_official_run(self):
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = runner.main([
                "mock-dry-run", "--project", str(PROJECT),
                "--repo-root", str(self.repo), "--timestamp", "dry-run",
            ])
        self.assertEqual(0, exit_code)
        summary = json.loads(output.getvalue())
        self.assertEqual(4, len(summary["worker_commands"]))
        self.assertEqual(runner.SHARD_MAP, summary["shard_map"])
        self.assertTrue(all(
            Path(command[1]).is_absolute()
            for command in summary["worker_commands"].values()
        ))
        self.assertFalse((self.repo / "results").exists())
        self.assertFalse((self.repo / "outputs").exists())

    def test_launch_is_concurrent_and_gpu_isolated(self):
        temporary = self.repo / "workers"
        temporary.mkdir()
        factory = FakePopenFactory()
        processes = runner.launch_workers(
            "python", PROJECT,
            PROJECT / runner.CONFIG_RELATIVE_PATH,
            temporary, "stamp", popen=factory,
        )
        exit_codes = runner.wait_workers(processes)
        self.assertEqual({str(index): 0 for index in range(4)}, exit_codes)
        self.assertEqual([0, 1, 2, 3], factory.wait_calls)
        self.assertEqual(
            ["0", "1", "2", "3"],
            [call["env"]["CUDA_VISIBLE_DEVICES"] for call in factory.calls],
        )
        for call in factory.calls:
            self.assertEqual("cuda:0", call["spec"]["local_device"])
            self.assertIn("single_gpu", call["command"])
        self.assertFalse((temporary / "reconstructions.jsonl").exists())
        self.assertFalse((temporary / "experiment.log").exists())

    def test_success_produces_one_sorted_official_run_and_one_copy(self):
        factory = FakePopenFactory()
        parent, copied = self._run(factory)
        self.assertTrue(parent.is_dir())
        self.assertTrue(copied.is_dir())
        official_runs = list(
            (self.repo / "results" / "invert_timestamp_runs"
             / runner.METHOD_DIRECTORY).iterdir()
        )
        self.assertEqual([parent], official_runs)
        copied_runs = list((self.repo / "实验" / "结果").iterdir())
        self.assertEqual([copied], copied_runs)
        self.assertFalse((copied / "airport").exists())
        self.assertFalse((copied / "medical").exists())
        self.assertFalse((copied / "worker_0").exists())
        self.assertTrue((copied / "worker_logs").is_dir())
        records = runner.load_jsonl(parent / "reconstructions.jsonl")
        self.assertEqual(10, len(records))
        self.assertEqual(
            [("airport", index) for index in range(5)]
            + [("medical", index) for index in range(5)],
            [(record["dataset_name"], record["dataset_sample_index"])
             for record in records],
        )
        self.assertEqual(1, len(list(parent.glob("resolved_config.json"))))
        self.assertEqual(1, len(list(parent.glob("experiment.log"))))
        log_text = (parent / "experiment.log").read_text(encoding="utf-8")
        self.assertEqual(1, log_text.count("===== overall average accuracy ====="))
        self.assertFalse(
            (self.repo / "outputs" / "suffix_v2_0_parallel"
             / "20260804-000000").exists()
        )

    def test_success_metadata_and_hashes_are_exact(self):
        parent, copied = self._run(FakePopenFactory(), "20260804-000001")
        resolved = runner.load_json(parent / "resolved_config.json")
        parallel = resolved["parallel_execution"]
        self.assertEqual("sample_parallel", parallel["mode"])
        self.assertEqual([0, 1, 2, 3], parallel["physical_gpu_ids"])
        self.assertEqual(runner.SHARD_MAP, parallel["shard_map"])
        self.assertFalse(parallel["future_multi_gpu_enabled"])
        self.assertEqual(
            {
                "suffix_v2_0_enabled": True,
                "legacy_suffix_enabled": False,
                "cgmr_enabled": False,
                "legacy_local_repair_enabled": False,
            },
            resolved["method_exclusivity"],
        )
        manifest = runner.load_json(parent / "run_manifest.json")
        self.assertTrue(manifest["overall_success"])
        self.assertTrue(manifest["merge_success"])
        self.assertTrue(manifest["output_validation_success"])
        self.assertEqual({str(index): 0 for index in range(4)}, manifest["worker_exit_codes"])
        for name in runner.CORE_ARTIFACTS:
            self.assertEqual(
                hashlib.sha256((parent / name).read_bytes()).hexdigest(),
                hashlib.sha256((copied / name).read_bytes()).hexdigest(),
            )

    def test_worker_logs_have_separate_stdout_stderr_sections(self):
        parent, _ = self._run(FakePopenFactory(), "20260804-000002")
        logs = list((parent / "worker_logs").glob("worker_*.log"))
        self.assertEqual(4, len(logs))
        for path in logs:
            text = path.read_text(encoding="utf-8")
            self.assertIn("===== STDOUT =====", text)
            self.assertIn("===== STDERR =====", text)

    def test_any_worker_failure_keeps_temp_and_has_no_success_merge_or_copy(self):
        timestamp = "20260804-000003"
        factory = FakePopenFactory(failing_worker=2)
        with self.assertRaises(runner.ParallelRunError):
            self._run(factory, timestamp)
        self.assertEqual([0, 1, 2, 3], factory.wait_calls)
        parent = (
            self.repo / "results" / "invert_timestamp_runs"
            / runner.METHOD_DIRECTORY / timestamp
        )
        self.assertTrue(parent.is_dir())
        self.assertFalse((parent / "reconstructions.jsonl").exists())
        self.assertFalse((self.repo / "实验" / "结果").exists())
        temporary = (
            self.repo / "outputs" / "suffix_v2_0_parallel" / timestamp
        )
        self.assertTrue(temporary.is_dir())
        self.assertTrue((temporary / "worker_2" / "stderr.log").is_file())
        manifest = runner.load_json(parent / "run_manifest.json")
        self.assertFalse(manifest["overall_success"])
        self.assertFalse(manifest["merge_success"])
        self.assertEqual(
            {"0": 3, "1": 3, "2": 2, "3": 2},
            manifest["worker_record_counts"],
        )
        self.assertIn("worker 2", manifest["failure_reason"])

    def test_merge_rejects_duplicate_missing_or_wrong_mapping(self):
        specs = [runner.worker_spec(i, i, "stamp") for i in range(4)]
        records = [
            synthetic_record(spec, index)
            for spec in specs
            for index in spec["assigned_global_indices"]
        ]
        runner.validate_merged_records(records)
        duplicate = list(records)
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaises(runner.ParallelRunError):
            runner.validate_merged_records(duplicate)
        wrong = [dict(record) for record in records]
        wrong[0]["dataset_sample_index"] = 4
        with self.assertRaises(runner.ParallelRunError):
            runner.validate_merged_records(wrong)

    def test_atomic_merge_leaves_no_temporary_file(self):
        specs = [runner.worker_spec(i, i, "stamp") for i in range(4)]
        records = [
            synthetic_record(spec, index)
            for spec in specs
            for index in spec["assigned_global_indices"]
        ]
        path = self.repo / "reconstructions.jsonl"
        runner.atomic_write_jsonl(path, runner.validate_merged_records(records))
        self.assertTrue(path.is_file())
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_static_source_does_not_claim_or_enable_tensor_parallel(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        for forbidden in ("DataParallel(", "DistributedDataParallel(", "tp_plan", "torchrun"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"future_multi_gpu_enabled": False', source)

    def test_legacy_runner_and_profile_remain_available(self):
        self.assertTrue((HERE / "runner.py").is_file())
        self.assertTrue(
            (PROJECT / "experiment_configs"
             / "l24_airport_medical_suffix_v1_2_3_no_cgmr.json").is_file()
        )


if __name__ == "__main__":
    unittest.main()
