import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


runner=module("runner_v222","实验/环境和实验/内部文件/runner_suffix_v2_2_2.py")
launcher=module("launcher_v222","实验/一键运行_suffix_v2_2_2.py")


def checkpoint_event(**overrides):
    event=dict(schema_version=4,candidate_top_k=3,acceptance_metric="pointwise_logmeanexp",
               acceptance_epsilon_source="per_window_repeated_forward_range",
               acceptance_calibration_repeats=3,accepted=False,repair_attempt_count=0)
    event.update(overrides)
    return event


class RunnerTests(unittest.TestCase):
    def test_event_contract_rejects_old_schema_and_invalid_acceptance(self):
        valid=checkpoint_event(accepted=True,repair_attempt_count=1,all_candidates_scored=True,
            acceptance_calibration_scores=[.25,.28125,.25],acceptance_epsilon=.03125,
            acceptance_threshold=.21875,D_win_before=.25,D_win_after=.1875)
        runner.validate_checkpoint_event(valid)
        for change in (dict(schema_version=3),dict(D_win_after=.21875),
                       dict(acceptance_epsilon=0.),dict(all_candidates_scored=False),
                       dict(acceptance_calibration_scores=[.25,float('nan'),.25])):
            with self.subTest(change=change),self.assertRaises(ValueError):
                runner.validate_checkpoint_event(dict(valid,**change))

    def test_summary_counts_calibration_cost_and_failures(self):
        events=[checkpoint_event(observation_forward_calls=1,calibration_forward_calls=2,
            candidate_forward_calls=3,downstream_forward_calls=6,acceptance_epsilon=.001),
            checkpoint_event(observation_forward_calls=1,calibration_forward_calls=1,
                             reason="acceptance_calibration_failed")]
        result=dict(pair_id="test:0",stage1_snapshot_sha256="fixture",
                    checkpoint=dict(complete_segment_count=2,events=events))
        row=runner.summarize_on([dict(accuracy=.5,suffix_reoptimization_v2_2_2_result=result)])["samples"][0]
        self.assertEqual(14,row["cp_forward_calls"])
        self.assertEqual(3,row["calibration_forward_calls"])
        self.assertEqual(1,row["calibration_failure_count"])
        self.assertEqual([.001],row["acceptance_epsilon_values"])

    def test_busy_gpu_blocks_before_launch(self):
        for outputs in (["1234"],["", "10, 0"],["", "0, 1000"]):
            run=mock.Mock(side_effect=[types.SimpleNamespace(stdout=text) for text in outputs])
            with self.assertRaises(RuntimeError):runner.ensure_gpu_idle("0",run)

    def test_local_cache_completeness_no_network_and_revision(self):
        temp_root=ROOT/"outputs/checkpoint_v222_impl"
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as name:
            path=Path(name)
            self.assertEqual("invalid",runner.inspect_model_cache(path)["model_cache_status"])
            runner.dump(path/"config.json",{"model_type":"qwen2","_commit_hash":"test-revision"})
            runner.dump(path/"tokenizer_config.json",{})
            runner.dump(path/"tokenizer.json",{})
            runner.dump(path/"model.safetensors.index.json",{"weight_map":{"x":"model-1.safetensors"}})
            self.assertEqual("invalid",runner.inspect_model_cache(path)["model_cache_status"])
            (path/"model-1.safetensors").write_bytes(b"fixture only")
            info=runner.inspect_model_cache(path)
            self.assertEqual("hit",info["model_cache_status"])
            self.assertEqual("test-revision",info["model_revision"])
            self.assertFalse(info["download_performed"])

    def test_layout_dry_run_does_not_launch_experiment_or_write(self):
        with mock.patch.object(runner.subprocess,"run",side_effect=AssertionError("no subprocess")), \
             mock.patch.object(runner,"dump",side_effect=AssertionError("no writes")), contextlib.redirect_stdout(io.StringIO()) as output:
            code=runner.main(["dry-run","--project",str(ROOT),"--runtime","unused","--result-root","unused",
                              "--model-path",str(ROOT/"outputs/checkpoint_v222_impl/unused_model")])
        self.assertEqual(0,code)
        plan=json.loads(output.getvalue())
        self.assertFalse(plan["real_model_loaded"])
        self.assertEqual(["cp_on"],plan["groups"])
        self.assertEqual({"on"},set(plan["configs"]))

    def test_launcher_forwards_paths_modes_and_exit_code(self):
        fake=mock.Mock(return_value=types.SimpleNamespace(returncode=7))
        self.assertEqual(7,launcher.main(["--dry-run","--model-path","path with spaces/model"],run=fake))
        command=fake.call_args.args[0]
        self.assertIn("dry-run",command); self.assertIn("path with spaces/model",command)
        self.assertEqual(str(ROOT),fake.call_args.kwargs["cwd"])

    def test_help_does_not_launch(self):
        fake=mock.Mock()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            launcher.main(["--help"],run=fake)
        self.assertEqual(0,error.exception.code); fake.assert_not_called()

    def test_pair_config_drift_rejected(self):
        configs={label:runner.load_config(ROOT/path) for label,path in runner.CONFIGS.items()}
        configs["on"][runner.PREFIX+"checkpoint_deviation_tau"]=.1
        with self.assertRaises(ValueError):
            runner.validate_configs(configs)

    def test_missing_config_and_include_cycle(self):
        temp_root=ROOT/"outputs/checkpoint_v222_impl"
        temp_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as name:
            path=Path(name)/"config.json"
            with self.assertRaises(FileNotFoundError): runner.load_config(path)
            runner.dump(path,{"include_configs":["config.json"]})
            with self.assertRaises(ValueError): runner.load_config(path)

    def test_mock_bundle_snapshot_modes_and_failure_stop(self):
        temp_root=ROOT/"outputs/checkpoint_v222_impl"
        temp_root.mkdir(parents=True,exist_ok=True)
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory(dir=temp_root) as name:
                project=Path(name)
                (project/"outputs").mkdir()
                plan=runner.preflight(ROOT,model_path=project/"unused_model")
                plan["pending_server_checks"]=[]
                calls=[]
                def fake_run(command,**kwargs):
                    if command[0]=="nvidia-smi":
                        return types.SimpleNamespace(stdout="" if "--query-compute-apps=pid" in command else "0, 0",returncode=0)
                    if command[0]=="git":return types.SimpleNamespace(stdout="test-revision",returncode=0)
                    config=runner.load_config(command[-1]); calls.append(config)
                    if fail:return types.SimpleNamespace(returncode=9)
                    snapshots=Path(config["suffix_v222_snapshot_dir"])
                    snapshots.mkdir(parents=True)
                    (snapshots/"fixture.pt").write_bytes(b"mock retained snapshot")
                    enabled=config[runner.PREFIX+"checkpoint_enabled"]
                    result=dict(formal_gt_blind=True,gt_accessed=False,pair_id="checkpoint_smoke:0",
                                stage1_snapshot_sha256="identical",second_stage_seconds=1.,
                                checkpoint=dict(complete_segment_count=1,events=[checkpoint_event()] if enabled else []))
                    result["reoptimization"]={"checkpoint":result["checkpoint"]}
                    record=dict(pair_id="checkpoint_smoke:0",dataset={"name":"checkpoint_smoke"},accuracy=.5,
                                selected_advanced_method=runner.METHOD,selected_candidate_reranking_method="none",
                                suffix_reoptimization_v2_2_2_result=result)
                    out=project/"results/invert_timestamp_runs"/runner.METHOD/str(len(calls))
                    out.mkdir(parents=True)
                    runner.dump(out/"resolved_config.json",dict(advanced_method={"name":runner.METHOD},advanced_methods={
                        "suffix_reoptimization_v2_2_2":dict(runner.FROZEN,checkpoint_enabled=enabled)}))
                    (out/"experiment.log").write_text("fixed summary",encoding="utf-8")
                    (out/"reconstructions.jsonl").write_text(json.dumps(record)+"\n",encoding="utf-8")
                    return types.SimpleNamespace(returncode=0)
                with mock.patch.object(runner,"preflight",return_value=plan):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            runner.run_bundle(project,project/"runtime",project/"bundle","python",smoke=True,run=fake_run)
                        self.assertEqual(1,len(calls))
                    else:
                        bundle=runner.run_bundle(project,project/"runtime",project/"bundle","python",smoke=True,run=fake_run)
                        self.assertEqual(["write"],[c["suffix_v222_snapshot_mode"] for c in calls])
                        self.assertTrue(calls[0][runner.PREFIX+"checkpoint_enabled"])
                        self.assertTrue((bundle/"snapshots/fixture.pt").is_file())
                        self.assertTrue((bundle/"cp_on_summary.json").is_file())
                        self.assertFalse((bundle/"paired_summary.json").exists())
                        self.assertEqual("complete",json.loads((bundle/"manifest.json").read_text())["status"])
                self.assertEqual([],list((project/"outputs").iterdir()))


if __name__ == "__main__":
    unittest.main()
