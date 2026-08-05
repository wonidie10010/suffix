import json
import os
import tempfile
import unittest

from dataset import load_dataset_samples
from invert import load_config


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_CONFIG = os.path.join(
    ROOT, "experiment_configs", "l24_airport_medical.json"
)


class MultiDatasetExperimentTests(unittest.TestCase):
    def test_multi_dataset_loading_preserves_order_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = os.path.join(directory, "first.json")
            second_path = os.path.join(directory, "second.json")
            with open(first_path, "w", encoding="utf-8") as f:
                json.dump(["a0", "a1"], f)
            with open(second_path, "w", encoding="utf-8") as f:
                json.dump(["b0", "b1"], f)

            specs = [
                {"name": "first", "path": first_path, "type": "local", "len": 2},
                {"name": "second", "path": second_path, "type": "local", "len": 2},
            ]
            samples, parameters = load_dataset_samples(dataset_specs=specs)

        self.assertEqual(specs, parameters)
        self.assertEqual(["a0", "a1", "b0", "b1"], [item["text"] for item in samples])
        self.assertEqual(
            ["first", "first", "second", "second"],
            [item["dataset"]["name"] for item in samples],
        )
        self.assertEqual([1, 2, 1, 2], [item["dataset"]["sample_number"] for item in samples])
        self.assertTrue(all(item["dataset"]["sample_count"] == 2 for item in samples))

    def test_single_dataset_arguments_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "single.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["only"], f)
            samples, parameters = load_dataset_samples(
                dataset_path=path,
                dataset_type="local",
                dataset_len=5,
            )

        self.assertEqual(1, len(samples))
        self.assertEqual("single", parameters[0]["name"])
        self.assertEqual(5, parameters[0]["len"])

    def test_multi_dataset_validation_rejects_short_or_duplicate_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(["one"], f)
            with self.assertRaisesRegex(ValueError, "requested 2 samples"):
                load_dataset_samples(dataset_specs=[{
                    "name": "short",
                    "path": path,
                    "type": "local",
                    "len": 2,
                }])
            with self.assertRaisesRegex(ValueError, "duplicate dataset name"):
                load_dataset_samples(dataset_specs=[
                    {"name": "same", "path": path, "type": "local", "len": 1},
                    {"name": "same", "path": path, "type": "local", "len": 1},
                ])

    def test_formal_config_selects_two_l24_datasets(self):
        config = load_config(EXPERIMENT_CONFIG)

        self.assertEqual(24, config["num_invert_layers"])
        self.assertEqual(1000, config["epoch"])
        self.assertEqual(0, config["seed"])
        self.assertEqual("v1.2.1", config["suffix_version"])
        self.assertEqual("v1.1", config["cgmr_version"])
        self.assertEqual(["airport", "medical"], [item["name"] for item in config["datasets"]])
        self.assertEqual([5, 5], [item["len"] for item in config["datasets"]])

    def test_main_experiment_path_does_not_write_excel(self):
        with open(os.path.join(ROOT, "invert.py"), "r", encoding="utf-8") as f:
            source = f.read()

        self.assertNotIn("write_summary_excel(", source)


if __name__ == "__main__":
    unittest.main()
