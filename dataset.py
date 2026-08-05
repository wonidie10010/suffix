from datasets import load_dataset
import json
import os
import torch
import numpy as np
import pandas as pd


VALID_DATASET_TYPES = {"local", "datasets", "github"}


class Dataset(object):
    def __init__(self, dataset_name, dataset_type):
        if dataset_type == "local":
            if not os.path.exists(dataset_name):
                raise FileNotFoundError("{} dataset does not exist!!".format(dataset_name))
            if "ECHR" in dataset_name:
                self.data = []
                for root, dirs, files in os.walk(dataset_name):
                    files.sort()
                    for i, file in enumerate(files):
                        with open(os.path.join(dataset_name, file), 'r', encoding="utf-8") as f:
                            d = json.load(f)
                        self.data.append(d["CONCLUSION"])
            else:
                with open(dataset_name, 'r', encoding="utf-8") as f:
                    self.data = json.load(f)
        elif dataset_type == "datasets":
            self.data = []
            data_hug = load_dataset(dataset_name)
            for key in data_hug.keys():
                if 'output' in data_hug[key].features:
                    for item in data_hug[key]['output']:
                        self.data.append(item)
        elif dataset_type == "github":
            if "skytrax-reviews-dataset" in dataset_name:
                if not os.path.exists(dataset_name):
                    raise FileNotFoundError("{} dataset does not exist!!".format(dataset_name))
                self.data = pd.read_csv(dataset_name, encoding="utf-8")['content']
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
                    
    def get_data(self):
        return self.data


def _default_dataset_name(path):
    filename = os.path.basename(os.path.normpath(path))
    name, _ = os.path.splitext(filename)
    return name or "dataset"


def load_dataset_samples(dataset_specs=None, dataset_path=None,
                         dataset_type=None, dataset_len=None):
    """Load one or more datasets and attach stable per-sample provenance."""
    is_multi_dataset = dataset_specs is not None
    if is_multi_dataset:
        if not isinstance(dataset_specs, list) or not dataset_specs:
            raise ValueError("datasets must be a non-empty list")
        specs = dataset_specs
    else:
        specs = [{
            "name": _default_dataset_name(dataset_path),
            "path": dataset_path,
            "type": dataset_type,
            "len": dataset_len,
        }]

    samples = []
    parameters = []
    seen_names = set()
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError("datasets[{}] must be an object".format(index))
        name = str(spec.get("name") or "").strip()
        path = spec.get("path")
        current_type = spec.get("type")
        requested_len = spec.get("len")
        if not name:
            raise ValueError("datasets[{}].name is required".format(index))
        if name in seen_names:
            raise ValueError("duplicate dataset name: {}".format(name))
        if not isinstance(path, str) or not path.strip():
            raise ValueError("datasets[{}].path is required".format(index))
        if current_type not in VALID_DATASET_TYPES:
            raise ValueError(
                "datasets[{}].type must be one of: datasets, github, local".format(index)
            )
        if isinstance(requested_len, bool) or not isinstance(requested_len, int) or requested_len <= 0:
            raise ValueError("datasets[{}].len must be a positive integer".format(index))

        loaded = Dataset(dataset_name=path, dataset_type=current_type)
        selected = list(loaded.get_data())[:requested_len]
        if is_multi_dataset and len(selected) != requested_len:
            raise ValueError(
                "dataset {} requested {} samples but only {} are available".format(
                    name, requested_len, len(selected)
                )
            )
        parameters.append({
            "name": name,
            "path": path,
            "type": current_type,
            "len": requested_len,
        })
        dataset_sample_count = len(selected)
        for sample_index, text in enumerate(selected):
            samples.append({
                "text": text,
                "dataset": {
                    "name": name,
                    "path": path,
                    "type": current_type,
                    "sample_index": sample_index,
                    "sample_number": sample_index + 1,
                    "sample_count": dataset_sample_count,
                },
            })
        seen_names.add(name)

    return samples, parameters
