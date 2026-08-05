import unittest
import sys
import types


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_nn_stub = types.ModuleType("torch.nn")
    torch_functional_stub = types.ModuleType("torch.nn.functional")
    torch_stub.nn = torch_nn_stub
    torch_nn_stub.functional = torch_functional_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = torch_nn_stub
    sys.modules["torch.nn.functional"] = torch_functional_stub

from suffix_optimization_methods.method_versions.suffix_reoptimization_v1_2 import (
    SuffixReoptimizationV12Config,
    _find_anomalies,
)


class SuffixReoptimizationV12AnomalyTests(unittest.TestCase):
    def adaptive_config(self, **overrides):
        values = {
            "anomaly_detection_mode": "adaptive",
            "adaptive_z_threshold": 1.5,
            "adaptive_drop_z_threshold": 1.5,
            "adaptive_min_std": 1e-6,
            "adaptive_min_points": 4,
        }
        values.update(overrides)
        return SuffixReoptimizationV12Config(**values)

    def test_smooth_sequence_does_not_trigger(self):
        config = self.adaptive_config()
        scores = [0.91, 0.90, 0.92, 0.91, 0.90, 0.92]

        anomalies = _find_anomalies(scores, scores, 0, config)

        self.assertEqual([], anomalies)

    def test_token_forward_adaptive_low_triggers_on_local_valley(self):
        config = self.adaptive_config()
        embedding_scores = [0.95, 0.94, 0.95, 0.94, 0.95, 0.94]
        token_scores = [0.92, 0.91, 0.90, 0.35, 0.91, 0.90]

        anomalies = _find_anomalies(embedding_scores, token_scores, 0, config)

        self.assertTrue(anomalies)
        self.assertEqual(3, anomalies[0]["position"])
        self.assertIn("token_forward_adaptive_low", anomalies[0]["reasons"])
        self.assertEqual("adaptive", anomalies[0]["anomaly_detection_mode"])

    def test_token_forward_drop_triggers_on_sudden_drop(self):
        config = self.adaptive_config()
        embedding_scores = [0.95, 0.94, 0.95, 0.94, 0.95, 0.94, 0.95]
        token_scores = [0.95, 0.94, 0.93, 0.70, 0.69, 0.68, 0.67]

        anomalies = _find_anomalies(embedding_scores, token_scores, 0, config)

        self.assertTrue(anomalies)
        self.assertEqual(3, anomalies[0]["position"])
        self.assertIn("token_forward_drop", anomalies[0]["reasons"])
        self.assertNotIn("token_forward_adaptive_low", anomalies[0]["reasons"])

    def test_scan_pos_skips_earlier_anomaly(self):
        config = self.adaptive_config()
        embedding_scores = [0.95, 0.94, 0.95, 0.94, 0.95, 0.94]
        token_scores = [0.92, 0.91, 0.35, 0.91, 0.90, 0.91]

        anomalies = _find_anomalies(embedding_scores, token_scores, 3, config)

        self.assertEqual([], anomalies)

    def test_threshold_mode_preserves_legacy_reason(self):
        config = SuffixReoptimizationV12Config(
            anomaly_detection_mode="threshold",
            hidden_low_threshold=0.50,
            hidden_drop_threshold=0.30,
            token_forward_low_threshold=0.75,
            min_anomaly_reasons=1,
        )
        embedding_scores = [0.95, 0.94, 0.93, 0.92]
        token_scores = [0.90, 0.88, 0.70, 0.89]

        anomalies = _find_anomalies(embedding_scores, token_scores, 0, config)

        self.assertEqual(1, len(anomalies))
        self.assertEqual(2, anomalies[0]["position"])
        self.assertEqual(["token_forward_low"], anomalies[0]["reasons"])
        self.assertEqual("threshold", anomalies[0]["anomaly_detection_mode"])


if __name__ == "__main__":
    unittest.main()
