import os
import unittest

from experiment_outputs import experiment_method_directory_name


class ExperimentOutputLayoutTests(unittest.TestCase):
    def test_baseline_only_uses_frozen_original_directory(self):
        self.assertEqual(
            experiment_method_directory_name(
                "frozen_original_baseline",
                "none",
            ),
            "frozen_original_baseline",
        )

    def test_suffix_only_uses_versioned_suffix_directory(self):
        self.assertEqual(
            experiment_method_directory_name(
                "suffix_reoptimization_v1.2.2",
                "none",
            ),
            "suffix_reoptimization_v1.2.2",
        )

    def test_baseline_with_cgmr_uses_cgmr_directory(self):
        self.assertEqual(
            experiment_method_directory_name(
                "frozen_original_baseline",
                "CGMR_v1.2",
            ),
            "frozen_original_baseline__CGMR_v1.2",
        )

    def test_suffix_with_cgmr_keeps_both_versions(self):
        self.assertEqual(
            experiment_method_directory_name(
                "suffix_reoptimization_v1.2.1",
                "CGMR_v1.1",
            ),
            "suffix_reoptimization_v1.2.1__CGMR_v1.1",
        )

    def test_complete_v123_with_cgmr_keeps_both_methods(self):
        self.assertEqual(
            experiment_method_directory_name(
                "suffix_v1.2.3",
                "CGMR_v1.0",
            ),
            "suffix_v1.2.3__CGMR_v1.0",
        )

    def test_missing_legacy_candidate_is_treated_as_none(self):
        self.assertEqual(
            experiment_method_directory_name("suffix_reoptimization"),
            "suffix_reoptimization",
        )

    def test_run_path_nests_timestamp_under_method_directory(self):
        method_directory = experiment_method_directory_name(
            "suffix_reoptimization_v1.3.1",
            "none",
        )
        run_dir = os.path.join(
            "results",
            "invert_timestamp_runs",
            method_directory,
            "20260729-120000",
        )

        self.assertEqual(
            run_dir,
            os.path.join(
                "results",
                "invert_timestamp_runs",
                "suffix_reoptimization_v1.3.1",
                "20260729-120000",
            ),
        )


if __name__ == "__main__":
    unittest.main()
