"""Regression: MMEngine BaseDataset clears data_list after serialization.

RTMPose preflight must inspect ds.get_data_info(i), not ds.data_list.
This test uses only the standard library; it does not need MMPose or a GPU.
"""
import unittest

from comparison_models.rtmpose.validate_fd import find_sample_with_flag


class SerializedDatasetStub:
    """Same observable API: records remain accessible after data_list clears."""

    def __init__(self):
        self.data_list = []
        self._records = [
            {"id": 1, "birdgaze_negative_scale": False, "birdgaze_zero_scale": False},
            {"id": 2, "birdgaze_negative_scale": True, "birdgaze_zero_scale": False},
            {"id": 3, "birdgaze_negative_scale": False, "birdgaze_zero_scale": True},
        ]

    def __len__(self):
        return len(self._records)

    def get_data_info(self, index):
        return dict(self._records[index])


class SerializedPreflightSelectionTests(unittest.TestCase):
    def test_flags_found_after_mmengine_clears_data_list(self):
        dataset = SerializedDatasetStub()
        self.assertEqual(dataset.data_list, [])
        self.assertEqual(find_sample_with_flag(dataset, "birdgaze_negative_scale"), 1)
        self.assertEqual(find_sample_with_flag(dataset, "birdgaze_zero_scale"), 2)

    def test_scenario_registry_supports_all(self):
        from comparison_models.registry import COMPARISON_MODELS, SCENARIOS
        model = COMPARISON_MODELS.models["rtmpose-m"]
        self.assertEqual(model.scenarios, SCENARIOS)
        self.assertEqual(model.smoke_module, model.train_module)

    def test_cli_dispatches_selected_scenario_through_guarded_launcher(self):
        from unittest.mock import patch
        from pathlib import Path
        from comparison_models import cli
        with patch.object(cli, "interpreter_for", return_value=Path("/fake/python")), \
                patch.object(cli, "_run", return_value=0) as run:
            self.assertEqual(cli.main(["test", "--model", "rtmpose-m", "--scenario", "all"]), 0)
            self.assertEqual([c.args[1][-3:] for c in run.call_args_list],
                             [["--scenario", sc, "--check-only"] for sc in ("FD", "OS", "CS")])

    def test_absent_flag_is_detected(self):
        self.assertIsNone(
            find_sample_with_flag(SerializedDatasetStub(), "not_present")
        )


if __name__ == "__main__":
    unittest.main()
