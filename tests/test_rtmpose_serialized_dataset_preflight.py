"""Regression: MMEngine BaseDataset clears data_list after serialization.

RTMPose preflight must inspect ds.get_data_info(i), not ds.data_list.
This test uses only the standard library; it does not need MMPose or a GPU.
"""
import unittest

from external_models.rtmpose.validate_fd import find_sample_with_flag


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

    def test_absent_flag_is_detected(self):
        self.assertIsNone(
            find_sample_with_flag(SerializedDatasetStub(), "not_present")
        )


if __name__ == "__main__":
    unittest.main()
