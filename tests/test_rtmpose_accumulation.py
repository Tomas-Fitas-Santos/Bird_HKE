"""Dependency-free checks of RTMPose-M BirdGaze accumulation boundaries."""
import math
import unittest

from external_models.rtmpose.accumulation import accumulation_group


class TestRTMPoseEpochAccumulation(unittest.TestCase):
    def test_full_dataset_exactly_once_per_epoch(self):
        images = 77740
        micro = 8
        accum = 8
        iters = math.ceil(images / micro)
        groups = [
            accumulation_group(images, micro, accum, idx)[0]
            for idx in range(iters)
            if accumulation_group(images, micro, accum, idx)[1]
        ]
        self.assertEqual(len(groups), 1215)
        self.assertEqual(groups[:-1], [64] * 1214)
        self.assertEqual(groups[-1], 44)
        self.assertEqual(sum(groups), images)

    def test_partial_microbatch_and_partial_accumulation(self):
        self.assertEqual(accumulation_group(70, 8, 8, 7), (64, True))
        self.assertEqual(accumulation_group(70, 8, 8, 8), (6, True))
        self.assertEqual(accumulation_group(70, 8, 8, 6), (64, False))

    def test_even_groups(self):
        self.assertEqual(accumulation_group(128, 8, 8, 7), (64, True))
        self.assertEqual(accumulation_group(128, 8, 8, 15), (64, True))

    def test_invalid_batch_plan(self):
        with self.assertRaises(ValueError):
            accumulation_group(0, 8, 8, 0)
        with self.assertRaises(ValueError):
            accumulation_group(64, 8, 8, 8)


if __name__ == "__main__":
    unittest.main()
