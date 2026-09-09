import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1] / 'Bird_HKE'
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dataset.birdgaze import coordinate_validity
    from dataset.birdgaze import infer_data_source
    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    'full dataset dependencies are not installed',
)
class DatasetVisibilityTests(unittest.TestCase):
    def test_source_inference_works_before_and_after_source_grouping(self):
        cases = {
            'FINETUNE/species/image.jpg': 'eBird',
            'eBird/FINETUNE/species/image.jpg': 'eBird',
            '123/species/image.jpg': 'NABirds',
            'NABirds/123/species/image.jpg': 'NABirds',
            'QWERTY/image.jpg': 'Animal Kingdom',
            'Animal_Kingdom/QWERTY/image.jpg': 'Animal Kingdom',
            'Cardinal/image.jpg': 'BirdSnap',
            'BirdSnap/Cardinal/image.jpg': 'BirdSnap',
        }
        for path, expected in cases.items():
            self.assertEqual(infer_data_source(path), expected)

    def test_nonzero_occluded_coordinate_remains_valid(self):
        joints = np.array([[12, 20], [0, 0], [-1, -1], [8, 9]], dtype=np.float32)
        visibility = np.array([0, 0, 0, 1], dtype=np.float32)
        self.assertEqual(
            coordinate_validity(joints, visibility=visibility).tolist(),
            [True, False, False, True],
        )

    def test_explicit_coordinate_mask_is_authoritative(self):
        joints = np.array([[12, 20], [5, 5], [2, 3], [8, 9]], dtype=np.float32)
        explicit = np.array([1, 0, 1, 0], dtype=np.float32)
        self.assertEqual(
            coordinate_validity(joints, explicit_valid=explicit).tolist(),
            [True, False, True, False],
        )


if __name__ == '__main__':
    unittest.main()
