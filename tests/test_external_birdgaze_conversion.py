"""Small fixture tests for the BirdGaze-to-COCO source adapter (no MMPose required)."""
import json
import tempfile
import unittest
from pathlib import Path

from external_models.convert_birdgaze_to_coco import SCENARIOS, convert_record, convert_split


class TestBirdGazeCocoConversion(unittest.TestCase):
    def test_dataset_folders_match_titan(self):
        self.assertEqual(SCENARIOS["FD"], "BirdGaze_Full_Dataset")
        self.assertEqual(SCENARIOS["OS"], "birdgaze_original_subset")
        self.assertEqual(SCENARIOS["CS"], "birdgaze_corrected_subset")

    def test_zero_available_landmarks_are_preserved_without_supervision(self):
        with tempfile.TemporaryDirectory() as folder:
            images = Path(folder)
            (images / "test.png").write_bytes(b"fixture")
            sample = {
                "image": "test.png", "center": [20, 20], "scale": 0.3,
                "joints": [[0, 0]] * 4,
                "joints_valid": [0, 0, 0, 0],
                "joints_vis": [0, 0, 0, 0],
            }
            image, ann = convert_record(sample, 1, images, (40, 40))
            self.assertEqual(image["id"], 1)
            self.assertEqual(ann["num_keypoints"], 0)
            self.assertEqual(ann["keypoints"], [0.0, 0.0, 0] * 4)

    def test_negative_scale_is_kept_as_rotation_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            images = Path(folder)
            (images / "negative.png").write_bytes(b"fixture")
            sample = {
                "image": "negative.png", "center": [20, 20], "scale": -0.09375,
                "joints": [[10, 15], [17, 20], [22, 20], [29, 25]],
            }
            image, ann = convert_record(sample, 1, images, (40, 40))
            self.assertEqual(image["id"], 1)
            self.assertTrue(ann["birdgaze_negative_scale"])
            self.assertEqual(ann["bbox"][2:], [18.75, 18.75])
            self.assertGreater(ann["area"], 0)

    def test_zero_scale_export_retains_original_case(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "zero.png").write_bytes(b"fixture")
            sample = {
                "image": "zero.png", "center": [20, 20], "scale": 0.0,
                "joints": [[15, 8], [12, 12], [-1, -1], [30, 20]],
            }
            _, ann = convert_record(sample, 1, root, (64, 32))
            self.assertTrue(ann["birdgaze_zero_scale"])
            self.assertFalse(ann["birdgaze_negative_scale"])
            self.assertEqual(ann["bbox"][2:], [1.0, 1.0])

    def test_corrected_bbox_is_retained_and_changes_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "crop.png").write_bytes(b"fixture")
            record = {
                "image": "crop.png", "center": [20, 16], "scale": 0.2,
                "joints": [[15, 8], [12, 12], [23, 12], [30, 20]],
            }
            _, original = convert_record(record, 1, root, (64, 32))
            _, corrected = convert_record({**record, "bbox": [2.1, 1.4, 60.8, 28.2]}, 1, root, (64, 32))
            self.assertIsNone(original["birdgaze_source_bbox"])
            self.assertEqual(corrected["birdgaze_source_bbox"], [2.1, 1.4, 60.8, 28.2])
            self.assertNotEqual(original, corrected)

    def test_source_crop_matches_original_floor_ceil_and_offset(self):
        import numpy as np
        from external_models.rtmpose.geometry import apply_source_crop
        full = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        rec = {
            "img": full.copy(), "img_shape": (6, 8),
            "birdgaze_source_bbox": [2.2, 1.1, 6.1, 5.1],
            "keypoints": np.array([[[4., 3.], [-1., -1.]]], dtype=np.float32),
            "bbox": np.array([[0., 0., 10., 10.]], dtype=np.float32),
        }
        result = apply_source_crop(rec)
        np.testing.assert_array_equal(result["img"], full[1:6, 2:7])
        np.testing.assert_array_equal(result["birdgaze_crop_offset"], [2., 1.])
        np.testing.assert_array_equal(result["keypoints"][0], [[2., 2.], [-3., -2.]])
        np.testing.assert_array_equal(result["bbox"][0], [-2., -1., 8., 9.])

    def test_invalid_source_crop_is_ignored_as_in_bird_hke(self):
        import numpy as np
        from external_models.rtmpose.geometry import apply_source_crop
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        r = {"img": img, "birdgaze_source_bbox": [4., 4., 3., 3.]}
        self.assertIs(apply_source_crop(r)["img"], img)

    def test_preserves_keypoint_order_visibility_and_unclipped_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "images" / "test.png").write_bytes(b"fixture")
            record = {
                "image": "test.png", "center": [5, 10], "scale": 0.2,
                "joints": [[2, 3], [6, 5], [0, 0], [10, 15]],
                "joints_vis": [1, 0, 0, 1],
                "joints_valid": [1, 1, 0, 1],
            }
            img, ann = convert_record(record, 1, root / "images", (40, 40))
            self.assertEqual(img["file_name"], "test.png")
            self.assertEqual(ann["bbox"], [-15.0, -10.0, 40.0, 40.0])
            self.assertEqual(ann["num_keypoints"], 3)
            self.assertEqual([ann["keypoints"][i] for i in (2, 5, 8, 11)], [2, 1, 0, 2])

    def test_converts_splits_without_modifying_source(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "annot").mkdir()
            Image.new("RGB", (64, 32)).save(root / "images" / "test.png")
            record = {
                "image": "test.png", "center": [20, 16], "scale": 0.2,
                "joints": [[15, 8], [12, 12], [23, 12], [30, 20]],
                "joints_vis": [1, 1, 1, 1],
            }
            src = root / "annot" / "train.json"
            src.write_text(json.dumps([record]), encoding="utf-8")
            original_bytes = src.read_bytes()
            result = convert_split(src, root / "images", root / "coco.json")
            output = json.loads((root / "coco.json").read_text())
            self.assertEqual(result["images"], 1)
            self.assertEqual(result["samples_without_valid_keypoints"], 0)
            self.assertEqual(output["images"][0]["width"], 64)
            self.assertEqual(output["images"][0]["height"], 32)
            self.assertEqual(output["annotations"][0]["num_keypoints"], 4)
            self.assertEqual(src.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
