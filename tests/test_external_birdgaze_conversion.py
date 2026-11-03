"""Small fixture tests for the BirdGaze-to-COCO source adapter (no MMPose required)."""
import json
import tempfile
import unittest
from pathlib import Path

from external_models.convert_birdgaze_to_coco import convert_record, convert_split


class TestBirdGazeCocoConversion(unittest.TestCase):
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
            self.assertEqual(output["images"][0]["width"], 64)
            self.assertEqual(output["images"][0]["height"], 32)
            self.assertEqual(output["annotations"][0]["num_keypoints"], 4)
            self.assertEqual(src.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
