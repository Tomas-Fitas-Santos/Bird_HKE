"""Stdlib-only tests; never import PyTorch, MMPose or existing model code."""
import json
from pathlib import Path
import tempfile
import unittest

from external_models.remaining_baselines.prepare import (
    BASELINES, SCENARIOS, ensure_under, prepare, reserve,
    validate_exports, validate_os_cs_distinction,
)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self, scenario="FD", *, altered=False):
        ds = self.root / "BirdGaze_v2" / SCENARIOS[scenario] / "annot"
        out = self.root / "external_models/generated" / scenario
        ds.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val"):
            source = [{"image": "a.jpg", "bbox": [1, 2, 3, 4] if altered else [0, 0, 3, 4]}]
            converted = {
                "images": [{"id": 1}],
                "annotations": [{"keypoints": [1, 2, 2] * 4,
                                 "birdgaze_source_bbox": source[0]["bbox"],
                                 "birdgaze_zero_scale": False,
                                 "birdgaze_negative_scale": False}],
            }
            (ds / f"{split}.json").write_text(json.dumps(source))
            (out / f"{split}_coco.json").write_text(json.dumps(converted))
        return ds.parent, out

    def test_two_models_have_distinct_run_roots(self):
        self.data()
        a, m1 = prepare("vitposepp-b", "FD", self.root, check_upstream=False)
        b, m2 = prepare("posebh-b", "FD", self.root, check_upstream=False)
        self.assertNotEqual(a, b)
        self.assertFalse(a.exists())
        self.assertFalse(b.exists())
        self.assertFalse(m1["training_ready"])
        self.assertFalse(m2["training_ready"])
        self.assertEqual(m1["annotations"]["train"]["count"], 1)

    def test_no_adapter_no_reservation(self):
        self.data()
        run, manifest = prepare("posebh-b", "FD", self.root, check_upstream=False)
        with self.assertRaisesRegex(ValueError, "not implemented"):
            reserve(run, manifest, self.root)
        self.assertFalse(run.exists())

    def test_refuses_overwriting_even_with_adapters(self):
        self.data()
        run, manifest = prepare("vitposepp-b", "FD", self.root, check_upstream=False)
        manifest["training_ready"] = True
        recipe = reserve(run, manifest, self.root)
        before = recipe.read_bytes()
        with self.assertRaises(FileExistsError):
            reserve(run, manifest, self.root)
        self.assertEqual(recipe.read_bytes(), before)

    def test_reject_path_escape(self):
        with self.assertRaises(ValueError):
            ensure_under(self.root / "other", self.root / "runs")
        with self.assertRaises(ValueError):
            ensure_under(self.root / "runs", self.root / "runs")

    def test_reject_lost_metadata(self):
        dataset, export = self.data()
        p = export / "train_coco.json"
        obj = json.loads(p.read_text())
        del obj["annotations"][0]["birdgaze_source_bbox"]
        p.write_text(json.dumps(obj))
        with self.assertRaisesRegex(ValueError, "stale crop metadata"):
            validate_exports(dataset, export)

    def test_os_cs_collision_detection(self):
        self.data("OS", altered=False)
        self.data("CS", altered=True)
        # Emulate the broken exporter dropping the original source bbox.
        os_dir = self.root / "external_models/generated/OS"
        cs_dir = self.root / "external_models/generated/CS"
        for split in ("train", "val"):
            (cs_dir / f"{split}_coco.json").write_bytes(
                (os_dir / f"{split}_coco.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "COCO exports match"):
            validate_os_cs_distinction(self.root)

    def test_registry_contains_only_remaining_models(self):
        self.assertEqual(set(BASELINES), {"vitposepp-b", "posebh-b"})


if __name__ == "__main__":
    unittest.main()
