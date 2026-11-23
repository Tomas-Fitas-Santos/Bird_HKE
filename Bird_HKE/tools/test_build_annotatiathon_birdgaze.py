"""Smoke tests: run with python -m unittest discover -s Bird_HKE/tools -p 'test_build_annotatiathon_birdgaze.py'."""
from __future__ import annotations
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

SCRIPT = Path(__file__).with_name("build_annotatiathon_birdgaze.py")


def make_ann(path: Path, width=80, height=60, *, duplicate=False):
    labels = [
        {"classTitle": "bird", "geometryType": "rectangle", "points": {"exterior": [[4, 2], [70, 50]]}},
        {"classTitle": "crown", "geometryType": "point", "points": {"exterior": [[30, 12]]}},
        {"classTitle": "left_eye_occ", "geometryType": "point", "points": {"exterior": [[35, 20]]}},
        {"classTitle": "right_eye", "geometryType": "point", "points": {"exterior": [[44, 21]]}},
        {"classTitle": "beak", "geometryType": "point", "points": {"exterior": [[55, 23]]}},
    ]
    if duplicate:
        labels.append(labels[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"size": {"width": width, "height": height}, "objects": labels}), encoding="utf-8")


class TestConversion(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.eann = self.root / "annotation eBird" / "ann"
        self.nann = self.root / "annotation NABirds" / "ann"
        self.eimgs = self.root / "images eBird"
        self.nimgs = self.root / "images NABirds"
        self.out = self.root / "BirdGaze output"
        for d in (self.eann, self.nann, self.eimgs, self.nimgs):
            d.mkdir(parents=True)

    def add_frame(self, folder: str, name: str = "f_1.jpg", ann_prefix=""):
        p = self.eimgs / folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (250, 250, 250)).save(p)
        make_ann(self.eann / ann_prefix / (name + ".json"))
        return p

    def add_nabirds(self, name: str):
        p = self.nimgs / name
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60)).save(p)
        make_ann(self.nann / (Path(name).name + ".json"))

    def run_build(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), "--ebird-ann", str(self.eann),
                               "--nabirds-ann", str(self.nann), "--ebird-images", str(self.eimgs),
                               "--nabirds-images", str(self.nimgs), "--output", str(self.out), *extra],
                              text=True, capture_output=True)

    def test_build_keeps_coords_and_paths(self):
        self.add_frame("Sterna/001", ann_prefix="Sterna/001")
        self.add_nabirds("015/xyz.jpg")
        p = self.run_build()
        self.assertEqual(p.returncode, 0, p.stderr)
        ebird = json.loads((self.out / "annot/ebird.json").read_text(encoding="utf-8"))
        nb = json.loads((self.out / "annot/nabirds.json").read_text(encoding="utf-8"))
        self.assertEqual(len(ebird), 1)
        self.assertEqual(len(nb), 1)
        self.assertEqual(ebird[0]["image"], "eBird/Sterna/001/f_1.jpg")
        self.assertEqual(ebird[0]["joints_vis"], [1, 0, 1, 1])
        self.assertEqual(ebird[0]["joints"], [[30.0, 12.0], [35.0, 20.0], [44.0, 21.0], [55.0, 23.0]])
        self.assertEqual(ebird[0]["scale"], 66 / 200)
        self.assertTrue((self.out / "images" / ebird[0]["image"]).is_file())
        # Only two annotation JSON files are produced. The occluded point is
        # preserved in joints and excluded from supervision by joints_vis=0.
        self.assertEqual({p.name for p in (self.out / "annot").glob("*.json")},
                         {"ebird.json", "nabirds.json"})
        self.assertFalse((self.out / "reports/supplemental_keypoints.jsonl").exists())

    def test_occluded_only_is_kept_in_source_json(self):
        self.add_frame("Sterna/001", ann_prefix="Sterna/001")
        ap = self.eann / "Sterna/001/f_1.jpg.json"
        raw = json.loads(ap.read_text(encoding="utf-8"))
        for obj in raw["objects"]:
            if obj.get("geometryType") == "point":
                obj["classTitle"] = obj["classTitle"].replace("_occ", "") + "_occ"
        ap.write_text(json.dumps(raw), encoding="utf-8")
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = json.loads((self.out / "annot/ebird.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["joints_vis"], [0, 0, 0, 0])
        self.assertEqual(records[0]["joints"][0], [30.0, 12.0])

    def test_audit_only_does_not_copy_media(self):
        self.add_frame("Sterna/001", ann_prefix="Sterna/001")
        p = self.run_build("--audit-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((self.out / "images").exists())
        self.assertTrue((self.out / "reports/manifest.csv").exists())

    def test_ambiguous_name_blocks_build(self):
        self.add_frame("A/111")
        p = self.eimgs / "B/222/f_1.jpg"
        p.parent.mkdir(parents=True)
        Image.new("RGB", (80, 60)).save(p)
        result = self.run_build()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.out / "images").exists())
        with (self.out / "reports/rejected_annotations.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn("ambiguous", rows[0]["reason"])

    def test_dimension_mismatch_blocks_build(self):
        self.add_frame("Sterna/001", ann_prefix="Sterna/001")
        make_ann(self.eann / "Sterna/001/f_1.jpg.json", width=40)
        result = self.run_build()
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.out / "images").exists())

    def test_duplicate_keypoint_blocks_build(self):
        self.add_frame("Sterna/001", ann_prefix="Sterna/001")
        make_ann(self.eann / "Sterna/001/f_1.jpg.json", duplicate=True)
        result = self.run_build()
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.out / "images").exists())

    def test_second_run_requires_explicit_resume(self):
        self.add_frame("Sterna/001", ann_prefix="Sterna/001")
        self.assertEqual(self.run_build().returncode, 0)
        self.assertEqual(self.run_build().returncode, 1)
        self.assertEqual(self.run_build("--skip-sizes-if-existing").returncode, 0)


if __name__ == "__main__":
    unittest.main()
