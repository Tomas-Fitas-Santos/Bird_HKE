"""Standalone synthetic tests for exact Nightjar UUID -> BirdGaze conversion.

Run: python -m unittest discover -s Bird_HKE/tools -p "test_build_annotatiathon_birdgaze.py" -v
"""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

SCRIPT = Path(__file__).with_name("build_annotatiathon_birdgaze.py")
EBIRD_UUID = "00003100-347e-48f2-a5c3-c0e9a09a51ff"
NABIRDS_UUID = "000255d3-5719-4636-8839-6803cc9e359f"
HEADERS = ["asset_id", "native_project_id", "relative_path", "kind", "size_bytes",
           "sha256", "frame_width", "frame_height", "video_group", "frame_index"]


def annotation(*, width=80, height=60, all_occluded=False, duplicate=False) -> dict:
    objects = [{
        "classTitle": "bird", "geometryType": "rectangle",
        "points": {"exterior": [[4, 2], [70, 50]]},
    }]
    pts = {"crown": [30, 12], "left_eye": [35, 20],
           "right_eye": [44, 21], "beak": [55, 23]}
    for name, xy in pts.items():
        title = name + "_occ" if all_occluded or name == "left_eye" else name
        objects.append({
            "classTitle": title, "geometryType": "point",
            "points": {"exterior": [xy]},
        })
    if duplicate:
        objects.append(objects[1])
    return {"size": {"width": width, "height": height}, "objects": objects}


class ConverterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ebird_ann = self.root / "EAnn" / "ann"
        self.nabirds_ann = self.root / "NAnn" / "ann"
        self.ebird_images = self.root / "eBird source"
        self.nabirds_images = self.root / "NABirds source"
        self.out = self.root / "Result"
        self.map = self.root / "Nightjar_media_asset_map.csv"
        for p in (self.ebird_ann, self.nabirds_ann, self.ebird_images, self.nabirds_images):
            p.mkdir(parents=True)
        self.rows = []

    def add_one(self, source="eBird", uid=None, *, all_occluded=False, duplicate=False,
                width=80, record_project=None, remove_original=False):
        is_ebird = source == "eBird"
        uid = uid or (EBIRD_UUID if is_ebird else NABIRDS_UUID)
        relative = ("Sterna/200836421/200836421_000043.jpg" if is_ebird
                    else "0813/0813-2201.jpg")
        original = (self.ebird_images if is_ebird else self.nabirds_images) / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (180, 90, 20)).save(original)
        img_bytes = original.read_bytes()
        annroot = self.ebird_ann if is_ebird else self.nabirds_ann
        (annroot / (uid + ".jpg.json")).write_text(
            json.dumps(annotation(width=width, all_occluded=all_occluded, duplicate=duplicate)),
            encoding="utf-8",
        )
        self.rows.append({
            "asset_id": uid,
            "native_project_id": str(record_project or (386759 if is_ebird else 386817)),
            "relative_path": ("videos/" if is_ebird else "images/") + relative,
            "kind": "video_frame" if is_ebird else "image",
            "size_bytes": len(img_bytes),
            "sha256": hashlib.sha256(img_bytes).hexdigest(),
            "frame_width": 80, "frame_height": 60,
            "video_group": "Sterna/200836421" if is_ebird else "",
            "frame_index": 43 if is_ebird else "",
        })
        if remove_original:
            original.unlink()

    def call(self, *args):
        with self.map.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(self.rows)
        return subprocess.run([
            sys.executable, str(SCRIPT),
            "--source-mode", "nightjar-map",
            "--ebird-ann", str(self.ebird_ann),
            "--nabirds-ann", str(self.nabirds_ann),
            "--ebird-images", str(self.ebird_images),
            "--nabirds-images", str(self.nabirds_images),
            "--output", str(self.out),
            "--media-map", str(self.map), *args
        ], text=True, capture_output=True)

    def test_both_sources_exact_mapping(self):
        self.add_one("eBird")
        self.add_one("NABirds")
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        e = json.loads((self.out / "annot/ebird.json").read_text(encoding="utf-8"))
        n = json.loads((self.out / "annot/nabirds.json").read_text(encoding="utf-8"))
        self.assertEqual(e[0]["image"], "eBird/Sterna/200836421/200836421_000043.jpg")
        self.assertEqual(n[0]["image"], "NABirds/0813/0813-2201.jpg")
        self.assertEqual(e[0]["joints_vis"], [1, 0, 1, 1])
        self.assertEqual(e[0]["joints"][1], [35.0, 20.0])
        self.assertEqual(e[0]["bbox"], [4.0, 2.0, 70.0, 50.0])
        self.assertTrue((self.out / "images" / e[0]["image"]).is_file())
        self.assertEqual({p.name for p in (self.out / "annot").glob("*.json")},
                         {"ebird.json", "nabirds.json"})
        with (self.out / "reports/manifest.csv").open(newline="", encoding="utf-8-sig") as f:
            records = list(csv.DictReader(f))
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["match_method"] == "nightjar_asset_relative_path" for r in records))

    def test_all_occluded_preserved(self):
        self.add_one(all_occluded=True)
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        e = json.loads((self.out / "annot/ebird.json").read_text(encoding="utf-8"))
        self.assertEqual(e[0]["joints_vis"], [0, 0, 0, 0])
        self.assertEqual(e[0]["joints"][0], [30.0, 12.0])

    def test_legacy_nabirds_indexed_as_video_frame(self):
        self.add_one("NABirds")
        self.rows[0]["kind"] = "video_frame"
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = json.loads((self.out / "annot/nabirds.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1)

    def test_nabirds_sha256_restores_image_after_relocation(self):
        self.add_one("NABirds")
        old = self.nabirds_images / "0813/0813-2201.jpg"
        relocated = self.nabirds_images / "original/NABirds/subfolder/newpath.jpg"
        relocated.parent.mkdir(parents=True, exist_ok=True)
        old.rename(relocated)
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = json.loads((self.out / "annot/nabirds.json").read_text(encoding="utf-8"))
        self.assertEqual(records[0]["image"], "NABirds/original/NABirds/subfolder/newpath.jpg")
        with (self.out / "reports/manifest.csv").open(newline="", encoding="utf-8-sig") as f:
            record = next(csv.DictReader(f))
        self.assertEqual(record["match_method"], "nightjar_source_sha256")

    def test_sha256_fallback_rejects_non_unique_media(self):
        self.add_one("NABirds")
        original = self.nabirds_images / "0813/0813-2201.jpg"
        relocated = self.nabirds_images / "two/duplicate.jpg"
        relocated.parent.mkdir(parents=True)
        relocated.write_bytes(original.read_bytes())
        moved = self.nabirds_images / "one/relocated.jpg"
        moved.parent.mkdir(parents=True)
        original.rename(moved)
        result = self.call("--audit-only")
        self.assertEqual(result.returncode, 2, result.stderr)
        with (self.out / "reports/rejected_annotations.csv").open(newline="", encoding="utf-8-sig") as f:
            record = next(csv.DictReader(f))
        self.assertIn("unique original", record["reason"])

    def test_audit_only(self):
        self.add_one()
        result = self.call("--audit-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.out / "images").exists())
        self.assertFalse((self.out / "annot").exists())
        self.assertTrue((self.out / "reports/validation_report.json").is_file())

    def test_missing_uuid_in_database_rejected(self):
        self.add_one()
        self.rows[0]["asset_id"] = NABIRDS_UUID
        result = self.call("--audit-only")
        self.assertEqual(result.returncode, 2)
        self.assertIn("0 matched", result.stdout)
        self.assertFalse((self.out / "annot").exists())

    def test_wrong_project_rejected(self):
        self.add_one(record_project=386817)
        result = self.call("--audit-only")
        self.assertEqual(result.returncode, 2)

    def test_missing_original_rejected(self):
        self.add_one(remove_original=True)
        result = self.call()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.out / "annot").exists())

    def test_image_dimension_mismatch_rejected(self):
        self.add_one(width=100)
        result = self.call("--audit-only")
        self.assertEqual(result.returncode, 2)

    def test_duplicate_keypoint_rejected(self):
        self.add_one(duplicate=True)
        result = self.call("--audit-only")
        self.assertEqual(result.returncode, 2)

    def test_source_sha_verification(self):
        self.add_one()
        result = self.call("--verify-sha256")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrong_source_sha_verification_rejected(self):
        self.add_one()
        self.rows[0]["sha256"] = "0" * 64
        result = self.call("--verify-sha256", "--audit-only")
        self.assertEqual(result.returncode, 2)

    def test_resume_explicit(self):
        self.add_one()
        self.assertEqual(self.call().returncode, 0)
        self.assertEqual(self.call().returncode, 1)
        self.assertEqual(self.call("--skip-sizes-if-existing").returncode, 0)

    def test_partial_build_requires_explicit_opt_in(self):
        self.add_one()
        self.add_one("NABirds", remove_original=True)
        self.assertEqual(self.call().returncode, 2)
        self.assertEqual(self.call("--allow-partial").returncode, 0)
        e = json.loads((self.out / "annot/ebird.json").read_text(encoding="utf-8"))
        n = json.loads((self.out / "annot/nabirds.json").read_text(encoding="utf-8"))
        self.assertEqual(len(e), 1)
        self.assertEqual(len(n), 0)


if __name__ == "__main__":
    unittest.main()
