"""Synthetic tests of full Supervisely export matching and hash validation."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

BUILDER = Path(__file__).with_name("build_annotatiathon_birdgaze.py")
UUID_E = "00003100-347e-48f2-a5c3-c0e9a09a51ff"
UUID_N = "000255d3-5719-4636-8839-6803cc9e359f"


def sample_annotation() -> dict:
    return {
        "size": {"width": 80, "height": 60},
        "objects": [
            {"classTitle": "bird", "geometryType": "rectangle",
             "points": {"exterior": [[4, 3], [76, 56]]}},
            {"classTitle": "crown", "geometryType": "point",
             "points": {"exterior": [[27, 13]]}},
            {"classTitle": "left_eye_occ", "geometryType": "point",
             "points": {"exterior": [[31, 22]]}},
            {"classTitle": "right_eye", "geometryType": "point",
             "points": {"exterior": [[43, 23]]}},
            {"classTitle": "beak", "geometryType": "point",
             "points": {"exterior": [[52, 27]]}},
        ],
    }


class ExportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.export_e = self.base / "eBird full export"
        self.export_n = self.base / "NABirds full export"
        self.old_e_ann = self.base / "Old E annotations"
        self.old_n_ann = self.base / "Old N annotations"
        self.orig_e = self.base / "eBird original"
        self.orig_n = self.base / "NABirds original"
        self.output = self.base / "result"
        for folder in (self.export_e, self.export_n, self.old_e_ann,
                       self.old_n_ann, self.orig_e, self.orig_n):
            folder.mkdir(parents=True)

    def make_one(self, *, source="eBird", original=True, image_info=True,
                 bad_hash=False, old_ann=False, original_name=None):
        is_ebird = source == "eBird"
        root = self.export_e if is_ebird else self.export_n
        uid = UUID_E if is_ebird else UUID_N
        relative = original_name or (
            "Sterna/200836421/200836421_000043.jpg"
            if is_ebird else "Species/015/IMG_025.jpg"
        )
        exportdir = root / "Complete video library · all species"
        imgdir = exportdir / "img"
        infodir = exportdir / "img_info"
        anndir = exportdir / "ann"
        imgdir.mkdir(parents=True, exist_ok=True)
        infodir.mkdir(parents=True, exist_ok=True)
        if not old_ann:
            anndir.mkdir(parents=True, exist_ok=True)
        annroot = (
            (self.old_e_ann if is_ebird else self.old_n_ann)
            if old_ann else anndir
        )
        original_root = self.orig_e if is_ebird else self.orig_n
        exported_img = imgdir / (uid + ".jpg")
        Image.new("RGB", (80, 60), (200, 110, 22) if is_ebird else (22, 180, 88)).save(exported_img)
        digest = hashlib.sha256(exported_img.read_bytes()).digest()
        if image_info:
            h = base64.b64encode(b"\x00" * 32 if bad_hash else digest).decode("ascii")
            (infodir / (exported_img.name + ".json")).write_text(
                json.dumps({"name": exported_img.name, "width": 80, "height": 60,
                            "hash": h}), encoding="utf-8"
            )
        (annroot / (exported_img.name + ".json")).write_text(
            json.dumps(sample_annotation()), encoding="utf-8"
        )
        orig = original_root / relative
        if original:
            orig.parent.mkdir(parents=True, exist_ok=True)
            orig.write_bytes(exported_img.read_bytes())
        return exported_img, orig

    def run_build(self, *flags):
        args = [
            sys.executable, str(BUILDER),
            "--source-mode", "exports",
            "--ebird-export", str(self.export_e),
            "--nabirds-export", str(self.export_n),
            "--ebird-ann", str(self.old_e_ann),
            "--nabirds-ann", str(self.old_n_ann),
            "--ebird-images", str(self.orig_e),
            "--nabirds-images", str(self.orig_n),
            "--media-map", str(self.base / "absent-map.csv"),
            "--output", str(self.output), *flags,
        ]
        return subprocess.run(args, text=True, capture_output=True)

    def read_annotation(self, src):
        return json.loads((self.output / "annot" / (src + ".json")).read_text(encoding="utf-8"))

    def test_both_export_sources_recover_original_paths(self):
        self.make_one(source="eBird")
        self.make_one(source="NABirds")
        p = self.run_build()
        self.assertEqual(p.returncode, 0, p.stderr)
        ebird = self.read_annotation("ebird")
        nabirds = self.read_annotation("nabirds")
        self.assertEqual(len(ebird), 1)
        self.assertEqual(len(nabirds), 1)
        self.assertEqual(ebird[0]["image"], "eBird/Sterna/200836421/200836421_000043.jpg")
        self.assertEqual(nabirds[0]["image"], "NABirds/Species/015/IMG_025.jpg")
        self.assertEqual(ebird[0]["joints_vis"], [1, 0, 1, 1])
        self.assertEqual(ebird[0]["joints"][1], [31.0, 22.0])
        self.assertTrue((self.output / "images" / ebird[0]["image"]).is_file())
        self.assertEqual({p.name for p in (self.output / "annot").glob("*.json")},
                         {"ebird.json", "nabirds.json"})
        with (self.output / "reports" / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(r["match_method"] == "sha256_original" for r in rows))
        self.assertTrue(all(r["sha256"] for r in rows))

    def test_export_only_preserves_uuid_images(self):
        self.make_one(source="eBird", original=False)
        self.make_one(source="NABirds", original=False)
        p = self.run_build()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.read_annotation("ebird")[0]["image"], "eBird/exported/" + UUID_E + ".jpg")
        self.assertEqual(self.read_annotation("nabirds")[0]["image"], "NABirds/exported/" + UUID_N + ".jpg")

    def test_matching_old_jsons_to_new_exported_images(self):
        self.make_one(source="eBird", old_ann=True)
        p = self.run_build("--audit-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        stats = json.loads((self.output / "reports" / "validation_report.json").read_text())
        self.assertEqual(stats["accepted"], 1)

    def test_image_info_wrong_hash_rejected_before_build(self):
        self.make_one(source="eBird", bad_hash=True)
        result = self.run_build()
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.output / "images").exists())
        with (self.output / "reports" / "rejected_annotations.csv").open(
                newline="", encoding="utf-8-sig") as f:
            rejected = list(csv.DictReader(f))
        self.assertIn("differs from img_info.hash", rejected[0]["reason"])

    def test_no_original_lookup_without_source_images(self):
        self.make_one(source="eBird", original=False, image_info=False)
        p = self.run_build("--no-original-lookup")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.read_annotation("ebird")[0]["image"], "eBird/exported/" + UUID_E + ".jpg")

    def test_strict_mode_blocks_unknown_original(self):
        self.make_one(source="eBird", original=False)
        p = self.run_build("--audit-only", "--require-original-paths")
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertFalse((self.output / "images").exists())

    def test_duplicated_original_hash_is_not_assigned_wrong_video(self):
        _, original = self.make_one(source="eBird")
        dupe = self.orig_e / "Sterna/anothervideo/duplicate.jpg"
        dupe.parent.mkdir(parents=True)
        dupe.write_bytes(original.read_bytes())
        p = self.run_build()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(self.read_annotation("ebird")[0]["image"], "eBird/exported/" + UUID_E + ".jpg")

    def test_audit_does_not_copy_or_create_annot(self):
        self.make_one()
        p = self.run_build("--audit-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((self.output / "images").exists())
        self.assertFalse((self.output / "annot").exists())

    def test_partial_rejected_blocks(self):
        self.make_one(source="eBird")
        self.make_one(source="NABirds", bad_hash=True)
        p = self.run_build()
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertFalse((self.output / "images").exists())


if __name__ == "__main__":
    unittest.main()
