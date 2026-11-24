"""Local-only tests for the Supervisely importer/exporter safety checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from supervisely_reannotation_io import (
    REQUIRED_CLASSES, preflight, verify_export, remote_upload,
)


def annotation(with_points=False):
    result = {
        "size": {"width": 80, "height": 60}, "objects": [{
            "classTitle": "bird", "geometryType": "rectangle",
            "points": {"exterior": [[3, 4], [68, 56]]},
        }],
    }
    if with_points:
        for title, x in (
            ("crown", 12), ("left_eye_occ", 24),
            ("right_eye", 38), ("beak", 50)
        ):
            result["objects"].append({
                "classTitle": title, "geometryType": "point",
                "points": {"exterior": [[x, 22]]},
            })
    return result


class ReannotationImportExportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sample = self.root / "sample"
        self.import_dir = self.sample / "supervisely_upload"
        self.export_dir = self.root / "downloaded_project"
        self.assignments = [
            {"sample_id": "F00001_example", "annotator": "Person A",
             "dataset_name": "Person_A", "image_filename": "F00001_example.jpg"},
            {"sample_id": "F00001_example", "annotator": "Person B",
             "dataset_name": "Person_B", "image_filename": "F00001_example.jpg"},
        ]
        self.import_dir.mkdir(parents=True)
        self.export_dir.mkdir(parents=True)
        meta = {
            "projectType": "images",
            "classes": [
                {"title": name, "shape": shape}
                for name, shape in REQUIRED_CLASSES.items()
            ],
        }
        (self.import_dir / "meta.json").write_text(json.dumps(meta))
        (self.export_dir / "meta.json").write_text(json.dumps(meta))
        reports = self.sample / "reports"
        reports.mkdir()
        with (reports / "assignments.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.assignments[0]))
            w.writeheader()
            w.writerows(self.assignments)
        for entry in self.assignments:
            for project_root, points in ((self.import_dir, False),
                                         (self.export_dir, True)):
                base = project_root / entry["dataset_name"]
                (base / "img").mkdir(parents=True)
                (base / "ann").mkdir(parents=True)
                (base / "img" / entry["image_filename"]).write_bytes(b"test-jpeg")
                (base / "ann" / (entry["image_filename"] + ".json")).write_text(
                    json.dumps(annotation(with_points=points)), encoding="utf-8"
                )

    def test_preflight_two_independent_assignments_and_no_landmark_leak(self):
        self.assertEqual(preflight(self.sample), (1, 2))

    def test_upload_dry_run_does_not_need_credentials(self):
        args = SimpleNamespace(sample=self.sample, workspace_id=777,
                               project_name="Bird study", execute=False)
        remote_upload(args)

    def test_preflight_rejects_keypoint_leak(self):
        p = (self.import_dir / "Person_A" / "ann" / "F00001_example.jpg.json")
        p.write_text(json.dumps(annotation(with_points=True)))
        with self.assertRaisesRegex(ValueError, "leaks"):
            preflight(self.sample)

    def test_preflight_rejects_missing_image(self):
        (self.import_dir / "Person_B" / "img" / "F00001_example.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            preflight(self.sample)

    def test_verify_export_collects_two_complete_independent_annotations(self):
        self.assertEqual(verify_export(self.sample, self.export_dir), 0)
        p = self.sample / "reports" / "reannotation_review_records.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(data["found_annotations"], 2)
        self.assertEqual(data["fully_annotated"], 2)
        self.assertEqual(data["issues"], 0)
        self.assertTrue(all(len(rec["landmarks"]) == 4 for rec in data["records"]))

    def test_verify_export_detects_missing_point(self):
        p = self.export_dir / "Person_B" / "ann" / "F00001_example.jpg.json"
        partial = annotation(with_points=True)
        partial["objects"].pop()
        p.write_text(json.dumps(partial))
        self.assertEqual(verify_export(self.sample, self.export_dir), 2)
        data = json.loads((self.sample / "reports" /
                           "reannotation_review_records.json").read_text())
        self.assertEqual(data["issues"], 1)
        self.assertEqual(data["fully_annotated"], 1)

    def test_verify_export_rejects_duplicate_landmarks(self):
        p = self.export_dir / "Person_A" / "ann" / "F00001_example.jpg.json"
        duplicate = annotation(with_points=True)
        duplicate["objects"].append(duplicate["objects"][1].copy())
        p.write_text(json.dumps(duplicate))
        self.assertEqual(verify_export(self.sample, self.export_dir), 2)

    def test_verify_export_detects_missing_exported_annotation(self):
        (self.export_dir / "Person_A" / "ann" / "F00001_example.jpg.json").unlink()
        self.assertEqual(verify_export(self.sample, self.export_dir), 2)
        data = json.loads((self.sample / "reports" /
                           "reannotation_review_records.json").read_text())
        self.assertEqual(data["found_annotations"], 1)


if __name__ == "__main__":
    unittest.main()
