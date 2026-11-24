"""Synthetic checks of double-blind Supervisely reannotation sample exporter."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from build_interannotator_sample import (
    Frame, ORIGINAL_QUOTAS, pairing, pose_distance,
    diversified_video_frames, main, image_annotation,
)
import random


class ReannotationSampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "birdgaze"
        self.out = Path(self.tmp.name) / "sample"
        (self.root / "annot").mkdir(parents=True)
        self.ebird = []
        self.nabirds = []
        self.quotas = Path(self.tmp.name) / "quotas.json"
        self.quotas.write_text(json.dumps({"Paulo": 2, "Luís": 2, "Riccardo": 2, "Romain": 2}),
                               encoding="utf-8")

    def add_frame(self, source="eBird", species="Sterna", video="v1",
                  number=0, pose_shift=0.0):
        if source == "eBird":
            relative = f"eBird/{species}/{video}/frame_{number:04d}.jpg"
        else:
            relative = f"NABirds/{species}/photo_{number:04d}.jpg"
        p = self.root / "images" / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (80, 60), (50 + number % 200, 120, 90)).save(p)
        shift = min(12.0, pose_shift)
        self_rec = {
            "image": relative,
            "bbox": [10, 6, 65, 54],
            "center": [37.5, 30],
            "scale": 0.3,
            "joints": [
                [25 + shift, 15], [29 + shift, 25],
                [40 + shift, 25], [45 + shift, 32]
            ],
            "joints_vis": [1, 1, 1, 1],
        }
        if source == "eBird":
            self.ebird.append(self_rec)
        else:
            self.nabirds.append(self_rec)
        return self_rec

    def write_annot(self):
        (self.root / "annot" / "ebird.json").write_text(
            json.dumps(self.ebird), encoding="utf-8"
        )
        (self.root / "annot" / "nabirds.json").write_text(
            json.dumps(self.nabirds), encoding="utf-8"
        )

    def run_main(self, *flags):
        with patch("sys.argv", [
            "build_interannotator_sample.py",
            "--dataset", str(self.root),
            "--output", str(self.out),
            "--quotas-json", str(self.quotas),
            "--min-frame-gap", "1",
            "--min-pose-distance", "0",
            "--max-per-video", "20",
            *flags,
        ]):
            return main()

    def test_doubled_quota_math_and_all_pairs_independent(self):
        quotas = {person: q * 2 for person, q in ORIGINAL_QUOTAS.items()}
        self.assertEqual(sum(quotas.values()), 4400)
        self.assertEqual(sum(quotas.values()) // 2, 2200)
        pairs = pairing(quotas, random.Random(20261124))
        self.assertEqual(len(pairs), 2200)
        self.assertTrue(all(a != b for a, b in pairs))
        assigned = Counter(person for pair in pairs for person in pair)
        self.assertEqual(assigned, Counter(quotas))
        self.assertEqual(quotas["Tomás Santos"], 600)

    def test_farthest_pose_picks_with_frame_gap(self):
        records = [self.add_frame(number=i, pose_shift=i * 2) for i in range(8)]
        self.write_annot()
        from build_interannotator_sample import load_frames
        grouped, _ = load_frames(self.root, False, 3)
        frames = next(iter(grouped.values()))
        selected = diversified_video_frames(frames, 3, 2, 0.01,
                                            random.Random(3))
        self.assertEqual(len(selected), 3)
        ordered = sorted(frames, key=lambda f: f.filename)
        positions = [ordered.index(frame) for frame in selected]
        self.assertTrue(all(abs(a - b) >= 2 for i, a in enumerate(positions)
                            for b in positions[i+1:]))
        self.assertTrue(all(pose_distance(a, b) >= .01
                            for i, a in enumerate(selected) for b in selected[i+1:]))

    def test_supervisely_project_has_two_copies_and_only_bbox(self):
        for species, video, idx in (
            ("Sterna", "v1", 0), ("Sterna", "v2", 1),
            ("Corvus", "v1", 2), ("Corvus", "v2", 3),
        ):
            self.add_frame(species=species, video=video, number=idx,
                           pose_shift=float(idx))
        self.write_annot()
        self.assertEqual(self.run_main(), 0)
        upload = self.out / "supervisely_upload"
        meta = json.loads((upload / "meta.json").read_text())
        titles = {c["title"] for c in meta["classes"]}
        self.assertEqual(len(titles), 9)
        self.assertIn("bird", titles)
        self.assertIn("crown", titles)
        self.assertEqual(len(list(upload.rglob("img/*.jpg"))), 8)
        self.assertEqual(len(list(upload.rglob("ann/*.json"))), 8)
        assignments = json.loads((self.out / "reports" / "selection_report.json").read_text())
        self.assertEqual(assignments["unique_images"], 4)
        self.assertEqual(assignments["independent_assignments"], 8)
        for filename in upload.rglob("ann/*.json"):
            ann = json.loads(filename.read_text())
            self.assertEqual(len(ann["objects"]), 1)
            self.assertEqual(ann["objects"][0]["classTitle"], "bird")
            self.assertEqual(ann["objects"][0]["geometryType"], "rectangle")
            self.assertEqual(ann["size"], {"width": 80, "height": 60})
        refs = self.out / "private_reference" / "event_reference_annotations.json"
        baselines = json.loads(refs.read_text())
        self.assertEqual(len(baselines), 4)
        self.assertEqual(len(next(iter(baselines.values()))["event_reference"]["joints"]), 4)
        self.assertFalse((upload / "event_reference_annotations.json").exists())
        self.assertEqual(len(list(upload.rglob("*reference*"))), 0)
        self.assertEqual(len(list(upload.iterdir())), 5)  # meta.json + four people

    def test_dry_run_does_not_copy_anything(self):
        for idx in range(4):
            self.add_frame(video=f"v{idx}", number=idx)
        self.write_annot()
        self.assertEqual(self.run_main("--dry-run"), 0)
        self.assertFalse(self.out.exists())

    def test_source_quota_balance_and_two_reannotations_each(self):
        for idx in range(6):
            self.add_frame(species=("Sterna" if idx < 3 else "Corvus"),
                           video=f"v{idx}", number=idx)
        self.write_annot()
        self.assertEqual(self.run_main(), 0)
        import csv
        with (self.out / "reports" / "assignments.csv").open(
                encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 8)
        grouped = Counter(row["sample_id"] for row in rows)
        self.assertEqual(set(grouped.values()), {2})
        by_person = Counter(row["annotator"] for row in rows)
        self.assertEqual(by_person, {"Paulo": 2, "Luís": 2, "Riccardo": 2, "Romain": 2})

    def test_not_enough_diverse_frames_fails_explicitly(self):
        for idx in range(4):
            self.add_frame(number=idx)  # only one video
        self.write_annot()
        with self.assertRaisesRegex(ValueError, "Cannot find"):
            self.run_main("--max-per-video", "1")

    def test_optional_nabirds_mix(self):
        for idx in range(3):
            self.add_frame(source="eBird", video=f"v{idx}", number=idx)
            self.add_frame(source="NABirds", species=f"species_{idx}", number=idx)
        self.write_annot()
        self.assertEqual(self.run_main("--nabirds-share", "0.5"), 0)
        report = json.loads((self.out / "reports" / "selection_report.json").read_text())
        self.assertEqual(report["unique_images"], 4)
        records = json.loads((self.out / "private_reference" /
                              "event_reference_annotations.json").read_text())
        self.assertEqual(Counter(item["source"] for item in records.values()),
                         {"eBird": 2, "NABirds": 2})

    def test_invalid_bbox_is_rejected_before_upload(self):
        with self.assertRaises(ValueError):
            image_annotation((80, 60), (10, 10, 100, 100))

    def test_zip_is_independent_of_private_reference(self):
        for idx in range(4):
            self.add_frame(video=f"v{idx}", number=idx)
        self.write_annot()
        self.assertEqual(self.run_main("--make-zip"), 0)
        import zipfile
        with zipfile.ZipFile(self.out / "supervisely_upload.zip") as z:
            names = z.namelist()
            self.assertIn("meta.json", names)
            self.assertFalse(any("private" in n or "reference" in n for n in names))
            self.assertEqual(sum(n.endswith(".jpg") for n in names), 8)


if __name__ == "__main__":
    unittest.main()
