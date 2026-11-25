"""Stdlib-only tests; never import PyTorch, MMPose or existing model code."""
import json
from pathlib import Path
import tempfile
import unittest

from comparison_models.common.preflight import (
    BASELINES, SCENARIOS, ensure_under, prepare, reserve,
    validate_exports, validate_os_cs_distinction,
    validate_standard_vitpose_config, validate_posebh_upstream_config,
)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self, scenario="FD", *, altered=False):
        ds = self.root / "BirdGaze_v2" / SCENARIOS[scenario] / "annot"
        out = self.root / "comparison_models/generated" / scenario
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
        a, m1 = prepare("vitpose-b", "FD", self.root, check_upstream=False)
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
        run, manifest = prepare("vitpose-b", "FD", self.root, check_upstream=False)
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
        os_dir = self.root / "comparison_models/generated/OS"
        cs_dir = self.root / "comparison_models/generated/CS"
        for split in ("train", "val"):
            (cs_dir / f"{split}_coco.json").write_bytes(
                (os_dir / f"{split}_coco.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "COCO exports match"):
            validate_os_cs_distinction(self.root)

    def test_standard_vitpose_architecture_guard(self):
        upstream = self.root / "official_vitpose_b.py"
        upstream.write_text(
            "model = dict(type='TopDown', pretrained=None, "
            "backbone=dict(type='ViT', embed_dim=768, depth=12, num_heads=12), "
            "keypoint_head=dict(type='TopdownHeatmapSimpleHead'))\n",
            encoding="utf-8",
        )
        validate_standard_vitpose_config(upstream)
        upstream.write_text(
            upstream.read_text().replace("depth=12", "depth=24"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Wrong ViTPose-B depth"):
            validate_standard_vitpose_config(upstream)
        upstream.write_text(
            upstream.read_text().replace("depth=24", "depth=12")
            .replace("pretrained=None", "pretrained='old_checkpoint.pth'"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Wrong ViTPose-B pretrained weights"):
            validate_standard_vitpose_config(upstream)

    def test_standard_vitpose_contract_points_to_official_single_task_config(self):
        cfg = BASELINES["vitpose-b"]
        self.assertEqual(cfg.run_folder, "ViTPose_B")
        self.assertEqual(cfg.upstream_folder, "vitpose")
        self.assertTrue(cfg.upstream_model_config.endswith(
            "ViTPose_base_simple_coco_256x192.py"))
        self.assertEqual(cfg.adapter_config,
                         "comparison_models/vitpose/recipe.py")
        self.assertNotIn("vitposepp-b", BASELINES)

    def test_pinned_posebh_moe_prototype_contract(self):
        config = self.root / "official_posebh_b.py"
        config.write_text(
            "num_in_class_proto=3\n"
            "model=dict(type='TopDownMoEProto',pretrained=None,"
            "multihead_pretrained=None,"
            "backbone=dict(type='ViTMoE',depth=12,embed_dim=768,"
            "num_heads=12,num_expert=6,part_features=192),"
            "proto_head=dict(type='KptProtoHead',"
            "extra=dict(num_in_class_proto=num_in_class_proto)))\n",
            encoding="utf-8",
        )
        validate_posebh_upstream_config(config)
        config.write_text(
            config.read_text().replace("num_expert=6", "num_expert=3"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "expert count"):
            validate_posebh_upstream_config(config)

    def test_posebh_registry_points_to_official_moe_config(self):
        entry = BASELINES["posebh-b"]
        self.assertEqual(entry.upstream_folder, "posebh")
        self.assertEqual(entry.run_folder, "PoseBH_B")
        self.assertTrue(entry.upstream_model_config.endswith("vitb_posebh.py"))
        self.assertEqual(entry.adapter_config,
                         "comparison_models/posebh/recipe.py")

    def test_registry_contains_only_remaining_models(self):
        self.assertEqual(set(BASELINES), {"vitpose-b", "posebh-b"})


if __name__ == "__main__":
    unittest.main()
