"""Unit/synthetic regression tests for the independent standard ViTPose-B stack."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
    from external_models.vitpose.dataset import (
        BirdGazeViTPoseDataset, affine_matrix, coordinate_valid,
        decode_image_coordinates, gaussian_targets,
    )
    from external_models.vitpose.model import ViTPoseB, masked_heatmap_mse
    from external_models.vitpose.train import lr_multiplier, require_idle_gpu
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "Requires torch, OpenCV, NumPy")
class ViTPoseBModelTests(unittest.TestCase):
    def test_full_architecture_shape_and_no_pretraining(self):
        model = ViTPoseB()
        self.assertEqual(len(model.blocks), 12)
        self.assertEqual(model.patch_embed.proj.kernel_size, (16, 16))
        self.assertEqual(model.patch_embed.proj.stride, (16, 16))
        self.assertEqual(model.patch_embed.proj.padding, (2, 2))
        self.assertEqual(model.pos_embed.shape, (1, 257, 768))
        self.assertEqual(model.final_layer.weight.shape, (4, 768, 3, 3))
        self.assertEqual(model.blocks[0].attn.num_heads, 12)
        self.assertEqual(model.blocks[0].mlp[0].out_features, 3072)

    def test_forward_backward_small_backbone(self):
        torch.manual_seed(2026)
        model = ViTPoseB(dim=48, depth=2, heads=3, drop_path_rate=0)
        x = torch.randn(1, 3, 256, 256)
        y = model(x)
        self.assertEqual(tuple(y.shape), (1, 4, 64, 64))
        target = torch.zeros_like(y)
        target[:, 0, 32, 32] = 1
        weight = torch.tensor([[1., 0., 0., 0.]])
        loss = masked_heatmap_mse(y, target, weight)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.patch_embed.proj.weight.grad).all())
        self.assertTrue(torch.isfinite(model.final_layer.weight.grad).all())

    def test_no_supervision_zero_loss_and_zero_gradients(self):
        model = ViTPoseB(dim=48, depth=1, heads=3, drop_path_rate=0)
        pred = model(torch.randn(1, 3, 256, 256))
        loss = masked_heatmap_mse(pred, torch.randn_like(pred), torch.zeros(1, 4))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertEqual(float(model.final_layer.weight.grad.abs().sum()), 0.)

    def test_fixed_input_size(self):
        with self.assertRaises(ValueError):
            ViTPoseB(image_size=192)


@unittest.skipUnless(HAS_DEPS, "Requires torch, OpenCV, NumPy")
class BirdGazeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "annot").mkdir()
        (self.root / "images").mkdir()
        # Deliberately asymmetrical BGR input: crop orientation and color must
        # remain the same as the original native JointsDataset pipeline.
        im = np.zeros((80, 100, 3), dtype=np.uint8)
        im[:, :50, 0] = 200
        im[:, 50:, 2] = 120
        cv2.imwrite(str(self.root / "images" / "bird.png"), im)
        self.record = {
            "image": "bird.png",
            "center": [50, 40],
            "scale": 0.5,
            "bbox": [10.1, 10.7, 90.1, 70.2],
            "joints": [[50, 20], [35, 30], [65, 30], [50, 58]],
            "joints_vis": [1, 0, 1, 0],
            "joints_valid": [1, 1, 1, 1],
        }
        for name in ("train", "val"):
            (self.root / "annot" / f"{name}.json").write_text(
                json.dumps([self.record]), encoding="utf-8")

    def test_source_crop_and_occluded_coordinate_supervision(self):
        ds = BirdGazeViTPoseDataset(self.root, "val")
        sample = ds[0]
        self.assertEqual(tuple(sample["image"].shape), (3, 256, 256))
        self.assertEqual(tuple(sample["target"].shape), (4, 64, 64))
        self.assertEqual(sample["offset"].tolist(), [10., 10.])
        self.assertEqual(sample["center"].tolist(), [40., 30.])
        self.assertEqual(sample["weight"].tolist(), [1., 1., 1., 1.])
        self.assertEqual(sample["eval_valid"].tolist(), [True, False, True, False])
        self.assertEqual(sample["gt"].tolist(), self.record["joints"])

    def test_signed_and_zero_scale_are_retained(self):
        for scale in (-0.5, 0):
            row = dict(self.record, scale=scale)
            (self.root / "annot" / "val.json").write_text(json.dumps([row]))
            sample = BirdGazeViTPoseDataset(self.root, "val")[0]
            self.assertEqual(sample["raw_scale"].item(), scale)
            self.assertTrue(torch.isfinite(sample["image"]).all())
            self.assertTrue(torch.isfinite(sample["target"]).all())

    def test_missing_coordinate_is_not_pose_supervision(self):
        row = dict(self.record, joints=[[0, 0], [20, 30], [60, 20], [0, 0]],
                   joints_valid=None, joints_vis=[0, 0, 1, 0])
        row.pop("joints_valid")
        (self.root / "annot" / "val.json").write_text(json.dumps([row]))
        sample = BirdGazeViTPoseDataset(self.root, "val")[0]
        self.assertEqual(sample["weight"].tolist(), [0., 1., 1., 0.])

    def test_source_crop_and_inverse_heatmap(self):
        heatmaps = np.full((1, 4, 64, 64), -1., dtype=np.float32)
        heatmaps[:, :, 32, 32] = 2.
        preds = decode_image_coordinates(
            heatmaps, np.array([[40, 30]], dtype=np.float32),
            np.array([0.625], dtype=np.float32),
            np.array([[10, 10]], dtype=np.float32),
        )
        np.testing.assert_allclose(preds[0, :, 0], [50.] * 4, atol=1e-5)
        np.testing.assert_allclose(preds[0, :, 1], [40.] * 4, atol=1e-5)

    def test_affine_matches_native_birdhke(self):
        from Bird_HKE.lib.utilities.transforms import get_affine_transform
        for scale in (0., -0.5, 0.75):
            for rotation in (0., 30.):
                expected = get_affine_transform(
                    np.array([40., 30.], dtype=np.float32),
                    np.array([scale, scale], dtype=np.float32),
                    rotation, [256, 256])
                actual = affine_matrix([40., 30.], scale, rotation, 256)
                np.testing.assert_allclose(actual, expected, atol=1e-5)


@unittest.skipUnless(HAS_DEPS, "Requires torch, OpenCV, NumPy")
class ViTPoseProtocolTests(unittest.TestCase):
    def test_lr_schedule(self):
        self.assertAlmostEqual(lr_multiplier(0), 0.1)
        self.assertAlmostEqual(lr_multiplier(5), 1.)
        self.assertAlmostEqual(lr_multiplier(100), 0.02)

    def test_gpu_busy_guard(self):
        from subprocess import CompletedProcess
        with patch("external_models.vitpose.train.subprocess.run",
                   return_value=CompletedProcess([], 0, stdout="284\n", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "Another GPU compute process"):
                require_idle_gpu()

    def test_loss_masks_occluded_with_coordinates(self):
        p = torch.zeros((1, 4, 64, 64), requires_grad=True)
        t = torch.ones_like(p)
        mask = torch.tensor([[1., 1., 0., 0.]])
        loss = masked_heatmap_mse(p, t, mask)
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        self.assertEqual(float(p.grad[0, 2].abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
