"""Unit/synthetic regression tests for the independent standard ViTPose-B stack."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
    from comparison_models.common.birdgaze.dataset import (
        BirdGazeCropDataset, affine_matrix, coordinate_valid,
        decode_image_coordinates, gaussian_targets,
    )
    from comparison_models.vitpose.model import (
        ViTPoseB, deterministic_bilinear_4x,
    )
    from comparison_models.vitpose.train import ViTPoseBTrainer
    from comparison_models.common.losses import masked_heatmap_mse
    from comparison_models.common.training_utils import (
        require_gpu_capacity, refresh_best_alias, recover_history,
    )
    from comparison_models.common.preflight import sha256
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

    def test_deterministic_bilinear_matches_reference_and_gradient(self):
        import torch.nn.functional as F
        torch.manual_seed(2026)
        x = torch.randn(1, 6, 16, 16, requires_grad=True)
        reference = F.interpolate(x, scale_factor=4, mode="bilinear",
                                  align_corners=False)
        adapted = deterministic_bilinear_4x(x)
        torch.testing.assert_close(adapted, reference, atol=2e-6, rtol=2e-6)
        reference.sum().backward(retain_graph=True)
        native_grad = x.grad.detach().clone()
        x.grad.zero_()
        adapted.sum().backward()
        torch.testing.assert_close(x.grad, native_grad, atol=2e-5, rtol=2e-5)

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
        ds = BirdGazeCropDataset(self.root, "val")
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
            sample = BirdGazeCropDataset(self.root, "val")[0]
            self.assertEqual(sample["raw_scale"].item(), scale)
            self.assertTrue(torch.isfinite(sample["image"]).all())
            self.assertTrue(torch.isfinite(sample["target"]).all())

    def test_missing_coordinate_is_not_pose_supervision(self):
        row = dict(self.record, joints=[[0, 0], [20, 30], [60, 20], [0, 0]],
                   joints_valid=None, joints_vis=[0, 0, 1, 0])
        row.pop("joints_valid")
        (self.root / "annot" / "val.json").write_text(json.dumps([row]))
        sample = BirdGazeCropDataset(self.root, "val")[0]
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
        from comparison_models.vitpose.recipe import RECIPE
        from comparison_models.common.training_utils import make_scheduler
        import torch as _torch
        optimizer = _torch.optim.SGD(
            [_torch.nn.Parameter(torch.zeros(1))], lr=RECIPE["lr"])
        iters_per_epoch = 100
        scheduler = make_scheduler(RECIPE, optimizer,
                                   iters_per_epoch=iters_per_epoch)
        optimizer.step()
        # Paper schedule: 500-iteration linear warmup from ratio 1e-3, then
        # step decay at epochs 81 and 95 of the 100-epoch budget.
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               RECIPE["lr"] * 0.001)
        for _ in range(500):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], RECIPE["lr"])
        for _ in range(81 * iters_per_epoch - 501):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], RECIPE["lr"])
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               RECIPE["lr"] * 0.1)
        for _ in range((95 - 81) * iters_per_epoch):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               RECIPE["lr"] * 0.01)

    def test_gpu_capacity_guard_refuses_low_free_memory(self):
        from subprocess import CompletedProcess
        with patch("comparison_models.common.training_utils.subprocess.run",
                   return_value=CompletedProcess([], 0,
                                                 stdout="1, 1024, 12288\n",
                                                 stderr="")), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1",
                                        "BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "",
                                        "BIRD_HKE_GPU_MIN_FREE_MIB": ""}):
            with self.assertRaisesRegex(RuntimeError, "has only 1024 MiB free"):
                require_gpu_capacity()

    def test_gpu_capacity_guard_allows_sharing_the_same_gpu(self):
        """A co-resident process must NOT block the run if memory is free.

        Both comparison models peak near 2 GiB, so several runs can share one
        card; determinism comes from the per-process RNG and fixed batch order,
        not from exclusive access.
        """
        from subprocess import CompletedProcess

        def fake_run(cmd, **_kwargs):
            if any("memory.free" in part for part in cmd):
                return CompletedProcess([], 0, stdout="1, 40000, 79253\n",
                                        stderr="")
            if any("index,uuid" in part for part in cmd):
                return CompletedProcess([], 0, stdout="1, GPU-bbb\n", stderr="")
            return CompletedProcess([], 0, stdout="GPU-bbb, 284\n", stderr="")

        with patch("comparison_models.common.training_utils.subprocess.run",
                   side_effect=fake_run), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1",
                                        "BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "",
                                        "BIRD_HKE_GPU_MIN_FREE_MIB": ""}):
            require_gpu_capacity()

    def test_gpu_capacity_guard_ignores_other_gpus(self):
        """A busy colleague GPU must not block a free assigned GPU."""
        from subprocess import CompletedProcess

        def fake_run(cmd, **_kwargs):
            if any("memory.free" in part for part in cmd):
                return CompletedProcess([], 0,
                                        stdout="0, 512, 94000\n1, 78000, 79253\n",
                                        stderr="")
            if any("index,uuid" in part for part in cmd):
                return CompletedProcess([], 0,
                                        stdout="0, GPU-aaa\n1, GPU-bbb\n",
                                        stderr="")
            return CompletedProcess([], 0, stdout="GPU-aaa, 284\n", stderr="")

        with patch("comparison_models.common.training_utils.subprocess.run",
                   side_effect=fake_run), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1",
                                        "BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "",
                                        "BIRD_HKE_GPU_MIN_FREE_MIB": ""}):
            require_gpu_capacity()

    def test_gpu_capacity_guard_floor_is_configurable(self):
        from subprocess import CompletedProcess
        with patch("comparison_models.common.training_utils.subprocess.run",
                   return_value=CompletedProcess([], 0,
                                                 stdout="0, 5000, 12288\n",
                                                 stderr="")), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0",
                                        "BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "",
                                        "BIRD_HKE_GPU_MIN_FREE_MIB": "6000"}):
            with self.assertRaisesRegex(RuntimeError, "6000 MiB"):
                require_gpu_capacity()

    def test_gpu_capacity_guard_can_be_overridden(self):
        from subprocess import CompletedProcess
        with patch("comparison_models.common.training_utils.subprocess.run",
                   return_value=CompletedProcess([], 0,
                                                 stdout="0, 1, 12288\n",
                                                 stderr="")), \
                patch.dict(os.environ,
                           {"BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "1"}):
            require_gpu_capacity()

    def test_gpu_capacity_guard_falls_back_to_torch_without_nvidia_smi(self):
        """Hosts whose driver ships without nvidia-smi must still be usable."""
        with patch("comparison_models.common.training_utils.subprocess.run",
                   side_effect=FileNotFoundError("nvidia-smi")), \
                patch("comparison_models.common.training_utils.torch.cuda.mem_get_info",
                      return_value=(40 * 2 ** 30, 48 * 2 ** 30)), \
                patch.dict(os.environ, {"BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "",
                                        "BIRD_HKE_GPU_MIN_FREE_MIB": ""}):
            require_gpu_capacity()

    def test_gpu_capacity_guard_fails_closed_without_any_probe(self):
        with patch("comparison_models.common.training_utils.subprocess.run",
                   side_effect=FileNotFoundError("nvidia-smi")), \
                patch("comparison_models.common.training_utils.torch.cuda.mem_get_info",
                      side_effect=RuntimeError("no CUDA device")), \
                patch.dict(os.environ, {"BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": ""}):
            with self.assertRaisesRegex(RuntimeError,
                                        "Cannot read GPU memory state"):
                require_gpu_capacity()

    def test_best_checkpoint_recovery_keeps_previous_slot(self):
        work = Path(tempfile.mkdtemp(prefix="vitpose-best-slots-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(work))
        first = work / "best_slot_a.pth"
        second = work / "best_slot_b.pth"
        first.write_bytes(b"previous committed best")
        a_digest = sha256(first)
        refresh_best_alias(work, first.name, a_digest)
        self.assertEqual((work / "model_best.pth").read_bytes(),
                         b"previous committed best")
        # New best slot is prepared, then the process crashes BEFORE writing
        # checkpoint_last. The old slot remains available for exact recovery.
        second.write_bytes(b"new best not yet committed")
        refresh_best_alias(work, first.name, a_digest)
        self.assertEqual((work / "model_best.pth").read_bytes(),
                         b"previous committed best")
        refresh_best_alias(work, second.name, sha256(second))
        self.assertEqual((work / "model_best.pth").read_bytes(),
                         b"new best not yet committed")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            refresh_best_alias(work, first.name, "bad")

    def test_interrupted_epoch_metric_history_recovery(self):
        with tempfile.TemporaryDirectory(prefix="vitpose-history-") as d:
            path = Path(d) / "training_history.jsonl"
            one = {"epoch": 1, "val_pck_005": 0.1}
            two = {"epoch": 2, "val_pck_005": 0.2}
            path.write_text(json.dumps(one) + "\n", encoding="utf-8")
            recover_history(Path(d), 2, two)
            lines = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(lines, [one, two])
            recover_history(Path(d), 2, two)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaisesRegex(RuntimeError, "disagrees"):
                recover_history(Path(d), 2, {"epoch": 2, "val_pck_005": 9.})

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
