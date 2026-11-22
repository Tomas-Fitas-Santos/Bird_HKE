"""CPU-only PoseBH-B BirdGaze adaptation tests; no training/checkpoint writes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
    from external_models.posebh.model import PoseBHBirdGazeB
    from external_models.posebh.dataset import (
        BirdGazePoseBHDataset, decode_image_coordinates)
    from external_models.posebh.birdgaze_posebh_b import RECIPE
    from external_models.posebh.train import (
        lr_multiplier, require_idle_gpu, recover_history, refresh_best_alias)
    from external_models.remaining_baselines.prepare import sha256
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@unittest.skipUnless(HAS_DEPS, "Requires torch, numpy, opencv")
class PoseBHModelTests(unittest.TestCase):
    def make_small(self):
        return PoseBHBirdGazeB(dim=96, depth=2, heads=3, part=24,
                               drop_path_rate=0)

    def test_moe_and_prototypes_have_correct_dimensions(self):
        model = self.make_small()
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(len(model.blocks[0].mlp.experts), 1)
        self.assertEqual(model.blocks[0].mlp.fc2.out_features, 72)
        self.assertEqual(model.blocks[0].mlp.experts[0].out_features, 24)
        self.assertEqual(tuple(model.prototypes.shape), (4, 3, 64))
        self.assertEqual(model.prototypes.requires_grad, False)
        self.assertEqual(RECIPE["weights"], "random_from_scratch_no_pretrained")
        self.assertEqual(RECIPE["heads"]["original_cross_dataset_css"], False)

    def test_forward_backward_prototype_and_regression_losses(self):
        torch.manual_seed(2026)
        model = self.make_small()
        model.train()
        image = torch.randn(1, 3, 256, 256)
        hm, emb = model(image)
        self.assertEqual(tuple(hm.shape), (1, 4, 64, 64))
        self.assertEqual(tuple(emb.shape), (1, 64, 64, 64))
        prt = model.prototype_heatmaps(emb)
        self.assertEqual(tuple(prt.shape), (1, 4, 64, 64))
        self.assertTrue(((prt >= 0) & (prt <= 1)).all())
        target = torch.zeros_like(hm)
        target[0, 0, 24, 24] = 1
        target[0, 1, 36, 24] = 1
        valid = torch.tensor([[1., 1., 0., 0.]])
        total, parts = model.loss_parts(hm, emb, target, valid, epoch=0)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(parts),
                         {"supervised", "prototype_mse", "prototype_contrast", "css"})
        self.assertEqual(parts["css"].detach().item(), 0)
        total.backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)
        self.assertIsNotNone(model.proto_head.weight.grad)
        self.assertTrue(torch.isfinite(model.proto_head.weight.grad).all())

    def test_prototype_updates_are_ema_and_checkpointed(self):
        model = self.make_small()
        assert "prototypes" in model.state_dict()
        before = model.prototypes.clone()
        features = torch.randn(1, 64, 64, 64)
        target = torch.zeros(1, 4, 64, 64)
        target[0, 0, :, :] = 1.0
        model.update_prototypes(features, target)
        self.assertTrue(torch.isfinite(model.prototypes).all())
        self.assertEqual(tuple(model.prototypes.shape), tuple(before.shape))
        self.assertTrue(torch.allclose(model.prototypes.norm(dim=-1),
                                       torch.ones(4, 3), atol=1e-5))
        state = model.state_dict()
        other = self.make_small()
        other.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(other.prototypes, model.prototypes))

    def test_sinkhorn_valid_and_no_unlabeled_class_cross_training(self):
        model = self.make_small()
        source = torch.tensor([[0.3, 0.6, 0.8], [0.8, 0.3, 0.5]])
        assignments = model.sinkhorn_assign(source)
        self.assertEqual(tuple(assignments.shape), (2, 3))
        self.assertTrue(torch.isfinite(assignments).all())
        self.assertTrue(torch.allclose(assignments.sum(dim=1),
                                       torch.ones(2), atol=1e-4))
        feats = torch.randn(1, 64, 64, 64)
        targets = torch.zeros(1, 4, 64, 64)
        contrast = model.prototype_contrast_loss(feats, targets)
        self.assertAlmostEqual(float(contrast.detach()), 0.0)

    def test_wrong_input_and_proto_shape_fail(self):
        model = self.make_small()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 192, 192))
        with self.assertRaises(ValueError):
            PoseBHBirdGazeB(dim=96, depth=2, heads=3, part=24,
                            variants=4)


@unittest.skipUnless(HAS_DEPS, "Requires torch, numpy, opencv")
class PoseBHSharedDatasetTests(unittest.TestCase):
    def test_same_source_crop_and_four_keypoints_as_vitpose(self):
        from external_models.vitpose.dataset import BirdGazeViTPoseDataset
        self.assertTrue(issubclass(BirdGazePoseBHDataset, BirdGazeViTPoseDataset))
        self.assertIsNotNone(decode_image_coordinates)


@unittest.skipUnless(HAS_DEPS, "Requires torch, numpy, opencv")
class PoseBHProtocolTests(unittest.TestCase):
    def test_schedule_and_gpu_busy_guard(self):
        self.assertAlmostEqual(lr_multiplier(0), 0.1)
        self.assertAlmostEqual(lr_multiplier(5), 1.)
        self.assertAlmostEqual(lr_multiplier(100), 0.02)
        from subprocess import CompletedProcess
        with patch("external_models.posebh.train.subprocess.run",
                   return_value=CompletedProcess([], 0, stdout="991\n", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "Another GPU compute process"):
                require_idle_gpu()

    def test_best_and_history_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            slot = directory / "best_slot_a.pth"
            slot.write_bytes(b"posebh best snapshot")
            refresh_best_alias(directory, slot.name, sha256(slot))
            self.assertEqual((directory / "model_best.pth").read_bytes(),
                             b"posebh best snapshot")
            row = {"epoch": 1, "val_pck_005": 0.25}
            recover_history(directory, 1, row)
            self.assertEqual(json.loads(
                (directory / "training_history.jsonl").read_text().strip()), row)
            recover_history(directory, 1, row)
            with self.assertRaises(RuntimeError):
                recover_history(directory, 1,
                                {"epoch": 1, "val_pck_005": 0.99})


if __name__ == "__main__":
    unittest.main()
