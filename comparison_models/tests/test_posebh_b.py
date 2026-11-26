"""CPU-only PoseBH-B BirdGaze adaptation tests; no training/checkpoint writes."""
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
    from comparison_models.posebh.model import PoseBHBirdGazeB
    from comparison_models.posebh.adapter_dataset import (
        BirdGazePoseBHDataset, decode_image_coordinates)
    from comparison_models.posebh.recipe import RECIPE
    from comparison_models.posebh.train import PoseBHTrainer
    from comparison_models.common.training_utils import (
        require_gpu_capacity, recover_history, refresh_best_alias)
    from comparison_models.common.preflight import sha256
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

    def test_unlabeled_keypoint_consistency_only_after_epoch_50(self):
        model = self.make_small()
        # Entire image has a prototype-consistent pseudo-keypoint, while
        # all four joints are unsupervised in this synthetic case.
        prototype = model.prototypes[0, 0].detach()
        embedding = prototype.view(1, 64, 1, 1).expand(
            1, 64, 64, 64).clone().requires_grad_(True)
        proto = model.prototype_heatmaps(embedding)
        prediction = (proto.detach() + 0.10).clone().requires_grad_(True)
        target = torch.zeros_like(prediction)
        missing = torch.zeros(1, 4)
        before, early = model.loss_parts(prediction, embedding, target,
                                         missing, epoch=49)
        after, mature = model.loss_parts(prediction, embedding, target,
                                        missing, epoch=50)
        self.assertEqual(float(early["css"].detach()), 0.)
        self.assertGreater(float(mature["css"].detach()), 0.)
        after.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.)

    def test_trainer_loss_matches_model_loss_parts_and_metrics(self):
        """The trainer must call the real model/loss API, not a stale one.

        Guards against the consolidation regression where the trainer asked the
        model for methods that never existed and for recipe keys the recipe
        does not define, which only failed once real training started.
        """
        torch.manual_seed(2026)
        model = self.make_small()
        model.train()
        image = torch.randn(1, 3, 256, 256)
        target = torch.zeros(1, 4, 64, 64)
        target[0, 0, 24, 24] = 1
        target[0, 2, 30, 40] = 1
        valid = torch.tensor([[1., 0., 1., 0.]])

        trainer = PoseBHTrainer()
        self.assertEqual(set(trainer.loss_parts),
                         {"supervised", "prototype_mse",
                          "prototype_contrast", "css"})
        # Reference from the model's own combination, computed before the
        # trainer's prototype EMA update changes the bank.
        with torch.no_grad():
            heatmaps, embedding = model(image)
        reference, ref_parts = model.loss_parts(
            heatmaps, embedding, target, valid, epoch=0)

        before = model.prototypes.clone()
        total, parts = trainer.loss(model, image, target, valid, epoch=0)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(parts), set(trainer.loss_parts))
        self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
        self.assertAlmostEqual(float(total.detach()), float(reference), places=6)
        for key, value in ref_parts.items():
            self.assertAlmostEqual(float(parts[key].detach()), float(value), places=6)
        # The prototype bank is refreshed from the batch, as upstream does.
        self.assertFalse(torch.equal(model.prototypes, before))
        total.backward()
        self.assertIsNotNone(model.heatmap_head.weight.grad)

    def test_recipe_refuses_pretrained_weights(self):
        trainer = PoseBHTrainer()
        trainer.ensure_recipe_is_from_scratch()  # must not raise
        original = RECIPE["weights"]
        try:
            RECIPE["weights"] = "pretrained_vitpose_plus_plus"
            with self.assertRaisesRegex(RuntimeError, "random weights"):
                trainer.ensure_recipe_is_from_scratch()
        finally:
            RECIPE["weights"] = original

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
        from comparison_models.common.birdgaze.dataset import BirdGazeCropDataset
        self.assertTrue(issubclass(BirdGazePoseBHDataset, BirdGazeCropDataset))
        self.assertIsNotNone(decode_image_coordinates)


@unittest.skipUnless(HAS_DEPS, "Requires torch, numpy, opencv")
class PoseBHProtocolTests(unittest.TestCase):
    def test_schedule_and_gpu_busy_guard(self):
        from comparison_models.posebh.recipe import RECIPE
        from comparison_models.common.training_utils import make_scheduler
        import torch as _torch
        optimizer = _torch.optim.SGD(
            [_torch.nn.Parameter(torch.zeros(1))], lr=RECIPE["lr"])
        iters_per_epoch = 100
        scheduler = make_scheduler(RECIPE, optimizer,
                                   iters_per_epoch=iters_per_epoch)
        optimizer.step()
        # Paper schedule: 500-iteration linear warmup from ratio 1e-3, then
        # step decay at epochs 50 and 90 of 100.
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               RECIPE["lr"] * 0.001)
        for _ in range(500):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], RECIPE["lr"])
        for _ in range(50 * iters_per_epoch - 501):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], RECIPE["lr"])
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               RECIPE["lr"] * 0.1)
        for _ in range((90 - 50) * iters_per_epoch):
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"],
                               RECIPE["lr"] * 0.01)
        from subprocess import CompletedProcess
        with patch("comparison_models.common.training_utils.subprocess.run",
                   return_value=CompletedProcess([], 0,
                                                 stdout="0, 512, 12288\n",
                                                 stderr="")), \
                patch.dict(os.environ, {"BIRD_HKE_SKIP_GPU_CAPACITY_CHECK": "",
                                        "BIRD_HKE_GPU_MIN_FREE_MIB": ""}):
            with self.assertRaisesRegex(RuntimeError, "has only 512 MiB free"):
                require_gpu_capacity()

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
