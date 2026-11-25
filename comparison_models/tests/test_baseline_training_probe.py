"""CPU-only checks for throwaway training probe and sample integrity."""
import json
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from comparison_models.common.training_probe import (
    checkpoint_epoch, select_real_examples,
)
from comparison_models.common import training_utils as vit_train


def record(x, scale=1, visible=None, valid=None):
    return {
        "joints": [[x, 10], [20, 20], [25, 25], [30, 30]],
        "scale": scale,
        **({"joints_vis": visible} if visible is not None else {}),
        **({"joints_valid": valid} if valid is not None else {}),
    }


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class ProbeDataTests(unittest.TestCase):
    def test_select_valid_real_images_and_exclude_zero_scale_validation(self):
        rows = [
            record(0, scale=0, visible=[1, 0, 0, 0],
                   valid=[1, 0, 0, 0]),
            record(20, scale=1, visible=[0, 0, 0, 0],
                   valid=[0, 0, 0, 0]),
            record(20, scale=1, visible=[1, 0, 0, 0],
                   valid=[1, 0, 0, 0]),
            record(24, scale=2, visible=[1, 0, 0, 0],
                   valid=[1, 0, 0, 0]),
        ]
        training = select_real_examples(DummyDataset(rows), 3)
        validation = select_real_examples(
            DummyDataset(rows), 2, validation=True)
        self.assertEqual(training.indices, [0, 2, 3])
        self.assertEqual(validation.indices, [2, 3])

    def test_reject_insufficient_or_unlabeled_data(self):
        dataset = DummyDataset([record(0, visible=[0, 0, 0, 0],
                                       valid=[0, 0, 0, 0])])
        with self.assertRaisesRegex(RuntimeError, "Only 0 eligible"):
            select_real_examples(dataset, 1, validation=True)
        with self.assertRaises(ValueError):
            select_real_examples(dataset, 0)

    def test_probe_checkpoint_uses_production_atomic_save_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = torch.nn.Linear(3, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 1.)
            scaler = torch.amp.GradScaler("cpu", enabled=False)
            sample = torch.randn(3, 3)
            target = torch.randn(3, 1)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(sample), target)
            loss.backward()
            optimizer.step()
            assert optimizer.state
            impl = types.SimpleNamespace(
                save_atomic=vit_train.save_atomic,
                sha256=vit_train.sha256,
                rng_snapshot=vit_train.rng_snapshot,
                manifest_digest=lambda manifest: "synthetic-probe-test",
                refresh_best_alias=vit_train.refresh_best_alias,
                recover_history=vit_train.recover_history,
            )
            validation = {"val_pck_005": 0.2, "val_landmarks": [1, 1, 1, 1]}
            best, slot, digest = checkpoint_epoch(
                impl, model, optimizer, scheduler, scaler, root,
                {"scenario": "FD"}, 0, -float("inf"), None, None,
                {"train_loss": float(loss.detach())}, validation)
            self.assertAlmostEqual(best, 0.2)
            self.assertEqual(slot, "best_slot_a.pth")
            self.assertTrue((root / "checkpoint_last.pth").exists())
            self.assertEqual(vit_train.sha256(root / "model_best.pth"),
                             digest)
            self.assertEqual(len(
                (root / "training_history.jsonl").read_text().splitlines()
            ), 1)
            state = torch.load(root / "checkpoint_last.pth",
                               map_location="cpu", weights_only=False)
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(state["recipe_hash"], "synthetic-probe-test")
            self.assertTrue(state["optimizer"]["state"])
            impl.recover_history(root, 1, state["metrics"])
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                impl.refresh_best_alias(root, slot, "incorrect-sha256")


if __name__ == "__main__":
    unittest.main()
