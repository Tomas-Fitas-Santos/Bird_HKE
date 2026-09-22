import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1] / 'Bird_HKE'
sys.path.insert(0, str(PROJECT_ROOT))

from lib.utilities.batching import accumulation_group_size
from lib.utilities.batching import resolve_batch_plan

try:
    import numpy as np
    import torch
    from lib.config.default import _C
    from lib.utilities.reproducibility import capture_rng_state
    from lib.utilities.reproducibility import protocol_hash
    from lib.utilities.reproducibility import restore_rng_state
    from lib.utilities.reproducibility import resume_environment
    from lib.utilities.reproducibility import seed_everything
    from lib.utilities.reproducibility import training_protocol
    from lib.utilities.utilities import atomic_torch_save
    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


class BatchPlanningTests(unittest.TestCase):
    @staticmethod
    def _training_config():
        return SimpleNamespace(
            BATCH_SIZE_PER_GPU=8,
            EFFECTIVE_BATCH_SIZE=64,
            GRAD_ACCUM_STEPS=0,
        )

    def test_automatic_accumulation_preserves_effective_batch(self):
        train_cfg = self._training_config()
        for devices, expected_steps in ((1, 8), (2, 4), (4, 2), (8, 1)):
            plan = resolve_batch_plan(train_cfg, devices)
            self.assertEqual(plan.accumulation_steps, expected_steps)
            self.assertEqual(plan.effective_batch_size, 64)

    def test_incompatible_device_count_is_rejected(self):
        train_cfg = self._training_config()
        with self.assertRaises(ValueError):
            resolve_batch_plan(train_cfg, 3)

    def test_short_final_accumulation_group_uses_its_real_size(self):
        sizes = [accumulation_group_size(i, 10, 4) for i in range(10)]
        self.assertEqual(sizes, [4, 4, 4, 4, 4, 4, 4, 4, 2, 2])


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    'full training dependencies (PyTorch and yacs) are not installed',
)
class RuntimeReproducibilityTests(unittest.TestCase):

    def test_atomic_checkpoint_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pth'
            atomic_torch_save({'epoch': 3}, checkpoint)
            with mock.patch(
                'lib.utilities.utilities.torch.save',
                side_effect=RuntimeError('simulated interrupted write'),
            ):
                with self.assertRaises(RuntimeError):
                    atomic_torch_save({'epoch': 4}, checkpoint)
            try:
                restored = torch.load(checkpoint, weights_only=False)
            except TypeError:
                restored = torch.load(checkpoint)
            self.assertEqual(restored['epoch'], 3)
            temporary_files = [
                path for path in checkpoint.parent.iterdir()
                if path.name.startswith(f'.{checkpoint.name}.')
            ]
            self.assertEqual(temporary_files, [])

    def test_rng_state_restores_all_cpu_sources_and_loader_generator(self):
        seed_everything(17)
        generator = torch.Generator().manual_seed(91)
        state = capture_rng_state(generator)

        expected_python = random.random()
        expected_numpy = np.random.rand()
        expected_torch = torch.rand(1)
        expected_generator = torch.rand(1, generator=generator)

        random.random()
        np.random.rand()
        torch.rand(1)
        torch.rand(1, generator=generator)
        restore_rng_state(state, generator)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))
        self.assertTrue(
            torch.equal(torch.rand(1, generator=generator), expected_generator)
        )

    def test_protocol_hash_does_not_depend_on_dataset_mount_path(self):
        first = _C.clone()
        second = _C.clone()
        first.defrost()
        second.defrost()
        first.DATASET.ROOT = '/machine-a/BirdGaze_v2'
        second.DATASET.ROOT = '/machine-b/BirdGaze_v2'
        first.freeze()
        second.freeze()
        plan = resolve_batch_plan(first.TRAIN, 1)
        self.assertEqual(
            protocol_hash(training_protocol(first, plan)),
            protocol_hash(training_protocol(second, plan)),
        )

    def test_protocol_hash_changes_when_uncertainty_training_changes(self):
        baseline = _C.clone()
        uncertainty = _C.clone()
        uncertainty.defrost()
        uncertainty.UNCERTAINTY.ENABLED = True
        uncertainty.freeze()
        plan = resolve_batch_plan(baseline.TRAIN, 1)
        self.assertNotEqual(
            protocol_hash(training_protocol(baseline, plan)),
            protocol_hash(training_protocol(uncertainty, plan)),
        )

    def test_protocol_hash_changes_when_validation_decoder_changes(self):
        first = _C.clone()
        second = _C.clone()
        second.defrost()
        second.TEST.FLIP_TEST = not first.TEST.FLIP_TEST
        second.freeze()
        plan = resolve_batch_plan(first.TRAIN, 1)
        self.assertNotEqual(
            protocol_hash(training_protocol(first, plan)),
            protocol_hash(training_protocol(second, plan)),
        )

    def test_resume_environment_ignores_report_timestamps(self):
        first = {
            'recorded_at_utc': 'first',
            'git_revision': 'abc',
            'python': '3.x',
            'pytorch': '2.x',
            'cuda_runtime': '12.x',
            'cudnn': 9000,
            'configured_gpu_ids': [0],
            'gpu_names': ['GPU'],
            'package_versions': {'torch': '2.x'},
        }
        second = dict(first, recorded_at_utc='second')
        self.assertEqual(resume_environment(first), resume_environment(second))


if __name__ == '__main__':
    unittest.main()
