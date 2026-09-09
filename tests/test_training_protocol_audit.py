import importlib.util
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPOSITORY_ROOT / 'Bird_HKE' / 'tools' / 'audit_training_protocol.py'
SPEC = importlib.util.spec_from_file_location('audit_training_protocol', AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TrainingProtocolAuditTests(unittest.TestCase):
    def test_every_experiment_matches_protocol(self):
        experiments = REPOSITORY_ROOT / 'Bird_HKE' / 'experiments'
        paths = list(AUDIT.experiment_files(experiments))
        self.assertEqual(len(paths), 18)
        failures = {}
        for path in paths:
            _config, _plan, problems = AUDIT.audit_file(path)
            if problems:
                failures[str(path.relative_to(REPOSITORY_ROOT))] = problems
        self.assertEqual(failures, {})

    def test_uncertainty_is_enabled_only_for_full_dataset_configs(self):
        experiments = REPOSITORY_ROOT / 'Bird_HKE' / 'experiments'
        counts = {'FD': 0, 'CS': 0, 'OS': 0}
        for path in AUDIT.experiment_files(experiments):
            config, _plan, _problems = AUDIT.audit_file(path)
            scenario = AUDIT.scenario_from_path(path)
            counts[scenario] += 1
            self.assertEqual(
                config['UNCERTAINTY']['ENABLED'],
                scenario == 'FD',
                path.name,
            )
        self.assertEqual(counts, {'FD': 6, 'CS': 6, 'OS': 6})

    def test_supported_gpu_counts_resolve_to_effective_batch_64(self):
        train = {
            'BATCH_SIZE_PER_GPU': 8,
            'EFFECTIVE_BATCH_SIZE': 64,
            'GRAD_ACCUM_STEPS': 0,
        }
        for devices, expected_accumulation in ((1, 8), (2, 4), (4, 2), (8, 1)):
            micro, accumulation, effective = AUDIT._resolve_batch_plan(
                train, devices
            )
            self.assertEqual(micro, 8)
            self.assertEqual(accumulation, expected_accumulation)
            self.assertEqual(effective, 64)

    def test_three_gpu_plan_is_rejected(self):
        train = {
            'BATCH_SIZE_PER_GPU': 8,
            'EFFECTIVE_BATCH_SIZE': 64,
            'GRAD_ACCUM_STEPS': 0,
        }
        with self.assertRaises(ValueError):
            AUDIT._resolve_batch_plan(train, 3)


if __name__ == '__main__':
    unittest.main()
