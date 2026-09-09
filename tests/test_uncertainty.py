import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1] / 'Bird_HKE'
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from lib.config.default import _C
    from lib.core.loss import ProbabilisticPoseLoss
    from lib.core.uncertainty import ensemble_uncertainty
    from lib.core.uncertainty import fit_hpd_mass_thresholds
    from lib.core.uncertainty import hpd_region
    from lib.core.uncertainty import probability_map_moments
    from models.common.uncertainty import ProbabilisticPoseOutput
    from models.common.uncertainty import spatial_probability
    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    'full uncertainty dependencies (PyTorch and yacs) are not installed',
)
class UncertaintyModelTests(unittest.TestCase):
    def test_spatial_distributions_are_normalized_and_nonnegative(self):
        logits = torch.randn(2, 4, 7, 9)
        for distribution in ('softmax', 'sparsemax'):
            probability = spatial_probability(logits, distribution)
            self.assertTrue(torch.all(probability >= 0))
            totals = probability.sum(dim=(2, 3))
            self.assertTrue(torch.allclose(totals, torch.ones_like(totals), atol=1e-5))

    def test_reliability_head_does_not_change_backbone_by_default(self):
        config = _C.clone()
        config.defrost()
        config.UNCERTAINTY.ENABLED = True
        config.freeze()
        module = ProbabilisticPoseOutput(config, in_channels=8, num_joints=4)
        features = torch.randn(2, 8, 6, 6, requires_grad=True)
        logits = torch.randn(2, 4, 6, 6, requires_grad=True)
        output = module(features, logits)
        (output['quality_logits'].sum() + output['visibility_logits'].sum()).backward()
        self.assertIsNone(features.grad)
        self.assertIsNone(logits.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in module.parameters()))

    def test_probabilistic_loss_is_finite_and_masks_unknown_visibility(self):
        config = _C.clone()
        config.defrost()
        config.UNCERTAINTY.ENABLED = True
        config.freeze()
        criterion = ProbabilisticPoseLoss(config)
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        quality = torch.randn(2, 4, requires_grad=True)
        visibility = torch.randn(2, 4, requires_grad=True)
        output = {
            'location_logits': logits,
            'probability_maps': torch.softmax(logits.flatten(2), dim=-1).reshape_as(logits),
            'quality_logits': quality,
            'visibility_logits': visibility,
        }
        target = torch.zeros(2, 4, 8, 8)
        target[:, :, 3, 5] = 1
        target_weight = torch.ones(2, 4, 1)
        target_weight[0, 3] = 0
        meta = {
            'visibility_known': torch.ones(2, 4, 1),
            'visibility_target': torch.ones(2, 4, 1),
        }
        meta['visibility_known'][1] = 0  # eBird-style unknown visibility
        loss = criterion(output, target, target_weight, meta)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(quality.grad)
        self.assertIsNotNone(visibility.grad)
        self.assertEqual(
            set(criterion.last_components),
            {'location', 'smoothness', 'quality', 'visibility'},
        )


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    'full uncertainty dependencies (PyTorch and yacs) are not installed',
)
class UncertaintyCalibrationTests(unittest.TestCase):
    def test_probability_moments_have_positive_semidefinite_covariance(self):
        probability = np.zeros((1, 1, 5, 5), dtype=np.float64)
        probability[0, 0, 2, 1] = 0.5
        probability[0, 0, 2, 3] = 0.5
        stats = probability_map_moments(probability)
        self.assertTrue(np.allclose(stats['mean'][0, 0], [2, 2]))
        eigenvalues = np.linalg.eigvalsh(stats['covariance'][0, 0])
        self.assertTrue(np.all(eigenvalues >= -1e-12))

    def test_ensemble_total_is_aleatoric_plus_epistemic(self):
        members = np.zeros((2, 1, 1, 5, 5), dtype=np.float64)
        members[0, 0, 0, 2, 1] = 1
        members[1, 0, 0, 2, 3] = 1
        result = ensemble_uncertainty(members)
        self.assertTrue(
            np.allclose(
                result['total_covariance'],
                result['aleatoric_covariance'] + result['epistemic_covariance'],
            )
        )
        self.assertGreater(result['epistemic_covariance'][0, 0, 0, 0], 0)

    def test_conformal_hpd_threshold_contains_calibration_targets(self):
        probability = np.full((4, 1, 3, 3), 0.025, dtype=np.float64)
        probability[:, 0, 1, 1] = 0.8
        coordinates = np.tile(np.array([[[1, 1]]]), (4, 1, 1))
        valid = np.ones((4, 1), dtype=bool)
        thresholds, counts = fit_hpd_mass_thresholds(
            probability, coordinates, valid, coverage=0.75
        )
        self.assertEqual(counts.tolist(), [4])
        region = hpd_region(probability[0, 0], thresholds[0])
        self.assertTrue(region[1, 1])


if __name__ == '__main__':
    unittest.main()
