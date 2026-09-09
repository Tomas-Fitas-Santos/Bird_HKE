import torch
import torch.nn as nn
import torch.nn.functional as F


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, lambda_weight=0.01, gamma=2.0):
        super(JointsMSELoss, self).__init__()
        self.use_target_weight = use_target_weight
        self.lambda_weight = lambda_weight
        self.gamma = gamma

    def forward(self, output, target, target_weight, meta=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            heatmap_gt_clamped = heatmap_gt.clamp(0.0, 1.0)
            weight = self.lambda_weight + (1.0 - self.lambda_weight) * (heatmap_gt_clamped ** self.gamma)
            if self.use_target_weight:
                weight = weight * target_weight[:, idx]
            diff = (heatmap_pred - heatmap_gt) ** 2
            loss += 0.5 * (diff * weight).mean()

        return loss / num_joints



class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight, meta=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class ProbabilisticPoseLoss(nn.Module):
    """ProbPose-style spatial risk plus quality and visibility supervision."""

    def __init__(self, cfg):
        super().__init__()
        uncertainty = cfg.UNCERTAINTY
        self.sigma_fraction = float(uncertainty.BKS_SIGMA_FRACTION)
        self.location_weight = float(uncertainty.LOCATION_WEIGHT)
        self.smoothness_weight = float(uncertainty.SMOOTHNESS_WEIGHT)
        self.quality_weight = float(uncertainty.QUALITY_WEIGHT)
        self.visibility_weight = float(uncertainty.VISIBILITY_WEIGHT)
        self.balance_visibility = bool(uncertainty.BALANCE_VISIBILITY_CLASSES)
        self.last_components = {}

    @staticmethod
    def _ground_truth_coordinates(target):
        batch_size, num_joints, _, width = target.shape
        indices = target.flatten(start_dim=2).argmax(dim=-1)
        x = torch.remainder(indices, width)
        y = torch.div(indices, width, rounding_mode='floor')
        return torch.stack((x, y), dim=-1).to(dtype=target.dtype)

    @staticmethod
    def _expected_similarity_maps(probability_maps, sigma):
        radius = max(1, int(round(3 * sigma)))
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=probability_maps.device,
            dtype=probability_maps.dtype,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
        kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        batch_size, num_joints, height, width = probability_maps.shape
        values = F.conv2d(
            probability_maps.reshape(batch_size * num_joints, 1, height, width),
            kernel.reshape(1, 1, *kernel.shape),
            padding=radius,
        )
        return values.reshape(batch_size, num_joints, height, width)

    def forward(self, output, target, target_weight, meta=None):
        if not isinstance(output, dict):
            raise TypeError(
                'ProbabilisticPoseLoss requires the dictionary returned when '
                'UNCERTAINTY.ENABLED is true.'
            )
        probability_maps = output['probability_maps']
        quality_logits = output['quality_logits']
        visibility_logits = output['visibility_logits']
        batch_size, num_joints, height, width = probability_maps.shape

        valid = target_weight[..., 0].to(probability_maps.device) > 0
        ground_truth = self._ground_truth_coordinates(target)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=probability_maps.device, dtype=probability_maps.dtype),
            torch.arange(width, device=probability_maps.device, dtype=probability_maps.dtype),
            indexing='ij',
        )
        dx = xx.view(1, 1, height, width) - ground_truth[..., 0, None, None]
        dy = yy.view(1, 1, height, width) - ground_truth[..., 1, None, None]
        sigma = max(self.sigma_fraction * min(height, width), 1e-6)
        similarity = torch.exp(-(dx.square() + dy.square()) / (2 * sigma ** 2))

        expected_risk = (probability_maps * (1.0 - similarity)).sum(dim=(2, 3))
        location_loss = _masked_mean(expected_risk, valid)

        horizontal_tv = (probability_maps[:, :, :, 1:] - probability_maps[:, :, :, :-1]).abs().mean()
        vertical_tv = (probability_maps[:, :, 1:, :] - probability_maps[:, :, :-1, :]).abs().mean()
        smoothness_loss = horizontal_tv + vertical_tv

        decoded_maps = self._expected_similarity_maps(probability_maps, sigma)
        predicted_indices = decoded_maps.flatten(start_dim=2).argmax(dim=-1)
        predicted_x = torch.remainder(predicted_indices, width).to(target.dtype)
        predicted_y = torch.div(
            predicted_indices, width, rounding_mode='floor'
        ).to(target.dtype)
        squared_error = (
            (predicted_x - ground_truth[..., 0]).square()
            + (predicted_y - ground_truth[..., 1]).square()
        )
        quality_target = torch.exp(-squared_error / (2 * sigma ** 2)).detach()
        quality_per_joint = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, reduction='none'
        )
        quality_loss = _masked_mean(quality_per_joint, valid)

        visibility_loss = probability_maps.sum() * 0.0
        if meta is not None and 'visibility_known' in meta:
            visibility_known = meta['visibility_known'].to(
                probability_maps.device, dtype=probability_maps.dtype
            )
            visibility_target = meta['visibility_target'].to(
                probability_maps.device, dtype=probability_maps.dtype
            )
            if visibility_known.ndim == 3:
                visibility_known = visibility_known[..., 0]
            if visibility_target.ndim == 3:
                visibility_target = visibility_target[..., 0]
            visibility_per_joint = F.binary_cross_entropy_with_logits(
                visibility_logits, visibility_target, reduction='none'
            )
            known = visibility_known > 0
            if self.balance_visibility and bool(known.any()):
                positive = known & (visibility_target > 0.5)
                negative = known & ~positive
                if bool(positive.any()) and bool(negative.any()):
                    known_count = known.sum().to(probability_maps.dtype)
                    weights = torch.where(
                        positive,
                        known_count / (2 * positive.sum().to(probability_maps.dtype)),
                        known_count / (2 * negative.sum().to(probability_maps.dtype)),
                    )
                    visibility_per_joint = visibility_per_joint * weights
            visibility_loss = _masked_mean(visibility_per_joint, known)

        total = (
            self.location_weight * location_loss
            + self.smoothness_weight * smoothness_loss
            + self.quality_weight * quality_loss
            + self.visibility_weight * visibility_loss
        )
        self.last_components = {
            'location': float(location_loss.detach()),
            'smoothness': float(smoothness_loss.detach()),
            'quality': float(quality_loss.detach()),
            'visibility': float(visibility_loss.detach()),
        }
        return total


def build_pose_criterion(cfg):
    """Construct the criterion selected by the optional UQ configuration."""
    if bool(cfg.UNCERTAINTY.ENABLED):
        return ProbabilisticPoseLoss(cfg)
    return JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
