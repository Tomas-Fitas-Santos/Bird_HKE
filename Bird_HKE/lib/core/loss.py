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


class AuxiliaryPoseUncertaintyLoss(nn.Module):
    """Original pose MSE plus detached auxiliary reliability supervision.

    The pose term is deliberately the exact :class:`JointsMSELoss` used when
    uncertainty is disabled.  The quality and visibility heads receive
    detached pose features (enforced by ``ProbabilisticPoseOutput``), so their
    losses cannot alter the pose heatmaps or backbone gradients.
    """

    def __init__(self, cfg):
        super().__init__()
        uncertainty = cfg.UNCERTAINTY
        self.pose_loss = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        )
        self.quality_pck_threshold = float(
            uncertainty.QUALITY_PCK_THRESHOLD
        )
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

    def forward(self, output, target, target_weight, meta=None):
        if not isinstance(output, dict):
            raise TypeError(
                'AuxiliaryPoseUncertaintyLoss requires the dictionary returned when '
                'UNCERTAINTY.ENABLED is true.'
            )
        location_logits = output['location_logits']
        quality_logits = output['quality_logits']
        visibility_logits = output['visibility_logits']
        _, _, height, width = location_logits.shape

        pose_loss = self.pose_loss(
            location_logits, target, target_weight, meta
        )

        valid = target_weight[..., 0].to(location_logits.device) > 0
        ground_truth = self._ground_truth_coordinates(target)
        predicted_indices = location_logits.detach().flatten(start_dim=2).argmax(dim=-1)
        predicted_x = torch.remainder(predicted_indices, width).to(target.dtype)
        predicted_y = torch.div(
            predicted_indices, width, rounding_mode='floor'
        ).to(target.dtype)
        # Match core.evaluate.accuracy exactly: coordinates are normalized by
        # heatmap_size / 10 and counted correct below the configured PCK
        # threshold (0.5 by default, equivalent to 5% of a square heatmap).
        normalized_x = (predicted_x - ground_truth[..., 0]) / (width / 10.0)
        normalized_y = (predicted_y - ground_truth[..., 1]) / (height / 10.0)
        normalized_distance = torch.sqrt(
            normalized_x.square() + normalized_y.square()
        )
        quality_target = (
            normalized_distance < self.quality_pck_threshold
        ).to(dtype=quality_logits.dtype).detach()
        quality_per_joint = F.binary_cross_entropy_with_logits(
            quality_logits, quality_target, reduction='none'
        )
        quality_loss = _masked_mean(quality_per_joint, valid)

        visibility_loss = visibility_logits.sum() * 0.0
        if meta is not None and 'visibility_known' in meta:
            visibility_known = meta['visibility_known'].to(
                location_logits.device, dtype=location_logits.dtype
            )
            visibility_target = meta['visibility_target'].to(
                location_logits.device, dtype=location_logits.dtype
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
                    known_count = known.sum().to(location_logits.dtype)
                    weights = torch.where(
                        positive,
                        known_count / (2 * positive.sum().to(location_logits.dtype)),
                        known_count / (2 * negative.sum().to(location_logits.dtype)),
                    )
                    visibility_per_joint = visibility_per_joint * weights
            visibility_loss = _masked_mean(visibility_per_joint, known)

        total = (
            pose_loss
            + self.quality_weight * quality_loss
            + self.visibility_weight * visibility_loss
        )
        self.last_components = {
            'pose': float(pose_loss.detach()),
            'quality': float(quality_loss.detach()),
            'visibility': float(visibility_loss.detach()),
        }
        return total


def build_pose_criterion(cfg):
    """Construct the criterion selected by the optional UQ configuration."""
    if bool(cfg.UNCERTAINTY.ENABLED):
        return AuxiliaryPoseUncertaintyLoss(cfg)
    return JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)


# Backward-compatible import name for external code.  The implementation is
# intentionally auxiliary-only despite the historical class name.
ProbabilisticPoseLoss = AuxiliaryPoseUncertaintyLoss
