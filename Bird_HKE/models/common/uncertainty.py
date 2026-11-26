"""Probabilistic output layers shared by every Bird_HKE pose architecture."""

from __future__ import annotations

import torch
import torch.nn as nn


def sparsemax(input_tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection from Martins & Astudillo (2016)."""
    shifted = input_tensor - input_tensor.max(dim=dim, keepdim=True).values
    sorted_values = torch.sort(shifted, dim=dim, descending=True).values
    size = shifted.size(dim)
    ranks_shape = [1] * shifted.ndim
    ranks_shape[dim] = size
    ranks = torch.arange(
        1, size + 1, device=shifted.device, dtype=shifted.dtype
    ).view(ranks_shape)
    cumulative = sorted_values.cumsum(dim)
    support = 1 + ranks * sorted_values > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau_sum = cumulative.gather(dim, support_size - 1)
    tau = (tau_sum - 1) / support_size.to(shifted.dtype)
    return torch.clamp(shifted - tau, min=0)


def spatial_probability(
    logits: torch.Tensor,
    distribution: str = 'softmax',
    temperature: float = 1.0,
) -> torch.Tensor:
    """Normalize each keypoint map into a probability distribution."""
    if logits.ndim != 4:
        raise ValueError('location logits must have shape [B, K, H, W]')
    temperature = max(float(temperature), 1e-6)
    flat = (logits / temperature).flatten(start_dim=2)
    name = str(distribution).lower()
    if name == 'softmax':
        probabilities = torch.softmax(flat, dim=-1)
    elif name == 'sparsemax':
        probabilities = sparsemax(flat, dim=-1)
    else:
        raise ValueError(
            f"Unknown spatial distribution {distribution!r}; "
            "expected 'softmax' or 'sparsemax'."
        )
    return probabilities.reshape_as(logits)


class KeypointReliabilityHead(nn.Module):
    """Estimate per-keypoint localization quality and visibility.

    The head receives both local, probability-weighted image features and the
    global head representation.  A learned joint embedding lets the shared MLP
    model the distinct roles of crown, eyes, and beak without duplicating a
    complete head for each landmark.
    """

    def __init__(
        self,
        in_channels: int,
        num_joints: int,
        hidden_channels: int = 128,
        joint_embed_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.joint_embedding = nn.Embedding(self.num_joints, joint_embed_dim)
        input_dim = 2 * int(in_channels) + int(joint_embed_dim) + 2
        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.quality = nn.Linear(int(hidden_channels), 1)
        self.visibility = nn.Linear(int(hidden_channels), 1)

    def forward(
        self, features: torch.Tensor, probability_maps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if probability_maps.shape[1] != self.num_joints:
            raise ValueError('probability map joint count does not match the head')

        batch_size, channels, _, _ = features.shape
        probabilities = probability_maps.flatten(start_dim=2)
        spatial_features = features.flatten(start_dim=2)
        local = torch.einsum('bkn,bcn->bkc', probabilities, spatial_features)
        global_features = features.mean(dim=(2, 3)).unsqueeze(1).expand(
            -1, self.num_joints, -1
        )

        eps = torch.finfo(probabilities.dtype).eps
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(eps))
        ).sum(dim=-1, keepdim=True)
        entropy = entropy / torch.log(
            torch.tensor(
                probabilities.shape[-1],
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
        )
        peak = probabilities.max(dim=-1, keepdim=True).values

        joint_ids = torch.arange(self.num_joints, device=features.device)
        joint_embedding = self.joint_embedding(joint_ids).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        joint_features = torch.cat(
            (local, global_features, joint_embedding, entropy, peak), dim=-1
        )
        hidden = self.shared(joint_features)
        return self.quality(hidden).squeeze(-1), self.visibility(hidden).squeeze(-1)


class ProbabilisticPoseOutput(nn.Module):
    """Turn shared spatial features and location logits into UQ outputs."""

    def __init__(self, cfg, in_channels: int, num_joints: int) -> None:
        super().__init__()
        uq = cfg['UNCERTAINTY']
        if bool(uq['RELIABILITY_GRADIENT_TO_BACKBONE']):
            raise ValueError(
                'Passive uncertainty requires '
                'RELIABILITY_GRADIENT_TO_BACKBONE: false'
            )
        if float(uq['HEAD_DROPOUT']) != 0.0:
            raise ValueError(
                'Passive uncertainty requires HEAD_DROPOUT: 0.0 so the '
                'auxiliary head cannot consume training RNG between pose steps'
            )
        self.distribution = uq['DISTRIBUTION']
        self.temperature = float(uq['TEMPERATURE'])
        self.reliability = KeypointReliabilityHead(
            in_channels=in_channels,
            num_joints=num_joints,
            hidden_channels=uq['HEAD_HIDDEN_CHANNELS'],
            joint_embed_dim=uq['JOINT_EMBED_DIM'],
            dropout=uq['HEAD_DROPOUT'],
        )

    def forward(
        self, features: torch.Tensor, location_logits: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        probability_maps = spatial_probability(
            location_logits, self.distribution, self.temperature
        )
        quality_logits, visibility_logits = self.reliability(
            features.detach(), probability_maps.detach()
        )
        return {
            'location_logits': location_logits,
            'probability_maps': probability_maps,
            'quality_logits': quality_logits,
            'visibility_logits': visibility_logits,
        }


def build_probabilistic_pose_output(
    cfg, in_channels: int, num_joints: int
) -> ProbabilisticPoseOutput:
    """Build the auxiliary head without advancing the pose-model RNG stream."""
    seed = int(cfg['REPRODUCIBILITY']['SEED']) + 104729
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ProbabilisticPoseOutput(cfg, in_channels, num_joints)
