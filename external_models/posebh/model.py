"""One-dataset PoseBH-B adaptation with prototype learning.

Reference: Jeong et al. CVPR 2025, pinned upstream PoseBH at
8f03438f913b559edb0a2beed54e77d1560edba3.
Original: ViTMoE (6 dataset experts), main heatmap head, KptProtoHead with
3 prototypes/keypoint, EMA prototypes, PPCE and cross-dataset supervision.
Single BirdGaze dataset only needs one active expert. Cross-dataset clustered
prototypes and cross-dataset CSS are **not** reproduced; instead we use
single-dataset unlabeled-joint consistency after epoch 50. These adaptations
must be disclosed in scientific reporting and are NOT the published benchmark.
No pretrained weights are loaded.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from external_models.vitpose.model import Attention, DropPath, masked_heatmap_mse


class SingleTaskMoEMLP(nn.Module):
    """One active expert from PoseBH's ViTMoE shared 576 + expert 192 FFN."""
    def __init__(self, dim: int, part: int = 192):
        super().__init__()
        if part <= 0 or part >= dim:
            raise ValueError("part must be strictly inside embedding dimension")
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim - part)
        self.experts = nn.ModuleList([nn.Linear(4 * dim, part)])
        self.drop = nn.Dropout(0.)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return torch.cat((self.fc2(x), self.experts[0](x)), dim=-1)


class PoseBHBlock(nn.Module):
    def __init__(self, dim, heads, part, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SingleTaskMoEMLP(dim, part)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def two_deconvolutions(in_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, 256, kernel_size=4,
                           stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 256, kernel_size=4,
                           stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
    )


class ProtoResidualNeck(nn.Module):
    """PoseBH 'res' neck with SiLU, 256-to-64 projection and residual convs."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.downsample = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64),
        )

    def forward(self, x):
        identity = self.downsample(x)
        return self.act(self.bn2(self.conv2(
            self.act(self.bn1(self.conv1(x))))) + identity)


class PoseBHBirdGazeB(nn.Module):
    """PoseBH-B 12-block ViTMoE + auxiliary and prototypical heads for four joints."""
    def __init__(self, dim=768, depth=12, heads=12, part=192,
                 keypoints=4, proto_dim=64, variants=3, drop_path_rate=0.3):
        super().__init__()
        if keypoints != 4 or proto_dim != 64 or variants != 3:
            raise ValueError("BirdGaze PoseBH uses four joints, 64-D, 3 prototypes")
        if dim % heads:
            raise ValueError("dim not divisible by attention heads")
        if dim <= part:
            raise ValueError("expert part must be smaller than hidden dimension")
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, 16, stride=16, padding=2)
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, dim))
        self.blocks = nn.ModuleList([
            PoseBHBlock(dim, heads, part, drop_path_rate * j / max(depth - 1, 1))
            for j in range(depth)
        ])
        self.last_norm = nn.LayerNorm(dim, eps=1e-6)
        self.heatmap_deconvs = two_deconvolutions(dim)
        self.heatmap_head = nn.Conv2d(256, keypoints, 1)
        self.proto_deconvs = two_deconvolutions(dim)
        self.proto_neck = ProtoResidualNeck()
        self.proto_head = nn.Conv2d(64, proto_dim, 3, padding=1)
        # EMA prototypes do not have gradients; retain them in model state.
        raw = torch.empty(keypoints, variants, proto_dim)
        nn.init.trunc_normal_(raw, std=0.02)
        self.register_buffer("prototypes", F.normalize(raw, dim=-1))
        self.keypoints = keypoints
        self.variants = variants
        self.gamma = 0.999
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.heatmap_head.weight, std=0.001)
        nn.init.zeros_(self.heatmap_head.bias)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, 256, 256):
            raise ValueError("PoseBH expects [B,3,256,256]")
        b = image.shape[0]
        x = self.patch_embed.proj(image)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for block in self.blocks:
            x = block(x)
        features = self.last_norm(x).transpose(1, 2).reshape(b, -1, h, w)
        heatmaps = self.heatmap_head(self.heatmap_deconvs(features))
        emb = self.proto_head(self.proto_neck(self.proto_deconvs(features)))
        assert tuple(heatmaps.shape[-2:]) == (64, 64)
        return heatmaps, emb

    def prototype_heatmaps(self, embedding):
        """Upstream KptPrototype uses abs cosine similarities, max over variants."""
        with torch.autocast(device_type=embedding.device.type, enabled=False):
            normalized = F.normalize(embedding.float(), dim=1)
            prototypes = F.normalize(self.prototypes.float(), dim=-1)
            sim = torch.einsum("bchw,kvc->bkvhw", normalized, prototypes).abs()
            return sim.amax(dim=2).clamp(1e-4, 1 - 1e-4)

    def sampled_prototypes(self, embedding, targets):
        """Select positive GT-heatmap pixels, as upstream KptProtoHead.hm_sample."""
        if embedding.shape[2:] != targets.shape[2:]:
            raise ValueError("Embedding and targets must have matching H,W")
        confidences, classes = targets.detach().float().max(dim=1)
        active = confidences > 0
        if not active.any():
            return None
        features = embedding.permute(0, 2, 3, 1)[active]
        return F.normalize(features.float(), dim=-1), classes[active], confidences[active]

    @staticmethod
    def sinkhorn_assign(score, iterations=3, epsilon=0.05):
        """Numerically stable, per-keypoint 3-way balanced soft assignment."""
        if score.numel() == 0:
            raise ValueError("Cannot assign empty prototype samples")
        q = torch.softmax(score.float() / epsilon, dim=1).T
        k, n = q.shape
        q /= q.sum().clamp_min(1e-8)
        for _ in range(iterations):
            q /= q.sum(dim=1, keepdim=True).clamp_min(1e-8)
            q /= k
            q /= q.sum(dim=0, keepdim=True).clamp_min(1e-8)
            q /= n
        return q.T

    def prototype_contrast_loss(self, embedding, targets):
        """PPC+PPD on within-landmark 3-prototype assignments.

        Upstream uses Sinkhorn + Gumbel, PPC/PPD weights and tiny global
        multiplication. Here assignments are deterministic Sinkhorn hard argmax
        so strict random-seed reproducibility is easier to audit.
        """
        sampled = self.sampled_prototypes(embedding, targets)
        if sampled is None:
            return embedding.sum() * 0.
        feats, classes, confidences = sampled
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        logits = (feats @ prototypes.reshape(12, 64).T).abs()
        labels = torch.full((len(classes),), -1, device=classes.device, dtype=torch.long)
        for k in range(4):
            mask = classes == k
            if not mask.any():
                continue
            # Three within-joint prototypes compete for each positive pixel.
            assignment = self.sinkhorn_assign(
                logits[mask, k * 3:(k + 1) * 3].detach())
            labels[mask] = assignment.argmax(dim=1) + k * 3
        valid = labels >= 0
        if not valid.any():
            return embedding.sum() * 0.
        ppc = F.cross_entropy(logits[valid], labels[valid])
        ppd = (1. - logits[valid].gather(1, labels[valid, None]).squeeze(1)).square().mean()
        return 1.25e-5 * (0.01 * ppc + 0.001 * ppd)

    @torch.no_grad()
    def update_prototypes(self, embedding, targets):
        """EMA update after backward; changes only this model's buffers."""
        sampled = self.sampled_prototypes(embedding.detach(), targets.detach())
        if sampled is None:
            return
        feats, classes, conf = sampled
        old = F.normalize(self.prototypes.float(), dim=-1)
        new = old.clone()
        sim = (feats @ old.reshape(12, 64).T).abs()
        predicted_class = sim.reshape(-1, 4, 3).amax(dim=-1).argmax(dim=-1)
        for k in range(4):
            selection = (classes == k) & (predicted_class == k)
            if not selection.any():
                continue
            assignments = self.sinkhorn_assign(
                sim[selection, k * 3:(k + 1) * 3])
            picked = assignments.argmax(dim=-1)
            for variant in range(3):
                chosen = picked == variant
                if chosen.any():
                    weight = conf[selection][chosen][:, None]
                    mean = F.normalize(
                        (feats[selection][chosen] * weight).sum(dim=0),
                        dim=0)
                    new[k, variant] = (
                        self.gamma * old[k, variant] +
                        (1 - self.gamma) * mean)
        self.prototypes.copy_(F.normalize(new, dim=-1))

    def loss_parts(self, heatmaps, embedding, targets, weights, epoch):
        prt_heatmaps = self.prototype_heatmaps(embedding)
        supervised = masked_heatmap_mse(heatmaps.float(), targets, weights)
        proto_heatmap = masked_heatmap_mse(prt_heatmaps, targets, weights)
        contrast = self.prototype_contrast_loss(embedding, targets)
        css = heatmaps.sum() * 0.
        if epoch >= 50:
            # Published PoseBH CSS uses *cross-dataset* pseudo-keypoints. In a
            # single BirdGaze task, agree-on-location pseudo-labels are only
            # used for missing coordinates. This is an explicit adaptation.
            missing = weights <= 0
            if missing.any():
                a = heatmaps.detach().float().flatten(2)
                b = prt_heatmaps.detach().float().flatten(2)
                p = a.argmax(dim=-1)
                q = b.argmax(dim=-1)
                delta = ((p % 64 - q % 64).float().square()
                         + (p // 64 - q // 64).float().square()).sqrt()
                valid = (missing & (delta <= 2.1)
                         & (a.max(dim=-1).values >= 0.25)
                         & (b.max(dim=-1).values >= 0.25))
                if valid.any():
                    # Teacher heatmap detached to avoid self-gradient leakage.
                    pseudo = prt_heatmaps.detach()
                    css = masked_heatmap_mse(heatmaps.float(), pseudo, valid.float())
        total = supervised + 3.33e-6 * proto_heatmap + contrast + 1.0e-3 * css
        return total, {"supervised": supervised, "prototype_mse": proto_heatmap,
                       "prototype_contrast": contrast, "css": css}
