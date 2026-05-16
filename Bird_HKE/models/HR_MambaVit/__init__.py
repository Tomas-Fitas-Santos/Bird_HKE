"""HR-MambaViT model (HRNet + parallel SSM + Attention branch)."""

from .pose_hr_mamba_vit import get_pose_net

__all__ = ['get_pose_net']
