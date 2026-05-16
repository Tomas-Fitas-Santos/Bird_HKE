from .HRNet import get_pose_net as get_pose_net_hrnet
from .VHR_BirdPose import get_pose_net as get_pose_net_vhr
from .HR_Mamba import get_pose_net as get_pose_net_mamba
from .HR_MambaVision import get_pose_net as get_pose_net_mamba_vision
from .HR_MambaVit import get_pose_net as get_pose_net_mamba_vit


def get_pose_net(cfg, is_train, **kwargs):
	name = str(cfg['MODEL'].get('NAME', '')).lower()

	# HR-MambaViT  (parallel SSM + Attention branch)
	if 'mamba_vit' in name:
		return get_pose_net_mamba_vit(cfg, is_train, **kwargs)

	# HR-Mamba  (pure SSM branch)
	extra = cfg['MODEL'].get('EXTRA', {})
	mamba_cfg = extra.get('MAMBA', None)
	if mamba_cfg is not None and mamba_cfg.get('PURE', False):
		return get_pose_net_mamba(cfg, is_train, **kwargs)

	# HR-MambaVision  (NVIDIA MambaVision adapter branch)
	if mamba_cfg is not None:
		return get_pose_net_mamba_vision(cfg, is_train, **kwargs)

	# VHR-BirdPose
	if 'vhr' in name or 'birdpose' in name:
		return get_pose_net_vhr(cfg, is_train, **kwargs)

	# HRNet (default)
	return get_pose_net_hrnet(cfg, is_train, **kwargs)


__all__ = ['get_pose_net']
