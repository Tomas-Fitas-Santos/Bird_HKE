"""RTMPose-M BirdGaze epoch-aligned, per-image gradient accumulation.

Upstream MMEngine OptimWrapper accumulates by global iteration index and
carries partial accumulation across epoch boundaries. Bird_HKE instead
flushes at every epoch boundary and normalizes the last smaller batch group
by its actual sample count. RTMCCHead's KLDiscretLoss returns a sum over
image samples / joints, so dividing by the count of images in the optimizer
group yields a sample-mean gradient consistent across full/partial groups.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper
from mmengine.registry import HOOKS, OPTIM_WRAPPERS


def accumulation_group(images_per_epoch: int, microbatch_size: int,
                       accumulation_steps: int, microbatch_index: int):
    """Return (group_images, update_now) for a zero-based epoch microbatch.

    The final group can have fewer than 64 samples; the last microbatch
    itself can have fewer than 8 images.
    """
    if images_per_epoch < 1 or microbatch_size < 1 or accumulation_steps < 1:
        raise ValueError("Invalid batch plan")
    epoch_iters = math.ceil(images_per_epoch / microbatch_size)
    if not 0 <= microbatch_index < epoch_iters:
        raise ValueError("Invalid microbatch index")
    group_start = (microbatch_index // accumulation_steps) * accumulation_steps
    group_stop = min(group_start + accumulation_steps, epoch_iters)
    group_start_image = group_start * microbatch_size
    group_stop_image = min(group_stop * microbatch_size, images_per_epoch)
    update_now = microbatch_index + 1 == group_stop
    return group_stop_image - group_start_image, update_now


@OPTIM_WRAPPERS.register_module()
class BirdGazeEpochOptimWrapper(OptimWrapper):
    """One optimizer step per full/partial group, never across epochs."""

    def __init__(self, *, images_per_epoch: int, microbatch_size: int = 8,
                 accumulative_counts: int = 8, **kwargs):
        self.images_per_epoch = int(images_per_epoch)
        self.microbatch_size = int(microbatch_size)
        self.epoch_iters = math.ceil(self.images_per_epoch / self.microbatch_size)
        self._group_samples = 0
        super().__init__(accumulative_counts=accumulative_counts, **kwargs)
        if self.images_per_epoch < 1:
            raise ValueError("Empty BirdGaze training dataset")

    def initialize_count_status(self, model, init_counts, max_counts):
        if init_counts % self.epoch_iters:
            raise RuntimeError(
                "Only epoch-boundary checkpoint resume is supported; "
                "optimizer partial groups cannot be silently discarded")
        if max_counts % self.epoch_iters:
            raise RuntimeError("Training iteration limit must end at an epoch boundary")
        super().initialize_count_status(model, init_counts, max_counts)

    def should_update(self):
        if self._inner_count == 0:
            return False
        index = (self._inner_count - 1) % self.epoch_iters
        _, update = accumulation_group(
            self.images_per_epoch, self.microbatch_size,
            self._accumulative_counts, index)
        return update

    def should_sync(self):
        index = self._inner_count % self.epoch_iters
        _, update = accumulation_group(
            self.images_per_epoch, self.microbatch_size,
            self._accumulative_counts, index)
        return update

    def scale_loss(self, loss):
        index = self._inner_count % self.epoch_iters
        group_images, _ = accumulation_group(
            self.images_per_epoch, self.microbatch_size,
            self._accumulative_counts, index)
        return loss / group_images


@HOOKS.register_module()
class BirdGazeRNGCheckpointHook(Hook):
    """Save/restore Python, NumPy, CPU and all CUDA RNG streams.

    With nonpersistent dataloader workers and epoch-boundary checkpoints,
    worker initialization is repeated each epoch. Determinism across distinct
    GPU architectures is still not guaranteed by CUDA.
    """

    priority = "VERY_HIGH"

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint["birdgaze_rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": [x.cpu() for x in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else [],
        }

    def after_load_checkpoint(self, runner, checkpoint):
        if not runner.cfg.get("resume", False):
            return
        state = checkpoint.get("birdgaze_rng_state")
        if state is None:
            raise RuntimeError(
                "Refusing resume: checkpoint has no BirdGaze RNG state")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        if torch.cuda.is_available():
            current = torch.cuda.device_count()
            saved = state["torch_cuda"]
            if len(saved) != current:
                raise RuntimeError(
                    "Refusing resume: CUDA device count changed "
                    f"({len(saved)} saved vs {current} available)")
            torch.cuda.set_rng_state_all(saved)

    def before_run(self, runner):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)
