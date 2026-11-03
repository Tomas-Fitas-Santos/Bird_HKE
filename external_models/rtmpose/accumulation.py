"""Pure-Python accumulation plan used by the MMEngine RTMPose wrapper."""
from __future__ import annotations
import math


def accumulation_group(images_per_epoch: int, microbatch_size: int,
                       accumulation_steps: int, microbatch_index: int):
    """Return (group_images, update_now) for a zero-based epoch microbatch."""
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
