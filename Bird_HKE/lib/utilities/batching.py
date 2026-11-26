"""Pure batch-planning helpers with no deep-learning runtime dependency."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchPlan:
    """Resolved batch parameters for the active hardware."""

    devices: int
    batch_size_per_device: int
    global_micro_batch: int
    accumulation_steps: int
    effective_batch_size: int


def resolve_batch_plan(train_cfg: Any, devices: int) -> BatchPlan:
    """Resolve and validate gradient accumulation for a requested batch size."""
    devices = int(devices)
    get_value = (
        train_cfg.get
        if isinstance(train_cfg, dict)
        else lambda key, default=None: getattr(train_cfg, key, default)
    )
    micro = int(get_value('BATCH_SIZE_PER_GPU'))
    target = int(get_value('EFFECTIVE_BATCH_SIZE'))
    configured_accum = int(get_value('GRAD_ACCUM_STEPS', 0))

    if devices < 1:
        raise ValueError('At least one training device is required.')
    if micro < 1 or target < 1:
        raise ValueError('Batch sizes must be positive integers.')

    global_micro = devices * micro
    if configured_accum == 0:
        if target % global_micro != 0:
            raise ValueError(
                'EFFECTIVE_BATCH_SIZE must be divisible by '
                'BATCH_SIZE_PER_GPU * number of devices: '
                f'{target} is not divisible by {micro} * {devices}.'
            )
        accumulation = target // global_micro
    elif configured_accum > 0:
        accumulation = configured_accum
        actual = global_micro * accumulation
        if actual != target:
            raise ValueError(
                f'Configured gradient accumulation produces effective batch '
                f'{actual}, but EFFECTIVE_BATCH_SIZE is {target}.'
            )
    else:
        raise ValueError('GRAD_ACCUM_STEPS must be zero (automatic) or positive.')

    return BatchPlan(
        devices=devices,
        batch_size_per_device=micro,
        global_micro_batch=global_micro,
        accumulation_steps=accumulation,
        effective_batch_size=target,
    )


def accumulation_group_size(batch_index: int, num_batches: int, steps: int) -> int:
    """Return the actual group size, including a possibly shorter final group."""
    if steps < 1:
        raise ValueError('steps must be positive')
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError('batch_index must identify a batch in the loader')
    group_start = (batch_index // steps) * steps
    return min(steps, num_batches - group_start)
