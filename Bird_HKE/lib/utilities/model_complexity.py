"""Consistent parameter and inference-compute accounting for Bird_HKE models.

The profiler uses the shapes observed during one real forward pass.  This is
important for the transformer branches, where an ``nn.Linear`` is applied to
every token, and for MambaVision, where attention is applied inside windows.

Conventions
-----------
* A multiply-accumulate (MAC) is reported as one operation in ``GMACs``.
* ``GFLOPs`` uses the hardware-oriented convention that one multiplication
  plus one addition is two floating-point operations, hence GFLOPs = 2*GMACs.
* Convolution, transposed convolution, linear projection, attention matrix
  products, Mamba selective scan, and the uncertainty-head spatial reduction
  are included.
* Bias addition, normalization, activation, softmax/sparsemax, interpolation,
  tensor rearrangement, and residual element-wise operations are excluded, as
  is conventional in architecture-complexity tables.

The selective-scan term follows the common Mamba/VMamba convention from
https://github.com/state-spaces/mamba/issues/110: 9*B*L*D*N, plus the D skip
and (where applicable) z gate terms.  Mamba projections and causal
convolutions are counted separately from their actual registered weights.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn


COUNTING_CONVENTION = (
    'Inference compute for one forward pass. GMACs count one multiply-add as '
    'one operation; GFLOPs count the multiplication and addition separately '
    '(GFLOPs = 2 x GMACs). Biases, normalization, activations, probability '
    'normalization, interpolation, tensor rearrangement, and residual adds '
    'are excluded.'
)

INCLUDED_OPERATIONS = (
    'convolution',
    'transposed_convolution',
    'linear_projection',
    'attention_qk_av',
    'mamba_projection',
    'mamba_causal_convolution',
    'mamba_selective_scan',
    'uncertainty_spatial_pooling',
)


def _product(values: Iterable[int]) -> int:
    return int(reduce(mul, (int(value) for value in values), 1))


def _primary_tensor(value: Any):
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for key in ('probability_maps', 'location_logits'):
            if key in value:
                tensor = _primary_tensor(value[key])
                if tensor is not None:
                    return tensor
        for nested in value.values():
            tensor = _primary_tensor(nested)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for nested in value:
            tensor = _primary_tensor(nested)
            if tensor is not None:
                return tensor
    return None


def _is_atomic_mamba(module: nn.Module) -> bool:
    class_name = module.__class__.__name__
    module_name = module.__class__.__module__
    if class_name == 'MambaVisionMixer':
        return True
    return (
        class_name in {'Mamba', 'Mamba2'}
        and module_name.startswith('mamba_ssm')
        and hasattr(module, 'A_log')
    )


def _is_attention(module: nn.Module) -> bool:
    if not hasattr(module, 'num_heads'):
        return False
    has_self_attention = hasattr(module, 'qkv') and hasattr(module, 'proj')
    has_cross_attention = all(
        hasattr(module, name) for name in ('wq', 'wk', 'wv', 'proj')
    )
    return has_self_attention or has_cross_attention


def _linear_weight_macs(module: nn.Module, tokens: int) -> int:
    weight = getattr(module, 'weight', None)
    return int(tokens * weight.numel()) if weight is not None else 0


def _mamba_operation_counts(
    module: nn.Module, input_tensor: torch.Tensor
) -> Dict[str, int]:
    if input_tensor.ndim != 3:
        raise ValueError(
            f'{module.__class__.__name__} input must have shape [B, L, D], '
            f'found {tuple(input_tensor.shape)}'
        )

    batch, sequence_length, _ = (int(value) for value in input_tensor.shape)
    tokens = batch * sequence_length

    projection_macs = 0
    for name in ('in_proj', 'x_proj', 'dt_proj', 'out_proj'):
        projection_macs += _linear_weight_macs(
            getattr(module, name, None), tokens
        )

    convolution_macs = 0
    for name in ('conv1d', 'conv1d_x', 'conv1d_z'):
        convolution = getattr(module, name, None)
        weight = getattr(convolution, 'weight', None)
        if weight is not None:
            # All Bird_HKE Mamba convolutions preserve sequence length.
            convolution_macs += batch * sequence_length * int(weight.numel())

    state_matrix = getattr(module, 'A_log', None)
    if state_matrix is None or state_matrix.ndim != 2:
        raise ValueError(
            f'{module.__class__.__name__} does not expose a two-dimensional '
            'A_log state matrix required for selective-scan accounting'
        )
    scan_channels, state_size = (int(value) for value in state_matrix.shape)
    scan_macs = 9 * batch * sequence_length * scan_channels * state_size

    # Every implementation used here has the learned D skip connection.
    if getattr(module, 'D', None) is not None:
        scan_macs += batch * sequence_length * scan_channels

    # mamba_ssm.Mamba passes z to selective_scan; MambaVisionMixer does not.
    if module.__class__.__name__ != 'MambaVisionMixer':
        scan_macs += batch * sequence_length * scan_channels

    return {
        'mamba_projection': projection_macs,
        'mamba_causal_convolution': convolution_macs,
        'mamba_selective_scan': scan_macs,
    }


@dataclass(frozen=True)
class ModelComplexity:
    model_class: str
    input_shapes: Tuple[Tuple[int, ...], ...]
    total_parameters: int
    trainable_parameters: int
    uncertainty_parameters: int
    macs_by_operation: Mapping[str, int]

    @property
    def non_trainable_parameters(self) -> int:
        return self.total_parameters - self.trainable_parameters

    @property
    def total_macs(self) -> int:
        return int(sum(self.macs_by_operation.values()))

    @property
    def gmacs(self) -> float:
        return self.total_macs / 1e9

    @property
    def gflops(self) -> float:
        return 2.0 * self.gmacs

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': 1,
            'model_class': self.model_class,
            'input_shapes': [list(shape) for shape in self.input_shapes],
            'batch_size': self.input_shapes[0][0] if self.input_shapes else None,
            'parameters': {
                'total': self.total_parameters,
                'trainable': self.trainable_parameters,
                'non_trainable': self.non_trainable_parameters,
                'uncertainty_head': self.uncertainty_parameters,
                'total_millions': round(self.total_parameters / 1e6, 6),
                'trainable_millions': round(
                    self.trainable_parameters / 1e6, 6
                ),
            },
            'compute': {
                'macs': self.total_macs,
                'gmacs': round(self.gmacs, 6),
                'flops': 2 * self.total_macs,
                'gflops': round(self.gflops, 6),
                'macs_by_operation': {
                    key: int(value)
                    for key, value in sorted(self.macs_by_operation.items())
                },
            },
            # Compatibility fields consumed by run_all.py.  params_M now
            # explicitly means trainable parameters, as used in the report.
            'params_M': round(self.trainable_parameters / 1e6, 6),
            'total_params_M': round(self.total_parameters / 1e6, 6),
            'gmacs': round(self.gmacs, 6),
            'gflops': round(self.gflops, 6),
            'counting_convention': COUNTING_CONVENTION,
            'included_operations': list(INCLUDED_OPERATIONS),
        }

    def format(self, verbose: bool = False) -> str:
        lines = [
            'Model Complexity',
            f'Model class: {self.model_class}',
            'Input shape(s): ' + ', '.join(str(shape) for shape in self.input_shapes),
            f'Total parameters: {self.total_parameters:,} '
            f'({self.total_parameters / 1e6:.6f} M)',
            f'Trainable parameters: {self.trainable_parameters:,} '
            f'({self.trainable_parameters / 1e6:.6f} M)',
            f'Non-trainable parameters: {self.non_trainable_parameters:,}',
            f'Uncertainty-head parameters: {self.uncertainty_parameters:,}',
            f'Inference compute: {self.gmacs:.6f} GMACs',
            f'Inference compute: {self.gflops:.6f} GFLOPs '
            '(multiply and add counted separately)',
            f'Convention: {COUNTING_CONVENTION}',
        ]
        if verbose:
            lines.append('MAC breakdown:')
            for name, value in sorted(self.macs_by_operation.items()):
                lines.append(
                    f'  {name}: {value:,} MACs ({value / 1e9:.6f} GMACs)'
                )
        return os.linesep.join(lines)


def profile_model_complexity(
    model: nn.Module, *input_tensors: torch.Tensor
) -> ModelComplexity:
    """Profile one inference forward pass using observed tensor shapes."""
    if not input_tensors:
        raise ValueError('at least one input tensor is required')

    named_modules = list(model.named_modules())
    atomic_mamba = [module for _, module in named_modules if _is_atomic_mamba(module)]
    atomic_descendants = set()
    for root in atomic_mamba:
        atomic_descendants.update(
            id(child) for child in root.modules() if child is not root
        )

    counts: Dict[str, int] = defaultdict(int)
    hooks = []

    def add_count(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f'negative operation count for {name}: {value}')
        counts[name] += int(value)

    def convolution_hook(module, inputs, output):
        input_tensor = _primary_tensor(inputs)
        output_tensor = _primary_tensor(output)
        if input_tensor is None or output_tensor is None:
            return
        kernel_volume = _product(module.kernel_size)
        if isinstance(
            module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            spatial_input = _product(input_tensor.shape[2:])
            macs = (
                int(input_tensor.shape[0])
                * int(module.in_channels)
                * spatial_input
                * (int(module.out_channels) // int(module.groups))
                * kernel_volume
            )
            add_count('transposed_convolution', macs)
        else:
            # weight.shape[1] is already in_channels/groups.  Dividing it by
            # groups again was the grouped-convolution bug in the old code.
            macs_per_output = _product(module.weight.shape[1:])
            add_count('convolution', output_tensor.numel() * macs_per_output)

    def linear_hook(module, inputs, output):
        output_tensor = _primary_tensor(output)
        if output_tensor is None:
            return
        # output.numel()/out_features includes batch, token and spatial axes.
        output_vectors = output_tensor.numel() // int(module.out_features)
        add_count(
            'linear_projection',
            output_vectors * int(module.in_features) * int(module.out_features),
        )

    def attention_hook(module, inputs, output):
        input_tensor = _primary_tensor(inputs)
        output_tensor = _primary_tensor(output)
        if input_tensor is None or output_tensor is None or input_tensor.ndim != 3:
            return
        batch = int(input_tensor.shape[0])
        key_length = int(input_tensor.shape[-2])
        query_length = (
            int(output_tensor.shape[-2])
            if output_tensor.ndim == 3
            else key_length
        )
        heads = int(module.num_heads)
        if hasattr(module, 'qkv'):
            projected = int(module.qkv.out_features) // 3
            head_dim = projected // heads
            query_length = key_length
        else:
            head_dim = int(module.wq.out_features) // heads
        # QK^T and attention*V each require the same number of MACs.
        add_count(
            'attention_qk_av',
            2 * batch * heads * query_length * key_length * head_dim,
        )

    def mamba_hook(module, inputs, _output):
        input_tensor = _primary_tensor(inputs)
        if input_tensor is None:
            return
        for name, value in _mamba_operation_counts(module, input_tensor).items():
            add_count(name, value)

    def uncertainty_hook(_module, inputs, _output):
        if len(inputs) < 2:
            return
        features = _primary_tensor(inputs[0])
        probabilities = _primary_tensor(inputs[1])
        if features is None or probabilities is None:
            return
        if features.ndim != 4 or probabilities.ndim != 4:
            return
        batch, channels, height, width = (int(value) for value in features.shape)
        joints = int(probabilities.shape[1])
        add_count(
            'uncertainty_spatial_pooling',
            batch * joints * channels * height * width,
        )

    convolution_types = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )

    for _, module in named_modules:
        if id(module) in atomic_descendants:
            continue
        if _is_atomic_mamba(module):
            hooks.append(module.register_forward_hook(mamba_hook))
        elif isinstance(module, convolution_types):
            hooks.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

        if _is_attention(module):
            hooks.append(module.register_forward_hook(attention_hook))
        if module.__class__.__name__ == 'KeypointReliabilityHead':
            hooks.append(module.register_forward_hook(uncertainty_hook))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(*input_tensors)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    uncertainty_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith('probabilistic_output.')
    )

    return ModelComplexity(
        model_class=model.__class__.__name__,
        input_shapes=tuple(tuple(int(value) for value in item.shape)
                           for item in input_tensors),
        total_parameters=int(total_parameters),
        trainable_parameters=int(trainable_parameters),
        uncertainty_parameters=int(uncertainty_parameters),
        macs_by_operation=dict(counts),
    )


def write_model_complexity_json(
    path: os.PathLike[str] | str,
    complexity: ModelComplexity,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    payload = complexity.to_dict()
    if metadata:
        payload['metadata'] = dict(metadata)
    destination = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    temporary = destination + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


__all__ = [
    'COUNTING_CONVENTION',
    'INCLUDED_OPERATIONS',
    'ModelComplexity',
    'profile_model_complexity',
    'write_model_complexity_json',
]
