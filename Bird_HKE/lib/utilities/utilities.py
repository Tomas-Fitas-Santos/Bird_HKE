# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------
import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn


def create_logger(cfg, cfg_name, phase='train', root_choice='log'):
    """Create logger and tensorboard dirs.
    root_choice: 'log' -> use LOG_DIR (never falls back to OUTPUT_DIR)
                 'output' -> use OUTPUT_DIR (never falls back to LOG_DIR)
                 'auto' -> legacy behavior (LOG_DIR > OUTPUT_DIR > ./output)
    """
    cwd = os.getcwd()
    if root_choice == 'log':
        if cfg.LOG_DIR:
            root_output_dir = Path(cfg.LOG_DIR)
        else:
            root_output_dir = Path(cwd) / 'log'
    elif root_choice == 'output':
        if cfg.OUTPUT_DIR:
            root_output_dir = Path(cfg.OUTPUT_DIR)
        else:
            root_output_dir = Path(cwd) / 'output'
    else:
        # legacy: prefer LOG_DIR, then OUTPUT_DIR
        root_output_dir = Path(cfg.LOG_DIR if cfg.LOG_DIR else (cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else (Path(cwd) / 'output')))
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True, exist_ok=True)

    # Use the root output directory directly (no per-dataset/model subfolders)
    final_output_dir = root_output_dir
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging to console only; do not create a dated log file.
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # No tensorboard logs: return None for tb_log_dir to disable TB output
    return logger, str(final_output_dir), None


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )
    elif cfg.TRAIN.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WD
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth', best_filename='model_best.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, best_filename))


def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            # Some modules (e.g. PatchEmbed) return tuples (tensor, meta).
            # Accept lists or tuples for both input and output and use the
            # primary tensor element for size/flops calculations.
            if isinstance(input[0], (list, tuple)):
                input = input[0]
            if isinstance(output, (list, tuple)):
                output = output[0]

            # Count parameters owned by this module (may be 0); we still
            # compute a global total later from model.parameters() to avoid
            # double-counting across module nesting.
            params = 0
            for param_ in module.parameters(recurse=False):
                params += param_.numel()

            flops = "Not Available"
            # Determine batch size if available
            try:
                batch_size = input[0].size(0)
            except Exception:
                batch_size = 1

            # Convolution FLOPs: batch * out_channels * (in_channels/groups) * kH * kW * out_h * out_w
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                w = module.weight.data.size()
                # w: (out_channels, in_channels, kH, kW)
                out_c = w[0]
                in_c = w[1]
                k_h = w[2] if len(w) > 2 else 1
                k_w = w[3] if len(w) > 3 else 1
                # output spatial dims (H_out, W_out)
                out_spatial = list(output.size())[2:]
                out_h = out_spatial[0] if len(out_spatial) > 0 else 1
                out_w = out_spatial[1] if len(out_spatial) > 1 else 1
                groups = getattr(module, 'groups', 1)
                in_per_group = in_c // groups if groups else in_c
                flops = (batch_size * out_c * in_per_group * k_h * k_w * out_h * out_w)
            elif isinstance(module, nn.Linear):
                # Linear: batch * out_features * in_features
                out_features = output.size(-1)
                in_features = input[0].size(-1)
                flops = (batch_size * out_features * in_features)

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    # Always include a compact header for the summary (no extra leading
    # blank lines). Start `details` with the main header; verbose adds table header.
    details = "Model Summary" + os.linesep
    if verbose:
        details += "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    # Use the canonical total parameter count from the model to avoid
    # double-counting due to nested modules. Per-layer params (in
    # `summary`) remain for verbose output.
    params_sum = sum(p.numel() for p in model.parameters())
    flops_sum = 0
    for layer in summary:
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    # Convert multiply-adds (MACs) to GFLOPs using 1e9 scaling.
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:.6f} GFLOPs".format(flops_sum/1e9) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details