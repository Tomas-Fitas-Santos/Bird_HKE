import time
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.evaluate import accuracy
from core.inference import get_pose_output_preds, get_max_preds
from models.common.uncertainty import spatial_probability
from utilities.transforms import flip_back
from utilities.vis import save_debug_images
from utilities.reproducibility import accumulation_group_size
from utilities.utilities import clip_pose_and_auxiliary_gradients


logger = logging.getLogger(__name__)

test_epoch = 0


def _last_output(outputs):
    return outputs[-1] if isinstance(outputs, list) else outputs


def _location_maps(output):
    # Training accuracy and debug images must use the exact pose heatmaps used
    # by the baseline, never the auxiliary probability representation.
    return output['location_logits'] if isinstance(output, dict) else output


def _compute_loss(criterion, outputs, target, target_weight, meta):
    if isinstance(outputs, list):
        return sum(
            criterion(item, target, target_weight, meta) for item in outputs
        )
    return criterion(outputs, target, target_weight, meta)


def _flip_back_output(output, flip_pairs, device):
    if not isinstance(output, dict):
        flipped = flip_back(output.detach().cpu().numpy(), flip_pairs)
        return torch.from_numpy(flipped.copy()).to(device)

    result = {}
    for key in ('location_logits', 'probability_maps'):
        flipped = flip_back(output[key].detach().cpu().numpy(), flip_pairs)
        result[key] = torch.from_numpy(flipped.copy()).to(device)
    for key in ('quality_logits', 'visibility_logits'):
        values = output[key].clone()
        for left, right in flip_pairs:
            values[:, [left, right]] = values[:, [right, left]]
        result[key] = values
    return result


def _average_pose_outputs(config, first, second):
    if not isinstance(first, dict):
        return (first + second) * 0.5
    logits = (first['location_logits'] + second['location_logits']) * 0.5
    return {
        'location_logits': logits,
        'probability_maps': spatial_probability(
            logits,
            distribution=config.UNCERTAINTY.DISTRIBUTION,
            temperature=config.UNCERTAINTY.TEMPERATURE,
        ),
        'quality_logits': (first['quality_logits'] + second['quality_logits']) * 0.5,
        'visibility_logits': (
            first['visibility_logits'] + second['visibility_logits']
        ) * 0.5,
    }

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, grad_accum_steps=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    if grad_accum_steps is None:
        grad_accum_steps = getattr(config.TRAIN, 'GRAD_ACCUM_STEPS', 1)
    accum_steps = int(grad_accum_steps)
    if accum_steps < 1:
        accum_steps = 1
    optimizer.zero_grad()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        input = input.cuda(non_blocking=True)
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        output = _last_output(outputs)
        loss = _compute_loss(criterion, outputs, target, target_weight, meta)
        location_output = _location_maps(output)

        num_images = input.size(0)
        losses.update(loss.item(), num_images)

        # compute gradient and do update step
        # The last accumulation group can be shorter than accum_steps. Divide
        # by its real size so its gradients are not silently underweighted.
        group_size = accumulation_group_size(i, len(train_loader), accum_steps)
        (loss / group_size).backward()
        should_step = ((i + 1) % accum_steps == 0) or ((i + 1) == len(train_loader))
        if should_step:
            clip_norm = getattr(config.TRAIN, 'CLIP_GRAD_NORM', 0.0)
            if clip_norm and clip_norm > 0:
                clip_pose_and_auxiliary_gradients(model, max_norm=clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        output_np = location_output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        _, avg_acc, cnt, pred = accuracy(output_np, target_np)
        
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            # Peak distance diagnostics (pixel and normalized PCK distance)
            try:
                pred_xy, _ = get_max_preds(output_np)
                gt_xy, _ = get_max_preds(target_np)
                vis = target_weight.detach().cpu().numpy()[:, :, 0] > 0.5
                if np.any(vis):
                    peak_dist = np.linalg.norm(pred_xy - gt_xy, axis=2)[vis].mean()
                    h = output_np.shape[2]
                    w = output_np.shape[3]
                    norm = np.array([h, w], dtype=np.float32) / 10.0
                    peak_dist_norm = np.linalg.norm((pred_xy - gt_xy) / norm, axis=2)[vis].mean()
                else:
                    peak_dist = 0.0
                    peak_dist_norm = 0.0
            except Exception:
                peak_dist = 0.0
                peak_dist_norm = 0.0
            components = getattr(criterion, 'last_components', {})
            component_text = ''.join(
                f'\tUQ-{name} {value:.5f}' for name, value in components.items()
            )
            msg = ('Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'PeakDist {peak_dist:.3f}\t' \
                  'PeakDistN {peak_dist_norm:.3f}'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc,
                      peak_dist=peak_dist, peak_dist_norm=peak_dist_norm)
                  + component_text)

            logger.info(msg)
            # Do not write per-batch text logs per user preference.

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer.add_scalar('train_acc', acc.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_debug_images(config, input, meta, target, pred*4, location_output,
                              prefix)
    # Return the epoch average training accuracy
    return acc.avg



val_losses = []
val_accuracies = []

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict, epoch):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):

            #input = input.permute(0, 3, 1, 2)
            # compute output
            input = input.cuda(non_blocking=True)
            outputs = model(input)
            output = _last_output(outputs)

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                output_flipped = _flip_back_output(
                    _last_output(outputs_flipped), val_dataset.flip_pairs, input.device
                )


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    if isinstance(output_flipped, dict):
                        output_flipped['location_logits'][:, :, :, 1:] = \
                            output_flipped['location_logits'].clone()[:, :, :, 0:-1]
                    else:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                output = _average_pose_outputs(config, output, output_flipped)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight, meta)
            location_output = _location_maps(output)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            output_np = location_output.cpu().numpy()
            target_np = target.cpu().numpy()
            _, avg_acc, cnt, pred = accuracy(output_np, target_np)

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            #print("Center:",c)
            s = meta['scale'].numpy()
            #print("Scale:",s)
            score = meta['score'].numpy()

            preds, maxvals = get_pose_output_preds(config, output, c, s)

            bbox_offset = meta.get('bbox_offset', None)
            if bbox_offset is not None:
                try:
                    offset_np = bbox_offset.numpy()
                    preds[:, :, 0] += offset_np[:, 0:1]
                    preds[:, :, 1] += offset_np[:, 1:2]
                except Exception:
                    pass

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                try:
                    pred_xy, _ = get_max_preds(output_np)
                    gt_xy, _ = get_max_preds(target_np)
                    vis = target_weight.detach().cpu().numpy()[:, :, 0] > 0.5
                    if np.any(vis):
                        peak_dist = np.linalg.norm(pred_xy - gt_xy, axis=2)[vis].mean()
                        h = output_np.shape[2]
                        w = output_np.shape[3]
                        norm = np.array([h, w], dtype=np.float32) / 10.0
                        peak_dist_norm = np.linalg.norm((pred_xy - gt_xy) / norm, axis=2)[vis].mean()
                    else:
                        peak_dist = 0.0
                        peak_dist_norm = 0.0
                except Exception:
                    peak_dist = 0.0
                    peak_dist_norm = 0.0
                msg = 'Validation: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                      'PeakDist {peak_dist:.3f}\t' \
                      'PeakDistN {peak_dist_norm:.3f}'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc, peak_dist=peak_dist,
                          peak_dist_norm=peak_dist_norm)
                logger.info(msg)
                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                save_debug_images(config, input, meta, target, pred*4, location_output,
                                  prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )


        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    # Return both the per-keypoint name_values and the overall perf indicator
    return name_values, perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
