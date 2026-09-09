# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# This file is adapted from the original codebase of Simple Baselines for Human Pose Estimation and Tracking, which is licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import json_tricks as json
from collections import OrderedDict
import json
import numpy as np
from scipy.io import savemat
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)


def infer_data_source(image_name):
    """Infer the source for both flat and source-grouped dataset layouts."""
    normalized = str(image_name).replace('\\', '/').strip('/')
    parts = normalized.split('/')
    lowered = [part.lower().replace(' ', '_') for part in parts]

    if any(part in ('ebird', 'e_bird') for part in lowered):
        return 'eBird'
    if any(part in ('nabirds', 'na_birds') for part in lowered):
        return 'NABirds'
    if any(part in ('birdsnap', 'bird_snap') for part in lowered):
        return 'BirdSnap'
    if any(part in ('animal_kingdom', 'animalkingdom') for part in lowered):
        return 'Animal Kingdom'

    folder = parts[0] if len(parts) > 1 else ''
    # ``FINETUNE`` is the legacy folder name of the eBird data source; it is
    # unrelated to a model-training phase.
    if folder.upper() == 'FINETUNE':
        return 'eBird'
    if folder.isdigit():
        return 'NABirds'
    if folder.isalpha() and folder.isupper():
        return 'Animal Kingdom'
    return 'BirdSnap'


def coordinate_validity(joints, explicit_valid=None, visibility=None):
    """Return which coordinates can supervise localization.

    A visibility value of zero does not automatically discard a non-sentinel
    coordinate.  This supports future fully annotated occlusions while still
    treating the common ``[0, 0]`` placeholder as absent.
    """
    xy = np.asarray(joints, dtype=np.float32)[:, :2]
    implicit = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0)
        & (xy[:, 1] >= 0)
        & np.any(xy != 0, axis=1)
    )
    if explicit_valid is not None:
        values = np.asarray(explicit_valid, dtype=np.float32).reshape(-1)
        return values > 0
    if visibility is None:
        return implicit
    visible = np.asarray(visibility, dtype=np.float32).reshape(-1) > 0
    return visible | implicit


class BirdGazeDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        '''
        0: Head_Mid_Top
        1: Eye_Left 
        2: Eye_Right 
        3: Mouth_Front_Top 
        '''
        ### Changes below ###
        self.num_joints = 4
        self.flip_pairs = [[1, 2]]
        self.parent_ids = [3,3,3,3]

        self.upper_body_ids = (0, 1, 2, 3)
        self.lower_body_ids = ()

        # Normalize Windows-style `root` when running under WSL so image IO works
        if os.name == 'posix' and isinstance(root, str):
            m = re.match(r'^([A-Za-z]):[\\/](.*)$', root)
            if m:
                drive = m.group(1).lower()
                rest = m.group(2).replace('\\', '/').replace('\\', '/')
                self.root = f'/mnt/{drive}/{rest}'
        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # Create train/val split
        file_name = os.path.join(
            self.root, self.annotation_dir, self.image_set+'.json'
        )
        # Normalize Windows-style paths when running under WSL
        def _normalize_wsl_path(p):
            if os.name == 'posix' and isinstance(p, str):
                m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
                if m:
                    drive = m.group(1).lower()
                    rest = m.group(2).replace('\\', '/').replace('\\', '/')
                    return f'/mnt/{drive}/{rest}'
            return p

        file_name = _normalize_wsl_path(file_name)
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=float)
            s = np.array([a['scale'], a['scale']], dtype=float)
            bbox = a.get('bbox', None)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                # c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            joints_3d = np.zeros((self.num_joints, 3), dtype=float)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=float)
            visibility_values = a.get('joints_vis')
            visibility_known = np.zeros((self.num_joints, 1), dtype=np.float32)
            visibility_target = np.zeros((self.num_joints, 1), dtype=np.float32)
            joints_3d_valid = np.zeros((self.num_joints, 3), dtype=np.float32)

            # Alteration here ****
            if self.image_set != '123abc':
                joints = np.array(a['joints'], dtype=np.float32)
                joints[:, 0:2] = joints[:, 0:2]
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                valid = coordinate_validity(
                    joints,
                    explicit_valid=a.get('joints_valid'),
                    visibility=visibility_values,
                ).astype(np.float32)
                joints_3d_valid[:, 0] = valid
                joints_3d_valid[:, 1] = valid

                if visibility_values is not None:
                    visible = (
                        np.asarray(visibility_values, dtype=np.float32).reshape(-1) > 0
                    ).astype(np.float32)
                    if visible.size != self.num_joints:
                        raise ValueError(
                            f"{image_name}: expected {self.num_joints} visibility "
                            f"values, found {visible.size}"
                        )
                    visibility_known[:, 0] = 1.0
                    visibility_target[:, 0] = visible
                    joints_3d_vis[:, 0] = visible
                    joints_3d_vis[:, 1] = visible
                else:
                    # eBird records have usable coordinates but no visibility
                    # labels.  Keep legacy visualization/augmentation behavior
                    # while masking these samples from the visibility loss.
                    joints_3d_vis[:, 0] = valid
                    joints_3d_vis[:, 1] = valid

            image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            # sanitize image_name separators (some annotations use Windows backslashes)
            image_name = image_name.replace('\\', '/')

            # Build candidate paths and silently support known Animal Kingdom
            # naming variants used across dataset releases.
            image_candidates = [image_name]
            is_animal_kingdom = (
                image_name.startswith('Animal_Kingdom/')
                or image_name.startswith('Animal Kingdom/')
            )
            if is_animal_kingdom:
                ak_variants = [
                    image_name,
                    image_name.replace('Animal_Kingdom/', 'Animal Kingdom/'),
                    image_name.replace('Animal Kingdom/', 'Animal_Kingdom/')
                ]
                for variant in ak_variants:
                    image_candidates.append(variant)
                    image_candidates.append(
                        re.sub(r'_f(\d+\.[A-Za-z0-9]+)$', r'_t\1', variant)
                    )

            # keep unique candidates while preserving order
            seen = set()
            image_candidates = [
                c for c in image_candidates if not (c in seen or seen.add(c))
            ]

            full_image = None
            if self.data_format == 'zip':
                full_image = os.path.join(self.root, image_dir, image_candidates[0])
                full_image = _normalize_wsl_path(full_image)
            else:
                for candidate in image_candidates:
                    candidate_full = os.path.join(self.root, image_dir, candidate)
                    candidate_full = _normalize_wsl_path(candidate_full)
                    if os.path.exists(candidate_full):
                        full_image = candidate_full
                        break

                # Fall back to the canonical annotation path if no local match is found.
                if full_image is None:
                    full_image = os.path.join(self.root, image_dir, image_candidates[0])
                    full_image = _normalize_wsl_path(full_image)

            gt_db.append(
                {
                    'image': full_image,
                    'center': c,
                    'scale': s,
                    'bbox': bbox,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'joints_3d_valid': joints_3d_valid,
                    'visibility_known': visibility_known,
                    'visibility_target': visibility_target,
                    'source': infer_data_source(image_name),
                    'filename': '',
                    'imgnum': 0,
                }
            )

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # Convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

#         if 'test' in cfg.DATASET.TEST_SET:
#             return {'Null': 0.0}, 0.0

        SC_BIAS = 1
        threshold = 0.05

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               cfg.DATASET.ANNOT_DIR,
                               '{}.json' #gt_{}.json'
                               .format(cfg.DATASET.TEST_SET)
                               )
        # Normalize Windows-style paths when running under WSL
        def _normalize_wsl_path(p):
            if os.name == 'posix' and isinstance(p, str):
                m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
                if m:
                    drive = m.group(1).lower()
                    rest = m.group(2).replace('\\', '/').replace('\\', '/')
                    return f'/mnt/{drive}/{rest}'
            return p

        gt_file = _normalize_wsl_path(gt_file)

        with open(gt_file) as f:
            gt_dict = json.load(f)
        
        ### Changes below ###

        # dataset_joints = gt_dict['dataset_joints']
        # jnt_visible = [v for k, v in gt_dict['joints_vis'].items()]
        # pos_gt_src = [v for k, v in gt_dict['joints'].items()]
        # scale = [v for k, v in gt_dict['scale'].items()]

        dataset_joints = [
            [
                "Head_Mid_Top"
            ],
            [
                "Eye_Left"
            ],
            [
                "Eye_Right"
            ],
            [
                "Mouth_Front_Top"
            ]
        ]

        jnt_visible = []
        for record in gt_dict:
            visibility = record.get('joints_vis')
            if visibility is None:
                visibility = coordinate_validity(
                    record['joints'],
                    explicit_valid=record.get('joints_valid'),
                ).astype(np.float32).tolist()
            jnt_visible.append(visibility)
        pos_gt_src = [x['joints'] for x in gt_dict]
        scale = [x['scale'] for x in gt_dict]

        scale=np.array(scale)
        scale=scale*200

        jnt_visible=np.transpose(jnt_visible, [1, 0])
        pos_pred_src = np.transpose(preds, [1, 2, 0])
        pos_gt_src=np.transpose(pos_gt_src, [1, 2, 0])
        dataset_joints=np.array(dataset_joints)

        head = np.where(dataset_joints == 'Head_Mid_Top')[0][0]

        tlefteye = np.where(dataset_joints == 'Eye_Left')[0][0]

        trighteye = np.where(dataset_joints == 'Eye_Right')[0][0]

        tmouth = np.where(dataset_joints == 'Mouth_Front_Top')[0][0]

        
        
        uv_error = pos_pred_src - pos_gt_src

        uv_err = np.linalg.norm(uv_error, axis=1)

#         headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
#         headsizes = np.linalg.norm(headsizes, axis=0)
        scale *= SC_BIAS
        headsizes=scale


        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)

        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                          jnt_visible)
        # Return PCK as fraction in [0,1] (not percent)
        PCKh = np.divide(np.sum(less_than_threshold, axis=1), jnt_count)
        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 4))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            # store as fraction in [0,1]
            pckAll[r, :] = np.divide(np.sum(less_than_threshold, axis=1),
                                     jnt_count)

#         PCKh = np.ma.array(PCKh, mask=False)
#         PCKh.mask[21:22] = True

#         jnt_count = np.ma.array(jnt_count, mask=False)
#         jnt_count.mask[21:22] = True

        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
        name_value = [
            ('Head', PCKh[head]),
            ('Eyes', 0.5 * (PCKh[tlefteye] + PCKh[trighteye])),
            ('Mouth', PCKh[tmouth]),
            ('Mean', np.sum(PCKh * jnt_ratio))
#             ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)
        return name_value, name_value['Mean']
