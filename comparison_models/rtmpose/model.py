"""RTMPose-M estimator that inverts BirdGaze's signed-scale crops.

MMPose TopdownPoseEstimator assumes zero crop rotation when recovering image
coordinates from decoded SimCC predictions. In BirdGaze FD, negative source
scales imply a 180-degree image warp, so an explicit inverse is required
during image validation. This does not alter the native RTMPose-M head/loss.
"""
from __future__ import annotations

import numpy as np
from mmpose.models.pose_estimators.topdown import TopdownPoseEstimator
from mmpose.registry import MODELS


@MODELS.register_module()
class BirdGazeRTMPoseEstimator(TopdownPoseEstimator):
    def add_pred_to_datasample(self, batch_pred_instances, batch_pred_fields,
                               batch_data_samples):
        samples = super().add_pred_to_datasample(
            batch_pred_instances, batch_pred_fields, batch_data_samples)
        for sample in samples:
            if not sample.metainfo.get("birdgaze_negative_scale", False):
                continue
            # MMPose maps SimCC coordinates into an unrotated bbox. The
            # original Bird_HKE negative scale reverses both axes; undo it
            # around the (possibly source-cropped) input center.
            center = np.asarray(
                sample.metainfo["input_center"], dtype=np.float32).reshape(1, 1, 2)
            keypoints = sample.pred_instances.keypoints
            keypoints[..., :2] = 2.0 * center - keypoints[..., :2]
            sample.pred_instances.keypoints = keypoints
        return samples
