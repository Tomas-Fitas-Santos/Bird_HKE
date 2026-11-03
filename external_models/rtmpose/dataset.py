"""MMPose CocoDataset adapter that preserves Bird_HKE's unclipped crop boxes."""
import numpy as np
from mmpose.datasets.datasets.body.coco_dataset import CocoDataset
from mmpose.registry import DATASETS, TRANSFORMS
from mmcv.transforms import BaseTransform


@DATASETS.register_module()
class BirdGazeCocoDataset(CocoDataset):
    def parse_data_info(self, raw_data_info):
        result = super().parse_data_info(raw_data_info)
        if result is not None:
            x, y, width, height = raw_data_info["raw_ann_info"]["bbox"]
            # The stock CocoDataset clips bboxes at the image boundary; the
            # existing Bird_HKE JointsDataset does not. Keep crop geometry.
            result["bbox"] = np.asarray(
                [[x, y, x + width, y + height]], dtype=np.float32
            )
            result["birdgaze_negative_scale"] = bool(
                raw_data_info["raw_ann_info"].get("birdgaze_negative_scale", False)
            )
        return result

    @staticmethod
    def _is_valid_instance(data_info):
        """Keep the original BirdGaze images with no supervised keypoints.

        The stock MMPose COCO loader removes samples when num_keypoints == 0.
        Bird_HKE keeps them with all target weights set to zero. Retain those
        images here too, while still rejecting invalid bounding boxes.
        """
        if data_info.get("num_keypoints") == 0:
            if data_info.get("iscrowd", 0):
                return False
            bbox = data_info.get("bbox")
            if bbox is None or not np.isfinite(bbox).all():
                return False
            x1, y1, x2, y2 = bbox[0]
            return bool(x2 > x1 and y2 > y1)
        return CocoDataset._is_valid_instance(data_info)


@TRANSFORMS.register_module()
class BirdGazeRestoreSignedScale(BaseTransform):
    """Restore Bird_HKE negative-scale crops via 180-degree rotation.

    Must follow RandomBBoxTransform (which replaces the angle) and precede
    TopdownAffine, in both the train and validation pipelines.
    """

    def transform(self, results):
        if results.get("birdgaze_negative_scale", False):
            existing = results.get("bbox_rotation")
            angle = float(np.asarray(existing).reshape(-1)[0]) if existing is not None else 0.0
            results["bbox_rotation"] = np.array([angle + 180.0], dtype=np.float32)
        return results
