"""MMPose CocoDataset adapter that preserves Bird_HKE's unclipped crop boxes."""
import numpy as np
from mmpose.datasets.datasets.body.coco_dataset import CocoDataset
from mmpose.registry import DATASETS


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
