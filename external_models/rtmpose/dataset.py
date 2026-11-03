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
