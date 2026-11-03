"""MMPose CocoDataset adapter that preserves Bird_HKE's unclipped crop boxes."""
import numpy as np
from mmpose.datasets.datasets.body.coco_dataset import CocoDataset
from mmpose.registry import DATASETS, TRANSFORMS
from mmcv.transforms import BaseTransform
from mmpose.datasets.transforms import TopdownAffine
from external_models.rtmpose.geometry import apply_source_crop


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
            raw = raw_data_info["raw_ann_info"]
            result["birdgaze_negative_scale"] = bool(raw.get("birdgaze_negative_scale", False))
            result["birdgaze_zero_scale"] = bool(raw.get("birdgaze_zero_scale", False))
            result["birdgaze_source_bbox"] = raw.get("birdgaze_source_bbox")
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



@TRANSFORMS.register_module()
class BirdGazeApplySourceCrop(BaseTransform):
    """Apply corrected/original BirdGaze source crop before pose affine."""

    def transform(self, results):
        return apply_source_crop(results)


@TRANSFORMS.register_module()
class BirdGazeTopdownAffine(TopdownAffine):
    """Preserve the source loader's degenerate zero-scale warp exactly.

    cv2.getAffineTransform() returns a zero matrix for zero-width source
    triangles: warpAffine therefore repeats the input's top-left pixel.
    Valid landmark targets all map to (0,0). For normal cases, defer to the
    upstream MMPose implementation.
    """

    def transform(self, results):
        if not results.get("birdgaze_zero_scale", False):
            return super().transform(results)
        w, h = self.input_size
        pixel = np.asarray(results["img"])[0, 0].copy()
        results["img"] = np.broadcast_to(pixel, (h, w, pixel.shape[-1])).copy()
        if results.get("keypoints") is not None:
            results["transformed_keypoints"] = np.zeros_like(results["keypoints"])
        else:
            results["transformed_keypoints"] = np.zeros([])
            results["keypoints_visible"] = np.ones((1, 1, 1))
        results["bbox_scale"] = np.zeros_like(results["bbox_scale"])
        results["input_size"] = (w, h)
        results["input_center"] = results["bbox_center"][0]
        results["input_scale"] = results["bbox_scale"][0]
        return results
