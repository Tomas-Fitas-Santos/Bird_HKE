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
        """Preserve the full BirdGaze sample set, including missing joints.

        Stock CocoDataset rejects rows with zero keypoints or with only
        sentinel coordinates. Bird_HKE retains them and masks their losses.
        Only reject genuinely invalid/crowd instances or invalid pose boxes.
        """
        if data_info.get("iscrowd", 0):
            return False
        bbox = data_info.get("bbox")
        if bbox is None or bbox.shape != (1, 4) or not np.isfinite(bbox).all():
            return False
        x1, y1, x2, y2 = bbox[0]
        return bool(x2 > x1 and y2 > y1)


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



@TRANSFORMS.register_module()
class BirdGazeRandomBBoxTransform(BaseTransform):
    """Match Bird_HKE's clipped-normal scale/rotation augmentation.

    Original JointsDataset uses s * clip(randn()*0.25+1, .75, 1.25)
    and, with probability 0.6, clip(randn()*30, -60, 60) degrees.
    The negative original scale is restored as an additional 180 degrees
    by BirdGazeRestoreSignedScale downstream.
    """

    def transform(self, results):
        n = len(results["bbox_scale"])
        factor = np.clip(
            np.random.randn(n, 1).astype(np.float32) * 0.25 + 1.0,
            0.75, 1.25,
        )
        rotation = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if __import__("random").random() <= 0.6:
                rotation[i] = np.clip(
                    np.random.randn() * 30.0, -60.0, 60.0)
        results["bbox_scale"] = results["bbox_scale"] * factor
        results["bbox_rotation"] = rotation
        return results
