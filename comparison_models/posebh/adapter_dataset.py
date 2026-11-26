"""PoseBH read-only BirdGaze adapter shared with the ViTPose benchmark.

Dataset/augmentation/decode are intentionally identical across the two
architectures for the scientific comparison. Importing this module cannot
train, write to, or alter ViTPose; paths and checkpoints remain separate.
The PoseBH recipe fingerprints the common source on each run.
"""
from comparison_models.common.birdgaze.dataset import (
    BirdGazeCropDataset,
    decode_image_coordinates,
    coordinate_valid,
    affine_matrix,
    gaussian_targets,
)


class BirdGazePoseBHDataset(BirdGazeCropDataset):
    """Four head keypoints, identical crop and supervision for PoseBH."""
