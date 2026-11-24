"""PoseBH read-only BirdGaze adapter shared with the ViTPose benchmark.

Dataset/augmentation/decode are intentionally identical across the two
architectures for the scientific comparison. Importing this module cannot
train, write to, or alter ViTPose; paths and checkpoints remain separate.
The PoseBH recipe fingerprints the common source on each run.
"""
from external_models.vitpose.dataset import (
    BirdGazeViTPoseDataset,
    decode_image_coordinates,
    coordinate_valid,
    affine_matrix,
    gaussian_targets,
)


class BirdGazePoseBHDataset(BirdGazeViTPoseDataset):
    """Four head keypoints, identical crop and supervision for PoseBH."""
