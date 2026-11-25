"""Shared infrastructure for the three external comparison models.

Everything in this package is model-agnostic: BirdGaze loading/geometry, the
COCO export, read-only preflight, dependency inspection, disposable training
probes, and the training protocol shared by the two single-task ViT adapters.

Import safety contract (enforced by ``comparison_models/tests/test_layout.py``):

* importing this package must not require MMPose, MMCV, MMEngine or CUDA;
* no module here may import a specific model package
  (``comparison_models.rtmpose`` / ``.vitpose`` / ``.posebh``);
* architecture-specific losses, heads and optimisers stay in the model
  packages and are deliberately *not* homogenised.
"""
