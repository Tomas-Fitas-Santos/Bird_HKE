"""BirdGaze data contract shared by every comparison model.

One source of truth for: the four-keypoint order
``[crown, left_eye, right_eye, beak_tip]``, the 256x256 input crop, the 64x64
heatmap grid, the source-bbox pre-crop, center/scale conversion, horizontal
flip eye swapping, absent-vs-annotated-occluded joint masks, negative and zero
crop scales, and the mapping of predictions back into original full-image
pixel coordinates.

RTMPose-M consumes the same contract through the MMPose pipeline transforms in
``comparison_models.rtmpose.dataset``; ViTPose-B and PoseBH-B consume it
directly through :mod:`comparison_models.common.birdgaze.dataset`.
"""
