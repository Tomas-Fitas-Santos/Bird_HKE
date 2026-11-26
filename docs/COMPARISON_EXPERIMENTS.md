# Comparison experiments

Nine single-seed runs compare the three external baselines against the original
Bird_HKE research models. This document fixes what is being compared, how, and
what still has to be verified before any number is reported.

## Matrix

| | FD | OS | CS |
|---|---|---|---|
| RTMPose-M | planned | refused by launcher | refused by launcher |
| ViTPose-B | planned | planned | planned |
| PoseBH-B (single-dataset adaptation) | planned | planned | planned |

* FD = `BirdGaze_v2/BirdGaze_Full_Dataset`, OS =
  `BirdGaze_v2/birdgaze_original_subset`, CS =
  `BirdGaze_v2/birdgaze_corrected_subset`.
* Seed 2026 for every run. Random initialisation, no pretrained weights, no
  reuse of native checkpoints.
* At most 100 epochs, AdamW lr `5e-4`, weight decay `0.01`, 5 warm-up epochs,
  cosine decay to `1e-5`, gradient clipping 1.0, physical batch 8, effective
  optimizer batch 64, 256x256 input, 64x64 heatmaps, Gaussian sigma 2.
* RTMPose-M follows the upstream 210-epoch MMEngine schedule; it is therefore
  reported as an architecture-faithful baseline rather than a
  protocol-identical one, and only FD is validated in this phase.

## Protocol parity

ViTPose-B and PoseBH-B execute the *same* code path for accumulation, AMP,
clipping, checkpointing, resume and PCK validation
(`comparison_models/common/training_utils.py`); only the architecture and loss
differ. Both consume the identical augmentation stream for a given epoch and
seed, so a difference in their curves is attributable to the model.

## Data rules

* `annot/train.json` updates weights.
* `annot/val.json` selects checkpoints and nothing else.
* The independently annotated videos are the final test set. They must not
  appear in training or checkpoint selection.
* Four keypoints in the order `[crown, left eye, right eye, beak tip]`.
* Missing and annotated-but-occluded coordinates are different states.
* Signed and zero crop scales keep their existing semantics.
* All predictions are decoded back to original full-image pixel coordinates
  before evaluation.

## Evaluation

Report on the held-out annotated videos with one shared pipeline: PCK under the
existing normalization convention, per-keypoint errors, and temporal metrics
under identical frame-matching and missing-landmark rules, with raw and
filtered results shown separately using fixed filter parameters for every
model. Image-validation PCK is a training diagnostic only. Bootstrap
uncertainty must resample video sequences, not frames.

## Status and open gates

| Item | Status |
|---|---|
| Shared BirdGaze conversion + OS/CS source-crop fidelity | implemented, CPU-tested |
| OS/CS identical-export guard | implemented (`preflight.validate_os_cs_distinction`) |
| ViTPose-B trainer | implemented; CPU-tested; **GPU smoke UNVERIFIED** |
| PoseBH-B trainer | implemented; CPU-tested; **GPU smoke UNVERIFIED** |
| ViTPose/PoseBH 2-epoch train+resume probe | implemented; **UNVERIFIED** (needs GPU) |
| RTMPose-M FD GPU data/model preflight | implemented; **UNVERIFIED** |
| RTMPose-M disposable optimizer/resume probe | **not implemented** - upstream MMEngine runner has no isolated mini-run harness |
| Any of the nine definitive runs | **not started** |

CPU CI being green is a merge requirement. GPU validation is a separate
full-training readiness gate. No GPU result has been produced for any of the
three models, and none should be reported until
`bash comparison_models/scripts/test_comparison_models.sh FD` passes on the
cluster.
