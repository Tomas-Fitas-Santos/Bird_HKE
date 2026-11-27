# Model families

The repository contains two groups of pose-estimation models that must never be
conflated. This document is the authoritative list of which model belongs where
and what is guaranteed about each group.

## Group 1 - Original Bird_HKE research models

Location: [`Bird_HKE/`](../Bird_HKE/). These are the architectures the thesis
contributes. Their architectures, training protocol, uncertainty
implementations, configuration files and trained-model paths are preserved
exactly as developed in Phases 1-3.

| Family | Config directory | Notes |
|---|---|---|
| HRNet (w32) | `Bird_HKE/experiments/HRNet/` | Baseline convolutional network |
| HR_Mamba | `Bird_HKE/experiments/HR_Mamba/` | Two fusion variants (`sum`, `concat_gate`) |
| HR_MambaViT | `Bird_HKE/experiments/HR_MambaViT/` | |
| HR_MambaVision | `Bird_HKE/experiments/HR_MambaVision/` | Integrates NVIDIA MambaVision (NVIDIA Source Code License-NC) |
| VHR-BirdPose | `Bird_HKE/experiments/VHR_BirdPose/` | Adapted from the upstream VHR-BirdPose release |

Eighteen experiment YAMLs (six families x FD/OS/CS) are enforced by
`Bird_HKE/tools/audit_training_protocol.py`, which also verifies the
`bird_hke_repro_v2` protocol, the effective batch of 64, and the Phase-2 policy
that uncertainty is enabled only for the six FD configs.

* Trainer: `Bird_HKE/tools/train.py`.
* Environment: the native `bird_hke-cu128` conda environment.
* Checkpoints: `Bird_HKE/trained_models/<dataset>/<family>/repro_v2|repro_v3/seed_2026/`.
* Uncertainty: `Bird_HKE/UNCERTAINTY.md`, `Bird_HKE/tools/calibrate_uncertainty.py`.

**No file in this group imports `comparison_models`.** That invariant is
asserted by `comparison_models/tests/test_consolidation.py`.

## Group 2 - External comparison models

Location: [`comparison_models/`](../comparison_models/). Three published
architectures retrained on BirdGaze so that the research models can be compared
against recognised baselines under the same data, protocol and evaluation.

| id | Model | Framework | Environment | Checkpoint root |
|---|---|---|---|---|
| `rtmpose-m` | RTMPose-M | official MMPose/MMEngine | `rtmpose-torch27-venv` | `Bird_HKE/trained_models/External_Baselines/RTMPose_M` |
| `vitpose-b` | ViTPose-B (standard single-task) | plain PyTorch | `vitpose-b-cu128-venv` | `.../External_Baselines/ViTPose_B` |
| `posebh-b` | PoseBH-B (single-dataset adaptation) | plain PyTorch | `posebh-b-cu128-venv` | `.../External_Baselines/PoseBH_B` |

Exactly nine experiments: 3 models x FD/OS/CS, seed 2026, random
initialisation, at most 100 epochs, no pretrained weights, no implicit reuse of
native weights.

### What is shared, and what is not

Shared, in `comparison_models/common/`:

* the BirdGaze data contract - COCO export, crop/affine dataset, heatmap
  targets, occlusion vs missing-keypoint semantics, OS/CS source-crop handling,
  coordinate decoding back to full-image pixels;
* the ViT primitives (`Attention`, `DropPath`, weight initialisation);
* the masked heatmap MSE loss;
* the epoch protocol - gradient accumulation, AMP, clipping, atomic
  checkpointing, strict full-state resume, best-slot alternation, history
  recovery, PCK validation.

Not shared:

* model architectures - each package owns its own `model.py`;
* trainers - `SingleTaskTrainer` subclasses supply only the model, the loss and
  the validation heatmap extraction;
* environments, checkpoints and logs.

**PoseBH no longer imports the ViTPose model module**, and the shared dataset
class is named `BirdGazeCropDataset` rather than after any one model. Both
invariants are asserted by tests.

### Naming rules

* Always write **`PoseBH-B (single-dataset adaptation)`** in READMEs, plots and
  reports. It uses one active MoE expert, four keypoints, three EMA prototypes
  per landmark and within-dataset consistency. It is **not** the paper's
  six-dataset pretrained method and must never be presented as a reproduction of
  its CPS/CSS cross-dataset results.
* `vitpose-b` is the standard single-task ViTPose-B
  (`ViTPose_base_simple_coco_256x192.py`), not ViTPose++.
* `rtmpose-m` uses the official MMPose implementation with a BirdGaze recipe;
  FD, OS and CS are supported; each definitive run requires GPU preflight.

## Cross-group rules

1. No runtime import in either direction.
2. Comparison models never write inside a native `trained_models/` experiment
   namespace; they use `External_Baselines/` exclusively.
3. The native environment is never used to run a comparison model, and each
   comparison trainer refuses to start outside its own interpreter.
4. Pinned upstream revisions are exact and upstream checkouts are never
   modified in place; RTMPose uses a verified import-only overlay.
5. All models report predictions in original full-image pixel coordinates and
   are evaluated on the same independently annotated held-out videos.
