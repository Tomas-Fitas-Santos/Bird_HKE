# PoseBH-B for BirdGaze — single-dataset adaptation

The official PoseBH [CVPR 2025 project](https://github.com/uyoung-jeong/PoseBH)
at commit `8f03438f913b559edb0a2beed54e77d1560edba3` uses
**six datasets, a 6-expert ViTMoE backbone, multiple supervised heads,
per-landmark prototypes, clustered prototype supervision (CPS), and
cross-dataset self-supervision (CSS)**. Its published training commands
start from ViTPose++ pretrained checkpoints.

Those assumptions **do not hold in our BirdGaze-only, from-scratch
comparison**. The code here is explicitly called a **single-dataset
PoseBH-B adaptation**, NOT a claimed reproduction of the multi-dataset
PoseBH-B results.

## What is actually retained and adapted

| Mechanism | BirdGaze PoseBH-B implementation |
|---|---|
| ViT-B backbone | 12 blocks, width 768, 12 heads, 16×16 patches |
| PoseBH MoE feed-forward | 576 shared and 192 expert dimensions; exactly **one active expert** for BirdGaze, not six untrained dataset experts |
| Auxiliary keypoint head | Independent 2× transposed-convolution heatmap head; four landmarks |
| Prototype embedding head | Independent 2× transposed-convolution + 64-D residual neck |
| Prototypes | **Three 64-D prototypes per keypoint**, with normalized absolute cosine similarity and EMA update (γ=0.999) |
| Prototype losses | Pixel Prototype CE / PPD, scaled using upstream weights, plus prototype heatmap MSE |
| Original multi-dataset CPS | **Unavailable** with one task; not claimed |
| Cross-dataset CSS | Replaced by **same-task high-confidence agreement on missing keypoints** after epoch 50; not equivalent to upstream CSS |
| Pretrained initialization | None. Strictly random initialization |
| Crop/augmentation | The **exact same read-only BirdGaze dataset adapter** as standard ViTPose-B |
| Model selection | Image-validation PCK@0.05 on prototype heatmaps; held-out annotated videos reserved for science evaluation |

The specific upstream basis is:
`configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mdt/vitb_posebh.py`,
`mmpose/models/backbones/vit_moe.py`,
`mmpose/models/heads/kpt_proto_head.py`,
`mmpose/models/heads/kpt_proto_module.py`, and PPCE/CSS losses.

**Research qualification:** because the multi-dataset approach is PoseBH's
central scientific contribution, these BirdGaze scores describe a controlled
single-dataset variant. They must not be presented as a published PoseBH
multi-dataset replication. A future complete multi-dataset PoseBH experiment
would require additional heterogeneous keypoint datasets.

## Isolation

- All new code stays in `external_models/posebh/` and `setup_posebh_b.sh`.
- Uses a **separate Python virtual environment**; inherits existing Torch
  wheel read-only, with no MMCV/MMPose installs in any active environment.
- Reads the existing BirdGaze FD, OS, CS images/annotations; does not
  modify the original images or native training code.
- Saves **only** under
  `Bird_HKE/trained_models/External_Baselines/PoseBH_B/<FD|OS|CS>/seed_2026/`.
  The ViTPose-B, RTMPose and six native Bird_HKE model output paths are not
  touched. The PoseBH manifest includes source hashes of the shared
  read-only BirdGaze preprocessing and the model modules.
- Fail-closed GPU PID check prevents launch if another compute job is already
  running; this is **not** a GPU reservation system, so researchers still need
  to schedule use of the workstation.
- A fresh start cannot overwrite an existing PoseBH run. Strict resume
  requires matching code/data/recipe/runtime, complete RNG/model/optimizer/
  scheduler/AMP states, best-checkpoint integrity and recovery of interrupted
  metrics writes.
- `--smoke-test` runs a real forward/backward/optimizer step, prototype EMA
  and prototype-decoder validation **without persistent output writes**.

## Preparing Titan — do not switch the existing training checkout

Use the **same dedicated experiment worktree** used for ViTPose. This makes
the two experiments share the immutable BirdGaze annotations while storing
independent outputs and interpreter environments:

```bash
cd /mnt/storage/TomasSantos/Bird_HKE-vitpose

# The feature worktree should already be checked out on vitpose-local.
# Inspect environment prerequisites before touching files.
bash setup_posebh_b.sh --inspect

# Fetch ONLY the official pinned PoseBH checkout.
bash setup_posebh_b.sh --sources

# Use the path to the existing compatible Torch/CUDA Python as a starting
# interpreter; creation NEVER pip-installs into that interpreter.
export POSEBH_BASE_PY="/path/to/working/torch-cu128/python"
bash setup_posebh_b.sh --create
bash setup_posebh_b.sh --verify
```

Default PoseBH interpreter:
`/home/students/conda-envs/posebh-b-cu128-venv/bin/python`.
Override using `POSEBH_ENV_PREFIX` to choose another **unused** isolated path.

The reference upstream project used Python 3.8/CUDA 11.3/MMCV 1.4.8.
We do **not** install that incompatible historical environment; the
independent implementation runs with the same modern Torch CUDA setup used
for ViTPose, in its own interpreter. It must be GPU-tested before use.

## CPU and real-data GPU tests — no training

```bash
POSEBH_PY=/home/students/conda-envs/posebh-b-cu128-venv/bin/python
$POSEBH_PY -m unittest discover -s tests -p 'test_posebh_b.py' -v
$POSEBH_PY -m unittest discover -s tests -p 'test_remaining_baseline_preflight.py' -v
$POSEBH_PY -m external_models.posebh.train --scenario FD --smoke-test
```

The preflight validates the upstream pinned commit and its actual ViTMoE +
prototype-head architecture, corresponding BirdGaze source/COCO annotation
records and source crop metadata. You must first generate corrected FD/OS/CS
COCO exports if they are missing (see `external_models/README.md`).
Neither unit tests nor smoke tests create model-training checkpoints.

To check **both** independent baseline models without starting training,
use `bash smoke_both_birdgaze_baselines.sh FD` after the two environments
are prepared. It runs tests and two GPU smoke tests sequentially.

## Training commands (ONLY after both models pass smoke tests)

```bash
$POSEBH_PY -m external_models.posebh.train --scenario FD --train
# If interrupted after a saved epoch checkpoint:
$POSEBH_PY -m external_models.posebh.train --scenario FD --resume
```

Use `--scenario OS` and `--scenario CS` for their independent new
directories, when those exports pass the OS/CS crop-collision guard.

Training is capped at 100 epochs with seed 2026, 256×256 inputs,
microbatch 2, gradient-accumulated batch 64, AdamW lr=0.0005, warmup/cosine
decay, AMP and gradient clipping. Image validation selects the best *prototype*
heatmap checkpoint, never the independent held-out annotated video data.

## Current qualification status

- Model, data interface, trainer, scripts and CPU tests: **committed**.
- CPU tests: **not yet executed on Titan**.
- Real BirdGaze GPU forward/backward, MoE and prototype losses, EMA and
  original-image decoder: **not yet GPU-verified**.
- Full 100-epoch training: **not started**.
- Published six-dataset PoseBH reproduction: **out of scope**.

No existing model checkpoints, training process or active environment was
changed during implementation.
