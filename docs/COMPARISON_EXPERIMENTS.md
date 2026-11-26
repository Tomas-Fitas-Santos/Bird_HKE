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
* At most 100 epochs, 256x256 input, 64x64 heatmaps, Gaussian sigma 2,
  effective optimizer batch 64.
* **Each baseline uses its own paper's optimizer and schedule**, not a shared
  one: ViTPose-B uses AdamW `5e-4`/`0.1` with layer decay 0.75 and step decay at
  epochs 81/95; PoseBH-B uses AdamW `1e-3`/`0.1` with layer decay 0.75 and step
  decay at epochs 50/90; RTMPose-M uses AdamW `4e-3`/`0.05`, clip 35,
  1000-iteration warmup, half-range cosine and `ExpMomentumEMA`. Epoch count is
  the only exempted hyperparameter. See
  [PAPER_FAITHFUL_PROTOCOL.md](./PAPER_FAITHFUL_PROTOCOL.md).
* RTMPose-M's upstream AP-10K schedule is 210 epochs; its cosine range and
  warmup are transposed onto the 100-epoch budget, so it is an
  optimizer-faithful baseline whose epoch budget and augmentation pipeline are
  Bird_HKE's. Only FD is validated in this phase.

## Protocol parity

ViTPose-B and PoseBH-B execute the *same* code path for accumulation, AMP,
clipping, checkpointing, resume and PCK validation
(`comparison_models/common/training_utils.py`); only the architecture, loss and
optimizer/schedule *values* differ. Both consume the identical augmentation
stream for a given epoch and seed, so a difference in their curves is
attributable to the model.

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

## Launching the definitive runs

`comparison_models/scripts/run_definitive_training.sh` starts the runs in
parallel *lanes* — sequential inside a lane, concurrent across lanes, one tmux
window each. Several lanes may name the same GPU, which is what makes the
FD-first layout work: both models peak near 2 GiB, so the two FD runs share the
fast card while the ~4x smaller OS and CS runs occupy another.

```bash
export VITPOSE_ENV_PREFIX=$HOME/venvs/vitpose-b-cu128-venv
export POSEBH_ENV_PREFIX=$HOME/venvs/posebh-b-cu128-venv
export PYTHONPATH=$PWD

# machine with the fast card: the two FD runs, one lane each so they overlap
bash comparison_models/scripts/run_definitive_training.sh --dry-run --workers 8 \
    1=vitpose-b:FD 1=posebh-b:FD

# the other machine: OS and CS, two runs per card
bash comparison_models/scripts/run_definitive_training.sh --dry-run --workers 4 \
    1=vitpose-b:OS vitpose-b:CS 0=posebh-b:OS posebh-b:CS
```

Drop `--dry-run` to launch. The script runs `cli preflight` for every requested
run *before* touching tmux, refuses a request that names the same run twice
(two trainers would fight over one `seed_2026` folder), skips any run whose
`final_model.pth` already exists, and never overwrites one. `--status` reports
DONE/PARTIAL/PENDING for all nine runs. An interrupted run is resumed with
`python -m comparison_models.cli resume --model M --scenario S`.

### Decode workers

Every BirdGaze sample is a full-resolution `cv2.imread` plus an affine warp and
a heatmap rasterisation, so training is decode-bound rather than GPU-bound: at
two workers an H100 PCIe sat at ~25 img/s for PoseBH-B while peaking at 2 GiB
of memory. The default is now `min(8, cpu_count)`, overridable per machine with
`BIRD_HKE_NUM_WORKERS`. Each lane runs one training at a time, so keep
`lanes x workers` below the core count — the launcher warns when it is not,
because oversubscribed decoders slow every run on the machine down at once. Two
FD runs in two lanes sharing one card are two lanes, so on a 16-core machine
`--workers 6` or `--workers 8` is the right range.

The worker count is part of the reproducibility contract: each worker is seeded
from the loader's base seed plus its own id, so changing it changes the
augmentation stream. It is therefore recorded in `runtime_fingerprint()` inside
the run manifest, and `--resume` refuses if it differs from the value the run
started with. Fix the count for a given run before the first epoch.

## Status and open gates

| Item | Status |
|---|---|
| Shared BirdGaze conversion + OS/CS source-crop fidelity | implemented, CPU-tested |
| OS/CS identical-export guard | implemented (`preflight.validate_os_cs_distinction`) |
| Per-model paper-faithful optimizer/schedule | implemented, CPU-tested ([details](./PAPER_FAITHFUL_PROTOCOL.md)) |
| ViTPose-B trainer | implemented; CPU-tested; **GPU smoke PASSED** (FD, real data) |
| PoseBH-B trainer | implemented; CPU-tested; **GPU smoke PASSED** (FD, real data) |
| ViTPose/PoseBH 2-epoch train+validate+resume probe | **PASSED** on GPU for both models |
| RTMPose-M paper recipe (values, optimizer build, LR trajectory, EMA) | **PASSED** via `comparison_models/rtmpose/verify_paper_recipe.py` |
| RTMPose-M FD GPU data/model preflight | implemented; **UNVERIFIED** - needs full mmcv |
| RTMPose-M end-to-end GPU training | **BLOCKED** - no mmcv 2.x build for this torch/CUDA |
| Any of the nine definitive runs | **not started** |

CPU CI being green is a merge requirement. GPU validation is a separate
full-training readiness gate. The GPU results below were produced on a single
RTX 5060 (8 GB) under WSL2 with the isolated per-model environments; they
validate the code path, not the final accuracy. None should be reported as a
result until `bash comparison_models/scripts/test_comparison_models.sh FD`
passes on the cluster.

### GPU validation already performed

Environment: WSL2 Ubuntu, Python 3.10, torch 2.7.0+cu128 (`sm_120`), one
`NVIDIA GeForce RTX 5060`. Each model ran in its own venv, satisfying
`ensure_isolated_runtime`, against the real FD export (77,740 train / 8,821
val images). The upstream sources were the pinned checkouts required by
`preflight.check_git_pin`.

| Model | Parameters | First-iteration loss | Peak GPU memory | 2-epoch probe | Resume |
|---|---|---|---|---|---|
| ViTPose-B | 85,871,620 | 0.0029182 | 1604.2 MiB | 0.0042437 → 0.0031960 | PASSED |
| PoseBH-B (single-dataset adaptation) | 94,473,796 | 0.0019805 | 1723.8 MiB | 0.0021073 → 0.0020328 | PASSED |

The probes use a small example subset, train two epochs, validate, write a full
checkpoint, restore optimizer/scheduler/AMP/RNG state from it, and delete
everything afterwards; no definitive run folder is created. Reproduce with:

```bash
python -m comparison_models.common.training_probe --model vitpose-b --scenario FD --train-examples 132 --val-examples 16
python -m comparison_models.common.training_probe --model posebh-b --scenario FD --train-examples 132 --val-examples 16
```

RTMPose-M cannot be executed in that environment: MMEngine training needs mmcv
2.x compiled with CUDA ops, and no build exists for torch 2.7 + CUDA 12.8
(`MMCV_WITH_OPS=0` produces `mmcv-lite`, which has no `mmcv._ext`). What *can*
be verified without those kernels is verified by
`comparison_models/rtmpose/verify_paper_recipe.py`, which resolves the recipe
through `_base_`, builds the real optimizer so `paramwise_cfg` is exercised,
drives a real `Runner` over all 100 epochs to confirm the warmup/cosine
trajectory, and builds the paper's `ExpMomentumEMA`. Its output:

| Check | Result |
|---|---|
| Recipe values vs pinned upstream (12 compared) | all identical; only epochs, batch, init and keypoint count differ |
| Backbone / head / preprocessor / test_cfg inheritance | matches upstream except the four documented overrides |
| `paramwise_cfg` on a built optimizer | `norm.weight`, `norm.bias`, `stem.bias`, `fc.bias` → `weight_decay=0`; conv/linear weights → `0.05` |
| `clip_grad` | `{'max_norm': 35, 'norm_type': 2}` applied by the wrapper |
| Warmup | starts at `4.000000e-08`, reaches `4.000000e-03` (`base_lr`) after 1000 iterations |
| Cosine | `4.000000e-03` at epoch 50 → `2.037492e-04` at epoch 99, floor `eta_min = 2.000000e-04 = base_lr * 0.05` |
| EMA | `ExpMomentumEMA`, `momentum=0.0002`, `gamma=2000`, `update_buffers=True`, priority 49 |

End-to-end RTMPose-M training remains gated on a machine with a full mmcv
build; the Titan procedure in
[`comparison_models/rtmpose/README.md`](../comparison_models/rtmpose/README.md)
is the route to close that gate.
