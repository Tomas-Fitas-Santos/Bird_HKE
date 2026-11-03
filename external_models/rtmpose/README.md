# RTMPose-M BirdGaze FD — Phase 3

This is the **first** of the three external architectures to be integrated.
The implementation is committed to the existing `phase-3-passive-uncertainty`
branch; no parallel setup branch is used. Do not start definitive training
until the actual Titan data/GPU preflight passes.

## Architecture and study protocol

- Upstream: official pinned MMPose RTMPose-M with CSPNeXt-M backbone and
  RTMCCHead / SimCC prediction and KLDiscretLoss.
- Four BirdGaze landmarks: crown, left eye, right eye, beak tip.
- From scratch: no pretrained weights or externally initialized checkpoint.
- Input: 256 × 256 RGB; only `BirdGaze_v2/BirdGaze_Full_Dataset/annot/train.json`
  is used for weight updates; `val.json` is for best-checkpoint selection.
- Train for 100 epochs; seed 2026; AdamW 0.0005 and weight decay 0.01;
  5-epoch linear warmup, cosine to 0.00001; norm clipping 1.0;
  physical microbatch 8 and **effective optimizer batch 64** (one GPU).
- Native coordinate supervision masking, original source crop offset,
  eye-swap under horizontal flip, clipped-normal scale/rotation augmentation,
  and BirdGaze's exceptional negative/zero-scale crop treatment.
- Epoch-aligned sample-weighted gradient accumulation: with 77,740 FD train
  records, 1,214 full groups of 64 plus one final group of 44 per epoch.
- Native model checkpoint selection uses image-validation PCK@0.05.
  **Final scientific model comparisons use the separate annotated video test
  set**, never the image validation score as a claimed final result.
- MMEngine checkpoints include optimizer, LR scheduler, Python/NumPy/torch
  RNG states through `BirdGazeRNGCheckpointHook`; resume is allowed from
  completed epoch checkpoints when the recipe hashes/versions match.

The reference source is the pinned MMPose checkout at
`.external_pose_sources/rtmpose` (commit
`759b39c13fea6ba094afc1fa932f51dc1b11cbf9`).
All code and configuration changes are kept in this repo, not in the upstream
checkout. This implementation requires a separate MMPose 1.x environment.

## 1. Checkout and data conversion (run on Titan)

Run commands in an ordinary interactive terminal; **do not run `set -e`
in your terminal**. Log long commands to files.

```bash
cd /mnt/storage/TomasSantos/Bird_HKE
git pull --ff-only origin phase-3-passive-uncertainty
python -m unittest discover -s tests -p 'test_rtmpose_accumulation.py' -v
python -m unittest discover -s tests -p 'test_external_birdgaze_conversion.py' -v

mkdir -p external_models/generated
python -u external_models/convert_birdgaze_to_coco.py --scenario FD \
  > external_models/generated/conversion_FD.log 2>&1
echo "FD conversion exit code: $?"
tail -n 35 external_models/generated/conversion_FD.log
```

The FD export should contain the complete 77,740 train and 8,821 validation
records present in the Titan dataset, including 2,447/322 negative-scale rows
and 33/0 zero-scale rows (train/val). A different dataset version may have
different counts; the runtime verifies equality with the source JSON.

## 2. Isolated RTMPose environment

First inspect the GPU compute architecture:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

The provided pinned prebuilt wheel recipe targets **Ampere/Hopper**
(compute capabilities below 10), not Blackwell. A Blackwell GPU needs a
newer CUDA/PyTorch/MMCV compatibility solution. Do not touch the existing
`/home/students/conda-envs/bird_hke-cu128` environment.

```bash
bash setup_rtmpose_env.sh > external_models/generated/setup_rtmpose.log 2>&1
echo "Environment setup exit code: $?"
tail -n 55 external_models/generated/setup_rtmpose.log

bash setup_rtmpose_env.sh --verify
```

The isolated environment is at
`/home/students/conda-envs/rtmpose-cu121`.
The known compatible stack is Python 3.10, PyTorch 2.1.0+cu121,
MMCV 2.1.0 (full CUDA-ops wheel), MMEngine 0.10.7, MMDetection 3.2.0,
and the pinned MMPose source. No compiler/toolkit install is required
for the known MMCV wheel.

## 3. Mandatory disposable real-data/GPU check

```bash
RTMP_PY=/home/students/conda-envs/rtmpose-cu121/bin/python

$RTMP_PY -m external_models.rtmpose.run_birdgaze --scenario FD --check-only \
  > external_models/generated/rtmpose_fd_preflight.log 2>&1
echo "Preflight exit code: $?"
tail -n 65 external_models/generated/rtmpose_fd_preflight.log
```

This checks dataset counts/no silent COCO filtering, real FD samples with
negative/zero scales, SimCC targets, a disposable full RTMPose-M
forward/backward/optimizer step, and validation decoding. It creates no
definitive run directory and changes no checkpoint. Fix any real runtime
failures before starting training.

## 4. Definitive FD training (only after preflight PASS)

```bash
$RTMP_PY -m external_models.rtmpose.run_birdgaze --scenario FD \
  > external_models/generated/rtmpose_fd_training.log 2>&1
echo "Training exit code: $?"
tail -n 75 external_models/generated/rtmpose_fd_training.log
```

The one official run writes only under
`Bird_HKE/trained_models/External_Baselines/RTMPose_M/FD/seed_2026`.

To continue after an interrupted epoch from the latest committed checkpoint:

```bash
$RTMP_PY -m external_models.rtmpose.run_birdgaze --scenario FD --resume \
  >> external_models/generated/rtmpose_fd_training.log 2>&1
echo "Resume exit code: $?"
tail -n 75 external_models/generated/rtmpose_fd_training.log
```

If a failure happened **before the first checkpoint** and the old recipe
manifest exists but no checkpoint was ever created, use `--retry-start`
to restart from random initialization. It will refuse if a checkpoint is
present. Existing native Bird_HKE checkpoints are never modified.

## Readiness and outstanding work

The code and isolated setup are committed, and the compatible MMCV wheel is
publicly available, but **this RTMPose integration has not yet passed
MMPose data/GPU tests on Titan**. Please run the preflight and share the
log or first error. Do not launch definitive training based only on
Python syntax or unit fixture tests.

ViTPose++-B and PoseBH-B remain to be integrated next; neither is trained
by this RTMPose launcher. Once all FD models have trained, use the same
independently annotated video benchmark and consistent detection/cropping
to compare all architectures.
