# RTMPose-M BirdGaze FD — Phase 3

> **Titan installation route (2026-11-03).** Titan has RTX 3090 (SM 8.6),
> an existing working PyTorch 2.7.0+cu128 environment, and matching
> `/usr/local/cuda-12.8/bin/nvcc`. The old PyTorch 2.1 installation
> failed while downloading a 2.2 GB wheel. The Conda alternative proposed
> 2.75 GB. **Do not run `setup_rtmpose_env.sh` on Titan.**
>
> The new [`setup_rtmpose_torch27.sh`](../../setup_rtmpose_torch27.sh)
> prepares a **separate, system-site-packages virtualenv** using the already
> installed Torch/CUDA, and compiles MMCV 2.2.0 CUDA ops for SM 8.6.
> It installs no packages into the existing `bird_hke-cu128` environment.
> Compatibility of MMCV 2.2.0's C++ source with torch 2.7.0 is **not yet
> validated**. Start with the read-only `--inspect` mode. The `--build`
> mode may use considerable CPU/RAM, but avoids re-downloading PyTorch/CUDA.
> Keep the build log for troubleshooting.

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

## 2. Isolated RTMPose environment using existing CUDA 12.8

Check prerequisites without installing anything:

```bash
bash setup_rtmpose_torch27.sh --inspect
```

The inspection prints the existing torch/CUDA versions, GPU architecture,
CUDA compiler, GCC/G++, and available disk space. Do not proceed to a build
unless they are as expected.

When the checks pass, build MMCV's full CUDA operators locally in a separate
virtualenv, limiting parallel compiler jobs:

```bash
mkdir -p external_models/generated
MAX_JOBS=2 bash setup_rtmpose_torch27.sh --build \
  > external_models/generated/rtmpose_torch27_build.log 2>&1
echo "Build exit code: $?"
tail -n 70 external_models/generated/rtmpose_torch27_build.log
```

If it succeeds:

```bash
bash setup_rtmpose_torch27.sh --verify
```

If the compile/install succeeds but verification fails with
`ModuleNotFoundError: No module named 'importlib_metadata'`, **do not
repeat the MMCV build**. The 2026-11-03 first Titan attempt reached this
condition. Pull the latest Phase 3 branch and run:

```bash
bash setup_rtmpose_torch27.sh --repair
```

The first two repair attempts exposed missing YAPF dependencies, followed
by a **NumPy ABI mismatch in `xtcocotools._mask`**. Titan confirmed that
the compiled `mmcv/_ext.cpython-310-x86_64-linux-gnu.so` exists and is
23,725,704 bytes. The only remaining uncertainty about it is runtime
compatibility with the installed Torch 2.7.0; building the file is not proof
that the CUDA kernels execute.

The NumPy repair succeeded on Titan (NumPy 1.26.4 inside the overlay,
shared PyTorch 2.7.0+cu128 intact). The subsequent import revealed a
different issue: the pinned MMPose package **unconditionally imports
`RTMOHead`**, even though the BirdGaze model only uses `RTMCCHead`.
`RTMOHead` requires MMDetection; MMDetection 3.3.0 explicitly rejects
MMCV 2.2.0.

The dedicated [MMPose overlay generator](make_overlay.py) copies the pinned
MMPose Python package to
`.external_pose_sources/rtmpose_birdgaze_overlay` and removes exactly
two unused `RTMOHead` import/export statements (four minimal string edits).
It verifies the upstream source against the pinned Git commit, creates a
manifest of SHA-256 checksums, keeps an idempotent overlay, and does **not**
modify the original upstream checkout or any RTMPose-M architecture, codec,
dataset or loss. `--repair` now regenerates/verifies this overlay before
its CUDA NMS smoke test.

```bash
git pull --ff-only origin phase-3-passive-uncertainty
bash setup_rtmpose_torch27.sh --repair
```

If this prints `MMCV CUDA operators: PASS`, run the real FD preflight in
section 3. **Do not rerun the long `--build` command for an MMDetection
import error.**

The virtualenv is `/home/students/conda-envs/rtmpose-torch27-venv`.
It inherits the existing Torch 2.7.0+cu128 binary package without modifying
it, but installs MMCV, MMEngine and MMPose into the virtualenv. No large
PyTorch/CUDA binary downloads are requested. This older MMCV source may need
compatibility fixes against PyTorch 2.7; review any compiler error before
trying another dependency stack.

## 3. Mandatory disposable real-data/GPU check

```bash
RTMP_PY=/home/students/conda-envs/rtmpose-torch27-venv/bin/python

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

**Titan config-parser fix (2026-11-03):** After the CUDA setup passed,
MMEngine 0.10.7 raised `ConfigParsingError` before loading any BirdGaze data.
The custom config declared Python imports before its legacy `_base_`
inheritance, causing MMEngine to detect the wrong lazy-import format.
The config now declares `_base_` before the imports and removes temporary
`Path` variables before serialization. This change does not alter the
model, dataset processing, hyperparameters or checkpoint protocol.
Pull the updated Phase 3 branch and rerun the **same** `--check-only` command;
there is no reason to reinstall/rebuild the environment.

**Serialized-dataset preflight correction (2026-11-03):** The next
`--check-only` run reached FD COCO loading but incorrectly reported
no `birdgaze_negative_scale` records. MMEngine BaseDataset defaults to
`serialize_data=True` and clears `ds.data_list` after serialization.
The validator now uses `ds.get_data_info(index)`, verifies the flags
in the converted annotations, and checks their propagation into the
packed MMPose samples. This does **not** disable serialization or drop
any negative-/zero-scale samples. A stdlib-only regression test is
`tests/test_rtmpose_serialized_dataset_preflight.py`. Pull and rerun
preflight without reinstalling or reconverting anything.

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

The code and isolated overlay setup are committed, but the
MMCV 2.2.0 source build and **MMPose data/GPU tests have not yet passed
on Titan**. Please run the preflight and share the
log or first error. Do not launch definitive training based only on
Python syntax or unit fixture tests.

ViTPose++-B and PoseBH-B remain to be integrated next; neither is trained
by this RTMPose launcher. Once all FD models have trained, use the same
independently annotated video benchmark and consistent detection/cropping
to compare all architectures.
