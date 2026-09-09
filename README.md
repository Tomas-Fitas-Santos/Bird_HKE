# Bird_HKE
Related code from the work titled: "Bird Head Keypoint Estimation for Archosaur Motion Insights". (Paper currently under review)

## Technical setup used in this work

- CUDA version: 12.8
- GPUs used:
  - NVIDIA GeForce RTX 5060: training of smaller models (HRNet, HR-MambaViT, HR-Mamba) and all evaluations.
  - NVIDIA H100: training of larger models ([VHR-BirdPose](https://github.com/LuoXishuang0712/VHR-BirdPose), HR-MambaVision (integrating [MambaVision](https://github.com/nvlabs/mambavision))).

## License and third-party code

This repository contains a mix of:

- Original code developed for this work
- Third-party code and derivatives from the MambaVision project

Important:

- The MambaVision-derived code is distributed under the NVIDIA Source Code License-NC.
- That license is non-commercial and applies to the corresponding files/folders.
- Keep the original copyright and license notices intact.

For path-level details and attribution, see `THIRD_PARTY_NOTICES.md`.

## Included in this repository

- Core code under `Bird_HKE/`:
  - `main.py` for a single simulation
  - `run_all.py` for batch simulations
  - `tools/`, `lib/`, `models/`, `dataset/`
  - `experiments/` YAML configs and `run_config.json`
- Project-level setup files:
  - `requirements.txt`
  - `.gitignore`

## Not included (intentionally)

To keep the repository lightweight and code-focused, these are excluded:

- `BirdGaze_v2/` (dataset used for training and evaluation)
- `Bird_HKE/trained_models/` (all trained models for this work)
- `Bird_HKE/logs/` and `Bird_HKE/videos_experiments/` (generated outputs)
- Large binary weights (`*.pt`, `*.pth`)

Create these folders locally and place your own data/models before running experiments.

### BirdGaze_v2 dataset details

BirdGaze_v2 includes:

- Full Original Dataset (FD): complete original annotations.
- Corrected Subset (CS): manually corrected subset for improved annotation quality.
- Original Subset (OS): same subset as CS but with original (uncorrected) annotations.
- eBird evaluation videos and associated annotations.
- eBird images/videos used in this work.

External-source media policy:

- eBird-origin media is distributed in BirdGaze_v2.
- Animal Kingdom, Birdsnap, and NABirds media are not redistributed; users must obtain them from original sources and follow their licenses.

For complete dataset structure and reconstruction steps, see the dataset README at `BirdGaze_v2/README.md`.

Download links:

- BirdGaze_v2 dataset ([Zenodo](https://doi.org/10.5281/zenodo.20241043))
- Trained models ([Google Drive](https://drive.google.com/file/d/1p1UCx_fpSFJxBxNxaT0qNn0L7vB-9t_t/view?usp=sharing))

## 1) Environment setup

This project was developed with Conda environments and validated on Linux-based systems.

- Linux: supported/recommended.
- Windows: use WSL (Windows Subsystem for Linux) as an alternative.
- macOS: WSL is not available on macOS; Linux compatibility is not guaranteed.

From the `GITHUB_REPO` root:

```bash
conda create -n bird_hke python=3.10 -y
conda activate bird_hke

# Install PyTorch stack for your specific system/CUDA first.
# Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Then install the remaining project dependencies.
pip install -r requirements.txt
```

Important compatibility note:

- PyTorch packages (`torch`, `torchvision`, `torchaudio`) are system-dependent (OS, CUDA, driver), so install them according to your platform.
- `mamba-ssm` depends on the installed PyTorch version and CUDA toolchain, so install a compatible release for your environment.
- [Mamba-SSM release list](https://github.com/state-spaces/mamba/releases)

For Mamba-based models, install `mamba-ssm` after PyTorch is installed and verified.

## 2) Train a new model

Training is config-driven. Choose one YAML in `Bird_HKE/experiments/...` and run:

```bash
python Bird_HKE/tools/train.py --cfg Bird_HKE/experiments/HR_Mamba/hr_mamba_CS_sum.yaml
```

### Controlled and reproducible training protocol

All experiment YAMLs use the same `bird_hke_repro_v2` protocol and read only
from the dataset's existing `annot/` folder. Dataset construction is a separate,
one-time process and is intentionally not part of this repository's recurring
training tools. Each dataset used here must already contain:

```text
dataset_root/
  annot/
    train.json
    val.json
    calibration.json
  images/
```

Image paths in the JSON records are relative to `images/` and must not begin
with `images/`. `train.json` updates weights, `val.json` selects checkpoints,
and `calibration.json` is used only after uncertainty training. The external
videos remain the final test set; the image validation and calibration splits
are not reported as a final test set.

Before starting a training campaign, verify the protocol from the repository root:

```bash
python Bird_HKE/tools/audit_training_protocol.py
python -m unittest discover -s tests
```

The shared baseline protocol fixes the following values across architectures:

- input/heatmap size: 256 x 256 / 64 x 64; Gaussian sigma: 2
- random initialization from scratch (no pretrained checkpoint)
- RGB input, horizontal flip, 0.25 scale jitter, and 30-degree rotation
- foreground-weighted, coordinate-availability-masked heatmap MSE
- AdamW, learning rate `5e-4`, weight decay `0.01`
- 100 epochs, 5 warm-up epochs, cosine decay to `1e-5`
- gradient clipping at 1.0
- seed 2026, deterministic PyTorch/cuDNN behavior, and TF32 disabled
- physical batch size 8 per GPU and effective optimizer batch size 64

The accumulation count is calculated at runtime. For example, one GPU uses
eight accumulation steps, two GPUs use four, and four GPUs use two. This lets
larger models run on stronger or multi-GPU machines without changing the
per-GPU BatchNorm batch or the effective optimizer batch. The effective batch
must divide exactly; incompatible GPU counts fail before training begins.

Each run writes `resolved_config.yaml` and `environment.json` to its log
directory. The latter records the Git revision, protocol hash, package/runtime
versions, GPU names, and resolved batch plan. Checkpoints also contain all RNG
states and the protocol hash, so an interrupted run resumes from the next epoch
with the same sampling and augmentation stream. Strict mode rejects legacy or
incompatible checkpoints instead of silently mixing protocols.

The CS and OS configs write to `repro_v2/baseline/seed_2026`; the FD configs
write to `repro_v2/uncertainty/seed_2026`. This preserves previously trained
models and prevents baseline and uncertainty checkpoints from being mixed.
`TRAIN.RESUME_FROM_CKPT: true` is safe within those directories: it resumes
only a matching reproducible run. For an independent repeat, use another seed
and separate output directories. All architectures in one comparison must use
the same seed set; seeds 2026, 2027, and 2028 are a reasonable three-run
campaign for reporting mean and standard deviation.

Deterministic settings and recorded environments make runs scientifically
reproducible, but bit-for-bit equality between different GPU architectures is
not guaranteed by CUDA. Comparisons should therefore use the same protocol and
seed set and report variation across seeds.

### FD uncertainty-aware training

The Phase-2 experiment policy enables uncertainty for all six Full Original
Dataset (FD) configs and disables it for every Corrected Subset (CS) and
Original Subset (OS) config. `audit_training_protocol.py` enforces this policy,
including the separate output namespaces. Run it before starting a campaign:

```bash
python Bird_HKE/tools/audit_training_protocol.py
```

On each training machine, run a one-batch forward/backward smoke test for the
FD model before starting its full job. It uses the configured physical batch
size and real training data, checks normalized probability maps, all UQ output
shapes, finite loss, and gradients in both the pose network and reliability
head. It does not write checkpoints or logs:

```bash
python Bird_HKE/tools/smoke_test_uncertainty.py \
  --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_FD.yaml
```

If it prints `Uncertainty training smoke test: PASS`, start the full run with
the same YAML:

```bash
python Bird_HKE/tools/train.py \
  --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_FD.yaml
```

Repeat those two commands with each of the six FD YAMLs. If the dataset is not
at the path stored in the YAML, append `DATASET.ROOT /absolute/dataset/path` to
both commands. Do not add `UNCERTAINTY.ENABLED true` to CS or OS runs.

When enabled, each keypoint output is a normalized spatial probability map.
Training minimizes expected bounded keypoint-similarity risk and jointly learns
two per-keypoint auxiliary outputs: localization quality and visibility.
Localization is supervised wherever a coordinate exists, including future
fully annotated occlusions. Visibility is supervised only where the JSON has a
visibility label; eBird samples without that field are masked from this loss.
Landmark-centred synthetic occlusions provide additional hidden-point examples
without inventing coordinates.

After selecting the best checkpoint with `val.json`, fit post-hoc temperatures
and per-keypoint split-conformal highest-density regions using only
`annot/calibration.json`:

```bash
python Bird_HKE/tools/calibrate_uncertainty.py \
  --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_FD.yaml \
  --checkpoint Bird_HKE/trained_models/BirdGaze_Full_Original_Dataset/HRNet_w32_birdgaze/repro_v2/uncertainty/seed_2026/model_best.pth \
  --output Bird_HKE/trained_models/BirdGaze_Full_Original_Dataset/HRNet_w32_birdgaze/repro_v2/uncertainty/seed_2026/uncertainty_calibration.json
```

For video inference with that model, set `UNCERTAINTY.CALIBRATION_FILE` to the
matching calibration JSON; the FD YAML already selects uncertainty and the
corresponding `TEST.POSE_MODEL_FILE`. Never combine a calibration file with a
different architecture, seed, or checkpoint; the video loader verifies the
stored checkpoint SHA-256. Uncertainty-enabled video runs write
`uncertainty_initial.json` with per-frame, per-keypoint quality, visibility
probability, covariance, major/minor standard deviation, entropy, and region
area. This file is produced even when the video has no ground truth.

See `Bird_HKE/UNCERTAINTY.md` for the design rationale, literature mapping,
annotation rules, limitations, and the calibration evidence to report.

For finetuning (if your workflow uses it):

```bash
python Bird_HKE/tools/finetune.py --cfg Bird_HKE/experiments/HR_Mamba/hr_mamba_CS_sum.yaml
```

Notes:

- Set dataset location in the selected YAML (`DATASET.ROOT`).
- Set output/checkpoint folders in YAML (`TRAIN.CKPT_DIR`, `TRAIN.LOG_DIR`).
- Set test model path in YAML (`TEST.POSE_MODEL_FILE`) for evaluation/inference.

## 3) Test all models on all videos (`run_all.py`)

`run_all.py` executes the full matrix `(config x video)` from `Bird_HKE/experiments/run_config.json`.

```bash
python Bird_HKE/run_all.py
```

Useful options:

```bash
# Validate paths and print commands only
python Bird_HKE/run_all.py --dry-run

# Use a custom run config file
python Bird_HKE/run_all.py --config Bird_HKE/experiments/run_config.json
```

### Run only selected simulations

Edit `Bird_HKE/experiments/run_config.json` to customize your experiments:

- `configs`: keep only the desired model configurations
- `videos`: keep only the target video files
- `filter_type`: optionally specify a motion filter

Then run `python Bird_HKE/run_all.py` again.

## 4) Run a single simulation (`main.py`)

Use `main.py` when you want one specific experiment:

```bash
python Bird_HKE/main.py \
  --cfg Bird_HKE/experiments/HR_Mamba/hr_mamba_CS_sum.yaml \
  --video BirdGaze_v2/eBird_videos_eval/annotated_videos/ColumbaPalumbus/ColumbaPalumbus.mp4 \
  --gt BirdGaze_v2/eBird_videos_eval/annotated_videos/ColumbaPalumbus/annot/ColumbaPalumbus.json \
  --write_obj --write_pose --filter_type one_euro
```

You can omit `--gt` for videos without ground-truth annotations.

## Expected local layout for running experiments

```text
Bird_HKE_REPOSITORY/
  READ_ME.md
  requirements.txt
  BirdGaze_v2/                     # create locally (download from Zenodo)
    README.md
    birdgaze_full_dataset/
      annot/
      images/
    birdgaze_corrected_subset/
      annot/
      images/
    birdgaze_original_subset/
      annot/
      images/
    eBird_videos_eval/
      annotated_videos/
      non_annotated_videos/
  Bird_HKE/
    main.py
    run_all.py
    experiments/
    tools/
    lib/
    models/
    dataset/
    trained_models/    # create locally (download from Google Drive)
```

  ## Acknowledgements

  We acknowledge and thank the original authors of MambaVision for their open-source release:

  - [MambaVision](https://github.com/nvlabs/mambavision) (NVIDIA)

 In this repository, HR-MambaVision refers to our own integration and experimentation code built for this work.

  We also acknowledge and thank the authors of Mamba:

  - [Mamba-SSM](https://github.com/state-spaces/mamba)

  We also acknowledge the original VHR-BirdPose authors whose code we adapted for the VHR-BirdPose branch:

  - [VHR-BirdPose](https://github.com/LuoXishuang0712/VHR-BirdPose)

  The VHR-BirdPose-related files in this repository preserve their upstream notices and remain subject to the licenses of their original components.

If you encounter any issues during environment setup or while using the code, please contact the authors at tomas.santos.work2002@gmail.com or tomas.dos.santos@tecnico.ulisboa.pt.
