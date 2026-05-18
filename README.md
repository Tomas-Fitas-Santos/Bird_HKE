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

Edit `Bird_HKE/experiments/run_config.json`:

- Keep only the wanted entries in `configs`
- Keep only the wanted entries in `videos`
- Adjust `flags` (for example `--write_obj`, `--write_pose`)
- Optionally change `filter_type`

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
    trained_models/    # create locally
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
