# Bird_HKE
Related code from the work titled: "Bird Head Keypoint Estimation for Archosaur Motion Insights". (Paper currently under review)

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

- `Bird_HKE/videos/` (input videos and annotations)
- `Bird_HKE/trained_models/` (trained checkpoints)
- `Bird_HKE/logs/` and `Bird_HKE/videos_experiments/` (generated outputs)
- Large binary weights (`*.pt`, `*.pth`)

Create these folders locally and place your own data/models before running experiments.

## 1) Environment setup

From the `GITHUB_REPO` root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

If you use Mamba-based models, also install `mamba-ssm` for your CUDA/PyTorch platform.

## 2) Train a new model

Training is config-driven. Choose one YAML in `Bird_HKE/experiments/...` and run:

```bash
python Bird_HKE/tools/train.py --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_CS.yaml
```

For finetuning (if your workflow uses it):

```bash
python Bird_HKE/tools/finetune.py --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_CS.yaml
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
  --cfg Bird_HKE/experiments/HRNet/hrnet_w32_birdgaze_CS.yaml \
  --video Bird_HKE/videos/video_annotations/ColumbaPalumbus/ColumbaPalumbus.mp4 \
  --gt Bird_HKE/videos/video_annotations/ColumbaPalumbus/annot/ColumbaPalumbus.json \
  --write_obj --write_pose --filter_type pose
```

You can omit `--gt` for videos without ground-truth annotations.

## Expected local layout for running experiments

```text
GITHUB_REPO/
  READ_ME.md
  requirements.txt
  Bird_HKE/
    main.py
    run_all.py
    experiments/
    tools/
    lib/
    models/
    dataset/
    videos/            # create locally
    trained_models/    # create locally
```

  ## Acknowledgements

  We acknowledge and thank the original authors of MambaVision for their open-source release:

  - MambaVision (NVIDIA): https://github.com/nvlabs/mambavision

  In this repository, HR-MambaVision refers to our own integration and experimentation code built for this work.

  We also acknowledge the original VHR-BirdPose authors whose code we adapted for the VHR-BirdPose branch:

  - VHR-BirdPose: https://github.com/LuoXishuang0712/VHR-BirdPose

  The VHR-BirdPose-related files in this repository preserve their upstream notices and remain subject to the licenses of their original components.
