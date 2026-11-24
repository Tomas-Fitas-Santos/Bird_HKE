# Standard ViTPose-B — isolated BirdGaze training

**Architecture:** standard **single-task** ViTPose-B, *not* ViTPose++.
Official pinned reference: [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
at `c050ed29112da7704797cc1a65af0234b525010d`,
`mmpose/models/backbones/vit.py` and
`mmpose/models/heads/topdown_heatmap_simple_head.py`. The reference
`ViTPose_base_simple_coco_256x192.py` defines ViT-B (12 blocks, hidden
dimension 768, 12 attention heads, 16×16 patches) and zero-deconv simple
heatmap head. This implementation uses independently written PyTorch layers
matching those operations, **not** the incompatible older MMPose runtime.
It uses no published/pretrained weights or ViTPose++ task adapters.

The BirdGaze adaptation changes the *input/output task*, not the ViT-B
architecture: 256×256 top-down crops, four keypoints
`[crown,left_eye,right_eye,beak_tip]`, 64×64 Gaussian heatmaps, and
four-channel output. The official code's learned positional token slot and
its 4× ReLU/bilinear/simple-convolution head are preserved.
The 4× bilinear interpolation uses equivalent **separable matrix multiplications**
instead of CUDA's native bilinear-backward kernel to support strict deterministic
training on PyTorch/CUDA versions where that kernel is nondeterministic.
A unit test checks both forward output and backward gradient parity against
`torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`.

## Safety and reproducibility

- This code lives **only** in `external_models/vitpose/`. It does not import
  or modify `Bird_HKE/tools/train.py`, its six model architectures,
  existing checkpoints, the RTMPose trainer, or the PoseBH code.
- Use a separate Python venv named `vitpose`; inherit existing Torch CUDA
  wheels in **read-only** system site-packages. No installation into the
  original Bird_HKE environment. No MMCV, MMPose or new CUDA downloads.
- The launcher refuses to run when `nvidia-smi` reports any GPU compute PID.
  That is a pre-start check; **coordinate cluster GPU scheduling as well**,
  since new jobs may start after the check.
- It checks a pinned clean upstream ViTPose checkout and compares the
  official *single-task* backbone config, never fetching pretrained weights.
- It fingerprints the source and converted FD/OS/CS annotations, upstream
  baseline config, adapter files, scenario, and Python/CUDA/PyTorch runtime.
- Training creates a brand-new directory only, under:
  `Bird_HKE/trained_models/External_Baselines/ViTPose_B/<FD|OS|CS>/seed_2026/`.
  It refuses to overwrite an existing run; `--resume` requires a matching
  manifest and a complete epoch checkpoint.
- Last checkpoint saves model, optimizer, scheduler, AMP scaler, and
  Python/NumPy/CPU/CUDA RNG state. The run uses deterministic epoch seeds for
  data ordering and dataloader workers. Deterministic CUDA algorithms are
  required: GPU preflight must PASS rather than silently downgrading.
- No run or checkpoint is created by `--smoke-test`.

## Titan installation (one-time)

Use a **separate Git worktree** for ViTPose, so you do not switch the branch,
change dependencies or write outputs in any checkout used by other models:

```bash
cd /mnt/storage/TomasSantos/Bird_HKE
git fetch origin

git worktree add -b vitpose-local \
  /mnt/storage/TomasSantos/Bird_HKE-vitpose \
  origin/feat/isolated-remaining-baselines-20261122

cd /mnt/storage/TomasSantos/Bird_HKE-vitpose
# Reuse the original images read-only, without duplicating the dataset.
ln -s /mnt/storage/TomasSantos/Bird_HKE/BirdGaze_v2 BirdGaze_v2

# Read-only inspection.
bash setup_vitpose_b.sh --inspect

# Fetch *only* the pinned standard ViTPose source. RTMPose/PoseBH untouched.
bash setup_vitpose_b.sh --sources

# Use the Python from the existing working Torch/CUDA installation only as
# a starting interpreter. The new venv and its site-packages are independent.
export VITPOSE_BASE_PY="/path/to/working/torch-cu128/python"
bash setup_vitpose_b.sh --create
bash setup_vitpose_b.sh --verify
```

The separate interpreter is
`/home/students/conda-envs/vitpose-b-cu128-venv/bin/python` by default.
Use `VITPOSE_ENV_PREFIX` if the cluster needs a different dedicated path.

No packages are installed into the inherited environment. If dependencies
are missing, install only into the **new** venv after checking compatibility,
not into native Bird_HKE.

## Data and real-data GPU preflight

The inputs are read from the same BirdGaze FD/OS/CS source annotations and
images as the existing models. Original data is never modified. Source bbox
crops use floor/ceil/clipping before pose affine, BGR normalization matches
Bird_HKE, and coordinate supervision retains annotated occluded points.
Missing points are masked. The left/right eye indices swap on horizontal
flip. FD zero/negative scales are retained (zero-scale validation entries
cannot contribute to a scale-normalized PCK metric).

The pre-existing crop-aware converter is used **only for checksums/integrity**;
ViTPose itself reads source annotations so center/scale and crop semantics
are not lost. Generate up-to-date exports, if missing, without overwriting
native training files:

```bash
python external_models/convert_birdgaze_to_coco.py --scenario FD
# Repeat for OS and CS when preparing those runs.
```

Run static/CPU regressions with the isolated interpreter:

```bash
VIT_PY=/home/students/conda-envs/vitpose-b-cu128-venv/bin/python
$VIT_PY -m unittest discover -s tests -p 'test_remaining_baseline_preflight.py' -v
$VIT_PY -m unittest discover -s tests -p 'test_vitpose_b.py' -v
```

With the target GPU idle, launch the actual disposable forward/backward test:

```bash
$VIT_PY -m external_models.vitpose.train --scenario FD --smoke-test
```

**Do not train if the real data/CUDA smoke test fails.** This preflight also
checks the signed crops, source/COCO counts, original model configuration
and resume recipe inputs. It has no persistent output writes.

## Full training-path test before 100 epochs

The standard `--smoke-test` checks a real forward/backward and single optimizer
step, but does **not** check the production epoch loop, validation selection
or checkpoint restoration. For those, run the new **disposable two-epoch
integration test** using only 12 real training images and 8 validation images:

```bash
$VIT_PY -m external_models.remaining_baselines.check_deps
$VIT_PY -m external_models.remaining_baselines.test_training \
  --model vitpose-b --scenario FD
```

The test calls the production training epoch and validation code, saves a
complete real checkpoint, destroys the first model instance, restores the
model/optimizer/scheduler/AMP/RNG state into a fresh instance, trains and
validates one additional epoch, and verifies checkpoint/best-weight/history
integrity. All large checkpoint files are temporary under the ignored
`Bird_HKE/trained_models/External_Baselines/_Training_Probes/` path and
are deleted when the test finishes. The definitive run path remains
untouched. Make sure **at least 5 GiB of scratch disk is free**.

To test both models in sequence (after preparing both separate venvs):

```bash
bash test_both_birdgaze_training.sh FD
```

This runs dependencies, CPU unit tests, the GPU smoke test and two-epoch
training/resume probes sequentially. Do not launch production training if
either probe fails.

## Training after smoke-test PASS

```bash
$VIT_PY -m external_models.vitpose.train --scenario FD --train
```

It trains from scratch for a maximum of 100 epochs with:
seed 2026, AdamW `lr=0.0005, weight_decay=0.01`, 5-epoch warmup,
cosine decay to `1e-5`, a physical microbatch of four, accumulated
effective batch 64, clipping at norm 1.0, FP16 AMP with dynamic scaler.
The final partial accumulation group is normalized by its real sample count
at each epoch boundary.  The best checkpoint is chosen **only on image
validation PCK@0.05**, normalized by source `scale × 200`. Validation does
not update any weights. The independently annotated videos are reserved for
the final scientific model comparison.

After an interruption **after an epoch checkpoint**:

```bash
$VIT_PY -m external_models.vitpose.train --scenario FD --resume
```

The run directory contains `training_recipe.json`,
`checkpoint_last.pth`, `model_best.pth`, `training_history.jsonl`,
and `final_model.pth` after completion. The best weights are also
kept in two alternating integrity-hashed `best_slot_a.pth` /
`best_slot_b.pth` slots. If a crash interrupts writing a newer best model,
the prior epoch's best slot remains intact, and `--resume` regenerates the
canonical `model_best.pth` from the last committed checkpoint. A crash before the first checkpoint
requires manual investigation and preserving/archiving the failed run; a
new start will not overwrite it. Training refuses stale live lock files until
the operator confirms no process is running.

The remaining OS/CS ViTPose-B experiments use the exact same code with
`--scenario OS` / `--scenario CS`, their own isolated run directories
and preserved crop distinctions. **PoseBH is out of scope for this stage.**

## Limitations

- **GPU tests and definitive training have not yet been executed on Titan.**
  The repo implementation is not evidence of a successfully trained model.
- This is a from-scratch **single-dataset BirdGaze comparison**. It is not a
  reproduction of ViTPose's published COCO pretrained baseline, its COCO AP,
  or its 210-epoch COCO training schedule.
- The ViTPose output is in heatmap coordinates. Video evaluation will need a
  model-specific inference adapter converting back to original image pixels
  before using the common annotated-video benchmark, detector/boxes and
  temporal metrics. Do **not** use val images as held-out video test.
- Upstream ViTPose and MMPose code retain their original licenses. This
  implementation should be reviewed for attribution before publication.
