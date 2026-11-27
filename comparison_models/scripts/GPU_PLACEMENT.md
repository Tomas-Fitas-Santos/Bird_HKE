# Placing the definitive runs on four GPUs

The rule: **each FD run gets a card to itself; OS and CS share a card, away from
FD.** FD is ~4x the data of OS and CS combined (77,740 train images vs 2,579 /
19,534), so it sets the wall clock and must not compete for SMs.

## Cards

`GPU=` in the launcher is a **CUDA** index, which is not necessarily the
`nvidia-smi` index. On cluster2 they are reversed: CUDA 0 is the H100 NVL and
CUDA 1 is the H100 PCIe. Confirm before launching:

```bash
python -c "import torch;[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

| Host | CUDA idx | Card | Role |
|---|---|---|---|
| cluster2 | 1 | H100 PCIe | `vitpose-b:FD` alone |
| cluster2 | 0 | H100 NVL | `posebh-b:FD` alone |
| cluster1 | 0 | RTX A6000 | all four OS/CS runs, one sequential lane |
| cluster1 | 1 | RTX 3080 | spare |

Both models peak near 2 GiB, so memory is never the constraint — SM contention
is. That is why the FD runs are separated even though they fit on one card.

## cluster2 — the two FD runs

```bash
cd ~/pose_estimation_clean/Bird_HKE
export VITPOSE_ENV_PREFIX=$HOME/venvs/vitpose-b-cu128-venv
export POSEBH_ENV_PREFIX=$HOME/venvs/posebh-b-cu128-venv
export COMPARISON_CLI_PYTHON=$(which python3)

bash comparison_models/scripts/run_definitive_training.sh --dry-run --workers 8 \
    1=vitpose-b:FD 0=posebh-b:FD
bash comparison_models/scripts/run_definitive_training.sh --workers 8 \
    1=vitpose-b:FD 0=posebh-b:FD
```

The NVL holds other users' memory but its SMs are idle, and our runs need only
~2 GiB each, so it is a fine home for one FD run. If the capacity guard reports
low free memory, that is the other users' allocation, not ours.

## cluster1 — OS and CS

Small enough to finish in hours. One lane, four runs, sequential:

```bash
cd ~/Bird_HKE
export VITPOSE_ENV_PREFIX=$HOME/venvs/vitpose-b-cu128-venv
export POSEBH_ENV_PREFIX=$HOME/venvs/posebh-b-cu128-venv
export COMPARISON_CLI_PYTHON=$HOME/miniconda3/envs/mamba/bin/python

bash comparison_models/scripts/run_definitive_training.sh --dry-run --workers 6 \
    0=vitpose-b:OS vitpose-b:CS posebh-b:OS posebh-b:CS
bash comparison_models/scripts/run_definitive_training.sh --workers 6 \
    0=vitpose-b:OS vitpose-b:CS posebh-b:OS posebh-b:CS
```

## Checking progress

```bash
bash comparison_models/scripts/run_definitive_training.sh --status
env -u LD_LIBRARY_PATH tmux attach -t birdhke-train    # Ctrl-b n cycles lanes
```

## Two things to keep consistent

* **`--workers` is part of the reproducibility contract.** Each dataloader
  worker is seeded from the loader seed plus its id, so the count feeds the
  augmentation RNG stream and is recorded in the run manifest. Resume refuses
  if it changed. Use the same value every time you launch a given run.
* **A run is tied to the GPU model it started on**, for the same reason — the
  device name is in the manifest. Launch each run on its final card from the
  start and it never matters.

If an aborted launch left a `seed_2026` directory behind, `reserve()` refuses to
overwrite it; delete that directory before relaunching the run.

## The remaining speed lever (optional, not applied)

PoseBH-B runs `microbatch: 2` — 32 sequential micro-steps per optimizer step —
which is why it tops out near 27 img/s on an H100 while ViTPose-B
(`microbatch: 4`) reaches ~66. Raising it is the biggest FD speedup available.
It is not a pure configuration change: PoseBH-B has `BatchNorm2d` in its neck
and deconvolution heads, so batch statistics change with the micro-batch and
the trained model changes with it. ViTPose-B has only `LayerNorm`, so for it the
effect is limited to fp16 accumulation order.

`microbatch` lives in `recipe.py`, which is hashed into the manifest, so a
change applies to fresh runs only. Nothing in `comparison_models/` shares hashed
files with native `Bird_HKE/`, so no comparison-model change can affect the
already-trained research models.

Measure first — `bottleneck_probe` is read-only, excluded from
`IMPLEMENTATION_FILES`, and cannot invalidate a live run:

```bash
CUDA_VISIBLE_DEVICES=0 $POSEBH_ENV_PREFIX/bin/python \
  -m comparison_models.common.bottleneck_probe --model posebh-b --scenario FD
```
