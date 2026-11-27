# External comparison models

This package holds the **three external pose-estimation baselines** used to
compare against the original Bird_HKE research models. It is deliberately a
separate group: nothing here is imported by `Bird_HKE/`, and nothing in
`Bird_HKE/` is imported at runtime by this package.

| Group | Location | Contents |
|---|---|---|
| Original research models | [`Bird_HKE/`](../Bird_HKE/) | HRNet, HR_Mamba, HR_MambaVision, HR_MambaViT, VHR_BirdPose (+ ViTPose variant), their uncertainty implementations, `experiments/*.yaml` configs and `trained_models/` |
| External comparison models | `comparison_models/` | RTMPose-M, standard ViTPose-B, PoseBH-B (single-dataset adaptation) |

The three comparison models share **data preprocessing only**. Each keeps its
own architecture, its own trainer, its own isolated Python environment and its
own checkpoint tree. They are never combined into one framework and never
share an interpreter.

## Layout

```
comparison_models/
  registry.py            machine-readable catalogue (ids, envs, entry points, outputs)
  cli.py                 unified, safe orchestration CLI
  common/
    birdgaze/            shared BirdGaze data contract
      conversion.py        source JSON -> COCO export
      dataset.py           crop/affine/heatmap dataset + coordinate decoding
      geometry.py          OS/CS original source-crop handling
      audit.py             read-only annotation audit
      inspect_os_cs.py     OS vs CS field comparison
    transformer.py       shared ViT primitives (Attention, DropPath, init)
    losses.py            shared masked heatmap MSE
    training_utils.py    shared epoch/AMP/checkpoint/resume protocol
    preflight.py         read-only recipe/export verification
    dependencies.py      per-environment import + CUDA report
    training_probe.py    disposable 2-epoch train/validate/resume probe
  rtmpose/               RTMPose-M (official MMPose/MMEngine)
  vitpose/               standard single-task ViTPose-B
  posebh/                PoseBH-B (single-dataset adaptation)
  scripts/               environment setup + qualification wrappers
  tests/                 CPU-only regression tests
```

## The three models

| id | Model | Environment | Entry point | Checkpoint root |
|---|---|---|---|---|
| `rtmpose-m` | RTMPose-M (official MMPose, BirdGaze recipe) | `rtmpose-torch27-venv` | `comparison_models.rtmpose.run_birdgaze` | `Bird_HKE/trained_models/External_Baselines/RTMPose_M` |
| `vitpose-b` | ViTPose-B (standard single-task baseline) | `vitpose-b-cu128-venv` | `comparison_models.vitpose.train` | `.../External_Baselines/ViTPose_B` |
| `posebh-b` | PoseBH-B (single-dataset adaptation) | `posebh-b-cu128-venv` | `comparison_models.posebh.train` | `.../External_Baselines/PoseBH_B` |

Run `python -m comparison_models.cli list` for the same table as JSON,
including the pinned upstream revisions and setup scripts.

PoseBH-B is an **adaptation**: one active MoE expert, four keypoints, three EMA
prototypes per landmark, trained from scratch on a single BirdGaze scenario. It
is **not** a reproduction of the published multi-dataset PoseBH method and its
cross-dataset CPS/CSS results, and must always be labelled
`PoseBH-B (single-dataset adaptation)` in plots and reports.

## Training fidelity

Each model is trained with **its own paper's optimizer and schedule** — base
learning rate, betas, weight decay, decay exemptions, layer-wise LR decay,
gradient clipping, warmup, decay schedule, EMA and AMP loss scale are all taken
verbatim from the pinned upstream config. The only exempted hyperparameter is
the epoch count, which Bird_HKE fixes at 100 and rescales the schedules to.

The three models still share the BirdGaze data contract, the augmentation
stream, the effective batch of 64, seed 2026 and random initialization. The
full per-model equivalence table and the list of deliberate deviations is in
[docs/PAPER_FAITHFUL_PROTOCOL.md](../docs/PAPER_FAITHFUL_PROTOCOL.md).

## Unified CLI

```bash
python -m comparison_models.cli list
python -m comparison_models.cli doctor
python -m comparison_models.cli preflight --model all --scenario FD
python -m comparison_models.cli test --model all --scenario FD
python -m comparison_models.cli train --model vitpose-b --scenario FD
python -m comparison_models.cli resume --model vitpose-b --scenario FD
```

* `list` and `doctor` import neither torch nor MMPose and touch no GPU.
* `preflight` verifies each model's isolated interpreter, package layout,
  pinned upstream checkout, MMPose overlay and BirdGaze source annotations.
  It fails closed and writes nothing.
* `test` runs each model's real-data GPU smoke check **sequentially**, inside
  that model's own interpreter. It never writes to a definitive run directory.
* `train` and `resume` are explicit, single-model, and never implied by any
  other subcommand.
* The interpreter for each model comes from `VITPOSE_ENV_PREFIX`,
  `POSEBH_ENV_PREFIX` and `RTMPOSE_ENV_PREFIX`, falling back to the paths in
  `registry.py`. The native `bird_hke-cu128` environment is never used.

Per-model entry points remain available for debugging, e.g.
`python -m comparison_models.vitpose.train --scenario FD --smoke-test`.

## Qualification before any definitive training

```bash
bash comparison_models/scripts/smoke_comparison_models.sh FD
bash comparison_models/scripts/test_comparison_models.sh FD
```

The first runs CPU tests, preflight and the sequential GPU smoke checks. The
second additionally runs the disposable 2-epoch train/validate/save/resume
probe for ViTPose-B and PoseBH-B under the ignored
`_Training_Probes/` scratch directory (5 GiB free required). Neither script
launches a 100-epoch run.

RTMPose-M supports FD, OS and CS through the shared CLI and launcher. **Known gap:** RTMPose-M has a real-data/model GPU preflight
(`comparison_models.rtmpose.validate_fd`) but no disposable optimizer/resume
mini-run harness, because its trainer is the upstream MMEngine runner. Its
optimizer, schedule and EMA are instead checked by
`comparison_models/rtmpose/verify_paper_recipe.py`, which drives a real MMEngine
`Runner` over the full 100 epochs but needs mmcv 2.x with CUDA ops for the
actual forward/backward pass. Where that build is unavailable, end-to-end
RTMPose-M training is **not** runtime-qualified and is reported as such.

## Upstream sources

```bash
bash comparison_models/scripts/setup_external_pose_sources.sh
```

Clones into the Git-ignored `.external_pose_sources/`. The pinned revisions are
exact and must not change:

| Model | Upstream | Pinned commit |
|---|---|---|
| RTMPose-M | https://github.com/open-mmlab/mmpose | `759b39c13fea6ba094afc1fa932f51dc1b11cbf9` |
| ViTPose-B | https://github.com/ViTAE-Transformer/ViTPose | `c050ed29112da7704797cc1a65af0234b525010d` |
| PoseBH-B | https://github.com/uyoung-jeong/PoseBH | `8f03438f913b559edb0a2beed54e77d1560edba3` |

RTMPose additionally needs an import-only overlay
(`bash comparison_models/scripts/setup_rtmpose_torch27.sh --repair`) that
patches **only** the unrelated `RTMOHead` registration; the upstream checkout
is verified unmodified against the pinned commit. Never install an older MMPose
fork into the ViTPose or PoseBH environments, and never modify the native
`bird_hke-cu128` environment.

## Data contract

Exactly nine single-seed experiments: 3 models x FD/OS/CS, seed 2026, from
scratch, at most 100 epochs, no pretrained weights.

* FD = `BirdGaze_v2/BirdGaze_Full_Dataset`, OS =
  `BirdGaze_v2/birdgaze_original_subset`, CS =
  `BirdGaze_v2/birdgaze_corrected_subset`.
* `annot/train.json` updates weights; `annot/val.json` is used **only** for
  checkpoint selection; the separately annotated videos are the final test set.
* Four landmarks in the fixed order `[crown, left eye, right eye, beak tip]`.
* Missing coordinates and annotated-but-occluded coordinates are distinct.
* Signed and zero crop scales keep their existing semantics.
* Predictions are mapped back to original full-image pixel coordinates before
  any shared evaluation.

Generate the shared COCO exports (Pillow only; no torch, MMPose or GPU):

```bash
python comparison_models/common/birdgaze/conversion.py --scenario FD
python comparison_models/common/birdgaze/conversion.py --scenario OS
python comparison_models/common/birdgaze/conversion.py --scenario CS
```

Outputs go to the Git-ignored `comparison_models/generated/`.

## Open data blocker: OS vs CS

A read-only audit on Titan found OS and CS exports that were byte-identical
even though the source JSON hashes differed. The cause was a dropped `bbox`
field: OS and CS share keypoint coordinates, validity, visibility, center and
scale, but differ in the **original source crop** applied before the pose
affine warp. The converter now records `birdgaze_source_bbox`, and
`common/birdgaze/geometry.py` reproduces the crop (floor/ceil, clipped to image
bounds, offset subtracted from keypoints and the pose box) before any
augmentation. `preflight.validate_os_cs_distinction` fails closed if OS and CS
artifacts become identical again.

Confirmed source-annotation counts:

| Partition | Train | Validation |
|---|---:|---:|
| FD source records | 77,740 | 8,821 |
| FD negative `scale` | 2,447 | 322 |
| FD zero `scale` | 33 | 0 |
| OS records with original `bbox` | 2,579 | 260 |
| CS records with corrected `bbox` | 19,534 | 2,302 |
| OS/CS differing `bbox` values | 19,527 | 2,301 |

FD negative scales are exported as positive box widths plus a
`birdgaze_negative_scale` flag and restored as a 180-degree rotation; zero
scales are kept with an explicit flag reproducing the original degenerate warp.
These exceptional cases still require on-cluster verification.

Regenerate and confirm OS and CS hashes now differ:

```bash
python -m unittest discover -s comparison_models/tests -t . -p 'test_external_birdgaze_conversion.py' -v
for scenario in OS CS FD; do
  python -u comparison_models/common/birdgaze/conversion.py --scenario "$scenario" \
    > "comparison_models/generated/conversion_${scenario}.log" 2>&1
  echo "$scenario exit=$?"
done
sha256sum comparison_models/generated/{OS,CS}/{train,val}_coco.json
```

## Final evaluation protocol

Image validation is for checkpoint selection only. All final checkpoints are
evaluated on the identical independently annotated held-out video collection,
in original video-frame coordinates, with the same PCK normalization,
per-keypoint errors, temporal metrics, frame-matching and missing-landmark
rules, and identical filter parameters across models. Keep video/source
leakage out of both the training and the checkpoint-selection sets. Report
image-validation scores as training diagnostics, not as the manuscript's model
comparison. Bootstrap resampling must resample video sequences, not frames.

## Tests

```bash
python -m unittest discover -s comparison_models/tests -t . -v
```

CPU-only, no GPU and no BirdGaze download required. `test_consolidation.py`
guards the two-group split: three distinct environments/entry points/output
roots, exact pinned revisions, `common` importable without MMPose, PoseBH not
importing the ViTPose model, no `external_models` imports anywhere, and no
runtime dependency between `comparison_models/` and `Bird_HKE/`.
