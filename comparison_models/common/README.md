# Isolated training preparation: ViTPose-B and PoseBH-B

This folder provides shared read-only preparation for the **two additional architectures**
identified in `comparison_models/README.md`. It does **not** modify or itself train the
six native Bird_HKE models, RTMPose-M, or the existing uncertainty pipeline.

## Scope and isolation contract

- Two pinned, clean upstream source checkouts:
  - ViTPose-B: `.external_pose_sources/vitpose` at
    `c050ed29112da7704797cc1a65af0234b525010d`. The official single-task
    reference config is `configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_simple_coco_256x192.py`.
  - PoseBH-B: `.external_pose_sources/posebh` at
    `8f03438f913b559edb0a2beed54e77d1560edba3`
- Keep **separate Python environments** for each architecture. Neither upstream
  project may install into the native `bird_hke-cu128` or the RTMPose venv.
- Read the same original FD/OS/CS annotations and corrected, crop-aware COCO
  exports already produced by `comparison_models/common/birdgaze/conversion.py`.
- Four landmarks in order: crown, left eye, right eye, beak tip.
  Input 256×256, seed 2026, 100 epochs maximum, from-scratch training.
- Separate new output prefixes only:
  - `Bird_HKE/trained_models/External_Baselines/ViTPose_B/{FD,OS,CS}/seed_2026/`
  - `Bird_HKE/trained_models/External_Baselines/PoseBH_B/{FD,OS,CS}/seed_2026/`
- Never reuse native weights, RTMPose output paths, calibration artifacts,
  checkpoint manifests, training processes or shared environment packages.
- Validation images select checkpoints; the held-out annotated videos are
  reserved for the final scientific comparison.

## Preparation commands (READ-ONLY)

From the `phase-3-passive-uncertainty` line of development or this feature
branch, after fetching pinned external sources and generating COCO exports:

```bash
python -m unittest discover -s comparison_models/tests -t . -p 'test_remaining_baseline_preflight.py' -v

python -m comparison_models.common.preflight --model vitpose-b --scenario FD
python -m comparison_models.common.preflight --model posebh-b --scenario FD
```

Repeat preflight for OS/CS when their COCO exports are ready. An OS/CS
conversion collision (different source annotations but identical converted
exports) is an error, not an acceptable result.

The preflight is read-only; it confirms pinned clean upstream source revisions,
source/export counts, four-keypoint record shapes, crop metadata and checksums.
For ViTPose-B it also statically checks the official single-task ViT-B backbone
(768 embedding, 12 layers, 12 attention heads, simple heatmap head).
Training will use BirdGaze-specific 256×256 input rather than the reference
COCO config's 256×192. The adaptation must keep the ViTPose-B architecture
while updating input/positional embedding handling and the four output joints.
The output is a per-model data/config recipe. It does **not** import MMPose or
instantiate/train any model, so it does not touch an ongoing GPU run.

## Implementation status and GPU gate

Both isolated adapters and guarded training launchers are now implemented:

- Standard ViTPose-B: [guide](../vitpose/README.md)
- **PoseBH-B single-dataset adaptation:** [guide](../posebh/README.md)

PoseBH's official published method depends on six heterogeneous datasets and
pretrained ViTPose++ weights. The code here adapts MoE and prototype losses to
a **single BirdGaze dataset, from random initialization**; published
cross-dataset CPS/CSS and pretrained results are **not reproduced**.
This distinction must be clearly disclosed when comparing results.

The prepared paths are:

- `comparison_models/vitpose/recipe.py`
- `comparison_models/posebh/recipe.py`

Both implement **separate** model architectures, Python environments,
run roots, manifests, strict resume and no-write real-data GPU smoke tests.
Before starting any 100-epoch training runs, run unit and GPU smoke tests
for **both** models sequentially:

```bash
bash comparison_models/scripts/smoke_comparison_models.sh FD
```

Repeat for OS and CS after validating the corrected crop-aware exports.
A smoke failure is a blocker; no demo or fallback trainer should run.
**No Titan smoke/100-epoch results have yet been verified.**
