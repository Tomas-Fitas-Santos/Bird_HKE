# Isolated training preparation: ViTPose++-B and PoseBH-B

This folder starts integrating the **two remaining external architectures**
identified in `external_models/README.md`. It does **not** change or train the
six native Bird_HKE models, RTMPose-M, or the existing uncertainty pipeline.

## Scope and isolation contract

- Two pinned, clean upstream source checkouts:
  - ViTPose++-B: `.external_pose_sources/vitposepp` at
    `c050ed29112da7704797cc1a65af0234b525010d`
  - PoseBH-B: `.external_pose_sources/posebh` at
    `8f03438f913b559edb0a2beed54e77d1560edba3`
- Keep **separate Python environments** for each architecture. Neither upstream
  project may install into the native `bird_hke-cu128` or the RTMPose venv.
- Read the same original FD/OS/CS annotations and corrected, crop-aware COCO
  exports already produced by `external_models/convert_birdgaze_to_coco.py`.
- Four landmarks in order: crown, left eye, right eye, beak tip.
  Input 256×256, seed 2026, 100 epochs maximum, from-scratch training.
- Separate new output prefixes only:
  - `Bird_HKE/trained_models/External_Baselines/ViTPosePP_B/{FD,OS,CS}/seed_2026/`
  - `Bird_HKE/trained_models/External_Baselines/PoseBH_B/{FD,OS,CS}/seed_2026/`
- Never reuse native weights, RTMPose output paths, calibration artifacts,
  checkpoint manifests, training processes or shared environment packages.
- Validation images select checkpoints; the held-out annotated videos are
  reserved for the final scientific comparison.

## Preparation commands (READ-ONLY)

From the `phase-3-passive-uncertainty` line of development or this feature
branch, after fetching pinned external sources and generating COCO exports:

```bash
python -m unittest discover -s tests -p 'test_remaining_baseline_preflight.py' -v

python -m external_models.remaining_baselines.prepare --model vitposepp-b --scenario FD
python -m external_models.remaining_baselines.prepare --model posebh-b --scenario FD
```

Repeat preflight for OS/CS when their COCO exports are ready. An OS/CS
conversion collision (different source annotations but identical converted
exports) is an error, not an acceptable result.

The preflight is read-only; it confirms pinned clean upstream source revisions,
source/export counts, four-keypoint record shapes, crop metadata and checksums.
The output is a per-model data/config recipe. It does **not** import MMPose or
instantiate/train any model, so it does not touch an ongoing GPU run.

## Deliberate training gate

**Not train-ready yet.** A model-specific config/adapter with a real
forward/backward + validation GPU smoke test must be implemented before
training is enabled. The planned local paths are:

- `external_models/vitposepp/birdgaze_vitposepp_b.py`
- `external_models/posebh/birdgaze_posebh_b.py`

The optional `--reserve-run` mode will only reserve a brand-new output
directory when the specified model adapter exists; it never overwrites or
resumes an existing directory. Do not use it until the model integration has
passed the architecture-specific checks. At this stage there is deliberately
no `--train` mode or command that launches an unadapted upstream trainer.

### Remaining implementation tasks

1. Implement ViTPose++-B native backbone/head and four-keypoint BirdGaze
   dataset wrapper **without silently treating ViTPose-B as ViTPose++-B**.
2. Adapt PoseBH-B prototype learning to the four-keypoint single-dataset
   setting; check the paper's multi-dataset self-supervision and scheduling
   assumptions before claiming an equivalent architecture.
3. Build model-specific isolated environment setup, disposable GPU training
   smoke tests, strict resume/RNG restoration and guarded launchers.
4. Verify source-crop handling including signed/zero FD scales and OS/CS
   bbox differences. Final evaluation must use independently annotated videos.

This staged approach is intentional: it establishes provable write boundaries
before implementing substantial, potentially incompatible upstream models.
