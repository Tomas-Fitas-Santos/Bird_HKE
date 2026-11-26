# Codex handoff — consolidate original and comparison models; integrate into `main`

Date: 2026-11-25  
Repository: `Tomas-Fitas-Santos/Bird_HKE`  
**Integration branch already created:** `chore/consolidate-native-comparison-models-20261125`  
Base: `feat/isolated-remaining-baselines-20261122`, which is a linear descendant of `phase-3-passive-uncertainty`.  
Target: `main` **after** organization, test repair, and a reviewed PR.

## Mission

Do the implementation **in the GitHub repository**, not in a ZIP or an external handoff folder. Fully organize the research code so there is an obvious, tested boundary between:
1. **Original Bird_HKE models and native training infrastructure** (all existing research architectures, configs, uncertainty, calibration, metrics, benchmarks); and
2. **Three external comparison architectures**: RTMPose-M, standard single-task ViTPose-B, and the explicitly **single-dataset PoseBH-B adaptation**.

All three comparison models must be co-located on **one integrated branch and eventually in `main`**. The native research workflows must continue to work, and none of their existing output directories or trained weights may be moved, overwritten, or reinterpreted.

Do NOT merge to `main` before completing the implementation and at least the available CPU/CI checks. DO NOT run full 100-epoch training during the reorganization. Do NOT change an active GPU process, existing Conda environment, Python interpreter, CUDA installation, or dataset.

## Verified branch topology — do not lose previous work

GitHub comparisons on 2026-11-25:
- `main` -> `phase-3-passive-uncertainty`: **104 commits ahead, 0 behind**.
- `phase-3-passive-uncertainty` -> `feat/isolated-remaining-baselines-20261122`: **65 commits ahead, 0 behind**.
- Thus the full comparison-model lineage is **169 commits ahead of `main`, 0 behind** at inspection.
- **RTMPose-M** was introduced in the Phase-3 lineage; **ViTPose-B** and **PoseBH-B** were added on the feature branch.
- A separate `feat/annotatiathon-birdgaze-builder-20261123` branch is **51 commits ahead of `main` and diverges from model work**; preserve that work as-is and do not silently mix it into this model PR. It needs its own subsequent integration plan/PR.

Starting commands in a **clean working tree** (do not operate on an active training checkout):

```bash
git fetch origin
git switch --track origin/chore/consolidate-native-comparison-models-20261125
# If the local branch already exists:
# git switch chore/consolidate-native-comparison-models-20261125
git status --short
git log --oneline --max-count=8
git merge-base --is-ancestor origin/main HEAD
```

Do not reset `main`, squash away Phase-1/2/3 commits, force-push, delete working branches, or merge the Annotatiathon branch as an incidental step.

## Current code to reorganize (inspect all imports and callers first)

Original code and configs: `Bird_HKE/models/`, `Bird_HKE/dataset/`, `Bird_HKE/lib/`, `Bird_HKE/tools/`, `Bird_HKE/experiments/`, `Bird_HKE/main.py`, `Bird_HKE/run_all.py`.

External code, all present in the source history:
- RTMPose-M: `external_models/rtmpose/`, `setup_rtmpose_torch27.sh`, `setup_rtmpose_env.sh`, plus RTMPose tests, MMPose overlay generator, model config and runner.
- ViTPose-B: `external_models/vitpose/`, `setup_vitpose_b.sh`.
- PoseBH-B: `external_models/posebh/`, `setup_posebh_b.sh`.
- Shared source export and audit: `external_models/convert_birdgaze_to_coco.py`, `external_models/audit_birdgaze_annotations.py`, `external_models/inspect_os_cs_keypoints.py`.
- Shared checks/probes: `external_models/remaining_baselines/`, `setup_external_pose_sources.sh`, `smoke_both_birdgaze_baselines.sh`, `test_both_birdgaze_training.sh`.
- Tests currently in `tests/test_external_birdgaze_conversion.py`, `tests/test_rtmpose*.py`, `tests/test_vitpose_b.py`, `tests/test_posebh_b.py`, `tests/test_remaining_baseline_preflight.py`, `tests/test_baseline_training_probe.py`.
- Existing CI: `.github/workflows/external_baseline_static.yml`.
- Existing docs: root `README.md`, `external_models/README.md`, the three model READMEs and native training/uncertainty docs.

**Do not assume the preceding lists are exhaustive.** Search imports and literal file paths throughout code, tests, shell scripts, CI, docs, and MMEngine configs before moving anything.

## Required target architecture

Create a clean top-level comparison area (recommended canonical package `comparison_models`) while leaving the original `Bird_HKE/` tree in place.

Suggested target layout (adjust if a more robust equivalent is warranted):

```bash
Bird_HKE/
  models/                       # ORIGINAL RESEARCH MODELS; preserve paths
  dataset/
  lib/
  tools/
  experiments/
  main.py
  run_all.py

comparison_models/
  __init__.py
  README.md                      # comparison-specific navigation and status
  common/
    __init__.py
    birdgaze/                    # shared source crop/annotation/geometry/decoding
    conversion.py                # COCO export, same data contract
    preflight.py                 # pinned source, integrity, collision checks
    dependencies.py              # read-only environment validator
    training_probe.py            # isolated disposable integration test
    training_utils.py            # genuinely shared resume/lock/metrics only
    transformer.py               # reusable Attention/DropPath if identical
  rtmpose/
    ...                          # config, dataset glue, runner, overlay
    README.md
  vitpose/
    ...                          # standard ViTPose-B implementation/trainer
    README.md
  posebh/
    ...                          # honest single-BirdGaze adaptation/trainer
    README.md
  scripts/
    ...                          # model setup, pinned sources, run validations
  tests/
    ...                          # tests for all three + common contracts

docs/
  MODEL_FAMILIES.md              # original vs comparison, architecture matrix
  COMPARISON_EXPERIMENTS.md      # data protocol, metrics, no-trained-weight claims
```

Use `git mv` for tracked files where appropriate to preserve history. Do not leave copies of whole model implementations in both locations. Root-level existing commands may be preserved via **thin compatibility wrappers** (no duplicated implementation) while the primary documented entry points use the new paths. If a move endangers MMPose's module registration or legacy CLI, keep a narrow compatibility shim and test it. Do not refactor native `Bird_HKE/models/` for appearance alone.

### Important shared-code refactors

1. **Remove model-to-model imports.** PoseBH currently imports `Attention`/`DropPath` directly from `external_models.vitpose.model` and inherits a ViTPose-named dataset. Move truly common transformer primitives and BirdGaze preprocessing to `comparison_models.common`, or use a clearly documented, independently tested equivalent. Neither model should require importing another model's implementation to train.
2. Share annotation keypoint order `[crown, left_eye, right_eye, beak_tip]`, image size 256×256, heatmap grid 64×64 where appropriate, center/scale/source-bbox conversion, flip eye indices, masks for absent vs annotated-occluded joints, negative/zero FD scales, and provenance/checksum reporting.
3. **Do NOT homogenize different architecture losses/heads.** RTMPose-M must retain CSPNeXt/RTMCC SimCC and its official loss/decoder; ViTPose-B stays standard ViT-B with simple heatmap head; PoseBH-B stays its documented one-task ViTMoE/prototype adaptation.
4. Consolidate source pin checking, dataset and train/val preflight, output-dir safety, complete resume/manifest hashing, and the test protocol **without changing behavior**. Keep architecture-specific optimizer, AMP, losses, LR policy and validation where they differ.
5. Update every reference to moved files: absolute and relative path computations (especially `Path(__file__).parents`), import statements, Bash calls, Python `-m` module entry points, dataset exporters, registration side effects, external MMPose overlays, code/recipe SHA-256 fingerprints, CI path triggers, Git ignores and docs.
6. **Avoid mutable circular imports** and ensure the new common package can be imported without MMPose/MMCV/CUDA. MMPose-only imports belong solely to the RTMPose adapter/runtime.
7. Do not copy `BirdGaze_v2`, videos, annotation JSON, large archives, compiled MMCV extensions, models, checkpoints, or `.external_pose_sources` into git. Ensure `.gitignore` covers any new generated subfolders, venvs, logs, temporary checkpoints and outputs.
8. Do not invalidate or overwrite existing output roots. Maintain distinct
   `Bird_HKE/trained_models/External_Baselines/{RTMPose_M,ViTPose_B,PoseBH_B}/{FD,OS,CS}/seed_2026/` (verify RTMPose's actual legacy path and preserve it). Native model outputs stay unchanged.
9. Preserve original third-party LICENSE/copyright notices and update `THIRD_PARTY_NOTICES.md` when introducing/relocating derivatives.

## Unified, safe CLI and dependency setup

Provide one easy-to-find, small orchestration CLI without trying to make all
architectures use one framework. Suggested interface:

```bash
python -m comparison_models.cli list
python -m comparison_models.cli preflight --model all --scenario FD
python -m comparison_models.cli test --model all --scenario FD
python -m comparison_models.cli test --model all --scenario OS
python -m comparison_models.cli test --model all --scenario CS
```

`list` and `preflight` must require no GPU; `test` may use GPU. The CLI
must select each **independent Python environment**, and run GPU jobs
**sequentially**, with no accidental training launch. Keep direct per-model
entry points available for debugging. Provide separate, model-specific
`--train` and `--resume` commands, **never run these by default**.

- RTMPose-M needs its isolated PyTorch/MMEngine/MMPose/MMCV environment and
  pre-existing pinned MMPose overlay; it may NOT share ViTPose/PoseBH's venv.
- Standard ViTPose-B and the PoseBH-B one-dataset adaptation use PyTorch,
  NumPy, OpenCV and Pillow for converters; do NOT install older MMPose
  forks into their environments.
- The scripts must inspect versions and missing imports **before**
  installing anything. Never replace the working system Torch/CUDA or
  modify the native `bird_hke-cu128` environment.
- The pinned source revisions must remain the exact tested upstream
  references: RTMPose `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`;
  ViTPose `c050ed29112da7704797cc1a65af0234b525010d`;
  PoseBH `8f03438f913b559edb0a2beed54e77d1560edba3`.

## Critical science and data safety

- Exactly **nine single-seed comparison experiments**: the 3 models × FD,
  OS, CS, seed 2026, from scratch, maximum 100 epochs; no pretrained
  checkpoints, no implicit native weights.
- FD = `BirdGaze_v2/BirdGaze_Full_Dataset`; OS =
  `BirdGaze_v2/birdgaze_original_subset`; CS =
  `BirdGaze_v2/birdgaze_corrected_subset`.
- Use original `annot/train.json` to update weights; `annot/val.json`
  exclusively for checkpoint selection; separately annotated videos are
  the final test set. **Do not leak test videos into training.**
- Signed and zero scales are special cases; preserve current semantics,
  including OS/CS source-crop distinctions. Fail closed if OS/CS COCO
  artifacts become identical despite meaningful source differences.
- Outputs must be mapped back to original full-image pixel coordinates
  before common scientific evaluation; include the original source-crop
  offset and a consistent keypoint order.
- PoseBH's released method is multi-dataset and pretrained. The in-repo
  adaptation uses **one active MoE expert**, four keypoints, prototype
  learning and within-dataset consistency. Call it **`PoseBH-B
  (single-dataset adaptation)`** throughout README, plots and reports.
  Never misrepresent it as a reproduction of the paper's six-dataset
  cross-dataset CPS/CSS results.
- Do not change any existing native experiment protocol, uncertainty
  configurations, production/saved checkpoint state or relative paths
  to make the new layout look cleaner.

## Tests and validation (mandatory)

1. Review `git diff --name-status origin/main...HEAD` to enumerate
   changed native files and confirm there were no unintended rewrites.
2. Run `python -m compileall -q comparison_models Bird_HKE` and
   `bash -n` on every setup/smoke/runner wrapper.
3. Run pre-existing native tests/audits (especially the existing
   reproducibility/uncertainty protocol and the legacy original trainer),
   allowing only **documented environment-specific skips**.
4. Run all existing external/RTMPose/crop tests, plus:
   `test_remaining_baseline_preflight.py`,
   `test_vitpose_b.py`, `test_posebh_b.py`,
   `test_baseline_training_probe.py` after moving/updating them.
   CI had previously verified 38 CPU tests for the two new adapters,
   but re-check after the refactor.
5. Add tests ensuring:
   - the three model IDs are discoverable and mapped to distinct environments,
     executable entry points, output roots and checkpoints;
   - `comparison_models.common` imports without MMPose;
   - PoseBH no longer imports the ViTPose model module;
   - all three are importable/locatable in the new layout;
   - original Bird_HKE trainer and all six experiment configs remain
     referenced and executable with their existing semantics;
   - generated COCO metadata, signed-scale, occluded-keypoint and
     OS/CS crop assertions remain valid;
   - `--test` performs no writes to definitive experiment dirs.
6. Extend CI to include **all three comparison models' CPU checks**,
   not only the two new adapters. On a real GPU where available, run all
   three model-specific real-data smoke checks sequentially.
7. Use small disposable **actual training/validation/checkpoint/resume**
   probes for ViTPose and PoseBH already present in the code. Add an
   equivalent isolated integration probe for RTMPose **if feasible without
   modifying its definitive trainer state**. If not, explicitly report
   RTMPose's remaining runtime qualification gap; do not claim three
   fully validated training workflows.
8. GPU-dependent checks on Titan may remain explicitly **UNVERIFIED**
   if the environment is unavailable; do not invent successful logs.
   CPU CI being green is a **merge requirement**, GPU validation a
   **full-training readiness gate**.
9. Verify all documentation commands against the actual new entry points.
   Remove obsolete statements claiming PoseBH or ViTPose is not implemented.

## Merge to main — only after working tree and CPU CI are green

1. Commit tested refactor to
   `chore/consolidate-native-comparison-models-20261125`.
2. Open a reviewed PR **against `main`**, retaining full Phase-1,
   Phase-2, Phase-3, RTMPose, ViTPose and PoseBH history. Summarize all
   169+ inherited commits separately from the organization-only diff.
3. Resolve conflicts by preserving the complete native Phase-3 history
   and the existing original model protocol. Avoid squashing/cherry-picking
   a small subset of model commits; do not force-push `main`.
4. Merge the PR using the repo's allowed method once reviews/required
   checks pass. Verify `origin/main` contains all three comparison
   packages, all native configs and the shared validation commands.
5. Do not delete the older feature branches or close PR #2 until you
   confirm its commits are integrated. If redundant afterward, add a
   clear note linking the new main PR before closing it.
6. Keep Annotatiathon feature branch intact and note its independent
   integration path in the final report.

## Deliverables Codex MUST report

- Link to final integration PR and merged `main` commit (or exact blocker
  if GitHub permissions/protections prevent merging).
- Before/after directory tree and model/CLI navigation commands.
- Confirmation all three external models are represented, with correct
  names and PoseBH qualification.
- Dependency matrix and how to create/verify each isolated environment.
- Exact FD/OS/CS smoke and disposable training-check commands.
- CPU test/CI result with counts, skipped tests and any failure details.
- GPU qualification status for each model separately, without guessing.
- No-accidental-changes audit of original native models/training/output paths.
- List of any unresolved issues before definitive 100-epoch training.

**Definition of done:** `main` contains all native research models and the
three comparison architectures under a clear separation with valid entry
points, docs, isolated environments and reproducible tests. Any GPU work
that cannot be verified remains visibly gated rather than treated as passed.
