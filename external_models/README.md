# External BirdGaze pose-estimation baselines

Goal: nine **single-seed, randomly initialized** experiments: RTMPose-M, ViTPose++-B, and PoseBH-B, each on FD / OS / CS. Keep the original training runs and video evaluation unchanged.

## Upstream source acquisition

From the root of the existing `phase-3-passive-uncertainty` checkout:

```bash
git pull --ff-only origin phase-3-passive-uncertainty
bash setup_external_pose_sources.sh
```

The sources are cloned to `.external_pose_sources/`, which is Git-ignored. The upstream commit IDs are pinned in the script, avoiding silent updates:

| Baseline | Upstream | Pinned Git commit |
|---|---|---|
| RTMPose-M | https://github.com/open-mmlab/mmpose | `759b39c13fea6ba094afc1fa932f51dc1b11cbf9` |
| ViTPose++-B | https://github.com/ViTAE-Transformer/ViTPose | `c050ed29112da7704797cc1a65af0234b525010d` |
| PoseBH-B | https://github.com/uyoung-jeong/PoseBH | `8f03438f913b559edb0a2beed54e77d1560edba3` |

Preserve upstream copyright/LICENSE when selectively importing any source. Do not commit full upstream checkouts, generated weights, or BirdGaze media.

## Do not mix third-party MMPose installations into the current Bird_HKE environment

RTMPose targets modern MMPose/MMEngine, whereas ViTPose++ and PoseBH contain older forked MMPose code. Use isolated environments for the upstream implementations. Keep the current Bird_HKE PyTorch/CUDA environment unchanged, including active runs.

## Work required before launching the nine training jobs

1. **BirdGaze adapters.** Use existing FD/OS/CS `annot/train.json` and `annot/val.json` and the corresponding `images/`. Map four landmarks **[crown, left eye, right eye, beak tip]**. Match the existing BirdGaze preprocessing, crops, horizontal flip (swap eyes), and annotation-availability masking. Distinguish missing coordinates from occluded coordinates that are actually annotated.
2. **RTMPose-M adapter.** Preserve CSPNeXt-M, its GAU/SimCC head, and SimCC loss/decode. Configure four output keypoints and a compatible MMPose dataset interface. Do not force HRNet heatmap loss onto this architecture.
3. **ViTPose++-B adapter.** Preserve the relevant ViTPose++ ViT-B components. Configure BirdGaze-only four-keypoint training with random weights. Document that this is *single-dataset training*, not a reproduction of the published multi-dataset pretrained configuration.
4. **PoseBH-B adaptation.** Verify prototype learning and associated losses remain well-defined with one dataset and four keypoints. The upstream PoseBH training config is multi-dataset and contains freeze/cluster scheduling. Do not claim to benchmark the published full multi-dataset method if these mechanisms cannot be preserved meaningfully.
5. **Reproducible runners.** Each model×dataset needs a fixed seed 2026, 256×256 input, correctly initialized weights, 100 epoch maximum, run-specific checkpoint directory, validation-only model selection, full model/optimizer/scheduler/RNG checkpoint restoration and a config/annotation fingerprint. Batch and optimizer must be comparable but architecture-specific heads/losses must be preserved.
6. **Shared predictions/evaluation.** Convert all keypoint predictions to original-image pixel coordinates in the existing BirdGaze keypoint order and use the same evaluation/videos, detector or ground-truth boxes, postprocessing, and metrics. Do not evaluate validation images as an independent test set.

**Readiness:** the source acquisition script only downloads pinned upstream source. Training configurations and adapters above are still required; do not treat the downloaded models as runnable BirdGaze experiments.

## Resume-policy update in this branch

`Bird_HKE/tools/train.py` now uses the checkpoint's `training_protocol_hash` as its reproducibility gate instead of checking the recorded Git commit or requiring a clean Git working tree. Model, optimizer, LR scheduler, and RNG checkpoint restoration remain strict. The code revision and runtime are still recorded as provenance. This applies to the **existing Bird_HKE trainer**; separate upstream trainers need equivalent guards if custom checkpoint resumption is required.
