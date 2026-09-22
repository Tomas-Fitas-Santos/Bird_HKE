# Bird_HKE passive uncertainty output

## Contract

Uncertainty is an auxiliary output attached to the pose model. Enabling it must
not change the keypoint estimation pipeline:

- the same image and augmentation reach the pose network;
- the same four heatmaps are trained with the original visibility-masked MSE;
- auxiliary inputs are detached from the pose graph;
- pose and auxiliary parameters use separate optimizer and gradient-clipping
  groups;
- the same heatmap argmax and quarter-pixel post-processing decode coordinates;
- calibration changes confidence and spatial uncertainty only, never the pose.

The smoke test and unit tests compare pose gradients and decoded coordinates
between the baseline and auxiliary-output paths. FD uncertainty runs use the
`bird_hke_repro_v3` protocol and `repro_v3/uncertainty/` directories so they
cannot resume checkpoints produced by the earlier pose-changing design. CS and
OS remain baseline-only `bird_hke_repro_v2` experiments.

## Outputs

For each estimated keypoint the model returns:

| Output | Meaning | Effect on pose |
|---|---|---|
| `location_logits` | Original MSE-trained pose heatmap | This is the pose output |
| `quality` | Calibrated probability that the decoded point is correct at training-log PCK@0.5 | None |
| `visibility_probability` | Probability that the point is visually observable where a visibility label exists | None |
| `probability_maps` and spatial statistics | Softmax representation of the frozen pose heatmap, temperature-calibrated after training | None |
| conformal region | Post-hoc keypoint region fitted on `annot/calibration.json` | None |

Quality and visibility are separate concepts. The keypoint confidence returned
beside the coordinates is quality only; it is not multiplied by visibility.

## How the auxiliary head learns

The reliability head follows the central idea of [On the Calibration of Human
Pose Estimation](https://arxiv.org/abs/2311.17105): learn localization quality
from pose features instead of interpreting the raw heatmap peak as confidence.
Its target is binary correctness under the same metric printed as `Accuracy`
during training. With the default heatmap metric this is PCK@0.5 after
normalization by heatmap size divided by ten, equivalent to a 3.2-pixel radius
on a 64 by 64 heatmap.

The head consumes detached global features, detached probability-weighted local
features, map entropy/peak, and keypoint identity. Only the head parameters
receive quality and visibility gradients. It has no dropout, so it cannot
advance the random-number stream used by a stochastic pose layer. Head
construction also uses an isolated random-number stream so adding the head does
not change pose initialization.

The head must still be trained to provide meaningful confidence. It can be
trained concurrently through this detached path, as implemented here, or in a
separate second stage with a frozen pose model. A pose-only checkpoint does not
contain a trained reliability head and cannot produce learned confidence merely
by enabling the option at inference time.

## Annotation contract

Coordinate availability and visibility are separate masks:

| Annotation case | Pose MSE | Visibility loss |
|---|---:|---:|
| Visible point with coordinates | yes | target 1 |
| Hidden point without coordinates | no | target 0 |
| Hidden point with coordinates | yes | target 0 |
| eBird coordinate without a visibility field | yes | masked |

No uncertainty-specific image augmentation is applied. In particular, the old
synthetic-occlusion path was removed because it changed the images used to
train the pose network.

## Calibration and reporting

After selecting `model_best.pth` using `val.json`,
`tools/calibrate_uncertainty.py` fits:

- a temperature for quality confidence;
- a temperature for visibility probability;
- a heatmap temperature and per-keypoint conformal mass for spatial regions.

These use only `annot/calibration.json` and never update model weights or
coordinates. Calibration artifacts are tied to the checkpoint SHA-256.

On annotated evaluation data report PCK/NME independently from uncertainty,
then report quality ECE, Brier score, error-confidence correlation,
risk-coverage curves, conformal coverage, and region area. Calibration-set
diagnostics are implementation checks; generalization claims require annotated
examples not used to fit calibration. For unannotated videos, uncertainty can
support triage and filtering but cannot establish ground-truth accuracy.

## Research basis

- [On the Calibration of Human Pose Estimation](https://arxiv.org/abs/2311.17105)
  motivates a learned, separate localization-quality prediction.
- [ProbPose](https://openaccess.thecvf.com/content/CVPR2025/html/Purkrabek_ProbPose_A_Probabilistic_Approach_to_2D_Human_Pose_Estimation_CVPR_2025_paper.html)
  motivates normalized per-keypoint spatial representations. Bird_HKE uses
  this only as a passive, post-hoc-calibrated representation; it does not adopt
  ProbPose's pose-training loss.
- [Conformal Keypoint Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Object_Pose_Estimation_With_Statistical_Guarantees_Conformal_Keypoint_Detection_and_CVPR_2023_paper.pdf)
  motivates calibrated spatial prediction regions.
- [Deep Ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
  motivates between-seed disagreement as epistemic uncertainty.
