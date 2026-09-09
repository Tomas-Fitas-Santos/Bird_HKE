# Bird_HKE uncertainty modelling

## Why the old heatmap score is not used

The maximum value or entropy of an MSE-trained heatmap is not a calibrated
probability. Its scale can change with architecture, loss balance, and image
content without a corresponding change in localization error. This matches the
Bird_HKE experiments in which heatmap-derived confidence did not predict the
actual keypoint error, and the analysis in [On the Calibration of Human Pose
Estimation](https://arxiv.org/abs/2311.17105).

When `UNCERTAINTY.ENABLED` is false, the repository retains that original
heatmap path only as the experimental baseline. It is not presented as reliable
uncertainty.

## Implemented design and its sources

| Component in Bird_HKE | Research idea used | Adaptation for bird-head landmarks |
|---|---|---|
| Normalized 2D distribution for every keypoint | [ProbPose (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Purkrabek_ProbPose_A_Probabilistic_Approach_to_2D_Human_Pose_Estimation_CVPR_2025_paper.html) treats keypoint location as a probability rather than an arbitrary heatmap score. | The existing final logits are normalized over the 64 by 64 grid with softmax by default; sparsemax remains an experiment option. |
| Expected bounded keypoint-similarity risk | ProbPose optimizes a task-aligned expected keypoint similarity over the spatial distribution. | Bird_HKE uses a bounded Gaussian similarity in heatmap coordinates, with one shared scale fraction across the four head points. This trains the complete distribution, not only its peak. |
| Explicit quality and visibility predictions | [CCNet](https://arxiv.org/abs/2311.17105) predicts pose accuracy and visibility from penultimate pose features rather than treating heatmap maxima as confidence. | A lightweight shared head consumes global features, probability-weighted local features, map entropy/peak, and a learned keypoint identity. It predicts expected localization similarity and visibility separately. |
| Split-conformal spatial regions | [Conformal Keypoint Detection (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Object_Pose_Estimation_With_Statistical_Guarantees_Conformal_Keypoint_Detection_and_CVPR_2023_paper.pdf) calibrates keypoint prediction sets to a requested marginal coverage. | `calibrate_uncertainty.py` fits a finite-sample highest-probability-density mass for each of the four landmarks using only `annot/calibration.json`. The resulting region can be non-circular and multi-modal. |
| Multiple independently trained seeds | [Deep Ensembles](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) use independently initialized models to improve predictive uncertainty and expose distribution shift. | `ensemble_uncertainty` averages probability maps and decomposes total coordinate covariance into within-model (aleatoric) and between-model (epistemic) parts. Use seeds 2026, 2027, and 2028 after validating one seed end to end. |

This is an adaptation, not a claim that Bird_HKE exactly reproduces any one
paper. In particular, ProbPose's out-of-window state is not implemented yet:
the present data has no consistent target saying that an annotated point lies
outside the bird crop. Adding an unsupervised outside class would create a
number that cannot be validated. It should be added only when the future fully
annotated image/video data supplies an explicit in-frame/out-of-frame target.

## Annotation contract

The loader no longer treats localization availability and visibility as the
same variable.

| Annotation case | Localization loss | Visibility loss |
|---|---:|---:|
| Visible point with coordinates | yes | target 1 |
| Current hidden point with no coordinates | no | target 0 |
| Future hidden point with coordinates | yes | target 0 |
| eBird point with coordinates but no `joints_vis` field | yes | masked |
| Synthetically masked point | yes, original coordinate retained | target 0 |

If present, a `joints_valid` array is the authoritative coordinate-availability
mask. Otherwise a finite, non-negative coordinate other than the `[0, 0]`
sentinel is considered annotated. `joints_vis` is never invented for eBird.

## Training behavior

The uncertainty mode changes the location objective, so baseline and
uncertainty runs are separate experimental conditions. Both use the same
images, split membership, augmentations, optimizer protocol, seed set, and
model-selection validation data.

In the checked-in Phase-2 experiment matrix, uncertainty is enabled only for
the six FD configurations. Every CS and OS configuration keeps the original
heatmap/MSE baseline, and the training-protocol audit rejects a config or run
directory that violates this scenario policy. Before a full FD job, run
`tools/smoke_test_uncertainty.py` with that YAML on its intended training
machine. A pass proves that one real batch can complete the probabilistic
forward pass, loss, backward pass, and optimizer update on that environment; it
does not replace the full validation and post-hoc calibration stages.

By default, gradients from the quality/visibility auxiliary head are stopped at
the shared feature map. The head is still learned in the same loop, but it
cannot improve its own loss by distorting the pose backbone. The probabilistic
location loss does update the complete pose network. Set
`RELIABILITY_GRADIENT_TO_BACKBONE: true` only as a documented ablation.

## Calibration and what can be claimed

After training, `calibrate_uncertainty.py` fits three temperatures (location,
quality, and visibility) and four conformal HPD mass thresholds. It also writes
sample counts, NLL, ECE, and Brier diagnostics. These diagnostics on the same
calibration split are implementation checks, not independent evidence that the
uncertainty generalizes.

For the paper, calibration must be evaluated on annotated examples not used to
fit the temperatures or conformal thresholds. The final annotated videos are
the preferred evaluation domain. Report at least:

- error-versus-uncertainty rank correlation per keypoint and source;
- risk-coverage/selective PCK curves, including area under the risk-coverage curve;
- quality ECE and Brier score;
- empirical 90% conformal coverage and mean region area per keypoint;
- visible versus occluded results when labels exist;
- in-domain versus new-species/video-domain results;
- all accuracy metrics both before and after uncertainty modelling.

For unannotated video frames, calibrated uncertainty can support triage,
filtering, and transparent reporting, but it does not become a substitute for
ground-truth accuracy. A defensible statement is that the model assigns a
calibrated prediction region under the exchangeability conditions validated on
the annotated evaluation data—not that an unannotated prediction is proven
correct.
