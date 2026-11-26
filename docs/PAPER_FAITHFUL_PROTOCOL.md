# Paper fidelity of the three comparison models

Every training hyperparameter of the three external baselines is taken from the
corresponding author's own config, at the commit pinned in
[`comparison_models/registry.py`](../comparison_models/registry.py). The single
exempted hyperparameter is the **number of epochs**, which Bird_HKE fixes at
100 for all three models.

Anything that is *not* a training hyperparameter — the dataset, the input
resolution, the number of keypoints, the augmentation stream and the
initialisation policy — is a Bird_HKE experimental-design decision and is
listed under [Deliberate deviations](#deliberate-deviations).

## Pinned sources

| Model | Upstream config used as the reference | Commit |
|---|---|---|
| RTMPose-M | `projects/rtmpose/rtmpose/animal_2d_keypoint/rtmpose-m_8xb64-210e_ap10k-256x256.py` (open-mmlab/mmpose) | `759b39c13fea6ba094afc1fa932f51dc1b11cbf9` |
| ViTPose-B | `configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py` (Vitae-transformer/ViTPose) | `c050ed2` |
| PoseBH-B | `configs/.../mdt/vitb_posebh.py` + `mmcv_custom/layer_decay_optimizer_constructor.py` (uyoung-jeong/PoseBH) | `8f03438f913b559edb0a2beed54e77d1560edba3` |

## Optimizer and schedule

Values are transcribed verbatim. "Rescaled" means the upstream value was
converted to the 100-epoch budget; the conversion is shown so it can be
re-checked.

| Setting | RTMPose-M paper | RTMPose-M here | ViTPose-B paper | ViTPose-B here | PoseBH-B paper | PoseBH-B here |
|---|---|---|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW | AdamW | AdamW | AdamW |
| Base LR | `4e-3` | `4e-3` | `5e-4` | `5e-4` | `1e-3` | `1e-3` |
| Betas | `(0.9, 0.999)` | `(0.9, 0.999)` | `(0.9, 0.999)` | `(0.9, 0.999)` | `(0.9, 0.999)` | `(0.9, 0.999)` |
| Weight decay | `0.05` | `0.05` | `0.1` | `0.1` | `0.1` | `0.1` |
| Decay exemptions | `norm_decay_mult=0`, `bias_decay_mult=0`, `bypass_duplicate=True` | same | 1-D / `.bias` / `pos_embed` exempt | same | 1-D / `.bias` / `pos_embed` exempt | same |
| Layer-wise LR decay | none (CNN) | none | `LayerDecayOptimizerConstructor`, `num_layers=12`, `layer_decay_rate=0.75` | same | same | same |
| Gradient clipping | `max_norm=35` | `max_norm=35` | `max_norm=1.` | `1.0` | `max_norm=1.` | `1.0` |
| Warmup | `LinearLR start_factor=1e-5`, 1000 iterations | same | `linear`, `warmup_iters=500`, `warmup_ratio=0.001` | same | same | same |
| Decay schedule | cosine, second half, `eta_min=base_lr*0.05`, iter-based | `begin=50`, `end=100`, `T_max=50` (rescaled from 105/210) | `policy='step'`, `step=[170, 200]` of 210 | `step=[81, 95]` of 100 (170/210→81, 200/210→95) | `policy='step'`, `step=[50, 90]` of 100 | `[50, 90]` unchanged |
| Schedule granularity | per iteration | per iteration | per iteration | per iteration | per iteration | per iteration |
| EMA | `ExpMomentumEMA`, `momentum=0.0002`, `update_buffers=True`, `priority=49` | same | none | none | none | none |
| AMP | TF32 off, no fp16 loss scale | same | none | fp16 dynamic GradScaler | `fp16 loss_scale init_scale=16384.0` | `grad_scaler_init_scale=16384.0` |
| Gradient clip order | unscale → clip → step | same | unscale → clip → step | same | unscale → clip → step | same |
| Epochs | 210 | **100** (exempt) | 210 | **100** (exempt) | 100 | 100 |
| Effective batch | 512 (64 × 8 GPUs) | **64** (see below) | 512 (64 × 8 GPUs) | **64** | 1024 (128 × 8 GPUs) | **64** |
| Seed | `21` (explicit in the config) | `2026` (Bird_HKE-wide) | upstream default | `2026` | upstream default | `2026` |

## Where the implementation lives

* **RTMPose-M** — [`comparison_models/rtmpose/birdgaze_rtmpose_m.py`](../comparison_models/rtmpose/birdgaze_rtmpose_m.py)
  inherits the upstream config through `_base_` and overrides only the fields
  above. MMEngine's own `AdamW`, `paramwise_cfg`, `LinearLR`,
  `CosineAnnealingLR`, `clip_grad` and `EMAHook` are used, so the semantics are
  upstream's, not a reimplementation.
* **ViTPose-B / PoseBH-B** — [`comparison_models/common/training_utils.py`](../comparison_models/common/training_utils.py)
  * [`layer_decay_param_groups()`](../comparison_models/common/training_utils.py)
    reproduces `LayerDecayOptimizerConstructor`: `patch_embed*`/`pos_embed` →
    layer 0, `blocks.N` → layer `N+1`, everything else (heads, necks) → the last
    layer; `lr_scale = rate ** (num_layers - layer_id - 1)` with
    `num_layers = depth + 2`; 1-D, `.bias` and `pos_embed` tensors get
    `weight_decay = 0`. It is re-expressed rather than imported because our
    modules are not wrapped in a `backbone.` prefix.
  * [`PaperScheduleLR`](../comparison_models/common/training_utils.py)
    is stepped once per optimizer step: linear warmup from `warmup_ratio` over
    `warmup_iters` iterations, then `gamma ** (milestones passed)` with
    milestones at `milestones_epochs * iters_per_epoch`.
  * [`make_optimizer()`](../comparison_models/common/training_utils.py) and
    [`make_scheduler()`](../comparison_models/common/training_utils.py) select
    the paper path when a recipe carries `schedule` / `lr_schedule` blocks, and
    fall back to the flat AdamW + epoch-cosine schedule used by the CPU smoke
    recipes.
  * [`make_grad_scaler()`](../comparison_models/common/training_utils.py) uses
    the recipe's `grad_scaler_init_scale`, so PoseBH starts AMP at 16384 like
    upstream.

These tests pin the values and the behaviour:
[`comparison_models/tests/test_paper_faithful_training.py`](../comparison_models/tests/test_paper_faithful_training.py),
plus `test_lr_schedule` in
[`test_vitpose_b.py`](../comparison_models/tests/test_vitpose_b.py) and
`test_schedule_and_gpu_busy_guard` in
[`test_posebh_b.py`](../comparison_models/tests/test_posebh_b.py).

## Deliberate deviations

These are not oversights. Each one is a Bird_HKE control that keeps the three
baselines comparable with each other and with the native research models.

| Deviation | Why |
|---|---|
| **100 epochs** for all three models | Explicitly exempted by the study protocol; the paper schedules are rescaled to it rather than dropped. |
| **Effective batch 64** on one GPU (upstream 512/1024) | The handoff fixes one optimizer batch for every comparison model so the runs are directly comparable and fit a single GPU. `auto_scale_lr` is therefore disabled for RTMPose and `base_lr` is used verbatim instead of being linearly scaled. |
| **Bird_HKE augmentation** (clipped-normal scale/rotation, eye-swap on horizontal flip, signed/zero crop-scale handling) instead of each paper's own pipeline | Shared preprocessing is the whole point of the comparison; per-model augmentation would confound architecture with augmentation. Consequences: RTMPose's two-stage `PipelineSwitchHook` (stage 2 at epoch 180 of 210) has no second stage to switch to and is not used, and ViTPose's COCO `HalfBody`/`urandom` translation and scale ranges do not apply to BirdGaze crops. |
| **Random initialisation, no pretrained weights** | Bird_HKE compares architectures trained from scratch on BirdGaze. Runtime guards reject pretrained initialization: `init_cfg=None` for the RTMPose backbone, `backbone.pretrained is None` in both recipes, and an assertion in the PoseBH trainer. |
| **Seed 2026** everywhere | Bird_HKE-wide reproducibility convention. |
| **PoseBH-B is a single-dataset adaptation** | One active MoE expert, four keypoints, no cross-dataset CPS/CSS. Upstream's `ProtoFreezeHook(50)`, `MultiheadFreezeHook`, `BackboneFreezeHook`, `CPSInitHook(50)` and `CSSEnableHook(50)` drive that multi-dataset machinery and are therefore not applicable. Always label it `PoseBH-B (single-dataset adaptation)`. |
| **fp16 dynamic GradScaler for ViTPose-B** | Upstream trains ViTPose-B in fp32. The shared AMP path is kept for memory reasons; it does not change the optimizer, the schedule or the clip norm. |
| **Validation every 10 epochs** (RTMPose `val_interval`) | Matches the upstream `evaluation interval=10`; kept, and it is also what the two custom trainers do. |

## Verification status

| Item | Status |
|---|---|
| Optimizer/schedule values transcribed from pinned upstream configs | done, reviewed line by line against the upstream files |
| Layer-decay constructor and iteration schedule unit-tested | done, CPU suite |
| Recipe values pinned by tests | done |
| ViTPose-B real GPU run with the paper optimizer/schedule | done — smoke and 2-epoch train/validate/resume probe on FD, RTX 5060 |
| PoseBH-B real GPU run with the paper optimizer/schedule | done — smoke and 2-epoch train/validate/resume probe on FD, RTX 5060 |
| RTMPose-M config loaded and executed | partially — `comparison_models/rtmpose/verify_paper_recipe.py` resolves the config through `_base_`, builds the real optimizer (so `paramwise_cfg` is exercised), drives a real `Runner` over all 100 epochs and builds the paper's `ExpMomentumEMA`. Full MMEngine training additionally needs mmcv 2.x with CUDA ops, for which no build exists for this torch/CUDA pair. |
| End-to-end RTMPose-M GPU training | **outstanding** — see [Status and open gates](./COMPARISON_EXPERIMENTS.md#status-and-open-gates) |

The exact upstream values the tests and scripts above are checked against:

| Model | Commit | Values read from the upstream config |
|---|---|---|
| ViTPose-B | `c050ed2` | AdamW `5e-4`, betas `(0.9, 0.999)`, `weight_decay=0.1`, `layer_decay_num_layers=12`, `layer_decay_rate=0.75`, `grad_clip max_norm=1.`, `warmup policy='linear' warmup_iters=500 warmup_ratio=0.001`, `step=[170, 200]` of 210, `gamma=0.1` |
| PoseBH-B | `8f03438` | AdamW `1e-3`, `weight_decay=0.1`, same layer-decay constructor and rate, `grad_clip max_norm=1.`, same warmup, `step=[50, 90]` of 100, `gamma=0.1`, `fp16 loss_scale init_scale=16384.0` |
| RTMPose-M | `759b39c` | AdamW `lr=4e-3`, `weight_decay=0.05`, `clip_grad max_norm=35 norm_type=2`, `paramwise_cfg norm_decay_mult=0 bias_decay_mult=0 bypass_duplicate=True`, `LinearLR start_factor=1e-5` over 1000 iterations, `CosineAnnealingLR eta_min=base_lr*0.05` `begin=105 end=210 T_max=105` iter-based, `EMAHook ExpMomentumEMA momentum=0.0002 update_buffers=True priority=49`, `val_interval=10` |
