"""Check the BirdGaze RTMPose-M recipe against the pinned upstream AP-10K recipe.

RTMPose-M cannot be trained on every machine: it needs mmcv 2.x compiled with
CUDA ops, and no such build exists for every torch/CUDA combination. This
script verifies the parts of the paper's training method that do not need those
kernels, using only ``mmengine`` and ``mmcv-lite``:

* the recipe resolves from the pinned upstream config through ``_base_``;
* the optimizer, weight decay, parameter-wise exemptions and gradient clipping
  match the paper, and MMEngine actually applies the exemptions;
* a real ``Runner`` driven over all 100 epochs reproduces the paper's warmup and
  cosine trajectory, reaching ``base_lr`` after warmup and ``base_lr * 0.05``
  at the end;
* the EMA hook is the paper's ``ExpMomentumEMA`` with the paper's settings.

Run from the repository root after fetching the pinned sources::

    bash comparison_models/scripts/setup_external_pose_sources.sh
    python comparison_models/rtmpose/verify_paper_recipe.py
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UPSTREAM_SOURCE = REPO_ROOT / ".external_pose_sources" / "rtmpose"
UPSTREAM_CONFIG = (
    UPSTREAM_SOURCE / "projects" / "rtmpose" / "rtmpose" / "animal_2d_keypoint"
    / "rtmpose-m_8xb64-210e_ap10k-256x256.py"
)
OUR_CONFIG = REPO_ROOT / "comparison_models" / "rtmpose" / "birdgaze_rtmpose_m.py"
SCENARIO = os.environ.get("BIRDGAZE_SCENARIO", "FD")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAILED: {message}")


def _load_configs():
    from mmengine.config import Config

    _require(UPSTREAM_CONFIG.is_file(), f"missing pinned upstream config {UPSTREAM_CONFIG}; "
                                       "run comparison_models/scripts/setup_external_pose_sources.sh")
    os.environ["BIRDGAZE_SCENARIO"] = SCENARIO
    os.environ.setdefault("BIRD_HKE_REPO_ROOT", str(REPO_ROOT))

    # `custom_imports` registers our mmpose subclasses, which needs the full
    # mmpose/mmdet stack. Let those imports fail so every training-method value
    # still resolves; the classes themselves are checked by the CPU suite.
    source = OUR_CONFIG.read_text(encoding="utf-8")
    _require("allow_failed_imports=False" in source,
             "expected birdgaze_rtmpose_m.py to forbid failed custom imports")
    rewritten = (
        source.replace("allow_failed_imports=False", "allow_failed_imports=True")
        # The temp copy sits outside the repo, so make the _base_ path absolute.
        .replace('"../../.external_pose_sources/',
                 f'"{(REPO_ROOT / ".external_pose_sources").as_posix()}/')
    )
    temp = Path(tempfile.gettempdir()) / "birdgaze_rtmpose_m_lenient.py"
    temp.write_text(rewritten, encoding="utf-8")
    return Config.fromfile(UPSTREAM_CONFIG), Config.fromfile(temp)


def check_values(up, ours) -> None:
    print("=== training method: upstream AP-10K vs BirdGaze recipe ===")
    rows = [
        ("optimizer type", up.optim_wrapper.optimizer.type, ours.optim_wrapper.optimizer.type),
        ("base_lr", up.optim_wrapper.optimizer.lr, ours.optim_wrapper.optimizer.lr),
        ("weight_decay", up.optim_wrapper.optimizer.weight_decay,
         ours.optim_wrapper.optimizer.weight_decay),
        ("paramwise_cfg", dict(up.optim_wrapper.paramwise_cfg),
         dict(ours.optim_wrapper.paramwise_cfg)),
        ("clip_grad", dict(up.optim_wrapper.clip_grad), dict(ours.optim_wrapper.clip_grad)),
        ("EMA hook", dict(up.custom_hooks[0]), dict(ours.custom_hooks[1])),
        ("warmup", dict(up.param_scheduler[0]), dict(ours.param_scheduler[0])),
        ("cosine eta_min", up.param_scheduler[1].eta_min, ours.param_scheduler[1].eta_min),
        ("cosine start fraction", up.param_scheduler[1].begin / up.train_cfg.max_epochs,
         ours.param_scheduler[1].begin / ours.train_cfg.max_epochs),
        ("cosine T_max fraction", up.param_scheduler[1].T_max / up.train_cfg.max_epochs,
         ours.param_scheduler[1].T_max / ours.train_cfg.max_epochs),
        ("cosine convert_to_iter_based", up.param_scheduler[1].convert_to_iter_based,
         ours.param_scheduler[1].convert_to_iter_based),
        ("val_interval", up.train_cfg.val_interval, ours.train_cfg.val_interval),
    ]
    mismatches = 0
    for name, upstream_value, our_value in rows:
        same = upstream_value == our_value
        mismatches += not same
        print(f"  {'SAME' if same else 'DIFF'}   {name:28s} upstream={upstream_value!r}  ours={our_value!r}")
    print(f"  (exempt)     epochs           upstream={up.train_cfg.max_epochs}  ours={ours.train_cfg.max_epochs}")
    print(f"  (deviation)  effective batch  upstream={up.train_dataloader.batch_size * 8}  "
          f"ours={ours.train_batch_size * ours.optim_wrapper.accumulative_counts}")
    print(f"  (deviation)  backbone init    upstream=pretrained checkpoint  ours={ours.model.backbone.init_cfg!r}")
    print(f"  (deviation)  keypoints        upstream={up.model.head.out_channels}  ours={ours.model.head.out_channels}")
    _require(mismatches == 0, f"{mismatches} training-method values differ from the paper")


def check_architecture(up, ours) -> None:
    print("\n=== architecture inherited from the upstream config ===")
    expected = {
        "backbone": {"_scope_", "init_cfg", "norm_cfg"},
        "head": {"out_channels"},
        "data_preprocessor": set(),
        "test_cfg": set(),
    }
    for key, allowed in expected.items():
        upstream_part, our_part = dict(up.model[key]), dict(ours.model[key])
        diff = {k for k in set(upstream_part) | set(our_part)
                if upstream_part.get(k) != our_part.get(k)}
        _require(diff <= allowed,
                 f"{key} differs from upstream in unexpected fields: {sorted(diff - allowed)}")
        print(f"  {key}: matches upstream except {sorted(diff) or 'nothing'}")


def _tiny_model():
    import torch
    import torch.nn as nn
    from mmengine.model import BaseModel

    class Model(BaseModel):
        """Small model with the norm/bias/weight naming paramwise_cfg keys on."""

        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 8, 3)
            self.norm = nn.BatchNorm2d(8)
            self.fc = nn.Linear(8, 4)

        def _predict(self, inputs):
            return self.fc(self.norm(self.stem(torch.stack(inputs))).flatten(2).mean(-1))

        def loss(self, inputs, data_samples):
            return {"loss": self._predict(inputs).square().mean()}

        def forward(self, inputs, data_samples=None, mode="tensor", **_ignored):
            if mode == "loss":
                return self.loss(inputs, data_samples)
            return self._predict(inputs)

    return Model


def check_optimizer(ours) -> None:
    from mmengine.optim import build_optim_wrapper

    print("\n=== optimizer built from our config (paramwise_cfg applied) ===")
    model_class = _tiny_model()
    model = model_class()
    wrapper = build_optim_wrapper(model, dict(
        type="OptimWrapper",
        optimizer=dict(copy.deepcopy(ours.optim_wrapper.optimizer)),
        paramwise_cfg=dict(ours.optim_wrapper.paramwise_cfg),
        clip_grad=dict(ours.optim_wrapper.clip_grad)))
    decay = {}
    for group in wrapper.param_groups:
        for param in group["params"]:
            for name, candidate in model.named_parameters():
                if candidate is param:
                    decay[name] = group["weight_decay"]
    for name in sorted(decay):
        print(f"  {name:12s} weight_decay={decay[name]}")
    print(f"  clip_grad kwargs: {wrapper.clip_grad_kwargs}")
    expected_decay = ours.optim_wrapper.optimizer.weight_decay
    _require(decay["norm.weight"] == 0.0 and decay["norm.bias"] == 0.0,
             "norm_decay_mult=0 not applied")
    _require(decay["stem.bias"] == 0.0 and decay["fc.bias"] == 0.0,
             "bias_decay_mult=0 not applied")
    _require(decay["stem.weight"] == expected_decay and decay["fc.weight"] == expected_decay,
             "conv/linear weights should keep the paper's weight decay")
    _require(dict(wrapper.clip_grad_kwargs or {}) == dict(ours.optim_wrapper.clip_grad),
             "gradient clipping not applied to the wrapper")
    print("  exemptions and clipping match the paper")


def check_schedule(ours) -> None:
    import torch
    from mmengine.dataset import BaseDataset
    from mmengine.hooks import Hook
    from mmengine.registry import DATASETS, HOOKS, MODELS
    from mmengine.runner import Runner

    print("\n=== schedule driven by a real Runner over the full run ===")
    # ExpMomentumEMA lives in mmpose's registry, whose package __init__ pulls in
    # mmcv's compiled ops. Import the single module and register it directly.
    from mmpose.engine.hooks.ema_hook import ExpMomentumEMA
    MODELS.register_module(module=ExpMomentumEMA, force=True)

    @DATASETS.register_module(force=True)
    class TinyDataset(BaseDataset):
        def load_data_list(self):
            return [{"inputs": torch.randn(3, 8, 8), "data_samples": {}}
                    for _ in range(400)]

    MODELS.register_module(module=_tiny_model(), name="TinyRTMPose", force=True)

    recorded = {}

    @HOOKS.register_module(force=True)
    class Recorder(Hook):
        priority = "LOWEST"

        def before_train(self, runner):
            recorded["pre"] = runner.optim_wrapper.param_groups[0]["lr"]

        def before_train_epoch(self, runner):
            recorded[runner.epoch] = runner.optim_wrapper.param_groups[0]["lr"]

    ow = dict(
        type="OptimWrapper",
        optimizer=dict(copy.deepcopy(ours.optim_wrapper.optimizer)),
        paramwise_cfg=dict(ours.optim_wrapper.paramwise_cfg),
        clip_grad=dict(ours.optim_wrapper.clip_grad))
    epochs = ours.train_cfg.max_epochs
    with tempfile.TemporaryDirectory() as work_dir:
        runner = Runner(
            model=dict(type="TinyRTMPose"),
            work_dir=work_dir,
            train_dataloader=dict(
                batch_size=2, dataset=dict(type="TinyDataset", pipeline=[]),
                sampler=dict(type="DefaultSampler", shuffle=True), num_workers=0),
            # val_interval is raised only so this schedule simulation does not
            # need a validation set; the configured value is compared above.
            train_cfg=dict(by_epoch=True, max_epochs=epochs, val_interval=10 ** 6),
            optim_wrapper=ow,
            param_scheduler=list(ours.param_scheduler),
            custom_hooks=[dict(type="Recorder"), dict(ours.custom_hooks[1])],
            default_hooks=dict(logger=None, checkpoint=None),
            randomness=dict(seed=ours.randomness.seed),
        )
        runner.train()
        built = [type(s).__name__ for s in runner.param_schedulers]
        ema = [h for h in runner._hooks if type(h).__name__ == "EMAHook"]

    base_lr = ours.optim_wrapper.optimizer.lr
    eta_min = ours.param_scheduler[1].eta_min
    cosine_begin = ours.param_scheduler[1].begin
    print(f"  schedulers built: {built}")
    print(f"  lr before epoch 0:            {recorded['pre']:.6e}  (base_lr {base_lr:.1e})")
    print(f"  lr at epoch 1:                {recorded[1]:.6e}")
    print(f"  lr at epoch 5 (post-warmup):  {recorded[5]:.6e}")
    print(f"  lr at cosine begin {cosine_begin}:     {recorded[cosine_begin]:.6e}")
    print(f"  lr at final epoch {epochs - 1}:        {recorded[epochs - 1]:.6e}")
    print(f"  cosine eta_min:               {eta_min:.6e}  (base_lr * 0.05 = {base_lr * 0.05:.6e})")
    _require(built == ["LinearLR", "CosineAnnealingLR"], "unexpected scheduler pair")
    _require(recorded["pre"] < base_lr, "warmup should start below base_lr")
    _require(abs(recorded[5] - base_lr) < 1e-6, "warmup should reach base_lr")
    _require(recorded[5] >= recorded[cosine_begin] > recorded[epochs - 1],
             "cosine decay should be monotone")
    _require(recorded[epochs - 1] >= eta_min * 0.999,
             "final lr should sit at the paper's cosine floor")
    _require(eta_min == base_lr * 0.05, "eta_min should be base_lr * 0.05")
    _require(ema and ema[0].ema_model is not None, "EMAHook did not build its EMA model")
    print(f"  EMA model: {type(ema[0].ema_model).__name__}, momentum={ema[0].ema_model.momentum}, "
          f"gamma={getattr(ema[0].ema_model, 'gamma', 'n/a')}, "
          f"update_buffers={ema[0].ema_model.update_buffers}")


def main() -> int:
    try:
        import mmengine  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "mmengine is required. Use the RTMPose environment described in "
            "comparison_models/rtmpose/README.md (mmengine plus mmcv-lite is enough "
            "for this check; full mmcv is only needed to train).")
    if not (REPO_ROOT / "comparison_models" / "generated" / SCENARIO / "train_coco.json").is_file():
        raise SystemExit(
            f"Run convert_birdgaze_to_coco.py for scenario {SCENARIO} first, or set "
            "BIRDGAZE_SCENARIO to a converted scenario.")
    up, ours = _load_configs()
    check_values(up, ours)
    check_architecture(up, ours)
    check_optimizer(ours)
    check_schedule(ours)
    print("\nRTMPose-M paper-recipe check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
