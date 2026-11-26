"""Shared, architecture-agnostic training machinery for the ViT comparison models.

ViTPose-B and PoseBH-B are trained by *identical* protocols: one visible GPU,
seed 2026, AdamW, warmup + cosine schedule, fp16 autocast with a gradient
scaler, epoch-aligned gradient accumulation, atomic checkpoints containing the
complete optimizer/scheduler/scaler/RNG state, alternating immutable "best"
slots, and a JSONL history that can be reconciled after an interrupted write.

Only the parts that are genuinely identical live here. Each model package keeps
its own architecture, loss, validation heatmap source, batch size and smoke
test, because those legitimately differ. Nothing in this module imports MMPose,
MMCV or CUDA at import time, and nothing here writes outside a run directory
that the caller has already reserved.
"""
from __future__ import annotations

# Must be set BEFORE the first CUDA context or CUDA BLAS operation.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_isolated_runtime(marker: str, documentation: str):
    """Refuse to run inside the native Bird_HKE or another model's environment.

    ``marker`` is the substring that must appear in the interpreter path or
    environment prefix, e.g. ``vitpose`` or ``posebh``.
    """
    executable = str(Path(sys.executable).resolve()).lower()
    prefix = str(Path(sys.prefix).resolve()).lower()
    if marker not in executable and marker not in prefix:
        raise RuntimeError(
            f"This model requires its own dedicated Python environment "
            f"containing '{marker}' in its path. See {documentation}; the "
            "native Bird_HKE environment is forbidden.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Require exactly one visible CUDA GPU (set CUDA_VISIBLE_DEVICES)")


#: Opt-out for hosts with no usable memory probe at all.
CAPACITY_GUARD_SKIP = "BIRD_HKE_SKIP_GPU_CAPACITY_CHECK"

#: Override for the free-memory floor, in MiB.
CAPACITY_GUARD_FLOOR = "BIRD_HKE_GPU_MIN_FREE_MIB"

#: Free memory a single comparison run needs before it starts. ViTPose-B and
#: PoseBH-B peak at ~2.2 GiB allocated at their fixed microbatch, and the
#: CUDA context, cuDNN workspaces and the fp16 gradient scaler add roughly
#: another GiB. Two runs therefore share a 12 GiB card comfortably.
DEFAULT_MIN_FREE_MIB = 4096


def _visible_free_memory():
    """Return [(label, free_mib, total_mib)] for this process's visible GPUs.

    torch is the only trustworthy source here: it already applies
    ``CUDA_VISIBLE_DEVICES`` *and* the host's device ordering. ``nvidia-smi``
    numbers GPUs independently, and on hosts where the two orders differ
    (e.g. PCI vs fast-first enumeration) filtering its output by the
    ``CUDA_VISIBLE_DEVICES`` index silently reads the wrong card.
    """
    rows = []
    for local_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(local_index)
        try:
            name = torch.cuda.get_device_name(local_index)
        except Exception:
            name = "cuda"
        rows.append((f"cuda:{local_index} ({name})",
                     free_bytes / 2 ** 20, total_bytes / 2 ** 20))
    return rows


def _normalize_gpu_uuid(value) -> str:
    """Canonical form for a GPU UUID so CUDA and nvidia-smi strings compare.

    ``nvidia-smi`` prints ``GPU-<hex>`` while the CUDA device property may or
    may not carry that prefix and may use dashes, so both sides are reduced to
    bare uppercase hex.
    """
    text = str(value).strip().upper().replace("-", "")
    return text[4:] if text.startswith("GPU") else text


def _visible_device_uuids():
    """Return the UUIDs of this process's visible GPUs, or None.

    The UUID is read from the CUDA context itself, which is the only identity
    that survives the two ways of numbering GPUs. ``torch.cuda.
    _get_nvml_device_index`` is *not* usable here despite its name: it returns
    ``_parse_visible_devices()[idx]``, i.e. the CUDA ordinal that
    ``CUDA_VISIBLE_DEVICES`` was set to, not an NVML index. Applying that
    number to ``nvidia-smi`` output is exactly the mistake that made an idle
    H100 PCIe report the ten co-tenants of a different card.
    """
    uuids = []
    for local_index in range(torch.cuda.device_count()):
        try:
            value = getattr(torch.cuda.get_device_properties(local_index),
                            "uuid", None)
        except Exception:
            return None
        if value is None:
            return None
        uuids.append(_normalize_gpu_uuid(value))
    return uuids or None


def _shared_process_count():
    """Best-effort count of other compute processes on this run's GPU(s).

    Informational only, and attributed by the UUID of the card the CUDA
    context is actually on, so a colleague on another card is never counted.
    Returns None when it cannot be determined -- including on hosts without
    nvidia-smi, which is normal on some cluster nodes.
    """
    wanted = _visible_device_uuids()
    if wanted is None:
        return None
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, check=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return None
    return len({
        int(pid)
        for line in apps.stdout.splitlines()
        for parts in [[p.strip() for p in line.split(",")]]
        if len(parts) == 2
        for gpu_uuid, pid in [parts]
        if pid.isdigit()
        and _normalize_gpu_uuid(gpu_uuid) in wanted
        and int(pid) != os.getpid()
    })


def require_gpu_capacity(min_free_mib: float | None = None):
    """Refuse to start unless the assigned GPU has enough free memory.

    The protocol needs an *isolated environment* and a *fixed* seed, neither of
    which is affected by another process sharing the card: each run gets its
    own CUDA context, and determinism comes from the per-process RNG plus the
    fixed batch order. The one thing co-tenancy can break is the run itself, by
    exhausting memory, so that is what is checked here.

    Only the device(s) this process can actually see are considered, so a
    colleague on another card of the same host never blocks this run. Other
    processes sharing the selected card are reported but permitted.

    This is a startup check, not a scheduler: a job starting later can still
    consume the memory, so plan the co-tenancy deliberately.
    """
    if os.environ.get(CAPACITY_GUARD_SKIP) == "1":
        print(f"GPU capacity check skipped via {CAPACITY_GUARD_SKIP}=1.",
              flush=True)
        return
    required = float(os.environ.get(CAPACITY_GUARD_FLOOR)
                     or (min_free_mib or DEFAULT_MIN_FREE_MIB))
    try:
        rows = _visible_free_memory()
    except Exception as exc:
        raise RuntimeError(
            "Cannot read GPU memory state from the CUDA context; refusing to "
            f"start. Set {CAPACITY_GUARD_SKIP}=1 to override.") from exc
    if not rows:
        raise RuntimeError("No CUDA device is visible to this process.")
    label, free, total = min(rows, key=lambda row: row[1])
    if free < required:
        raise RuntimeError(
            f"{label} has only {free:.0f} MiB free of {total:.0f} MiB; this "
            f"run needs at least {required:.0f} MiB. Choose a freer GPU, raise "
            f"{CAPACITY_GUARD_FLOOR}, or set {CAPACITY_GUARD_SKIP}=1 if you "
            "have verified the memory.")
    shared = _shared_process_count()
    note = ("exclusive" if shared == 0 else
            f"sharing with {shared} other compute process(es)"
            if shared else "")
    print(f"{label}: {free:.0f}/{total:.0f} MiB free "
          f"(>= {required:.0f} MiB required)"
          + (f", {note}" if note else ""), flush=True)



def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def runtime_fingerprint() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        # Worker count feeds the augmentation RNG stream (each worker is
        # seeded from the loader seed plus its id), so it is part of the
        # reproducibility contract and resume refuses if it changed.
        "dataloader_workers": dataloader_workers(),
    }


WORKERS_ENV = "BIRD_HKE_NUM_WORKERS"
DEFAULT_MAX_WORKERS = 8
PREFETCH_FACTOR = 4


def dataloader_workers():
    """Decode/augment workers per loader.

    Every sample is a full-resolution ``cv2.imread`` plus an affine warp and a
    heatmap rasterisation, so a single worker tops out well below what either
    ViT can consume -- two workers left the GPU starved and the FD scenario
    (77,740 train images) is entirely decode-bound. The count is recorded in
    the run manifest because it changes the augmentation RNG stream: each
    worker is seeded from the loader's base seed plus its own id, so a run is
    only reproducible at the same worker count.
    """
    override = os.environ.get(WORKERS_ENV, "").strip()
    if override:
        try:
            workers = int(override)
        except ValueError:
            raise RuntimeError(f"{WORKERS_ENV} must be an integer, got {override!r}")
        if workers < 0:
            raise RuntimeError(f"{WORKERS_ENV} must be >= 0, got {workers}")
        return workers
    return min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1)


def make_epoch_loader(recipe, dataset, epoch, *, train):
    """Epoch-specific sampler/worker seeds for exact epoch-boundary resume."""
    seed = int(recipe["seed"]) + epoch * 1009 + (0 if train else 800000)
    sampler_gen = torch.Generator().manual_seed(seed)
    worker_gen = torch.Generator().manual_seed(seed + 29)
    sampler = RandomSampler(dataset, generator=sampler_gen) if train else None
    workers = dataloader_workers()
    return DataLoader(
        dataset, batch_size=recipe["microbatch"],
        sampler=sampler, shuffle=False, drop_last=False,
        num_workers=workers, pin_memory=True,
        generator=worker_gen,
        # Must stay False: a worker's RNG is seeded from the base seed drawn
        # off this loader's generator, and the loader is rebuilt every epoch
        # with an epoch-specific seed. Keeping workers alive across epochs
        # would freeze the augmentation stream.
        persistent_workers=False,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
    )


def layer_decay_param_groups(model, base_lr, weight_decay, num_layers,
                             layer_decay_rate):
    """Parameter groups for the ViT layer-decay used by ViTPose and PoseBH.

    Mirrors ``LayerDecayOptimizerConstructor`` from the ViTPose/PoseBH
    repositories (``mmcv_custom/layer_decay_optimizer_constructor.py``),
    re-expressed for this repository's parameter names -- those models wrap
    the transformer in ``backbone.``, ours does not:

    * ``pos_embed``/``patch_embed`` -> layer 0, ``blocks.N`` -> layer N+1,
      anything else (heads, necks) -> the last layer;
    * 1-D tensors, ``.bias`` tensors and ``pos_embed`` get weight decay 0;
    * ``lr_scale = layer_decay_rate ** (num_layers - layer_id - 1)`` with
      ``num_layers = depth + 2``.

    Frozen parameters are skipped, so a freeze/unfreeze stage can be resumed
    without changing the group layout.
    """
    depth_plus_two = int(num_layers) + 2
    rate = float(layer_decay_rate)
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split(".")
        if name.startswith("patch_embed") or name == "pos_embed":
            layer_id = 0
        elif parts[0] == "blocks" and len(parts) > 1 and parts[1].isdigit():
            layer_id = int(parts[1]) + 1
        else:
            layer_id = depth_plus_two - 1
        undecayed = (len(parameter.shape) == 1 or name.endswith(".bias")
                     or "pos_embed" in name)
        key = (layer_id, 0 if undecayed else 1)
        if key not in groups:
            scale = rate ** (depth_plus_two - layer_id - 1)
            groups[key] = {
                "params": [],
                "param_names": [],
                "weight_decay": 0.0 if undecayed else float(weight_decay),
                "lr_scale": scale,
                "lr": scale * float(base_lr),
                "group_name": "layer_%d_%s"
                              % (key[0], "no_decay" if undecayed else "decay"),
            }
        groups[key]["params"].append(parameter)
        groups[key]["param_names"].append(name)
    return [groups[key] for key in sorted(groups)]


def make_optimizer(recipe, model):
    schedule = recipe.get("schedule") or {}
    if schedule.get("type") == "vit_layer_decay":
        groups = layer_decay_param_groups(
            model, recipe["lr"], recipe["weight_decay"],
            schedule["num_layers"], schedule["layer_decay_rate"])
        return torch.optim.AdamW(groups, lr=recipe["lr"],
                                 weight_decay=recipe["weight_decay"],
                                 betas=tuple(recipe.get("betas", (0.9, 0.999))))
    return torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"],
        weight_decay=recipe["weight_decay"],
        betas=tuple(recipe.get("betas", (0.9, 0.999))),
    )


def lr_multiplier(recipe, epoch):
    warmup = recipe["warmup_epochs"]
    min_factor = recipe["min_lr"] / recipe["lr"]
    if epoch < warmup:
        return 0.1 + 0.9 * epoch / max(1, warmup)
    progress = min(1., (epoch - warmup) / max(1, recipe["epochs"] - warmup))
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))


class PaperScheduleLR(torch.optim.lr_scheduler.LRScheduler):
    """Iteration-based linear warmup plus epoch-milestone step decay.

    ViTPose and PoseBH both train with ``warmup='linear',
    warmup_iters=500, warmup_ratio=0.001`` followed by ``policy='step'``
    decay at epoch milestones (``step=[170, 200]`` of 210 epochs for ViTPose,
    ``step=[50, 90]`` of 100 epochs for PoseBH). MMPose applies both per
    optimizer iteration, so this scheduler is stepped once per optimizer
    step and ``milestones_epochs`` are converted to iterations with
    ``iters_per_epoch`` (optimizer steps per epoch).

    ``lr_lambda`` remains available for the epoch-level cosine fallback used
    by recipes without a ``schedule`` block.
    """

    steps_per_iteration = True

    def __init__(self, optimizer, warmup_iters, warmup_ratio,
                 milestones_epochs, iters_per_epoch, gamma=0.1,
                 last_epoch=-1):
        self.warmup_iters = int(warmup_iters)
        self.warmup_ratio = float(warmup_ratio)
        self.milestones_epochs = tuple(int(epoch) for epoch in milestones_epochs)
        self.iters_per_epoch = max(1, int(iters_per_epoch))
        self.gamma = float(gamma)
        self.milestones_iters = tuple(
            epoch * self.iters_per_epoch for epoch in self.milestones_epochs)
        # LRScheduler steps once during __init__, so start one iteration
        # early to make the first parameter update use iteration 0.
        self.iterations = -1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_iters and self.iterations < self.warmup_iters:
            progress = self.iterations / self.warmup_iters
            factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * progress
        else:
            factor = self.gamma ** sum(
                1 for milestone in self.milestones_iters
                if self.iterations >= milestone)
        return [base_lr * factor for base_lr in self.base_lrs]

    def step(self):
        self.iterations += 1
        super().step()

    def state_dict(self):
        state = super().state_dict()
        state.pop("optimizer", None)
        return state


def make_scheduler(recipe, optimizer, iters_per_epoch):
    """Build the scheduler a recipe asks for.

    Recipes with a ``schedule`` block get the paper's iteration-based
    warmup + step decay; anything else keeps the epoch-level cosine used by
    the smoke tests.
    """
    schedule = recipe.get("lr_schedule") or {}
    if schedule.get("type") == "vit_step":
        return PaperScheduleLR(
            optimizer,
            warmup_iters=schedule["warmup_iters"],
            warmup_ratio=schedule["warmup_ratio"],
            milestones_epochs=schedule["milestones_epochs"],
            iters_per_epoch=iters_per_epoch,
            gamma=schedule.get("gamma", 0.1),
        )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: lr_multiplier(recipe, epoch))



def rng_snapshot():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(snapshot):
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise RuntimeError("Checkpoint is missing complete RNG state")
    if len(snapshot["torch_cuda"]) != torch.cuda.device_count():
        raise RuntimeError("Checkpoint CUDA device count mismatch")
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def manifest_digest(document) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def write_json_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def save_atomic(payload, path):
    """Write a checkpoint by atomic rename inside the exclusively locked run."""
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def recover_history(run_dir, committed_epoch, committed_metrics):
    """Repair an interrupted append after checkpoint_last was atomically saved.

    Only the single committed metric row may be appended. Refuse malformed,
    duplicate, or inconsistent history instead of inventing measurements.
    """
    path = Path(run_dir) / "training_history.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(epochs) + 1)):
        raise RuntimeError("Training history contains gaps or duplicate epochs")
    if len(epochs) > committed_epoch:
        raise RuntimeError("History extends past the committed checkpoint")
    if len(epochs) == committed_epoch:
        if rows[-1] != committed_metrics:
            raise RuntimeError("Training history disagrees with last checkpoint")
        return
    if len(epochs) == committed_epoch - 1 and committed_metrics is not None:
        if committed_metrics.get("epoch") != committed_epoch:
            raise RuntimeError("Checkpoint metric row has incorrect epoch")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(committed_metrics, sort_keys=True) + "\n")
        return
    raise RuntimeError("Training history cannot be safely recovered")


def refresh_best_alias(run_dir, slot, digest):
    """Keep canonical best checkpoint synchronized with committed epoch state.

    Alternating immutable-per-epoch best slots mean an interrupted checkpoint
    write cannot destroy the best state referenced by the preceding checkpoint.
    """
    if slot not in ("best_slot_a.pth", "best_slot_b.pth"):
        raise RuntimeError(f"Invalid best slot in checkpoint: {slot!r}")
    run_dir = Path(run_dir)
    source = run_dir / slot
    if not source.is_file() or sha256(source) != digest:
        raise RuntimeError("Recorded best checkpoint hash mismatch; refusing resume")
    target = run_dir / "model_best.pth"
    if target.is_file() and sha256(target) == digest:
        return
    tmp = run_dir / f".model_best.pth.{os.getpid()}.tmp"
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_run_lock(folder):
    lock = Path(folder) / ".training_active.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Training lock already present: {lock}; inspect running jobs "
            "and recover the lock manually only after confirming it is stale") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


class SingleTaskTrainer:
    """Protocol shared by the two single-task ViT comparison trainers.

    Subclasses supply the pieces that legitimately differ per architecture:
    ``new_model``, ``loss`` and ``validation_heatmaps``. Everything else --
    accumulation, AMP, clipping, checkpointing, resume, history recovery and
    PCK validation -- is executed once, here, for both models.
    """

    #: Extra metric keys emitted by ``loss`` beyond ``train_loss``.
    loss_parts: tuple[str, ...] = ()

    def __init__(self, recipe, *, name: str, environment_marker: str,
                 documentation: str):
        self.recipe = recipe
        self.name = name
        self.environment_marker = environment_marker
        self.documentation = documentation

    # -- architecture hooks ------------------------------------------------
    def new_model(self):
        raise NotImplementedError

    def loss(self, model, images, targets, weights, epoch):
        """Return (loss, {extra metric name: scalar tensor})."""
        raise NotImplementedError

    def validation_heatmaps(self, model, images):
        raise NotImplementedError

    def ensure_recipe_is_from_scratch(self):
        """Models must refuse any pretrained initialization."""

    # -- setup -------------------------------------------------------------
    def ensure_isolated_runtime(self):
        ensure_isolated_runtime(self.environment_marker, self.documentation)
        self.ensure_recipe_is_from_scratch()

    def set_determinism(self):
        set_determinism(int(self.recipe["seed"]))

    def make_epoch_loader(self, dataset, epoch, *, train):
        return make_epoch_loader(self.recipe, dataset, epoch, train=train)

    def make_optimizer(self, model):
        return make_optimizer(self.recipe, model)

    def lr_multiplier(self, epoch):
        return lr_multiplier(self.recipe, epoch)

    def iters_per_epoch(self, dataset_size):
        """Optimizer steps per epoch under effective-batch accumulation."""
        return max(1, math.ceil(int(dataset_size) / int(self.recipe["effective_batch"])))

    def make_scheduler(self, optimizer, dataset_size):
        return make_scheduler(self.recipe, optimizer,
                              self.iters_per_epoch(dataset_size))

    def make_grad_scaler(self):
        # PoseBH's config uses fp16=dict(loss_scale=dict(init_scale=16384.0)).
        return torch.amp.GradScaler(
            "cuda", enabled=True,
            init_scale=float(self.recipe.get("grad_scaler_init_scale", 65536.0)))

    # -- protocol helpers exposed for the disposable integration probe -----
    # The probe must exercise the exact production implementations, so they
    # are reachable from the trainer object rather than re-imported by name.
    manifest_digest = staticmethod(manifest_digest)
    refresh_best_alias = staticmethod(refresh_best_alias)
    recover_history = staticmethod(recover_history)
    save_atomic = staticmethod(save_atomic)
    rng_snapshot = staticmethod(rng_snapshot)
    restore_rng = staticmethod(restore_rng)
    sha256 = staticmethod(sha256)

    def require_gpu_capacity(self):
        require_gpu_capacity()

    # -- epochs ------------------------------------------------------------
    def training_epoch(self, model, optimizer, scaler, loader, dataset_size,
                       epoch, scheduler=None):
        recipe = self.recipe
        model.train()
        optimizer.zero_grad(set_to_none=True)
        count, running_loss, group_seen = 0, 0., 0
        parts_total = {key: 0. for key in self.loss_parts}
        group_size = min(recipe["effective_batch"], dataset_size)
        tick = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            images = batch["image"].cuda(non_blocking=True)
            targets = batch["target"].cuda(non_blocking=True)
            weights = batch["weight"].cuda(non_blocking=True)
            micro_count = images.size(0)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, parts = self.loss(model, images, targets, weights, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite train loss epoch={epoch} step={step}")
            # Normalize by actual number of images, including final partial group.
            scaler.scale(loss * (micro_count / group_size)).backward()
            running_loss += float(loss.detach().item()) * micro_count
            for key in parts_total:
                parts_total[key] += float(parts[key].detach().item()) * micro_count
            count += micro_count
            group_seen += micro_count
            if group_seen == group_size:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), recipe["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                group_seen = 0
                group_size = min(recipe["effective_batch"], dataset_size - count)
                # The papers' warmup/step schedule is iteration based, so it
                # advances once per optimizer step. Epoch-level schedulers are
                # still advanced once per epoch by ``save_epoch``.
                if getattr(scheduler, "steps_per_iteration", False):
                    scheduler.step()
            if step % 100 == 0:
                print(f"{self.name} epoch {epoch + 1}: {count}/{dataset_size} "
                      f"images, loss={running_loss / count:.8f}", flush=True)
        if count != dataset_size or group_seen != 0:
            raise RuntimeError(
                f"Epoch did not consume the entire dataset: {count}/{dataset_size}")
        return {"train_loss": running_loss / max(1, count),
                **{f"train_{key}": value / max(1, count)
                   for key, value in parts_total.items()},
                "train_seconds": round(time.perf_counter() - tick, 2)}

    @torch.no_grad()
    def validation_epoch(self, model, dataset, epoch):
        recipe = self.recipe
        model.eval()
        count = 0
        good = np.zeros(4, dtype=np.int64)
        total = np.zeros(4, dtype=np.int64)
        loader = self.make_epoch_loader(dataset, epoch, train=False)
        tick = time.perf_counter()
        for batch in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                predicted = self.validation_heatmaps(
                    model, batch["image"].cuda(non_blocking=True))
            heatmaps = predicted.float().cpu().numpy()
            xy = decode_to_image_coordinates(
                heatmaps, batch["center"].numpy(), batch["scale"].numpy(),
                batch["offset"].numpy(),
            )
            truth = batch["gt"].numpy()
            valid = batch["eval_valid"].numpy().astype(bool)
            scale = np.abs(batch["raw_scale"].numpy())
            threshold = 0.05 * scale[:, None] * 200.
            error = np.linalg.norm(xy - truth, axis=-1)
            # Zero original scale has no valid PCK normalizer. Keep these rows in
            # the dataset/training but exclude them from the validation metric.
            mask = valid & (scale[:, None] > 0) & np.isfinite(error)
            good += ((error <= threshold) & mask).sum(axis=0)
            total += mask.sum(axis=0)
            count += len(truth)
        scores = np.divide(good, total, out=np.full(4, np.nan), where=total > 0)
        value = float(good.sum() / total.sum()) if total.sum() else float("nan")
        if not np.isfinite(value):
            raise RuntimeError(
                "No labeled validation landmarks available for checkpoint selection")
        return {
            "val_pck_005": value,
            "pck_per_keypoint": [None if np.isnan(x) else float(x) for x in scores],
            "val_landmarks": total.tolist(),
            "val_images": count,
            "val_seconds": round(time.perf_counter() - tick, 2),
        }

    # -- checkpointing -----------------------------------------------------
    def save_epoch(self, model, optimizer, scheduler, scaler, run_dir, manifest,
                   epoch, best, best_slot, best_digest, stats, validation):
        """Commit one epoch: best slot, full state, alias and history row."""
        score = float(validation["val_pck_005"])
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation PCK in checkpoint")
        updated = score > best
        best = max(best, score)
        record = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "base_lr": float(self.recipe["lr"]),
            "lr_group_count": len(optimizer.param_groups),
            **stats, **validation,
            "best_val_pck_005": best,
            "peak_cuda_memory_mib": round(
                torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
        }
        if updated:
            best_slot = ("best_slot_b.pth" if best_slot == "best_slot_a.pth"
                         else "best_slot_a.pth")
            save_atomic(model.state_dict(), run_dir / best_slot)
            best_digest = sha256(run_dir / best_slot)
        if not getattr(scheduler, "steps_per_iteration", False):
            scheduler.step()
        save_atomic({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "rng": rng_snapshot(),
            "recipe_hash": manifest_digest(manifest),
            "best_pck": best,
            "best_slot": best_slot,
            "best_sha256": best_digest,
            "metrics": record,
        }, run_dir / "checkpoint_last.pth")
        # The committed row is appended by recover_history, which also repairs
        # the case where the process died between checkpoint_last and the log.
        if updated:
            refresh_best_alias(run_dir, best_slot, best_digest)
        recover_history(run_dir, epoch + 1, record)
        return best, best_slot, best_digest, record

    def load_epoch(self, model, optimizer, scheduler, scaler, run_dir, manifest):
        """Strict full-state resume. Returns the resume state dictionary."""
        # This pickle file is generated only by our own training process,
        # inside the locked dedicated run directory. Never load arbitrary
        # externally supplied checkpoints with weights_only=False.
        state = torch.load(Path(run_dir) / "checkpoint_last.pth",
                           map_location="cpu", weights_only=False)
        if state.get("recipe_hash") != manifest_digest(manifest):
            raise RuntimeError("Checkpoint recipe fingerprint mismatch")
        begin_epoch = int(state["epoch"])
        if begin_epoch < 1 or begin_epoch > int(self.recipe["epochs"]):
            raise RuntimeError(f"Invalid resume epoch: {begin_epoch}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["grad_scaler"])
        digest = state.get("best_sha256")
        if not isinstance(digest, str):
            raise RuntimeError("Checkpoint missing best-checkpoint integrity hash")
        refresh_best_alias(run_dir, state["best_slot"], digest)
        recover_history(run_dir, begin_epoch, state.get("metrics"))
        restore_rng(state["rng"])
        return state

    # -- full run ----------------------------------------------------------
    def train(self, *, run_dir, manifest, train_data, valid_data, mode,
              begin_message=None):
        """Run (or resume) the 100-epoch protocol inside an exclusive lock."""
        run_dir = Path(run_dir)
        with exclusive_run_lock(run_dir):
            self.set_determinism()
            model = self.new_model().cuda()
            optimizer = self.make_optimizer(model)
            scheduler = self.make_scheduler(
                optimizer, len(train_data))
            scaler = self.make_grad_scaler()
            begin_epoch, best, best_slot, best_digest = 0, -float("inf"), None, None
            if mode == "resume":
                state = self.load_epoch(model, optimizer, scheduler, scaler,
                                        run_dir, manifest)
                begin_epoch = int(state["epoch"])
                best = float(state["best_pck"])
                best_slot = state.get("best_slot")
                best_digest = state.get("best_sha256")
                print(f"Restored full training state at epoch {begin_epoch}",
                      flush=True)
                if begin_message:
                    print(begin_message, flush=True)

            if begin_epoch >= int(self.recipe["epochs"]):
                print("All epochs are already complete. No changes.", flush=True)
                return
            metrics_file = run_dir / "training_history.jsonl"
            for epoch in range(begin_epoch, int(self.recipe["epochs"])):
                loader = self.make_epoch_loader(train_data, epoch, train=True)
                stats = self.training_epoch(
                    model, optimizer, scaler, loader, len(train_data), epoch,
                    scheduler=scheduler)
                validation = self.validation_epoch(model, valid_data, epoch)
                best, best_slot, best_digest, metrics = self.save_epoch(
                    model, optimizer, scheduler, scaler, run_dir, manifest,
                    epoch, best, best_slot, best_digest, stats, validation)
                assert metrics_file.is_file()
                print(json.dumps(metrics, sort_keys=True), flush=True)
            save_atomic(model.state_dict(), run_dir / "final_model.pth")
            print(f"{self.name} training COMPLETE: {run_dir}", flush=True)


def decode_to_image_coordinates(heatmaps, center, scale, offset):
    """Re-export so trainers share one decoding entry point."""
    from comparison_models.common.birdgaze.dataset import (
        decode_image_coordinates,
    )
    return decode_image_coordinates(heatmaps, center, scale, offset)
