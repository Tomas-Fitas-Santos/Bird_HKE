#!/usr/bin/env python3
"""Machine-readable catalogue of the three external comparison models.

The registry is deliberately import-light: it must work on a machine with no
GPU, no torch and no MMPose, because ``python -m comparison_models list`` and
``... preflight`` are meant to be runnable anywhere. Model modules are only
imported by the model's own entry point, inside its own virtualenv.

Every entry records the four things that must stay separate:

* ``environment``  - isolated interpreter/venv the model requires,
* ``entry_point``  - module that actually runs smoke/test/train,
* ``output_root``  - the only directory that model may ever write,
* ``upstream``     - pinned upstream revision (never modified in place).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: BirdGaze scenario codes shared by all three models.
SCENARIOS = ("FD", "OS", "CS")


@dataclass(frozen=True)
class ComparisonModel:
    model_id: str
    display_name: str
    package: str
    #: Isolated environment marker; the model refuses to run in the native
    #: Bird_HKE interpreter. ``environment_hint`` names the expected venv.
    environment_marker: str
    environment_hint: str
    #: Importable module invoked for a real-data GPU smoke test / training.
    smoke_module: str
    train_module: str
    #: Directory (relative to the repository root) this model may write into.
    output_root: str
    #: Disposable probe root, kept outside every definitive run directory.
    probe_root: str
    upstream_repository: str
    upstream_revision: str
    #: FD/OS/CS the model is currently validated for.
    scenarios: tuple[str, ...] = SCENARIOS
    #: True when the model needs MMPose/MMEngine rather than plain PyTorch.
    requires_mmpose: bool = False
    #: Extra shell setup script that creates the isolated environment.
    setup_script: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Registry:
    models: dict[str, ComparisonModel] = field(default_factory=dict)

    def ids(self) -> tuple[str, ...]:
        return tuple(self.models)

    def resolve(self, value: str) -> tuple[ComparisonModel, ...]:
        if value == "all":
            return tuple(self.models.values())
        try:
            return (self.models[value],)
        except KeyError:
            raise KeyError(
                f"Unknown comparison model {value!r}; "
                f"choose one of {', '.join(self.ids())} or 'all'") from None


COMPARISON_MODELS = Registry({
    "rtmpose-m": ComparisonModel(
        model_id="rtmpose-m",
        display_name="RTMPose-M (official MMPose, BirdGaze recipe)",
        package="comparison_models/rtmpose",
        environment_marker="rtmpose",
        environment_hint="/home/students/conda-envs/rtmpose-torch27-venv",
        smoke_module="comparison_models.rtmpose.run_birdgaze",
        train_module="comparison_models.rtmpose.run_birdgaze",
        output_root="Bird_HKE/trained_models/External_Baselines/RTMPose_M",
        probe_root="Bird_HKE/trained_models/External_Baselines/_Training_Probes/RTMPose_M",
        upstream_repository="https://github.com/open-mmlab/mmpose",
        upstream_revision="759b39c13fea6ba094afc1fa932f51dc1b11cbf9",
        scenarios=SCENARIOS,
        requires_mmpose=True,
        setup_script="comparison_models/scripts/setup_rtmpose_torch27.sh",
        notes="All scenarios supported; definitive training requires real-data GPU preflight.",
    ),
    "vitpose-b": ComparisonModel(
        model_id="vitpose-b",
        display_name="ViTPose-B (standard single-task baseline)",
        package="comparison_models/vitpose",
        environment_marker="vitpose",
        environment_hint="/home/students/conda-envs/vitpose-b-cu128-venv",
        smoke_module="comparison_models.vitpose.train",
        train_module="comparison_models.vitpose.train",
        output_root="Bird_HKE/trained_models/External_Baselines/ViTPose_B",
        probe_root="Bird_HKE/trained_models/External_Baselines/_Training_Probes/ViTPose_B",
        upstream_repository="https://github.com/ViTAE-Transformer/ViTPose",
        upstream_revision="c050ed29112da7704797cc1a65af0234b525010d",
        setup_script="comparison_models/scripts/setup_vitpose_b.sh",
        notes="Trained from scratch; no ImageNet or COCO weights are loaded.",
    ),
    "posebh-b": ComparisonModel(
        model_id="posebh-b",
        display_name="PoseBH-B (single-dataset adaptation)",
        package="comparison_models/posebh",
        environment_marker="posebh",
        environment_hint="/home/students/conda-envs/posebh-b-cu128-venv",
        smoke_module="comparison_models.posebh.train",
        train_module="comparison_models.posebh.train",
        output_root="Bird_HKE/trained_models/External_Baselines/PoseBH_B",
        probe_root="Bird_HKE/trained_models/External_Baselines/_Training_Probes/PoseBH_B",
        upstream_repository="https://github.com/Innse/PoseBH",
        upstream_revision="8f03438f913b559edb0a2beed54e77d1560edba3",
        setup_script="comparison_models/scripts/setup_posebh_b.sh",
        notes="PoseBH architecture adapted to one BirdGaze scenario per run; "
              "it is not the multi-dataset PoseBH paper configuration.",
    ),
})


def resolve_scenarios(value: str) -> tuple[str, ...]:
    if value == "all":
        return SCENARIOS
    if value not in SCENARIOS:
        raise KeyError(f"Unknown scenario {value!r}; choose FD, OS, CS or 'all'")
    return (value,)
