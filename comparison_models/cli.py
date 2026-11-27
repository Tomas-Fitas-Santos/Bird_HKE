#!/usr/bin/env python3
"""One small, safe entry point for the three external comparison models.

The CLI is an orchestrator, not a unified framework: each model keeps its own
architecture, its own isolated virtualenv and its own trainer. This module
only discovers the models, checks their prerequisites, and dispatches work to
the correct interpreter **sequentially**.

  python -m comparison_models.cli list
  python -m comparison_models.cli preflight --model all --scenario FD
  python -m comparison_models.cli test     --model all --scenario FD
  python -m comparison_models.cli train    --model vitpose-b --scenario FD
  python -m comparison_models.cli resume   --model vitpose-b --scenario FD

Safety contract:

* ``list`` and ``preflight`` never import torch, MMPose or a model, and never
  touch a GPU. They work on a laptop.
* ``test`` runs each model's real-data GPU smoke check inside that model's own
  environment, one model at a time. It performs no writes to any definitive
  experiment directory.
* ``train``/``resume`` are explicit, single-model commands and are never
  implied by any other subcommand.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from comparison_models.registry import (
    COMPARISON_MODELS, ROOT, SCENARIOS, ComparisonModel, resolve_scenarios,
)

#: Environment variables that locate an isolated interpreter per model.
ENVIRONMENT_VARIABLES = {
    "rtmpose-m": "RTMPOSE_ENV_PREFIX",
    "vitpose-b": "VITPOSE_ENV_PREFIX",
    "posebh-b": "POSEBH_ENV_PREFIX",
}


def interpreter_for(model: ComparisonModel) -> Path | None:
    """Return the model's isolated Python, or None when it is unavailable.

    Resolution order: an explicit ``*_ENV_PREFIX`` override, then the venv
    recorded in the registry hint. The current interpreter is only accepted
    when it already carries the model's environment marker, which is exactly
    the rule the model's own trainer enforces.

    The path is deliberately *not* resolved: ``venv/bin/python`` is normally a
    symlink to the base interpreter, and following it would hand back the base
    Python, which does not activate the venv and therefore fails
    ``ensure_isolated_runtime``.
    """
    current = Path(sys.executable)
    if model.environment_marker in str(current).lower() or \
            model.environment_marker in str(current.resolve()).lower():
        return current
    override = os.environ.get(ENVIRONMENT_VARIABLES.get(model.model_id, ""), "")
    candidate = override or model.environment_hint
    if not candidate:
        return None
    path = Path(candidate)
    for probe in ((path / "bin" / "python"), (path / "Scripts" / "python.exe"), path):
        if probe.is_file():
            return probe
    return None


def _base_env(model: ComparisonModel) -> dict:
    env = dict(os.environ)
    env["BIRD_HKE_REPO_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), env.get("PYTHONPATH", "")))
    if model.requires_mmpose:
        upstream = ROOT / ".external_pose_sources" / "rtmpose"
        overlay = ROOT / ".external_pose_sources" / "rtmpose_birdgaze_overlay"
        env["PYTHONPATH"] = os.pathsep.join(
            (str(overlay), str(ROOT), str(upstream), env.get("PYTHONPATH", "")))
    return env


def _run(model: ComparisonModel, argv: list[str], env: dict) -> int:
    print(f"\n=== {model.display_name} :: {' '.join(argv[1:])} ===", flush=True)
    print(f"interpreter: {argv[0]}", flush=True)
    return subprocess.run(argv, cwd=ROOT, env=env, check=False).returncode


def cmd_list(_args) -> int:
    print(json.dumps({
        "native_research_package": "Bird_HKE/",
        "comparison_package": "comparison_models/",
        "scenarios": list(SCENARIOS),
        "models": [
            {
                "id": m.model_id,
                "name": m.display_name,
                "package": m.package,
                "environment": m.environment_hint,
                "environment_available": interpreter_for(m) is not None,
                "entry_point": m.train_module,
                "smoke_entry_point": m.smoke_module,
                "output_root": m.output_root,
                "probe_root": m.probe_root,
                "scenarios": list(m.scenarios),
                "requires_mmpose": m.requires_mmpose,
                "upstream": f"{m.upstream_repository}@{m.upstream_revision}",
                "setup_script": m.setup_script,
                "notes": m.notes,
            }
            for m in COMPARISON_MODELS.models.values()
        ],
    }, indent=2, sort_keys=True))
    return 0


def cmd_preflight(args) -> int:
    """Dependency, layout and data checks. No GPU, no torch, no writes."""
    models = COMPARISON_MODELS.resolve(args.model)
    scenarios = resolve_scenarios(args.scenario)
    failures = []
    for model in models:
        package = ROOT / model.package
        if not package.is_dir():
            failures.append(f"{model.model_id}: package directory missing: {model.package}")
            continue
        interpreter = interpreter_for(model)
        print(f"[{model.model_id}] package={model.package}")
        print(f"  environment : {model.environment_hint} "
              f"({'found' if interpreter else 'NOT FOUND'})")
        print(f"  output root : {model.output_root}")
        print(f"  upstream    : {model.upstream_revision[:12]} "
              f"({model.upstream_repository})")
        if interpreter is None:
            failures.append(
                f"{model.model_id}: isolated environment not found at "
                f"{model.environment_hint}; create it with "
                f"bash {model.setup_script}")
            continue
        for scenario in scenarios:
            if scenario not in model.scenarios:
                print(f"  {scenario}: SKIPPED - not validated for this model "
                      f"({model.notes})")
                continue
            dataset = ROOT / "BirdGaze_v2" / {
                "FD": "BirdGaze_Full_Dataset",
                "OS": "birdgaze_original_subset",
                "CS": "birdgaze_corrected_subset",
            }[scenario]
            present = all((dataset / "annot" / f"{split}.json").is_file()
                          for split in ("train", "val"))
            print(f"  {scenario}: source annotations "
                  f"{'present' if present else 'MISSING'} at {dataset.name}")
            if not present:
                failures.append(f"{model.model_id}/{scenario}: missing "
                                f"{dataset / 'annot'}")
        if model.requires_mmpose:
            upstream = ROOT / ".external_pose_sources" / "rtmpose"
            overlay = ROOT / ".external_pose_sources" / "rtmpose_birdgaze_overlay"
            if not (upstream / "tools" / "train.py").is_file():
                failures.append(
                    f"{model.model_id}: pinned MMPose checkout missing; run "
                    "bash comparison_models/scripts/setup_external_pose_sources.sh")
            if not (overlay / "manifest.json").is_file():
                failures.append(
                    f"{model.model_id}: MMPose import overlay missing; run "
                    f"bash {model.setup_script} --repair")
        # Ask the model's own interpreter whether its imports resolve. This is
        # the only way to check MMPose without importing it in this process.
        code = (
            "import importlib, sys\n"
            f"importlib.import_module({model.train_module!r})\n"
            + ("importlib.import_module('mmpose')\n" if model.requires_mmpose else "")
            + "print('imports OK', sys.executable)\n"
        )
        result = subprocess.run([str(interpreter), "-c", code], cwd=ROOT,
                                env=_base_env(model), capture_output=True,
                                text=True, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            failures.append(f"{model.model_id}: imports failed in "
                            f"{interpreter}: {detail[-1] if detail else 'no output'}")
        else:
            print(f"  imports     : {(result.stdout or '').strip()}")
    if failures:
        print("\nPREFLIGHT FAILED", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        return 1
    print("\nPreflight OK: environments, layouts and source annotations verified. "
          "No training was launched.", flush=True)
    return 0


def cmd_test(args) -> int:
    """Real-data GPU smoke checks, sequentially, one model at a time."""
    models = COMPARISON_MODELS.resolve(args.model)
    scenarios = resolve_scenarios(args.scenario)
    rc = 0
    for model in models:
        for scenario in scenarios:
            if scenario not in model.scenarios:
                print(f"[{model.model_id}] {scenario}: SKIPPED - {model.notes}")
                continue
            interpreter = interpreter_for(model)
            if interpreter is None:
                print(f"[{model.model_id}] {scenario}: SKIPPED - environment "
                      f"{model.environment_hint} is unavailable", flush=True)
                rc = max(rc, 2)
                continue
            if model.requires_mmpose:
                argv = [str(interpreter), "-m", model.smoke_module,
                        "--scenario", scenario, "--check-only"]
            else:
                argv = [str(interpreter), "-m", model.smoke_module,
                        "--scenario", scenario, "--smoke-test"]
            rc = max(rc, _run(model, argv, _base_env(model)))
    print("\nSmoke summary:", "PASS" if rc == 0 else
          ("SKIPPED (environment unavailable)" if rc == 2 else "FAILED"), flush=True)
    return rc


def cmd_train(args) -> int:
    """Explicit training. Never reached from list/preflight/test."""
    model, = COMPARISON_MODELS.resolve(args.model)
    scenario, = resolve_scenarios(args.scenario)
    if scenario not in model.scenarios:
        print(f"{model.model_id} refuses {scenario}: {model.notes}", flush=True)
        return 1
    interpreter = interpreter_for(model)
    if interpreter is None:
        print(f"Refusing to train: {model.environment_hint} is unavailable.",
              flush=True)
        return 1
    if model.requires_mmpose:
        argv = [str(interpreter), "-m", model.train_module,
                "--scenario", scenario]
        if args.resume:
            argv.append("--resume")
    else:
        argv = [str(interpreter), "-m", model.train_module,
                "--scenario", scenario, "--resume" if args.resume else "--train"]
    return _run(model, argv, _base_env(model))


def cmd_doctor(_args) -> int:
    """Report interpreter/package availability without importing them."""
    report = {"python": sys.executable, "repo_root": str(ROOT),
              "models": {}}
    for model in COMPARISON_MODELS.models.values():
        interpreter = interpreter_for(model)
        entry = {"environment_hint": model.environment_hint,
                 "interpreter": str(interpreter) if interpreter else None}
        if interpreter:
            probe = subprocess.run(
                [str(interpreter), "-c",
                 "import sys;print(sys.version.split()[0])"],
                capture_output=True, text=True, check=False)
            entry["python_version"] = (probe.stdout or "").strip() or None
        report["models"][model.model_id] = entry
    report["outputs"] = {
        m.model_id: str(ROOT / m.output_root) for m in COMPARISON_MODELS.models.values()}
    report["outputs_exist"] = {
        m.model_id: (ROOT / m.output_root).exists()
        for m in COMPARISON_MODELS.models.values()}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m comparison_models.cli",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="Show the three comparison models")
    listing.set_defaults(func=cmd_list)

    for name, func, help_text in (
        ("preflight", cmd_preflight,
         "Check environments, layouts and data. No GPU, no writes."),
        ("test", cmd_test,
         "Real-data GPU smoke checks, run sequentially. No writes."),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--model", default="all",
                        choices=[*COMPARISON_MODELS.ids(), "all"])
        sp.add_argument("--scenario", default="FD",
                        choices=[*SCENARIOS, "all"])
        sp.set_defaults(func=func)

    for name, resume, help_text in (
        ("train", False, "Start a definitive run. Explicit and single-model only."),
        ("resume", True, "Resume a definitive run from its epoch checkpoint."),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--model", required=True, choices=COMPARISON_MODELS.ids())
        sp.add_argument("--scenario", required=True, choices=SCENARIOS)
        sp.add_argument("--resume", action="store_true", default=resume,
                        help=argparse.SUPPRESS)
        sp.set_defaults(func=cmd_train)

    doctor = sub.add_parser("doctor", help="Report interpreter availability")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
