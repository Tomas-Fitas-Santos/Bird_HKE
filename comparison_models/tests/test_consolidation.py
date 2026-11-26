"""Layout, isolation and safety guarantees for the two-group repository split.

These are stdlib/CPU-only checks. They exist so that a future refactor cannot
silently re-couple the native Bird_HKE research models with the external
comparison models, or point a comparison model at a shared environment or a
shared checkpoint directory.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from comparison_models.registry import COMPARISON_MODELS, ROOT, SCENARIOS

REPO = ROOT


def imported_root_modules(path: Path) -> set[str]:
    """Top-level package names a file actually imports (not just mentions)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def module_source(dotted: str) -> str:
    spec = importlib.util.find_spec(dotted)
    if spec is None or not spec.origin:
        raise AssertionError(f"{dotted} is not importable in the new layout")
    return Path(spec.origin).read_text(encoding="utf-8")


class RegistryTests(unittest.TestCase):
    def test_three_models_are_discoverable(self):
        self.assertEqual(set(COMPARISON_MODELS.ids()),
                         {"rtmpose-m", "vitpose-b", "posebh-b"})

    def test_models_use_distinct_environments(self):
        environments = {m.environment_hint for m in COMPARISON_MODELS.models.values()}
        self.assertEqual(len(environments), 3, environments)
        for model in COMPARISON_MODELS.models.values():
            self.assertNotEqual(model.environment_hint, "")
            # The native research environment must never be a comparison env.
            self.assertNotIn("bird_hke-cu128", model.environment_hint)
            self.assertIn(model.environment_marker,
                          model.environment_hint.lower().replace(
                              "rtmpose-torch27", "rtmpose"))

    def test_models_have_distinct_entry_points_and_output_roots(self):
        for attribute in ("train_module", "output_root", "probe_root", "package"):
            values = [getattr(m, attribute) for m in COMPARISON_MODELS.models.values()]
            self.assertEqual(len(set(values)), 3, f"{attribute} is not distinct")
        for model in COMPARISON_MODELS.models.values():
            self.assertTrue(model.output_root.startswith(
                "Bird_HKE/trained_models/External_Baselines/"))
            self.assertNotIn("_Training_Probes", model.output_root)
            self.assertIn("_Training_Probes", model.probe_root)

    def test_pinned_upstream_revisions_are_exact(self):
        self.assertEqual(
            COMPARISON_MODELS.models["rtmpose-m"].upstream_revision,
            "759b39c13fea6ba094afc1fa932f51dc1b11cbf9")
        self.assertEqual(
            COMPARISON_MODELS.models["vitpose-b"].upstream_revision,
            "c050ed29112da7704797cc1a65af0234b525010d")
        self.assertEqual(
            COMPARISON_MODELS.models["posebh-b"].upstream_revision,
            "8f03438f913b559edb0a2beed54e77d1560edba3")

    def test_posebh_is_named_as_single_dataset_adaptation(self):
        self.assertIn("single-dataset adaptation",
                      COMPARISON_MODELS.models["posebh-b"].display_name)

    def test_scenarios_are_shared(self):
        self.assertEqual(SCENARIOS, ("FD", "OS", "CS"))
        self.assertEqual(COMPARISON_MODELS.models["rtmpose-m"].scenarios, ("FD",))


class LayoutTests(unittest.TestCase):
    def test_all_three_packages_exist(self):
        for model in COMPARISON_MODELS.models.values():
            package = REPO / model.package
            self.assertTrue((package / "__init__.py").is_file(), package)
            self.assertTrue((package / "README.md").is_file(), package)

    def test_shared_common_package_exists(self):
        common = REPO / "comparison_models/common"
        for name in ("__init__.py", "transformer.py", "losses.py",
                     "training_utils.py", "preflight.py",
                     "birdgaze/__init__.py", "birdgaze/dataset.py",
                     "birdgaze/conversion.py", "birdgaze/geometry.py"):
            self.assertTrue((common / name).is_file(), common / name)

    def test_old_external_models_tree_is_gone(self):
        self.assertFalse((REPO / "external_models").exists())
        self.assertFalse((REPO / "comparison_models/remaining_baselines").exists())

    def test_moved_scripts_live_in_the_package(self):
        for model in COMPARISON_MODELS.models.values():
            self.assertTrue((REPO / model.setup_script).is_file(), model.setup_script)
        self.assertFalse((REPO / "setup_vitpose_b.sh").exists())
        # Native environment tooling intentionally stays at the repository root.
        self.assertTrue((REPO / "setup_linux_training_env.sh").is_file())

    def test_script_root_resolution_points_at_repository_root(self):
        for script in (REPO / "comparison_models/scripts").glob("*.sh"):
            text = script.read_text(encoding="utf-8")
            if "BASH_SOURCE" not in text:
                continue
            self.assertIn('dirname "${BASH_SOURCE[0]}")/../.."', text,
                          f"{script.name} does not resolve the repository root")


class IsolationTests(unittest.TestCase):
    def test_common_imports_without_mmpose(self):
        code = (
            "import sys\n"
            "for blocked in ('mmpose', 'mmengine', 'mmcv', 'mmdet'):\n"
            "    sys.modules[blocked] = None\n"
            "import importlib\n"
            "for name in ('comparison_models.common.transformer',\n"
            "             'comparison_models.common.losses',\n"
            "             'comparison_models.common.preflight',\n"
            "             'comparison_models.common.birdgaze.dataset'):\n"
            "    importlib.import_module(name)\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_cli_list_and_preflight_do_not_import_torch(self):
        code = (
            "import sys\n"
            "sys.modules['torch'] = None\n"
            "from comparison_models.cli import main\n"
            "assert main(['list']) == 0\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_posebh_does_not_import_the_vitpose_model(self):
        source = module_source("comparison_models.posebh.model")
        self.assertNotIn("comparison_models.vitpose", source)
        self.assertNotIn("external_models", source)
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertFalse({m for m in imported if "vitpose" in m}, imported)

    def test_posebh_dataset_no_longer_inherits_a_vitpose_named_class(self):
        source = module_source("comparison_models.posebh.adapter_dataset")
        self.assertIn("BirdGazeCropDataset", source)
        self.assertNotIn("BirdGazeViTPoseDataset", source)

    def test_no_python_module_is_imported_from_the_old_package(self):
        offenders = []
        for path in REPO.rglob("*.py"):
            if any(part in {".git", "__pycache__", ".external_pose_sources"}
                   for part in path.parts):
                continue
            for root_module in imported_root_modules(path):
                if root_module == "external_models":
                    offenders.append(str(path.relative_to(REPO)))
                    break
        self.assertEqual(offenders, [])

    def test_comparison_runtime_code_never_imports_native_research_package(self):
        # Tests may deliberately cross-check against native geometry helpers;
        # the shipped comparison runtime must not depend on them.
        offenders = []
        for path in (REPO / "comparison_models").rglob("*.py"):
            if "__pycache__" in path.parts or "tests" in path.parts:
                continue
            if "Bird_HKE" in imported_root_modules(path):
                offenders.append(str(path.relative_to(REPO)))
        self.assertEqual(offenders, [])

    def test_native_research_package_never_imports_comparison_models(self):
        offenders = []
        for path in (REPO / "Bird_HKE").rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if "comparison_models" in imported_root_modules(path):
                offenders.append(str(path.relative_to(REPO)))
        self.assertEqual(offenders, [])


class NativePreservationTests(unittest.TestCase):
    def test_native_trainer_and_experiment_configs_are_intact(self):
        self.assertTrue((REPO / "Bird_HKE/main.py").is_file())
        self.assertTrue((REPO / "Bird_HKE/run_all.py").is_file())
        configs = sorted((REPO / "Bird_HKE/experiments").rglob("*.yaml"))
        self.assertEqual(len(configs), 18, [c.name for c in configs])
        self.assertEqual({c.parent.name for c in configs},
                         {"HRNet", "HR_Mamba", "HR_MambaVision", "HR_MambaViT",
                          "VHR_BirdPose"})
        self.assertTrue((REPO / "Bird_HKE/lib").is_dir())
        self.assertTrue((REPO / "Bird_HKE/tools/audit_training_protocol.py").is_file())

    def test_native_trained_model_roots_are_untouched_by_comparison_layout(self):
        for model in COMPARISON_MODELS.models.values():
            self.assertNotIn("comparison_models", model.output_root)


class CliSafetyTests(unittest.TestCase):
    def test_train_requires_an_explicit_single_model(self):
        result = subprocess.run(
            [sys.executable, "-m", "comparison_models.cli", "train",
             "--model", "all", "--scenario", "FD"],
            cwd=REPO, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_preflight_without_environments_fails_closed_without_training(self):
        env = {"VITPOSE_ENV_PREFIX": str(REPO / "does-not-exist"),
               "POSEBH_ENV_PREFIX": str(REPO / "does-not-exist"),
               "RTMPOSE_ENV_PREFIX": str(REPO / "does-not-exist")}
        import os
        full = dict(os.environ, **env)
        result = subprocess.run(
            [sys.executable, "-m", "comparison_models.cli", "preflight",
             "--model", "all", "--scenario", "FD"],
            cwd=REPO, capture_output=True, text=True, check=False, env=full)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PREFLIGHT FAILED", result.stdout)
        # A failed preflight must not create any definitive run directory.
        for model in COMPARISON_MODELS.models.values():
            self.assertFalse((REPO / model.output_root).exists(),
                             model.output_root)

    def test_list_output_is_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "comparison_models.cli", "list"],
            cwd=REPO, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual({m["id"] for m in payload["models"]},
                         set(COMPARISON_MODELS.ids()))
        self.assertEqual(payload["native_research_package"], "Bird_HKE/")


if __name__ == "__main__":
    unittest.main()
