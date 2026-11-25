#!/usr/bin/env python3
"""Create a reproducible MMPose RTMPose-only import overlay, without touching upstream.

Pinned MMPose imports its optional RTMOHead unconditionally, which requires
MMDetection. MMDetection 3.3.0 rejects MMCV 2.2.0. This overlay removes
*only RTMOHead registration/imports* and preserves the CSPNeXt-M backbone,
RTMCC head, loss, transforms, configuration, and pinned upstream checkout.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / ".external_pose_sources" / "rtmpose"
OVERLAY = ROOT / ".external_pose_sources" / "rtmpose_birdgaze_overlay"
PIN = "759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
PATCHES = {
    "models/heads/hybrid_heads/__init__.py": [
        ("from .rtmo_head import RTMOHead\n", ""),
        ("'YOLOXPoseHead', 'RTMOHead'", "'YOLOXPoseHead'"),
    ],
    "models/heads/__init__.py": [
        ("from .hybrid_heads import DEKRHead, RTMOHead, VisPredictHead",
         "from .hybrid_heads import DEKRHead, VisPredictHead"),
        ("'InternetHead', 'RTMWHead', 'RTMOHead'",
         "'InternetHead', 'RTMWHead'"),
    ],
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_bytes(rel: str) -> bytes:
    git_rel = "mmpose/" + rel
    expected = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "show", f"{PIN}:{git_rel}"])
    local = (UPSTREAM / git_rel).read_bytes()
    if local != expected:
        raise RuntimeError(
            f"Refusing overlay: upstream source modified: {git_rel}")
    return local


def expected_patch(rel: str, before: bytes) -> bytes:
    current = before.decode("utf-8")
    for old, new in PATCHES[rel]:
        if current.count(old) != 1:
            raise RuntimeError(
                f"Unexpected upstream MMPose code in {rel}: {old!r}")
        current = current.replace(old, new, 1)
    return current.encode("utf-8")


def expected_manifest() -> dict:
    return {
        "origin": "open-mmlab/mmpose",
        "pinned_revision": PIN,
        "purpose": "Disable unrelated RTMOHead import requiring MMDetection",
        "changes": {
            rel: {
                "upstream_sha256": sha256(source_bytes(rel)),
                "overlay_sha256": sha256(expected_patch(rel, source_bytes(rel))),
            }
            for rel in sorted(PATCHES)
        },
    }


def verify_overlay(manifest: dict) -> None:
    path = OVERLAY / "manifest.json"
    if not path.is_file():
        raise RuntimeError(
            f"Existing overlay missing manifest: {OVERLAY}; preserve and inspect it")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != manifest:
        raise RuntimeError(
            f"Existing overlay differs from pinned MMPose recipe: {OVERLAY}")
    for rel, hashes in manifest["changes"].items():
        target = OVERLAY / "mmpose" / rel
        if not target.is_file() or sha256(target.read_bytes()) != hashes["overlay_sha256"]:
            raise RuntimeError(f"Existing overlay has drifted: {target}")


def main() -> None:
    actual = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != PIN:
        raise RuntimeError(
            f"Pinned upstream mismatch: {actual}; expected {PIN}")
    manifest = expected_manifest()
    if OVERLAY.exists():
        verify_overlay(manifest)
        print("RTMPose-only overlay already exists and matches pinned recipe:", OVERLAY)
        return
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=OVERLAY.parent, prefix=".rtmpose_overlay_build_"
    ) as temp:
        build = Path(temp) / "overlay"
        src_pkg = UPSTREAM / "mmpose"
        dst_pkg = build / "mmpose"
        shutil.copytree(
            src_pkg, dst_pkg,
            ignore=shutil.ignore_patterns(".mim", "__pycache__", "*.pyc"),
        )
        for rel in sorted(PATCHES):
            before = source_bytes(rel)
            (dst_pkg / rel).write_bytes(expected_patch(rel, before))

        # MMPose's mmengine config scope uses mmpose/.mim/configs.
        # Link back to original untouched, pinned config files.
        original_mim = src_pkg / ".mim"
        if original_mim.exists():
            (dst_pkg / ".mim").symlink_to(original_mim.resolve(),
                                          target_is_directory=True)
        else:
            (dst_pkg / ".mim").mkdir()
            (dst_pkg / ".mim" / "configs").symlink_to(
                (UPSTREAM / "configs").resolve(), target_is_directory=True)
        build.mkdir(exist_ok=True)
        (build / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(build, OVERLAY)
    verify_overlay(manifest)
    print("Created RTMPose-only MMPose overlay:", OVERLAY)
    print("Skipped only unused RTMOHead imports; upstream checkout unchanged.")


if __name__ == "__main__":
    main()
