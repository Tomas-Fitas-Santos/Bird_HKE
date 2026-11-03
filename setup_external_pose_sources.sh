#!/usr/bin/env bash
# Fetch the exact upstream source revisions used for external BirdGaze baselines.
# Run from any directory: bash setup_external_pose_sources.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES="$ROOT/.external_pose_sources"
mkdir -p "$SOURCES"

fetch_source() {
    local name="$1" url="$2" revision="$3" path="$SOURCES/$1"
    if [[ -e "$path" && ! -d "$path/.git" ]]; then
        echo "ERROR: $path exists but is not a Git clone" >&2
        return 1
    fi
    if [[ ! -d "$path/.git" ]]; then
        echo "Cloning $name..."
        git clone --quiet "$url" "$path"
    fi
    if [[ -n "$(git -C "$path" status --porcelain)" ]]; then
        echo "ERROR: $name has uncommitted edits; refusing to change checkout" >&2
        return 1
    fi
    if ! git -C "$path" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -C "$path" fetch origin "$revision"
    fi
    git -C "$path" -c advice.detachedHead=false checkout --quiet --detach "$revision"
    local actual
    actual="$(git -C "$path" rev-parse HEAD)"
    [[ "$actual" == "$revision" ]] || { echo "ERROR: unexpected $name revision" >&2; return 1; }
    printf '%-13s %s\n' "$name" "$actual"
}

fetch_source rtmpose https://github.com/open-mmlab/mmpose.git 759b39c13fea6ba094afc1fa932f51dc1b11cbf9
fetch_source vitposepp https://github.com/ViTAE-Transformer/ViTPose.git c050ed29112da7704797cc1a65af0234b525010d
fetch_source posebh https://github.com/uyoung-jeong/PoseBH.git 8f03438f913b559edb0a2beed54e77d1560edba3

echo "Upstream sources are pinned in $SOURCES."
echo "No packages were installed and no training was started."
echo "See external_models/README.md for dataset adapters and training requirements."
