# Annotatiathon 2026 → BirdGaze (eBird + NABirds)

This converter builds a **single BirdGaze-compatible dataset**, with only two
annotation files: annot/ebird.json and annot/nabirds.json.

The 2026 Supervisely exports have ONLY per-image annotation JSONs, with names
such as 00003100-347e-48f2-a5c3-c0e9a09a51ff.jpg.json. There are no exported
Supervisely images. Original media come from the user-provided eBird and
NABirds source directories. Neither source needs to be exported again.

## Exact provenance: recovered from the Nightjar source code

The Supervisely-Monitoring-App repository uploads official source frames with:

~~~python
def source_image_name(asset):
    return asset.id + Path(asset.relative_path).suffix.lower()
~~~

See src/annotatiathon/native_media_pool.py. Consequently the UUID in each
Supervisely source-image annotation filename equals media_assets.id in the
Nightjar PostgreSQL database. The original image path is media_assets.relative_path.

The authoritative relationship is:

~~~text
<uuid>.jpg.json
  -> media_assets.id = <uuid>
  -> media_datasets.native_project_id = 386759 or 386817
  -> media_assets.relative_path / sha256 / size_bytes
  -> original eBird frame or NABirds photograph
~~~

Mounted Nightjar datasets typically store relative paths like
videos/<species>/<video>/<frame>.jpg and images/<species>/<photo>.jpg.
The converter removes the leading videos/ or images/ prefix when resolving
files under the original local media root.

If the original files were reorganized after import and the relative path no
longer resolves, it searches the specified source root using the **exact
SHA-256 stored by Nightjar**, constrained by original byte size. There must be
exactly one matching original. It never guesses a match from image dimensions,
list order, nearby frames, or similar-looking image content.

## Source directories

- eBird annotations: E:\Annotatiathon 2026\386759_Nightjar · Official Annotatiathon Dataset · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_425800a7-4c09-4ebf-b6ef-2e78d7e62541]\Complete video library · all species\ann
- NABirds annotations: E:\Annotatiathon 2026\386817_Nightjar · NABirds · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_b7ef26cf-2084-4a52-9293-19a46aceac87]\Complete video library · all species\ann
- eBird frames: E:\Official-Annotatiathon-Dataset\Official Annotatiathon Dataset
- NABirds images: E:\BirdGaze_v2\BirdGaze_Full_Dataset\images\NABirds
- Media mapping CSV: E:\Annotatiathon 2026\Nightjar_media_asset_map.csv
- Output: E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026

All five path arguments have these defaults.

## Step 1. Export the READ-ONLY media map from Nightjar

Use the **original** Supervisely-Monitoring-App deployment and its original
PostgreSQL volume. Do not create another empty PostgreSQL database or run a
deployment/reset. Keep .env.live private.

From the Nightjar repo directory (the one containing compose.live.yaml and
.env.live), run the script in this Bird_HKE branch:

~~~powershell
& "C:\Users\User\Desktop\IST\Mestrado\Tese\pose_estimation_clean\Bird_HKE\Bird_HKE\tools\export_nightjar_media_map.ps1"
~~~

Alternatively, run it from elsewhere using -NightjarRepo with the actual
Supervisely-Monitoring-App repository path.

The script runs a read-only SELECT against the running Nightjar PostgreSQL
container, exports only media_assets belonging to Supervisely project IDs
386759 and 386817, and copies a UTF-8 CSV to
E:\Annotatiathon 2026\Nightjar_media_asset_map.csv. No credentials are printed;
no Nightjar or Supervisely objects are modified.

It must find rows for both projects. Verify:

~~~powershell
Import-Csv "E:\Annotatiathon 2026\Nightjar_media_asset_map.csv" |
    Group-Object native_project_id |
    Select-Object Name, Count
~~~

If it reports no rows, check the ORIGINAL live Nightjar database, not the
Supervisely annotation files; the UUID mapping cannot be reconstructed from
annotation JSONs and original media filenames alone.

## Step 2. Audit mappings and coordinates

From the Bird_HKE repository root:

~~~powershell
python -m pip install Pillow
python Bird_HKE/tools/build_annotatiathon_birdgaze.py --audit-only
~~~

The audit validates exact UUID/project matching, existence of original files,
file-size and image-dimension agreement, bird bounding boxes, duplicate and
out-of-bounds joints. If the original path changed, it may scan images to find
an exact SHA-256 match.

Expected source annotation counts:
- eBird: 50,447
- NABirds: 48,562
- Combined: 99,009

Inspect:
- reports/validation_report.json (counts and match methods)
- reports/rejected_annotations.csv (all failures)
- reports/manifest.csv (complete UUID -> original filename -> output mapping)

The audit does NOT copy any media or produce annotation JSON outputs.
It returns 0 for zero rejections, 2 if any annotations were rejected, and 1
for an operational error.

## Step 3. Build after audit passes

~~~powershell
python Bird_HKE/tools/build_annotatiathon_birdgaze.py
~~~

The output is:

~~~text
BirdGaze_Annotatiathon_2026/
  images/
    eBird/<species>/<video>/<frame>.jpg
    NABirds/<relative-original-image-path>
  annot/
    ebird.json
    nabirds.json
  reports/
    manifest.csv
    rejected_annotations.csv
    validation_report.json
~~~

ebird.json and nabirds.json are arrays of BirdGaze records with image,
bbox, center, scale, joints and joints_vis. Keypoint order is crown, left eye,
right eye, beak. Occluded keypoints retain their annotated coordinates, but
are masked using joints_vis=0. Unannotated keypoints are [-1,-1] with
joints_vis=0. The scale follows the existing BirdGaze loader's expected
pixel_std=200 convention.

The converter does not generate train.json, val.json, test.json or
occlusion-only annotation files.

Source files are untouched. Default mode **copies** each original image.
Optional --materialize hardlink uses hardlinks within the same disk volume,
but then the original and output file share bytes and must be treated as
immutable. Use --skip-sizes-if-existing after an interrupted copy to resume.
Use --verify-sha256 during audit for additional whole-file integrity
verification (can require reading 100+ GB). Without this flag, source paths
are checked by dimensions and database-recorded byte size. Hash restoration
after path mismatch always verifies SHA-256.

By default any rejection blocks the build. The --allow-partial option can
produce a known incomplete subset and records all exclusions in reports;
do not use it for the official complete dataset.

## Synthetic tests

~~~powershell
python -m unittest discover -s Bird_HKE/tools -p "test_build_annotatiathon_birdgaze.py" -v
~~~

Tests exercise exact ID mapping, source project isolation, original paths,
occlusion preservation, source mismatch rejection, exact SHA-256 fallback,
ambiguous duplicate images, audit-only behavior, and safe resumability.
Actual E: drive media and the live PostgreSQL database must be checked by
running the audit on the user's machine.
