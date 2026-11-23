# Annotatiathon 2026 to BirdGaze dataset

Standalone, read-only-source converter for the two Supervisely image annotation exports collected during the 2026 Nightjar Annotatiathon.

This utility does not modify pose models, existing BirdGaze annotations, source images, or original Supervisely JSON files.

## Source paths (configured as defaults)

- eBird per-frame JSONs: E:\Annotatiathon 2026\386759_Nightjar · Official Annotatiathon Dataset · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_425800a7-4c09-4ebf-b6ef-2e78d7e62541]\Complete video library · all species\ann
- NABirds image JSONs: E:\Annotatiathon 2026\386817_Nightjar · NABirds · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_b7ef26cf-2084-4a52-9293-19a46aceac87]\Complete video library · all species\ann
- eBird frames: E:\Official-Annotatiathon-Dataset\Official Annotatiathon Dataset
- NABirds images: E:\BirdGaze_v2\BirdGaze_Full_Dataset\images\NABirds
- New output (default): E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026

## Important: UUID filename matching in Supervisely exports

The Annotatiathon Supervisely exports contain annotation JSONs named with UUIDs, not the original frame/image filenames. Original media cannot be associated with annotations by basename alone. The builder therefore looks for exported images in the sibling `img` folder of each Supervisely `ann` directory (or explicit `--ebird-export-images` and `--nabirds-export-images` overrides).

Resolution order:

1. Match an original by a unique filename if possible.
2. Find the corresponding UUID image in the Supervisely exported `img/` folder and match its **exact SHA-256 hash** to original media of the same size. A unique exact match restores the original relative path, including eBird species/video/frame directories.
3. If no unique original is found, preserve the verified Supervisely export image under `images/eBird/exported/<uuid>.jpg` or `images/NABirds/exported/<uuid>.jpg`. The image remains usable in BirdGaze, but the original video sequence cannot be recovered from the UUID alone.

The `reports/manifest.csv` file includes a `media_match` column describing which path was used, and `reports/validation_report.json` gives per-source match counts. To require exact original matches and never allow the exported UUID fallback, add `--require-original`. To bypass hashing and use the exported image immediately when names differ, add `--no-hash-originals`. **Do not infer temporal order from exported UUID filenames.**

If the export includes only `ann/` and no matching `img/`, and no separate UUID-to-original mapping exists, matching is impossible from filenames alone. Export the project images too, or provide an authoritative manifest linking the UUID names to original image paths.

## Run on the Windows machine holding E:

In PowerShell, from the repository root, first confirm both exported media folders exist:

~~~powershell
Get-ChildItem "E:\Annotatiathon 2026" -Directory -Filter "386759_*" |
    ForEach-Object { Get-ChildItem $_.FullName -Directory -Recurse -Filter img | Select-Object -First 3 FullName }
Get-ChildItem "E:\Annotatiathon 2026" -Directory -Filter "386817_*" |
    ForEach-Object { Get-ChildItem $_.FullName -Directory -Recurse -Filter img | Select-Object -First 3 FullName }
~~~

If these return no image directories, the source exports may be annotation-only. The builder cannot safely reconstruct the match without additional image data or mapping.

Then run the builder:

~~~powershell
python -m pip install Pillow

# First audit all annotation-to-image matches without copying media:
python Bird_HKE/tools/build_annotatiathon_birdgaze.py --audit-only

# Review reports:
# E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026\reports\validation_report.json
# E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026\reports\rejected_annotations.csv

# Only after mismatches are resolved:
python Bird_HKE/tools/build_annotatiathon_birdgaze.py
~~~

Audit returns exit code 0 when all annotations pass, 2 when some are rejected, and 1 on a general error. Rejected annotations prevent bulk copying by default. The option --allow-partial permits excluding invalid records while recording all exclusions; do not use partial mode for the authoritative complete dataset.

For interrupted runs, use --skip-sizes-if-existing. This checks only file sizes, not full file hashes. The default media mode is copy. The option --materialize hardlink is much faster and uses little extra disk space when sources and output are on the same volume, but shares underlying image bytes: use it only when both copies will be treated as immutable.

Override any source directory using --ebird-ann, --nabirds-ann, --ebird-images, --nabirds-images and --output.

## Output

~~~text
BirdGaze_Annotatiathon_2026/
  images/
    eBird/<species>/<video>/<frame>.jpg
    NABirds/<original-relative-image-path>
  annot/
    ebird.json
    nabirds.json
  reports/
    validation_report.json
    rejected_annotations.csv
    manifest.csv
~~~

**Exactly two annotation JSON files are produced:** annot/ebird.json (all accepted eBird frames) and annot/nabirds.json (all accepted NABirds photographs). No train, validation, test, or occlusion-only split files are written. Images remain separated by source and retain their original relative directory structure. Fully occluded samples are kept in their source JSON, although normal supervised evaluation may exclude them later because zero-visible samples make some metrics undefined. If you ran an earlier version of the converter, it does not delete older extra annotation JSON files; remove those manually if you want the directory to contain only the two current outputs.

## BirdGaze format

~~~json
{
  "image": "eBird/Species/Video/Video_000001.jpg",
  "center": [230.0, 156.0],
  "scale": 1.06,
  "bbox": [124.0, 87.0, 336.0, 225.0],
  "joints": [[230, 101], [234, 120], [250, 127], [270, 144]],
  "joints_vis": [1, 0, 1, 1]
}
~~~

Compatible with Bird_HKE/dataset/birdgaze.py. Four joints in order crown, left eye, right eye, beak, absolute pixel coordinates. Visible joint = 1, occluded or unannotated joint = 0. Actual annotated occluded coordinates remain in joints even though masked by joints_vis=0; missing/unannotated points use [-1,-1]. Bird bounding boxes are [x1,y1,x2,y2] in image pixels; scale is the bounding box longest dimension / 200 before the existing loader's 1.25x padding.

The annotated occluded coordinates are preserved **inside the same two BirdGaze annotation JSONs** in the joints arrays; no separate annotation metadata JSON or JSONL is produced.

## Matching and validation

Each Supervisely image annotation is first matched to original media using a unique basename when possible, and then through the exported image and SHA-256 comparison against the original media, constrained by byte size. For UUID exports without a unique exact original, the corresponding exported image itself is used as the authoritative annotated media (unless --require-original is set). Missing exported media, dimension mismatches, invalid bbox coordinates, missing all four keypoints, duplicate keypoints, and out-of-image keypoints cause explicit rejections. SHA-256 matches may fail when images were re-encoded, even if they look identical; the safe fallback preserves the exported media instead.

This converter expects image-format Supervisely objects[], a single rectangular bird label, and point labels crown, left_eye, right_eye, beak, plus their _occ variants. It does not support Supervisely *video-format* exports. Unknown classes are ignored, not remapped. Multiple bounding boxes are rejected rather than assigning points ambiguously.

All conversion logic has been tested on synthetic Supervisely annotations, not the user's actual E: directories; those must be validated locally before the dataset can be regarded as complete.

## Test suite

~~~powershell
python -m unittest discover -s Bird_HKE/tools -p "test_build_annotatiathon_birdgaze.py" -v
~~~

Ten tests cover correct conversion and paths, exactly two output annotations, preserved occluded coordinates including all-occluded samples, audit-only safety, ambiguous image names, mismatched dimensions, duplicate keypoints, explicit reruns, UUID content-hash mapping, exported-media fallback and original-only strict mode.
