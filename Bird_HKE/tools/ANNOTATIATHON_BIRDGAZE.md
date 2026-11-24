# BirdGaze Annotatiathon 2026: eBird + NABirds

This converter creates a new BirdGaze dataset with **exactly two annotation JSONs**,
one for the annotated eBird frames and one for NABirds photos. It does not
modify source images, Supervisely annotations or training code.

## New primary sources

The latest full Supervisely exports are:

- **eBird:** E:\Annotatiathon 2026\Nightjar_Annotatiathon_eBird_videos
- **NABirds:** E:\Annotatiathon 2026\Nightjar_Annotatiathon_NABirds

Expected Supervisely layout (the containing project/dataset folder can be nested):

~~~text
Nightjar_Annotatiathon_eBird_videos/
  [project/dataset/]
    ann/
      <uuid>.jpg.json
    img/
      <uuid>.jpg
    img_info/
      <uuid>.jpg.json

Nightjar_Annotatiathon_NABirds/
  [project/dataset/]
    ann/
    img/
    img_info/
~~~

The builder searches recursively for the named folders. When an export
contains only img/ and img_info/, it uses the **older** annotation JSONs from
the previous source locations below and pairs them to the newly exported
images by exact UUID filename.

Previous annotation directories:

- E:\Annotatiathon 2026\386759_Nightjar · Official Annotatiathon Dataset · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_425800a7-4c09-4ebf-b6ef-2e78d7e62541]\Complete video library · all species\ann
- E:\Annotatiathon 2026\386817_Nightjar · NABirds · Official Event [NJ_ff71f54f-9f77-45c8-8504-c9c4028b0249_b7ef26cf-2084-4a52-9293-19a46aceac87]\Complete video library · all species\ann

The original media roots (optional for restoring the pre-export filename and
video structure) are:

- E:\Official-Annotatiathon-Dataset\Official Annotatiathon Dataset
- E:\BirdGaze_v2\BirdGaze_Full_Dataset\images\NABirds

## Matching algorithm

1. Pair each annotation file ending in .json with the exported image with the
   **same complete filename** (e.g. ann/<uuid>.jpg.json -> img/<uuid>.jpg).
   Files in a matching dataset's sibling img/ folder take precedence over a
   global unique-basename lookup; ambiguous matches fail.
2. Open matching img_info/<uuid>.jpg.json, when present. Supervisely ImageInfo
   hash is Base64-encoded SHA256 of file contents. Hash the exported image
   bytes and verify the metadata SHA256. Incorrect hashes are rejected, not
   silently ignored.
3. To restore **species/video/frame filenames**, optionally use a Nightjar
   media asset mapping CSV if one is already available. The original file
   must have the same exact byte hash as the exported image.
4. Without the CSV, hash original eBird frames/NABirds images once, restricted
   to file sizes occurring in the exported dataset. A single original having
   the same SHA256 allows restoring its path. Duplicates and missing exact
   matches DO NOT get arbitrarily assigned to a source video.
5. When an original match cannot be made, the exported image remains a valid
   image, stored under eBird/exported/<uuid>.jpg or
   NABirds/exported/<uuid>.jpg. It can be used for image pose training, but the
   original video timing/order cannot be reconstructed from that UUID alone.
   See the CSV manifest for each image's provenance.

The exact content hash match is the strongest possible association between
exported and original file bytes. If an image has been re-encoded, its SHA256
changes; the converter preserves the correctly paired Supervisely image
rather than guessing a source filename from visual similarity.

## Running on Windows

From the Bird_HKE repository root:

~~~powershell
git fetch origin
git switch feat/annotatiathon-birdgaze-builder-20261123
git pull --ff-only origin feat/annotatiathon-birdgaze-builder-20261123

python -m pip install Pillow

# Run synthetic tests first:
python -m unittest discover -s Bird_HKE/tools -p "test_build_annotatiathon*.py" -v

# Audit the current Annotatiathon collection; include annotation-only issues
# in the pending-correction category, without copying images yet:
python -u Bird_HKE/tools/build_annotatiathon_birdgaze.py --source-mode exports --retain-annotation-issues --audit-only

# After confirming that non-annotation integrity errors (rejected) are zero:
python -u Bird_HKE/tools/build_annotatiathon_birdgaze.py --source-mode exports --retain-annotation-issues
~~~

**This new export-based mode does NOT require access to the Nightjar database,
Docker, PostgreSQL, or a separate media-mapping CSV.**

Optional arguments:

- --source-mode exports: explicitly select full export mode
- --no-original-lookup: faster scan using exported images and JSONs only;
  all paths stay under eBird/exported/ or NABirds/exported/
- --require-original-paths: reject any image without an unambiguous original
  filename recovered from UUID mapping or exact file-content SHA256
- --ignore-nightjar-map: ignore an existing optional mapping CSV and use
  content hashes alone for original lookups
- --audit-only: generate reports, without copying any media
- --retain-annotation-issues: keep images with known fixable annotation errors
  in the dataset and copy their original annotation JSONs to
  pending_corrections/ rather than treating them as usable training labels
- --use-corrected-annotations: apply edited copies from
  pending_corrections/annotations/ (requires --skip-sizes-if-existing
  when the dataset images were already materialized)
- --allow-partial: build an explicitly incomplete subset despite rejections;
  do NOT use for the authoritative dataset
- --materialize hardlink: faster, no duplicate bytes, but only on the same
  disk volume and the output then shares bytes with the source
- --skip-sizes-if-existing: resume interrupted copy using file sizes

The previous original-mapping-only pipeline can still be selected explicitly
with --source-mode nightjar-map and requires the Nightjar mapping CSV.

## Output

Default destination:
E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026

~~~text
BirdGaze_Annotatiathon_2026/
  images/
    eBird/
      <species>/<video>/<frame>.jpg  (when original hash match is unique)
      exported/<uuid>.jpg           (otherwise)
    NABirds/
      <original relative path>      (when hash match is unique)
      exported/<uuid>.jpg           (otherwise)
  annot/
    ebird.json                   # validated training records only
    nabirds.json                 # validated training records only
  pending_corrections/
    annotations/eBird/<uuid>.jpg.json
    annotations/NABirds/<uuid>.jpg.json
    image_info/eBird/<uuid>.jpg.json
    image_info/NABirds/<uuid>.jpg.json
  reports/
    validation_report.json
    pending_corrections.csv     # original paths, hashes, issue, status
    rejected_annotations.csv    # integrity and non-label failures
    skipped_unannotated.csv
    manifest.csv                # valid and pending images; annotation_status
~~~

Each annotation has BirdGaze fields image, bbox, center, scale, joints,
joints_vis. The four ordered joints are crown, left eye, right eye, and
beak. Occluded points preserve their annotated coordinates, but use
joints_vis=0; missing points use [-1, -1]. 'scale' is the longest bbox
dimension divided by 200, before the existing loader's 1.25 padding.

Unannotated images are normal in a Supervisely export: an image can have an
`ann/*.json` file with `"objects": []` even though no one labeled it.
The new export builder classifies these as **skipped_unannotated**, not
rejected. Their images are not copied to the dataset. Verified annotated
images produce BirdGaze records. With `--retain-annotation-issues`, known
annotation-only issues (duplicate or missing bird/keypoints, image dimensions
disagreeing with the annotation) become **pending_corrections**. Their images
stay under images/eBird/ and images/NABirds/ with recovered original paths
where available; raw Supervisely JSONs and image-info metadata are retained
for later manual correction. They do **not** appear in ebird.json/nabirds.json
until validated. Image/annotation pairing failures, invalid image hashes and
other non-label problems are **rejected** and still block the official build.
The audit reconciles
`accepted + pending_corrections + skipped_unannotated + rejected = annotation_files`.

Use the following command to quickly inspect the last run's rejection
categories before re-running the full original-image SHA-256 scan:

~~~powershell
$report = "E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026\reports\rejected_annotations.csv"
Import-Csv $report | Group-Object source, reason |
    Sort-Object Count -Descending | Select-Object -First 25 Count, Name |
    Format-Table -AutoSize -Wrap
~~~

The next audit prints `valid`, `pending correction`, `unannotated/skipped`,
and `rejected`. Outputs include `reports/pending_corrections.csv` with
exact issue and hash provenance for the images needing correction. If any
non-label rejections remain, inspect them before building. The `--no-original-lookup` option allows a faster
annotation-classification-only audit, but does not restore eBird video paths.
Do a full audit without that option before building the temporal dataset.

## Correcting annotations later

1. Open `reports/pending_corrections.csv` to identify each image and
   the reason its annotation needs review. The image is already located
   under `images/<source>/<original hierarchy>` when an original
   SHA-256 path match exists.
2. Edit the *copy* in `pending_corrections/annotations/<source>/` (or
   replace that copied JSON with a corrected Supervisely annotation).
   Never edit the original export's ann/ files. Keep its matching image
   and UUID filename unchanged.
3. Apply corrected files to the training JSONs using:

~~~powershell
python -u Bird_HKE/tools/build_annotatiathon_birdgaze.py --source-mode exports --retain-annotation-issues --use-corrected-annotations --skip-sizes-if-existing
~~~

Any still-invalid corrections remain `pending_correction`. Corrected
and valid labels move into the appropriate BirdGaze training JSON.
The original Supervisely export is never modified; the correction copies
are independent. The updated audit remains safe:

~~~powershell
python -u Bird_HKE/tools/build_annotatiathon_birdgaze.py --source-mode exports --retain-annotation-issues --use-corrected-annotations --audit-only
~~~

A `--skip-sizes-if-existing` rebuild is designed for the same existing
dataset output (it checks file sizes rather than rehashing existing copies).
Do not use that flag when the existing output images cannot be trusted.

The audit manifest records matched image files, their SHA256, optional
image-info paths, any original source path, and match method:
- nightjar_uuid_sha256
- sha256_original
- supervisely_export

Expected annotation inputs based on the previous failed audit:
- eBird: 50,447
- NABirds: 48,562
- Total: 99,009

The prior audit identified 74,532 validated annotations, 63 annotation
issues, and 24,414 empty annotations. With the retention flag, the expected
output is 74,595 images, 63 records in pending_corrections.csv, two
training annotation JSONs containing 74,532 valid records, and zero
non-label rejections (if no other errors emerge).
**Unannotated images are skipped and do not block building.**
Known annotation-only issues may be retained only with
`--retain-annotation-issues`; other rejections block the build by default.

**Warning:** The converter cannot preserve eBird temporal identity when an
exported UUID image has no verified original path. Do not construct temporal
training sequences from UUID sorting; use only records with recovered video
paths (or the authoritative Nightjar mapping).

## Tests

Two test modules cover original Nightjar-map mode and new full-export mode:

~~~powershell
python -m unittest discover -s Bird_HKE/tools -p "test_build_annotatiathon*.py" -v
~~~

These are synthetic fixtures. No access to the user's E: drive exists in this
development environment, so the real dataset is not marked validated until
the user's local audit has passed.
