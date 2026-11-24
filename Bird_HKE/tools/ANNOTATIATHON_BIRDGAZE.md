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

# Audit only (does not copy images or write annotation outputs):
python Bird_HKE/tools/build_annotatiathon_birdgaze.py --audit-only

# After inspecting reports and confirming zero rejections:
python Bird_HKE/tools/build_annotatiathon_birdgaze.py
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
    ebird.json
    nabirds.json
  reports/
    validation_report.json
    rejected_annotations.csv
    manifest.csv
~~~

Each annotation has BirdGaze fields image, bbox, center, scale, joints,
joints_vis. The four ordered joints are crown, left eye, right eye, and
beak. Occluded points preserve their annotated coordinates, but use
joints_vis=0; missing points use [-1, -1]. 'scale' is the longest bbox
dimension divided by 200, before the existing loader's 1.25 padding.

The audit manifest records matched image files, their SHA256, optional
image-info paths, any original source path, and match method:
- nightjar_uuid_sha256
- sha256_original
- supervisely_export

Expected annotation inputs based on the previous failed audit:
- eBird: 50,447
- NABirds: 48,562
- Total: 99,009

The new export may have more or fewer files; compare actual counts.
Any rejected annotation blocks the build by default.

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
