# BirdGaze inter-annotator agreement study (2026)

## Goal and sampling unit

Build a **double-blind reannotation study** from the validated BirdGaze Annotatiathon dataset:

- Each selected image is assigned to **exactly two different annotators**.
- All selected images were annotated at the event (present in the validated
  \`annot/ebird.json\` or \`annot/nabirds.json\`), not unannotated/empty.
- The new annotators see **only the original bird bounding box**, not any
  event keypoints. The nine Nightjar point classes remain available so
  reviewers can add landmark coordinates and visible/occluded classes.
- Event reference keypoints are stored in **private_reference/**, explicitly
  **outside the Supervisely upload package**.
- eBird frames are stratified by species and video, separated in sorted frame
  order, and selected using farthest-point normalized 2D landmark geometry
  to seek distinct head poses within each video.
- The random seed and selection settings are stored with the sample for audit.

This is inter-annotator reliability measurement, **not** a new ground-truth
gold-standard dataset. The event annotations are a useful third point of
comparison but may contain errors of their own.

## Allocation: doubled two-week plan

| Annotator | Total assignments | Per week | Approx/day over 14 days |
| --- | ---: | ---: | ---: |
| Paulo Lobato Correia | 200 | 100 | 14.3 |
| Ricardo Araújo | 400 | 200 | 28.6 |
| Luís Almeida | 200 | 100 | 14.3 |
| Tomás Santos | 600 | 300 | 42.9 |
| Riccardo Rocchi | 1,000 | 500 | 71.4 |
| Filippo Bertozzo | 800 | 400 | 57.1 |
| Romain David | 400 | 200 | 28.6 |
| Miguel Marx | 400 | 200 | 28.6 |
| Victor Carvalho | 400 | 200 | 28.6 |
| **Total** | **4,400** | **2,200** | **314.3** |

Total: **2,200 different images × 2 independent people per image**.

If Tomás is actually limited to **30 images/day**, 14 days permits 420
rather than 600; if everyone else stays on double quotas, the adjusted
study is 4,220 assignments / 2,110 unique images. Pass
\`--tomas-daily-cap 30\` to select that plan.

## Inputs

Default local BirdGaze source dataset:

\`E:\Annotatiathon 2026\BirdGaze_Annotatiathon_2026\`

Required:

~~~text
images/eBird/<species>/<video>/<frame>.jpg
images/NABirds/<species>/<...>/<photo>.jpg  (only if optional NABirds share > 0)
annot/ebird.json
annot/nabirds.json                            (only if optional NABirds share > 0)
~~~

Only validated image annotations are eligible. Unannotated images, the 63
pending-correction labels, and eBird frames with missing video identity are
excluded. This uses **original eBird video hierarchy**, not UUID sorting.

## Run

From your Bird_HKE repository root:

~~~powershell
git fetch origin
git switch feat/annotatiathon-birdgaze-builder-20261123
git pull --ff-only origin feat/annotatiathon-birdgaze-builder-20261123
python -m pip install Pillow

# Synthetic tests, no full dataset copies:
python -m unittest discover -s Bird_HKE/tools -p "test_build_interannotator_sample.py" -v

# Plan only: check diversity capacity, allocations, and image paths:
python -u Bird_HKE/tools/build_interannotator_sample.py --dry-run

# Write the import-ready Supervisely folder and the private reference:
python -u Bird_HKE/tools/build_interannotator_sample.py
~~~

Default **eBird only** focuses on the requested diverse video poses.
To include 20% labeled NABirds photos, use \`--nabirds-share 0.20\`
during both dry-run and build. E.g. 1,760 eBird frames and 440 NABirds
photos in the standard 2,200-image plan. The sample still requires exactly
two reviewers for every selected image.

If the eBird sample cannot satisfy the strict pose/video constraints,
the program stops with a diagnostic; it **does not silently duplicate
frames**. Adjust \`--max-per-video 16\` or \`--min-pose-distance 0.08\`
explicitly, rerun \`--dry-run\` first, and record the applied settings.
Default parameters: maximum 12 frames per video; minimum 6 positions
apart in sorted frame order; normalized landmark pose distance ≥0.10
for any selected pair within the same video; three annotated head
landmarks minimum; seed 20261124.

For custom people/quotas, pass \`--quotas-json "path\to\quotas.json"\`
with a JSON object mapping person name to integer image counts.
The sum must be even; nobody may receive more assignments than the
number of unique frames.

Optional \`--make-zip\` creates a potentially large
\`supervisely_upload.zip\`. The standard import app accepts a **folder**
as well as an archive, so avoid unnecessary zip/compression when local
folder mounting is available. Output and source folders must be distinct
and the script refuses to overwrite a nonempty output directory.
Do **not** run a build twice to the same destination; use a new output
path for a different sample design.

## Output and access controls

Default output:
\`E:\Annotatiathon 2026\BirdGaze_Reannotation_2026\`

~~~text
BirdGaze_Reannotation_2026/
  supervisely_upload/              # UPLOAD THIS DIRECTORY ONLY
    meta.json                       # nine classes: one bird box + eight point states
    Paulo_Lobato_Correia/
      img/F00001_<hash>.jpg
      ann/F00001_<hash>.jpg.json     # only bird rectangle; no keypoints
    Ricardo_Araujo/
      img/...
      ann/...
    [one dataset per person, nine total]
  private_reference/               # NEVER upload, never share with annotators
    selected_frames.csv             # real species, videos, paths, reviewer pairs
    event_reference_annotations.json# original event keypoints + bbox
  reports/
    assignments.csv                 # sample IDs and assigned person; no event labels
    workloads.csv
    pair_counts.csv
    selection_report.json
  supervisely_upload.zip            # optional, contains upload folder ONLY
~~~

Images have pseudonymous \`F00001_<hash>.jpg\` names, consistent for the
two assigned copies. The file name is the stable linkage key when
downloading reviewers' completed annotations. **Share only the assigned
labeling job with each annotator**, not the full researcher's private files.
These are nine datasets in one project; upload/import does not automatically
create an independently permissioned Supervisely labeling job.

The file layout matches Supervisely's documented
**Import Images in Supervisely Format** app: a project-level \`meta.json\`
and per-dataset \`img/\` + \`ann/\` folders. In the Supervisely web interface,
import the project from the \`supervisely_upload\` folder (or optional ZIP),
then create the nine labeling jobs and assign each dataset to its named
participant. The imported project includes the keypoint classes even though
only the original bounding box is present in annotations.

**Do not upload** \`private_reference\` or the entire parent output directory:
doing so could expose original event landmarks and bias independent ratings.

## Subsequent statistics

Once both completed annotation projects/datasets are exported, match on the
pseudonymous sample ID and original assignment rows:

- two-annotator coordinate disagreement in pixels and normalized by original
  bird bbox size (report median, percentile distribution, thresholds);
- per-point PCK at preregistered bbox-relative thresholds;
- per-point visibility/occlusion agreement and Cohen's kappa where possible;
- frame-level pose variation and discrepancy by species/video;
- optional separate comparisons between each reviewer and the **original
  event reference** (not automatically the ground truth).

Do not average two annotators' coordinates and call them gold labels without
a reconciliation/adjudication protocol. Store completed reviewer results
independently before any discussion or corrections.

## Boundaries and honest limitations

The sampling algorithm measures pose diversity using the **four existing 2D
head landmarks relative to the bird bbox**; this reduces near-identical
annotated poses but does not guarantee different 3D head orientations.
Temporal separation uses sorted source frame filenames, not actual
timestamp/fps information. The original event author identity is not
present in consolidated BirdGaze records, so avoiding reassigning someone
their own original event frame requires an additional author mapping.

This creates a **Supervisely-native import artifact**, not a direct injection
into the Nightjar Monitoring App database or automatic job creation.
