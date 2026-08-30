# V3 three-class dataset

V3 is an independent dataset for the fixed ontology `laptop`, `keyboard`, and `cup`. It does not modify or replace V1/V2, and it does not change `configs/classes.txt`.

## Sources and scope

- User domain: 119 accepted P01 images mapped through the verified V2 capture-session manifest. The one raw image previously rejected by manual quality review remains untouched in `data/raw/P01` and is recorded in `data/v3/manifests/excluded_images.csv`.
- Auxiliary domain: official COCO 2017 `train2017` detection images and `instances_train2017.json`. A deterministic candidate pool of 176 images was downloaded from the official COCO S3 bucket; 150 were retained after balancing the combined train instance distribution.
- COCO labels retain only category names queried as `laptop`, `keyboard`, and `cup`; `iscrowd` and invalid boxes are excluded. COCO is train-only.

The 119 accepted user images are copied byte-for-byte from raw. Their V3 labels were reconstructed in the new class space, all 119 previews were reviewed a second time, and mouse/headphones are never V3 targets. Empty label files are retained for 31 hard-negative images.

## Split policy

Capture sessions are indivisible:

| Split | User sessions | User images | laptop | keyboard | cup |
| --- | --- | ---: | ---: | ---: | ---: |
| train | S01, S02 | 88 | 32 | 19 | 47 |
| val | S05 | 5 | 5 | 0 | 4 |
| test | S03, S04 | 26 | 7 | 2 | 14 |

The split prioritizes capture-session isolation and gives the formal user test non-zero coverage for all three classes. Keyboard cannot also appear in validation without splitting a session or removing keyboard coverage from train/test, because keyboard occurs only in S02 and S04. The resulting validation set therefore has zero keyboard instances; this is an explicit data limitation, not a random-split omission.

Final YOLO train is 88 user-train images plus 150 COCO train2017 images. Validation and test are user-only. COCO does not enter the formal user-domain test.

## Counts

| Section | Images | Objects | laptop | keyboard | cup |
| --- | ---: | ---: | ---: | ---: | ---: |
| all user | 119 | 130 | 44 | 21 | 65 |
| retained COCO | 150 | 1011 | 394 | 263 | 354 |
| combined train | 238 | 1109 | 426 | 282 | 401 |
| user val | 5 | 9 | 5 | 0 | 4 |
| user test | 26 | 23 | 7 | 2 | 14 |

The canonical machine-readable counts are in `data/v3/manifests/dataset_summary.csv` and `split_summary.csv`.

## Reproduction and validation

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\build_v3_dataset.py
.\.venv\Scripts\python.exe scripts\validate_v3_dataset.py
```

The builder downloads only the official annotation archive and a deterministic 176-image candidate pool, then retains 150. Images, previews, YOLO materializations, and download caches are intentionally Git-ignored. Labels, manifests, splits, configuration, documentation, and scripts are tracked.

Validation checks class and box validity, image/label pairing, user session and path isolation, COCO-ID uniqueness, user/COCO exact hashes, user-only val/test, and materialized split integrity. Two byte-identical user pairs already accepted in V2 occur within their respective single capture sessions; they are non-critical warnings because neither pair crosses a split or source.
