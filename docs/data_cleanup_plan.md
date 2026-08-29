# P01 V2 data cleanup plan

## KEEP

- `data/raw/`: immutable original photographs, including every P01 source file.
- `configs/classes.txt`, V1 experiment records under `results/`, and local model
  weights: source evidence or reproducibility records.
- Existing P01 V1 labels and manifests while they are copied into `data/v1/`.

## REBUILD

- `data/v2/images/`, `data/v2/labels/`, `data/v2/previews/`, manifests, splits,
  and YOLO materialisation.  These are deterministic outputs of the audited V2
  build script and raw data.

## DELETE

- `data/curated/`, `data/curated_labels/`, `data/session_reorg/`, and
  `data/splits_v2/`: the local-only failed V2 interpretation, all reproducible
  and not the sole copy of any image or V1 label.
- Old generated `data/annotation_preview/` and `data/yolo_dataset/` materialised
  copies after V1 metadata is safely archived; their raw and canonical labels
  remain elsewhere.

## ARCHIVE

- Copy the V1 canonical labels, split manifests, and dataset configuration to
  `data/v1/`.  `data/v1/README.md` records their original locations and the fact
  that the V1 baseline was not retrained.
