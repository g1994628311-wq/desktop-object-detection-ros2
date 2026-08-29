# Final data structure

- `data/raw/`: immutable source images; never delete or edit.
- `data/v1/`: tracked V1 labels, train/val/test path lists, and a `metadata/`
  archive containing the original dataset configuration and split summary.
- `data/v2/labels`, `manifests`, `splits`, and `yolo/dataset.yaml`: tracked V2
  source-of-truth metadata.
- `data/v2/images`, `previews`, and `yolo/images|labels`: generated local
  copies; safe to regenerate with `scripts/build_p01_v2.py`.

The obsolete `curated*`, `session_reorg`, `splits_v2`, old annotation previews,
and generated V1 YOLO copies were removed in the earlier cleanup because raw
images and archived V1 labels remain the authoritative inputs. This rebuild
created no temporary data directory; its only audit-only coordinate grids and
raw contact sheets live outside the repository visualization workspace.
