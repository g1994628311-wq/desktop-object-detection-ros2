# Desktop5 v2 YOLO dataset

Classes: mouse, external keyboard, laptop, cup/thermos, headband/over-ear headphones.

The v1 strict validation and test boundaries are preserved. P02 and P04 are train-only, so the new data cannot leak into held-out evaluation. Empty label files are intentional negatives.

All 948 image/label pairs decode and pass YOLO coordinate validation. See `manifests/` for the split, class counts, annotation provenance, and duplicate audit.
