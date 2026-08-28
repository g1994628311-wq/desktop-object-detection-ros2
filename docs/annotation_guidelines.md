# Ground Truth Annotation Guidelines

The sole class authority is `configs/classes.txt`: 0 mouse, 1 keyboard, 2 laptop, 3 cup, 4 headphones.

## Box policy

- Inspect every image and label every visible qualifying instance, regardless of filename.
- Use one tight box per physical instance.
- Mouse excludes its pad/long cable. Keyboard means an independent external keyboard.
- Laptop covers the full visible computer; its built-in keyboard is not a separate object.
- Cup includes mugs, tumblers, and thermos cups, including handle/lid.
- Headphones means over-ear/headband headphones; exclude a long cable.
- Label light/moderate occlusion; exclude an unidentifiable severe fragment and record it for review.
- Clip out-of-frame targets at the image boundary.
- A verified negative image has a same-name empty TXT file.

YOLO rows are `class_id x_center y_center width height`. Centers must be in `[0,1]`, sizes in `(0,1]`, and derived boxes must remain inside the image.

Run `python scripts/validate_labels.py` after each session. Previews under `data/annotation_preview/` are local generated artifacts; canonical annotations live under `data/labels/`.
