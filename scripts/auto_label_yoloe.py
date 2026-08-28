from pathlib import Path

import cv2
from ultralytics import YOLOE


# ============================================================
# Paths
# ============================================================

SOURCE_DIR = Path("data/raw/P01/S01")
LABEL_DIR = Path("data/labels/P01/S01")
PREVIEW_DIR = Path("data/previews/P01/S01")

MODEL_PATH = "yoloe-26s-seg.pt"

MAX_IMAGES = None

# S01 is a basic/single-class collection session.
# We deliberately use a relatively low threshold for recall,
# because all labels will still be manually reviewed.


# ============================================================
# Final dataset classes
# ============================================================

TARGET_NAMES = {
    0: "mouse",
    1: "keyboard",
    2: "laptop",
    3: "cup",
    4: "headphones",
}


# Filename type -> final class + open-vocabulary prompts
TYPE_CONFIG = {

    "MOU": {
        "class_id": 0,
        "class_name": "mouse",
        "threshold": 0.08,
        "prompts": [
            "computer mouse",
            "wireless computer mouse",
            "wired computer mouse",
            "PC mouse",
        ],
    },

    "KEY": {
        "class_id": 1,
        "class_name": "keyboard",
        "threshold": 0.08,
        "prompts": [
            "computer keyboard",
            "external keyboard",
            "mechanical keyboard",
            "PC keyboard",
        ],
    },

    "LAP": {
        "class_id": 2,
        "class_name": "laptop",
        "threshold": 0.08,
        "prompts": [
            "laptop computer",
            "notebook computer",
            "portable computer",
        ],
    },

    "CUP": {
        "class_id": 3,
        "class_name": "cup",
        "threshold": 0.05,
        "prompts": [
            "cup",
            "drinking cup",
            "travel mug",
            "thermos",
            "thermos cup",
            "vacuum flask",
            "insulated bottle",
            "insulated tumbler",
            "water bottle",
            "water cup",
        ],
    },

    "HDP": {
        "class_id": 4,
        "class_name": "headphones",
        "threshold": 0.05,
        "prompts": [
            "headphones",
            "over-ear headphones",
            "full-size headphones",
            "wireless headphones",
            "headband headphones",
            "gaming headset",
            "over-ear headset",
            "earmuff headphones",
        ],
    },
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def get_images(directory: Path):

    images = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images = sorted(images)

    if MAX_IMAGES is not None:
        images = images[:MAX_IMAGES]

    return images


def get_type_code(image_path: Path):

    # Example:
    # P01_S01_CUP_0001.jpg
    #
    # parts:
    # P01 / S01 / CUP / 0001

    parts = image_path.stem.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Invalid filename format: {image_path.name}"
        )

    return parts[2]


def xyxy_to_yolo(box, image_width, image_height):

    x1, y1, x2, y2 = box

    x_center = ((x1 + x2) / 2) / image_width
    y_center = ((y1 + y2) / 2) / image_height

    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return (
        x_center,
        y_center,
        width,
        height,
    )


def process_single_class(
    model,
    image_path,
    config,
):

    prompts = config["prompts"]

    final_class_id = config["class_id"]
    final_class_name = config["class_name"]

    threshold = config["threshold"]

    model.set_classes(prompts)

    results = model.predict(
        source=str(image_path),
        conf=threshold,
        agnostic_nms=True,
        max_det=20,
        verbose=False,
    )

    result = results[0]

    image = result.orig_img.copy()

    image_height, image_width = image.shape[:2]
    image_area = image_width * image_height

    candidates = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box

            box_width = x2 - x1
            box_height = y2 - y1

            box_area = box_width * box_height
            area_ratio = box_area / image_area

            # Filter extremely tiny / implausibly huge detections
            if area_ratio < 0.015:
                continue

            if area_ratio > 0.90:
                continue

            # S01 contains one main object.
            # Prefer boxes that have both good confidence
            # and cover a reasonable portion of the object.
            rank_score = float(score) * (area_ratio ** 0.5)

            candidates.append(
                {
                    "box": box,
                    "confidence": float(score),
                    "rank": rank_score,
                }
            )

    labels = []

    # --------------------------------------------------------
    # IMPORTANT:
    # S01 is a single-object session.
    # Keep ONLY the best candidate.
    # --------------------------------------------------------

    if candidates:

        best = max(
            candidates,
            key=lambda item: item["rank"]
        )

        box = best["box"]
        score = best["confidence"]

        x1, y1, x2, y2 = box

        (
            x_center,
            y_center,
            width,
            height,
        ) = xyxy_to_yolo(
            box,
            image_width,
            image_height,
        )

        labels.append(
            (
                final_class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

        # Preview
        x1_i = int(x1)
        y1_i = int(y1)
        x2_i = int(x2)
        y2_i = int(y2)

        cv2.rectangle(
            image,
            (x1_i, y1_i),
            (x2_i, y2_i),
            (0, 255, 0),
            2,
        )

        text = (
            f"{final_class_name} "
            f"{score:.2f}"
        )

        cv2.putText(
            image,
            text,
            (
                x1_i,
                max(y1_i - 8, 20),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return labels, image

    prompts = config["prompts"]

    final_class_id = config["class_id"]
    final_class_name = config["class_name"]

    # All prompts describe the SAME final class.
    model.set_classes(prompts)

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,

        # Very useful here:
        # suppress strongly overlapping detections produced
        # by different synonym prompts.
        agnostic_nms=True,

        verbose=False,
    )

    result = results[0]

    image = result.orig_img.copy()

    image_height, image_width = image.shape[:2]

    labels = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box

            (
                x_center,
                y_center,
                width,
                height,
            ) = xyxy_to_yolo(
                box,
                image_width,
                image_height,
            )

            labels.append(
                (
                    final_class_id,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )

            # Preview
            x1_i = int(x1)
            y1_i = int(y1)
            x2_i = int(x2)
            y2_i = int(y2)

            cv2.rectangle(
                image,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 0),
                2,
            )

            text = (
                f"{final_class_name} "
                f"{score:.2f}"
            )

            cv2.putText(
                image,
                text,
                (
                    x1_i,
                    max(y1_i - 8, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return labels, image


def save_labels(labels, output_path):

    lines = []

    for (
        class_id,
        x_center,
        y_center,
        width,
        height,
    ) in labels:

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}"
        )

    LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_images(SOURCE_DIR)

    if not images:
        raise RuntimeError(
            "No images found."
        )

    print("=" * 70)
    print("YOLOE PRE-ANNOTATION V0.2")
    print("=" * 70)

    print(f"Source: {SOURCE_DIR}")
    print(f"Images: {len(images)}")

    print()

    model = YOLOE(MODEL_PATH)

    total_detections = 0

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        type_code = get_type_code(
            image_path
        )

        print(
            f"[{index:02d}/{len(images):02d}] "
            f"{image_path.name}"
        )

        # Negative sample
        if type_code == "NEG":

            label_path = (
                LABEL_DIR /
                f"{image_path.stem}.txt"
            )

            label_path.write_text(
                "",
                encoding="utf-8",
            )

            continue

        if type_code not in TYPE_CONFIG:

            print(
                f"    WARNING: unsupported type "
                f"{type_code}"
            )

            continue

        config = TYPE_CONFIG[type_code]

        labels, preview = (
            process_single_class(
                model,
                image_path,
                config,
            )
        )

        label_path = (
            LABEL_DIR /
            f"{image_path.stem}.txt"
        )

        preview_path = (
            PREVIEW_DIR /
            f"{image_path.stem}.jpg"
        )

        save_labels(
            labels,
            label_path,
        )

        cv2.imwrite(
            str(preview_path),
            preview,
        )

        total_detections += len(labels)

        print(
            f"    target class: "
            f"{config['class_name']}"
        )

        print(
            f"    detections: "
            f"{len(labels)}"
        )

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)

    print(
        f"Processed images: "
        f"{len(images)}"
    )

    print(
        f"Total detections: "
        f"{total_detections}"
    )

    print()
    print(
        "All outputs are PRE-ANNOTATIONS "
        "and require manual review."
    )


if __name__ == "__main__":
    main()