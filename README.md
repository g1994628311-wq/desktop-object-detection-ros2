# Desktop Object Detection with YOLO and ROS2

This project implements a real-time desktop object detection system using YOLO on NVIDIA Jetson and publishes detection results through ROS2.

## Dataset status

The current P01 collection has 87 manually annotated images and 168 instances. Canonical labels are in `data/labels/`; session-grouped manifests are in `data/splits/`.

Run `python scripts/validate_labels.py` and `python scripts/build_yolo_dataset.py` to validate and rebuild the local YOLO dataset. See `docs/annotation_guidelines.md` and `docs/dataset_summary.md`. Training and deployment are outside the current phase.

## Target Classes

| ID | Class |
|---|---|
| 0 | mouse |
| 1 | keyboard |
| 2 | laptop |
| 3 | cup |
| 4 | headphones |

## Project Requirements

- Detect at least two object classes
- Use self-collected and manually annotated data
- Run real-time object detection on NVIDIA Jetson
- Display class name, bounding box and confidence
- Publish detection results through ROS2
- Recognition accuracy >= 80% on 20 test objects
- Detection speed >= 5 FPS
- Save test results and representative failure cases

## Pipeline

Data Collection
-> Annotation
-> Dataset Validation
-> YOLO Training
-> PC Evaluation
-> Jetson TensorRT Deployment
-> ROS2 Publisher
-> Final Evaluation

## Evaluation / Results

Final model: YOLO11n

Manual real-world evaluation: 20/20 objects correctly recognized
Accuracy: 100%
Requirement: >=80%
Status: PASS

This manual USB-camera acceptance result is distinct from held-out Test mAP, the YOLO11n / YOLO11s comparison, and USB Camera FPS. It records only the confirmed 20-object manual test and is not a claim of overall model accuracy.

Jetson real-device deployment: PASS
Jetson average FPS: 55 FPS
Jetson requirement: >=5 FPS
ROS2 real-device runtime: PASS

For runnable entry points, see `scripts/`, `docs/ros2_detector_usage.md`, `docs/jetson_deployment.md`, and `docs/final_demo_guide.md`.

## Repository Structure

    configs/     Configuration files
    data/        Dataset documentation
    docs/        Project documentation
    scripts/     Data processing and training scripts
    models/      Model documentation
    results/     Evaluation results
    ros2_ws/     ROS2 workspace

Large image datasets, videos and model binaries are not committed directly to this repository.
