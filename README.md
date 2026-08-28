# Desktop Object Detection with YOLO and ROS2

This project implements a real-time desktop object detection system using YOLO on NVIDIA Jetson and publishes detection results through ROS2.

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

## Repository Structure

    configs/     Configuration files
    data/        Dataset documentation
    docs/        Project documentation
    scripts/     Data processing and training scripts
    models/      Model documentation
    results/     Evaluation results
    ros2_ws/     ROS2 workspace

Large image datasets, videos and model binaries are not committed directly to this repository.
