# Final Code Index

- `scripts/train_final_yolo_comparison.py`: local YOLO11n/YOLO11s training and offline evaluation pipeline.
- `scripts/validate_final_yolo_dataset.py`: pre-training dataset integrity and label validation.
- `scripts/usb_camera_model_test.py`: standalone USB Camera YOLO11n inference.
- `scripts/check_jetson_env.py`, `probe_cameras.py`, `jetson_realtime_detect.py`: Jetson deployment utilities.
- `ros2_ws/src/desktop_object_detector/desktop_object_detector/detector_node.py`: ROS2 detection publisher.
- `ros2_ws/src/desktop_object_detector/launch/detector.launch.py`: ROS2 launch entry.
- `ros2_ws/src/desktop_object_detector/package.xml`, `setup.py`: ROS2 package metadata and console script registration.
