# ROS2 detector usage

Windows: `cd ros2_ws; colcon build --symlink-install; .\install\setup.ps1; ros2 run desktop_object_detector desktop_object_detector`.

Linux/Jetson: `cd ros2_ws && colcon build --symlink-install && source install/setup.bash && ros2 run desktop_object_detector desktop_object_detector`.

Inspect: `ros2 topic list`, `ros2 topic echo /detections_json`, `ros2 topic hz /detections_json`, `ros2 topic echo /detections`.

Teacher demo: terminal A starts the node; terminal B echoes `/detections_json`. Windows defaults are USB camera index 1 and DSHOW; Jetson camera/path are configurable parameters. The node publishes `/detections` (`vision_msgs/msg/Detection2DArray`), `/detections_json` (`std_msgs/msg/String`), and optionally `/detection_image` (`sensor_msgs/msg/Image`).
