# Final Demo Guide

1. **Real-time detection:** show mouse, keyboard, laptop, cup and headphones with bbox, class, confidence and FPS.
2. **Multi-object:** show mouse+keyboard or cup+headphones simultaneously.
3. **ROS2:** terminal A runs `ros2 run desktop_object_detector desktop_object_detector`; terminal B runs `ros2 topic echo /detections_json`.
4. **Topics:** run `ros2 topic list` and show `/detections`, `/detections_json`, `/detection_image`.
5. **Performance:** show real Jetson runtime evidence; do not quote an FPS unless its benchmark file is available.
6. **Final result:** manual test 20/20, 100%, requirement >=80%, PASS.

Suggested 2–3 minute recording: 0:00 project/camera; 0:15 single classes; 0:50 multi-object; 1:10 Jetson; 1:40 ROS2 echo; 2:10 acceptance result.
