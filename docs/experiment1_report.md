# 实验一 目标检测与识别实验

## 1. 实验目的与要求
完成不少于两类桌面物体的自采集、标注、训练、Jetson 实时识别和 ROS2 发布；验收包含 20 个物体 >=80%、Jetson >=5 FPS 和结果保存。

## 2. 环境、数据与标注
五类为 mouse、keyboard、laptop、cup、headphones。数据经人工标注为 YOLO `class_id x_center y_center width height`，并在训练前检查标签、边界和 Train/Val/Test split。

## 3. 模型训练与测试
使用 COCO pretrained YOLO11 进行 Fine-tuning；训练代码为 `scripts/train_final_yolo_comparison.py`，最终部署为 YOLO11n `runs/detect/yolo11n_final/weights/best.pt`。固定 Test 结果：P=.842738，R=.725178，mAP50=.768520，mAP50-95=.419046。YOLO11n 较小，更适合边缘部署。

## 4. 实时、Jetson 与 ROS2
USB Camera 由 OpenCV 读取，推理显示 bbox、类别、置信度。Jetson 实机部署、ROS2 runtime 和 Jetson+ROS2 验收均已通过；ROS2 节点为 `desktop_object_detector`，发布 `/detections`、`/detections_json` 和可选 `/detection_image`。本地没有可引用的 Jetson FPS 数值，因此不报告数字。

## 5. 验收与错误分析
人工实物测试 20/20 正确、100%、PASS；这不是总体理论准确率。真实离线错误分析保留 false negative、false positive、class confusion 和 localization error。

## 6. 结论
类别数、训练、结果保存、Jetson 和 ROS2 均满足项目要求。最终视频与真实 Jetson FPS 记录仍需在课程提交材料中补充。
