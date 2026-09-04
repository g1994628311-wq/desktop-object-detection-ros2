# Final Teacher Questions

1. **Why YOLO11?** It provides a practical accuracy/speed detector pipeline.
2. **Why YOLO11n?** Its smaller size better fits edge deployment.
3. **Why COCO pretraining?** It transfers general visual features to a smaller custom dataset.
4. **Train/Val/Test?** Train updates weights, Val selects/monitors, Test is final untouched evaluation.
5. **mAP50 vs mAP50-95?** The latter averages stricter IoU thresholds.
6. **Precision/Recall?** Precision measures correctness of predictions; Recall measures found ground truth.
7. **Why best.pt?** It is the best Validation checkpoint, unlike the final-epoch last.pt.
8. **Why early stopping?** It avoids continuing when Validation no longer improves.
9. **Why Jetson?** PC proof is not edge-device deployment proof.
10. **ROS2 role?** It communicates structured detections to other robotic components.
11. **Why Detection2DArray and JSON?** Standard typed messages serve systems; JSON is easy to inspect live.
12. **Confidence/NMS?** Confidence filters weak boxes; NMS removes duplicate overlapping boxes.
13. **Pipeline FPS?** It includes camera, inference, postprocess and drawing, not only network time.
14. **Why 20 objects?** It is the course real-world acceptance protocol; 20/20 passed but is not global accuracy.
15. **Limitations?** Keyboard Recall is weaker in the offline Test and performance depends on scene conditions.
