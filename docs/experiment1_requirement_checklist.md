# 实验一课程要求对照表

| Requirement | Evidence | Status |
|---|---|---|
| >=2 classes | Five classes: mouse, keyboard, laptop, cup, headphones | PASS |
| Self-collected dataset / annotation | Dataset manifests and YOLO labels under `data/` | PASS |
| Model training | `scripts/train_final_yolo_comparison.py` and recorded results | PASS |
| Jetson deployment | User-confirmed real-device acceptance | PASS |
| bbox/class/confidence display | USB and ROS2 inference programs | PASS |
| ROS2 publishing | `desktop_object_detector`; `/detections`, `/detections_json`, `/detection_image` | PASS |
| Simultaneous >=2 classes | Final demo procedure requires two-class frame | EVIDENCE MISSING |
| 20-object accuracy | `results/final_acceptance/manual_20_object_evaluation.md`: 20/20 | PASS |
| Jetson >=5 FPS | Jetson runtime acceptance confirmed; measured FPS record not found | EVIDENCE MISSING |
| Saved results | Offline metrics, manual acceptance and error CSV are retained | PASS |
| Typical error cases | `results/final_model_comparison/error_cases_yolo11n.csv` | PASS |
