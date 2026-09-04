# Final Results Summary

## Dataset

The frozen five-class dataset contains mouse, keyboard, laptop, cup and headphones and passed integrity, label and split validation.

## Offline Test Evaluation

YOLO11n Test: Precision 0.842738, Recall 0.725178, mAP50 0.768520, mAP50-95 0.419046.

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| mouse | .931163 | .786494 | .848984 | .585096 |
| keyboard | .894553 | .206924 | .310810 | .163985 |
| laptop | .487545 | .952381 | .875237 | .460737 |
| cup | .968152 | .759036 | .889439 | .473281 |
| headphones | .932275 | .921053 | .918127 | .412129 |

## Comparison and deployment

YOLO11n was selected over YOLO11s because it is smaller and better suited to edge deployment; the YOLO11s accuracy benefit was limited in the recorded comparison.

## Manual 20-Object Evaluation

20 tested, 20 correct, 0 incorrect: 100%, requirement >=80%, PASS. This is a real USB-camera acceptance result, not theoretical overall model accuracy.

## ROS2 and Jetson

ROS2 node `desktop_object_detector`, Jetson deployment and Jetson+ROS2 real-device acceptance: PASS. No numerical Jetson FPS is recorded here.

## Final Status

System Acceptance: PASS.
