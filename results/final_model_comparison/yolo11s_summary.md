# yolo11s final fine-tuning

正式训练前进行数据完整性校验，发现 1 个贴近图像边缘的 bbox 因归一化小数舍入产生 0.0005 越界，经原图视觉确认后裁剪至合法图像边界并重新归一化。该修复发生在任何模型训练和测试之前。

- Pretrained weights: `yolo11s.pt`
- Run: `D:\codes\desktop-object-detection-ros2\runs\detect\yolo11s_final`
- Best epoch: 18; completed: 38/100; training seconds: 3175.534
- Val: P=0.649330, R=0.500538, mAP50=0.541854, mAP50-95=0.355897
- Test: P=0.877823, R=0.710737, mAP50=0.794705, mAP50-95=0.420278

| Class | P | R | AP50 | AP50-95 | Instances |
|---|---:|---:|---:|---:|---:|
| mouse | 0.9451754165497337 | 0.8018662482133734 | 0.8362328570585073 | 0.5561364910601684 | 86 |
| keyboard | 0.9045571403381266 | 0.20121951219512196 | 0.39378025678194073 | 0.20302969214798172 | 164 |
| laptop | 0.8104344061009175 | 0.8809523809523809 | 0.9126537569973752 | 0.5024342477328037 | 42 |
| cup | 0.8654366625123123 | 0.7749078035461846 | 0.8836355085205403 | 0.41865620036829576 | 83 |
| headphones | 0.8635118400986445 | 0.8947368421052632 | 0.9472246143055226 | 0.4211316776618679 | 38 |

Error counts at conf=0.25: {'false_negative': 147, 'false_positive': 177, 'low_confidence_correct': 27, 'class_confusion': 6, 'localization_error': 10}. Test was used once after fixing best.pt and did not participate in selection or tuning.
