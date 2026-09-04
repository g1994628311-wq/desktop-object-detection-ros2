# 5–8 分钟代码讲解稿

先从数据开始。原始图片经过人工标注后转为 YOLO 标签；训练前我运行数据集验证代码，检查图片和标签是否一一对应、类别 ID 是否是五类范围、bbox 是否在 0 到 1 内，以及同一采集 session 是否泄露到不同 split。这样避免训练结束才发现数据问题。

训练部分使用 COCO 预训练 YOLO11，而不是随机初始化。预训练模型已经学到通用边缘、纹理和形状；我用五类桌面物体继续微调，这就是迁移学习。`model.train()` 由 Ultralytics 封装了前向传播、损失计算、反向传播和优化器更新。100 epochs 是最多训练轮数，patience=20 允许早停，所以不一定跑满。训练中看 Validation 选 best.pt，最后一轮的 last.pt 不一定最好。

完成训练后，固定 best.pt 再做 Test。Precision 看预测正确性，Recall 看真实物体有没有被找回；mAP50 使用 IoU 0.5，mAP50-95 更严格。Test 不参与调参，否则会造成评价偏差。除了总体指标，我也检查五类的逐类指标，尤其关注 keyboard。

最后演示推理。USB 摄像头是 index 1，Windows 下用 DSHOW。程序只打开摄像头一次，然后持续 `cap.read()`，把每帧直接传给 `model.predict(frame)`；不会让 YOLO 再抢占摄像头。推理输出框、类别和置信度，NMS 会去掉同一物体的重复框。前 30 帧 warm-up 不计入，随后用 300 帧平均 Pipeline FPS，因为它包含采集、推理、后处理和绘制，最接近真实应用。

总结：验证代码保证数据可信，训练代码使用迁移学习并选择 best.pt，推理代码把模型接到真实 USB 摄像头，同时保存视频、CSV 和 JSONL，形成可解释、可复现的完整流程。

常见追问速答：预训练可减少小数据集训练难度；Test 不能参与训练；Session 不能跨 split 以防数据泄露；Confidence 低会更多误检；YOLO11n 较小更适合实时部署；Inference 不会更新参数。
