#!/usr/bin/env python3
"""训练并评估最终五分类 YOLO11n/YOLO11s 实验。

本脚本演示 Transfer Learning（迁移学习）：从 COCO 预训练权重开始，
再对 mouse、keyboard、laptop、cup、headphones 进行 Fine-tuning（微调）。
"""
from __future__ import annotations

# ============================================================
# 1. 依赖与统一路径：Path 管理文件；CSV/Markdown 保存实验记录；
#    Ultralytics 将前向传播、loss、反向传播和 optimizer.step() 封装进 train()。
# ============================================================
import csv
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("PYTHONUTF8", "1")

import torch
import ultralytics
from ultralytics import YOLO

# ============================================================
# 2. 固定数据、输出与类别配置。dataset.yaml 定义 Train/Val/Test 路径。
# ============================================================
DATA = ROOT / "data/yolo_dataset_final/dataset.yaml"
DATA_ROOT = DATA.parent
RUNS = ROOT / "runs/detect"
RESULTS = ROOT / "results/final_model_comparison"
NAMES = ("mouse", "keyboard", "laptop", "cup", "headphones")
FIX_NOTE = "正式训练前进行数据完整性校验，发现 1 个贴近图像边缘的 bbox 因归一化小数舍入产生 0.0005 越界，经原图视觉确认后裁剪至合法图像边界并重新归一化。该修复发生在任何模型训练和测试之前。"


def unique_name(base: str) -> str:
    if not (RUNS / base).exists(): return base
    i = 2
    while (RUNS / f"{base}_{i}").exists(): i += 1
    return f"{base}_{i}"

#读取中的所有 YOLO .txt 文件，然后统计每一类有多少个真实目标。
def class_instances(split: str) -> Counter:
    counts = Counter()
    for path in (DATA_ROOT / "labels" / split).glob("*.txt"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip(): counts[int(line.split()[0])] += 1
    return counts

#评价模块，负责提取整体指标：
#Precision
#Recall
#mAP50
#mAP50-95
def aggregate(metrics):
    b=metrics.box
    return {"precision":float(b.mp),"recall":float(b.mr),"mAP50":float(b.map50),"mAP50_95":float(b.map)}

#统计每一类的指标
def per_class(split: str, metrics):
    b=metrics.box; instances=class_instances(split); index={int(c):i for i,c in enumerate(b.ap_class_index)}
    rows=[]
    for cls,name in enumerate(NAMES):
        i=index.get(cls)
        rows.append({"split":split,"class_id":cls,"class_name":name,"precision":"N/A" if i is None else float(b.p[i]),"recall":"N/A" if i is None else float(b.r[i]),"AP50":"N/A" if i is None else float(b.ap50[i]),"AP50_95":"N/A" if i is None else float(b.maps[cls]),"instances":instances[cls]})
    return rows

#计算预测框和真实框的重叠程度
def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.0

#错误分析
def error_analysis(model, model_key: str):
    rows=[]; source=DATA_ROOT/"images/test"; label_dir=DATA_ROOT/"labels/test"
    for result in model.predict(source=str(source),imgsz=640,conf=.25,device=0,verbose=False,stream=True):
        image=Path(result.path); h,w=result.orig_shape; gt=[]
        for line in (label_dir/f"{image.stem}.txt").read_text(encoding="utf-8-sig").splitlines():
            if not line.strip(): continue
            c,x,y,bw,bh=map(float,line.split()); gt.append((int(c),((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)))
        pred=[(int(c),tuple(map(float,box)),float(conf)) for box,c,conf in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        used_g,used_p=set(),set(); candidates=sorted(((iou(g[1],p[1]),gi,pi) for gi,g in enumerate(gt) for pi,p in enumerate(pred) if g[0]==p[0]),reverse=True)
        for overlap,gi,pi in candidates:
            if overlap>=.5 and gi not in used_g and pi not in used_p:
                used_g.add(gi); used_p.add(pi)
                if pred[pi][2]<.5: rows.append([image.name,NAMES[gt[gi][0]],NAMES[pred[pi][0]],pred[pi][2],"low_confidence_correct",f"IoU={overlap:.3f}"])
        for gi,g in enumerate(gt):
            if gi not in used_g:
                same=max((iou(g[1],p[1]) for p in pred if p[0]==g[0]),default=0)
                rows.append([image.name,NAMES[g[0]],"","","localization_error" if same>=.1 else "false_negative",f"best same-class IoU={same:.3f}"])
        for pi,p in enumerate(pred):
            if pi in used_p: continue
            overlap,cls=max(((iou(p[1],g[1]),g[0]) for g in gt),default=(0,-1))
            kind="class_confusion" if overlap>=.5 and cls!=p[0] else "false_positive"
            rows.append([image.name,"" if cls<0 else NAMES[cls],NAMES[p[0]],p[2],kind,"no ground truth" if cls<0 else f"IoU={overlap:.3f}"])
    out=RESULTS/f"error_cases_{model_key}.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["image","ground_truth","prediction","confidence","error_type","notes"]); w.writerows(rows)
    sample=sorted(source.glob("*"))[:12]
    model.predict(source=[str(p) for p in sample],imgsz=640,conf=.25,device=0,save=True,project=str(RESULTS/"predictions"),name=model_key,exist_ok=False,verbose=False)
    return Counter(r[4] for r in rows)


def driver():
    try: return subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True,encoding="utf-8").strip().splitlines()[0]
    except Exception: return "unknown"


def train_one(model_key: str):
    """训练一个 n/s 模型并以 best.pt 做 Val/Test 评估。

    epochs=100 是最多 100 个 Epoch（完整遍历训练集），patience=20 是
    Early Stopping（早停）容忍轮数，因此训练不一定跑满 100 轮。best.pt 是
    Validation 指标最佳权重；last.pt 仅是最后一轮权重。
    """
    weight=f"{model_key}.pt"; run_name=unique_name(f"{model_key}_final")#加载预训练权重，Ultralytics 封装好的训练接口
    started=datetime.now().astimezone(); clock=time.perf_counter(); model=YOLO(weight,task="detect")
    os.chdir(DATA_ROOT)
    # model.train(): image -> resize/augmentation -> forward -> 与 Ground Truth 比较
    # -> box/cls/dfl loss -> backward -> optimizer update。Batch=16，device=0 为 GPU。
    trained=model.train(data=str(DATA),project=str(RUNS),name=run_name,epochs=100,imgsz=640,batch=16,device=0,patience=20,seed=42,deterministic=True,pretrained=True,optimizer="auto",plots=True,save=True,workers=4,cache=False)
    training_seconds=time.perf_counter()-clock; run_dir=Path(trained.save_dir); best=run_dir/"weights/best.pt"
    if not best.exists(): raise RuntimeError(f"{model_key} missing best.pt")
    rows=list(csv.DictReader((run_dir/"results.csv").open(encoding="utf-8-sig")))
    best_epoch=max(range(len(rows)),key=lambda i:.1*float(rows[i]["metrics/mAP50(B)"])+.9*float(rows[i]["metrics/mAP50-95(B)"]))+1
    fixed=YOLO(str(best),task="detect")
    # Validation 用于训练期间选择/早停；最终 Test 只在固定 best.pt 后评价，不能调参。
    val=fixed.val(data=str(DATA),split="val",imgsz=640,batch=16,device=0,workers=4,plots=True,project=str(run_dir),name="val_eval")
    test=fixed.val(data=str(DATA),split="test",imgsz=640,batch=16,device=0,workers=4,plots=True,project=str(run_dir),name="test_eval")
    val_a,test_a=aggregate(val),aggregate(test); pc=per_class("val",val)+per_class("test",test)
    with (RESULTS/f"{model_key}_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=("split","precision","recall","mAP50","mAP50_95")); w.writeheader(); w.writerow({"split":"val",**val_a}); w.writerow({"split":"test",**test_a})
    with (RESULTS/f"{model_key}_per_class_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=pc[0].keys()); w.writeheader(); w.writerows(pc)
    errors=error_analysis(fixed,model_key)
    params=sum(p.numel() for p in fixed.model.parameters()); size=best.stat().st_size/1024/1024; ended=datetime.now().astimezone()
    env={"date_started":started.isoformat(),"date_ended":ended.isoformat(),"Python":platform.python_version(),"PyTorch":torch.__version__,"CUDA":torch.version.cuda,"Ultralytics":ultralytics.__version__,"GPU":torch.cuda.get_device_name(0),"driver":driver(),"pretrained_weights":weight,"imgsz":640,"batch":16,"epochs_requested":100,"epochs_completed":len(rows),"best_epoch":best_epoch,"early_stopping":len(rows)<100,"seed":42,"deterministic":True,"training_time_seconds":f"{training_seconds:.3f}","run_dir":str(run_dir.resolve()),"best.pt":str(best.resolve()),"best_size_mb":f"{size:.3f}","parameters":params}
    (RESULTS/f"{model_key}_environment.txt").write_text("".join(f"{k}: {v}\n" for k,v in env.items()),encoding="utf-8")
    test_pc=[r for r in pc if r["split"]=="test"]
    table="\n".join(f"| {r['class_name']} | {r['precision']} | {r['recall']} | {r['AP50']} | {r['AP50_95']} | {r['instances']} |" for r in test_pc)
    summary=f"# {model_key} final fine-tuning\n\n{FIX_NOTE}\n\n- Pretrained weights: `{weight}`\n- Run: `{run_dir}`\n- Best epoch: {best_epoch}; completed: {len(rows)}/100; training seconds: {training_seconds:.3f}\n- Val: P={val_a['precision']:.6f}, R={val_a['recall']:.6f}, mAP50={val_a['mAP50']:.6f}, mAP50-95={val_a['mAP50_95']:.6f}\n- Test: P={test_a['precision']:.6f}, R={test_a['recall']:.6f}, mAP50={test_a['mAP50']:.6f}, mAP50-95={test_a['mAP50_95']:.6f}\n\n| Class | P | R | AP50 | AP50-95 | Instances |\n|---|---:|---:|---:|---:|---:|\n{table}\n\nError counts at conf=0.25: {dict(errors)}. Test was used once after fixing best.pt and did not participate in selection or tuning.\n"
    (RESULTS/f"{model_key}_summary.md").write_text(summary,encoding="utf-8")
    return {"model":model_key,"parameters":params,"model_size_mb":size,"precision":test_a["precision"],"recall":test_a["recall"],"mAP50":test_a["mAP50"],"mAP50_95":test_a["mAP50_95"],"best_epoch":best_epoch,"epochs_completed":len(rows),"training_time":training_seconds,"best":best.resolve(),"val":val_a,"test_pc":test_pc,"errors":errors}


def comparison(a,b):
    """比较 nano(n，较快较小) 与 small(s，容量更大) 的整体和逐类指标。"""
    fields=("model","parameters","model_size_mb","precision","recall","mAP50","mAP50_95","best_epoch","training_time")
    with (RESULTS/"model_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for x in (a,b): w.writerow({k:x[k] for k in fields})
    ap={x["model"]:{r["class_name"]:r for r in x["test_pc"]} for x in (a,b)}
    lines=["# Final model comparison","",FIX_NOTE,"","## Aggregate test metrics","","| Model | P | R | mAP50 | mAP50-95 | Best epoch | Training seconds |","|---|---:|---:|---:|---:|---:|---:|"]
    for x in (a,b): lines.append(f"| {x['model']} | {x['precision']:.6f} | {x['recall']:.6f} | {x['mAP50']:.6f} | {x['mAP50_95']:.6f} | {x['best_epoch']} | {x['training_time']:.3f} |")
    lines += ["","## Per-class test comparison","","| Class | YOLO11n AP50 | YOLO11s AP50 | Difference | YOLO11n Recall | YOLO11s Recall | Recall difference |","|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        n,s=ap["yolo11n"][name],ap["yolo11s"][name]; lines.append(f"| {name} | {float(n['AP50']):.6f} | {float(s['AP50']):.6f} | {float(s['AP50'])-float(n['AP50']):+.6f} | {float(n['recall']):.6f} | {float(s['recall']):.6f} | {float(s['recall'])-float(n['recall']):+.6f} |")
    winner=max((a,b),key=lambda x:x["mAP50_95"])["model"]
    lines += ["","## Interpretation","",f"Accuracy winner: **{winner}**. Deployment candidate remains **yolo11n** until measured Jetson/TensorRT FPS and accuracy are evaluated; both n and s should be benchmarked when the accuracy gap is material.","","Test results were not used for model selection, hyperparameter adjustment, confidence tuning, or dataset changes."]
    (RESULTS/"model_comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable; refusing full training")
    RESULTS.mkdir(parents=True,exist_ok=True)
    n=train_one("yolo11n")
    s=train_one("yolo11s")
    comparison(n,s)
    print(f"FINAL_COMPLETE n_best={n['best']} s_best={s['best']}")


if __name__ == "__main__": main()
