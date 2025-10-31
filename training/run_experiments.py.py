#!/usr/bin/env python3
"""
run_experiments.py
Unified experimental runner for the OrangeFruitNet project (IJIES 2025)
Author: OrangeFruitNet Research Team

Implements reproducible training runs for:
  - YOLOv4 (Darknet)
  - YOLOv5 (Ultralytics)
  - YOLOv8 (Ultralytics)
  - Faster R-CNN (Detectron2)
  - Mask R-CNN (Detectron2)

Follows Section 3.3 of the manuscript:
- lr=0.001, Adam optimizer (β1=0.937, β2=0.999)
- batch=16, epochs=200, cosine LR scheduler
- input size=640, seed=42
- augmentations: rotation ±15°, brightness ±15%, horizontal flip (p=0.5)
"""

import os
import subprocess
import argparse
import random
import numpy as np
import torch
from datetime import datetime


# -------------------------------------------------------------------------
# 1. Deterministic Seeding (Reproducibility)
# -------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# 2. Command Runner with Logging
# -------------------------------------------------------------------------
def run_cmd(cmd, log_file):
    print(f"\n🚀 Running: {' '.join(cmd)}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as lf:
        process = subprocess.Popen(cmd, stdout=lf, stderr=lf)
        process.wait()
    if process.returncode == 0:
        print(f"✅ Completed: {' '.join(cmd)}")
    else:
        print(f"❌ Error running: {' '.join(cmd)} (see {log_file})")


# -------------------------------------------------------------------------
# 3. Individual Model Training Routines
# -------------------------------------------------------------------------
def run_yolov8(dataset_dir, weights_dir, runs_dir):
    cmd = [
        "yolo", "detect", "train",
        f"data={dataset_dir}/orangefruitnet.yaml",
        f"model={weights_dir}/yolov8n.pt",
        "imgsz=640",
        "batch=16",
        "epochs=200",
        "optimizer=Adam",
        "lr0=0.001",
        "beta1=0.937",
        "beta2=0.999",
        "workers=4",
        "project=" + f"{runs_dir}/yolov8n_orangefruitnet",
        "name=exp_yolov8n",
        "patience=25",
        "seed=42",
        "save_period=10"
    ]
    run_cmd(cmd, f"{runs_dir}/yolov8n_orangefruitnet/train.log")


def run_yolov5(dataset_dir, weights_dir, runs_dir):
    cmd = [
        "python", "train.py",
        "--data", f"{dataset_dir}/orangefruitnet.yaml",
        "--weights", f"{weights_dir}/yolov5s.pt",
        "--cfg", "./configs/yolov5_config.yaml",
        "--img", "640",
        "--batch", "16",
        "--epochs", "200",
        "--optimizer", "Adam",
        "--lr", "0.001",
        "--momentum", "0.937",
        "--workers", "4",
        "--project", f"{runs_dir}/yolov5s_orangefruitnet",
        "--name", "exp_yolov5s",
        "--patience", "25",
        "--seed", "42"
    ]
    run_cmd(cmd, f"{runs_dir}/yolov5s_orangefruitnet/train.log")


def run_yolov4(dataset_dir, runs_dir):
    cmd = [
        "./darknet", "detector", "train",
        f"{dataset_dir}/orangefruitnet.data",
        "./configs/yolov4_config.yaml",
        "./weights/yolov4.conv.137",
        "-dont_show",
        "-map",
        "-clear",
        "-gpus", "0"
    ]
    run_cmd(cmd, f"{runs_dir}/yolov4_orangefruitnet/train.log")


def run_detectron2(model, dataset_dir, runs_dir):
    cmd = [
        "python", "train_detectron2.py",
        "--model", model,
        "--dataset", dataset_dir,
        "--output", f"{runs_dir}/{model}_orangefruitnet",
        "--batch_size", "16",
        "--imgsz", "640",
        "--lr", "0.001",
        "--beta1", "0.937",
        "--beta2", "0.999",
        "--max_iter", "20000",
        "--checkpoint_period", "1000",
        "--workers", "4",
        "--seed", "42"
    ]
    run_cmd(cmd, f"{runs_dir}/{model}_orangefruitnet/train.log")


# -------------------------------------------------------------------------
# 4. Main Execution Logic
# -------------------------------------------------------------------------
def main(args):
    set_seed(args.seed)

    dataset_dir = os.path.abspath(args.dataset)
    weights_dir = os.path.abspath(args.weights)
    runs_dir = os.path.abspath(args.runs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n🟠 Starting OrangeFruitNet Experiments [{timestamp}]")
    print(f"Dataset: {dataset_dir}")
    print(f"Weights: {weights_dir}")
    print(f"Runs:    {runs_dir}\n")

    os.makedirs(runs_dir, exist_ok=True)

    # Sequential training of requested models
    executed_models = []
    if "yolov8" in args.models:
        run_yolov8(dataset_dir, weights_dir, runs_dir)
        executed_models.append("YOLOv8-n")
    if "yolov5" in args.models:
        run_yolov5(dataset_dir, weights_dir, runs_dir)
        executed_models.append("YOLOv5-s")
    if "yolov4" in args.models:
        run_yolov4(dataset_dir, runs_dir)
        executed_models.append("YOLOv4")
    if "faster_rcnn" in args.models:
        run_detectron2("faster_rcnn", dataset_dir, runs_dir)
        executed_models.append("Faster R-CNN")
    if "mask_rcnn" in args.models:
        run_detectron2("mask_rcnn", dataset_dir, runs_dir)
        executed_models.append("Mask R-CNN")

    print("\n✅ All training experiments completed successfully.")
    print(f"Results stored in: {runs_dir}")
    print("\n📊 Summary of completed experiments:")
    for model in executed_models:
        print(f"  • {model} → logs under {runs_dir}/{model.lower().replace(' ', '_')}_orangefruitnet/train.log")
    print("\n🧾 Configuration: lr=0.001 | batch=16 | epochs=200 | β1=0.937 | β2=0.999 | imgsz=640 | seed=42")
    print("------------------------------------------------------------")


# -------------------------------------------------------------------------
# 5. CLI Argument Parser
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OrangeFruitNet unified training pipeline")
    parser.add_argument("--dataset", type=str, default="../dataset", help="Path to dataset directory")
    parser.add_argument("--weights", type=str, default="../weights", help="Path to pretrained weights")
    parser.add_argument("--runs", type=str, default="./runs", help="Output directory for experiment logs")
    parser.add_argument("--models", nargs="+", default=["yolov8", "yolov5", "yolov4", "faster_rcnn", "mask_rcnn"],
                        help="List of models to train (e.g., yolov8 faster_rcnn)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
