## 🟠 OrangeFruitNet – Real-Time Deployment Guide

### 📘 Overview
This guide documents the inference and deployment procedures for the **OrangeFruitNet** framework on embedded devices such as **NVIDIA Jetson Nano** and **Raspberry Pi 4**, reproducing the results reported in  
📄 *OrangeFruitNet: YOLO-Based Orange Fruit Detection and Yield Estimation in Orchard Environments* (IJIES 2025).

All models were exported to **TorchScript** for device-friendly execution with reduced memory footprint and improved portability.

---

### ⚙️ 1  Environment Setup

#### 🧩 Dependencies
Install Python ≥ 3.8 and the required libraries:

```bash
pip install ultralytics torch torchvision opencv-python numpy
```

> 💡 On Jetson Nano, use NVIDIA’s JetPack >= 5.0 which includes CUDA-enabled PyTorch.

#### 📂 Directory Structure
```
OrangeFruitNet-Project/
│
├── weights/
│   ├── yolov8n.pt
│   ├── yolov5s.pt
│   ├── yolov4.weights
│   ├── faster_rcnn.pth
│   ├── mask_rcnn.pth
│   └── checksums.txt
│
├── deployment/
│   ├── export_torchscript.py
│   ├── inference_device.py
│   └── README_deployment.md
```

---

### 🧠 2  Model Export (TorchScript)

Export trained YOLO models to TorchScript for embedded inference:

```bash
python deployment/export_torchscript.py --weights weights/yolov8n.pt --output weights/yolov8n.torchscript
```

This produces a lightweight `.torchscript` file usable on both Jetson Nano and Raspberry Pi 4.

---

### 🎥 3  Real-Time Inference

Run real-time detection using the integrated inference script:

```bash
python deployment/inference_device.py   --weights weights/yolov8n.torchscript   --source 0   --device cuda:0   --imgsz 640   --conf 0.25
```

**Parameters**
| Argument | Description | Default |
|-----------|--------------|----------|
| `--weights` | Path to model weights (.pt or .torchscript) | Required |
| `--source` | Camera index (0) or video/image path | 0 |
| `--device` | `cpu`, `cuda`, or `cuda:0` | cpu |
| `--imgsz` | Inference image size | 640 |
| `--conf` | Detection confidence threshold | 0.25 |

Press `q` to stop live display.

---

### 📊 4  Performance Evaluation (Table 10 Reproduction)

To replicate Table 10 (“Real-time embedded performance”), use the inference script on the device and record average FPS as reported below (these are the values reported in the manuscript):

| Device | Model | FPS (avg) | Power (W) | Notes |
|---------|--------|-----------|-----------|-------|
| Jetson Nano | YOLOv4-Darknet53 | 6 | 10 | Baseline YOLOv4 on Jetson Nano |
| Jetson Nano | YOLOv5-s | 8 | 10 | Lightweight YOLOv5-s; real-time on Nano |
| Jetson Nano | YOLOv8-n | 7 | 10 | YOLOv8-n accuracy vs speed tradeoff |
| Raspberry Pi 4 | YOLOv4-Darknet53 | 3 | 8 | CPU-bound |
| Raspberry Pi 4 | YOLOv5-s | 2.5 | 8 | Lightweight variant on Pi |
| Raspberry Pi 4 | YOLOv8-n | 2 | 8 | CPU-bound, lower FPS |

> Notes:
> - Power values are approximate measured draw under typical inference (see manuscript Table 10 for exact measurement setup).
> - For exact reproduction, run `python deployment/inference_device.py --weights <path> --source <camera_or_video> --device cpu` (on Pi use `--device cpu`).

---

### 🧾 5  Troubleshooting

| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| *TorchScript load error* | PyTorch version mismatch | Re-export with local PyTorch version |
| *Low FPS on Pi 4* | Missing OpenBLAS / NEON | Enable via `sudo apt install libopenblas-dev` |
| *No display window* | Running headless | Use `--source video.mp4` and log FPS only |

---

### 🔒 6  Reproducibility & Citation

All deployment configurations, pretrained weights, and scripts are provided under  
📂 `/deployment` and 📂 `/weights` in the public repository:  
👉 https://github.com/OrangeFruitNet-Project/IJIES-2025

If you use this deployment setup, please cite:

> **M. Khedkar et al.**, *“OrangeFruitNet: YOLO-Based Orange Fruit Detection and Yield Estimation in Orchard Environments,.....
