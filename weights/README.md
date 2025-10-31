# OrangeFruitNet Project – Pretrained Weights Documentation

This folder contains **verified pretrained weights** used for all experiments reported in the manuscript:

> **"OrangeFruitNet: A Dataset and On-Device Real-Time YOLO-Based Framework for Orange Fruit Detection and Yield Estimation in Orchard Environments"**

All weights were downloaded from **official model repositories** (Ultralytics, AlexeyAB Darknet, and Detectron2 Model Zoo).  
Each file was verified using SHA-256 checksums listed in [`weights/checksums.txt`](./checksums.txt).

---

## 🧠 Model Weights Overview

| Model Name | Framework | Source | Dataset | File | License | Usage in Manuscript |
|-------------|------------|---------|----------|--------|----------|----------------------|
| **YOLOv8-n** | Ultralytics | [Hugging Face – Ultralytics/YOLOv8](https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt) | COCO | `yolov8n_coco.pt` | GPLv3 (Ultralytics) | Fine-tuning and evaluation (YOLOv8-n baseline and main model) |
| **YOLOv5-s** | Ultralytics | [GitHub – YOLOv5 Release v6.2](https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt) | COCO | `yolov5s_coco.pt` | GPLv3 (Ultralytics) | Fine-tuning and real-time benchmarking (YOLOv5-s lightweight variant) |
| **YOLOv4-Darknet53** | AlexeyAB Darknet | [GitHub – AlexeyAB/darknet](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) | COCO | `yolov4.weights` | Public domain / GPLv3 | Used as legacy YOLO baseline (comparative analysis) |
| **Faster R-CNN (R50-FPN)** | Detectron2 | [Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) | COCO | `faster_rcnn_r50_fpn_3x_coco.pth` | Apache 2.0 | Benchmark for comparison with YOLO-based detectors |
| **Mask R-CNN (R50-FPN)** | Detectron2 | [Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) | COCO | `mask_rcnn_r50_fpn_3x_coco.pth` | Apache 2.0 | Instance segmentation and dense canopy comparison |

---

## 🔒 Checksum Verification

After downloading all weights (via `get_pretrained.sh`), verify their integrity using:
```bash
sha256sum -c weights/checksums.txt
