# 🟧 OrangeFruitNet: A Dataset and On-Device Real-Time YOLO-Based Framework for Orange Fruit Detection and Yield Estimation

## 📄 Overview
This repository implements the complete experimental framework described in the manuscript:  
**“OrangeFruitNet: A Dataset and On-Device Real-Time YOLO-Based Framework for Orange Fruit Detection and Yield Estimation in Orchard Environments.”**

The project integrates **dataset preparation**, **YOLO-based training**, **cross-dataset benchmarking**, **yield estimation**, and **embedded deployment** on **Jetson Nano** and **Raspberry Pi 4**.

## 📘 Directory Structure
```
OrangeFruitNet-Framework/
├── dataset/
├── augmentation/
├── weights/
├── configs/
├── training/
├── evaluation/
├── deployment/
├── yield_estimation/
├── utils/
├── scripts/
├── outputs/
└── README.md
```

## 🚀 How to Run Experiments
### 1️⃣ Setup Environment
```bash
conda create -n orangefruit python=3.10 -y
conda activate orangefruit
pip install -r requirements.txt
```
### 2️⃣ Download Pretrained Weights
```bash
bash scripts/get_pretrained.sh
```
### 3️⃣ Train All Models
```bash
bash training/train_all.sh
```
### 4️⃣ Evaluate and Generate Tables
```bash
python evaluation/eval_cross_dataset.py
python evaluation/produce_table.py
```
### 5️⃣ Yield Estimation and Regression Validation
```bash
python yield_estimation/yield_estimation.py
```
### 6️⃣ Export Models for Embedded Devices
```bash
python deployment/export_torchscript.py
```
### 7️⃣ On-device Inference
```bash
python deployment/inference_device.py --device jetson
```

## 🧩 Reproducibility
- Random seed fixed (`42`)
- Augmentations: rotation +15°, horizontal flip (p=0.5), brightness ±15%
- Loss weighting & hyperparameters match Section 3.3 of manuscript
- Splits: 80:10:10 (train:val:test)

## 📊 Outputs
Results saved under `outputs/`:
```
outputs/
├── logs/
├── plots/
├── tables/
├── models/
└── reports/
```

## 🧠 Citation
```
M. Khedkar, V. Sambhe, and V. Dhore, 
"OrangeFruitNet: A Dataset and On-Device Real-Time YOLO-Based Framework 
for Orange Fruit Detection and Yield Estimation in Orchard Environments," 
........
```

## 🪪 License
Released under **CC BY-NC 4.0 International License**.
