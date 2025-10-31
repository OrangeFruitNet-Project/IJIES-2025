#!/bin/bash
# ============================================================
# OrangeFruitNet - Unified Training Pipeline (as per manuscript)
# ============================================================
# Trains YOLOv4, YOLOv5, YOLOv8, Faster R-CNN, and Mask R-CNN
# under the same experimental settings described in Section 3.3
# ============================================================

set -e  # stop on first error
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

ROOT_DIR=$(pwd)
DATASET_DIR="${ROOT_DIR}/dataset"
WEIGHTS_DIR="${ROOT_DIR}/weights"
RUNS_DIR="${ROOT_DIR}/runs"

mkdir -p "$RUNS_DIR"

echo "============================================================"
echo " 1️⃣  Training YOLOv8-n on OrangeFruitNet"
echo "============================================================"
yolo detect train \
  data=${DATASET_DIR}/orangefruitnet.yaml \
  model=${WEIGHTS_DIR}/yolov8n.pt \
  imgsz=640 \
  batch=16 \
  epochs=200 \
  optimizer=Adam \
  lr0=0.001 \
  beta1=0.937 \
  beta2=0.999 \
  workers=4 \
  seed=42 \
  project=${RUNS_DIR}/yolov8n_orangefruitnet \
  name=exp_yolov8n \
  save_period=10 \
  patience=25 \
  augment=True

echo "============================================================"
echo " 2️⃣  Training YOLOv5-s on OrangeFruitNet"
echo "============================================================"
python train.py \
  --data ${DATASET_DIR}/orangefruitnet.yaml \
  --weights ${WEIGHTS_DIR}/yolov5s.pt \
  --cfg ${ROOT_DIR}/configs/yolov5_config.yaml \
  --img 640 \
  --batch 16 \
  --epochs 200 \
  --optimizer Adam \
  --lr 0.001 \
  --momentum 0.937 \
  --workers 4 \
  --project ${RUNS_DIR}/yolov5s_orangefruitnet \
  --name exp_yolov5s \
  --patience 25 \
  --seed 42

echo "============================================================"
echo " 3️⃣  Training YOLOv4 on OrangeFruitNet (Darknet)"
echo "============================================================"
./darknet detector train \
  ${DATASET_DIR}/orangefruitnet.data \
  ${ROOT_DIR}/configs/yolov4_config.cfg \
  ${WEIGHTS_DIR}/yolov4.weights \
  -dont_show \
  -map \
  -clear \
  -gpus 0

echo "============================================================"
echo " 4️⃣  Training Faster R-CNN (Detectron2)"
echo "============================================================"
python train_detectron2.py \
  --config-file ${ROOT_DIR}/configs/faster_rcnn_orangefruitnet.yaml \
  --num-gpus 1 \
  OUTPUT_DIR ${RUNS_DIR}/faster_rcnn_orangefruitnet

echo "============================================================"
echo " 5️⃣  Training Mask R-CNN (Detectron2)"
echo "============================================================"
python train_detectron2.py \
  --config-file ${ROOT_DIR}/configs/mask_rcnn_orangefruitnet.yaml \
  --num-gpus 1 \
  OUTPUT_DIR ${RUNS_DIR}/mask_rcnn_orangefruitnet

echo "============================================================"
echo "✅ All models trained successfully!"
echo "Logs and checkpoints saved under: ${RUNS_DIR}/"
echo "============================================================"
