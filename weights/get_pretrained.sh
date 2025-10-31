#!/bin/bash
# Script to download pretrained weights for OrangeFruitNet experiments

mkdir -p weights

echo "Downloading YOLOv8-n pretrained weights..."
wget -O weights/yolov8n_coco.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt

echo "Downloading YOLOv5-s pretrained weights..."
wget -O weights/yolov5s_coco.pt https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt

echo "Downloading YOLOv4-Darknet53 pretrained weights..."
wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

echo "Note: Detectron2 model zoo links do not provide direct .pth download — will fetch automatically using model_zoo in code."
echo "You can optionally download Faster R-CNN via:"
echo "    weights/faster_rcnn_r50_fpn_3x_coco.pth"
echo "    from model_zoo: detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137257794"
echo "And Mask R-CNN via:"
echo "    weights/mask_rcnn_r50_fpn_3x_coco.pth"
echo "    from model_zoo: detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/162397801"

echo "All done."
