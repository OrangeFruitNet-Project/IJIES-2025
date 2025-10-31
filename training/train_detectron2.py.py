#!/usr/bin/env python3
"""
train_detectron2.py
training script for Faster R-CNN and Mask R-CNN
as described in the OrangeFruitNet manuscript.
Implements unified reproducible training pipeline with Adam optimizer,
cosine LR scheduler, fixed seeds, and augmentations consistent with Section 3.3 and 3.3.

# Train Faster R-CNN
python train_detectron2.py --model faster_rcnn --dataset ../dataset --output ./runs/faster_rcnn_orangefruitnet

# Train Mask R-CNN
python train_detectron2.py --model mask_rcnn --dataset ../dataset --output ./runs/mask_rcnn_orangefruitnet

"""

import os
import argparse
import random
import numpy as np
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.utils.logger import setup_logger

setup_logger()


# -------------------------------------------------------------------------
# 1. Deterministic seed setup (for reproducibility)
# -------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------------------
# 2. Custom dataset loader (uses COCO format JSON files)
# -------------------------------------------------------------------------
def register_orangefruitnet(dataset_dir):
    from detectron2.data.datasets import register_coco_instances
    train_json = os.path.join(dataset_dir, "annotations/train.json")
    val_json = os.path.join(dataset_dir, "annotations/val.json")
    img_dir = os.path.join(dataset_dir, "images")

    register_coco_instances("orangefruitnet_train", {}, train_json, img_dir)
    register_coco_instances("orangefruitnet_val", {}, val_json, img_dir)


# -------------------------------------------------------------------------
# 3. Custom mapper for augmentations (Section 3.2)
# -------------------------------------------------------------------------
def custom_mapper(dataset_dict):
    dataset_dict = dataset_dict.copy()
    image = utils.read_image(dataset_dict["file_name"], format="RGB")
    aug = T.AugmentationList([
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomRotation(angle=[-15, 15]),
        T.RandomBrightness(0.85, 1.15)
    ])
    image, transforms = T.apply_augmentations(aug, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1)).contiguous()
    annos = [utils.transform_instance_annotations(obj, transforms, image.shape[1:]) 
             for obj in dataset_dict.pop("annotations")]
    dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[1:])
    return dataset_dict


class OrangeTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


# -------------------------------------------------------------------------
# 4. Main training routine
# -------------------------------------------------------------------------
def main(args):
    set_seed(args.seed)
    dataset_dir = os.path.abspath(args.dataset)
    register_orangefruitnet(dataset_dir)

    cfg = get_cfg()

    # Select model architecture
    if args.model.lower() == "mask_rcnn":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    cfg.DATASETS.TRAIN = ("orangefruitnet_train",)
    cfg.DATASETS.TEST = ("orangefruitnet_val",)
    cfg.DATALOADER.NUM_WORKERS = args.workers

    # Optimization parameters (per manuscript Section 3.3)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.OPTIMIZER = "Adam"
    cfg.SOLVER.BETAS = (args.beta1, args.beta2)
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.INPUT.MIN_SIZE_TRAIN = (args.imgsz,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.imgsz
    cfg.INPUT.MIN_SIZE_TEST = args.imgsz
    cfg.INPUT.MAX_SIZE_TEST = args.imgsz
    cfg.INPUT.FORMAT = "RGB"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    cfg.SOLVER.SEED = args.seed
    cfg.OUTPUT_DIR = os.path.abspath(args.output)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Pretrained weights (COCO)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" if args.model.lower() == "mask_rcnn"
        else "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )

    print("============================================")
    print(f"Training {args.model.upper()} on OrangeFruitNet")
    print(f"Batch Size: {args.batch_size}, LR: {args.lr}, Epochs ≈ {args.max_iter / (len(DatasetCatalog.get('orangefruitnet_train')) / args.batch_size):.1f}")
    print("============================================")

    trainer = OrangeTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"✅ Training complete. Model checkpoints saved to: {cfg.OUTPUT_DIR}")


# -------------------------------------------------------------------------
# 5. CLI arguments
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster/Mask R-CNN on OrangeFruitNet (Detectron2)")
    parser.add_argument("--model", type=str, default="faster_rcnn", choices=["faster_rcnn", "mask_rcnn"])
    parser.add_argument("--dataset", type=str, default="../dataset")
    parser.add_argument("--output", type=str, default="./runs/detectron2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.937)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--max_iter", type=int, default=20000)
    parser.add_argument("--checkpoint_period", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
