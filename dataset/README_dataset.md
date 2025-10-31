# 🟠 OrangeFruitNet Dataset

### Overview
**OrangeFruitNet** is a dedicated annotated dataset developed for orange fruit detection and yield estimation in real-world orchard conditions.  
The dataset captures key field challenges such as **occlusion, clustering, illumination variability,** and **leaf-fruit confusion**, providing a benchmark for reproducible deep learning research in precision agriculture.

### Dataset Structure
```
orangefruitnet/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── orangefruitnet_manifest.csv
```

### Manifest File (`orangefruitnet_manifest.csv`)
The manifest provides metadata for **1,750 original annotated images** (expanded to 7,000 via augmentations in experiments).  
Each row corresponds to a single image and includes:

| Column | Description |
|---------|-------------|
| `split` | Dataset split (train / valid / test) |
| `image_id` | File name of the image |
| `image_path` | Relative path to the image file |
| `width`, `height` | Image dimensions in pixels |
| `label_path` | Corresponding YOLO-format annotation file |
| `num_objects` | Number of annotated fruits in the image |
| `class_ids` | Encoded class IDs (`0: flower`, `1: green orange`, `2: orange`) |
| `annotation_format` | “YOLO” (normalized x_center, y_center, width, height) |

Example entry:
```csv
split,image_id,image_path,width,height,label_path,num_objects,class_ids,annotation_format
train,IMG_01234.jpg,train/images/IMG_01234.jpg,640,480,train/labels/IMG_01234.txt,6,[2],YOLO
```

### Dataset Access
The full dataset (images and annotations) is available under **Creative Commons (CC BY 4.0)** via Roboflow:

🔗 [Orange Fruit Counting Dataset – Roboflow Universe](https://universe.roboflow.com/fruit-estimations/orange-fruit-counting/dataset/3)

This manifest ensures reproducibility even without direct image files, allowing reviewers to validate:
- Annotation structure and labeling format  
- Split ratios and data diversity  
- Consistency with model training configurations in the manuscript

### Citation
If you use OrangeFruitNet in your research, please cite:
> Khedkar, M., *et al.* “OrangeFruitNet: YOLO-based Detection and Yield Estimation Framework for Precision Citrus Farming.” .....................
