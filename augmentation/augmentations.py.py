# augmentation/augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.MotionBlur(p=0.2),
        A.HueSaturationValue(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def apply_transform(image, bboxes, category_ids):
    aug = get_train_transforms()
    res = aug(image=image, bboxes=bboxes, category_ids=category_ids)
    return res['image'], res['bboxes'], res['category_ids']
