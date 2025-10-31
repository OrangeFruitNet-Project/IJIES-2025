# utils/reproducibility.py
"""
Reproducibility utilities for OrangeFruitNet experiments.
Ensures deterministic behavior across YOLOv4/YOLOv5/YOLOv8 and Detectron2 runs.

Implements:
- Random seed initialization
- cuDNN deterministic settings
- Logging of environment versions
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    Args:
        seed (int): Random seed (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures reproducible CuDNN operations (slower but deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"[INFO] Reproducibility initialized with seed = {seed}")


def log_environment():
    """
    Log the active environment versions for transparency.
    """
    import platform
    import torch
    import albumentations
    import cv2

    print("\n[INFO] ==== Environment Information ====")
    print(f"Python version       : {platform.python_version()}")
    print(f"PyTorch version      : {torch.__version__}")
    print(f"CUDA available       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device          : {torch.cuda.get_device_name(0)}")
    print(f"Albumentations ver.  : {albumentations.__version__}")
    print(f"OpenCV version       : {cv2.__version__}")
    print("[INFO] ==================================\n")
