# deploy/export_torchscript.py
from ultralytics import YOLO
import torch
import argparse

def export_torchscript(weights_path, output_path, imgsz=640):
    model = YOLO(weights_path)
    # export to torchscript via model.export
    model.export(format='torchscript', imgsz=imgsz, dynamic=False, opset=11, file=output_path)

if __name__ == "__main__":
    import sys
    weights = sys.argv[1]
    out = sys.argv[2]
    export_torchscript(weights, out)
