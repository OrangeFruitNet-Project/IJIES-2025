"""
Inference script for OrangeFruitNet models on edge devices.
Implements real-time detection and FPS measurement for Jetson Nano and Raspberry Pi 4,
consistent with Section 4.4 and Table 10 of the OrangeFruitNet manuscript.
"""

import time
import cv2
import torch
from ultralytics import YOLO


def run_inference(weights, source=0, device="cpu", imgsz=640, conf=0.25):
    """
    Run real-time inference on a video source (camera or file).

    Args:
        weights (str): Path to model weights (.pt or .torchscript).
        source (str or int): Camera index (0) or path to video/image.
        device (str): 'cpu', 'cuda', or 'cuda:0'.
        imgsz (int): Inference image size (default 640).
        conf (float): Confidence threshold (default 0.25).
    """
    print(f"\n[INFO] Loading model from {weights} on {device} ...")
    model = YOLO(weights)
    model.to(device)
    model.conf = conf

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    frame_count = 0
    start_time = time.time()

    print("[INFO] Starting real-time inference ... Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, imgsz=imgsz, device=device, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        cv2.putText(
            annotated_frame, f"FPS: {fps:.2f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow("OrangeFruitNet Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Average FPS: {fps:.2f}")
    print("[INFO] Inference completed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OrangeFruitNet - Real-Time Inference Script")
    parser.add_argument("--weights", required=True, help="Path to model weights file (.pt or .torchscript)")
    parser.add_argument("--source", default=0, help="Camera index or path to input video/image")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (Jetson Nano uses 'cuda:0')")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    run_inference(args.weights, args.source, args.device, args.imgsz, args.conf)
