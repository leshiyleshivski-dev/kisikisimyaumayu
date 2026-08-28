"""Train and export the small TIMBER CUT branch segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--name", default="branches")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = YOLO(args.model)
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=4,
        device=0,
        workers=4,
        patience=40,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        seed=20260828,
        deterministic=True,
        cache="disk",
        close_mosaic=20,
        degrees=2.0,
        translate=0.04,
        scale=0.12,
        perspective=0.0002,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.20,
        hsv_v=0.25,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    exported = YOLO(str(best)).export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        dynamic=False,
    )
    print(exported)
