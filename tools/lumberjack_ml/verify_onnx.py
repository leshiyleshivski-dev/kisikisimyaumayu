"""Verify the exact OpenCV ONNX cascade shipped with the application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kisiki.modules.lumberjack.ml_vision import MODEL_CROP, find_branch_targets_ml
from kisiki.modules.lumberjack.vision import REFERENCE_HEIGHT, REFERENCE_WIDTH, TABLE_ZONE


def read_labels(path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = np.asarray([float(value) for value in line.split()[1:]])
        points = values.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(points.astype(np.int32))
    return polygons


def greedy_matches(
    points: list[tuple[int, int]], polygons: list[np.ndarray], tolerance: float = 32.0,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for point_index, point in enumerate(points):
        for polygon_index, polygon in enumerate(polygons):
            distance = cv2.pointPolygonTest(polygon, point, True)
            if distance >= -tolerance:
                candidates.append((-distance, point_index, polygon_index))
    matched_points: set[int] = set()
    matched_polygons: set[int] = set()
    for _distance, point_index, polygon_index in sorted(candidates):
        if point_index in matched_points or polygon_index in matched_polygons:
            continue
        matched_points.add(point_index)
        matched_polygons.add(polygon_index)
    return len(matched_polygons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()

    zone_left = round(REFERENCE_WIDTH * TABLE_ZONE[0])
    zone_top = round(REFERENCE_HEIGHT * TABLE_ZONE[1])
    zone_width = round(REFERENCE_WIDTH * TABLE_ZONE[2])
    crop_left = round(zone_width * MODEL_CROP[0])
    totals = [0, 0, 0]

    for crop_path in sorted((args.dataset / "images" / "val").glob("*.png")):
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            raise RuntimeError(f"Unreadable validation image: {crop_path}")
        frame = np.zeros((REFERENCE_HEIGHT, REFERENCE_WIDTH, 3), dtype=np.uint8)
        frame[
            zone_top:zone_top + crop.shape[0],
            zone_left:zone_left + crop.shape[1],
        ] = crop

        targets = find_branch_targets_ml(frame)
        if targets is None:
            raise RuntimeError("The shipped ONNX models could not be loaded")
        model_points = [
            (target.x - zone_left - crop_left, target.y - zone_top)
            for target in targets
        ]

        model_image_path = (
            args.dataset.parent / "dataset-crop-corrected-val"
            / "images" / "val" / crop_path.name
        )
        model_image = cv2.imread(str(model_image_path), cv2.IMREAD_COLOR)
        if model_image is None:
            raise RuntimeError(f"Unreadable cropped validation image: {model_image_path}")
        height, width = model_image.shape[:2]
        labels = read_labels(
            args.dataset.parent / "dataset-crop-corrected-val"
            / "labels" / "val" / f"{crop_path.stem}.txt",
            width,
            height,
        )
        matches = greedy_matches(model_points, labels)
        totals[0] += len(labels)
        totals[1] += len(model_points)
        totals[2] += matches
        print(
            f"{crop_path.stem}: real={len(labels)}, "
            f"predicted={len(model_points)}, matched={matches}"
        )

    print(f"TOTAL: real={totals[0]}, predicted={totals[1]}, matched={totals[2]}")
    if not totals[0] == totals[1] == totals[2]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
