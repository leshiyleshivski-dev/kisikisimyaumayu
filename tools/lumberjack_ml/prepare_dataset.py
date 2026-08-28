"""Build a YOLO instance-segmentation dataset from painted branch scribbles.

The annotator paints branches over an unchanged PNG. Dark and light strokes
are treated as separate instance groups. The original frame is copied into the
dataset; only the difference between the painted image and its original is
used for labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


DARK_OR_LIGHT_DELTA = 28
MIN_COMPONENT_AREA = 80


def stroke_masks(original: np.ndarray, annotated: np.ndarray) -> list[np.ndarray]:
    """Return connected dark/light strokes as separate binary instances."""
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.int16)
    annotated_gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY).astype(np.int16)
    changed = np.max(cv2.absdiff(original, annotated), axis=2) > 12
    groups = (
        ((original_gray - annotated_gray) > DARK_OR_LIGHT_DELTA) & changed,
        ((annotated_gray - original_gray) > DARK_OR_LIGHT_DELTA) & changed,
    )
    instances: list[np.ndarray] = []
    kernel = np.ones((9, 9), dtype=np.uint8)
    for group in groups:
        joined = cv2.morphologyEx(
            group.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel,
        )
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(joined)
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) < MIN_COMPONENT_AREA:
                continue
            mask = np.zeros_like(joined)
            mask[labels == component] = 255
            instances.append(mask)
    return instances


def mask_polygon(mask: np.ndarray) -> list[tuple[float, float]] | None:
    """Convert one painted instance to a compact normalized polygon."""
    height, width = mask.shape
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.002), True)
    if len(polygon) < 3:
        return None
    return [
        (float(point[0][0]) / width, float(point[0][1]) / height)
        for point in polygon
    ]


def crop_box(
    shape: tuple[int, ...], crop: tuple[float, float, float, float] | None,
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    if crop is None:
        return 0, 0, width, height
    left, top, right, bottom = crop
    return (
        round(width * left),
        round(height * top),
        round(width * right),
        round(height * bottom),
    )


def build_dataset(
    originals: Path,
    annotations: Path,
    output: Path,
    crop: tuple[float, float, float, float] | None = None,
    corrections: dict | None = None,
    train_files: set[str] | None = None,
) -> None:
    image_count = 0
    instance_count = 0
    split_counts = {"train": 0, "val": 0}
    for split in split_counts:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    for annotated_path in sorted(annotations.glob("*.png")):
        original_path = originals / annotated_path.name
        original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
        annotated = cv2.imread(str(annotated_path), cv2.IMREAD_COLOR)
        if original is None:
            raise FileNotFoundError(f"Missing original: {original_path}")
        if annotated is None:
            raise FileNotFoundError(f"Unreadable annotation: {annotated_path}")
        if original.shape != annotated.shape:
            raise ValueError(
                f"Shape changed for {annotated_path.name}: "
                f"{original.shape} != {annotated.shape}"
            )

        left, top, right, bottom = crop_box(original.shape, crop)
        original_crop = original[top:bottom, left:right]

        # test3 is the newest recording and remains unseen unless a corrected
        # frame is explicitly promoted to training.
        split = (
            "val"
            if annotated_path.stem.startswith("test3_")
            and annotated_path.name not in (train_files or set())
            else "train"
        )
        label_lines: list[str] = []
        for mask in stroke_masks(original, annotated):
            ignored_points = (corrections or {}).get(annotated_path.name, {}).get(
                "ignore_points", []
            )
            if any(
                0 <= int(point[0]) < mask.shape[1]
                and 0 <= int(point[1]) < mask.shape[0]
                and np.any(
                    mask[
                        max(0, int(point[1]) - 16):int(point[1]) + 17,
                        max(0, int(point[0]) - 16):int(point[0]) + 17,
                    ]
                )
                for point in ignored_points
            ):
                continue
            polygon = mask_polygon(mask[top:bottom, left:right])
            if polygon is None:
                continue
            coordinates = " ".join(f"{value:.6f}" for point in polygon for value in point)
            label_lines.append(f"0 {coordinates}")

        if not label_lines:
            raise ValueError(f"No branch strokes found in {annotated_path.name}")
        if not cv2.imwrite(str(output / "images" / split / original_path.name), original_crop):
            raise RuntimeError(f"Could not write cropped image: {original_path.name}")
        (output / "labels" / split / f"{annotated_path.stem}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8",
        )
        image_count += 1
        instance_count += len(label_lines)
        split_counts[split] += 1

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: branch\n",
        encoding="utf-8",
    )
    print(
        f"Built {image_count} images / {instance_count} instances "
        f"(train={split_counts['train']}, val={split_counts['val']})"
    )
    print(dataset_yaml)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Optional normalized crop bounds.",
    )
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--train-file", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    corrections = (
        json.loads(args.corrections.read_text(encoding="utf-8"))
        if args.corrections
        else None
    )
    build_dataset(
        args.originals,
        args.annotations,
        args.output,
        tuple(args.crop) if args.crop else None,
        corrections,
        set(args.train_file),
    )
