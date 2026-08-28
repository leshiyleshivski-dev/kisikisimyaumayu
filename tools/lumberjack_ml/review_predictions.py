from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


COLORS = [
    (255, 0, 255),
    (0, 165, 255),
    (255, 255, 0),
    (255, 80, 80),
    (80, 80, 255),
    (180, 0, 255),
]


def read_labels(path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()[1:]]
        points = np.asarray(values, dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(points.astype(np.int32))
    return polygons


def mask_click_point(mask: np.ndarray) -> tuple[int, int]:
    binary = (mask > 0.5).astype(np.uint8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, _, _, point = cv2.minMaxLoc(distance)
    return int(point[0]), int(point[1])


def greedy_matches(
    click_points: list[tuple[int, int]], polygons: list[np.ndarray], tolerance: float = 32.0
) -> tuple[int, list[bool]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, point in enumerate(click_points):
        for gt_index, polygon in enumerate(polygons):
            distance = cv2.pointPolygonTest(polygon, point, True)
            if distance >= -tolerance:
                candidates.append((-distance, pred_index, gt_index))

    matched_predictions: set[int] = set()
    matched_ground_truth: set[int] = set()
    for _, pred_index, gt_index in sorted(candidates):
        if pred_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
    return len(matched_ground_truth), [i in matched_predictions for i in range(len(click_points))]


def deduplicate_predictions(
    masks: list[np.ndarray], scores: list[float], distance: float = 30.0,
) -> tuple[list[np.ndarray], list[float], list[tuple[int, int]]]:
    """Keep the strongest prediction from each click-sized cluster."""
    ordered = sorted(range(len(masks)), key=lambda index: -scores[index])
    kept_masks: list[np.ndarray] = []
    kept_scores: list[float] = []
    kept_points: list[tuple[int, int]] = []
    for index in ordered:
        point = mask_click_point(masks[index])
        if any(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2 <= distance ** 2
            for other in kept_points
        ):
            continue
        kept_masks.append(masks[index])
        kept_scores.append(scores[index])
        kept_points.append(point)
    return kept_masks, kept_scores, kept_points


def render(
    model: YOLO,
    image_path: Path,
    label_path: Path,
    confidence: float,
) -> tuple[np.ndarray, int, int, int]:
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    ground_truth = read_labels(label_path, width, height)

    result = model.predict(
        source=image,
        imgsz=960,
        conf=confidence,
        iou=0.5,
        retina_masks=True,
        verbose=False,
    )[0]
    canvas = image.copy()

    for polygon in ground_truth:
        cv2.polylines(canvas, [polygon], True, (50, 255, 50), 3, cv2.LINE_AA)

    masks: list[np.ndarray] = []
    scores: list[float] = []
    if result.masks is not None:
        masks = [mask.cpu().numpy() for mask in result.masks.data]
    if result.boxes is not None:
        scores = result.boxes.conf.cpu().numpy().tolist()

    masks, scores, click_points = deduplicate_predictions(masks, scores)
    matched, prediction_matches = greedy_matches(click_points, ground_truth)

    overlay = canvas.copy()
    for index, (mask, point) in enumerate(zip(masks, click_points)):
        color = COLORS[index % len(COLORS)]
        binary = mask > 0.5
        overlay[binary] = (
            0.45 * overlay[binary] + 0.55 * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)
        circle_color = (0, 255, 0) if prediction_matches[index] else (0, 0, 255)
        cv2.circle(overlay, point, 8, circle_color, -1, cv2.LINE_AA)
        cv2.circle(overlay, point, 12, (255, 255, 255), 2, cv2.LINE_AA)
        score = scores[index] if index < len(scores) else 0.0
        cv2.putText(
            overlay,
            f"{index + 1}:{score:.2f}",
            (point[0] + 12, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    title = (
        f"{image_path.stem}  conf={confidence:.2f}  "
        f"GT={len(ground_truth)}  predicted={len(masks)}  click-matches={matched}"
    )
    cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay, len(ground_truth), len(masks), matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = parser.parse_args()

    model = YOLO(str(args.model))
    image_paths = sorted((args.dataset / "images" / "val").glob("*.png"))
    rows: list[np.ndarray] = []
    for confidence in args.confidences:
        rendered: list[np.ndarray] = []
        totals = [0, 0, 0]
        for image_path in image_paths:
            label_path = args.dataset / "labels" / "val" / f"{image_path.stem}.txt"
            panel, gt_count, prediction_count, match_count = render(
                model, image_path, label_path, confidence
            )
            rendered.append(panel)
            totals[0] += gt_count
            totals[1] += prediction_count
            totals[2] += match_count
        rows.append(cv2.hconcat(rendered))
        print(
            f"conf={confidence:.2f}: GT={totals[0]}, predicted={totals[1]}, "
            f"click-matches={totals[2]}"
        )

    montage = cv2.vconcat(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage):
        raise RuntimeError(f"Could not write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
