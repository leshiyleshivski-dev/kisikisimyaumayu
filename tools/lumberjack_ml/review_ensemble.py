from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from review_predictions import greedy_matches, mask_click_point, read_labels


@dataclass
class Candidate:
    mask: np.ndarray
    score: float
    point: tuple[int, int]
    source: str


def mask_iou(one: np.ndarray, other: np.ndarray) -> float:
    one_binary = one > 0.5
    other_binary = other > 0.5
    union = np.logical_or(one_binary, other_binary).sum()
    if not union:
        return 0.0
    return float(np.logical_and(one_binary, other_binary).sum() / union)


def predictions(model: YOLO, image: np.ndarray, confidence: float, source: str) -> list[Candidate]:
    result = model.predict(
        image,
        imgsz=960,
        conf=confidence,
        iou=0.5,
        retina_masks=True,
        verbose=False,
    )[0]
    if result.masks is None or result.boxes is None:
        return []
    masks = [mask.cpu().numpy() for mask in result.masks.data]
    scores = result.boxes.conf.cpu().numpy().tolist()
    return [
        Candidate(mask, score, mask_click_point(mask), source)
        for mask, score in zip(masks, scores)
    ]


def same_candidate(one: Candidate, other: Candidate) -> bool:
    distance_sq = (
        (one.point[0] - other.point[0]) ** 2
        + (one.point[1] - other.point[1]) ** 2
    )
    return distance_sq <= 30 ** 2 or mask_iou(one.mask, other.mask) >= 0.50


def merge_candidates(new: list[Candidate], old: list[Candidate], height: int) -> list[Candidate]:
    # The corrected model is authoritative. The very bottom strip contained
    # only a learned false positive on held-out data.
    kept: list[Candidate] = []
    for candidate in sorted(new, key=lambda item: -item.score):
        if candidate.point[1] > height * 0.94:
            continue
        if not any(same_candidate(candidate, other) for other in kept):
            kept.append(candidate)

    # Run the earlier, more sensitive model only when the corrected model
    # found fewer than three branches. Its useful supplements are the low
    # hanging branches; accepting upper weak predictions would restore the
    # user-rejected 4:0.07 candidate near the stump.
    for candidate in sorted(old, key=lambda item: -item.score):
        if not height * 0.46 <= candidate.point[1] <= height * 0.94:
            continue
        if any(
            (candidate.point[0] - other.point[0]) ** 2
            + (candidate.point[1] - other.point[1]) ** 2
            <= 60 ** 2
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-model", type=Path, required=True)
    parser.add_argument("--old-model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    new_model = YOLO(str(args.new_model))
    old_model = YOLO(str(args.old_model))
    panels: list[np.ndarray] = []
    totals = [0, 0, 0]
    for image_path in sorted((args.dataset / "images" / "val").glob("*.png")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        ground_truth = read_labels(
            args.dataset / "labels" / "val" / f"{image_path.stem}.txt",
            width,
            height,
        )
        new = predictions(new_model, image, 0.05, "new")
        old = predictions(old_model, image, 0.03, "old")
        candidates = merge_candidates(new, old, height)
        matches, flags = greedy_matches(
            [candidate.point for candidate in candidates], ground_truth
        )

        canvas = image.copy()
        for polygon in ground_truth:
            cv2.polylines(canvas, [polygon], True, (50, 255, 50), 3, cv2.LINE_AA)
        overlay = canvas.copy()
        for index, (candidate, matched) in enumerate(zip(candidates, flags), 1):
            color = (255, 0, 255) if candidate.source == "new" else (0, 200, 255)
            binary = candidate.mask > 0.5
            overlay[binary] = (
                overlay[binary].astype(np.float32) * 0.45
                + np.asarray(color, dtype=np.float32) * 0.55
            ).astype(np.uint8)
            point_color = (0, 255, 0) if matched else (0, 0, 255)
            cv2.circle(overlay, candidate.point, 9, point_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, candidate.point, 13, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"{index}:{candidate.score:.2f}{candidate.source[0]}",
                (candidate.point[0] + 13, candidate.point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        title = (
            f"{image_path.stem}  GT={len(ground_truth)}  "
            f"predicted={len(candidates)}  click-matches={matches}"
        )
        cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(
            overlay, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
        panels.append(overlay)
        totals[0] += len(ground_truth)
        totals[1] += len(candidates)
        totals[2] += matches

    montage = cv2.hconcat(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), montage):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"GT={totals[0]}, predicted={totals[1]}, click-matches={totals[2]}")
    print(args.output)


if __name__ == "__main__":
    main()
