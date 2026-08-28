"""Small ONNX instance-segmentation detector for TIMBER CUT branches."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .vision import BranchTarget, _zone, to_reference


MODEL_WIDTH = 960
MODEL_HEIGHT = 480
MODEL_CROP = (0.18, 0.0, 0.90, 0.69)
PRIMARY_CONFIDENCE = 0.05
SENSITIVE_CONFIDENCE = 0.03
SAME_CLICK_DISTANCE = 60
MASK_DUPLICATE_IOU = 0.50


def _resource_path(*parts: str) -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[3]))
    return root.joinpath(*parts)


PRIMARY_MODEL = _resource_path(
    "assets", "vision", "lumberjack_branches_primary.onnx",
)
SENSITIVE_MODEL = _resource_path(
    "assets", "vision", "lumberjack_branches_sensitive.onnx",
)


@dataclass
class _Prediction:
    mask: np.ndarray
    score: float
    point: tuple[int, int]
    source: str


class _OnnxSegmenter:
    def __init__(self, path: Path) -> None:
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.output_names = self.net.getUnconnectedOutLayersNames()

    @staticmethod
    def _letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        ratio = min(MODEL_WIDTH / width, MODEL_HEIGHT / height)
        resized_width = round(width * ratio)
        resized_height = round(height * ratio)
        resized = cv2.resize(
            image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR,
        )
        left = round((MODEL_WIDTH - resized_width) / 2 - 0.1)
        right = MODEL_WIDTH - resized_width - left
        top = round((MODEL_HEIGHT - resized_height) / 2 - 0.1)
        bottom = MODEL_HEIGHT - resized_height - top
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, ratio, left, top

    @staticmethod
    def _click_point(mask: np.ndarray) -> tuple[int, int]:
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        _minimum, _maximum, _minimum_point, maximum_point = cv2.minMaxLoc(distance)
        return int(maximum_point[0]), int(maximum_point[1])

    @staticmethod
    def _native_mask(
        coefficients: np.ndarray,
        prototypes: np.ndarray,
        box: np.ndarray,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_shape
        channels, proto_height, proto_width = prototypes.shape
        logits = coefficients @ prototypes.reshape(channels, -1)
        mask = 1.0 / (1.0 + np.exp(-logits.reshape(proto_height, proto_width)))

        gain = min(proto_height / height, proto_width / width)
        pad_width = (proto_width - width * gain) / 2
        pad_height = (proto_height - height * gain) / 2
        left = round(pad_width - 0.1)
        right = round(proto_width - pad_width + 0.1)
        top = round(pad_height - 0.1)
        bottom = round(proto_height - pad_height + 0.1)
        mask = cv2.resize(
            mask[top:bottom, left:right], (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        x1, y1, x2, y2 = box.astype(int)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        binary = np.zeros((height, width), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            binary[y1:y2, x1:x2] = mask[y1:y2, x1:x2] > 0.5
        return binary

    def predict(
        self, image: np.ndarray, confidence: float, source: str,
    ) -> list[_Prediction]:
        padded, ratio, pad_left, pad_top = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 255.0, (MODEL_WIDTH, MODEL_HEIGHT),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        detections = next(output for output in outputs if output.ndim == 3)
        prototypes = next(output for output in outputs if output.ndim == 4)[0]

        height, width = image.shape[:2]
        found: list[_Prediction] = []
        for row in detections[0]:
            score = float(row[4])
            if score < confidence:
                continue
            box = row[:4].astype(np.float32)
            box[[0, 2]] = (box[[0, 2]] - pad_left) / ratio
            box[[1, 3]] = (box[[1, 3]] - pad_top) / ratio
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            mask = self._native_mask(row[6:], prototypes, box, (height, width))
            if not mask.any():
                continue
            found.append(_Prediction(mask, score, self._click_point(mask), source))
        return found


_primary: _OnnxSegmenter | None = None
_sensitive: _OnnxSegmenter | None = None
_models_failed = False


def _models() -> tuple[_OnnxSegmenter, _OnnxSegmenter] | None:
    global _primary, _sensitive, _models_failed
    if _models_failed:
        return None
    try:
        if _primary is None:
            _primary = _OnnxSegmenter(PRIMARY_MODEL)
        if _sensitive is None:
            _sensitive = _OnnxSegmenter(SENSITIVE_MODEL)
    except (cv2.error, OSError):
        _models_failed = True
        return None
    return _primary, _sensitive


def _mask_iou(one: np.ndarray, other: np.ndarray) -> float:
    union = np.logical_or(one, other).sum()
    if not union:
        return 0.0
    return float(np.logical_and(one, other).sum() / union)


def _same_prediction(one: _Prediction, other: _Prediction) -> bool:
    distance = (
        (one.point[0] - other.point[0]) ** 2
        + (one.point[1] - other.point[1]) ** 2
    )
    return distance <= 30 ** 2 or _mask_iou(one.mask, other.mask) >= MASK_DUPLICATE_IOU


def _merge_predictions(
    primary: list[_Prediction], sensitive: list[_Prediction], height: int,
) -> list[_Prediction]:
    kept: list[_Prediction] = []
    for candidate in sorted(primary, key=lambda item: -item.score):
        if candidate.point[1] > height * 0.94:
            continue
        if not any(_same_prediction(candidate, other) for other in kept):
            kept.append(candidate)

    for candidate in sorted(sensitive, key=lambda item: -item.score):
        if not height * 0.46 <= candidate.point[1] <= height * 0.94:
            continue
        if any(
            (candidate.point[0] - other.point[0]) ** 2
            + (candidate.point[1] - other.point[1]) ** 2
            <= SAME_CLICK_DISTANCE ** 2
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def find_branch_targets_ml(image: np.ndarray) -> list[BranchTarget] | None:
    """Return learned branch click points, or ``None`` when models are absent."""
    loaded = _models()
    if loaded is None or image is None or image.size == 0:
        return None

    reference, scale_x, scale_y = to_reference(image)
    zone, zone_left, zone_top = _zone(reference)
    height, width = zone.shape[:2]
    crop_left = round(width * MODEL_CROP[0])
    crop_top = round(height * MODEL_CROP[1])
    crop_right = round(width * MODEL_CROP[2])
    crop_bottom = round(height * MODEL_CROP[3])
    crop = zone[crop_top:crop_bottom, crop_left:crop_right]
    if crop.size == 0:
        return []

    primary_model, sensitive_model = loaded
    primary = primary_model.predict(crop, PRIMARY_CONFIDENCE, "primary")
    sensitive = sensitive_model.predict(crop, SENSITIVE_CONFIDENCE, "sensitive")
    merged = _merge_predictions(primary, sensitive, crop.shape[0])
    return [
        BranchTarget(
            round((zone_left + crop_left + candidate.point[0]) * scale_x),
            round((zone_top + crop_top + candidate.point[1]) * scale_y),
            candidate.score,
            "сучок",
        )
        for candidate in merged
    ]
