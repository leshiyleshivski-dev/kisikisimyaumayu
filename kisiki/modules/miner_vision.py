"""Computer vision helpers for the 2560x1440 and 1920x1080 quarry mini-game."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache

import cv2
import numpy as np


ORE_TYPES: tuple[tuple[str, str], ...] = (
    ("iron", "Железная"),
    ("silver", "Серебряная"),
    ("copper", "Медная"),
    ("tin", "Оловянная"),
    ("gold", "Золотая"),
    ("manganese", "Марганцевая"),
    ("silicon", "Кремниевая"),
    ("chrome", "Хромовая"),
    ("nickel", "Никелевая"),
)

ORE_NAMES = dict(ORE_TYPES)

# Ratios are shared by the 16:9 layouts. Full HD frames are normalized to the
# 2K reference before fixed-size component and OCR checks are applied.
OVERLAY_SAMPLE_RATIO = (0.30, 0.20, 0.42, 0.58)
TARGET_SEARCH_RATIO = (0.37, 0.24, 0.26, 0.49)
TOAST_TEXT_RATIO = (0.412, 0.935, 0.148, 0.037)
MINING_PROGRESS_RATIO = (0.82, 0.90, 0.175, 0.065)
REFERENCE_WIDTH = 2560
REFERENCE_HEIGHT = 1440


def is_supported_2k(width: int, height: int) -> bool:
    """Allow tiny border/screenshot differences around a 2560x1440 client."""
    return abs(int(width) - 2560) <= 8 and abs(int(height) - 1440) <= 8


def is_supported_full_hd(width: int, height: int) -> bool:
    """Allow small client-border differences around 1920x1080."""
    return abs(int(width) - 1920) <= 8 and abs(int(height) - 1080) <= 8


def is_supported_miner_resolution(width: int, height: int) -> bool:
    return is_supported_2k(width, height) or is_supported_full_hd(width, height)


def _normalize_detection_frame(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return a 2K-scale frame and factors for mapping points back to input."""
    height, width = image.shape[:2]
    if is_supported_full_hd(width, height):
        normalized = cv2.resize(
            image, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_LINEAR,
        )
        return normalized, width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT
    return image, 1.0, 1.0


def _ratio_crop(image: np.ndarray, ratio: tuple[float, float, float, float]) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    x_ratio, y_ratio, width_ratio, height_ratio = ratio
    left = max(0, round(width * x_ratio))
    top = max(0, round(height * y_ratio))
    right = min(width, left + max(1, round(width * width_ratio)))
    bottom = min(height, top + max(1, round(height * height_ratio)))
    return image[top:bottom, left:right], left, top


def miner_overlay_visible(image: np.ndarray) -> bool:
    """Recognize the large pink sorting rug without relying on fixed pixels."""
    if image is None or image.size == 0:
        return False
    image, _scale_x, _scale_y = _normalize_detection_frame(image)
    sample, _left, _top = _ratio_crop(image, OVERLAY_SAMPLE_RATIO)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    pink = cv2.inRange(hsv, (145, 35, 65), (179, 255, 255))
    ratio = float(np.count_nonzero(pink)) / max(1, pink.size)
    return ratio >= 0.34


def mining_progress_visible(image: np.ndarray) -> bool:
    """Recognize the thin blue ``Добыча руды`` progress bar at bottom-right.

    Merely seeing an E key press is not proof that Majestic accepted the rock
    interaction.  The progress bar is the first reliable acknowledgement from
    the game, so the controller uses it as a hard gate before sending strikes.
    """
    if image is None or image.size == 0:
        return False
    image, _scale_x, _scale_y = _normalize_detection_frame(image)
    progress, _left, _top = _ratio_crop(image, MINING_PROGRESS_RATIO)
    hsv = cv2.cvtColor(progress, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 80, 70), (125, 255, 255))
    blue = cv2.morphologyEx(
        blue, cv2.MORPH_CLOSE, np.ones((3, 5), dtype=np.uint8),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(blue)
    frame_width = image.shape[1]
    for component in range(1, count):
        _x, _y, width, height, area = map(int, stats[component])
        fill = area / max(1, width * height)
        if (
            width >= max(18, round(frame_width * 0.007))
            and 2 <= height <= 18
            and width / max(1, height) >= 4.0
            and area >= 40
            and fill >= 0.45
        ):
            return True
    return False


def _large_rock_mask(search: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    blue_gray = cv2.inRange(hsv, (82, 15, 25), (138, 255, 220))
    blue_gray = cv2.morphologyEx(
        blue_gray, cv2.MORPH_CLOSE, np.ones((31, 31), dtype=np.uint8),
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(blue_gray)
    clean = np.zeros_like(blue_gray)
    for component in range(1, component_count):
        _x, _y, width, height, area = map(int, stats[component])
        if 2_000 <= area <= 260_000 and 60 <= width <= 900 and 60 <= height <= 650:
            clean[labels == component] = 255
    if not np.count_nonzero(clean):
        clean = blue_gray
    return cv2.dilate(clean, np.ones((13, 13), dtype=np.uint8))


def _component_targets(
    candidate: np.ndarray,
    rock_mask: np.ndarray,
    *,
    kind: str,
    offset_x: int,
    offset_y: int,
    hue_channel: np.ndarray | None = None,
    saturation_channel: np.ndarray | None = None,
    minimum_area: int | None = None,
) -> list[tuple[int, int, float, str]]:
    raw_mask = candidate & (rock_mask > 0)
    mask = np.uint8(raw_mask) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    targets: list[tuple[int, int, float, str]] = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        touches_search_side_or_bottom = (
            x <= 3
            or x + width >= candidate.shape[1] - 3
            or y + height >= candidate.shape[0] - 3
        )
        fill = area / max(1, width * height)
        if kind == "color":
            component_mask = labels == component
            raw_coverage = float(np.mean(raw_mask[component_mask]))
            accepted = (
                (minimum_area or 45) <= area <= 1_000
                and 7 <= width <= 60
                and 7 <= height <= 60
                and fill >= 0.16
                and (area >= 200 or fill >= 0.48)
                # Closing is useful for the faceted ore sprites, but can also
                # inflate a handful of warm rock pixels into a convincing
                # component.  Require a real coloured core and never click a
                # component clipped by the search window.
                and raw_coverage >= 0.24
                and not touches_search_side_or_bottom
            )
            score = 100.0 + min(area, 700) / 100.0 + fill
        else:
            accepted = (
                150 <= area <= 4_000
                and 12 <= width <= 74
                and 12 <= height <= 100
                and 0.45 <= width / max(1, height) <= 2.2
                and fill >= 0.28
                and not touches_search_side_or_bottom
            )
            if accepted and saturation_channel is not None:
                component_mask = labels == component
                nearby = cv2.dilate(
                    np.uint8(component_mask) * 255,
                    np.ones((21, 21), dtype=np.uint8),
                ) > 0
                ring = nearby & ~component_mask & (rock_mask > 0)
                component_saturation = float(np.mean(saturation_channel[component_mask]))
                ring_saturation = (
                    float(np.mean(saturation_channel[ring]))
                    if np.any(ring) else component_saturation
                )
                component_hue = (
                    float(np.mean(hue_channel[component_mask]))
                    if hue_channel is not None else 90.0
                )
                ring_hue = (
                    float(np.mean(hue_channel[ring]))
                    if hue_channel is not None and np.any(ring) else component_hue
                )
                hue_distance = abs(component_hue - ring_hue)
                hue_distance = min(hue_distance, 180.0 - hue_distance)
                # Black and silver inclusions can have almost exactly the same
                # brightness as the rock (especially under sunny lighting).
                # Their stable distinction is a compact, low-saturation body
                # surrounded by the much bluer rock surface.
                accepted = (
                    area >= 400
                    and (
                        fill >= 0.42
                        or (area >= 1_000 and fill >= 0.40)
                        # Some nearly black inclusions have a jagged one-pixel
                        # rim which lowers their fill just below 40%.  Only
                        # allow that relaxation when the core is essentially
                        # colourless and the surrounding rock is strongly blue.
                        or (
                            area >= 1_000
                            and fill >= 0.39
                            and component_saturation <= 12
                            and ring_saturation - component_saturation >= 80
                        )
                    )
                    and component_saturation <= 40
                    and (
                        ring_saturation - component_saturation >= 40
                        # In flat sunlight both a silver inclusion and its rock
                        # can have low saturation.  The inclusion still has a
                        # warm/neutral hue while the rock remains distinctly
                        # blue, which is a safer signal than weakening the
                        # saturation threshold for every rock facet.
                        or (component_hue <= 45 and hue_distance >= 55)
                    )
                )
            score = 40.0 + min(area, 900) / 180.0 + fill * 2.0
        if accepted:
            center_x, center_y = centroids[component]
            targets.append((
                offset_x + round(float(center_x)),
                offset_y + round(float(center_y)),
                score,
                kind,
            ))
    return targets


def _merge_nearby_targets(targets: list[tuple[int, int, float, str]]) -> list[tuple[int, int, float, str]]:
    merged: list[tuple[int, int, float, str]] = []
    for target in sorted(targets, key=lambda item: item[2], reverse=True):
        x, y, _score, _kind = target
        if any((x - other[0]) ** 2 + (y - other[1]) ** 2 <= 24 ** 2 for other in merged):
            continue
        merged.append(target)
    return merged


def find_ore_targets(image: np.ndarray) -> list[tuple[int, int, float, str]]:
    """Return likely ore inclusion centers in client-relative coordinates.

    Bright warm/cyan inclusions are tried first. Neutral inclusions are kept as
    a lower-priority fallback because black and silver pieces share colors with
    cracks in the rock. The controller rescans after every click and remembers
    attempted positions, so harmless false candidates are not clicked twice.
    """
    if image is None or image.size == 0:
        return []
    image, scale_x, scale_y = _normalize_detection_frame(image)
    if not miner_overlay_visible(image):
        return []
    search, offset_x, offset_y = _ratio_crop(image, TARGET_SEARCH_RATIO)
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    rock_mask = _large_rock_mask(search)

    warm = (hue < 38) & (saturation > 115) & (value > 130)
    cyan = (hue >= 75) & (hue <= 115) & (saturation > 70) & (value > 150)
    # Neutral ore is genuinely close to the rock colour, but its useful core is
    # almost unsaturated.  The previous threshold (S < 52) also admitted blue
    # rock facets and cracks, producing a route through arbitrary points on an
    # empty stone.  Keep only the compact black/silver cores; coloured shells
    # around them are already handled by the higher-priority masks above.
    neutral = (saturation < 38) & ((value < 90) | (value > 105))

    # Warm and cyan masks are processed independently.  Joining them before
    # component analysis could bridge differently coloured ore pixels via an
    # anti-aliased edge and make the resulting component too large to accept.
    targets = _component_targets(
        warm, rock_mask, kind="color", offset_x=offset_x, offset_y=offset_y,
    )
    targets.extend(_component_targets(
        cyan, rock_mask, kind="color", offset_x=offset_x, offset_y=offset_y,
        minimum_area=250,
    ))
    targets.extend(_component_targets(
        neutral, rock_mask, kind="neutral", offset_x=offset_x, offset_y=offset_y,
        hue_channel=hue, saturation_channel=saturation,
    ))
    merged = _merge_nearby_targets(targets)
    if scale_x == 1.0 and scale_y == 1.0:
        return merged
    return [
        (round(x * scale_x), round(y * scale_y), score, kind)
        for x, y, score, kind in merged
    ]


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RgbQuad(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", ctypes.c_ubyte),
        ("rgbGreen", ctypes.c_ubyte),
        ("rgbRed", ctypes.c_ubyte),
        ("rgbReserved", ctypes.c_ubyte),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("bmiHeader", _BitmapInfoHeader), ("bmiColors", _RgbQuad * 1)]


@lru_cache(maxsize=96)
def _render_word_mask(text: str, face: str, font_height: int = 19) -> np.ndarray:
    """Render a Cyrillic candidate with Windows GDI; no external OCR is needed."""
    gdi32 = ctypes.windll.gdi32
    canvas_width, canvas_height = 360, 48
    info = _BitmapInfo()
    info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
    info.bmiHeader.biWidth = canvas_width
    info.bmiHeader.biHeight = -canvas_height
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    info.bmiHeader.biCompression = 0
    bits = ctypes.c_void_p()
    dc = gdi32.CreateCompatibleDC(0)
    bitmap = gdi32.CreateDIBSection(dc, ctypes.byref(info), 0, ctypes.byref(bits), 0, 0)
    if not dc or not bitmap or not bits.value:
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if dc:
            gdi32.DeleteDC(dc)
        raise OSError("Windows could not create an OCR bitmap")
    old_bitmap = gdi32.SelectObject(dc, bitmap)
    font = gdi32.CreateFontW(
        -font_height, 0, 0, 0, 700, 0, 0, 0, 204, 0, 0, 4, 0, face,
    )
    old_font = gdi32.SelectObject(dc, font)
    try:
        gdi32.SetBkColor(dc, 0)
        gdi32.SetTextColor(dc, 0xFFFFFF)
        gdi32.TextOutW(dc, 2, 2, text, len(text))
        raw_type = ctypes.c_ubyte * (canvas_width * canvas_height * 4)
        raw = np.ctypeslib.as_array(raw_type.from_address(bits.value))
        image = raw.reshape(canvas_height, canvas_width, 4)[:, :, 0].copy()
    finally:
        gdi32.SelectObject(dc, old_font)
        gdi32.SelectObject(dc, old_bitmap)
        gdi32.DeleteObject(font)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(dc)
    rows, columns = np.where(image > 20)
    if not len(columns):
        raise OSError("Windows could not render the OCR word")
    return image[rows.min():rows.max() + 1, columns.min():columns.max() + 1]


def extract_ore_notification_line(image: np.ndarray) -> np.ndarray | None:
    """Extract the white `Вы собрали … руда!` line from the bottom toast."""
    if image is None or image.size == 0:
        return None
    image, _scale_x, _scale_y = _normalize_detection_frame(image)
    toast, _left, _top = _ratio_crop(image, TOAST_TEXT_RATIO)
    hsv = cv2.cvtColor(toast, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 145), (179, 115, 255))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(white)
    line = np.zeros_like(white)
    for component in range(1, count):
        _x, _y, width, height, area = map(int, stats[component])
        if 2 <= width <= 30 and 8 <= height <= 20 and area >= 10:
            line[labels == component] = 255
    rows, columns = np.where(line > 0)
    if not len(columns):
        return None
    line = line[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    if not (215 <= line.shape[1] <= 290 and 12 <= line.shape[0] <= 24):
        return None
    return line


def _projection(mask: np.ndarray) -> np.ndarray:
    values = (mask > 20).mean(axis=0).astype(np.float32)
    return cv2.GaussianBlur(values.reshape(1, -1), (0, 0), 1.0).ravel()


def _word_score(observed: np.ndarray, candidate: np.ndarray) -> float:
    observed_profile, candidate_profile = _projection(observed), _projection(candidate)
    width_penalty = abs(len(observed_profile) - len(candidate_profile)) / max(
        len(observed_profile), len(candidate_profile), 1,
    )
    size = max(len(observed_profile), len(candidate_profile))
    observed_scaled = cv2.resize(
        observed_profile.reshape(1, -1), (size, 1), interpolation=cv2.INTER_LINEAR,
    ).ravel()
    candidate_scaled = cv2.resize(
        candidate_profile.reshape(1, -1), (size, 1), interpolation=cv2.INTER_LINEAR,
    ).ravel()
    return float(np.mean(np.abs(observed_scaled - candidate_scaled)) + width_penalty * 0.7)


def classify_ore_notification(line: np.ndarray) -> tuple[str | None, float]:
    """Classify one of nine fixed ore names by its projected glyph shape."""
    # The fixed phrase uses 104 px before and 49 px after the ore at 2K.
    if line is None or line.shape[1] <= 154:
        return None, 1.0
    word = line[:, 104:line.shape[1] - 49]
    fonts = ("Arial", "Segoe UI", "Tahoma", "Roboto")
    font_heights = (16, 17, 18, 19, 20)
    scored: list[tuple[float, str]] = []
    try:
        for ore_key, title in ORE_TYPES:
            # Word-only matching is sharp for the long names, but the narrow
            # GTA font can make "Никелевая" resemble "Хромовая" after both
            # profiles are scaled.  Combine it with a fixed-size comparison of
            # the complete toast: the prefix/suffix provide a scale reference
            # and preserve the real difference in title width.
            word_score = min(
                _word_score(word, _render_word_mask(title, face, font_height))
                for face in fonts
                for font_height in font_heights
            )
            line_score = min(
                _word_score(
                    line,
                    _render_word_mask(f"Вы собрали {title} руда!", face),
                )
                for face in fonts
            )
            score = (word_score + line_score) / 2.0
            scored.append((score, ore_key))
    except (OSError, AttributeError):
        return None, 1.0
    best_score, best_key = min(scored)
    return (best_key if best_score <= 0.23 else None), best_score


def read_ore_notification(image: np.ndarray) -> tuple[bool, str | None, float]:
    """Return (toast present, ore key or None, confidence score)."""
    line = extract_ore_notification_line(image)
    if line is None:
        return False, None, 1.0
    ore_key, score = classify_ore_notification(line)
    return True, ore_key, score
