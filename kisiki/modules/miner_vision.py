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
# The right-hand rock can extend past 65% of the frame on some camera angles.
# Keep the crop centered on the rug, but include the full outer edge of both
# rocks so a final inclusion there cannot be silently clipped.
TARGET_SEARCH_RATIO = (0.35, 0.24, 0.33, 0.49)
TOAST_TEXT_RATIO = (0.412, 0.935, 0.148, 0.037)
MINING_PROGRESS_RATIO = (0.82, 0.90, 0.175, 0.065)
REFERENCE_WIDTH = 2560
REFERENCE_HEIGHT = 1440
# A worked-out stone keeps shrinking; only props and rug pattern are excluded.
ROCK_MIN_AREA = 16_000
# Wider than any ore sprite, so a top-hat keeps inclusions and drops the rock.
# A rectangle is what makes this affordable: OpenCV runs it as two separable
# passes, while the same job with an elliptical element costs forty times more
# and would stall the collecting loop between clicks.
INCLUSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71))


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
            # The real track is about 230 px wide at 2K. The old floor of 18 px
            # accepted any blue sliver, and at night the bluish ground under
            # the player produced one every few frames - the module then took
            # a walk between rocks for an accepted mining interaction.
            width >= max(120, round(frame_width * 0.05))
            and 2 <= height <= 18
            and width / max(1, height) >= 8.0
            and area >= 400
            and fill >= 0.45
        ):
            return True
    return False


def mining_progress_fill(image: np.ndarray) -> float | None:
    """Return how full the ``Добыча руды`` bar is, or ``None`` if it is absent.

    Every accepted pickaxe swing moves the bar one step of about a tenth of
    its length. Reading that step is what lets the controller strike in the
    rhythm the game actually grants instead of on a fixed timer: the character
    used to finish a swing, stand idle for half a second and only then get the
    next click.
    """
    if image is None or image.size == 0:
        return None
    image, _scale_x, _scale_y = _normalize_detection_frame(image)
    progress, _left, _top = _ratio_crop(image, MINING_PROGRESS_RATIO)
    hsv = cv2.cvtColor(progress, cv2.COLOR_BGR2HSV)
    track = cv2.morphologyEx(
        cv2.inRange(hsv, (90, 80, 70), (125, 255, 255)),
        cv2.MORPH_CLOSE, np.ones((3, 5), dtype=np.uint8),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(track)
    frame_width = image.shape[1]
    # The widget draws the track and its lit head as separate strips; take the
    # span of every bar-shaped piece rather than betting on one of them.
    pieces = [
        (x, y, width, height)
        for x, y, width, height, area in (map(int, stats[component]) for component in range(1, count))
        if width >= max(120, round(frame_width * 0.05))
        and 2 <= height <= 18
        and width / max(1, height) >= 8.0
        and area >= 400
        and area / max(1, width * height) >= 0.45
    ]
    if not pieces:
        return None
    left = min(x for x, _y, _w, _h in pieces)
    right = max(x + width for x, _y, width, _h in pieces)
    top = min(y for _x, y, _w, _h in pieces)
    bottom = max(y + height for _x, y, _w, height in pieces)
    # The filled head of the bar is the same hue but noticeably lighter.
    band = hsv[max(0, top - 2):bottom + 2, left:right]
    filled = cv2.inRange(band, (90, 60, 150), (125, 255, 255))
    columns = np.where(filled.sum(axis=0) > 0)[0]
    if not len(columns):
        return 0.0
    return float(min(1.0, (int(columns.max()) + 1) / max(1, right - left)))


def _filled_contour(mask: np.ndarray) -> np.ndarray:
    """Fill a silhouette to its outer contour.

    The ore itself is often outside the stone threshold. Filling the external
    contour keeps black, gold and copper inclusions inside the clickable area.
    """
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _looks_like_stone(
    filled: np.ndarray, width: int, height: int, area: int, rug: np.ndarray,
) -> bool:
    """Judge one filled silhouette against the shape a stone always has."""
    if not (ROCK_MIN_AREA <= area <= 260_000):
        return False
    if not (90 <= width <= 900 and 90 <= height <= 650):
        return False
    if not 0.45 <= width / max(1, height) <= 2.2:
        return False
    if np.count_nonzero(filled) / max(1, width * height) < 0.60:
        return False
    ring = (
        cv2.dilate(filled, np.ones((41, 41), dtype=np.uint8)) > 0
    ) & (filled == 0)
    return not (np.any(ring) and float(np.mean(rug[ring] > 0)) < 0.55)


def _split_touching_stones(
    filled: np.ndarray, rug: np.ndarray,
) -> list[np.ndarray]:
    """Pull a blob of two neighbouring stones apart and judge them separately.

    Two stones lying side by side bridge into a single component: the closing
    that heals speckle in the colour mask also closes the narrow strip of rug
    between them. The merged bounding box is then mostly that gap, its fill
    lands near 0.53 and the pair is thrown away -- on a night table that left
    the whole tray to the "anything that is not rug" rescue below, which has no
    notion of where one stone ends, and a rock's worth of ore went uncollected.
    """
    # The ladder has to reach far enough to break the widest bridge the rug
    # closing can build: on the 10-43-16 table the two stones only came apart
    # at 41, and stopping at 27 left the pair rejected. Trying the small
    # kernels first keeps a narrow bridge from over-eroding the stones.
    for kernel_size in (9, 17, 25, 33, 41):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(filled, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(eroded)
        cores = [core for core in range(1, count) if stats[core, 4] >= ROCK_MIN_AREA // 3]
        if len(cores) < 2:
            continue
        accepted: list[np.ndarray] = []
        for core in cores:
            grown = cv2.dilate(np.uint8(labels == core) * 255, kernel) & filled
            part_count, part_labels, part_stats, _part_centroids = (
                cv2.connectedComponentsWithStats(grown)
            )
            if part_count < 2:
                continue
            largest = max(range(1, part_count), key=lambda part: part_stats[part, 4])
            _x, _y, width, height, area = map(int, part_stats[largest])
            part = _filled_contour(np.uint8(part_labels == largest) * 255)
            if _looks_like_stone(part, width, height, area, rug):
                accepted.append(part)
        # One good half and one bad one means the blob was a stone touching a
        # prop, not a pair of stones. Only a clean split is trusted.
        if len(accepted) >= 2:
            return accepted
    return []


def _stones_from(candidate: np.ndarray, rug: np.ndarray) -> np.ndarray:
    """Keep the components of ``candidate`` that behave like a stone.

    A stone shrinks as it is worked out, so bulk area is not what makes it a
    stone. What holds for every one of them is a compact, roughly equant
    silhouette lying on an unbroken pink field. The rug's patterned border,
    the props on its corners and the progress note fail one of those: they are
    either ragged, far too elongated, or not surrounded by rug.
    """
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate)
    stones = np.zeros_like(candidate)
    for component in range(1, component_count):
        _x, _y, width, height, area = map(int, stats[component])
        if area < ROCK_MIN_AREA:
            continue
        filled = _filled_contour(np.uint8(labels == component) * 255)
        if _looks_like_stone(filled, width, height, area, rug):
            stones |= filled
            continue
        for part in _split_touching_stones(filled, rug):
            stones |= part
    return stones


def _large_rock_mask(search: np.ndarray) -> np.ndarray:
    """Return the stones lying on the rug, filled to their outer contour."""
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    blue_gray = cv2.inRange(hsv, (82, 15, 25), (138, 255, 220))
    # A smaller closing kernel prevents the two rocks from being merged with
    # the rug into one oversized component before the contour fill happens.
    blue_gray = cv2.morphologyEx(
        blue_gray, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8),
    )
    rug = cv2.inRange(hsv, (145, 35, 65), (179, 255, 255))
    clean = _stones_from(blue_gray, rug)
    if not np.count_nonzero(clean):
        # Sunset and storm light can push a stone right out of the blue/grey
        # window, and then nothing on the table was clickable at all. Anything
        # compact that is *not* rug is a stone by elimination.
        not_rug = cv2.morphologyEx(
            cv2.bitwise_not(cv2.morphologyEx(
                cv2.inRange(hsv, (140, 25, 50), (179, 255, 255)),
                cv2.MORPH_CLOSE, np.ones((15, 15), dtype=np.uint8),
            )),
            cv2.MORPH_OPEN, np.ones((9, 9), dtype=np.uint8),
        )
        clean = _stones_from(not_rug, rug)
    if not np.count_nonzero(clean):
        clean = blue_gray
    return cv2.dilate(clean, np.ones((9, 9), dtype=np.uint8))


def _is_compact(labels: np.ndarray, component: int, area: int) -> bool:
    """Ore sprites are convex lumps; rock shading that survives a filter is not.

    This one test removed almost every false click on a finished table, where
    facets, cracks and rim shadows used to be offered as ore and were then
    clicked three times each before being retired.
    """
    component_mask = np.uint8(labels == component) * 255
    contours, _hierarchy = cv2.findContours(
        component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
    return area / max(1.0, hull_area) >= 0.65


def _hue_gap(hue: np.ndarray | float, reference: float) -> np.ndarray | float:
    """Shortest distance on the circular OpenCV hue scale (0..179)."""
    distance = np.abs(np.asarray(hue, dtype=np.float32) - reference)
    return np.minimum(distance, 180.0 - distance)


def _rock_relative_targets(
    search: np.ndarray,
    rock_mask: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, float, str]]:
    """Find compact patches whose colour differs from the stone they sit on.

    Fixed HSV windows cannot describe ore across weather, time of day and rock
    type, which is why a single dark inclusion was regularly left behind on a
    table where the bright ones were collected. What does hold everywhere is
    that an inclusion differs from *its own* stone: another hue, or far less
    colour than the stone (black, grey and white ore). The deviation is passed
    through a morphological top-hat, so only structures smaller than the
    kernel survive - the stone's own facets, cracks and shadows are large and
    share its colour, and drop out before anything is thresholded.
    """
    body = rock_mask > 0
    if not np.any(body):
        return []
    # Ore lies on the stone, never on its silhouette. Dark bays along the rim
    # are the one shape that survives the top-hat while not being ore at all.
    inner = cv2.erode(rock_mask, np.ones((27, 27), dtype=np.uint8)) > 0
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    # Stones are not convex, and the mask is grown outwards to keep ore that
    # sits right on the edge. Both let rug inside the searched area, and pink
    # rug against a blue stone is the strongest colour deviation on the whole
    # table: it used to form a ring along the entire silhouette that swallowed
    # every inclusion near the edge into one ragged, unclickable blob.
    rug = cv2.dilate(
        cv2.inRange(hsv, (140, 30, 45), (179, 255, 255)),
        np.ones((3, 3), dtype=np.uint8),
    ) > 0
    body &= ~rug
    hue, saturation, value = [channel.astype(np.float32) for channel in cv2.split(hsv)]
    rock_hue = float(np.median(hue[body]))
    rock_saturation = float(np.median(saturation[body]))

    deviation = np.uint8(np.clip(
        _hue_gap(hue, rock_hue) * 3.0 + np.clip(rock_saturation - saturation, 0, 255),
        0, 255,
    ))
    brightness = np.uint8(value)
    candidate = cv2.morphologyEx(deviation, cv2.MORPH_TOPHAT, INCLUSION_KERNEL) >= 150
    # A grey stone gives no colour contrast at all; there brightness is the
    # only thing an inclusion can differ in. The threshold stays high and fixed
    # on purpose: every attempt to scale it down to the stone's own smoothness
    # started offering shadowed folds as ore on the recorded tables.
    candidate |= cv2.morphologyEx(
        brightness, cv2.MORPH_TOPHAT, INCLUSION_KERNEL) >= 150
    candidate |= cv2.morphologyEx(
        brightness, cv2.MORPH_BLACKHAT, INCLUSION_KERNEL) >= 150
    candidate &= body

    mask = np.uint8(candidate) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    targets: list[tuple[int, int, float, str]] = []
    for component in range(1, count):
        _x, _y, width, height, area = map(int, stats[component])
        if not (150 <= area <= 6_000 and 10 <= width <= 95 and 10 <= height <= 95):
            continue
        if not 0.4 <= width / max(1, height) <= 2.5:
            continue
        fill = area / max(1, width * height)
        if fill < 0.35:
            continue
        if not _is_compact(labels, component, area):
            continue
        # A gap in the stone is rug through and through; a red or copper
        # sprite only has a rim of pixels that read as rug, so the bar is
        # set where a whole patch of rug sits and a sprite never does.
        if float(np.mean(rug[labels == component])) >= 0.5:
            continue
        center_x, center_y = centroids[component]
        if not inner[
            min(inner.shape[0] - 1, max(0, round(float(center_y)))),
            min(inner.shape[1] - 1, max(0, round(float(center_x)))),
        ]:
            continue
        targets.append((
            offset_x + round(float(center_x)),
            offset_y + round(float(center_y)),
            60.0 + min(area, 2_000) / 200.0 + fill * 4.0,
            "ore",
        ))
    return targets


def _component_targets(
    candidate: np.ndarray,
    rock_mask: np.ndarray,
    *,
    kind: str,
    offset_x: int,
    offset_y: int,
    hue_channel: np.ndarray | None = None,
    saturation_channel: np.ndarray | None = None,
    value_channel: np.ndarray | None = None,
    minimum_area: int | None = None,
    minimum_core_value: float | None = None,
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
                and 0.3 <= width / max(1, height) <= 3.0
                and fill >= 0.16
                and (area >= 200 or fill >= 0.48)
                # Closing is useful for the faceted ore sprites, but can also
                # inflate a handful of warm rock pixels into a convincing
                # component.  Require a real coloured core and never click a
                # component clipped by the search window.
                and raw_coverage >= 0.24
                and not touches_search_side_or_bottom
            )
            if accepted and value_channel is not None and minimum_core_value is not None:
                raw_component = component_mask & raw_mask
                accepted = bool(np.any(raw_component)) and (
                    float(np.mean(value_channel[raw_component])) >= minimum_core_value
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


def _adaptive_contrast_targets(
    search: np.ndarray,
    rock_mask: np.ndarray,
    *,
    offset_x: int,
    offset_y: int,
) -> list[tuple[int, int, float, str]]:
    """Find compact local outliers relative to the surface of each rock.

    Ore sprites are not guaranteed to keep one absolute HSV colour: sunlight,
    weather and post-processing change both the inclusion and the stone around
    it.  This fallback compares every pixel with a blurred local model of that
    *same* stone.  It is intentionally used only when the stricter colour and
    neutral detectors found nothing, so rock texture cannot outrank a known ore
    colour.
    """
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(search, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)

    # A broad blur is a local estimate of the uninterrupted rock surface.  Lab
    # distance catches hue changes while the S/V terms retain black and silver
    # inclusions whose hue is unstable at low saturation.
    local_lab = cv2.GaussianBlur(lab, (0, 0), 13.0)
    local_saturation = cv2.GaussianBlur(saturation, (0, 0), 13.0)
    local_value = cv2.GaussianBlur(value, (0, 0), 13.0)
    lab_delta = np.linalg.norm(lab - local_lab, axis=2)
    saturation_delta = np.abs(saturation - local_saturation)
    value_delta = np.abs(value - local_value)

    candidate = (
        ((lab_delta >= 25.0) & ((saturation_delta >= 20.0) | (value_delta >= 18.0)))
        | ((local_value - value >= 24.0) & (value <= 100.0))
        | ((value - local_value >= 28.0) & (saturation_delta >= 12.0))
    )
    candidate &= rock_mask > 0
    mask = np.uint8(candidate) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    targets: list[tuple[int, int, float, str]] = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        fill = area / max(1, width * height)
        if not (
            150 <= area <= 3_200
            and 10 <= width <= 86
            and 10 <= height <= 110
            and 0.35 <= width / max(1, height) <= 2.8
            and fill >= 0.35
        ):
            continue
        if not _is_compact(labels, component, area):
            continue

        component_mask = labels == component
        nearby = cv2.dilate(
            np.uint8(component_mask) * 255,
            np.ones((25, 25), dtype=np.uint8),
        ) > 0
        ring = nearby & ~component_mask & (rock_mask > 0)
        if np.count_nonzero(ring) < 80:
            continue

        component_lab = np.mean(lab[component_mask], axis=0)
        ring_lab = np.mean(lab[ring], axis=0)
        colour_contrast = float(np.linalg.norm(component_lab - ring_lab))
        value_contrast = abs(float(np.mean(value[component_mask]) - np.mean(value[ring])))
        saturation_contrast = abs(
            float(np.mean(saturation[component_mask]) - np.mean(saturation[ring]))
        )
        if not (
            colour_contrast >= 22.0
            or value_contrast >= 24.0
            or saturation_contrast >= 32.0
        ):
            continue

        center_x, center_y = centroids[component]
        score = 20.0 + colour_contrast + value_contrast * 0.25 + fill * 2.0
        targets.append((
            offset_x + round(float(center_x)),
            offset_y + round(float(center_y)),
            score,
            "adaptive",
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
        # Cyan inclusions lose more edge pixels than warm ones after GTA's
        # scaling/JPEG-like capture softness, especially when they are the
        # final tiny piece. The rock contour and compact-shape checks keep the
        # lower area threshold constrained to the actual stone.
        minimum_area=150,
        value_channel=value, minimum_core_value=175,
    ))
    targets.extend(_component_targets(
        neutral, rock_mask, kind="neutral", offset_x=offset_x, offset_y=offset_y,
        hue_channel=hue, saturation_channel=saturation,
    ))
    # The absolute masks are precise but blind to ore they were not tuned for.
    # The rock-relative pass is run on every frame, not only as a rescue, so a
    # dark inclusion is collected in the same sweep as the bright ones instead
    # of being left on a table the module already considers finished.
    targets.extend(_rock_relative_targets(
        search, rock_mask, offset_x=offset_x, offset_y=offset_y,
    ))
    # Local contrast stays the last resort for ore that matches neither its
    # own stone nor any known colour.
    if not targets:
        targets = _adaptive_contrast_targets(
            search, rock_mask, offset_x=offset_x, offset_y=offset_y,
        )
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
