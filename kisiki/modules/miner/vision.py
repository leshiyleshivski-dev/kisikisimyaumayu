"""Разбор кадра карьера в GTA: стол, полоса добычи, вкрапления, уведомление.

Модуль не создаёт окон и не отправляет ввод: он получает кадр и отвечает, что
сейчас на экране. Так же живут пара ``blackjack.py`` / ``blackjack_vision.py``
и начинка ``poker/`` — распознавание проверяется на сохранённых кадрах без
запуска CustomTkinter.

Области заданы долями клиентской области, а размеры знаков и ядер — в пикселях
эталонного кадра 2560x1440. Кадр другого разрешения приводится к эталону в
``to_reference()``, поэтому 1920x1080 считается теми же числами, а координаты
целей возвращаются обратно в систему исходного кадра.

Замеры сняты по записи карьера от 26.08.2026 (38 камней) и разобраны в
``tests/fixtures/miner/README.md``.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import lru_cache
from typing import NamedTuple

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
ORE_KEYS = tuple(key for key, _title in ORE_TYPES)

REFERENCE_WIDTH = 2560
REFERENCE_HEIGHT = 1440

# Ковёр занимает середину экрана целиком; кусок взят с запасом внутрь, чтобы
# в него не попадали ни земля, ни подсказки по краям кадра.
RUG_SAMPLE_RATIO = (0.30, 0.20, 0.42, 0.58)
RUG_SAMPLE_SHARE = 0.30
# Полоса «Добыча руды» в правом нижнем углу: замерена как 226x8 в (2280,1334).
MINING_BAR_RATIO = (0.86, 0.915, 0.13, 0.025)
# Плашка уведомления «Вы собрали … руда!» по низу экрана, замерена в
# (1001,1336)-(1542,1402). Текст в ней прижат влево, а не отцентрован.
TOAST_RATIO = (0.34, 0.9278, 0.32, 0.0458)

PINK_RUG = ((145, 30, 55), (179, 255, 255))
# Камень на ковре синевато-серый при любом освещении записи: hue 104-124,
# насыщенность от 78. Земля и бумажка «прогресс сбора руды» тёплые (hue 7-20)
# и в эти границы не попадают.
STONE_HUE = (98, 128)
STONE_MIN_SATURATION = 65
STONE_VALUE = (50, 140)
# Камни записи занимают 44 000-107 000 px², сточенный — меньше. Порог взят
# заметно ниже самого мелкого: отбор всё равно держится на цвете и форме.
STONE_MIN_AREA = 18_000
STONE_MIN_FILL = 0.52
# Шейка между слипшимися камнями бывает и в девять пикселей, и в сорок.
STONE_SPLIT_EROSIONS = (9, 17, 25, 33, 41)

# Шире любого спрайта руды: top-hat этим ядром оставляет вкрапление и убирает
# сам камень. Прямоугольник, а не эллипс, не для красоты — OpenCV раскладывает
# его на два прохода, и кадр разбирается за десятки миллисекунд вместо секунд.
INCLUSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (71, 71))
# После top-hat крупинка руды сидит в верхнем проценте яркости камня: медиана
# фона 20-29, девяностый процентиль 49-92, а сами вкрапления 200-255.
INCLUSION_THRESHOLD = 150
INCLUSION_AREA = (28, 2600)
# Плотность цели: её площадь к площади выпуклой оболочки. Отсекает не руду, а
# растянутые кляксы — трещины, кайму ковра, лучи вспышки сбора.
#
# Порог стоял на 0,62 и резал ровно посередине живого облака. Бледный
# серебристый кристалл полупрозрачный: от камня у него отклоняется только
# светящийся контур и рёбра граней, а плоская середина — нет, и после top-hat
# он выходит не пятном, а полым рваным кольцом. Оболочка у кольца вдвое больше
# его самого, и плотность падает в 0,53-0,62 — под самый порог.
#
# По 1550 кадрам столов записи «тест 2»: у принятых целей минимум 0,621, у
# отброшенных максимум 0,620. Зазора между двумя кучами нет — это одна куча,
# разрезанная константой. Ниже 0,50 не живёт вообще ничего (самая рыхлая цель
# записи — 0,413), поэтому порог опущен туда: он возвращает 503 цели из 509
# отброшенных и не пускает ничего нового.
INCLUSION_MIN_SOLIDITY = 0.50
# Крупинка руды поперёк не больше сорока пикселей: цели ближе этого — куски
# одного блестящего спрайта, а не две руды.
INCLUSION_MERGE_DISTANCE = 30
# Стрелка курсора на камне — белое пятно с тёмной обводкой, и своё отклонение
# от камня у неё не хуже, чем у руды: на записи «тест 2» она набирала 245 при
# 212 у настоящего кристалла и вставала первой в очереди. Кликать её бесполезно
# — мышь уже там, — и стол 3:42 простоял из-за этого пять секунд.
#
# Поэтому под курсором отклонение гасится. Не цель отбрасывается, а пиксели:
# крупинка, накрытая стрелкой, остаётся целью по тому, что торчит из-под неё.
# Замерено вычитанием кадров: след стрелки 11x18 при 2560x1440. Рамка взята с
# запасом — на 1920x1080 тот же курсор после приведения к эталону крупнее.
CURSOR_PATCH = (-3, -3, 16, 26)
# Ниже этой яркости насыщенность пикселя ничего не значит — см. отбор целей.
LIT_ENOUGH_FOR_SATURATION = 65
# Руда отражает свет, щель между гранями — нет. Цель, в которой света нет
# вовсе, — это тень, и кликать по ней бесполезно: см. отбор целей.
LIT_ENOUGH_FOR_ORE = 25
# Тёмные выемки по силуэту камня похожи на чёрную руду. Центр цели обязан
# лежать внутри камня, ужатого на эту кайму.
INCLUSION_EDGE_MARGIN = 19

# Слова уведомления. «собрали» и «руда!» в нём не меняются, и по их ширине
# видно и масштаб текста, и что читается именно та фраза.
TOAST_ANCHOR = "собрали"
TOAST_TAIL = "руда!"
TOAST_ANCHOR_WIDTH = (60, 78)
TOAST_TAIL_WIDTH = (36, 50)
TOAST_ORE_WIDTH = (58, 150)
TOAST_WORD_GAP = 6
TOAST_ACCEPT_DISTANCE = 0.16
TOAST_WIDTH_TOLERANCE = 7.0
TOAST_FACES = ("Segoe UI", "Calibri", "Arial", "Tahoma", "Franklin Gothic Medium")
TOAST_HEIGHTS = (19, 20, 21, 22, 23, 24)


class OreTarget(NamedTuple):
    """Вкрапление руды на камне: куда нажимать и насколько уверенно."""

    x: int
    y: int
    score: float
    kind: str


class ToastReading(NamedTuple):
    """Что удалось прочитать в плашке «Вы собрали … руда!»."""

    visible: bool
    ore: str | None
    distance: float


def supported_resolution(width: int, height: int) -> bool:
    """2560x1440 и 1920x1080 с запасом на рамку окна."""
    return (
        (abs(int(width) - 2560) <= 8 and abs(int(height) - 1440) <= 8)
        or (abs(int(width) - 1920) <= 8 and abs(int(height) - 1080) <= 8)
    )


def to_reference(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Привести кадр к 2560x1440 и вернуть множители обратного пересчёта."""
    height, width = image.shape[:2]
    if width == REFERENCE_WIDTH and height == REFERENCE_HEIGHT:
        return image, 1.0, 1.0
    scaled = cv2.resize(
        image, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA,
    )
    return scaled, width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT


def _crop(
    image: np.ndarray, ratio: tuple[float, float, float, float],
) -> tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    left, top = round(width * ratio[0]), round(height * ratio[1])
    right = min(width, left + round(width * ratio[2]))
    bottom = min(height, top + round(height * ratio[3]))
    return image[top:bottom, left:right], left, top


def sorting_table_visible(image: np.ndarray) -> bool:
    """Открыт ли стол сортировки — розовый ковёр во весь центр экрана.

    На кадрах записи доля розового в центральном куске равна 0,55-0,68 на
    столе и не превышает 0,003 где угодно ещё, так что порог здесь не спорный.
    """
    if image is None or image.size == 0:
        return False
    reference, _scale_x, _scale_y = to_reference(image)
    sample, _left, _top = _crop(reference, RUG_SAMPLE_RATIO)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    pink = cv2.inRange(hsv, *PINK_RUG)
    return float(np.count_nonzero(pink)) / max(1, pink.size) >= RUG_SAMPLE_SHARE


def _mining_bar_box(reference: np.ndarray) -> tuple[int, int, int, int] | None:
    crop, left, top = _crop(reference, MINING_BAR_RATIO)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 80, 55), (115, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 7), dtype=np.uint8))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(blue)
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        # Настоящая дорожка — 226x8. Синеватый обрывок земли под ногами такой
        # длины не даёт, и на ночной прогулке полоса больше не мерещится.
        if (
            width >= 150
            and 3 <= height <= 16
            and width / max(1, height) >= 10.0
            and area >= 600
            and area / max(1, width * height) >= 0.55
        ):
            return left + x, top + y, width, height
    return None


def mining_bar_visible(image: np.ndarray) -> bool:
    """Идёт ли добыча: игра сама рисует полосу, пока камень принимает удары."""
    if image is None or image.size == 0:
        return False
    reference, _scale_x, _scale_y = to_reference(image)
    return _mining_bar_box(reference) is not None


def mining_bar_fill(image: np.ndarray) -> float | None:
    """Насколько полоса заполнена, или ``None``, если её нет.

    Заполненная часть светлая (V около 250), пустая — тёмно-бирюзовая (V около
    137), и порог посередине разводит их без единого промаха на записи. Каждый
    засчитанный удар двигает полосу примерно на десятую часть — по этому шагу
    помощник и ловит ритм, который даёт игра, вместо стрельбы по таймеру.
    """
    if image is None or image.size == 0:
        return None
    reference, _scale_x, _scale_y = to_reference(image)
    box = _mining_bar_box(reference)
    if box is None:
        return None
    left, top, width, height = box
    band = reference[top:top + height, left:left + width]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    lit = cv2.inRange(hsv, (95, 60, 150), (115, 255, 255))
    columns = np.where(lit.sum(axis=0) > 0)[0]
    if not len(columns):
        return 0.0
    return float(min(1.0, (int(columns.max()) + 1) / max(1, width)))


def _rug_mask(hsv: np.ndarray) -> np.ndarray:
    rug = cv2.inRange(hsv, *PINK_RUG)
    # Смыкание залечивает крапинки узора. Оно только добавляет розовое, поэтому
    # узкая полоска ковра между двумя соприкасающимися камнями от него не
    # исчезает — а именно на ней камни и разделяются.
    #
    # Ядро маленькое не случайно. Красная руда по цвету от ковра неотличима и
    # попадает в эту маску; ядром 25x25 её раздувало до края камня, выемка
    # выходила на силуэт, и заливка контура её уже не закрывала. На тридцати
    # семи столах записи разница между 25 и 9 — шестнадцать найденных
    # вкраплений, а число самих камней не меняется вовсе.
    return cv2.morphologyEx(rug, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))


def _fill_interior(mask: np.ndarray) -> np.ndarray:
    """Залить силуэт до внешнего контура.

    Красная руда по оттенку от розового ковра неотличима: hue у неё уходит за
    180 и возвращается в тот же диапазон 145-179. Поэтому красное вкрапление
    попадает в маску ковра и **вырезает себе дырку в собственном камне** — на
    столе 3:23 записи так пропадало единственное оставшееся вкрапление. Заливка
    внешнего контура возвращает его внутрь камня. Выемки по силуэту при этом
    остаются снаружи: контур повторяет очертания, а не обтягивает их выпукло.
    """
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _stone_masks(hsv: np.ndarray) -> list[np.ndarray]:
    """Камни как дырки внутри ковра.

    Искать камень своим цветом — путь, на котором он теряется при каждом
    новом освещении. Ковёр же на столе всегда один и тот же, и всё, что лежит
    внутри его силуэта дыркой, — это камень. Реквизит по краям (лампа, бочка,
    подушки) и бумажка «прогресс сбора руды» край силуэта режут, дырками не
    становятся и в отбор не попадают вовсе.
    """
    rug = _rug_mask(hsv)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(rug)
    if count < 2:
        return []
    largest = max(range(1, count), key=lambda component: stats[component][4])
    silhouette = (labels == largest).astype(np.uint8) * 255
    holes = cv2.morphologyEx(
        cv2.bitwise_and(_fill_interior(silhouette), cv2.bitwise_not(rug)),
        cv2.MORPH_OPEN, np.ones((9, 9), dtype=np.uint8),
    )
    hole_count, hole_labels, hole_stats, _hole_centroids = (
        cv2.connectedComponentsWithStats(holes)
    )
    masks: list[np.ndarray] = []
    for component in range(1, hole_count):
        area = int(hole_stats[component][4])
        if area < STONE_MIN_AREA:
            continue
        piece = (hole_labels == component).astype(np.uint8) * 255
        # Не прошёл по форме — не выбрасываем, а пробуем разнять. Два камня
        # разделяет полоска ковра, но в тени она сама перестаёт быть розовой,
        # и пара слипается в один блоб.
        candidates = [piece] if _looks_like_stone(piece) else _split_touching_stones(piece)
        for candidate in candidates:
            if _stone_colour_fits(hsv, candidate):
                masks.append(_fill_interior(candidate))
    return masks


def _looks_like_stone(mask: np.ndarray) -> bool:
    """Камень занимает заметную часть своей рамки; слипшаяся пара — нет."""
    columns = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if not len(columns) or not len(rows):
        return False
    width = int(columns[-1] - columns[0] + 1)
    height = int(rows[-1] - rows[0] + 1)
    area = int(np.count_nonzero(mask))
    return area >= STONE_MIN_AREA and area / max(1, width * height) >= STONE_MIN_FILL


def _stone_colour_fits(hsv: np.ndarray, mask: np.ndarray) -> bool:
    sample = hsv[mask > 0]
    if not len(sample):
        return False
    hue, saturation, value = (float(np.median(sample[:, index])) for index in range(3))
    return (
        STONE_HUE[0] <= hue <= STONE_HUE[1]
        and saturation >= STONE_MIN_SATURATION
        and STONE_VALUE[0] <= value <= STONE_VALUE[1]
    )


def _split_touching_stones(blob: np.ndarray) -> list[np.ndarray]:
    """Развести слипшуюся пару эрозией и нарастить куски обратно.

    Лестница нужна целиком: шейка между камнями бывает и в девять пикселей, и
    в сорок. Принимается только чистое разделение — два и более куска, каждый
    похожий на камень по отдельности. Камень, прилипший к реквизиту, так не
    пройдёт и останется отброшенным, как и был.

    Разнимать важно не только ради формы: цвет-эталон для поиска вкраплений
    берётся с самого камня, а на столе 0:33 записи от 09:50 один камень почти
    чёрный, второй сине-серый. Общая на двоих медиана не годится ни одному.
    """
    for size in STONE_SPLIT_EROSIONS:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        cores = cv2.erode(blob, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(cores)
        seeds = [
            component for component in range(1, count)
            if stats[component][4] >= STONE_MIN_AREA // 4
        ]
        if len(seeds) < 2:
            continue
        grown = [
            cv2.bitwise_and(
                cv2.dilate((labels == seed).astype(np.uint8) * 255, kernel), blob,
            )
            for seed in seeds
        ]
        if all(_looks_like_stone(piece) for piece in grown):
            return grown
    return []


def _hue_gap(hue: np.ndarray, reference: float) -> np.ndarray:
    raw = np.abs(hue.astype(np.int16) - int(round(reference)))
    return np.minimum(raw, 180 - raw)


def _is_compact(mask: np.ndarray, area: int) -> bool:
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return False
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    return area / hull_area >= INCLUSION_MIN_SOLIDITY


def _is_lit(value: np.ndarray, blob: np.ndarray) -> bool:
    """Светится ли цель хоть сколько-нибудь.

    Руда — предмет, лежащий на камне: он отражает свет, и даже тёмная крупинка
    внутри освещена. Щель между гранями света не отражает вовсе, и её яркость
    падает в ноль. Отклонение от камня у такой щели при этом огромное — ровно
    потому, что чернее уже некуда, — и до этой проверки она проходила как руда
    убедительнее настоящей.

    Отсюда правило: там, где света нет, руды нет. Берётся девяностый процентиль,
    а не максимум: одинокий блик на кромке щели не должен её спасать.

    Порог заведомо ниже самого камня: в отбор камней проходит только силуэт с
    медианной яркостью от ``STONE_VALUE[0]``, так что весь камень целиком под
    эту проверку не уйдёт ни при каком освещении.
    """
    return float(np.percentile(value[blob], 90)) >= LIT_ENOUGH_FOR_ORE


def _blank_cursor(deviation: np.ndarray, cursor: tuple[int, int] | None) -> None:
    """Погасить отклонение под стрелкой курсора — прямо в ``deviation``.

    Стрелка отклоняется от камня не хуже руды и на записи вставала первой в
    очереди, а клик по ней уходит в никуда: мышь и так уже там. Гасятся именно
    пиксели, а не готовая цель, — крупинку под стрелкой видно по тому, что
    из-под неё торчит, и она остаётся целью.
    """
    if cursor is None:
        return
    x, y = cursor
    dx, dy, width, height = CURSOR_PATCH
    # Обрезаем по самому кадру, а не полагаемся на срез: у отрицательного конца
    # среза numpy отсчитывает от другого края и гасит пол-камня.
    left = max(0, x + dx)
    top = max(0, y + dy)
    right = min(deviation.shape[1], x + dx + width)
    bottom = min(deviation.shape[0], y + dy + height)
    if left >= right or top >= bottom:
        return
    deviation[top:bottom, left:right] = 0


def _targets_on_stone(
    hsv: np.ndarray, body: np.ndarray, cursor: tuple[int, int] | None = None,
) -> list[tuple[int, int, float, str]]:
    """Вкрапление — это мелкое отклонение от цвета своего же камня.

    Эталон берётся с самого камня, поэтому правило работает и на сером камне
    в тени, и на синем на закате. Отклонение прогоняется через top-hat: грани,
    трещины и тени крупные, ядро их съедает, а крупинка руды остаётся.
    """
    if not np.any(body):
        return []
    rows, columns = np.where(body > 0)
    # Ядро top-hat смотрит на 35 пикселей вокруг, поэтому камень режется с
    # запасом: без него у самой кромки открывающее преобразование опирается на
    # пустоту, и весь силуэт вспыхивает ложной каймой.
    pad = INCLUSION_KERNEL.shape[0] // 2 + 5
    top = max(0, int(rows.min()) - pad)
    bottom = min(hsv.shape[0], int(rows.max()) + pad + 1)
    left = max(0, int(columns.min()) - pad)
    right = min(hsv.shape[1], int(columns.max()) + pad + 1)
    patch = hsv[top:bottom, left:right]
    stone = body[top:bottom, left:right]
    inside = stone > 0

    sample = patch[inside]
    hue_reference = float(np.median(sample[:, 0]))
    saturation_reference = float(np.median(sample[:, 1]))
    value_reference = float(np.median(sample[:, 2]))

    hue_gap = _hue_gap(patch[:, :, 0], hue_reference) * 3
    saturation_gap = np.abs(patch[:, :, 1].astype(np.int16) - int(saturation_reference))
    # Насыщенность в темноте — не цвет, а шум: у почти чёрного пикселя S
    # уползает к 255 просто оттого, что делить не на что. Тёмная фаска по
    # нижнему краю камня из-за этого давала отклонение 173 при пороге 150 и
    # выглядела рудой убедительнее самой руды: на столе 2:33 записи от 09:50
    # таких ложных целей было шесть из десяти, и одна из них закрывала собой
    # настоящее красное вкрапление. Оттенок в темноте ещё различим — у тёмной
    # руды он расходится с камнем втрое сильнее, чем у фаски, — поэтому
    # гасится только насыщенность.
    saturation_gap[patch[:, :, 2] < LIT_ENOUGH_FOR_SATURATION] = 0
    value_gap = np.abs(patch[:, :, 2].astype(np.int16) - int(value_reference)) * 3 // 2
    deviation = np.maximum(np.maximum(hue_gap, saturation_gap), value_gap)
    deviation = np.clip(deviation, 0, 255).astype(np.uint8)
    _blank_cursor(
        deviation, None if cursor is None else (cursor[0] - left, cursor[1] - top),
    )

    peaks = cv2.morphologyEx(deviation, cv2.MORPH_TOPHAT, INCLUSION_KERNEL)
    strong = (peaks >= INCLUSION_THRESHOLD).astype(np.uint8) * 255
    core = cv2.erode(
        stone, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (INCLUSION_EDGE_MARGIN * 2 + 1,) * 2,
        ),
    )
    strong = cv2.bitwise_and(strong, core)
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(strong)
    found: list[tuple[int, int, float, str]] = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if not INCLUSION_AREA[0] <= area <= INCLUSION_AREA[1]:
            continue
        center_x, center_y = (int(round(value)) for value in centroids[component])
        if not core[center_y, center_x]:
            continue
        piece = (labels == component).astype(np.uint8) * 255
        if not _is_compact(piece[y:y + height, x:x + width], area):
            continue
        blob = piece > 0
        if not _is_lit(patch[:, :, 2], blob):
            continue
        score = float(np.mean(peaks[blob]))
        coloured = float(np.mean(hue_gap[blob])) >= 30
        found.append((
            left + center_x, top + center_y, score,
            "цветное" if coloured else "тёмное",
        ))
    return found


def _merge_nearby(
    targets: list[tuple[int, int, float, str]],
) -> list[tuple[int, int, float, str]]:
    """Оставить от одной крупинки одну цель.

    Спрайт руды блестит, и порог режет его на два-три куска: помощник тратил
    на такую крупинку три клика вместо одного, а на цель уходит 0,6 с при столе
    длиной 3,9 с по медиане.
    Слипшиеся ближе тридцати пикселей цели — это одна цель, и берётся у неё
    лучший скор.
    """
    kept: list[tuple[int, int, float, str]] = []
    for x, y, score, kind in targets:
        if any(
            (x - other_x) ** 2 + (y - other_y) ** 2 <= INCLUSION_MERGE_DISTANCE ** 2
            for other_x, other_y, _score, _kind in kept
        ):
            continue
        kept.append((x, y, score, kind))
    return kept


def find_ore_targets(
    image: np.ndarray, cursor: tuple[int, int] | None = None,
) -> list[OreTarget]:
    """Вкрапления руды на камнях стола, самые уверенные первыми.

    ``cursor`` — где сейчас стрелка мыши в координатах кадра. Её пиксели из
    разбора выпадают: сама по себе стрелка на камне выглядит рудой не хуже
    настоящей, а клик по ней уходит в никуда. Без точки помощник разбирает
    кадр как раньше — зрение само по себе о мыши ничего не знает.
    """
    if image is None or image.size == 0:
        return []
    reference, scale_x, scale_y = to_reference(image)
    if cursor is not None:
        cursor = (round(cursor[0] / scale_x), round(cursor[1] / scale_y))
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    found: list[tuple[int, int, float, str]] = []
    for body in _stone_masks(hsv):
        found.extend(_targets_on_stone(hsv, body, cursor))
    found.sort(key=lambda target: -target[2])
    return [
        OreTarget(round(x * scale_x), round(y * scale_y), score, kind)
        for x, y, score, kind in _merge_nearby(found)
    ]


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("bmiHeader", _BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]


RUSSIAN_CHARSET = 204
# В GDI OPAQUE это 2, а TRANSPARENT — 1. С непрозрачным фоном по умолчанию
# белый и текст белый: слово выходит сплошным прямоугольником своей ширины.
TRANSPARENT_BACKGROUND = 1
BLACK_BRUSH = 4


@lru_cache(maxsize=256)
def render_word(text: str, face: str, height: int) -> np.ndarray | None:
    """Нарисовать слово системным шрифтом и вернуть его маску.

    Эталоны знаков карт покер нарезает из самой игры, и здесь это было бы
    честнее. Но за двенадцать минут карьера выпали четыре вида руды из девяти,
    а остальные пять нарезать не с чего. Шрифты Windows дают все девять сразу,
    и сравнение всё равно идёт не по буквам, а по силуэту слова.
    """
    canvas_width, canvas_height = 512, 64
    gdi32, user32 = ctypes.windll.gdi32, ctypes.windll.user32
    screen = user32.GetDC(0)
    device = gdi32.CreateCompatibleDC(screen)
    info = _BitmapInfo()
    info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
    info.bmiHeader.biWidth = canvas_width
    info.bmiHeader.biHeight = -canvas_height
    info.bmiHeader.biPlanes = 1
    info.bmiHeader.biBitCount = 32
    bits = ctypes.c_void_p()
    bitmap = gdi32.CreateDIBSection(
        device, ctypes.byref(info), 0, ctypes.byref(bits), None, 0,
    )
    font = gdi32.CreateFontW(
        -height, 0, 0, 0, 700, 0, 0, 0, RUSSIAN_CHARSET, 0, 0, 4, 0, face,
    )
    try:
        gdi32.SelectObject(device, bitmap)
        gdi32.SelectObject(device, font)
        rectangle = wintypes.RECT(0, 0, canvas_width, canvas_height)
        user32.FillRect(device, ctypes.byref(rectangle), gdi32.GetStockObject(BLACK_BRUSH))
        gdi32.SetBkColor(device, 0x000000)
        gdi32.SetBkMode(device, TRANSPARENT_BACKGROUND)
        gdi32.SetTextColor(device, 0xFFFFFF)
        gdi32.TextOutW(device, 6, 6, text, len(text))
        buffer = (ctypes.c_ubyte * (canvas_width * canvas_height * 4)).from_address(
            bits.value,
        )
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(
            canvas_height, canvas_width, 4,
        ).copy()
    finally:
        gdi32.DeleteObject(font)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(device)
        user32.ReleaseDC(0, screen)
    mask = (frame[:, :, 0] > 110).astype(np.uint8) * 255
    rows, columns = np.where(mask > 0)
    if not len(columns):
        return None
    return mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]


def _profile(mask: np.ndarray) -> np.ndarray:
    values = (mask > 20).mean(axis=0).astype(np.float32)
    return cv2.GaussianBlur(values.reshape(1, -1), (0, 0), 1.2).ravel()


def _shape_distance(observed: np.ndarray, candidate: np.ndarray) -> float:
    first, second = _profile(observed), _profile(candidate)
    size = max(len(first), len(second), 1)
    penalty = abs(len(first) - len(second)) / size
    first = cv2.resize(
        first.reshape(1, -1), (size, 1), interpolation=cv2.INTER_LINEAR,
    ).ravel()
    second = cv2.resize(
        second.reshape(1, -1), (size, 1), interpolation=cv2.INTER_LINEAR,
    ).ravel()
    return float(np.mean(np.abs(first - second)) + penalty * 0.8)


def _toast_words(image: np.ndarray) -> list[np.ndarray]:
    """Разрезать строку уведомления на слова по разрывам между знаками."""
    crop, _left, _top = _crop(image, TOAST_RATIO)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (179, 95, 255))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(white)
    glyphs = np.zeros_like(white)
    height = crop.shape[0]
    for component in range(1, count):
        _x, y, width, glyph_height, area = map(int, stats[component])
        if not (2 <= width <= 34 and 8 <= glyph_height <= 26 and area >= 12):
            continue
        # Строка стоит по середине плашки; зелёная галочка слева и края
        # плашки в эту полосу не попадают.
        if not 10 <= y + glyph_height / 2 <= height - 8:
            continue
        glyphs[labels == component] = 255
    columns = np.where(glyphs.sum(axis=0) > 0)[0]
    if not len(columns):
        return []
    runs: list[tuple[int, int]] = []
    start = columns[0]
    for previous, current in zip(columns, columns[1:]):
        if current - previous > TOAST_WORD_GAP:
            runs.append((start, previous))
            start = current
    runs.append((start, columns[-1]))
    words = []
    for left, right in runs:
        if right - left + 1 < 4:
            continue
        piece = glyphs[:, left:right + 1]
        rows = np.where(piece.sum(axis=1) > 0)[0]
        words.append(piece[rows.min():rows.max() + 1])
    return words


def _ore_word(words: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    """Найти в строке название руды и заодно масштаб текста.

    Фраза целиком — «Вы собрали … руда!». Название всегда стоит между
    «собрали» и «руда!», и требовать обоих соседей стоит не из педантизма: так
    чужое уведомление не превратится в добытую руду, а ширина «собрали» даёт
    опору, по которой ширины девяти названий сравниваются в пикселях.
    """
    for index, word in enumerate(words):
        if index == 0 or index + 1 >= len(words):
            continue
        anchor = words[index - 1].shape[1]
        tail = words[index + 1].shape[1]
        width = word.shape[1]
        if not TOAST_ORE_WIDTH[0] <= width <= TOAST_ORE_WIDTH[1]:
            continue
        if not TOAST_ANCHOR_WIDTH[0] <= anchor <= TOAST_ANCHOR_WIDTH[1]:
            continue
        if not TOAST_TAIL_WIDTH[0] <= tail <= TOAST_TAIL_WIDTH[1]:
            continue
        return word, float(anchor)
    return None


def classify_ore_word(word: np.ndarray, anchor_width: float) -> tuple[str | None, float]:
    """Опознать название руды по силуэту слова и его ширине.

    Одного силуэта мало: у «Кремниевая» и «Оловянная» он расходится в третьем
    знаке после запятой. Зато ширина «собрали» рядом задаёт масштаб шрифта, и
    предсказанная ширина названия ложится в считаные пиксели от снятой с
    экрана — этим и отсекаются похожие соседи.
    """
    observed_width = word.shape[1]
    scored: list[tuple[float, str]] = []
    for key, title in ORE_TYPES:
        best = None
        for face in TOAST_FACES:
            anchor = render_word(TOAST_ANCHOR, face, 22)
            if anchor is None:
                continue
            for height in TOAST_HEIGHTS:
                candidate = render_word(title, face, height)
                reference = render_word(TOAST_ANCHOR, face, height)
                if candidate is None or reference is None:
                    continue
                expected = candidate.shape[1] * anchor_width / reference.shape[1]
                if abs(expected - observed_width) > TOAST_WIDTH_TOLERANCE:
                    continue
                distance = _shape_distance(word, candidate)
                if best is None or distance < best:
                    best = distance
        if best is not None:
            scored.append((best, key))
    if not scored:
        return None, 1.0
    distance, key = min(scored)
    return (key if distance <= TOAST_ACCEPT_DISTANCE else None), distance


def read_ore_toast(image: np.ndarray) -> ToastReading:
    """Прочитать плашку «Вы собрали … руда!» внизу экрана."""
    if image is None or image.size == 0:
        return ToastReading(False, None, 1.0)
    reference, _scale_x, _scale_y = to_reference(image)
    picked = _ore_word(_toast_words(reference))
    if picked is None:
        return ToastReading(False, None, 1.0)
    word, anchor_width = picked
    key, distance = classify_ore_word(word, anchor_width)
    return ToastReading(True, key, distance)
