"""Разбор кадра стола блэкджека в Majestic.

Модуль ничего не знает об окнах, вводе и статистике: он получает кадр и
отвечает, что сейчас на экране. Жизненный цикл и фокус игры остаются в
``blackjack.py``, а распознавание отсюда проверяется на сохранённых кадрах
без запуска CustomTkinter.

Все области заданы долями клиентской области, поэтому одинаково работают
на 2560x1440 и на других разрешениях 16:9.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..core import resource_path

# Подпись «У вас [N] , У дилера [M]» — фиксированная строка HUD по центру
# экрана, а не надпись в мире: при движении камеры она не уезжает.
HAND_LABEL_RATIO = (0.400, 0.7118, 0.200, 0.0452)
# Две строки правого столбца подсказок. Нижняя — широкая тёмная панель
# («ЭТА СТАВКА» или «ВРЕМЯ»), верхняя отличает ставку от хода: при ставке
# там вторая тёмная панель, во время хода — белая кнопка «СЪЕСТЬ ЗА».
PROMPT_TOP_ROW_RATIO = (0.8574, 0.8715, 0.1328, 0.0382)
PROMPT_BOTTOM_ROW_RATIO = (0.8574, 0.9132, 0.1328, 0.0326)
# Уведомление о результате раздачи. Плашка центрирована по ширине экрана,
# поэтому ищем цветную полоску на её правом крае в полосе правее центра.
TOAST_BAND_RATIO = (0.360, 0.9236, 0.330, 0.0500)
TOAST_ACCENT_X_RATIO = (0.560, 0.680)
TOAST_PLATE_WIDTH_RATIO = 0.130
# Плашка тёмная почти во всю высоту полосы, но текст разрывает её на
# несколько отрезков, поэтому от каждого требуем лишь часть ширины.
TOAST_PLATE_DARK_RATIO = 0.80
TOAST_PLATE_MIN_WIDTH_RATIO = 0.050
# Полоска результата шириной шесть пикселей на 1440p. Пробуем только её
# левый край: дальше вплотную начинается сукно того же зелёного тона.
TOAST_ACCENT_PROBE_RATIO = 0.0016
TOAST_ACCENT_HEIGHT_RATIO = 0.55

GLYPH_WIDTH, GLYPH_HEIGHT = 16, 24
# Высота цифры и скобки в подписи: 19 и 24 пикселя на 1440p.
GLYPH_MIN_HEIGHT_RATIO = 0.0110
GLYPH_MAX_HEIGHT_RATIO = 0.0250
DIGIT_MATCH_THRESHOLD = 0.55
# Между цифрами одного числа зазор в пару пикселей, между «[11]» и «[10]»
# — почти двести. Порог берём с большим запасом в обе стороны.
DIGIT_GROUP_GAP_RATIO = 0.020

_DIGIT_TEMPLATES: list[np.ndarray] | None = None


def _ratio_crop(image: np.ndarray, ratio: tuple[float, float, float, float]) -> np.ndarray:
    height, width = image.shape[:2]
    x, y, region_width, region_height = ratio
    return image[
        max(0, round(height * y)):min(height, round(height * (y + region_height))),
        max(0, round(width * x)):min(width, round(width * (x + region_width))),
    ]


def digit_templates() -> list[np.ndarray] | None:
    """Полоса из десяти нормализованных цифр интерфейса Majestic."""
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        strip = cv2.imread(
            str(resource_path("assets", "vision", "blackjack_digits.png")),
            cv2.IMREAD_GRAYSCALE,
        )
        if strip is None or strip.shape != (GLYPH_HEIGHT, GLYPH_WIDTH * 10):
            return None
        _DIGIT_TEMPLATES = [
            strip[:, digit * GLYPH_WIDTH:(digit + 1) * GLYPH_WIDTH] > 127
            for digit in range(10)
        ]
    return _DIGIT_TEMPLATES


def _text_mask(crop: np.ndarray) -> np.ndarray:
    """Оставить только светлый и обесцвеченный текст интерфейса.

    Кремовая разметка сукна («BLACKJACK PAYS 3 TO 2») бывает такой же яркой,
    но заметно цветнее, поэтому ограничение по насыщенности убирает её и
    оставляет подпись целой.
    """
    channels = crop[:, :, :3].astype(np.int16)
    span = channels.max(axis=2) - channels.min(axis=2)
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.int16)
    return ((gray > 170) & (span < 28)).astype(np.uint8) * 255


def _glyphs(mask: np.ndarray, frame_height: int) -> list[tuple[int, int, int, np.ndarray]]:
    minimum = frame_height * GLYPH_MIN_HEIGHT_RATIO
    maximum = frame_height * GLYPH_MAX_HEIGHT_RATIO
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    found = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if minimum <= height <= maximum and width >= 2 and area >= 8:
            bitmap = (labels[y:y + height, x:x + width] == index).astype(np.uint8) * 255
            found.append((x, width, height, bitmap))
    found.sort(key=lambda glyph: glyph[0])
    return found


def _similarity(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    return int(np.logical_and(first, second).sum()) / union if union else 0.0


def _read_digit(bitmap: np.ndarray, templates: list[np.ndarray]) -> tuple[int, float]:
    normalized = cv2.resize(
        bitmap, (GLYPH_WIDTH, GLYPH_HEIGHT), interpolation=cv2.INTER_AREA
    ) > 127
    scores = [(_similarity(normalized, template), digit) for digit, template in enumerate(templates)]
    best_score, best_digit = max(scores)
    return best_digit, best_score


def hand_totals(frame: np.ndarray | None) -> tuple[int, int] | None:
    """Прочитать «У вас [N] , У дилера [M]» и вернуть пару чисел.

    Возвращает ``None``, пока подписи нет или в ней распознана не ровно
    пара чисел: вслепую нажимать по одной найденной цифре опаснее, чем
    подождать следующий кадр.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    templates = digit_templates()
    if templates is None:
        return None
    crop = _ratio_crop(frame, HAND_LABEL_RATIO)
    if crop.size == 0:
        return None
    height = frame.shape[0]
    groups: list[list[tuple[int, int]]] = []
    gap = max(6, round(frame.shape[1] * DIGIT_GROUP_GAP_RATIO))
    previous_right = None
    for x, width, glyph_height, bitmap in _glyphs(_text_mask(crop), height):
        digit, score = _read_digit(bitmap, templates)
        if score < DIGIT_MATCH_THRESHOLD:
            continue
        if previous_right is None or x - previous_right > gap:
            groups.append([])
        groups[-1].append((digit, x))
        previous_right = x + width
    numbers = [int("".join(str(digit) for digit, _x in group)) for group in groups if group]
    if len(numbers) != 2 or numbers[0] > 40 or numbers[1] > 40:
        return None
    return numbers[0], numbers[1]


def _row_tones(frame: np.ndarray, ratio: tuple[float, float, float, float]) -> tuple[float, float]:
    crop = _ratio_crop(frame, ratio)
    if crop.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray < 70)), float(np.mean(gray > 170))


def table_phase(frame: np.ndarray | None) -> str | None:
    """Определить, что предлагает интерфейс стола прямо сейчас.

    ``bet`` — открыта ставка («РАЗМЕР СТАВКИ» и «ЭТА СТАВКА»),
    ``action`` — идёт ход («ВРЕМЯ» и «Ещё / Достаточно / Дабл»),
    ``wait`` — подсказок нет, стол раздаёт карты,
    ``None`` — интерфейс не опознан, работать вслепую нельзя.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    top_dark, top_bright = _row_tones(frame, PROMPT_TOP_ROW_RATIO)
    bottom_dark, bottom_bright = _row_tones(frame, PROMPT_BOTTOM_ROW_RATIO)
    if bottom_dark >= 0.80:
        if top_dark >= 0.80:
            return "bet"
        if top_bright >= 0.10 and top_dark < 0.60:
            return "action"
        return None
    if bottom_bright >= 0.10 and bottom_dark < 0.60:
        return "wait"
    return None


def _dark_runs(columns: np.ndarray) -> list[tuple[int, int]]:
    """Отрезки подряд идущих отмеченных колонок как пары «первая, последняя»."""
    padded = np.concatenate(([False], columns, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(first), int(stop) - 1) for first, stop in zip(starts, stops)]


def classify_round_toast(frame: np.ndarray | None) -> str | None:
    """Найти уведомление о результате раздачи и вернуть ``win`` или ``loss``.

    Одного цветного пятна мало: зелёное сукно стола и красные фишки дают
    такие же оттенки. Опорой служит тёмная плашка уведомления — цвет читаем
    только у её правого края, где игра рисует полоску результата.

    Искать полоску отдельным пятном нельзя: зелёная полоска победы вплотную
    примыкает к такому же зелёному сукну и слипается с ним в одно широкое
    пятно. Красная так не слипается, поэтому проигрыши считались, а победы
    молча уходили в «раздачи без выигрыша».
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        return None
    band = _ratio_crop(frame, TOAST_BAND_RATIO)
    if band.size == 0:
        return None
    frame_width = frame.shape[1]
    gray = cv2.cvtColor(band[:, :, :3], cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(band[:, :, :3], cv2.COLOR_BGR2HSV)
    masks = {
        "win": cv2.inRange(
            hsv, np.array((35, 90, 90), dtype=np.uint8),
            np.array((95, 255, 255), dtype=np.uint8),
        ),
        "loss": cv2.bitwise_or(
            cv2.inRange(
                hsv, np.array((0, 90, 90), dtype=np.uint8),
                np.array((12, 255, 255), dtype=np.uint8),
            ),
            cv2.inRange(
                hsv, np.array((150, 90, 90), dtype=np.uint8),
                np.array((179, 255, 255), dtype=np.uint8),
            ),
        ),
    }
    plate_width = max(40, round(frame_width * TOAST_PLATE_WIDTH_RATIO))
    minimum_plate = max(40, round(frame_width * TOAST_PLATE_MIN_WIDTH_RATIO))
    probe = max(2, round(frame_width * TOAST_ACCENT_PROBE_RATIO))
    band_left = round(frame_width * TOAST_BAND_RATIO[0])
    lower = round(frame_width * TOAST_ACCENT_X_RATIO[0]) - band_left
    upper = round(frame_width * TOAST_ACCENT_X_RATIO[1]) - band_left
    dark = np.mean(gray < 80, axis=0) >= TOAST_PLATE_DARK_RATIO
    for first, last in _dark_runs(dark):
        if last - first + 1 < minimum_plate:
            continue
        accent = last + 1
        if not lower <= accent <= upper or accent < plate_width:
            continue
        text = float(np.mean(gray[:, accent - plate_width:accent] > 150))
        if not 0.005 <= text <= 0.30:
            continue
        for result, mask in masks.items():
            stripe = mask[:, accent:accent + probe]
            if stripe.size and float(np.mean(stripe > 0)) >= TOAST_ACCENT_HEIGHT_RATIO:
                return result
    return None


class HandReader:
    """Отличает мягкую руку от жёсткой по тому, как рос счёт.

    Игра показывает одно число вместо карт, но во время раздачи это число
    меняется по одной карте, а туз всегда приходит как 11. Этого хватает:

    * первая карта «11» или прибавка ровно на 11 — в руке появился туз,
      который сейчас считается за одиннадцать, то есть рука мягкая;
    * если после добора счёт не вырос, значит туз упал в единицу и рука
      стала жёсткой: мягкие 16 (A+5) плюс десятка снова показывают 16.

    Класс не смотрит в кадр и не шлёт нажатий: он получает уже
    подтверждённый счёт и отвечает, какой строкой стратегии играть.
    """

    ACE = 11

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Начать новую раздачу."""
        self.total = 0
        self.soft = False
        self.awaiting_card = False

    def draw(self) -> None:
        """Запомнить, что мы попросили ещё одну карту."""
        self.awaiting_card = True

    def update(self, total: int, *, settled: bool = False) -> bool:
        """Принять подтверждённый счёт и вернуть «рука сейчас мягкая».

        ``settled`` означает, что стол снова просит ход: карта точно
        доехала, и счёт на экране больше не может быть прошлым.
        """
        if not self.awaiting_card:
            if total > self.total:
                self._take(total)
                self.total = total
            return self.soft
        if total > self.total:
            self._take(total)
            self.total = total
            self.awaiting_card = False
        elif settled:
            # Карта пришла, а счёт не вырос — туз перестал считаться за 11.
            self.soft = False
            self.total = total
            self.awaiting_card = False
        return self.soft

    def _take(self, total: int) -> None:
        if self.total == 0:
            self.soft = total == self.ACE
        elif total - self.total == self.ACE:
            self.soft = True


def decide_move(player: int, dealer: int, *, soft: bool = False) -> str:
    """Базовая стратегия: вернуть ``hit`` или ``stand``.

    Мягкую руку перебрать нельзя, поэтому она играется отдельной строкой:
    до восемнадцати добираем всегда, а на мягких восемнадцати встаём только
    против слабой карты дилера. Пока карта дилера закрыта (``0``), играем по
    правилу самого дилера — добираем до семнадцати.

    Сплита стол не предлагает, а дабл модуль не берёт, поэтому пары и
    удвоения в таблице не нужны.
    """
    if soft:
        if player >= 19:
            return "stand"
        if player == 18:
            return "hit" if dealer <= 0 or dealer >= 9 else "stand"
        return "hit"
    if player >= 17:
        return "stand"
    if player <= 11:
        return "hit"
    if dealer <= 0:
        return "hit"
    if player == 12:
        return "stand" if 4 <= dealer <= 6 else "hit"
    return "stand" if 2 <= dealer <= 6 else "hit"
