"""Разбор кадра покерного стола в GTA.

Модуль ничего не знает об окнах, вводе и статистике: он получает кадр и
отвечает, что сейчас на столе. Такое разделение уже используется парой
``blackjack.py`` / ``blackjack_vision.py`` и позволяет проверять
распознавание на сохранённых кадрах без запуска CustomTkinter.

Все области заданы долями клиентской области. Замеры сняты на 2560x1440;
геометрия и разбор кадров описаны в ``poker-plan/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from ...core import resource_path

RANKS = "23456789TJQKA"
SUITS = "cdhs"

CARD_WIDTH_RATIO = 107 / 2560
CARD_HEIGHT_RATIO = 157 / 1440

# Пять общих карт по центру стола: шаг между слотами 113,4 пикселя.
BOARD_SLOTS: tuple[tuple[float, float], ...] = tuple(
    ((1000 + 113.4 * index) / 2560, 453 / 1440) for index in range(5)
)

# Шесть мест за столом. Карты каждого места лежат на своём постоянном пятне,
# а не там, где сидит игрок: своё место определяется отдельно.
SEAT_SLOTS: tuple[tuple[tuple[float, float], ...], ...] = tuple(
    tuple((x / 2560, y / 1440) for x in xs)
    for xs, y in (
        ((193, 305), 23),    # верх-слева
        ((2151, 2263), 0),   # верх-справа, карта срезана верхним краем экрана
        ((193, 305), 620),   # слева-середина
        ((2151, 2263), 620), # справа-середина
        ((780, 892), 905),   # низ-слева
        ((1567, 1679), 905), # низ-справа
    )
)

# Сброшенная рубашкой карта лежит плашмя и рисуется ниже и мельче лицевой,
# поэтому у неё своя рамка: одна на обе карты места. Замерено по кадрам, у
# верхнего правого места лицевая карта срезана краем экрана, и вывести рамку
# смещением от неё нельзя.
SEAT_BACK_RATIOS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x / 2560, y / 1440, width / 2560, height / 1440)
    for x, y, width, height in (
        (188, 128, 232, 78),
        (2148, 128, 232, 78),
        (188, 728, 232, 78),
        (2148, 728, 232, 78),
        (766, 1012, 238, 82),
        (1553, 1012, 238, 82),
    )
)

# Карта не всегда лежит ровно на номинальном месте: на средних местах она
# уезжает вверх примерно на 50 пикселей, когда в панели появляется строка
# комбинации, а на вскрытии игра приподнимает карты победной комбинации на
# сорок. Поэтому карту ищут в полосе, а не берут жёстким прямоугольником.
CARD_SEARCH_UP_RATIO = 75 / 1440
CARD_SEARCH_DOWN_RATIO = 35 / 1440
CARD_ROW_COVERAGE = 0.6

SEAT_NAMES = (
    "верх-слева", "верх-справа", "слева-середина",
    "справа-середина", "низ-слева", "низ-справа",
)

# Порядок мест по кругу — в нём игра раздаёт блайнды и в нём же ходят. Выведен
# не из симметрии стола, а проверен по записи: на шести разных положениях
# кнопки малый блайнд каждый раз оказывался на следующем месте этого круга, а
# большой — на втором.
RING_ORDER = (0, 1, 3, 5, 4, 2)

# Метка дилера у правого края панели игрока. Рамки замерены по кадрам: у
# средних мест панель уезжает вверх на 49 пикселей, поэтому там рамка высокая.
DEALER_RATIOS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x / 2560, y / 1440, width / 2560, height / 1440)
    for x, y, width, height in (
        (406, 198, 68, 56),    # верх-слева
        (2362, 198, 68, 56),   # верх-справа
        (406, 748, 68, 136),   # слева-середина
        (2366, 748, 68, 136),  # справа-середина
        (992, 1078, 68, 56),   # низ-слева
        (1778, 1078, 68, 56),  # низ-справа
    )
)
# Залитый квадратик метки — это около 690 малиновых пикселей. Порог взят вдвое
# ниже: метку частично закрывает курсор, а спутать её в этой рамке не с чем.
DEALER_INK_FLOOR = 350

# Метка «WIN» у победителя: залитая золотом плашка справа от панели, вплотную
# к её рамке, 59 × 37 пикселей. Держится она того же правого края панели, что
# и метка дилера, и уезжает вместе с ней, поэтому рамки выведены смещением от
# уже замеренных: метка дилера стоит внутри панели, а «WIN» — сразу за ней.
# Рамка взята с запасом, потому что искомое из неё выбирается формой, а не
# положением.
WIN_RATIOS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x + 58 / 2560, y - 8 / 1440, 84 / 2560, height + 34 / 1440)
    for x, y, _width, height in DEALER_RATIOS
)
# Золото на кадре есть и без победителя: сукно расписано жёлтым, и логотип
# «PLAY» лежит ровно под нижним правым местом. Отличается метка тем, что она
# залита целиком, а разметка — тонкие линии в шесть пикселей. Поэтому маска
# сначала съедается эрозией: линии исчезают, плашка остаётся.
WIN_ERODE_RATIO = 9 / 1440
# После эрозии метка даёт около 750 пикселей, а вся жёлтая разметка сукна на
# двадцати четырёх опорных кадрах — ноль. Порог взят вдвое ниже: метку может
# накрыть курсор, а перепутать её после эрозии уже не с чем.
WIN_INK_FLOOR = 300

POT_RATIO = (1133 / 2560, 264 / 1440, 294 / 2560, 118 / 1440)
# Полоса плашек под бордом: на неё игра раскладывает банк по горшкам, когда
# кто-то ушёл в олл-ин на меньшую сумму. Плашки стоят через 240 пикселей и
# висят по центру стола, поэтому рамка широкая: пятерых горшков в записях не
# попадалось, но шесть мест их допускают, и узкая рамка срезала бы крайний.
# «Общий банк» над бордом эти горшки уже включает — полоса не прибавка к
# нему, а его разбор (проверено на записях, см. ``poker-plan/README.md``).
SIDE_POT_ROW_RATIO = (740 / 2560, 645 / 1440, 1080 / 2560, 75 / 1440)
# Текст кнопки хода целиком: и слово («CALL», «CHECK»), и сумма. Рамка
# начинается правее значка «+»: значок нарисован тем же малиновым, что и
# сумма, и в маску попадать не должен.
CALL_TEXT_RATIO = (1930 / 2560, 1258 / 1440, 270 / 2560, 70 / 1440)
CALL_TEXT_SCALE = 2  # мелкие цифры слипаются, поэтому область увеличиваем
QUICK_BET_ROW_RATIO = (2200 / 2560, 1205 / 1440, 340 / 2560, 50 / 1440)
# Кнопка «FOLD» и место под кнопкой повышения. Когда доплата больше стека,
# повышать нечем: игра убирает и ряд быстрых размеров, и ползунок, и саму
# кнопку повышения, оставляя только «ALL IN» и «FOLD». Правая колонка внизу
# при этом пустеет — по ней такой экран и отличается от преактива, где на том
# же месте стоит четвёртая кнопка.
FOLD_BUTTON_RATIO = (1860 / 2560, 1344 / 1440, 340 / 2560, 66 / 1440)
UNDER_RAISE_RATIO = (2200 / 2560, 1344 / 1440, 340 / 2560, 66 / 1440)
BUTTON_LIT_SHARE = 0.5
EMPTY_SLOT_SHARE = 0.2
# Правая кнопка: «BET 50», когда доплачивать нечего, и «RAISE 1 000», когда
# есть. Рамка начинается правее галочки — она нарисована тем же малиновым,
# что и сумма, и в маску попадать не должна.
BET_TEXT_RATIO = (2270 / 2560, 1258 / 1440, 270 / 2560, 70 / 1440)
# Ползунок размера ставки под кнопками. Малиновый бегунок появляется у левого
# края дорожки: пока он там, на кнопке написан минимум, который игра примет.
# В покое бегунок стоит на 5 % дорожки, один шаг сдвигает его ещё на 3 % —
# порог посередине, иначе подвинутый на шаг ползунок сойдёт за нетронутый.
SLIDER_RATIO = (2255 / 2560, 1345 / 1440, 220 / 2560, 50 / 1440)
SLIDER_MIN_SHARE = 0.07

# Ставки, лежащие на сукне. Плашка ставки стоит ближе к центру стола, чем
# карты места, и держится их верхнего края: у карт лицом она выше, у чужой
# рубашки — ниже (рубашка лежит плашмя и дальше от центра), а на средних
# местах вся стопка уезжает вверх, когда в панели появляется строка
# комбинации. Поэтому у средних и нижних мест области высокие — они
# покрывают все положения. Замеры по кадрам: верхние места 318, средние 528,
# 566 и 678, нижние 847 и 959.
#
# У верхних мест области узкие нарочно. Под панелью игрока игра рисует меню
# «ПОКУПКА ЗА ⊙» — и золотая кнопка этого меню ложится ровно в широкую рамку
# места «верх-слева». Помощник принимал её за плашку ставки и либо молчал,
# либо считал чужую ставку своей: доплата выходила нулём там, где на кнопке
# стояло «CALL 250».
BET_RATIOS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x / 2560, y / 1440, width / 2560, height / 1440)
    for x, y, width, height in (
        (240, 296, 110, 60),    # верх-слева
        (2190, 296, 110, 60),   # верх-справа
        (200, 500, 260, 215),   # слева-середина
        (2150, 500, 260, 215),  # справа-середина
        (790, 820, 260, 185),   # низ-слева
        (1580, 820, 260, 185),  # низ-справа
    )
)

# Стек в панели игрока: та же золотая фишка и цифры справа от неё. Панель
# стоит под картами места и на средних местах уезжает вверх вместе с ними —
# там область высокая. Замеры по кадрам: верхние места 254, средние 805 и
# 854, нижние 1136.
STACK_RATIOS: tuple[tuple[float, float, float, float], ...] = tuple(
    (x / 2560, y / 1440, width / 2560, height / 1440)
    for x, y, width, height in (
        (225, 240, 175, 45),    # верх-слева
        (2181, 240, 175, 45),   # верх-справа
        (225, 790, 175, 95),    # слева-середина
        (2185, 790, 175, 95),   # справа-середина
        (811, 1122, 175, 45),   # низ-слева
        (1597, 1122, 175, 45),  # низ-справа
    )
)
# Сумму всегда подписывает золотая фишка ⊙ слева от цифр — и на плашке банка,
# и на плашке ставки. От неё и пляшем: сама плашка по яркости почти не
# отличается от сукна (75 против 66), а фишка видна и на сукне, и на панели.
CHIP_MIN_SIDE_RATIO = 10 / 1440
CHIP_MAX_SIDE_RATIO = 30 / 1440
CHIP_AREA_FLOOR = 60
# Подпись действия в панели игрока написана золотом («Пропустил ход»), и её
# буквы проходят и по цвету, и по кругломерности. Стек ищется рядом с ними,
# поэтому там фишка требуется крупная: настоящая — 17–21 пиксель, буква — 13.
PANEL_CHIP_MIN_SIDE_RATIO = 15 / 1440
AMOUNT_WIDTH_RATIO = 160 / 2560  # сколько места справа от фишки занимают цифры
AMOUNT_SCALE = 2

GLYPH_WIDTH, GLYPH_HEIGHT = 24, 32
DIGIT_WIDTH, DIGIT_HEIGHT = 18, 26

# --- журнал событий ---

# Строка событий в левом нижнем углу: игра сама пишет, кто что сделал, готовым
# текстом и в правильном порядке. Это надёжнее, чем сравнивать кадры между
# собой: при четырёх кадрах в секунду быстрый фолд теряется, а дырявый журнал
# для статистики хуже, чем никакого.
#
# Снизу и по бокам рамка постоянная, а вверх журнал растёт: свёрнутый
# показывает одну строку, развёрнутый — десять. Поэтому рамка взята сразу на
# все десять, а сколько строк на кадре, видно по чернилам.
LOG_RATIO = (56 / 2560, 1000 / 1440, 495 / 2560, 387 / 1440)
# Порог чернил берётся от собственного фона плашки, как у карт — от их
# белизны: плашка бывает то полупрозрачной (яркость 105), то почти белой
# (231), и абсолютный порог годился бы ровно для одной из них.
LOG_INK_SHARE = 0.72
# Темнее этого фона плашки на кадре нет вовсе: сукно даёт 65–75.
LOG_BACKGROUND_FLOOR = 95
LOG_LINE_INK = 3        # столько чернил в ряду, чтобы счесть его строкой
LOG_LINE_MIN = 8        # строка ниже этого — обрывок анимации, а не текст
LOG_WORD_GAP = 5        # пробел между словами шире межбуквенного зазора
# Имя игрока написано малиновым, время и действие — серым. Тот же приём, каким
# на кнопке хода отделяется сумма от слова, только наоборот.
LOG_NAME_INK = 3

WORD_WIDTH, WORD_HEIGHT = 96, 16
# Слова нарезаны из самой игры и лежат в ``assets/vision/poker_words.png`` в
# этом порядке: пропустил, поставил, уровнял, выиграл, передана, Началась,
# Игра, Началась, Игра, сбросил, повысил, Игра, уровнял. Одно слово встречается
# в полосе не по разу нарочно: плашка журнала то светлеет, то гаснет, и буквы
# на ней прорисовываются чуть по-разному.
LOG_ACTIONS = (
    "чек", "ставка", "колл", "выигрыш", "кнопка", "начало", "конец",
    "начало", "конец", "фолд", "рейз", "конец", "колл",
)
# Разные слова похожи друг на друга максимум на 0,48 («поставил» и «повысил»),
# а своё слово узнаётся в среднем на 0,88. Порог посередине.
LOG_MATCH_FLOOR = 0.6
# За какими действиями в строке стоит сумма. «Пропустил ход» числа не несёт, и
# разбирать ради него остаток строки незачем: каждое слово стоит своих
# миллисекунд, а строк на кадре десять.
LOG_AMOUNTS = frozenset({"ставка", "рейз", "выигрыш"})
# Отпечаток имени: та же картинка, растянутая по высоте. Читать буквы не нужно
# — нужно лишь узнавать, что это тот же человек, что в прошлой раздаче.
NAME_WIDTH, NAME_HEIGHT = 128, 16

# Цифры интерфейса примерно в 1/55 высоты экрана. Границы заданы от кадра, а
# не от вырезанной области: одна и та же цифра встречается и на широкой плашке
# банка, и на узкой кнопке колла.
DIGIT_MIN_HEIGHT_RATIO = 0.009
DIGIT_MAX_HEIGHT_RATIO = 0.032

# Ниже этого совпадения знак считается неразобранным: лучше промолчать, чем
# подсказать по выдуманной карте.
CARD_MATCH_FLOOR = 0.55
DIGIT_MATCH_FLOOR = 0.62
# Меньше этого малинового пятна на кнопке — значит суммы там нет вовсе и на
# кнопке написано «CHECK». Отличать «доплаты нет» от «не разобрал сумму»
# обязательно: иначе непрочитанный колл выглядел бы бесплатным чеком.
CALL_INK_FLOOR = 40
# Кнопка под курсором мыши (и заранее выбранный преактив) заливается малиновым
# целиком, а текст на ней становится белым. Малинового на такой кнопке 93 %
# против 1 % у обычной — порог посередине берётся с большим запасом.
CALL_HIGHLIGHT_SHARE = 0.5
# На залитой кнопке слово и сумма написаны одним цветом, и делит их только
# пробел: внутри слова буквы стоят в 2–8 пикселях друг от друга, а перед
# суммой зазор вдвое шире. Доля от высоты знака, крупные экраны и мелкие.
GLYPH_GAP_SHARE = 0.35
# Курсор мыши поверх кнопки — такое же белое пятно, как буквы. Отсекается по
# строке: весь текст кнопки стоит на одной базовой линии, курсор — нет.
GLYPH_BASELINE_SHARE = 0.2

_RANK_TEMPLATES: list[np.ndarray] | None = None
_SUIT_TEMPLATES: list[np.ndarray] | None = None
_DIGIT_TEMPLATES: list[np.ndarray] | None = None
_WORD_TEMPLATES: list[np.ndarray] | None = None


def _load_strip(name: str, width: int, height: int, count: int) -> list[np.ndarray] | None:
    strip = cv2.imread(str(resource_path("assets", "vision", name)), cv2.IMREAD_GRAYSCALE)
    if strip is None or strip.shape != (height, width * count):
        return None
    return [strip[:, index * width:(index + 1) * width] > 127 for index in range(count)]


def rank_templates() -> list[np.ndarray] | None:
    """Тринадцать знаков ранга, вырезанных из самой игры."""
    global _RANK_TEMPLATES
    if _RANK_TEMPLATES is None:
        _RANK_TEMPLATES = _load_strip("poker_ranks.png", GLYPH_WIDTH, GLYPH_HEIGHT, 13)
    return _RANK_TEMPLATES


def suit_templates() -> list[np.ndarray] | None:
    global _SUIT_TEMPLATES
    if _SUIT_TEMPLATES is None:
        _SUIT_TEMPLATES = _load_strip("poker_suits.png", GLYPH_WIDTH, GLYPH_HEIGHT, 4)
    return _SUIT_TEMPLATES


def word_templates() -> list[np.ndarray] | None:
    """Слова журнала событий, нарезанные из самой игры."""
    global _WORD_TEMPLATES
    if _WORD_TEMPLATES is None:
        _WORD_TEMPLATES = _load_strip(
            "poker_words.png", WORD_WIDTH, WORD_HEIGHT, len(LOG_ACTIONS)
        )
    return _WORD_TEMPLATES


def digit_templates() -> list[np.ndarray] | None:
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        _DIGIT_TEMPLATES = _load_strip("poker_digits.png", DIGIT_WIDTH, DIGIT_HEIGHT, 10)
    return _DIGIT_TEMPLATES


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Совпадение площадей двух масок: общее к суммарному.

    Мерка одна на весь модуль: ею узнаются знаки карт, цифры, слова строки
    событий — и отпечатки имён в ``players.py``. Точное равенство не годится
    нигде: картинка на экране то светлеет, то гаснет, и одинаковыми две маски
    не бывают почти никогда.
    """
    union = int(np.logical_or(first, second).sum())
    return int(np.logical_and(first, second).sum()) / union if union else 0.0


def _crop(frame: np.ndarray, ratio: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, region_width, region_height = ratio
    left, top = round(width * x), round(height * y)
    return frame[
        max(0, top):min(height, top + round(height * region_height)),
        max(0, left):min(width, left + round(width * region_width)),
    ]


def card_crop(frame: np.ndarray, slot: tuple[float, float]) -> np.ndarray:
    """Прямоугольник ровно на номинальном месте, без поиска."""
    return _crop(frame, (slot[0], slot[1], CARD_WIDTH_RATIO, CARD_HEIGHT_RATIO))


def _colour_span(crop: np.ndarray) -> np.ndarray:
    """Разброс каналов пикселя: у белой бумаги он мал, у сукна и рубашки нет.

    Считается вычитанием готовых карт цвета, а не по оси массива: разбор
    двенадцати карт мест идёт на каждом кадре, и разница выходит вчетверо.
    """
    blue, green, red = cv2.split(crop[:, :, :3])
    return cv2.subtract(
        cv2.max(cv2.max(blue, green), red), cv2.min(cv2.min(blue, green), red)
    )


def _paper_mask(crop: np.ndarray) -> np.ndarray:
    """Пиксели бумаги: светлые и без цвета."""
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    return (gray > 120) & (_colour_span(crop) < 60)


def _paper_rows(strip: np.ndarray) -> np.ndarray:
    """Строки полосы, целиком занятые бумагой карты."""
    return np.mean(_paper_mask(strip), axis=1) > CARD_ROW_COVERAGE


def card_at(frame: np.ndarray, slot: tuple[float, float]) -> np.ndarray:
    """Карта места с поправкой на сдвиг по вертикали.

    Ищем сплошную полосу бумаги вокруг номинального места и берём карту от её
    верхнего края. Без этого своя рука пропадала, как только открывался борд:
    карта уезжает вверх, а прямоугольник остаётся.
    """
    height = frame.shape[0]
    card_height = round(height * CARD_HEIGHT_RATIO)
    nominal = round(height * slot[1])
    top = max(0, nominal - round(height * CARD_SEARCH_UP_RATIO))
    bottom = min(height, nominal + card_height + round(height * CARD_SEARCH_DOWN_RATIO))
    left = round(frame.shape[1] * slot[0])
    strip = frame[top:bottom, left:left + round(frame.shape[1] * CARD_WIDTH_RATIO)]
    if strip.size == 0:
        return card_crop(frame, slot)

    rows = _paper_rows(strip)
    best_start, best_length, start = None, 0, None
    for index, filled in enumerate(rows):
        if filled and start is None:
            start = index
        elif not filled and start is not None:
            if index - start > best_length:
                best_start, best_length = start, index - start
            start = None
    if start is not None and len(rows) - start > best_length:
        best_start, best_length = start, len(rows) - start
    if best_start is None or best_length < card_height * 0.6:
        return card_crop(frame, slot)
    found = top + best_start
    return frame[found:min(height, found + card_height), left:left + strip.shape[1]]


def is_face_down(card: np.ndarray) -> bool:
    """Рубашка: фиолетовый ромбовый узор поверх белой карты."""
    if card.size == 0:
        return False
    hsv = cv2.cvtColor(card[:, :, :3], cv2.COLOR_BGR2HSV)
    purple = np.mean((hsv[:, :, 0] > 115) & (hsv[:, :, 0] < 165) & (hsv[:, :, 1] > 60))
    return bool(purple > 0.08)


def is_face_up(card: np.ndarray) -> bool:
    """Лицевая карта: светлое поле без узора рубашки.

    Порог светлого нарочно низкий: на вскрытии игра приглушает карты, не
    вошедшие в победную комбинацию, и они темнее сукна.
    """
    if card.size == 0 or is_face_down(card):
        return False
    return bool(np.mean(_paper_mask(card)) > 0.5)


def _ink_parts(card: np.ndarray) -> list[tuple[np.ndarray, bool]] | None:
    """Знак ранга и знак масти, каждый обрезан по чернилам и нормализован.

    Порог чернил считается от собственной белизны карты, а не абсолютный:
    иначе приглушённая на вскрытии карта не читается вовсе.
    """
    gray = cv2.cvtColor(card[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper = float(np.percentile(gray, 85))
    ink = (gray < paper * 0.72).astype(np.uint8)
    height, width = ink.shape
    margin_x, margin_y = max(1, width // 13), max(1, height // 31)
    ink[:, :margin_x] = 0
    ink[:, width - margin_x:] = 0
    ink[:margin_y, :] = 0
    ink[height - margin_y:, :] = 0
    halves = (
        ink * (np.arange(height)[:, None] < height * 0.52),
        ink * (np.arange(height)[:, None] >= height * 0.48),
    )
    parts = []
    for half in halves:
        rows, columns = np.nonzero(half.astype(np.uint8))
        if rows.size < 30:
            return None
        patch = half[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
        piece = card[rows.min():rows.max() + 1, columns.min():columns.max() + 1, :3].astype(np.int16)
        red = float(np.mean(piece[:, :, 2] - piece[:, :, 0])) > 25
        parts.append((
            cv2.resize(
                patch.astype(np.uint8) * 255, (GLYPH_WIDTH, GLYPH_HEIGHT),
                interpolation=cv2.INTER_AREA,
            ) > 127,
            red,
        ))
    return parts


def read_card(card: np.ndarray) -> str | None:
    """Лицевая карта → «Ah», «Td»; рубашка, пустой слот или мусор → None."""
    ranks, suits = rank_templates(), suit_templates()
    if ranks is None or suits is None or not is_face_up(card):
        return None
    parts = _ink_parts(card)
    if parts is None:
        return None
    (rank_mask, _rank_red), (suit_mask, suit_red) = parts
    rank_score, rank_index = max(
        (similarity(rank_mask, template), index) for index, template in enumerate(ranks)
    )
    pool = (1, 2) if suit_red else (0, 3)  # d, h против c, s
    suit_score, suit_index = max((similarity(suit_mask, suits[index]), index) for index in pool)
    if min(rank_score, suit_score) < CARD_MATCH_FLOOR:
        return None
    return RANKS[rank_index] + SUITS[suit_index]


def board_cards(frame: np.ndarray) -> tuple[str, ...]:
    """Открытые общие карты слева направо; закрытые слоты пропускаются."""
    found = []
    for slot in BOARD_SLOTS:
        card = read_card(card_at(frame, slot))
        if card is None:
            break
        found.append(card)
    return tuple(found)


def seat_cards(frame: np.ndarray, seat: int) -> tuple[str, ...]:
    """Открытые карты одного места; у закрытых карт ответ пустой."""
    found = [read_card(card_at(frame, slot)) for slot in SEAT_SLOTS[seat]]
    return tuple(card for card in found if card is not None)


def seats_with_cards(frame: np.ndarray) -> tuple[int, ...]:
    """Места, у которых сейчас есть карты — лицом или рубашкой.

    Это и есть число участников раздачи: сбросивший карты игрок остаётся за
    столом, но его карты со стола убирают.
    """
    seats = []
    for index, slots in enumerate(SEAT_SLOTS):
        face_up = any(is_face_up(card_at(frame, slot)) for slot in slots)
        if face_up or is_face_down(_crop(frame, SEAT_BACK_RATIOS[index])):
            seats.append(index)
    return tuple(seats)


def hero_seat(frame: np.ndarray) -> int | None:
    """Своё место: единственное, где карты лежат лицом.

    На вскрытии лицом лежат несколько рук, и однозначного ответа нет — тогда
    ответ ``None``, а вызывающий держит место, найденное раньше.
    """
    face_up = [
        index for index, slots in enumerate(SEAT_SLOTS)
        if any(is_face_up(card_at(frame, slot)) for slot in slots)
    ]
    return face_up[0] if len(face_up) == 1 else None


def _light_text_mask(crop: np.ndarray) -> np.ndarray:
    """Белые цифры интерфейса без золотой фишки и цветной разметки сукна."""
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    return ((gray > 170) & (_colour_span(crop) < 40)).astype(np.uint8)


def _lit_text_mask(crop: np.ndarray) -> np.ndarray:
    """Надпись на залитой кнопке: порог между заливкой и текстом ищется сам.

    Жёсткий порог белизны тут не годится. Сумма на залитой кнопке написана
    бледнее слова, и на «CALL 250» цифры рассыпались в крошку — помощник
    решал, что суммы нет вовсе, и советовал бесплатный чек вместо колла.
    """
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    _level, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.uint8)


def _red_text_mask(crop: np.ndarray) -> np.ndarray:
    """Малиновая сумма на белой кнопке: «CALL 500» пишет её отдельным цветом.

    Порог высокий нарочно: на мягком пороге соседние цифры слипаются краями в
    одно пятно и число прочитать уже нельзя.
    """
    channels = crop[:, :, :3].astype(np.int16)
    return (
        (channels[:, :, 2] - channels[:, :, 1] > 70)
        & (channels[:, :, 2] - channels[:, :, 0] > 70)
    ).astype(np.uint8)


class _Glyph(NamedTuple):
    """Один знак текста: где стоит и как выглядит."""

    left: int
    right: int
    bottom: int
    height: int
    bitmap: np.ndarray


def _spots(mask: np.ndarray, area_floor: int = 30) -> list[_Glyph]:
    """Все заметные пятна маски слева направо, без разбора, знак это или нет."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    found = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if width < 2 or height < 2 or area < area_floor:
            continue
        found.append(_Glyph(
            left=x, right=x + width, bottom=y + height, height=height,
            bitmap=(labels[y:y + height, x:x + width] == index),
        ))
    return sorted(found, key=lambda glyph: glyph.left)


def _is_glyph_sized(glyph: _Glyph, frame_height: int, scale: int) -> bool:
    """Пятно ростом со знак интерфейса: не пылинка и не картинка."""
    return (
        frame_height * DIGIT_MIN_HEIGHT_RATIO * scale
        <= glyph.height
        <= frame_height * DIGIT_MAX_HEIGHT_RATIO * scale
        and glyph.right - glyph.left >= 3
    )


def _glyphs(mask: np.ndarray, frame_height: int, scale: int = 1) -> list[_Glyph]:
    """Знаки маски слева направо; слишком мелкое и крупное отброшено."""
    return [
        glyph for glyph in _spots(mask)
        if _is_glyph_sized(glyph, frame_height, scale)
    ]


def _same_line(glyphs: list[_Glyph]) -> list[_Glyph]:
    """Знаки одной строки: общая базовая линия и общий рост.

    Число всегда написано одним кеглем в одну строку, поэтому всё, что рядом
    оказалось крупнее или сидит ниже, — соседний значок интерфейса, а не
    цифра. Ради этого правило и появилось: круглая кнопка «$» из меню покупки
    стоит вплотную к ставке места «верх-слева», и ставка 250 читалась как
    2501 — тихо неверное число вместо честного ответа.
    """
    if not glyphs:
        return []
    baseline = float(np.median([glyph.bottom for glyph in glyphs]))
    tall = float(np.median([glyph.height for glyph in glyphs]))
    return [
        glyph for glyph in glyphs
        if abs(glyph.bottom - baseline) <= tall * GLYPH_BASELINE_SHARE
        and abs(glyph.height - tall) <= tall * GLYPH_BASELINE_SHARE
    ]


def _digits_of(glyphs: list[_Glyph]) -> int | None:
    """Знаки подряд как одно число; хоть один знак не цифра — ``None``.

    Слипшиеся краями цифры выглядят одним широким пятном. Раньше такое пятно
    молча выбрасывалось, и «244 000» превращалось в «2 000» — тихо неверное
    число хуже честного молчания, поэтому теперь оно тоже даёт ``None``.
    """
    templates = digit_templates()
    if templates is None or not glyphs:
        return None
    digits = []
    for glyph in glyphs:
        if glyph.right - glyph.left > glyph.height * 1.2:
            return None
        normalized = cv2.resize(
            glyph.bitmap.astype(np.uint8) * 255, (DIGIT_WIDTH, DIGIT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ) > 127
        score, digit = max(
            (similarity(normalized, template), value)
            for value, template in enumerate(templates)
        )
        if score < DIGIT_MATCH_FLOOR:
            return None
        digits.append(str(digit))
    return int("".join(digits))


def _read_digits(
    mask: np.ndarray, lower_half: bool, frame_height: int, scale: int = 1
) -> int | None:
    height = mask.shape[0]
    glyphs = [
        glyph for glyph in _glyphs(mask, frame_height, scale)
        if not (lower_half and glyph.bottom - glyph.height / 2 < height * 0.45)
    ]
    return _digits_of(_same_line(glyphs))


def _gold_chips(
    area: np.ndarray, frame_height: int, min_side_ratio: float = CHIP_MIN_SIDE_RATIO
) -> list[tuple[int, int, int, int]]:
    """Значки ⊙ в области: слева направо, каждый как ``(x, y, ширина, высота)``.

    Фишка нарисована золотым и почти круглая — этого хватает, чтобы отличить
    её и от сукна, и от жёлтой разметки стола, которая заметно крупнее.
    """
    hsv = cv2.cvtColor(area[:, :, :3], cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (hue > 15) & (hue < 35) & (saturation > 110) & (value > 130)
    ).astype(np.uint8)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    smallest = frame_height * min_side_ratio
    tallest = frame_height * CHIP_MAX_SIDE_RATIO
    rows, columns = mask.shape
    found = []
    for index in range(1, count):
        x, y, width, height, chip_area = (int(item) for item in stats[index])
        if x == 0 or y == 0 or x + width == columns or y + height == rows:
            # Пятно, срезанное краем области, мерить нечем: жёлтая разметка
            # сукна за границей рамки обрезается ровно до размеров фишки, и
            # ставка соседнего места отсчитывалась от неё, а не от плашки.
            continue
        if not (smallest <= width <= tallest and smallest <= height <= tallest):
            continue
        if abs(width - height) > tallest * 0.2 or chip_area < CHIP_AREA_FLOOR:
            continue
        found.append((x, y, width, height))
    return sorted(found)


def _amount_right_of_chip(
    frame: np.ndarray,
    chip: tuple[int, int, int, int],
    left: int,
    top: int,
    stop: int,
) -> int | None:
    """Число справа от фишки ⊙, стоящей в кадре на ``left + x``, ``top + y``.

    ``stop`` — где обрываются цифры: у одинокой плашки это просто длина
    числа, а в полосе горшков — фишка следующей плашки, иначе её сумма
    приклеилась бы к предыдущей.
    """
    height = frame.shape[0]
    x, y, chip_width, chip_height = chip
    margin = round(height * 6 / 1440)
    digits = frame[
        max(0, top + y - margin):top + y + chip_height + margin,
        left + x + chip_width + 2:stop,
    ]
    if digits.size == 0:
        return None
    bigger = cv2.resize(
        digits, None, fx=AMOUNT_SCALE, fy=AMOUNT_SCALE, interpolation=cv2.INTER_CUBIC
    )
    return _read_digits(
        _light_text_mask(bigger), lower_half=False,
        frame_height=height, scale=AMOUNT_SCALE,
    )


def _amount_beside_chip(
    frame: np.ndarray,
    ratio: tuple[float, float, float, float],
    min_side_ratio: float = CHIP_MIN_SIDE_RATIO,
) -> int | None:
    """Сумма рядом со значком ⊙ в области; ``0`` — значка там нет вовсе.

    Так читаются и банк, и ставки на сукне, и стеки: цифры всегда стоят
    справа от фишки, а сама плашка под ними по яркости почти сливается с
    сукном. Фишку ищем в области, а цифры отсчитываем уже от неё по кадру:
    рамка отвечает за то, где лежит плашка, а не за то, какой длины число.
    """
    height, width = frame.shape[:2]
    area = _crop(frame, ratio)
    if area.size == 0:
        return 0
    chips = _gold_chips(area, height, min_side_ratio)
    if not chips:
        return 0
    left, top = round(width * ratio[0]), round(height * ratio[1])
    start = left + chips[0][0] + chips[0][2] + 2
    return _amount_right_of_chip(
        frame, chips[0], left, top, start + round(width * AMOUNT_WIDTH_RATIO)
    )


def pot_size(frame: np.ndarray) -> int | None:
    """Банк с плашки «Общий банк».

    ``0`` — плашки нет вовсе, банк ещё не собран: на префлопе блайнды лежат
    ставками на сукне. ``None`` — плашка есть, а число не сложилось.
    """
    if not _gold_chips(_crop(frame, POT_RATIO), frame.shape[0]):
        return 0
    return _read_digits(
        _light_text_mask(_crop(frame, POT_RATIO)), lower_half=True,
        frame_height=frame.shape[0],
    )


def side_pots(frame: np.ndarray) -> tuple[int | None, ...]:
    """Горшки на полосе под бордом слева направо; ``()`` — полосы нет вовсе.

    Полоса появляется, когда кто-то ушёл в олл-ин на меньшую сумму, чем
    поставили остальные, и банк раскладывается по горшкам. ``None`` на месте
    горшка — плашка есть, а число не сложилось.

    Прибавлять эти суммы к «Общему банку» нельзя: плашка над бордом их уже
    включает. На пяти случаях из двух записей сумма полосы сошлась с ней
    знак в знак — 2 000 + 5 000 = 7 000, 121 500 + 57 000 + 65 500 = 244 000.
    Полоса нужна другим: пока анимация сгребает фишки, плашка отстаёт от
    полосы, и по расхождению видно, что банку сейчас верить нельзя.
    """
    height, width = frame.shape[:2]
    area = _crop(frame, SIDE_POT_ROW_RATIO)
    if area.size == 0:
        return ()
    chips = _gold_chips(area, height)
    if not chips:
        return ()
    left = round(width * SIDE_POT_ROW_RATIO[0])
    top = round(height * SIDE_POT_ROW_RATIO[1])
    limit = left + area.shape[1]
    room = round(width * AMOUNT_WIDTH_RATIO)
    found = []
    for index, chip in enumerate(chips):
        stop = min(limit, left + chip[0] + chip[2] + 2 + room)
        if index + 1 < len(chips):
            # Соседняя плашка стоит в 240 пикселях, и длина числа до неё не
            # дотягивается. Но упереться в её фишку всё равно надо: иначе
            # горшок в шесть знаков склеился бы со следующим.
            stop = min(stop, left + chips[index + 1][0])
        found.append(_amount_right_of_chip(frame, chip, left, top, stop))
    return tuple(found)


def bets_on_felt(frame: np.ndarray) -> tuple[int | None, ...]:
    """Ставки этой улицы перед каждым из шести мест.

    ``0`` — плашки перед местом нет, ``None`` — плашка есть, но число не
    разобралось. Плашка «Общий банк» этих ставок не считает: на её месте
    17 250 перед игроками лежали ещё 2 500 и 16 500. Без них шансы банка
    выходят завышенными, и помощник требует от руки больше, чем нужно.
    """
    return tuple(_amount_beside_chip(frame, ratio) for ratio in BET_RATIOS)


def stack_size(frame: np.ndarray, seat: int | None) -> int | None:
    """Стек игрока на месте ``seat``: цифры справа от фишки в его панели.

    ``0`` — панели на кадре нет или фишек у игрока не осталось; ставить в
    обоих случаях всё равно нечего. ``None`` — панель есть, а число не
    сложилось.
    """
    if seat is None:
        return None
    return _amount_beside_chip(frame, STACK_RATIOS[seat], PANEL_CHIP_MIN_SIDE_RATIO)


def _button_amount(
    spots: list[_Glyph], frame_height: int, scale: int, lit: bool
) -> int | None:
    """Сумма из знаков на кнопке хода; ``0`` — суммы на кнопке нет.

    На обычной кнопке малиновым написана только сумма, слово чёрное — значит
    все знаки маски и есть число. На залитой кнопке белым написано и слово, и
    сумма, и делит их зазор вдвое шире межбуквенного: слово стоит первым, всё
    остальное — сумма. «CHECK» остаётся одним куском, и это честный ноль.

    Куски после слова склеиваются обратно в одно число: разряды игра тоже
    делит пробелом, и по одному последнему куску «CALL 1 250» читалось как
    250 — впятеро дешевле настоящего колла.
    """
    glyphs = [spot for spot in spots if _is_glyph_sized(spot, frame_height, scale)]
    if not glyphs:
        # На залитой кнопке всегда есть хотя бы слово, а на белой малиновое
        # пятно уже проверено вызывающим. Пусто здесь значит одно: знаки
        # слиплись с курсором в пятно не по росту, и читать нечего.
        return None
    kept = _same_line(glyphs)
    if not kept:
        return None
    baseline = float(np.median([glyph.bottom for glyph in kept]))
    tall = float(np.median([glyph.height for glyph in kept]))
    if lit:
        amount = _trailing_digits(_words(kept, tall))
    else:
        amount = _digits_of(kept)

    # Курсор мыши ниже строки строке не мешает — а вот курсор поверх цифры
    # съедает её целиком: «CALL 250» читалось как «2», и помощник уверенно
    # советовал колл во сто раз дешевле настоящего. Обрезки съеденной цифры
    # остаются лежать в строке, по ним такое чтение и отменяется.
    top = baseline - tall * 1.3
    for spot in spots:
        if spot in kept or spot.bottom - spot.height >= baseline or spot.bottom <= top:
            continue  # знак строки или пятно вне её
        if spot.left < kept[-1].right and spot.right > kept[0].left:
            return None  # съел знак посреди надписи
        gap = spot.left - kept[-1].right
        if 0 <= gap <= tall and (amount or gap > tall * GLYPH_GAP_SHARE):
            # Либо съел последнюю цифру суммы, либо стоит за пробелом — там,
            # где сумма и пишется. Буква того же слова примыкает вплотную.
            return None
    return amount


def _trailing_digits(words: list[list[_Glyph]]) -> int | None:
    """Число из последних кусков надписи; ``0`` — цифр на кнопке нет.

    Раньше правило было «слово первое, всё остальное — сумма», и на кнопке
    `ALL IN 4 500` оно спотыкалось: первым куском уходило `ALL`, а `IN` в
    число не складывалось, и помощник молчал ровно на том решении, где на кону
    весь стек. Идём с конца, пока куски читаются цифрами: `CALL 1 250` даёт
    1250, `ALL IN 4 500` — 4500, а `CHECK` — честный ноль.
    """
    tail: list[_Glyph] = []
    for word in reversed(words):
        if _digits_of(word) is None:
            break
        tail = word + tail
    if not tail:
        return 0
    return _digits_of(tail)


def _words(glyphs: list[_Glyph], tall: float) -> list[list[_Glyph]]:
    """Знаки, разбитые по зазорам: внутри слова буквы стоят плотнее."""
    words: list[list[_Glyph]] = [[glyphs[0]]]
    for previous, glyph in zip(glyphs, glyphs[1:]):
        if glyph.left - previous.right > tall * GLYPH_GAP_SHARE:
            words.append([])
        words[-1].append(glyph)
    return words


def _amount_on_button(
    frame: np.ndarray, ratio: tuple[float, float, float, float]
) -> int | None:
    """Сумма, написанная на кнопке хода; ``0`` — суммы на ней нет.

    Кнопка под курсором мыши заливается малиновым, а текст на ней белеет.
    Раньше заливка читалась как одно сплошное пятно суммы, и помощник молчал
    ровно в тот миг, когда игрок тянулся к кнопке.
    """
    crop = _crop(frame, ratio)
    if crop.size == 0:
        return None
    bigger = cv2.resize(
        crop, None, fx=CALL_TEXT_SCALE, fy=CALL_TEXT_SCALE, interpolation=cv2.INTER_CUBIC
    )
    red = _red_text_mask(bigger)
    lit = bool(np.mean(red) > CALL_HIGHLIGHT_SHARE)
    if lit:
        mask = _lit_text_mask(bigger)
    else:
        if int(red.sum()) < CALL_INK_FLOOR:
            return 0  # чёрное слово на белой кнопке и ни одной малиновой цифры
        mask = red
    return _button_amount(_spots(mask), frame.shape[0], CALL_TEXT_SCALE, lit)


def call_amount(frame: np.ndarray) -> int | None:
    """Сумма к уравниванию с левой кнопки хода.

    ``0`` — доплачивать нечего: на кнопке написано «CHECK». ``None`` — сумма
    там есть, но не разобралась; советовать по такому кадру нельзя. Ответ
    имеет смысл только на своём ходу: сумму «CALL 500» показывает и преактив.
    """
    return _amount_on_button(frame, CALL_TEXT_RATIO)


def bet_offer(frame: np.ndarray) -> int | None:
    """Сумма на правой кнопке: «BET 50» без доплаты и «RAISE 1 000» с ней.

    Это то, что игра поставит по нажатию, — сколько сейчас выставлено
    ползунком. ``0`` — кнопки нет или на ней нет суммы.
    """
    return _amount_on_button(frame, BET_TEXT_RATIO)


def bet_slider_at_minimum(frame: np.ndarray) -> bool:
    """Стоит ли бегунок размера ставки у левого края дорожки."""
    crop = _crop(frame, SLIDER_RATIO)
    if crop.size == 0:
        return False
    columns = np.flatnonzero(_red_text_mask(crop).any(axis=0))
    if columns.size == 0:
        return False
    return bool(columns.min() <= crop.shape[1] * SLIDER_MIN_SHARE)


def minimum_bet(frame: np.ndarray) -> int | None:
    """Наименьшая ставка, которую примет игра, — пока ползунок не тронут.

    Ниже минимума ползунок не опускается, поэтому у нетронутого ползунка на
    кнопке написан как раз он. Бегунок сдвинули — судить о минимуме уже не по
    чему, и ответ ``None``: советовать размер меньше разрешённого хуже, чем
    не называть границу вовсе.
    """
    if not bet_slider_at_minimum(frame):
        return None
    return bet_offer(frame) or None


def sane_minimum(
    minimum: int | None, *, my_bet: int = 0, to_call: int | None = 0,
    stack: int | None = None,
) -> int | None:
    """Минимум с кнопки, если он вообще возможен по правилам игры.

    Курсор, севший на цифру, ломает не только колл: на кадре 155505 записи
    `8.mp4` на кнопке написано `RAISE 4 000`, а прочиталось 654 000. Дальше
    беда шла не от самого числа, а от того, что минимум работает полом для
    всех четырёх быстрых кнопок: они разом подтянулись к нему, обрезались
    стеком и схлопнулись в одну — `ALL IN`. Совет на этом кадре звучал «жми
    ALL IN, поднять до 84 000» при банке 12 000 и паре королей с четвёркой,
    а кадром раньше и кадром позже — «жми MIN, 4 000».

    Проверяем двумя правилами, которые в покере верны всегда, а не подобраны
    на глаз:

    * больше своих фишек в банк не положить — минимум не бывает выше стека;
    * минимальное повышение — это чужая ставка плюс её же последний шаг,
      значит вдвое больше старшей ставки на столе оно быть не может.

    Не прошло — ``None``: без минимума помощник обойдётся, а с выдуманным
    отправит игрока в олл-ин.
    """
    if minimum is None or minimum <= 0:
        return minimum
    if stack is not None and minimum > my_bet + stack:
        return None
    if to_call and minimum > 2 * (my_bet + to_call):
        return None
    return minimum


def due_from_bets(bets: tuple[int | None, ...], seat: int | None) -> int | None:
    """Доплата, посчитанная по ставкам на сукне: старшая ставка минус своя.

    Второе мнение о кнопке хода. Курсор, севший ровно в пробел «CALL 50»,
    оставляет от надписи одно слово, и кнопка выглядит как «CHECK» — а
    бесплатный чек вместо колла худшая из ошибок помощника. Ставки же лежат
    на сукне отдельно от курсора.
    """
    if seat is None or any(bet is None for bet in bets):
        return None
    return max(bets) - (bets[seat] or 0)


def _lit_share(frame: np.ndarray, ratio: tuple[float, float, float, float]) -> float:
    """Какая доля рамки занята светлой заливкой кнопки."""
    crop = _crop(frame, ratio)
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 225))


def all_in_only(frame: np.ndarray) -> bool:
    """Нарисован ли экран без повышения: только «ALL IN» и «FOLD».

    Игра показывает его, когда доплата больше стека: повышать не на что.
    Отличается тем, что правая нижняя клетка пуста — там нет ни ползунка, как
    на обычном ходу, ни четвёртой кнопки, как у преактива. Одной кнопки
    «FOLD» мало: она горит на всех трёх экранах.

    Раньше такой экран не считался своим ходом вовсе, и помощник молчал на
    семи решениях из сорока в записи ``6.mp4`` — на всех, где на кону стоял
    весь стек.
    """
    return (
        _lit_share(frame, FOLD_BUTTON_RATIO) > BUTTON_LIT_SHARE
        and _lit_share(frame, UNDER_RAISE_RATIO) < EMPTY_SLOT_SHARE
        and _lit_share(frame, QUICK_BET_ROW_RATIO) < EMPTY_SLOT_SHARE
    )


def is_my_turn(frame: np.ndarray) -> bool:
    """Ход мой: либо есть ряд быстрых размеров, либо экран «ALL IN / FOLD».

    По цвету кнопок судить нельзя: заранее выбранный преактив подсвечивается
    так же, а сумму «CALL 500» преактив тоже показывает. Ряд
    «MIN / 3 BB / BANK / ALL IN» и ползунок появляются только на своём ходу —
    но когда доплата больше стека, игра не рисует и их.
    """
    return _lit_share(frame, QUICK_BET_ROW_RATIO) > 0.55 or all_in_only(frame)


def dealer_seat(frame: np.ndarray) -> int | None:
    """Место с меткой ``D``; ``None`` — метки на кадре не видно.

    Метка — залитый малиновым квадратик со скруглением у правого края панели
    игрока, 29 × 30 пикселей. Без неё нельзя назвать позицию, а без позиции
    префлоп-таблица бессмысленна: одна и та же рука с кнопки открывается, а с
    ранней позиции сбрасывается.

    Рамки замерены по кадрам, а не выведены из панели: у средних мест панель
    уезжает вверх на 49 пикселей вместе с картами, и метка едет с ней.
    """
    best: tuple[int, int] | None = None
    for seat, ratio in enumerate(DEALER_RATIOS):
        crop = _crop(frame, ratio)
        if crop.size == 0:
            continue
        blue, green, red = (crop[:, :, index].astype(int) for index in range(3))
        ink = int(
            np.count_nonzero(
                (red > 140) & (red - green > 90) & (red - blue > 70)
            )
        )
        if ink >= DEALER_INK_FLOOR and (best is None or ink > best[1]):
            best = (seat, ink)
    return None if best is None else best[0]


def _solid_gold(crop: np.ndarray, frame_height: int) -> int:
    """Сколько в области залитого золота — без тонких линий жёлтой разметки.

    Маска съедается эрозией на толщину этих линий: разметка сукна и рамка
    победной панели исчезают целиком, а плашка «WIN» остаётся почти вся.
    Размер ядра берётся от кадра, а не в пикселях: на другом разрешении и
    линии тоньше.
    """
    hsv = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (
        (hue > 15) & (hue < 35) & (saturation > 110) & (value > 130)
    ).astype(np.uint8)
    side = max(3, int(round(frame_height * WIN_ERODE_RATIO)) | 1)
    return int(cv2.erode(mask, np.ones((side, side), np.uint8)).sum())


def winning_seats(frame: np.ndarray) -> tuple[int, ...]:
    """Места с меткой «WIN»; пусто — банк ещё не отдан.

    Метку игра рисует, когда банк уходит игроку, и мест в ответе может быть
    несколько: разделённый банк отмечается у каждого, кому он достался.

    Советовать по ней нечего — раздача уже сыграна, — зато по ней считаются
    сами раздачи: сколько их было и чем кончились. Всё остальное в журнале
    держится на этом счёте.
    """
    return tuple(
        seat for seat, ratio in enumerate(WIN_RATIOS)
        if (crop := _crop(frame, ratio)).size
        and _solid_gold(crop, frame.shape[0]) >= WIN_INK_FLOOR
    )


# --- журнал событий ---


def _fit(mask: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Кусок текста в общий холст: масштаб по высоте, ширина едет за ней.

    Растягивать по ширине нельзя: у слова она и есть главная примета — «ход»
    от «пропустил» отличается прежде всего длиной. Поэтому высота приводится к
    общей, ширина считается по ней, а остаток холста остаётся пустым.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if not rows.size or not columns.size:
        return None
    trimmed = mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    wide = max(1, min(width, int(round(trimmed.shape[1] * height / trimmed.shape[0]))))
    resized = cv2.resize(
        trimmed.astype(np.uint8) * 255, (wide, height), interpolation=cv2.INTER_AREA
    ) > 127
    canvas = np.zeros((height, width), dtype=bool)
    canvas[:, :wide] = resized
    return canvas


def _runs(
    values: np.ndarray, floor: int, gap: int = 1, least: int = 1
) -> list[tuple[int, int]]:
    """Отрезки, где значение держится выше порога: строки в плашке, слова в строке."""
    spans: list[tuple[int, int]] = []
    start, empty = None, 0
    for index, value in enumerate(values):
        if value > floor:
            if start is None:
                start = index
            empty = 0
        elif start is not None:
            empty += 1
            if empty >= gap:
                end = index - empty + 1
                if end - start >= least:
                    spans.append((start, end))
                start = None
    if start is not None and len(values) - start >= least:
        spans.append((start, len(values)))
    return spans


def _log_ink(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Чернила журнала и малиновые имена в нём; ``None`` — журнала не видно.

    Сперва ищется сама плашка. Рамка взята на десять строк, а журнал бывает
    свёрнут в одну — и тогда девять десятых рамки занимает стол, который
    темнее любых чернил. Плашка — это светлые ряды, доходящие до низа рамки:
    вверх она растёт, а низ у неё всегда на месте.

    Порог чернил берётся уже от фона самой плашки: она бывает то
    полупрозрачной (яркость 105), то почти белой (231). Тот же приём, что и с
    картами, где порог берётся от белизны бумаги.
    """
    crop = _crop(frame, LOG_RATIO)
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY).astype(int)
    # Ряд считается плашкой, когда светлая в нём хотя бы половина: длинная
    # строка текста закрывает до трети ряда, и по одной медиане плашка на ней
    # обрывалась — журнал начинался с середины.
    light = (gray >= LOG_BACKGROUND_FLOOR).mean(axis=1) >= 0.5
    if not light[-1]:
        return None
    dark = np.flatnonzero(~light)
    top = int(dark[-1]) + 1 if dark.size else 0
    band = gray[top:]
    if band.shape[0] < LOG_LINE_MIN:
        return None
    level = float(np.percentile(band, 85))
    ink = band < level * LOG_INK_SHARE
    colours = crop[top:, :, :3].astype(int)
    blue, green, red = (colours[:, :, index] for index in range(3))
    return ink, ink & (red - green > 40) & (red - blue > 40)


def _log_time(word: np.ndarray) -> int | None:
    """Отметка «[11:58:03]» в секундах от полуночи; ``None`` — не прочиталась.

    Скобки и двоеточия отсеиваются сами: ни одна цифра на них не похожа.
    Остаться должно ровно шесть знаков — по этому времени строки журнала potом
    и различаются между кадрами.
    """
    # Скобки отсеиваются ростом: они выше цифр, а двоеточия ниже. Без этого
    # закрывающая скобка читалась как единица, знаков выходило семь, и время
    # не складывалось вовсе.
    spots = _same_line(_spots(word.astype(np.uint8), area_floor=6))
    if len(spots) != 6:
        # Шесть знаков — часы, минуты и секунды. Больше или меньше значит, что
        # знаки слиплись или рассыпались, и гадать по ним нечего.
        return None
    digits = [value for spot in spots if (value := _digits_of([spot])) is not None]
    if len(digits) != 6:
        return None
    hours = digits[0] * 10 + digits[1]
    minutes = digits[2] * 10 + digits[3]
    seconds = digits[4] * 10 + digits[5]
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _log_action(word: np.ndarray) -> str:
    """Что означает первое слово после имени; пусто — слово незнакомое.

    Незнакомое слово — это не беда, а обычное дело: игрок вошёл за стол,
    вышел, купил фишек. Такую строку журнал пропускает молча, вместо того
    чтобы гадать.
    """
    templates = word_templates()
    shape = _fit(word, WORD_WIDTH, WORD_HEIGHT)
    if templates is None or shape is None:
        return ""
    score, action = max(
        (similarity(shape, template), LOG_ACTIONS[index])
        for index, template in enumerate(templates)
    )
    return action if score >= LOG_MATCH_FLOOR else ""


@dataclass(frozen=True, eq=False)
class LogEvent:
    """Строка журнала: кто, что сделал и на сколько.

    ``name`` — не имя, а его отпечаток: картинка малиновых букв, приведённая к
    общему размеру. Читать имя незачем — достаточно узнавать, что это тот же
    человек, что в раздаче номер сорок семь. Распознавание букв переменной
    ширины было бы отдельным большим проектом, а отпечаток сравнивается
    совпадением площадей, как эталоны карт.
    """

    action: str
    amount: int | None = None
    at: int | None = None
    name: np.ndarray | None = None


def event_log(frame: np.ndarray) -> tuple[LogEvent, ...]:
    """Разобранные строки журнала событий, от старой к новой.

    Свёрнутый журнал показывает одну строку, развёрнутый — десять; строки в
    ответе идут в том же порядке, в каком их пишет игра.
    """
    parsed = _log_ink(frame)
    if parsed is None:
        return ()
    ink, crimson = parsed
    events = []
    for top, bottom in _runs(ink.sum(axis=1), LOG_LINE_INK, 1, LOG_LINE_MIN):
        line, red = ink[top:bottom], crimson[top:bottom]
        words = _runs(line.sum(axis=0), 0, LOG_WORD_GAP)
        if len(words) < 2:
            continue
        crimsons = [red[:, left:right].sum() > LOG_NAME_INK for left, right in words]
        # Имя стоит сразу после времени и только там. Малиновым игра пишет ещё
        # и название комбинации в конце строки — принять его за имя нельзя.
        name = None
        first = 1
        if crimsons[1]:
            while first + 1 < len(words) and crimsons[first + 1]:
                first += 1
            name = _fit(
                line[:, words[1][0]:words[first][1]], NAME_WIDTH, NAME_HEIGHT
            )
            first += 1
        if first >= len(words):
            continue
        action = _log_action(line[:, words[first][0]:words[first][1]])
        if not action:
            continue
        # Сумма — первый подряд идущий кусок цифр после слова действия. Кусков
        # может быть несколько: разряды игра делит пробелом, и «2 500» — это
        # два слова, а не два числа. Ищем её только там, где она бывает:
        # «пропустил ход» числа не несёт, а разбор каждого слова не бесплатный.
        amount, digits = None, []
        for left, right in (words[first + 1:] if action in LOG_AMOUNTS else ()):
            value = _digits_of(_spots(line[:, left:right].astype(np.uint8), area_floor=6))
            if value is None:
                if digits:
                    break
                continue
            digits.append(str(value))
        if digits:
            amount = int("".join(digits))
        if action == "ставка" and amount is None:
            # «поставил малый блайнд» — это не ставка по своей воле, а долг
            # перед раздачей, и в статистику агрессии он не идёт.
            action = "блайнд"
        events.append(LogEvent(
            action=action, amount=amount,
            at=_log_time(line[:, words[0][0]:words[0][1]]), name=name,
        ))
    return tuple(events)


def seats_in_order(live_seats, dealer: int | None) -> tuple[int, ...]:
    """Живые места по кругу, начиная со следующего за кнопкой.

    Это и порядок блайндов, и порядок хода: малый блайнд ставит первое место
    после кнопки, большой — второе. Когда игроков двое, «второе» заворачивается
    обратно на кнопку — поэтому за столом на двоих большой блайнд ставит сам
    дилер. Отдельного правила для этого не нужно, общее уже даёт верный ответ:
    проверено по записи на шести разных положениях кнопки.
    """
    if dealer is None:
        return ()
    live = set(live_seats)
    start = RING_ORDER.index(dealer)
    wheel = RING_ORDER[start + 1:] + RING_ORDER[:start + 1]
    return tuple(seat for seat in wheel if seat in live)


def seat_positions(live_seats, dealer: int | None) -> dict[int, str]:
    """Место → позиция: ``SB``, ``BB``, ``BTN``, ``CO``, ``MP``, ``UTG``.

    Блайнды называются по блайндам, остальные — по расстоянию назад от кнопки.
    За столом на двоих кнопка и есть большой блайнд, и имя блайнда старше:
    решение там принимается против уже поставленных денег.
    """
    order = seats_in_order(live_seats, dealer)
    if len(order) < 2:
        return {}
    names: dict[int, str] = {}
    # Кнопка идёт последней в круге ходов, поэтому считаем назад от конца.
    for step, seat in enumerate(reversed(order)):
        names[seat] = ("BTN", "CO", "MP")[step] if step < 3 else "UTG"
    names[order[0]] = "SB"
    names[order[1]] = "BB"
    return names


def blinds_from_bets(
    bets: tuple[int | None, ...], live_seats, dealer: int | None, board=()
) -> tuple[int, int] | None:
    """Малый и большой блайнд по сукну — пока никто не успел повысить.

    Берём только чистое начало раздачи: борд пуст, ставки стоят ровно у двух
    мест, старшая вдвое больше младшей, и лежат они там, где положено по
    кругу. Любое отклонение — значит кто-то уже сходил, и по такому кадру
    размер блайнда не определить. Ошибиться тут нельзя: глубина стека в
    блайндах решает, какой из двух режимов включить.

    Про борд — не лишняя строчка, а разбор живой ошибки. После флопа ставка и
    ответное повышение сплошь и рядом складываются в ту же картинку: на кадре
    122205 записи `8.mp4` на борде `7♠ A♥ K♦` лежали 7 000 и 3 500, больше
    ничьих ставок не было, и блайнды прочитались как 3 500 / 7 000 при
    настоящих 250 / 500. Дальше глубина стека выходила 4 ставки вместо
    шестидесяти, префлоп уходил в режим «олл-ин или пас», и помощник советовал
    ва-банк на ровном месте — три раздачи подряд, пока не попался чистый старт.
    Блайнды на столе видны только до флопа, и после него им взяться неоткуда.
    """
    if board:
        return None
    order = seats_in_order(live_seats, dealer)
    if len(order) < 2 or any(bet is None for bet in bets):
        return None
    small, big = bets[order[0]], bets[order[1]]
    if not small or not big or small * 2 != big:
        return None
    if any(bets[seat] for seat in order[2:]):
        return None
    return int(small), int(big)


@dataclass(frozen=True)
class TableState:
    """Что удалось прочитать с одного кадра. Ничего не додумывает."""

    board: tuple[str, ...]
    hole: tuple[str, ...]
    hero_seat: int | None
    players: int
    pot: int | None
    bets: tuple[int | None, ...]
    to_call: int | None
    my_turn: bool
    showdown: bool
    stack: int | None = None
    min_bet: int | None = None
    dealer: int | None = None
    blinds: tuple[int, int] | None = None
    live_seats: tuple[int, ...] = ()
    # Доплата больше стека: игра убрала кнопку повышения, и советовать рейз
    # некуда — остались только «ALL IN» и «FOLD».
    all_in_only: bool = False
    # Чужая ставка целиком, без оглядки на наш стек. ``to_call`` — сколько мы
    # заплатим, а это — сколько он поставил, и числа расходятся, когда фишек
    # у нас меньше. Первое нужно для шансов банка, второе — чтобы прочитать
    # его диапазон: ставка в банк и ставка в четверть банка говорят разное.
    faced_bet: int | None = None
    # Горшки с полосы под бордом. Не прибавка к банку — его разбор: плашка
    # «Общий банк» их уже включает. Второй счёт того же числа, и заодно
    # признак, что за столом есть олл-ин на меньшую сумму.
    side_pots: tuple[int | None, ...] = ()
    # Места с меткой «WIN»: банк отдан им. Советовать тут нечего, зато по этой
    # метке журнал считает раздачи и их исход.
    winners: tuple[int, ...] = ()
    # Что каждое место показало лицом. Своя рука лежит так всю раздачу, чужие
    # — только на вскрытии, и это единственный случай, когда чужие карты
    # вообще видны. Пусто там, где место карт не показывало.
    shown: tuple[tuple[str, ...], ...] = ()
    # Разобранные строки журнала событий, от старой к новой. На кадре их
    # видно десять — это окно, а не поток: одни и те же строки приходят снова
    # и снова, пока журнал не прокрутится.
    events: tuple[LogEvent, ...] = ()

    @property
    def position(self) -> str | None:
        """Своя позиция за столом: ``BTN``, ``SB``, ``BB`` и так далее."""
        if self.hero_seat is None:
            return None
        return seat_positions(self.live_seats, self.dealer).get(self.hero_seat)

    @property
    def my_bet(self) -> int | None:
        """Сколько своих денег уже стоит на этой улице."""
        if self.hero_seat is None:
            return None
        return self.bets[self.hero_seat]

    @property
    def money_in_play(self) -> int | None:
        """Всё, что уже поставлено: плашка банка плюс ставки на сукне.

        Именно от этого числа считаются шансы банка. ``None`` — что-то из
        слагаемых не прочиталось, и складывать нечего: заниженный банк тихо
        превращает выгодный колл в фолд.
        """
        if self.pot is None or any(bet is None for bet in self.bets):
            return None
        if self.split_pot and self.pot != sum(self.side_pots):
            # Плашка банка и полоса горшков считают одно и то же, и в покое
            # сходятся знак в знак. Расходятся они, только пока плашка
            # догоняет полосу анимацией: на кадре записи ``6.mp4`` она
            # показывала 9 842 при настоящих 13 000. Такой банк занижен на
            # четверть, а по нему считаются шансы банка.
            return None
        return self.pot + sum(bet for bet in self.bets if bet)

    @property
    def split_pot(self) -> bool:
        """Банк разложен по горшкам: за столом есть олл-ин на меньшую сумму."""
        return bool(self.side_pots) and all(pot is not None for pot in self.side_pots)


def table_state(frame: np.ndarray, known_seat: int | None = None) -> TableState:
    """Собрать состояние стола из кадра.

    ``known_seat`` — место, найденное на прошлых кадрах: на вскрытии лицом
    лежит несколько рук, и своё место по кадру уже не отличить.
    """
    # Карты мест вырезаются один раз на кадр: их двенадцать, и поиск полосой
    # для каждой стоит дороже всего остального разбора вместе взятого.
    seat_cards_found = [[card_at(frame, slot) for slot in slots] for slots in SEAT_SLOTS]
    face_up = [
        index for index, cards in enumerate(seat_cards_found)
        if any(is_face_up(card) for card in cards)
    ]
    seats = [
        index for index in range(len(SEAT_SLOTS))
        if index in face_up or is_face_down(_crop(frame, SEAT_BACK_RATIOS[index]))
    ]
    showdown = len(face_up) > 1
    winners = winning_seats(frame)
    # Пока горит метка «WIN», банк ещё отдают, а единственные открытые карты
    # на столе — карты победителя. Своими их считать нельзя: на этих кадрах
    # помощник пересаживался к соседу, показывал его руку своей и записывал
    # её в журнал отдельной раздачей — в записи `8.mp4` так родились две
    # раздачи-двойника с одним бордом.
    seat = face_up[0] if len(face_up) == 1 and not winners else known_seat
    hole = ()
    if seat is not None:
        read = (read_card(card) for card in seat_cards_found[seat])
        hole = tuple(card for card in read if card is not None)
    # Чужие карты видны только на вскрытии, и только там их и разбираем:
    # каждая карта стоит своих миллисекунд, а на обычном кадре у соперников
    # лежат рубашки.
    shown = []
    for index in range(len(SEAT_SLOTS)):
        if index == seat:
            shown.append(hole)
        elif showdown and index in face_up:
            read = (read_card(card) for card in seat_cards_found[index])
            shown.append(tuple(card for card in read if card is not None))
        else:
            shown.append(())
    bets = bets_on_felt(frame)
    stack = stack_size(frame, seat)
    shove_screen = all_in_only(frame)
    pots = side_pots(frame)
    split = bool(pots) and all(pot is not None for pot in pots)
    # Кнопка и ставки считают одно и то же двумя путями. Расходятся — значит
    # что-то прочитано неверно, и честнее промолчать, чем выбрать наугад.
    button, by_bets = call_amount(frame), due_from_bets(bets, seat)
    faced = by_bets
    if shove_screen and by_bets is not None and stack:
        # На экране без повышения кнопка называет не всю чужую ставку, а то,
        # что реально уйдёт из стека: больше своих фишек в банк не положить.
        # Без этой поправки два счёта расходятся и экран молчит.
        by_bets = min(by_bets, stack)
    to_call = button if button is not None else by_bets
    if button is not None and by_bets is not None and button != by_bets:
        to_call = None
    if split:
        # Пока банк разложен по горшкам, ставки на сукне доплату не считают:
        # перед ушедшим в олл-ин лежит вся его ставка, а уравнивать её не
        # надо — лишнее уже сложено в горшок. На кадре ``side-pots-in-hand``
        # у соперников по 1 500 против нашей 500, а игра пишет «CHECK».
        # Второго счёта тут нет, и кнопка остаётся одна: не прочиталась —
        # молчим, как молчали бы при расхождении.
        to_call, faced = button, None
    dealer = dealer_seat(frame)
    board = board_cards(frame)
    return TableState(
        board=board,
        hole=hole,
        hero_seat=seat,
        players=len(seats),
        pot=pot_size(frame),
        bets=bets,
        to_call=to_call,
        my_turn=is_my_turn(frame),
        showdown=showdown,
        stack=stack,
        all_in_only=shove_screen,
        min_bet=sane_minimum(
            minimum_bet(frame), my_bet=(bets[seat] or 0) if seat is not None else 0,
            to_call=to_call, stack=stack,
        ),
        dealer=dealer,
        faced_bet=faced,
        blinds=blinds_from_bets(bets, seats, dealer, board),
        live_seats=tuple(seats),
        side_pots=pots,
        winners=winners,
        shown=tuple(shown),
        events=event_log(frame),
    )
