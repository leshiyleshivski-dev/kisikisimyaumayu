"""Разбор кадра лесопилки в GTA: полоса рубки, стол с бревном и ветки на нём.

Модуль не создаёт окон и не отправляет ввод: он получает кадр и отвечает, что
сейчас на экране. Так же живут ``miner/vision.py``, ``blackjack_vision.py`` и
начинка ``poker/`` — распознавание проверяется на сохранённых кадрах без
запуска CustomTkinter.

Области заданы долями клиентской области, а размеры ядер и целей — в пикселях
эталонного кадра 2560x1440. Кадр другого разрешения приводится к эталону в
``to_reference()``, поэтому 1920x1080 считается теми же числами, а координаты
целей возвращаются обратно в систему исходного кадра.

Замеры сняты по записи лесопилки от 27.08.2026 (пять деревьев, пять столов) и
разобраны в ``tests/fixtures/lumberjack/README.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np


REFERENCE_WIDTH = 2560
REFERENCE_HEIGHT = 1440

# Полоса «Рубка дерева» в правом нижнем углу. Это тот же виджет игры, что и
# «Добыча руды» у шахтёра, и стоит он на том же месте: замерен как 226x8 в
# (2280,1334) на десятой секунде записи.
CHOPPING_BAR_RATIO = (0.86, 0.915, 0.13, 0.025)

# Где вообще живёт стол: ковёр занимает середину экрана, ветки торчат вверх за
# его дальний край. Рамка взята с запасом внутрь от краёв кадра — в неё не
# должны попадать ни зелёный текст чата слева вверху, ни зелень миникарты.
TABLE_ZONE = (0.24, 0.22, 0.54, 0.50)
# Ковёр стола — искусственный газон, и разводит его с живой зеленью леса
# насыщенность, а не оттенок. У ковра S=246 в медиане и днём, и ночью; у листвы
# ивы, что свисает над камерой, — 111 при 178 в девяносто пятом перцентиле, у
# сухой травы под ногами 50, у ночного подлеска и того меньше.
#
# Прежний порог стоял на S=70 и держался только потому, что запись была одна и
# ночная. Днём под него попадает весь лес: доля «газона» в зоне стола не
# опускалась ниже 0,15 даже на прогулке, помощник считал стол открытым всё
# время и щёлкал по листве — это и есть та «рубка воздуха», из-за которой
# правился этот разбор.
MAT = ((30, 170, 40), (70, 255, 255))
# Доля ковра в зоне стола: 0,148-0,152 на дневной записи, 0,180-0,188 на
# ночной, и не больше 0,006 в лесу без стола. Порог посередине этой пропасти.
MAT_SHARE = 0.06
# Кусок ковра мельче этого — блик на листе, а не половина ковра.
MAT_PIECE_MIN_AREA = 8_000
MAT_MIN_AREA = 25_000
# Бревно поперёк больше 250 пикселей и занимает половину ковра. Дырка мельче
# двадцати тысяч — не бревно, а тень или прореха в газоне.
LOG_MIN_AREA = 20_000

# Радиус кружка, которым бревно отделяется от веток. Ствол поперёк больше 250
# пикселей и кружок в себя вмещает, ветка — меньше сорока и не вмещает.
# Размыкание считается не ядром, а двумя преобразованиями расстояния: тело —
# это всё, что ближе радиуса к центрам вписанных кружков. Так оно выходит
# круглым (ядро-прямоугольник съедало углы наискось и вместе с ними основания
# веток) и стоит те же миллисекунды.
TWIG_RADIUS = 20.0
# Отросток мельче этого — крапинка коры, крупнее — целый сук вместе с куском
# ствола, разнятые не по делу.
TWIG_AREA = (80, 20_000)
# Толщина — наибольший вписанный в отросток кружок. Снизу порог отсекает
# проволочную ногу лампы (2,3 пикселя на ночной записи), сверху — куски самого
# ствола, которые размыкание не дожевало.
TWIG_THICKNESS = (3.0, 26.0)
# Вынос — насколько далеко отросток уходит от тела бревна. Он заменил прежнюю
# вытянутость, и не ради красоты: вытянутость мерила рамку куска, а кусок у
# основания ветки обрезан телом, и у толстого сука она падала ниже, чем у
# кромки бревна. Вынос меряет то самое, что и значит «торчит»: у веток записей
# 6-80 пикселей, у ноги лампы, попавшей в силуэт целиком, — 210-272.
TWIG_REACH = (6.0, 140.0)
# И глубина: насколько отросток уходит внутрь ковра от его края. Кромка самого
# ковра — тёмная земляная обшивка вдоль края — попадает в оболочку клином и
# читается веткой: клин длинный, тонкий и бурый, все прочие пороги он проходит.
# Но лежит он вплотную к краю (глубина 11-18), а ветки растут из бревна на
# середине ковра (41-195). Порог посередине.
TWIG_DEPTH = 25.0
# Ветка бурая. Тень бревна на ковре в дырку тоже попадает — она темнее порога
# ковра, — но оттенок у неё остаётся зелёным: на записях у теней hue 39-56, у
# веток 8-24. Разводит их именно оттенок, а не яркость.
TWIG_HUE_MAX = 30
# И ветка отражает свет — но куда меньше, чем казалось по ночной записи. По
# обеим записям кандидаты делятся на две кучи: 28-35 и 68 и выше, между ними
# пусто. Нижняя куча — ветки в тени бревна, и прежний порог 55 выбрасывал их все,
# а это каждая пятая цель. Порог опущен под обе кучи; он остался страховкой
# на случай дырки, в которой нет предмета вовсе, а не признаком ветки. Берётся
# он по девяностому процентилю, а не по максимуму: одинокий блик не спасает.
TWIG_LIT = 25
# Ветка поперёк не больше сорока пикселей: цели ближе этого — куски одного
# сука, а не две ветки.
TWIG_MERGE_DISTANCE = 46
# Толстый отросток называется суком, тонкий — прутиком. Разделены они не ради
# слов: сук виден и с первого кадра, а прутик надо разглядывать, и в статусе
# помощника это единственная подсказка, за чем он сейчас гоняется.
TWIG_STOUT_THICKNESS = 8.0

# --- гребёнка ---------------------------------------------------------------
# Половина веток лежит вдоль бревна и за его очертания почти не выходит: по
# размеченной записи центр ветки стоит в 0-15 пикселях от контура, а у каждой
# четвёртой — на 50-75 внутрь. Отростком такую не взять, цветом тоже: у ветки
# hue 8-16 при S 96-148, у коры hue 8-12 при S 121-181, кучи налезают друг на
# друга. Зато промах по бревну не стоит ничего — игра на него не отвечает.
#
# Поэтому то, чего зрение не разобрало, помощник добирает частой гребёнкой по
# самому бревну. Шаг 42 пикселя взят по размеру веток (27-200 поперёк): на
# размеченных столах цели и гребёнка вместе накрывают 25 веток из 26.
COMB_STEP = 42
# Отступ от края бревна: у самого контура работают отростки, а гребёнка нужна
# внутри.
COMB_MARGIN = 12

# Стрелка мыши цвета не имеет вовсе: белое пятно 10x15 с тёмной обводкой.
# Замерено на кадре 31:06 ночной записи. Ни ковром, ни деревом она не
# является, и в силуэт бревна ей попадать незачем — клик по ней уходит туда,
# где мышь и так уже стоит.
POINTER_ACHROMATIC = ((0, 0, 175), (180, 55, 255))


class BranchTarget(NamedTuple):
    """Ветка на бревне: куда нажимать и насколько она заметна."""

    x: int
    y: int
    score: float
    kind: str


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


def _zone(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Кусок кадра со столом.

    Разбор идёт по нему, а не по всему кадру, и это не экономия ради экономии:
    в зоне пикселей втрое меньше, и столько же работы у морфологии. Один и тот
    же стол разбирается за 18 миллисекунд по зоне и за 49 по всему кадру.
    """
    return _crop(image, TABLE_ZONE)


# ------------------------------------------------------------- полоса рубки


def _chopping_bar_box(reference: np.ndarray) -> tuple[int, int, int, int] | None:
    crop, left, top = _crop(reference, CHOPPING_BAR_RATIO)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 80, 55), (115, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((3, 7), dtype=np.uint8))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(blue)
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        # Настоящая дорожка — 226x8. Синеватый просвет между листьями такой
        # длины не даёт, и на ночной прогулке полоса не мерещится.
        if (
            width >= 150
            and 3 <= height <= 16
            and width / max(1, height) >= 10.0
            and area >= 600
            and area / max(1, width * height) >= 0.55
        ):
            return left + x, top + y, width, height
    return None


def chopping_bar_visible(image: np.ndarray) -> bool:
    """Идёт ли рубка: игра сама рисует полосу, пока дерево принимает удары."""
    if image is None or image.size == 0:
        return False
    reference, _scale_x, _scale_y = to_reference(image)
    return _chopping_bar_box(reference) is not None


def chopping_bar_fill(image: np.ndarray) -> float | None:
    """Насколько полоса заполнена, или ``None``, если её нет.

    Заполненная часть светлая, пустая — тёмно-бирюзовая, и порог посередине
    разводит их. Каждый засчитанный удар двигает полосу на десятую часть: по
    записи ступеньки равны 0,086-0,110, а промежуток между ними — 1,00-1,20 с
    при медиане 1,10. По этому шагу помощник и ловит ритм, который задаёт
    игра, вместо стрельбы по таймеру.
    """
    if image is None or image.size == 0:
        return None
    reference, _scale_x, _scale_y = to_reference(image)
    box = _chopping_bar_box(reference)
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


# -------------------------------------------------------------------- стол


def log_table_visible(image: np.ndarray) -> bool:
    """Открыт ли стол с бревном — по ковру в середине экрана.

    Ковёр здесь ищется по насыщенности, а не по «зелёному вообще». Живая
    зелень леса такой насыщенности не даёт ни при каком свете, и порог 0,06
    лежит между 0,006 у леса без стола и 0,148 у самого бедного стола.
    """
    if image is None or image.size == 0:
        return False
    reference, _scale_x, _scale_y = to_reference(image)
    sample, _left, _top = _crop(reference, TABLE_ZONE)
    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    mat = cv2.inRange(hsv, *MAT)
    return float(np.count_nonzero(mat)) / max(1, mat.size) >= MAT_SHARE


def _mat_mask(hsv: np.ndarray) -> np.ndarray:
    mat = cv2.inRange(hsv, *MAT)
    # Смыкание залечивает просветы между травинками. Наружу маска от него не
    # растёт, поэтому очертания ковра остаются теми же.
    return cv2.morphologyEx(mat, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8))


def _mat_hull(mat: np.ndarray) -> np.ndarray | None:
    """Ковёр целиком, вместе с бревном поверх него.

    Просто «самый большой зелёный кусок» ковром не является: бревно лежит
    поперёк и режет ковёр надвое, а на трёх столах ночной записи из пяти
    большей оказывалась дальняя половина.

    Половины сводит обратно выпуклая оболочка. Ковёр — плоский
    четырёхугольник, и оболочка его зелёных кусков и есть он сам: за углы она
    не выходит, а бревно между половинами накрывает целиком. Прежде половины
    сводило смыкание ядром шире бревна, и на ночной записи это сходило с рук.
    Днём не сошло: ядро в 261 пиксель раздувает наискось скошенные края, и
    оболочка стола вылезала в лес на добрую сотню пикселей.
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mat)
    if count < 2:
        return None
    keep = [
        component for component in range(1, count)
        if int(stats[component][4]) >= MAT_PIECE_MIN_AREA
    ]
    if not keep or max(int(stats[component][4]) for component in keep) < MAT_MIN_AREA:
        return None
    pieces = np.isin(labels, keep).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(
        pieces, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    hull = np.zeros_like(mat)
    cv2.drawContours(
        hull, [cv2.convexHull(np.vstack(contours))], -1, 255, cv2.FILLED,
    )
    return hull


def _pointer_free(hsv: np.ndarray) -> np.ndarray:
    """Всё, кроме стрелки мыши.

    Стрелка — белое пятно без цвета, и дыркой в ковре она становится наравне с
    веткой. Клик по ней не делает ничего: мышь и так уже там. Гасится она по
    цвету, а не по координате от системы — ветка, накрытая стрелкой, обязана
    остаться видной по тому, что из-под неё торчит.
    """
    return cv2.bitwise_not(cv2.inRange(hsv, *POINTER_ACHROMATIC))


def _log_mask(hsv: np.ndarray, mat: np.ndarray, hull: np.ndarray) -> np.ndarray | None:
    """Бревно вместе с ветками — дырка в ковре.

    Тёплой маски здесь больше нет, и это разница между двумя записями. Ночью
    лес вокруг стола холодный и тёмный, и «всё тёплое рядом» означало ветку.
    Днём вокруг стола сухая трава, стволы и палая листва — тёплое всё, и та же
    маска приводила бревно за руку в лес: силуэт разрастался на пол-экрана, а
    ветками читались травинки.

    Платой идёт ветка, целиком легшая выше дальнего края ковра. Но ветка
    растёт из бревна, а бревно лежит на ковре, и над ковром проходит хотя бы её
    основание — целится помощник всё равно в самое толстое место, то есть
    ближе к основанию.
    """
    holes = cv2.morphologyEx(
        cv2.bitwise_and(hull, cv2.bitwise_not(mat)),
        cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8),
    )
    holes = cv2.bitwise_and(holes, _pointer_free(hsv))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(holes)
    if count < 2:
        return None
    largest = max(range(1, count), key=lambda component: stats[component][4])
    if int(stats[largest][4]) < LOG_MIN_AREA:
        return None
    return (labels == largest).astype(np.uint8) * 255


def _twigs(log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Отростки силуэта и две карты, по которым они меряются.

    Тело бревна — объединение вписанных в него кружков радиуса ``TWIG_RADIUS``.
    Считается оно двумя преобразованиями расстояния: первое даёт толщину в
    каждой точке (и центры кружков — там, где толщина не меньше радиуса),
    второе — расстояние до ближайшего такого центра. Ближе радиуса к центру —
    тело, дальше — отросток.

    Так размыкание выходит круглым, а стоит по-прежнему миллисекунды. Ядром
    его считать нельзя: прямоугольник дёшев, но съедает наискось и вместе с
    углами забирает основания веток, а круглое ядро того же размера считается
    тридцать две миллисекунды вместо полутора.
    """
    thickness = cv2.distanceTransform(log, cv2.DIST_L2, 5)
    core = (thickness >= TWIG_RADIUS).astype(np.uint8)
    if not core.any():
        return None
    reach = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
    body = ((reach <= TWIG_RADIUS) & (log > 0)).astype(np.uint8) * 255
    twigs = cv2.morphologyEx(
        cv2.bitwise_and(log, cv2.bitwise_not(body)),
        cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8),
    )
    return twigs, thickness, reach


def _merge_nearby(
    targets: list[tuple[int, int, float, str]],
) -> list[tuple[int, int, float, str]]:
    """Оставить от одной ветки одну цель.

    Сук с развилкой размыкание режет на два-три куска, и помощник тратил бы на
    него три клика вместо одного — а стол живёт по медиане три секунды.
    """
    kept: list[tuple[int, int, float, str]] = []
    for x, y, score, kind in targets:
        if any(
            (x - other_x) ** 2 + (y - other_y) ** 2 <= TWIG_MERGE_DISTANCE ** 2
            for other_x, other_y, _score, _kind in kept
        ):
            continue
        kept.append((x, y, score, kind))
    return kept


def find_branch_targets(image: np.ndarray) -> list[BranchTarget]:
    """Ветки на бревне, самые торчащие первыми.

    Заметность здесь — вынос: насколько далеко отросток уходит от тела бревна.
    Сук, торчащий на полсотни пикселей, виден с первого кадра и при любом
    освещении, а прутик у самого ствола приходится разглядывать. Стол живёт три
    секунды, и начинать с того, что видно наверняка, дешевле, чем гоняться за
    сомнительным.

    Целится помощник в самое толстое место отростка, а не в середину рамки:
    точка берётся как центр наибольшего вписанного кружка и лежит внутри ветки
    по определению. Центр тяжести у развилки попадает в воздух между сучьями,
    и клик уходит в ковёр.
    """
    if image is None or image.size == 0:
        return []
    reference, scale_x, scale_y = to_reference(image)
    zone, zone_left, zone_top = _zone(reference)
    if zone.size == 0:
        return []
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mat = _mat_mask(hsv)
    hull = _mat_hull(mat)
    if hull is None:
        return []
    log = _log_mask(hsv, mat, hull)
    if log is None:
        return []
    grown = _twigs(log)
    if grown is None:
        return []
    twigs, thickness, reach = grown
    # Глубина внутрь ковра: ею отсеивается его собственная кромка.
    depth = cv2.distanceTransform(hull, cv2.DIST_L2, 5)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(twigs)
    found: list[tuple[int, int, float, str]] = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if not TWIG_AREA[0] <= area <= TWIG_AREA[1]:
            continue
        piece = labels[y:y + height, x:x + width] == component
        inside = thickness[y:y + height, x:x + width].copy()
        inside[~piece] = 0.0
        thick = float(inside.max())
        if not TWIG_THICKNESS[0] <= thick <= TWIG_THICKNESS[1]:
            continue
        stands_out = float(reach[y:y + height, x:x + width][piece].max()) - TWIG_RADIUS
        if not TWIG_REACH[0] <= stands_out <= TWIG_REACH[1]:
            continue
        if float(depth[y:y + height, x:x + width][piece].max()) < TWIG_DEPTH:
            continue
        patch = hsv[y:y + height, x:x + width]
        if float(np.median(patch[:, :, 0][piece])) > TWIG_HUE_MAX:
            continue
        if float(np.percentile(patch[:, :, 2][piece], 90)) < TWIG_LIT:
            continue
        local_y, local_x = np.unravel_index(int(np.argmax(inside)), inside.shape)
        found.append((
            zone_left + x + int(local_x), zone_top + y + int(local_y), stands_out,
            "сук" if thick >= TWIG_STOUT_THICKNESS else "прутик",
        ))
    found.sort(key=lambda target: -target[2])
    return [
        BranchTarget(round(x * scale_x), round(y * scale_y), score, kind)
        for x, y, score, kind in _merge_nearby(found)
    ]


def log_comb(image: np.ndarray) -> list[tuple[int, int]]:
    """Точки по всему бревну — куда тыкать, когда видимые ветки кончились.

    Это не распознавание, и выдавать его за распознавание нечестно. Это
    признание: половина веток лежит вдоль бревна, отростком не торчит и от коры
    ни цветом, ни яркостью не отличается — мерили. Зато клик мимо ветки игре
    ничего не говорит, а стол стоит открытым, пока на бревне остаётся хоть
    одна.

    Точки идут змейкой вдоль бревна: так курсор не мечется через весь стол и
    успевает больше. Первым делом помощник всё равно бьёт по разобранным
    целям — гребёнка добирает остаток.
    """
    if image is None or image.size == 0:
        return []
    reference, scale_x, scale_y = to_reference(image)
    zone, zone_left, zone_top = _zone(reference)
    if zone.size == 0:
        return []
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    mat = _mat_mask(hsv)
    hull = _mat_hull(mat)
    if hull is None:
        return []
    log = _log_mask(hsv, mat, hull)
    if log is None:
        return []
    inside = cv2.erode(
        log, np.ones((COMB_MARGIN * 2 + 1,) * 2, dtype=np.uint8),
    )
    rows = np.nonzero(inside.any(axis=1))[0]
    if not len(rows):
        return []
    points: list[tuple[int, int]] = []
    for step, y in enumerate(range(int(rows.min()), int(rows.max()) + 1, COMB_STEP)):
        columns = np.nonzero(inside[y])[0]
        if not len(columns):
            continue
        line = list(range(int(columns.min()), int(columns.max()) + 1, COMB_STEP))
        # Змейкой: чётные ряды слева направо, нечётные — обратно.
        for x in (line if step % 2 == 0 else line[::-1]):
            if inside[y, x]:
                points.append((
                    round((zone_left + x) * scale_x), round((zone_top + y) * scale_y),
                ))
    return points
