"""Recognition and solvers for Volt's current-conduction grid."""

from __future__ import annotations

import heapq
import threading
import time

import cv2
import numpy as np

from ...core import (
    CURRENT_DIRECTIONS, CURRENT_GRID_COLUMNS, CURRENT_GRID_RATIO,
    CURRENT_GRID_ROWS, PORT_LEFT, PORT_RIGHT, VK_A, VK_D, VK_S,
    VK_SPACE, VK_W, find_game_window, user32,
)


class CurrentGridMixin:
    @staticmethod
    def rotate_ports(mask: int, turns: int) -> int:
        turns %= 4
        for _ in range(turns):
            mask = ((mask << 1) & 0b1111) | ((mask >> 3) & 1)
        return mask

    @classmethod
    def orientations(cls, mask: int) -> tuple[int, ...]:
        variants: list[int] = []
        for turns in range(4):
            value = cls.rotate_ports(mask, turns)
            if value not in variants:
                variants.append(value)
        return tuple(variants)

    @staticmethod
    def current_ports_connected(mask: int, first: int, second: int) -> bool:
        """Проверить внутреннее соединение двух портов одной плитки.

        У игровых углов, тройников и крестовин все видимые ветви соединены в
        центре. Отдельная проверка нужна, чтобы поиск маршрута не принимал
        отсутствующий порт за допустимый выход.
        """
        return first != second and bool(mask & first) and bool(mask & second)

    @staticmethod
    def component_box(mask: np.ndarray, min_width: int, min_height: int) -> tuple[int, int, int, int] | None:
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        candidates = [tuple(map(int, row)) for row in stats[1:] if row[2] >= min_width and row[3] >= min_height]
        if not candidates:
            return None
        x, y, width, height, _area = max(candidates, key=lambda row: row[4])
        return x, y, width, height

    @staticmethod
    def side_has_wire(mask: np.ndarray, x: int, y: int, radius: int) -> bool:
        patch = mask[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        return patch.size > 0 and int(np.count_nonzero(patch)) >= max(5, patch.size // 7)

    @staticmethod
    def side_has_colour(mask: np.ndarray, x: int, y: int, radius: int) -> bool:
        """Цветной провод тоньше серого, поэтому для клемм нужен мягче порог."""
        patch = mask[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        return patch.size > 0 and int(np.count_nonzero(patch)) >= max(3, patch.size // 30)

    @staticmethod
    def current_row_from_y(y: float, cell_size: float) -> int | None:
        """Привязать экранную координату к строке, не округляя соседнюю клетку."""
        row = round(y / cell_size - 0.5)
        if not 0 <= row < CURRENT_GRID_ROWS:
            return None
        center = (row + 0.5) * cell_size
        return row if abs(y - center) <= cell_size * 0.34 else None

    @classmethod
    def detect_current_terminals(
        cls,
        image: np.ndarray,
        grid_x: int,
        grid_y: int,
        grid_width: int,
        grid_height: int,
        cell_size: float,
    ) -> set[tuple[int, int, int]] | None:
        """Найти реальные внешние клеммы по корпусам и кабелю.

        Оранжевые участки внутри первой колонки не являются входами. Поэтому
        красные источники определяются по круглым кнопкам в металлических
        корпусах слева, а строка выхода — по отдельному зелёному кабелю справа
        от печатной платы. Эти признаки не меняются при вращении плиток.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        left_x1 = max(0, grid_x - round(cell_size * 1.48))
        left_x2 = min(image.shape[1], grid_x - round(cell_size * 0.02))
        left_band = gray[grid_y:grid_y + grid_height, left_x1:left_x2]
        if left_band.size == 0:
            return None
        circles = cv2.HoughCircles(
            cv2.medianBlur(left_band, 5), cv2.HOUGH_GRADIENT,
            dp=1.1, minDist=max(18, round(cell_size * 0.28)),
            param1=90, param2=18,
            minRadius=max(7, round(cell_size * 0.09)),
            maxRadius=max(12, round(cell_size * 0.25)),
        )
        source_candidates: dict[int, tuple[float, float]] = {}
        if circles is not None:
            expected_x = cell_size * 0.93
            patch_radius = max(12, round(cell_size * 0.34))
            for center_x, center_y, radius in circles[0]:
                if not cell_size * 0.76 <= center_x <= cell_size * 1.08:
                    continue
                if not cell_size * 0.10 <= radius <= cell_size * 0.24:
                    continue
                row = cls.current_row_from_y(float(center_y), cell_size)
                if row is None:
                    continue
                x, y = round(float(center_x)), round(float(center_y))
                patch = left_band[
                    max(0, y - patch_radius):y + patch_radius + 1,
                    max(0, x - patch_radius):x + patch_radius + 1,
                ]
                # У корпуса средняя яркость около 95; у красной платы и
                # пустого поля она ниже 40.
                brightness = float(np.mean(patch)) if patch.size else 0.0
                if brightness < 68.0:
                    continue
                score = brightness - abs(float(center_x) - expected_x) * 0.35
                previous = source_candidates.get(row)
                if previous is None or score > previous[0]:
                    source_candidates[row] = (score, float(center_y))
        source_rows = sorted(source_candidates)
        if not 1 <= len(source_rows) <= 3:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green = ((hue >= 35) & (hue <= 95) & (saturation >= 120) & (value >= 80)).astype(np.uint8)
        # Печатная плата заканчивается примерно через три клетки после поля;
        # дальше остаётся только яркий петлеобразный выходной кабель.
        right_x1 = min(image.shape[1], grid_x + grid_width + round(cell_size * 3.0))
        right_x2 = min(image.shape[1], grid_x + grid_width + round(cell_size * 5.2))
        right_band = green[grid_y:grid_y + grid_height, right_x1:right_x2]
        if right_band.size == 0:
            return None
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(right_band)
        target_candidates: list[tuple[int, float]] = []
        for index in range(1, count):
            _x, _y, width, height, area = map(int, stats[index])
            if width < cell_size * 0.65 or height < cell_size * 0.48 or area < cell_size * cell_size * 0.04:
                continue
            # Центроид петли смещён к более яркой верхней дуге; середина её
            # ограничивающего прямоугольника совпадает с центром корпуса.
            row = cls.current_row_from_y(float(_y + height / 2), cell_size)
            if row is not None:
                target_candidates.append((area, row))
        if not target_candidates:
            return None
        _area, target_row = max(target_candidates)
        terminals = {(row, 0, PORT_LEFT) for row in source_rows}
        terminals.add((target_row, CURRENT_GRID_COLUMNS - 1, PORT_RIGHT))
        return terminals

    def capture_current_grid(
        self,
        captured: tuple[np.ndarray, tuple[int, int, int, int]] | None = None,
    ) -> tuple[list[int], set[tuple[int, int, int]], tuple[int, int]] | None:
        """Считать разъёмы поля 8×8, наружные контакты и выделенную плитку.

        Внутри поля провода бывают серыми или оранжевыми, поэтому маска учитывает
        оба типа. Для внешних клемм дополнительно ищутся насыщенные красный и
        зелёный кабели — это не позволяет принять обычную линию сетки за контакт.
        """
        captured = captured or self.capture_game_image()
        if captured is None:
            return None
        _image, (_left, _top, width, height) = captured
        ratio_x, ratio_y, ratio_width, ratio_height = CURRENT_GRID_RATIO
        # Поле всегда квадратное. Раньше ширина и высота брались из разных
        # коэффициентов: на реальном экране клетки получались 109×103 и
        # последняя колонка с зелёным контактом съезжала вправо.
        cell_size = min(
            width * ratio_width / CURRENT_GRID_COLUMNS,
            height * ratio_height / CURRENT_GRID_ROWS,
        )
        grid_width = round(cell_size * CURRENT_GRID_COLUMNS)
        grid_height = round(cell_size * CURRENT_GRID_ROWS)
        grid_x = round(width * ratio_x)
        grid_y = round(height * ratio_y)
        terminal_padding = max(12, round(cell_size * 0.42))
        if grid_width < CURRENT_GRID_COLUMNS * 20 or grid_height < CURRENT_GRID_ROWS * 20:
            return None
        # Зелёный выход располагается сразу за правой границей поля, поэтому
        # сохраняем его узкую полосу вместе с самой сеткой.
        context = _image[
            grid_y:grid_y + grid_height,
            grid_x:grid_x + grid_width + terminal_padding,
        ]
        if context.shape[0] != grid_height or context.shape[1] < grid_width:
            return None
        image = context[:, :grid_width]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Скрин мини-игры затемнён полупрозрачным слоем GTA, поэтому у серых
        # проводов яркость заметно ниже, чем у белого интерфейса.
        gray_wire = (saturation <= 95) & (value >= 42)
        orange_wire = (hue >= 4) & (hue <= 28) & (saturation >= 55) & (value >= 55)
        wire_mask = (gray_wire | orange_wire).astype(np.uint8)
        cell_x, cell_y = image.shape[1] / CURRENT_GRID_COLUMNS, image.shape[0] / CURRENT_GRID_ROWS
        if abs(cell_x - cell_y) > max(cell_x, cell_y) * 0.12:
            return None
        # Пробу берём внутри плитки, не у самой границы. На сложных полях
        # металлический стык между соседними клетками попадал в прежнее окно
        # (0.41 клетки с радиусом 0.105) и создавал несуществующую ветвь.
        radius = max(2, round(min(cell_x, cell_y) * 0.070))
        masks: list[int] = []
        for row in range(CURRENT_GRID_ROWS):
            for column in range(CURRENT_GRID_COLUMNS):
                center_x = round((column + 0.5) * cell_x)
                center_y = round((row + 0.5) * cell_y)
                mask = 0
                for dy, dx, port, _opposite, _key in CURRENT_DIRECTIONS:
                    probe_x = round(center_x + dx * cell_x * 0.36)
                    probe_y = round(center_y + dy * cell_y * 0.36)
                    if self.side_has_wire(wire_mask, probe_x, probe_y, radius):
                        mask |= port
                masks.append(mask)

        terminals = self.detect_current_terminals(
            _image, grid_x, grid_y, grid_width, grid_height, cell_size,
        )
        if terminals is None:
            return None

        selected_mask = ((hue >= 72) & (hue <= 105) & (saturation >= 85) & (value >= 95)).astype(np.uint8)
        selected = self.component_box(selected_mask, max(8, round(cell_x * 0.40)), max(8, round(cell_y * 0.40)))
        if not selected:
            return None
        selected_x, selected_y, selected_width, selected_height = selected
        selection = (
            min(CURRENT_GRID_ROWS - 1, max(0, round((selected_y + selected_height / 2) / cell_y - 0.5))),
            min(CURRENT_GRID_COLUMNS - 1, max(0, round((selected_x + selected_width / 2) / cell_x - 0.5))),
        )
        return masks, terminals, selection

    @classmethod
    def solve_current_grid(cls, masks: list[int], terminals: set[tuple[int, int, int]], limit_seconds: float = 3.5) -> list[int] | None:
        """Решить повороты плиток через ограничения соседей и поиск с отсечениями."""
        rows, columns = CURRENT_GRID_ROWS, CURRENT_GRID_COLUMNS
        if len(masks) != rows * columns:
            return None
        terminal_map = {(row, column): port for row, column, port in terminals}
        domains = [list(cls.orientations(mask)) for mask in masks]

        def prune(state: list[list[int]]) -> list[list[int]] | None:
            changed = True
            while changed:
                changed = False
                for row in range(rows):
                    for column in range(columns):
                        index = row * columns + column
                        allowed = state[index]
                        for dr, dc, port, opposite, _key in CURRENT_DIRECTIONS:
                            next_row, next_column = row + dr, column + dc
                            if 0 <= next_row < rows and 0 <= next_column < columns:
                                neighbour = next_row * columns + next_column
                                filtered = [candidate for candidate in allowed if any(bool(candidate & port) == bool(other & opposite) for other in state[neighbour])]
                            else:
                                # У плиток могут быть свободные концы на
                                # краю поля; это не ошибка. Ограничиваем
                                # только реальные цветные клеммы.
                                required = terminal_map.get((row, column)) == port
                                filtered = [candidate for candidate in allowed if bool(candidate & port)] if required else allowed
                            if not filtered:
                                return None
                            if len(filtered) != len(allowed):
                                state[index] = filtered
                                allowed = filtered
                                changed = True
            return state

        def terminals_connected(solution: list[int]) -> bool:
            terminal_cells = {(row, column) for row, column, _port in terminals}
            sources = {(row, column) for row, column, _port in terminals if column == 0}
            targets = {(row, column) for row, column, _port in terminals if column == columns - 1}
            if not sources or not targets:
                return False
            for source in sources:
                seen = {source}
                queue = list(seen)
                while queue:
                    row, column = queue.pop()
                    mask = solution[row * columns + column]
                    for dr, dc, port, opposite, _key in CURRENT_DIRECTIONS:
                        next_row, next_column = row + dr, column + dc
                        if mask & port and 0 <= next_row < rows and 0 <= next_column < columns:
                            if solution[next_row * columns + next_column] & opposite and (next_row, next_column) not in seen:
                                seen.add((next_row, next_column))
                                queue.append((next_row, next_column))
                if targets & seen:
                    return True
            return False

        deadline = time.monotonic() + limit_seconds
        initial = prune([domain[:] for domain in domains])
        if initial is None:
            return None

        def search(state: list[list[int]]) -> list[int] | None:
            if time.monotonic() >= deadline:
                return None
            unresolved = [index for index, domain in enumerate(state) if len(domain) > 1]
            if not unresolved:
                solution = [domain[0] for domain in state]
                return solution if terminals_connected(solution) else None
            index = min(unresolved, key=lambda item: len(state[item]))
            for candidate in state[index]:
                branch = [domain[:] for domain in state]
                branch[index] = [candidate]
                filtered = prune(branch)
                if filtered is not None:
                    solved = search(filtered)
                    if solved is not None:
                        return solved
            return None

        return search(initial)

    @classmethod
    def solve_current_path(cls, masks: list[int], terminals: set[tuple[int, int, int]]) -> list[int] | None:
        """Подключить к правому выходу каждый реальный красный источник.

        У правого корпуса два индикатора: решение с одним подключённым входом
        оставляет мини-игру открытой. Ищем несколько совместимых простых путей,
        объединяем их в одну сеть и только затем выбираем поворот каждой плитки.
        """
        rows, columns = CURRENT_GRID_ROWS, CURRENT_GRID_COLUMNS
        if len(masks) != rows * columns:
            return None
        sources = sorted((row, column) for row, column, port in terminals if column == 0 and port == PORT_LEFT)
        targets = [(row, column) for row, column, port in terminals if column == columns - 1 and port == PORT_RIGHT]
        if not sources or len(targets) != 1:
            return None
        target = targets[0]
        deadline = time.monotonic() + 3.5

        def port_between(first: tuple[int, int], second: tuple[int, int]) -> int | None:
            delta_row, delta_column = second[0] - first[0], second[1] - first[1]
            for dr, dc, port, _opposite, _key in CURRENT_DIRECTIONS:
                if (dr, dc) == (delta_row, delta_column):
                    return port
            return None

        def requirements(path: tuple[tuple[int, int], ...]) -> dict[int, int] | None:
            required: dict[int, int] = {}
            for position, cell in enumerate(path):
                if position == 0:
                    incoming = PORT_LEFT
                else:
                    incoming = port_between(cell, path[position - 1])
                    if incoming is None:
                        return None
                if position == len(path) - 1:
                    outgoing = PORT_RIGHT
                else:
                    outgoing = port_between(cell, path[position + 1])
                    if outgoing is None:
                        return None
                bits = incoming | outgoing
                index = cell[0] * columns + cell[1]
                if not any(
                    cls.current_ports_connected(orientation, incoming, outgoing)
                    for orientation in cls.orientations(masks[index])
                ):
                    return None
                required[index] = bits
            return required

        def enumerate_paths(source: tuple[int, int], max_paths: int = 400) -> list[dict[int, int]]:
            """Вернуть короткие физически возможные пути без повторов клеток."""
            sequence = 0
            start_path = (source,)
            heap: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = [
                (abs(source[0] - target[0]) + abs(source[1] - target[1]), 0, sequence, start_path),
            ]
            found: list[dict[int, int]] = []
            expansions = 0
            while heap and len(found) < max_paths and expansions < 90_000 and time.monotonic() < deadline:
                _estimate, distance, _order, path = heapq.heappop(heap)
                expansions += 1
                current = path[-1]
                if current == target:
                    candidate = requirements(path)
                    if candidate is not None:
                        found.append(candidate)
                    continue
                entered = PORT_LEFT if len(path) == 1 else port_between(current, path[-2])
                if entered is None:
                    continue
                for dr, dc, outgoing, opposite, _key in CURRENT_DIRECTIONS:
                    next_cell = (current[0] + dr, current[1] + dc)
                    if not (0 <= next_cell[0] < rows and 0 <= next_cell[1] < columns):
                        continue
                    if next_cell in path:
                        continue
                    current_index = current[0] * columns + current[1]
                    needed = entered | outgoing
                    if not any(
                        cls.current_ports_connected(orientation, entered, outgoing)
                        for orientation in cls.orientations(masks[current_index])
                    ):
                        continue
                    next_index = next_cell[0] * columns + next_cell[1]
                    if next_cell == target:
                        next_needed = opposite | PORT_RIGHT
                        if not any(
                            cls.current_ports_connected(orientation, opposite, PORT_RIGHT)
                            for orientation in cls.orientations(masks[next_index])
                        ):
                            continue
                    elif not any(
                        any(
                            cls.current_ports_connected(orientation, opposite, candidate_port)
                            for _dr, _dc, candidate_port, _candidate_opposite, _key
                            in CURRENT_DIRECTIONS
                            if candidate_port != opposite
                        )
                        for orientation in cls.orientations(masks[next_index])
                    ):
                        continue
                    next_path = path + (next_cell,)
                    sequence += 1
                    heuristic = abs(next_cell[0] - target[0]) + abs(next_cell[1] - target[1])
                    heapq.heappush(heap, (distance + 1 + heuristic, distance + 1, sequence, next_path))
            return found

        path_sets = [enumerate_paths(source) for source in sources]
        if any(not candidates for candidates in path_sets):
            return None

        def merge_requirements(base: dict[int, int], addition: dict[int, int]) -> dict[int, int] | None:
            merged = base.copy()
            for index, bits in addition.items():
                combined = merged.get(index, 0) | bits
                if not any((orientation & combined) == combined for orientation in cls.orientations(masks[index])):
                    return None
                merged[index] = combined
            return merged

        def materialize(required: dict[int, int]) -> tuple[list[int], int] | None:
            solved = masks[:]
            turns_total = 0
            for index, bits in required.items():
                choices: list[tuple[int, int]] = []
                for orientation in cls.orientations(masks[index]):
                    if (orientation & bits) != bits:
                        continue
                    turns = cls.rotation_count(masks[index], orientation)
                    if turns is not None:
                        choices.append((turns, orientation))
                if not choices:
                    return None
                turns, solved[index] = min(choices)
                turns_total += turns
            return solved, turns_total

        # Сначала перебираем источник с меньшим числом вариантов: это резко
        # сокращает комбинации на сложных досках.
        path_sets.sort(key=len)
        best: tuple[int, list[int]] | None = None

        def combine(position: int, merged: dict[int, int]) -> None:
            nonlocal best
            if time.monotonic() >= deadline:
                return
            if position == len(path_sets):
                materialized = materialize(merged)
                if materialized is None:
                    return
                solved, turns = materialized
                # Среди одинаковых по поворотам решений предпочитаем сеть с
                # меньшим числом затронутых плиток.
                score = turns * 100 + len(merged)
                if best is None or score < best[0]:
                    best = (score, solved)
                return
            for candidate in path_sets[position]:
                combined = merge_requirements(merged, candidate)
                if combined is not None:
                    combine(position + 1, combined)
                if time.monotonic() >= deadline:
                    return

        combine(0, {})
        if best is None:
            return None
        solved = best[1]
        return solved if cls.current_grid_complete(solved, terminals) else None

    @classmethod
    def current_grid_complete(cls, masks: list[int], terminals: set[tuple[int, int, int]]) -> bool:
        """Проверить, что каждый левый источник действительно дошёл до выхода."""
        rows, columns = CURRENT_GRID_ROWS, CURRENT_GRID_COLUMNS
        if len(masks) != rows * columns:
            return False
        sources = [
            (row, column, port)
            for row, column, port in terminals
            if column == 0 and port == PORT_LEFT
        ]
        targets = {
            (row, column, port)
            for row, column, port in terminals
            if column == columns - 1 and port == PORT_RIGHT
        }
        if not sources or not targets:
            return False
        for source in sources:
            if not masks[source[0] * columns + source[1]] & source[2]:
                return False
            queue = [source]
            seen = {source}
            while queue:
                row, column, entered_port = queue.pop()
                mask = masks[row * columns + column]
                # Сначала распространяем ток внутри плитки. Для крестовины
                # это только противоположный конец того же провода.
                for _dr, _dc, internal_port, _opposite, _key in CURRENT_DIRECTIONS:
                    internal = (row, column, internal_port)
                    if (
                        internal not in seen and
                        cls.current_ports_connected(mask, entered_port, internal_port)
                    ):
                        seen.add(internal)
                        queue.append(internal)

                # Затем переходим через границу к совпадающему порту соседа.
                for dr, dc, port, opposite, _key in CURRENT_DIRECTIONS:
                    next_row, next_column = row + dr, column + dc
                    if entered_port != port or not (0 <= next_row < rows and 0 <= next_column < columns):
                        continue
                    if not masks[next_row * columns + next_column] & opposite:
                        continue
                    endpoint = (next_row, next_column, opposite)
                    if endpoint not in seen:
                        seen.add(endpoint)
                        queue.append(endpoint)
            if not targets & seen:
                return False
        return True

    @classmethod
    def rotation_count(cls, current: int, target: int) -> int | None:
        for turns in range(4):
            if cls.rotate_ports(current, turns) == target:
                return turns
        return None

    def build_key_plan(self, masks: list[int], solved: list[int], selection: tuple[int, int]) -> list[int] | None:
        turns = [self.rotation_count(current, target) for current, target in zip(masks, solved)]
        if any(value is None for value in turns):
            return None
        current_row, current_column = selection
        actions: list[int] = []
        targets = [index for index, value in enumerate(turns) if value]
        targets.sort(key=lambda index: (index // CURRENT_GRID_COLUMNS, index % CURRENT_GRID_COLUMNS))
        for index in targets:
            row, column = divmod(index, CURRENT_GRID_COLUMNS)
            while current_row > row:
                actions.append(VK_W)
                current_row -= 1
            while current_row < row:
                actions.append(VK_S)
                current_row += 1
            while current_column > column:
                actions.append(VK_A)
                current_column -= 1
            while current_column < column:
                actions.append(VK_D)
                current_column += 1
            actions.extend([VK_SPACE] * int(turns[index] or 0))
        return actions

    def recognize_and_solve(
        self,
        captured: tuple[np.ndarray, tuple[int, int, int, int]] | None = None,
    ) -> None:
        if not self.running or not self.active:
            return
        grid = self.capture_current_grid(captured)
        if grid is None:
            self.resume_monitoring("Мини-игры пока нет — продолжаю наблюдение.", 450)
            return
        masks, terminals, selection = grid
        sources = sum(column == 0 for _row, column, _port in terminals)
        self.current_recovery_round = 0
        self.current_verify_misses = 0
        self.target.set(f"Сетка найдена · красных входов: {sources}")
        self.status.set("Строю общую схему для всех входов…")

        generation = self.detection_generation

        def solve_in_background() -> None:
            solved = self.solve_current_path(masks, terminals)
            self.after(0, lambda: self.finish_solution(masks, solved, selection, generation))

        threading.Thread(target=solve_in_background, daemon=True, name="kiski-current-solver").start()

    def finish_solution(
        self,
        masks: list[int],
        solved: list[int] | None,
        selection: tuple[int, int],
        generation: int | None = None,
    ) -> None:
        if generation is not None and generation != self.detection_generation:
            return
        if not self.running or not self.active:
            return
        if solved is None:
            self.resume_monitoring(
                "Поле тока распознано, но схема пока не подтверждена. Повторю автоматически.",
                900,
            )
            return
        plan = self.build_key_plan(masks, solved, selection)
        if plan is None:
            self.resume_monitoring(
                "Не удалось составить безопасную последовательность. Продолжаю наблюдение.",
                900,
            )
            return
        if not plan:
            # solve_current_path возвращает результат только после проверки
            # связности всех источников, поэтому нулевой план уже подтверждён.
            self.resume_monitoring("Обе цепи уже собраны — продолжаю автодетект.", 1200)
            return
        self.actions = plan
        self.action_kind = "current"
        self.action_interval = 135
        self.target.set(f"Обе цепи рассчитаны · действий: {len(plan)}")
        self.status.set("Подключаю оба красных входа. Не трогай клавиатуру до проверки.")
        self.run_next_action()

    def verify_current_completion(self) -> None:
        """Не объявлять успех, пока оба индикатора не получили питание."""
        self.action_job = None
        if not self.running or not self.active or self.action_kind != "current":
            return
        if (
            not self.game_window or
            not find_game_window(self.process.get()) or
            user32.GetForegroundWindow() != self.game_window
        ):
            self.stop("Проверка тока остановлена: окно GTA больше не активно.")
            return
        captured = self.capture_current_grid()
        if captured is None:
            # После успеха окно мини-игры исчезает. Несколько чтений защищают
            # от единичного кадра анимации или задержки отрисовки.
            self.current_verify_misses += 1
            if self.current_verify_misses < 3:
                self.action_job = self.after(260, self.verify_current_completion)
                return
            self.resume_monitoring(
                "Обе цепи подключены: мини-игра завершилась. Жду следующий вызов.",
                1400,
            )
            return
        self.current_verify_misses = 0
        masks, terminals, selection = captured
        if self.current_grid_complete(masks, terminals):
            self.resume_monitoring(
                "Обе цепи подключены и подтверждены. Продолжаю автодетект.",
                1400,
            )
            return
        if self.current_recovery_round >= 2:
            self.stop("После повторной проверки одна из цепей не подключена. Ввод остановлен.")
            return
        self.current_recovery_round += 1
        self.target.set(f"Достраиваю вторую цепь · попытка {self.current_recovery_round}/2")
        self.status.set("Поле осталось открытым — пересчитываю только недостающие повороты.")

        generation = self.detection_generation

        def solve_again() -> None:
            solved = self.solve_current_path(masks, terminals)
            self.after(0, lambda: self.finish_solution(masks, solved, selection, generation))

        threading.Thread(target=solve_again, daemon=True, name="kiski-current-recheck").start()
