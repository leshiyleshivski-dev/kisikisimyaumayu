"""Sparse-graph recognition and routing for Volt's electro-maze."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ...core import VK_A, VK_D, VK_S, VK_W


class MazeMixin:
    @staticmethod
    def is_maze_screen(image: np.ndarray) -> bool:
        """Быстро отличить зелёную плату лабиринта от остальных экранов.

        Полное восстановление графа через HoughCircles занимает несколько
        секунд на 1440p. Для автодетекта сначала достаточно дешёвого цветового
        признака; маршрут после этого строится отдельно в рабочем потоке.
        """
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        center = hsv[
            int(height * 0.16):int(height * 0.90),
            int(width * 0.20):int(width * 0.80),
        ]
        if center.size == 0:
            return False
        hue, saturation, value = cv2.split(center)
        board_ratio = float(np.mean(
            (hue >= 35) & (hue <= 95) &
            (saturation >= 55) & (value >= 20)
        ))
        return board_ratio >= 0.45

    @staticmethod
    def maze_endpoint(hsv: np.ndarray, colour: str) -> tuple[int, int] | None:
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        if colour == "green":
            mask = ((hue >= 38) & (hue <= 95) & (saturation >= 110) & (value >= 100)).astype(np.uint8)
        else:
            mask = (((hue <= 9) | (hue >= 170)) & (saturation >= 115) & (value >= 100)).astype(np.uint8)
        _count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates = [(stats[index], centroids[index]) for index in range(1, len(stats)) if 18 <= stats[index, 2] <= 90 and 18 <= stats[index, 3] <= 90 and stats[index, 4] >= 120]
        if not candidates:
            return None
        _box, center = max(candidates, key=lambda item: int(item[0][4]))
        return round(float(center[0])), round(float(center[1]))

    @staticmethod
    def maze_has_link(mask: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> bool:
        """Проверить непрерывную дорожку между двумя реальными контактами."""
        x1, y1 = first
        x2, y2 = second
        length = max(abs(x2 - x1), abs(y2 - y1))
        if length < 24:
            return False
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        trim = max(6, min(14, length // 7))
        count = max(12, length - trim * 2)
        xs = np.linspace(x1, x2, count + trim * 2).round().astype(int)[trim:-trim]
        ys = np.linspace(y1, y2, count + trim * 2).round().astype(int)[trim:-trim]
        if not len(xs):
            return False
        # Hough может сместить центр кольца на 1–2 пикселя. Проверяем узкий
        # поперечный коридор, но требуем непрерывность почти по всей длине.
        coverage: list[bool] = []
        radius = max(2, min(mask.shape) // 480)
        for x, y in zip(xs, ys):
            if horizontal:
                y0, y1_bound = max(0, y - radius), min(mask.shape[0], y + radius + 1)
                coverage.append(bool(np.any(mask[y0:y1_bound, np.clip(x, 0, mask.shape[1] - 1)])))
            else:
                x0, x1_bound = max(0, x - radius), min(mask.shape[1], x + radius + 1)
                coverage.append(bool(np.any(mask[np.clip(y, 0, mask.shape[0] - 1), x0:x1_bound])))
        if float(np.mean(coverage)) < 0.82:
            return False
        longest_gap = current_gap = 0
        for present in coverage:
            current_gap = 0 if present else current_gap + 1
            longest_gap = max(longest_gap, current_gap)
        return longest_gap <= max(4, round(len(coverage) * 0.07))

    @classmethod
    def analyze_maze_route(cls, image: np.ndarray) -> list[int] | None:
        """Построить кратчайший путь только по реально видимым контактам.

        Линии лабиринта образуют не регулярную таблицу, а разреженный граф.
        Создание полного декартова произведения X×Y добавляло несуществующие
        узлы, из-за чего маршрут разрастался до десятков лишних шагов и мог
        разворачиваться. Здесь вершинами служат только найденные кольца, A и Б.
        """
        height, width = image.shape[:2]
        # У лабиринта почти весь центральный экран занят зелёной платой. Эта
        # дешёвая проверка не даёт красным/зелёным разъёмам «Проведения тока»
        # запустить тяжёлый поиск окружностей и выдать себя за A/Б.
        if not cls.is_maze_screen(image):
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        start = cls.maze_endpoint(hsv, "green")
        goal = cls.maze_endpoint(hsv, "red")
        if start is None or goal is None:
            return None

        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray_wire = (saturation <= 82) & (value >= 92)
        # Уже пройденный участок становится жёлтым. Это всё ещё тот же провод,
        # и исключать его из графа нельзя.
        yellow_wire = (hue >= 17) & (hue <= 38) & (saturation >= 90) & (value >= 105)
        green_endpoint = (hue >= 35) & (hue <= 95) & (saturation >= 75) & (value >= 75)
        red_endpoint = ((hue <= 10) | (hue >= 170)) & (saturation >= 80) & (value >= 80)
        wire_mask = (gray_wire | yellow_wire | green_endpoint | red_endpoint).astype(np.uint8)
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi_x0, roi_x1 = int(width * 0.12), int(width * 0.93)
        roi_y0, roi_y1 = int(height * 0.18), int(height * 0.84)
        search_gray = gray[roi_y0:roi_y1, roi_x0:roi_x1]
        circles = cv2.HoughCircles(
            cv2.medianBlur(search_gray, 5), cv2.HOUGH_GRADIENT,
            dp=1.1, minDist=max(16, min(width, height) // 65),
            param1=80, param2=12,
            minRadius=max(4, min(width, height) // 230),
            maxRadius=max(8, min(width, height) // 85),
        )
        if circles is None:
            return None

        merge_distance = max(12, min(width, height) // 100)
        nodes: list[tuple[int, int]] = [start, goal]
        for circle in circles[0]:
            point = (round(float(circle[0])) + roi_x0, round(float(circle[1])) + roi_y0)
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                continue
            if any(np.hypot(point[0] - existing[0], point[1] - existing[1]) <= merge_distance for existing in nodes):
                continue
            nodes.append(point)

        alignment = max(7, min(width, height) // 135)
        minimum_step = max(22, min(width, height) // 55)
        neighbours: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {node: [] for node in nodes}
        directions = (
            (0, -1, VK_W),
            (1, 0, VK_D),
            (0, 1, VK_S),
            (-1, 0, VK_A),
        )
        for node in nodes:
            x, y = node
            for delta_x, delta_y, key in directions:
                candidates: list[tuple[int, tuple[int, int]]] = []
                for other in nodes:
                    if other == node:
                        continue
                    offset_x, offset_y = other[0] - x, other[1] - y
                    if delta_x:
                        if offset_x * delta_x < minimum_step or abs(offset_y) > alignment:
                            continue
                        distance = abs(offset_x)
                    else:
                        if offset_y * delta_y < minimum_step or abs(offset_x) > alignment:
                            continue
                        distance = abs(offset_y)
                    candidates.append((distance, other))
                for _distance, other in sorted(candidates, key=lambda item: item[0]):
                    if cls.maze_has_link(wire_mask, node, other):
                        neighbours[node].append((other, key))
                        break

        # Ложные окружности от винтов и деталей остаются изолированными и не
        # могут попасть в маршрут. BFS гарантирует кратчайший путь без возвратов.
        queue: deque[tuple[int, int]] = deque([start])
        previous: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for other, key in neighbours.get(current, []):
                if other not in previous:
                    previous[other] = (current, key)
                    queue.append(other)
        if goal not in previous:
            return None
        route: list[int] = []
        current = goal
        while previous[current] is not None:
            parent, key = previous[current] or ((0, 0), VK_W)
            route.append(key)
            current = parent
        route.reverse()
        return route

    def detect_maze_route(self) -> list[int] | None:
        """Распознать «Электро-лабиринт» и вернуть путь от A до Б."""
        captured = self.capture_game_image()
        if captured is None:
            return None
        image, _bounds = captured
        return self.analyze_maze_route(image)
