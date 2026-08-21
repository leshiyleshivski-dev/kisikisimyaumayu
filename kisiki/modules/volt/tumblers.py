"""Recognition and solving for Volt's switch-board mini-game."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ...core import cursor_position, find_game_window, send_left_click, user32


class TumblerMixin:
    @staticmethod
    def tumbler_layout_signature(
        positions: list[tuple[int, int]],
        bounds: tuple[int, int, int, int],
    ) -> tuple[int, ...] | None:
        """Подтвердить, что цветные детали образуют центральную ровную панель.

        Красные фонари, стоп-сигналы и элементы HUD во время езды иногда
        давали четыре подходящих цветных контура. Настоящие тумблеры всегда
        образуют выровненную прямоугольную решётку в центре мини-игры.
        """
        left, top, width, height = bounds
        if len(positions) < 4 or width <= 0 or height <= 0:
            return None
        relative = [((x - left) / width, (y - top) / height) for x, y in positions]
        if any(not 0.25 <= x <= 0.75 or not 0.38 <= y <= 0.88 for x, y in relative):
            return None

        ordered = sorted(relative, key=lambda point: (point[1], point[0]))
        rows: list[list[tuple[float, float]]] = []
        for point in ordered:
            if not rows or point[1] - float(np.mean([item[1] for item in rows[-1]])) > 0.045:
                rows.append([point])
            else:
                rows[-1].append(point)
        if len(rows) < 2 or len(rows) > 5:
            return None
        column_count = len(rows[0])
        if column_count < 2 or any(len(row) != column_count for row in rows):
            return None

        sorted_rows = [sorted(row) for row in rows]
        reference_x = [point[0] for point in sorted_rows[0]]
        if any(
            abs(point[0] - reference_x[index]) > 0.022
            for row in sorted_rows[1:]
            for index, point in enumerate(row)
        ):
            return None
        x_gaps = [right - left_x for left_x, right in zip(reference_x, reference_x[1:])]
        if not x_gaps or min(x_gaps) < 0.025 or max(x_gaps) > min(x_gaps) * 1.45:
            return None
        row_y = [float(np.mean([point[1] for point in row])) for row in sorted_rows]
        y_gaps = [bottom - upper for upper, bottom in zip(row_y, row_y[1:])]
        if not y_gaps or min(y_gaps) < 0.04 or max(y_gaps) > min(y_gaps) * 1.55:
            return None

        center_x = float(np.mean(reference_x))
        center_y = float(np.mean(row_y))
        if not 0.38 <= center_x <= 0.62 or not 0.48 <= center_y <= 0.80:
            return None
        # Крупные корзины не реагируют на дрожание распознавания в 1–2 px,
        # но отличают другую случайную группу объектов на следующем кадре.
        return (
            len(rows), column_count,
            round(center_x * 20), round(center_y * 20),
            round((reference_x[-1] - reference_x[0]) * 20),
            round((row_y[-1] - row_y[0]) * 20),
        )

    def detect_tumbler_layout(
        self,
        captured: tuple[np.ndarray, tuple[int, int, int, int]] | None = None,
    ) -> tuple[list[tuple[int, int]], list[bool]] | None:
        """Найти тумблеры в любом числе строк и столбцов.

        Размер поля не задан: отбираем одинаковые красные и зелёные рычаги,
        затем группируем их по фактическим Y-координатам. Поэтому 2×3, 3×4,
        5×N и другие раскладки не зависят от захардкоженного количества. Оба
        цвета нужны для повторного запуска на уже частично собранной панели.
        """
        captured = captured or self.capture_game_image()
        if captured is None:
            return None
        image, (left, top, width, height) = captured
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red = ((hue <= 9) | (hue >= 170)) & (saturation >= 115) & (value >= 100)
        green = (hue >= 35) & (hue <= 90) & (saturation >= 90) & (value >= 90)
        switch_colours = (red | green).astype(np.uint8)
        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(switch_colours)
        # На 2560×1440 внутренняя красная часть рычага имеет ширину ~54 px.
        # Предыдущий порог 60 px отбрасывал ровно настоящие тумблеры.
        min_width, min_height = max(18, width // 50), max(28, height // 30)
        max_width, max_height = max(min_width + 1, width // 9), max(min_height + 1, height // 7)
        candidates: list[tuple[int, int, int, int, int, bool]] = []
        for label, row in enumerate(stats[1:], start=1):
            x, y, item_width, item_height, area = map(int, row)
            if not (min_width <= item_width <= max_width and min_height <= item_height <= max_height):
                continue
            component = labels[y:y + item_height, x:x + item_width] == label
            green_pixels = int(np.count_nonzero(green[y:y + item_height, x:x + item_width] & component))
            red_pixels = int(np.count_nonzero(red[y:y + item_height, x:x + item_width] & component))
            candidates.append((x, y, item_width, item_height, area, green_pixels > red_pixels))
        if len(candidates) < 4:
            return None
        median_width = float(np.median([item[2] for item in candidates]))
        median_height = float(np.median([item[3] for item in candidates]))
        candidates = [item for item in candidates if 0.55 <= item[2] / median_width <= 1.75 and 0.55 <= item[3] / median_height <= 1.75]
        candidates.sort(key=lambda item: (item[1], item[0]))
        row_tolerance = max(18, int(median_height * 0.62))
        rows: list[list[tuple[int, int, int, int, int, bool]]] = []
        for item in candidates:
            if not rows or item[1] - rows[-1][0][1] > row_tolerance:
                rows.append([item])
            else:
                rows[-1].append(item)
        rows = [row for row in rows if len(row) >= 2]
        if len(rows) < 2:
            return None
        # Не используем порядок найденных контуров: строим порядок строго
        # сверху вниз, слева направо, чтобы раскладка была стабильной.
        ordered = [item for row in rows for item in sorted(row, key=lambda item: item[0])]
        positions = [(left + x + item_width // 2, top + y + item_height // 2) for x, y, item_width, item_height, _area, _on in ordered]
        states = [on for _x, _y, _item_width, _item_height, _area, on in ordered]
        if self.tumbler_layout_signature(positions, (left, top, width, height)) is None:
            return None
        return positions, states

    def read_tumbler_switch_states(self) -> list[bool] | None:
        """Прочитать цвета по зафиксированным координатам переключателей."""
        captured = self.capture_game_image()
        if captured is None:
            return None
        image, (left, top, width, height) = captured
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        radius_x = max(16, width // 100)
        radius_y = max(24, height // 32)
        states: list[bool] = []
        for screen_x, screen_y in self.tumbler_positions:
            x, y = screen_x - left, screen_y - top
            x0, x1 = max(0, x - radius_x), min(width, x + radius_x + 1)
            y0, y1 = max(0, y - radius_y), min(height, y + radius_y + 1)
            if x0 >= x1 or y0 >= y1:
                return None
            crop_hue = hue[y0:y1, x0:x1]
            crop_saturation = saturation[y0:y1, x0:x1]
            crop_value = value[y0:y1, x0:x1]
            green = np.count_nonzero(
                (crop_hue >= 35) & (crop_hue <= 90) &
                (crop_saturation >= 90) & (crop_value >= 90)
            )
            red = np.count_nonzero(
                ((crop_hue <= 9) | (crop_hue >= 170)) &
                (crop_saturation >= 115) & (crop_value >= 100)
            )
            if green + red < 30:
                return None
            states.append(int(green) > int(red))
        return states

    @staticmethod
    def analyze_tumbler_gauges(
        image: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]] | None:
        """Считать положения стрелок и красные цели обоих приборов.

        Показания A и V имеют разные шкалы, поэтому сравнивать их числа между
        собой нельзя. Всё переводится в долю своей полуокружности: 0 — левый
        край шкалы, 1 — правый. Красная метка распознаётся отдельно на каждом
        приборе и тем самым задаёт настоящий диапазон решения.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        upper = gray[: int(height * 0.62), :]
        min_radius = max(55, min(width, height) // 12)
        max_radius = max(min_radius + 1, min(width, height) // 6)
        circles = cv2.HoughCircles(
            cv2.GaussianBlur(upper, (5, 5), 0), cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(120, min(width, height) // 5), param1=85, param2=34,
            minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is None:
            return None
        candidates = [tuple(map(float, circle)) for circle in circles[0]]
        pairs: list[tuple[float, tuple[float, float, float], tuple[float, float, float]]] = []
        for first_index, first in enumerate(candidates):
            for second in candidates[first_index + 1:]:
                mean_radius = (first[2] + second[2]) / 2.0
                separation = abs(first[0] - second[0])
                radius_difference = abs(first[2] - second[2]) / mean_radius
                y_difference = abs(first[1] - second[1]) / mean_radius
                separation_ratio = separation / mean_radius
                if radius_difference > 0.18 or y_difference > 0.32 or not 2.25 <= separation_ratio <= 3.55:
                    continue
                if max(first[1], second[1]) > height * 0.55:
                    continue
                score = radius_difference * 3.0 + y_difference * 2.0 + abs(separation_ratio - 2.82)
                pairs.append((score, first, second))
        if not pairs:
            return None
        _score, first, second = min(pairs, key=lambda item: item[0])
        selected = sorted((first, second), key=lambda circle: circle[0])

        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red = (((hue <= 10) | (hue >= 170)) & (saturation >= 90) & (value >= 90))

        def normalized_angle(xs: np.ndarray, ys: np.ndarray, center_x: float, center_y: float) -> np.ndarray:
            angles = np.arctan2(ys - center_y, xs - center_x)
            angles = np.where(angles < 0, angles + 2 * np.pi, angles)
            return np.clip((angles - np.pi) / np.pi, 0.0, 1.0)

        def read_one(circle: tuple[float, float, float]) -> tuple[float, tuple[float, float]] | None:
            center_x, center_y, radius = circle
            y0, y1 = max(0, int(center_y - radius)), min(height, int(center_y + radius) + 1)
            x0, x1 = max(0, int(center_x - radius)), min(width, int(center_x + radius) + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            distances = np.hypot(xx - center_x, yy - center_y)

            target_pixels = red[y0:y1, x0:x1] & (distances >= radius * 0.67) & (distances <= radius * 0.96) & (yy <= center_y + radius * 0.08)
            target_y, target_x = np.nonzero(target_pixels)
            if len(target_x) < 12:
                return None
            target_values = normalized_angle(target_x + x0, target_y + y0, center_x, center_y)
            target_values = target_values[(target_values >= 0.01) & (target_values <= 0.99)]
            if len(target_values) < 12:
                return None
            target_center = float(np.median(target_values))
            target_spread = max(0.030, float(np.percentile(target_values, 90) - np.percentile(target_values, 10)) * 0.70)
            target_range = (max(0.0, target_center - target_spread), min(1.0, target_center + target_spread))

            # Стрелка — непрерывная тёмная полоса от ступицы к шкале. Берём
            # только внутреннюю часть циферблата, чтобы цифры и риски не могли
            # победить её по суммарной темноте.
            angles = np.linspace(np.pi * 1.005, np.pi * 1.995, 241)
            radial = np.linspace(radius * 0.18, radius * 0.66, 72)
            scores: list[float] = []
            for angle in angles:
                perpendicular_x, perpendicular_y = -np.sin(angle), np.cos(angle)
                samples: list[np.ndarray] = []
                for offset in (-2.0, 0.0, 2.0):
                    xs = np.clip((center_x + np.cos(angle) * radial + perpendicular_x * offset).round().astype(int), 0, width - 1)
                    ys = np.clip((center_y + np.sin(angle) * radial + perpendicular_y * offset).round().astype(int), 0, height - 1)
                    samples.append(gray[ys, xs].astype(np.float32))
                darkness = 255.0 - np.mean(np.stack(samples), axis=0)
                # Длинная стрелка должна быть тёмной почти по всему лучу;
                # усечённое среднее подавляет отдельные буквы шкалы.
                scores.append(float(np.mean(np.partition(darkness, -48)[-48:])))
            best = int(np.argmax(scores))
            if scores[best] < 42:
                return None
            needle = best / (len(angles) - 1)
            return needle, target_range

        readings = [read_one(circle) for circle in selected]
        if any(reading is None for reading in readings):
            return None
        left_reading, right_reading = readings
        assert left_reading is not None and right_reading is not None
        return (left_reading[0], right_reading[0]), (left_reading[1], right_reading[1])

    def read_tumbler_gauges(self) -> tuple[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]] | None:
        """Снять экран и прочитать оба прибора тумблеров."""
        captured = self.capture_game_image()
        if captured is None:
            return None
        image, _bounds = captured
        return self.analyze_tumbler_gauges(image)

    @staticmethod
    def tumbler_distance(value: float, target: tuple[float, float]) -> float:
        """Расстояние показания до допустимой красной зоны."""
        low, high = target
        if value < low:
            return low - value
        if value > high:
            return value - high
        return 0.0

    @classmethod
    def choose_tumbler_solution(
        cls,
        baseline: tuple[float, float],
        effects: list[tuple[float, float]],
        targets: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[list[int], float]:
        """Перебрать состояния и привести каждый прибор в его красную зону."""
        if len(effects) > 18:
            return [], float("inf")
        best_key: tuple[float, float, float, float, int] | None = None
        best_mask = 0
        for mask in range(1 << len(effects)):
            left_value, right_value = baseline
            for index, (left_effect, right_effect) in enumerate(effects):
                if mask & (1 << index):
                    left_value += left_effect
                    right_value += right_effect
            left_error = cls.tumbler_distance(left_value, targets[0])
            right_error = cls.tumbler_distance(right_value, targets[1])
            left_center = (targets[0][0] + targets[0][1]) / 2.0
            right_center = (targets[1][0] + targets[1][1]) / 2.0
            left_center_error = abs(left_value - left_center)
            right_center_error = abs(right_value - right_center)
            count = mask.bit_count()
            # Любое попадание обеих стрелок в зоны лучше промаха. Среди всех
            # попаданий выбираем не край диапазона, а наиболее устойчивую
            # комбинацию около центров красных меток.
            key = (
                max(left_error, right_error),
                left_error + right_error,
                max(left_center_error, right_center_error),
                left_center_error + right_center_error,
                count,
            )
            if best_key is None or key < best_key:
                best_key, best_mask = key, mask
        selection = [index for index in range(len(effects)) if best_mask & (1 << index)]
        return selection, best_key[0] if best_key is not None else float("inf")

    def click_tumbler(self, index: int) -> bool:
        if not (0 <= index < len(self.tumbler_positions)):
            return False
        x, y = self.tumbler_positions[index]
        start_x, start_y = cursor_position()
        distance = max(abs(x - start_x), abs(y - start_y))
        steps = max(5, min(18, (distance + 39) // 40))
        for step in range(1, steps + 1):
            progress = step / steps
            eased = progress * progress * (3.0 - 2.0 * progress)
            next_x = round(start_x + (x - start_x) * eased)
            next_y = round(start_y + (y - start_y) * eased)
            if not user32.SetCursorPos(next_x, next_y):
                return False
            if step < steps:
                time.sleep(0.008)
        time.sleep(0.035)
        return send_left_click(0.055)

    def run_tumbler_step(self) -> None:
        self.action_job = None
        if not self.running or not self.active:
            return
        # Рычаг после клика меняет цвет и перестаёт быть «красным кандидатом».
        # Поэтому повторный поиск раскладки здесь обрывал измерение на 3–4-м
        # переключателе. Координаты фиксируются перед первым действием.
        if not self.game_window or not find_game_window(self.process.get()):
            self.stop("Окно GTA закрыто. Тумблеры остановлены.")
            return
        if user32.GetForegroundWindow() != self.game_window:
            self.stop("Тумблеры остановлены: GTA больше не является активным окном.")
            return
        total = len(self.tumbler_positions)
        if self.tumbler_phase == "normalize":
            while self.tumbler_index < total and not self.tumbler_initial_states[self.tumbler_index]:
                self.tumbler_index += 1
            if self.tumbler_index >= total:
                self.tumbler_phase = "baseline"
                self.tumbler_index = 0
                self.action_job = self.after(850, self.run_tumbler_step)
                return
            if not self.click_tumbler(self.tumbler_index):
                self.stop("Windows не принял клик по тумблеру.")
                return
            self.tumbler_initial_states[self.tumbler_index] = False
            self.tumbler_index += 1
            self.target.set(f"Сбрасываю тумблеры · {self.tumbler_index}/{total}")
            self.action_job = self.after(340, self.run_tumbler_step)
            return
        if self.tumbler_phase == "baseline":
            states = self.read_tumbler_switch_states()
            if states is None:
                self.stop("Не удалось подтвердить положения тумблеров. Ввод остановлен.")
                return
            if any(states):
                self.tumbler_initial_states = states
                self.tumbler_index = 0
                self.tumbler_phase = "normalize"
                self.status.set("Один клик не зарегистрирован — повторяю сброс.")
                self.action_job = self.after(160, self.run_tumbler_step)
                return
            gauges = self.read_tumbler_gauges()
            if gauges is None:
                self.stop("Тумблеры сброшены, но стрелки приборов не распознаны.")
                return
            self.tumbler_baseline, self.tumbler_targets = gauges
            self.tumbler_previous = self.tumbler_baseline
            self.tumbler_effects = []
            self.tumbler_index = 0
            self.tumbler_phase = "measure_on"
            self.status.set("Последовательно включаю тумблеры и измеряю фактический прирост.")
            self.action_job = self.after(180, self.run_tumbler_step)
            return
        if self.tumbler_phase == "measure_on":
            if self.tumbler_index >= total:
                self.tumbler_solution, error = self.choose_tumbler_solution(
                    self.tumbler_baseline, self.tumbler_effects, self.tumbler_targets,
                )
                # Каждый тумблер измеряется отдельно и затем снова выключается,
                # поэтому перед применением вся панель находится в чистом нуле.
                self.tumbler_apply = list(self.tumbler_solution)
                self.tumbler_index = 0
                self.tumbler_phase = "apply"
                self.target.set(f"Сочетание найдено · {len(self.tumbler_solution)} из {total}")
                self.status.set(f"Промах до красных зон: {error * 100:.1f}%. Применяю сочетание.")
                self.action_job = self.after(180, self.run_tumbler_step)
                return
            if not self.click_tumbler(self.tumbler_index):
                self.stop("Windows не принял клик по тумблеру.")
                return
            self.tumbler_phase = "measure_read"
            self.tumbler_click_retries = 0
            self.tumbler_settle_retries = 0
            self.action_job = self.after(900, self.run_tumbler_step)
            return
        if self.tumbler_phase == "measure_read":
            states = self.read_tumbler_switch_states()
            if states is None:
                self.stop("Не удалось проверить клик по тумблеру. Ввод остановлен.")
                return
            if not states[self.tumbler_index]:
                if self.tumbler_click_retries >= 1 or not self.click_tumbler(self.tumbler_index):
                    self.stop("Тумблер не переключился после повторного клика. Ввод остановлен.")
                    return
                self.tumbler_click_retries += 1
                self.action_job = self.after(900, self.run_tumbler_step)
                return
            gauges = self.read_tumbler_gauges()
            if gauges is None:
                self.stop("Не удалось считать стрелки после переключения. Ввод остановлен.")
                return
            self.tumbler_pending_values, _targets = gauges
            self.tumbler_phase = "measure_confirm"
            self.action_job = self.after(180, self.run_tumbler_step)
            return
        if self.tumbler_phase == "measure_confirm":
            gauges = self.read_tumbler_gauges()
            if gauges is None:
                self.stop("Не удалось повторно считать стрелки после переключения.")
                return
            changed, _targets = gauges
            drift = max(
                abs(changed[0] - self.tumbler_pending_values[0]),
                abs(changed[1] - self.tumbler_pending_values[1]),
            )
            if drift > 0.0125 and self.tumbler_settle_retries < 3:
                self.tumbler_pending_values = changed
                self.tumbler_settle_retries += 1
                self.status.set("Жду, пока стрелки остановятся перед замером.")
                self.action_job = self.after(180, self.run_tumbler_step)
                return
            stable = (
                (changed[0] + self.tumbler_pending_values[0]) / 2.0,
                (changed[1] + self.tumbler_pending_values[1]) / 2.0,
            )
            effect = (
                stable[0] - self.tumbler_baseline[0],
                stable[1] - self.tumbler_baseline[1],
            )
            self.tumbler_effects.append(effect)
            # Возвращаем только что измеренный тумблер в ноль. Так ошибка
            # анимации или одного раннего кадра не накапливается к 6–8 рычагу.
            if not self.click_tumbler(self.tumbler_index):
                self.stop("Windows не принял обратный клик по тумблеру.")
                return
            self.tumbler_phase = "measure_off"
            self.tumbler_click_retries = 0
            self.action_job = self.after(700, self.run_tumbler_step)
            return
        if self.tumbler_phase == "measure_off":
            states = self.read_tumbler_switch_states()
            if states is None:
                self.stop("Не удалось проверить возврат тумблера в ноль.")
                return
            if states[self.tumbler_index]:
                if self.tumbler_click_retries >= 1 or not self.click_tumbler(self.tumbler_index):
                    self.stop("Тумблер не вернулся в ноль после повторного клика.")
                    return
                self.tumbler_click_retries += 1
                self.action_job = self.after(700, self.run_tumbler_step)
                return
            self.tumbler_index += 1
            self.tumbler_phase = "measure_on"
            self.target.set(f"Измеряю тумблеры · {self.tumbler_index}/{total}")
            self.status.set("Каждый прирост считаю от одного и того же нулевого положения.")
            self.action_job = self.after(180, self.run_tumbler_step)
            return
        if self.tumbler_phase == "apply":
            if self.tumbler_index >= len(self.tumbler_apply):
                self.tumbler_phase = "verify"
                self.action_job = self.after(600, self.run_tumbler_step)
                return
            if not self.click_tumbler(self.tumbler_apply[self.tumbler_index]):
                self.stop("Windows не принял клик по тумблеру.")
                return
            self.tumbler_index += 1
            self.action_job = self.after(260, self.run_tumbler_step)
            return
        if self.tumbler_phase == "verify":
            final_gauges = self.read_tumbler_gauges()
            if final_gauges is None:
                self.target.set("Тумблеры настроены")
                self.resume_monitoring("Тумблеры применены. Продолжаю ждать следующий вызов.", 1400)
                return
            final_values, final_targets = final_gauges
            left_error = self.tumbler_distance(final_values[0], final_targets[0])
            right_error = self.tumbler_distance(final_values[1], final_targets[1])
            if max(left_error, right_error) <= 0.018:
                self.target.set("Тумблеры настроены")
                self.resume_monitoring(
                    "Тумблеры настроены: обе стрелки в красных зонах. Продолжаю автодетект.",
                    1400,
                )
                return
            actual_states = self.read_tumbler_switch_states()
            if actual_states is not None:
                self.tumbler_solution = [index for index, enabled in enumerate(actual_states) if enabled]
            if self.tumbler_correction_round < 2:
                anchored_left, anchored_right = final_values
                for index in self.tumbler_solution:
                    anchored_left -= self.tumbler_effects[index][0]
                    anchored_right -= self.tumbler_effects[index][1]
                corrected, error = self.choose_tumbler_solution(
                    (anchored_left, anchored_right), self.tumbler_effects, final_targets,
                )
                previous = set(self.tumbler_solution)
                desired = set(corrected)
                changes = sorted(previous ^ desired)
                if changes:
                    self.tumbler_solution = corrected
                    self.tumbler_apply = changes
                    self.tumbler_index = 0
                    self.tumbler_phase = "apply"
                    self.tumbler_correction_round += 1
                    self.target.set(
                        f"Уточняю сочетание · попытка {self.tumbler_correction_round}/2"
                    )
                    self.status.set(
                        f"Учитываю фактические стрелки; осталось до зон {error * 100:.1f}%."
                    )
                    self.action_job = self.after(220, self.run_tumbler_step)
                    return
            self.stop(
                f"Сочетание не подтверждено: промах A {left_error * 100:.1f}%, "
                f"V {right_error * 100:.1f}%. Ввод остановлен."
            )
