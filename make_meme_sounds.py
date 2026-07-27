"""Генерирует шесть коротких оригинальных мультяшных WAV-эффектов."""

from __future__ import annotations

import math
import random
import struct
import wave
from pathlib import Path


RATE = 22050
OUT = Path(__file__).resolve().parent / "sounds"
random.seed(20260724)


def envelope(t: float, duration: float, attack: float = 0.01) -> float:
    if t < attack:
        return t / attack
    return max(0.0, 1 - (t - attack) / max(0.001, duration - attack))


def make(name: str, duration: float, sample) -> None:
    values = []
    for i in range(int(RATE * duration)):
        t = i / RATE
        value = max(-1.0, min(1.0, sample(t)))
        values.append(struct.pack("<h", int(value * 32767)))
    with wave.open(str(OUT / name), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(RATE)
        file.writeframes(b"".join(values))


def sine(freq, t):
    return math.sin(math.tau * freq * t)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # Самоуверенный cartoon-pop: две упругие ноты.
    make("smug_pop.wav", 0.26, lambda t: 0.55 * envelope(t, 0.26) * sine(510 + 750 * t, t) + (0.22 * envelope(t - .10, .16) * sine(1280, t) if t > .10 else 0))

    # Бонго: два низких удара с затухающим шумом.
    def bongo(t):
        result = 0.0
        for start, freq in ((0.0, 180), (.13, 245)):
            if start <= t < start + .15:
                u = t - start
                result += (0.58 * sine(freq * (1 - 0.35 * u), u) + 0.08 * random.uniform(-1, 1)) * envelope(u, .15, .002)
        return result
    make("bongo_bonk.wav", 0.31, bongo)

    # Апельсин: классический мультяшный boing вниз-вверх.
    make("orange_boing.wav", 0.36, lambda t: 0.62 * envelope(t, .36, .004) * sine(780 - 1500 * t + 2100 * t * t, t))

    # Бафф: короткий драматичный басовый удар и металлический акцент.
    make("buff_dramatic.wav", 0.36, lambda t: 0.72 * envelope(t, .22, .002) * sine(130 - 230 * t, t) + (0.22 * envelope(t - .11, .20) * sine(920, t) if t > .11 else 0))

    # Клавишник: три клик-удара и «компьютерный» звон.
    def keyboard(t):
        result = 0.0
        for start in (0.0, .07, .14):
            if start <= t < start + .05:
                u = t - start
                result += (0.34 * random.uniform(-1, 1) + 0.18 * sine(1900, u)) * envelope(u, .05, .001)
        if t > .19:
            result += .30 * envelope(t - .19, .15, .004) * sine(1319, t)
        return result
    make("keyboard_clack.wav", .35, keyboard)

    # Grumpy: комичный descending fail-trombone без заимствованной записи.
    make("grumpy_fail.wav", .48, lambda t: 0.58 * envelope(t, .48, .01) * (sine(360 - 420 * t, t) + .22 * sine(2 * (360 - 420 * t), t)))


if __name__ == "__main__":
    main()
