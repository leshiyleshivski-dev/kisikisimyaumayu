"""Создаёт ICO с PNG апельсинового котика без сторонних библиотек."""

from pathlib import Path
import struct


root = Path(__file__).resolve().parent
png = (root / "kiski kartinki" / "03_orange_meme_cat.png").read_bytes()
icon = root / "orange_cat.ico"

# Windows поддерживает PNG внутри ICO. Нулевые width/height означают 256 px.
header = struct.pack("<HHH", 0, 1, 1)
entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22)
icon.write_bytes(header + entry + png)
print(icon)
