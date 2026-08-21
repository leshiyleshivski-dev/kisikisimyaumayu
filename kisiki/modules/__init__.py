"""Secret game modules exposed by the application."""

from .bongo import BongoModule
from .buff import BuffTimingModule
from .miner import MinerModule
from .roulette import RouletteModule
from .volt import ElectricianModule

__all__ = [
    "RouletteModule", "BongoModule", "BuffTimingModule", "ElectricianModule",
    "MinerModule",
]
