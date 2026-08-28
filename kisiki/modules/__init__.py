"""Secret game modules exposed by the application."""

from .blackjack import BlackjackModule
from .phone import PhoneModule
from .builder import BuilderModule
from .lumberjack import LumberjackModule
from .miner import MinerModule, fresh_daily_stats
from .roulette import RouletteModule
from .race_bettor import RaceBettorModule
from .slot_spinner import SlotSpinnerModule
from .poker import PokerModule
from .volt import ElectricianModule

__all__ = [
    "RouletteModule", "PhoneModule", "BuilderModule", "ElectricianModule",
    "MinerModule",
    "LumberjackModule",
    "fresh_daily_stats",
    "RaceBettorModule",
    "SlotSpinnerModule",
    "BlackjackModule",
    "PokerModule",
]
