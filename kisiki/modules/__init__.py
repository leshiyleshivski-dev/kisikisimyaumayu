"""Secret game modules exposed by the application."""

from .blackjack import BlackjackModule
from .phone import PhoneModule
from .builder import BuilderModule
from .miner import MinerModule
from .roulette import RouletteModule
from .race_bettor import RaceBettorModule
from .slot_spinner import SlotSpinnerModule
from .poker import PokerModule
from .volt import ElectricianModule

__all__ = [
    "RouletteModule", "PhoneModule", "BuilderModule", "ElectricianModule",
    "MinerModule",
    "RaceBettorModule",
    "SlotSpinnerModule",
    "BlackjackModule",
    "PokerModule",
]
