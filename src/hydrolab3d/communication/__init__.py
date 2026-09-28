"""Deterministic lightweight underwater communication."""

from .acoustic_channel import AcousticChannel
from .packet import Packet

__all__ = ["AcousticChannel", "Packet"]
