"""Explicit communication packet data."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Packet:
    sender: str
    recipient: str
    payload: Any
    sent_time: float
    deliver_time: float

    @property
    def age(self) -> float:
        """Scheduled channel latency, in seconds (not wall-clock packet age)."""
        return self.deliver_time - self.sent_time
