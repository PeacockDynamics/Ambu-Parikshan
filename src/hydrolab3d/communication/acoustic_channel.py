"""Selective-fidelity underwater communication channel."""

from __future__ import annotations

import heapq
import copy
from dataclasses import dataclass, field

import numpy as np

from .packet import Packet


@dataclass
class AcousticChannel:
    """Range, fixed latency, seeded loss, and blackout-aware packet channel."""

    max_range: float
    latency: float = 0.0
    packet_loss: float = 0.0
    seed: int | None = None
    blackouts: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.max_range, self.latency, self.packet_loss = float(self.max_range), float(self.latency), float(self.packet_loss)
        if not np.isfinite(self.max_range) or self.max_range <= 0 or not np.isfinite(self.latency) or self.latency < 0 or not 0 <= self.packet_loss <= 1:
            raise ValueError("max_range must be positive, latency non-negative, and packet_loss in [0, 1]")
        self._rng = np.random.default_rng(self.seed)
        self._queue: list[tuple[float, int, Packet]] = []
        self._sequence = 0

    def reset(self) -> None:
        """Clear in-flight packets and restore seeded loss draws.

        Configured blackout membership is retained; it is channel configuration,
        not transient packet state.
        """
        self._rng = np.random.default_rng(self.seed); self._queue.clear(); self._sequence = 0

    def send(self, sender: str, recipient: str, payload, sent_time: float, sender_position, recipient_position) -> bool:
        if not isinstance(sender, str) or not isinstance(recipient, str) or not sender or not recipient:
            raise ValueError("sender and recipient must be non-empty strings")
        sent_time = float(sent_time)
        if not np.isfinite(sent_time) or sent_time < 0:
            raise ValueError("sent_time must be finite and non-negative")
        sender_position, recipient_position = np.asarray(sender_position, dtype=float), np.asarray(recipient_position, dtype=float)
        if sender_position.shape != (3,) or recipient_position.shape != (3,) or not np.all(np.isfinite(sender_position)) or not np.all(np.isfinite(recipient_position)):
            raise ValueError("positions must be finite vectors with shape (3,)")
        if sender in self.blackouts or recipient in self.blackouts or np.linalg.norm(sender_position - recipient_position) > self.max_range or self._rng.random() < self.packet_loss:
            return False
        # Packets represent the message at send time, not a mutable sender view.
        packet = Packet(sender, recipient, copy.deepcopy(payload), sent_time, sent_time + self.latency)
        heapq.heappush(self._queue, (packet.deliver_time, self._sequence, packet)); self._sequence += 1
        return True

    def receive(self, time: float) -> list[Packet]:
        time = float(time)
        if not np.isfinite(time) or time < 0:
            raise ValueError("receive time must be finite and non-negative")
        packets = []
        while self._queue and self._queue[0][0] <= time:
            packets.append(heapq.heappop(self._queue)[2])
        return packets
