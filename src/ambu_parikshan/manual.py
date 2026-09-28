"""Backend-independent keyboard-to-wrench mapping."""

import numpy as np


class ManualActionMapper:
    """Maintain pressed keys and emit the standard Simulator wrench."""
    _axes = {"w": (0, 1), "s": (0, -1), "a": (1, -1), "d": (1, 1), "r": (2, 1), "f": (2, -1), "q": (5, 1), "e": (5, -1)}
    def __init__(self, force=10.0, moment=5.0): self.force, self.moment, self._pressed = float(force), float(moment), set()
    def press(self, key):
        if key == "space": self._pressed.clear()
        elif key in self._axes: self._pressed.add(key)
    def release(self, key): self._pressed.discard(key)
    def reset(self): self._pressed.clear()
    def action(self):
        result = np.zeros(6)
        for key in self._pressed:
            axis, sign = self._axes[key]; result[axis] += sign * (self.force if axis < 3 else self.moment)
        return result
