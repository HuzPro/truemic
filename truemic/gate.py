"""Fail-closed delay-line gate.

Audio leaves the gate DELAY samples after it entered, by which time the
decision loop has scored the rolling window that contains it. Optimistic
gating would leak 1-3 s of impostor audio before the first verdict lands.
"""

import statistics
from collections import deque

import numpy as np


class DelayGate:
    def __init__(self, delay_samples: int, fade_samples: int):
        self._fifo = np.zeros(delay_samples, dtype=np.float32)
        self._fade_samples = fade_samples
        self._gain = 0.0
        self._target = 0.0

    def set_allow(self, allow: bool) -> None:
        self._target = 1.0 if allow else 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        self._fifo = np.concatenate([self._fifo, block.astype(np.float32, copy=False)])
        out = self._fifo[: len(block)].copy()
        self._fifo = self._fifo[len(block):]
        return out * self._envelope(len(out))

    @property
    def gain(self) -> float:
        return self._gain

    def _envelope(self, n: int) -> np.ndarray:
        if self._gain == self._target:
            linear = np.full(n, self._gain, dtype=np.float32)
        else:
            step = 1.0 / self._fade_samples
            direction = 1.0 if self._target > self._gain else -1.0
            ramp = self._gain + direction * step * np.arange(1, n + 1, dtype=np.float32)
            lo, hi = sorted((self._gain, self._target))
            linear = np.clip(ramp, lo, hi)
            self._gain = float(linear[-1])
        return 0.5 - 0.5 * np.cos(np.pi * linear)


class GateDecision:
    """Security-asymmetric verdicts (the speech-enhancement field's 10:1
    keep-the-speech bias, inverted): opening needs consecutive median-smoothed
    agreement on the reliable long window; closing is immediate on a raw drop
    of EITHER the long window or the recent one. The recent-window close is
    what stops an impostor riding the tail of the genuine speaker's rolling
    window. A silence timeout fails the gate closed."""

    MEDIAN_SPAN = 3
    RECENT_CLOSE_CONFIRMATIONS = 2  # the 1 s window is noisy; one dip is not an impostor

    def __init__(
        self,
        open_threshold: float,
        close_threshold: float,
        silence_timeout: float = 3.0,
        open_confirmations: int = 2,
        recent_close_threshold: float | None = None,
    ):
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        # 1 s scores live on their OWN scale, well below 3 s scores for the same
        # speaker (measured live: window 0.95, recent 0.6) — the recent path is
        # inert until calibration supplies its scale.
        self.recent_close_threshold = recent_close_threshold
        self.silence_timeout = silence_timeout
        self.open_confirmations = open_confirmations
        self._open = False
        self._streak = 0
        self._recent_low_streak = 0
        self._recent_scores: deque[float] = deque(maxlen=self.MEDIAN_SPAN)
        self._last_voice_at: float | None = None

    @property
    def is_open(self) -> bool:
        return self._open

    def update(
        self, window_score: float | None, recent_score: float | None, now: float
    ) -> bool:
        if window_score is None:
            return self._on_silence(now)
        self._last_voice_at = now
        if self._open:
            if self._recent_disagrees(recent_score):
                self._recent_low_streak += 1
            else:
                self._recent_low_streak = 0
            drops = (
                window_score < self.close_threshold
                or self._recent_low_streak >= self.RECENT_CLOSE_CONFIRMATIONS
            )
            if drops:
                self._close()
            return self._open
        self._recent_scores.append(window_score)
        if statistics.median(self._recent_scores) >= self.open_threshold:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.open_confirmations and not self._recent_disagrees(
            recent_score
        ):
            self._open = True
        return self._open

    def _recent_disagrees(self, recent_score: float | None) -> bool:
        if self.recent_close_threshold is None or recent_score is None:
            return False
        return recent_score < self.recent_close_threshold

    def _on_silence(self, now: float) -> bool:
        silent_too_long = (
            self._last_voice_at is None or now - self._last_voice_at > self.silence_timeout
        )
        if silent_too_long and self._open:
            self._close()
        return self._open

    def _close(self) -> None:
        self._open = False
        self._streak = 0
        self._recent_low_streak = 0
        self._recent_scores.clear()
