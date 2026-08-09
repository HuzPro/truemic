"""Microphone identity from near-silence frames.

Forensic mic identification (FFT histograms of quiet frames, ~93% over 7 mics
in the literature) inverted into a live capture-chain check: the enrolled
mic's noise floor has a spectral shape; an injected clone played through a
virtual cable doesn't have it. Indicator by default — physics, not proof.
"""

import numpy as np

from truemic import config

QUIET_VAD_PROBABILITY = 0.35
MIN_QUIET_FRAMES = 20
EMA_WEIGHT = 0.05
ZERO_HISTORY_FRAMES = 160  # ~5 s of 32 ms frames


def spectrum_of_frames(frames: np.ndarray) -> np.ndarray:
    """Mean log-magnitude spectrum of (N, frame) quiet frames, L2-normalized."""
    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    mean_log = np.log10(magnitude + 1e-10).mean(axis=0)
    return mean_log / np.linalg.norm(mean_log)


def fingerprint_from_recording(recording: np.ndarray, vad) -> np.ndarray | None:
    """Enrollment-side: fingerprint the mic from the quiet gaps between words."""
    import torch

    frame_size = config.VAD_FRAME_SAMPLES
    n_frames = len(recording) // frame_size
    frames = recording[: n_frames * frame_size].reshape(n_frames, frame_size)
    vad._model.reset_states()
    with torch.no_grad():
        probs = np.array(
            [vad._model(torch.from_numpy(f), config.SAMPLE_RATE).item() for f in frames]
        )
    vad._model.reset_states()
    quiet = frames[probs < QUIET_VAD_PROBABILITY]
    quiet = quiet[np.abs(quiet).max(axis=1) > 0]  # digital silence carries no mic
    if len(quiet) < MIN_QUIET_FRAMES:
        energies = frames.std(axis=1)
        quiet = frames[energies <= np.percentile(energies, 10)]
    if len(quiet) < 3:
        return None
    return spectrum_of_frames(quiet)


class RollingMicPrint:
    """Runtime side: EMA the live quiet-frame spectrum, compare to enrollment."""

    def __init__(self, enrolled: np.ndarray | None):
        self._enrolled = enrolled
        self._live: np.ndarray | None = None
        self._quiet_frames_seen = 0
        self._zero_flags: list[bool] = []

    def update(self, frame: np.ndarray, voiced: bool) -> None:
        is_zero = float(np.abs(frame).max()) == 0.0
        self._zero_flags.append(is_zero)
        if len(self._zero_flags) > ZERO_HISTORY_FRAMES:
            self._zero_flags.pop(0)
        if voiced or is_zero or self._enrolled is None:
            return
        spectrum = spectrum_of_frames(frame[None])
        self._live = (
            spectrum
            if self._live is None
            else (1 - EMA_WEIGHT) * self._live + EMA_WEIGHT * spectrum
        )
        self._quiet_frames_seen += 1

    def match(self) -> float | None:
        """Cosine vs the enrolled mic, None until there's enough evidence."""
        if self._enrolled is None or self._quiet_frames_seen < MIN_QUIET_FRAMES:
            return None
        live = self._live / np.linalg.norm(self._live)
        return float(np.dot(live, self._enrolled))

    def zero_fraction(self) -> float:
        """Bit-exact zero frames — real mics have a noise floor; injected
        virtual-cable silence doesn't."""
        if not self._zero_flags:
            return 0.0
        return sum(self._zero_flags) / len(self._zero_flags)
