"""AudioSeal embed/detect adapters, causal streaming checkpoints.

Chunked calls into the non-causal checkpoint reset model state every 0.5 s —
a measured 10x amplitude discontinuity at each chunk boundary. The streaming
checkpoint with a held-open cache embeds continuously.

WM_ALPHA trades imperceptibility for channel robustness (published sweep:
1.0 -> 0.671 pristine detector confidence, 3.0 -> 0.999 including AAC) —
calibrate on-site like the similarity threshold.
"""

import numpy as np

from truemic import config
from truemic.tokens import TOKEN_BITS


class Watermarker:
    def __init__(self, device: str = "cpu", alpha: float | None = None):
        from audioseal import AudioSeal

        self._device = device
        self._alpha = config.WM_ALPHA if alpha is None else alpha
        self._model = AudioSeal.load_generator("audioseal_wm_streaming").to(device).eval()
        # streaming_forever, not the streaming() context manager: the context's
        # teardown crashes in moshi's _stop_streaming (torch isinstance bug) and
        # a half-torn cache corrupts every later embed. Forever-mode has no
        # teardown; reset_streaming() re-inits state in place.
        self._model.encoder.streaming_forever(1)

    def embed(self, chunk: np.ndarray, token: int) -> np.ndarray:
        import torch

        wav = torch.from_numpy(chunk.astype(np.float32))[None, None, :].to(self._device)
        message = torch.tensor(
            [_token_to_bits(token)], dtype=torch.int32, device=self._device
        )
        with torch.no_grad():
            watermark = self._model.get_watermark(wav, message=message)
        n = watermark.shape[-1]  # may run short while the cache warms
        return (wav[..., :n] + self._alpha * watermark).squeeze().cpu().numpy()

    def reset(self, alpha: float | None = None) -> None:
        """Fresh causal cache (and optionally a new alpha) — for offline tools;
        the live sender never needs this."""
        if alpha is not None:
            self._alpha = alpha
        self._model.encoder.reset_streaming()


PHASE_PERIOD = 320  # the streaming detector's conv hop: decode only works aligned
LOCK_CRISPNESS = 0.25  # bit crispness at true phase ~0.7, one sample off ~0.05
SEARCH_BATCH = 32
SCAN_COOLDOWN_START = 1.5  # full 320-phase scans are ~40x a tick; back off while unlocked
SCAN_COOLDOWN_MAX = 8.0
SILENCE_RMS = 1e-4  # silence can't carry a watermark; don't scan it


class WatermarkReader:
    """Detector plus sample-exact phase lock.

    Measured: the streaming pair decodes message bits ONLY at sample-exact
    alignment with the embedder's 320-sample frame grid (bit crispness 0.75
    at the true phase, 0.05 one sample off; detection score alone plateaus
    over +-16 and cannot be the criterion). Codecs add constant sub-grid
    delays (Opus: 215 samples) — but at the right phase the token survives
    Opus 32k with zero bit errors. So: one-time batched scan of all 320
    phases by crispness, then +-1 tracking (phase is constant on a cable,
    drifts ~1 sample/s across clock domains such as the air-gap beat).
    DETECT_HOP must stay a multiple of PHASE_PERIOD or the lock walks.
    """

    def __init__(self, device: str = "cpu"):
        from audioseal import AudioSeal

        self._device = device
        self._detector = (
            AudioSeal.load_detector("audioseal_detector_streaming").to(device).eval()
        )
        self._phase: int | None = None
        self._next_scan_at = 0.0
        self._scan_cooldown = SCAN_COOLDOWN_START

    def read_soft(self, window: np.ndarray) -> tuple[float, np.ndarray]:
        """(fraction-of-samples detection score, 16 soft bit probabilities).

        The shipped checkpoints have their input normalizer disabled, so we
        RMS-normalize here — otherwise quiet received audio silently degrades.
        """
        import torch

        window = window.astype(np.float32)
        rms = float(np.sqrt(np.mean(window**2)))
        if rms > 1e-5:
            window = window * (config.WM_TARGET_RMS / rms)
        wav = torch.from_numpy(window)[None, None, :].to(self._device)
        with torch.no_grad():
            result, message = self._detector(wav)
        score = float((result[:, 1, :] > 0.5).float().mean())
        return score, message.squeeze().cpu().numpy()

    def read(self, window: np.ndarray) -> tuple[float, int]:
        score, probabilities = self.read_soft(window)
        return score, _bits_to_token([int(p >= 0.5) for p in probabilities])

    def read_locked(self, window: np.ndarray) -> tuple[float, np.ndarray]:
        """Phase-searched read. `window` must carry PHASE_PERIOD extra samples
        beyond the detection length."""
        assert len(window) > PHASE_PERIOD, "window too short for phase search"
        span = len(window) - PHASE_PERIOD

        had_lock = self._phase is not None
        if had_lock:
            phases = [(self._phase + d) % PHASE_PERIOD for d in (-1, 0, 1)]
            best, score, soft, crisp = self._best_phase(window, span, phases)
            if crisp >= LOCK_CRISPNESS:
                self._phase = best
                self._scan_cooldown = SCAN_COOLDOWN_START
                return score, soft
            self._phase = None
        import time

        now = time.monotonic()
        silent = float(np.sqrt(np.mean(window**2))) < SILENCE_RMS
        # a lock that just broke (stream glitch shifted the phase) rescans
        # immediately — the cooldown only paces repeated fruitless scans
        if silent or (now < self._next_scan_at and not had_lock):
            return self.read_soft(window[:span])
        self._next_scan_at = now + self._scan_cooldown
        self._scan_cooldown = min(self._scan_cooldown * 2, SCAN_COOLDOWN_MAX)
        best, score, soft, crisp = self._best_phase(
            window, span, list(range(PHASE_PERIOD))
        )
        if crisp >= LOCK_CRISPNESS:
            self._phase = best
            self._scan_cooldown = SCAN_COOLDOWN_START
        return score, soft

    def _best_phase(self, window, span, phases):
        """Batch-detect the given phase offsets, pick by bit crispness."""
        import torch

        best = (phases[0], 0.0, np.full(TOKEN_BITS, 0.5, dtype=np.float32), -1.0)
        for i in range(0, len(phases), SEARCH_BATCH):
            batch_phases = phases[i : i + SEARCH_BATCH]
            stack = np.stack([window[p : p + span] for p in batch_phases])
            rms = np.sqrt(np.mean(stack**2, axis=1, keepdims=True))
            stack = np.where(rms > 1e-5, stack * (config.WM_TARGET_RMS / rms), stack)
            wav = torch.from_numpy(stack[:, None, :].astype(np.float32)).to(self._device)
            with torch.no_grad():
                result, message = self._detector(wav)
            scores = (result[:, 1, :] > 0.5).float().mean(dim=1).cpu().numpy()
            probs = message.cpu().numpy()
            crisps = np.abs(probs - 0.5).mean(axis=1) * 2
            top = int(np.argmax(crisps))
            if crisps[top] > best[3]:
                best = (batch_phases[top], float(scores[top]), probs[top], float(crisps[top]))
        return best

    def unlock(self) -> None:
        self._phase = None
        self._next_scan_at = 0.0
        self._scan_cooldown = SCAN_COOLDOWN_START


def majority_token(soft_probabilities: list[np.ndarray]) -> int:
    """Sum (p - 0.5) per bit across detection ticks, then threshold: two or
    three ticks of soft accumulation fix an isolated garbled tick, letting
    token matching stay exact."""
    votes = np.sum(np.stack(soft_probabilities) - 0.5, axis=0)
    return _bits_to_token([int(v >= 0) for v in votes])


def _token_to_bits(token: int) -> list[int]:
    return [(token >> (TOKEN_BITS - 1 - i)) & 1 for i in range(TOKEN_BITS)]


def _bits_to_token(bits: list[int]) -> int:
    token = 0
    for bit in bits:
        token = (token << 1) | int(bit)
    return token
