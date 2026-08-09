"""Shared audio plumbing: thread-safe sample FIFO, rolling window, recording."""

import threading

import numpy as np
import sounddevice as sd

from truemic import config


class SampleQueue:
    """Float32 sample FIFO between the processing thread and an audio callback.

    Underruns pad with zeros; `padded_samples` counts them — every pad inserts
    samples into the output timeline and shifts the receiver's watermark phase,
    so this number should stay at zero on a healthy input device."""

    def __init__(self):
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.padded_samples = 0

    def put(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buffer = np.concatenate([self._buffer, samples])

    def pop(self, n: int) -> np.ndarray:
        with self._lock:
            taken, self._buffer = self._buffer[:n], self._buffer[n:]
            if len(taken) < n:
                self.padded_samples += n - len(taken)
        if len(taken) < n:
            taken = np.concatenate([taken, np.zeros(n - len(taken), dtype=np.float32)])
        return taken

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class RollingWindow:
    def __init__(self, seconds: float, sample_rate: int = config.SAMPLE_RATE):
        self._samples = int(seconds * sample_rate)
        self._buffer = np.zeros(0, dtype=np.float32)

    def append(self, block: np.ndarray) -> None:
        self._buffer = np.concatenate([self._buffer, block])[-self._samples :]

    def window(self) -> np.ndarray:
        return self._buffer.copy()

    @property
    def full(self) -> bool:
        return len(self._buffer) >= self._samples


def synthetic_speech(seconds: float) -> np.ndarray:
    """Deterministic speech-shaped signal: AM harmonics (voicing) plus a
    band-limited noise bed (fricative energy). The broadband part matters —
    watermark message bits need spectral room, like real speech has."""
    from scipy.signal import butter, sosfilt

    t = np.arange(int(seconds * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    carrier = sum(np.sin(2 * np.pi * f * t) / i for i, f in enumerate((140, 280, 420, 560), 1))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)
    bandpass = butter(4, (1000, 6000), btype="band", fs=config.SAMPLE_RATE, output="sos")
    fricatives = sosfilt(bandpass, np.random.default_rng(1234).standard_normal(len(t)))
    return (0.12 * carrier * envelope + 0.03 * fricatives).astype(np.float32)


def add_noise_at_snr(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    signal_power = float(np.mean(wav**2)) + 1e-12
    noise_power = float(np.mean(noise**2))
    scale = np.float32(np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10))))
    return wav + scale * noise


def add_synthetic_reverb(
    wav: np.ndarray, rng: np.random.Generator, t60_seconds: float = 0.3
) -> np.ndarray:
    """Exponential-decay noise RIR — the standard cheap room simulator."""
    from scipy.signal import fftconvolve

    t = np.arange(int(t60_seconds * config.SAMPLE_RATE)) / config.SAMPLE_RATE
    rir = rng.standard_normal(len(t)).astype(np.float32) * np.exp(-t * (6.9 / t60_seconds))
    rir[0] = 1.0
    wet = fftconvolve(wav, rir)[: len(wav)].astype(np.float32)
    gain = np.float32(np.sqrt(np.mean(wav**2) / (np.mean(wet**2) + 1e-12)))
    return wet * gain


def record(seconds: float, device: int | None, prompt: str) -> np.ndarray:
    """Blocking mono 16 kHz recording with a live level meter."""
    print(prompt)
    blocks: list[np.ndarray] = []
    total = int(seconds * config.SAMPLE_RATE)

    def callback(indata, frames, time_info, status):
        blocks.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=config.BLOCK_SAMPLES,
        device=device,
        callback=callback,
    ):
        recorded = 0
        while recorded < total:
            sd.sleep(200)
            recorded = sum(len(b) for b in blocks)
            level = float(np.abs(blocks[-1]).max()) if blocks else 0.0
            bar = "#" * int(level * 40)
            print(f"\r  {recorded / config.SAMPLE_RATE:5.1f}s / {seconds:.0f}s  |{bar:<40}|", end="")
    print()
    return np.concatenate(blocks)[:total]
