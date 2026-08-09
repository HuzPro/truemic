"""Offline end-to-end check: token → AudioSeal embed → gate → detect → verify.

Needs the AudioSeal weights (downloaded on first import) — no mic, no cable.
Run with: python -m pytest tests/test_roundtrip.py -v
"""

import time

import numpy as np
import pytest

from truemic import config
from truemic.audio import synthetic_speech as speechlike
from truemic.gate import DelayGate
from truemic.tokens import current_slot, token_for_slot, verify_token

KEY = b"\x07" * 32


@pytest.fixture(scope="module")
def watermarker():
    from truemic.watermark import Watermarker

    return Watermarker()


@pytest.fixture(scope="module")
def reader():
    from truemic.watermark import WatermarkReader

    return WatermarkReader()


def test_embedded_token_survives_the_open_gate(watermarker, reader):
    now = time.time()
    slot = current_slot(now, config.TOKEN_SLOT_SECONDS)
    token = token_for_slot(KEY, slot)

    gate = DelayGate(
        delay_samples=int(config.GATE_DELAY_SECONDS * config.SAMPLE_RATE),
        fade_samples=int(config.CROSSFADE_MS / 1000 * config.SAMPLE_RATE),
    )
    gate.set_allow(True)

    chunk_samples = int(config.WM_CHUNK_SECONDS * config.SAMPLE_RATE)
    audio = speechlike(4.0)
    emitted = []
    for start in range(0, len(audio), chunk_samples):
        chunk = watermarker.embed(audio[start : start + chunk_samples], token)
        emitted.append(gate.process(chunk))
    out = np.concatenate(emitted)

    window = out[-int(config.DETECT_WINDOW_SECONDS * config.SAMPLE_RATE) :]
    probability, detected = reader.read(window)
    assert probability >= config.WM_DETECT_THRESHOLD
    assert (detected ^ token).bit_count() <= config.TOKEN_MAX_BIT_ERRORS
    assert (
        verify_token(
            KEY, detected, now, config.TOKEN_SLOT_SECONDS,
            max_bit_errors=config.TOKEN_MAX_BIT_ERRORS,
        )
        == slot
    )


def test_muted_audio_carries_no_watermark(watermarker, reader):
    token = token_for_slot(KEY, 42)
    gate = DelayGate(
        delay_samples=int(config.GATE_DELAY_SECONDS * config.SAMPLE_RATE),
        fade_samples=int(config.CROSSFADE_MS / 1000 * config.SAMPLE_RATE),
    )
    gate.set_allow(False)

    chunk_samples = int(config.WM_CHUNK_SECONDS * config.SAMPLE_RATE)
    audio = speechlike(4.0)
    emitted = []
    for start in range(0, len(audio), chunk_samples):
        chunk = watermarker.embed(audio[start : start + chunk_samples], token)
        emitted.append(gate.process(chunk))
    out = np.concatenate(emitted)

    assert np.allclose(out, 0.0)
    probability, _ = reader.read(out[-int(1.5 * config.SAMPLE_RATE) :])
    assert probability < config.WM_DETECT_THRESHOLD


def test_stale_token_is_rejected(watermarker, reader):
    stale_slot = current_slot(time.time(), config.TOKEN_SLOT_SECONDS) - 10
    token = token_for_slot(KEY, stale_slot)
    audio = speechlike(2.0)
    marked = watermarker.embed(audio, token)
    probability, detected = reader.read(marked)
    assert probability >= config.WM_DETECT_THRESHOLD
    assert (detected ^ token).bit_count() <= config.TOKEN_MAX_BIT_ERRORS
    assert (
        verify_token(
            KEY, detected, time.time(), config.TOKEN_SLOT_SECONDS,
            max_bit_errors=config.TOKEN_MAX_BIT_ERRORS,
        )
        is None
    )


def test_soft_attenuation_would_leak_the_token(watermarker, reader):
    """Documents the landmine that makes the hard mute mandatory: AudioSeal
    survives uniform attenuation, so a -20 dB 'polite' floor (field range is
    -12..-30 dB) hands the receiver a perfectly readable token. Measured
    boundary: survives -26 dB, dies by -32 dB — inaudible anyway. If this
    test ever fails, a floor became safe."""
    token = token_for_slot(KEY, 99)
    marked = watermarker.embed(speechlike(2.0), token)
    probability, detected = reader.read(marked * 0.1)
    assert probability >= config.WM_DETECT_THRESHOLD
    assert (detected ^ token).bit_count() <= config.TOKEN_MAX_BIT_ERRORS


def test_phase_lock_recovers_a_misaligned_capture(watermarker, reader):
    """The streaming pair decodes bits only at sample-exact alignment with the
    embedder's 320-sample grid. A real capture (or a codec's constant delay)
    starts at an arbitrary phase, so the receiver scans by bit crispness."""
    from truemic.watermark import PHASE_PERIOD, majority_token

    token = token_for_slot(KEY, 7)
    marked = watermarker.embed(speechlike(4.0), token)
    misaligned = marked[137:]  # arbitrary sub-grid shift, like a codec delay

    span = int(config.DETECT_WINDOW_SECONDS * config.SAMPLE_RATE)
    _, naive_soft = reader.read_soft(misaligned[:span])
    assert majority_token([naive_soft]) != token  # documents the sensitivity

    reader.unlock()
    locked_score, soft = reader.read_locked(misaligned[: span + PHASE_PERIOD])
    assert locked_score >= 0.9
    assert majority_token([soft]) == token
    assert reader._phase == (PHASE_PERIOD - 137) % PHASE_PERIOD


def test_embed_runs_faster_than_real_time(watermarker):
    chunk = speechlike(config.WM_CHUNK_SECONDS)
    watermarker.embed(chunk, 1)  # warm-up
    started = time.perf_counter()
    for _ in range(4):
        watermarker.embed(chunk, 1)
    per_chunk = (time.perf_counter() - started) / 4
    print(f"\nembed: {per_chunk:.3f}s per {config.WM_CHUNK_SECONDS}s chunk "
          f"(load {per_chunk / config.WM_CHUNK_SECONDS:.0%})")
    assert per_chunk < config.WM_CHUNK_SECONDS
