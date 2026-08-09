"""TrueMic sender: mic → VAD → speaker verify → watermark → delay-line gate → virtual cable.

The watermark is embedded BEFORE the gate on purpose: muting multiplies the
watermarked signal to zero, so denied audio carries no token. Watermarking
after the gate would stamp a valid token onto the muted silence — an impostor
would show as verified. For the same reason the mute is a hard zero: AudioSeal
survives uniform attenuation, so a -40 dB "polite" floor would still leak a
readable token (tests/test_roundtrip.py proves it). Naturalness comes from
unwatermarked comfort noise mixed in after the gate instead.
"""

import argparse
import gc
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from truemic import config
from truemic import profile as profile_store
from truemic.audio import RollingWindow, SampleQueue
from truemic.devices import find_device
from truemic.gate import DelayGate, GateDecision
from truemic.micprint import RollingMicPrint
from truemic.tokens import current_slot, token_for_slot
from truemic.verify import VoiceActivity, create_verifier
from truemic.watermark import Watermarker

OUTPUT_PREFILL_SECONDS = 1.0  # jitter slack for network mics (WO Mic); adds latency


def main() -> None:
    args = parse_args()
    config.apply_profile("low-power" if args.low_power else "default")
    prof = profile_store.load(Path(args.profile_dir))
    if args.engine:
        prof.engine = args.engine
    open_threshold = prof.open_threshold
    close_threshold = prof.close_threshold
    if args.threshold is not None:
        open_threshold = args.threshold
        close_threshold = args.threshold - config.THRESHOLD_CLOSE_GAP

    print(f"Loading models (engine={prof.engine}, "
          f"open={open_threshold:.2f}, close={close_threshold:.2f})...")
    verifier = create_verifier(prof)
    vad = VoiceActivity()
    watermarker = None if args.no_watermark else Watermarker(alpha=args.alpha)
    mic_guard = RollingMicPrint(prof.mic_print)
    config.apply_torch_threads(config.SENDER_TORCH_THREADS)
    _realtime_hygiene()

    in_device = find_device(args.input, "input")
    out_device = find_device(args.output, "output")
    out_rate = args.out_rate
    out_block = round(config.BLOCK_SAMPLES * out_rate / config.SAMPLE_RATE)

    gate = DelayGate(
        delay_samples=int(config.GATE_DELAY_SECONDS * config.SAMPLE_RATE),
        fade_samples=int(config.CROSSFADE_MS / 1000 * config.SAMPLE_RATE),
    )
    decision = GateDecision(
        open_threshold=open_threshold,
        close_threshold=close_threshold,
        silence_timeout=config.SILENCE_TIMEOUT_SECONDS,
        open_confirmations=config.OPEN_CONFIRMATIONS,
        recent_close_threshold=(
            prof.recent_close_threshold if args.threshold is None else None
        ),
    )
    ring = RollingWindow(config.WINDOW_SECONDS)
    in_queue: queue.Queue[np.ndarray] = queue.Queue()
    out_queue = SampleQueue()
    out_queue.put(np.zeros(int(OUTPUT_PREFILL_SECONDS * out_rate), dtype=np.float32))
    comfort_rng = np.random.default_rng()
    comfort_amp = 10 ** (config.COMFORT_NOISE_DBFS / 20)

    def in_callback(indata, frames, time_info, status):
        in_queue.put(indata[:, 0].copy())

    def out_callback(outdata, frames, time_info, status):
        outdata[:, 0] = out_queue.pop(frames)

    wm_pending = np.zeros(0, dtype=np.float32)
    chunk_samples = int(config.WM_CHUNK_SECONDS * config.SAMPLE_RATE)
    recent_samples = int(config.RECENT_WINDOW_SECONDS * config.SAMPLE_RATE)
    # phase-offset decisions from watermark chunk boundaries so the two
    # inference bursts never land on the same tick
    last_decide = time.monotonic() + config.DECIDE_EVERY_SECONDS / 2
    embed_seconds_ema = 0.0
    muted_voiced_seconds = 0.0

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=config.BLOCK_SAMPLES,
        device=in_device,
        callback=in_callback,
    ), sd.OutputStream(
        samplerate=out_rate,
        channels=1,
        dtype="float32",
        blocksize=out_block,
        device=out_device,
        callback=out_callback,
    ):
        mode = "badge-only (no muting)" if args.badge_only else "gating"
        print(f"TrueMic {mode} for {prof.name!r} -> {args.output!r}. Ctrl+C stops.\n")
        try:
            while True:
                block = in_queue.get()
                ring.append(block)
                for frame, voiced in vad.feed(block):
                    mic_guard.update(frame, voiced)
                verifier.feed(block)
                wm_pending = np.concatenate([wm_pending, block])

                if len(wm_pending) >= chunk_samples:
                    chunk, wm_pending = wm_pending[:chunk_samples], wm_pending[chunk_samples:]
                    allow = decision.is_open
                    chunk_has_voice = (
                        vad.recent_voiced_fraction(config.WM_CHUNK_SECONDS) > 0
                    )
                    if (
                        watermarker is not None
                        and chunk_has_voice  # watermarking silence beeps and buys nothing
                        and (not args.badge_only or allow)
                    ):
                        started = time.perf_counter()
                        slot = current_slot(time.time(), config.TOKEN_SLOT_SECONDS)
                        chunk = watermarker.embed(chunk, token_for_slot(prof.token_key, slot))
                        embed_seconds_ema = (
                            0.8 * embed_seconds_ema + 0.2 * (time.perf_counter() - started)
                        )
                    if args.badge_only:
                        out = chunk
                    else:
                        out = gate.process(chunk)
                        out += (
                            comfort_amp
                            * (1.0 - gate.gain)
                            * comfort_rng.standard_normal(len(out)).astype(np.float32)
                        )
                    if out_rate != config.SAMPLE_RATE:
                        import soxr

                        out = soxr.resample(out, config.SAMPLE_RATE, out_rate).astype(np.float32)
                    out_queue.put(out)

                now = time.monotonic()
                if now - last_decide >= config.DECIDE_EVERY_SECONDS and ring.full:
                    last_decide = now
                    window = ring.window()
                    window_voiced = vad.recent_voiced_fraction(config.WINDOW_SECONDS)
                    window_score = (
                        verifier.score(window)
                        if window_voiced >= config.MIN_VOICED_FRACTION
                        else None
                    )
                    recent_score = None
                    if window_score is not None:
                        recent_voiced = vad.recent_voiced_fraction(config.RECENT_WINDOW_SECONDS)
                        if recent_voiced >= config.MIN_VOICED_FRACTION:
                            recent_score = verifier.score(window[-recent_samples:])
                    allow = decision.update(window_score, recent_score, now)
                    mic_match = mic_guard.match()
                    injected = mic_guard.zero_fraction() >= config.MIC_ZERO_FRACTION_ALARM
                    mic_alarm = injected or (
                        mic_match is not None and mic_match < config.MIC_MATCH_THRESHOLD
                    )
                    if args.strict_mic and mic_alarm:
                        allow = False
                    gate.set_allow(allow)
                    if not allow and window_voiced >= config.MIN_VOICED_FRACTION:
                        muted_voiced_seconds += config.DECIDE_EVERY_SECONDS
                    print_status(
                        allow, window_score, recent_score, window_voiced,
                        embed_seconds_ema, in_queue.qsize(), args.badge_only,
                        mic_alarm, injected,
                        out_queue.padded_samples / out_rate,
                    )
        except KeyboardInterrupt:
            print(f"\nStopped. Muted-while-voiced total: {muted_voiced_seconds:.1f}s "
                  "(includes impostor speech — that's the point).")


def _realtime_hygiene() -> None:
    gc.collect()
    gc.freeze()
    try:
        import psutil

        p = psutil.Process()
        if hasattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS"):
            p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        elif hasattr(p, "nice"):
            try:
                p.nice(-5)
            except Exception:
                pass
    except Exception:
        pass  # priority is best-effort; HIGH would starve a 2-core shell



def print_status(
    allow: bool,
    window_score: float | None,
    recent_score: float | None,
    voiced: float,
    embed_seconds: float,
    backlog_blocks: int,
    badge_only: bool,
    mic_alarm: bool = False,
    injected: bool = False,
    padded_seconds: float = 0.0,
) -> None:
    state = ("SIGN " if allow else "PASS ") if badge_only else ("OPEN " if allow else "MUTED")
    fmt = lambda s: f"{s:5.2f}" if s is not None else "  -- "
    wm_load = embed_seconds / config.WM_CHUNK_SECONDS
    lagging = "  LAGGING" if backlog_blocks > 20 else ""
    mic = "  INJECTED?" if injected else ("  MIC?" if mic_alarm else "")
    drops = f"  drops={padded_seconds:.1f}s" if padded_seconds > 0.05 else ""
    print(
        f"\r[{state}] score={fmt(window_score)} recent={fmt(recent_score)} "
        f"voiced={voiced:.2f} wm_load={wm_load:4.0%}{drops}{mic}{lagging}   ",
        end="",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrueMic voice-authentication virtual mic")
    parser.add_argument("--in", dest="input", default=None, help="input device name substring")
    parser.add_argument(
        "--out", dest="output", default=config.DEFAULT_OUT_DEVICE,
        help="output device name substring (VB-CABLE 'CABLE Input')",
    )
    parser.add_argument("--out-rate", type=int, default=config.DEFAULT_OUT_RATE)
    parser.add_argument("--dir", dest="profile_dir", default=config.DEFAULT_PROFILE_DIR)
    parser.add_argument("--engine", choices=["campp", "ecapa", "eagle"], default=None)
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the open threshold (close follows at -gap)")
    parser.add_argument("--no-watermark", action="store_true")
    parser.add_argument("--alpha", type=float, default=None,
                        help="watermark strength override (default config.WM_ALPHA)")
    parser.add_argument(
        "--strict-mic", action="store_true",
        help="close the gate when the capture chain doesn't match the enrolled "
        "microphone (default: indicator only)",
    )
    parser.add_argument(
        "--badge-only", action="store_true",
        help="never mute; watermark-sign only verified speech (fallback posture "
        "when calibration shows the voices can't be separated)",
    )
    parser.add_argument(
        "--low-power", action="store_true",
        help="settings for 2-core laptops (slower decide cadence)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
