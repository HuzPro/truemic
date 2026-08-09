"""Per-stage cost on THIS machine, as a fraction of one core, plus a config
recommendation. Run on every box the demo will touch — especially weak laptops.
"""

import argparse
import time

import numpy as np

from truemic import config
from truemic.audio import synthetic_speech


def timed(fn, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage cost on this machine")
    parser.add_argument("--low-power", action="store_true")
    if parser.parse_args().low_power:
        config.apply_profile("low-power")

    hop = synthetic_speech(config.DECIDE_EVERY_SECONDS)
    window = synthetic_speech(config.WINDOW_SECONDS)
    wm_chunk = synthetic_speech(config.WM_CHUNK_SECONDS)
    detect_window = synthetic_speech(config.DETECT_WINDOW_SECONDS)

    print("Loading models...")
    from truemic.verify import CamppVerifier, VoiceActivity, load_cohort
    from truemic.watermark import Watermarker, WatermarkReader

    vad = VoiceActivity()
    verifier = CamppVerifier(
        np.random.default_rng(0).standard_normal(512).astype(np.float32), load_cohort()
    )
    watermarker = Watermarker()
    reader = WatermarkReader()
    config.apply_torch_threads(config.SENDER_TORCH_THREADS)

    recent = synthetic_speech(config.RECENT_WINDOW_SECONDS)
    stages = [
        ("VAD (streaming hop)", timed(lambda: vad.feed(hop)), config.DECIDE_EVERY_SECONDS),
        ("Verify (window)", timed(lambda: verifier.score(window)), config.DECIDE_EVERY_SECONDS),
        ("Verify (recent)", timed(lambda: verifier.score(recent)), config.DECIDE_EVERY_SECONDS),
        ("WM embed (chunk)", timed(lambda: watermarker.embed(wm_chunk, 7)), config.WM_CHUNK_SECONDS),
        ("WM detect (window)", timed(lambda: reader.read(detect_window)), config.DETECT_HOP_SECONDS),
    ]

    print(f"\n{'stage':<20} {'ms/call':>9} {'every':>7} {'core load':>10}")
    for name, seconds, period in stages:
        print(f"{name:<20} {seconds * 1000:>7.1f}ms {period:>6.2f}s {seconds / period:>9.0%}")

    sender_load = sum(s / p for _, s, p in stages[:4])
    receiver_load = stages[4][1] / stages[4][2]
    print(f"\nSender ~{sender_load:.0%} of one core, receiver ~{receiver_load:.0%}.")
    if sender_load > 0.6:
        print(
            "Recommendation: run the sender with --low-power "
            "(longer decide interval and watermark chunks)."
        )
    else:
        print("Recommendation: defaults are fine on this machine.")


if __name__ == "__main__":
    main()
