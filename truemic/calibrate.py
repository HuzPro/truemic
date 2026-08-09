"""Calibrate open/close thresholds on the actual room, mic, and speaker.

Percentiles over overlapping windows are unstable (~12 effective samples in
30 s), so thresholds come from means and standard deviations: the genuine
floor at mean-1.65sd (~5% FRR) and, when an impostor recording exists, an
impostor ceiling at mean+2.5sd with the thresholds placed inside the gap.
"""

import argparse
from pathlib import Path

import numpy as np

from truemic import config
from truemic import profile as profile_store
from truemic.audio import record
from truemic.devices import find_device
from truemic.verify import VoiceActivity, create_verifier

GENUINE_FLOOR_SIGMA = 1.65  # one-sided ~5% FRR
IMPOSTOR_CEILING_SIGMA = 2.5
OPEN_GAP_POSITION = 0.6  # open threshold sits 60% of the way up the gap
CLOSE_GAP_POSITION = 0.3


def main() -> None:
    args = parse_args()
    prof = profile_store.load(Path(args.profile_dir))
    verifier = create_verifier(prof)
    vad = VoiceActivity()
    device = find_device(args.input, "input")

    recording = record(
        args.seconds, device, f"{prof.name}, speak naturally for {args.seconds:.0f}s..."
    )
    half = len(recording) // 2
    if _refine_print(prof, verifier, vad, recording[:half]):
        verifier = create_verifier(prof)  # reload with the enriched print
    genuine = scores_for_recording(verifier, vad, recording[half:])
    if len(genuine) < 5:
        raise SystemExit("Too few scoreable voiced windows — check the mic and try again.")
    genuine = np.asarray(genuine)
    floor = float(genuine.mean() - GENUINE_FLOOR_SIGMA * genuine.std())
    print(
        f"Genuine: n={len(genuine)}  mean={genuine.mean():.3f}  "
        f"sd={genuine.std():.3f}  floor={floor:.3f}"
    )

    impostor = None
    if args.impostor:
        impostor_recording = record(
            args.seconds, device, f"Impostor (teammate), speak for {args.seconds:.0f}s..."
        )
        genuine_rms = float(np.sqrt(np.mean(recording**2)))
        impostor_rms = float(np.sqrt(np.mean(impostor_recording**2)))
        if impostor_rms < config.CALIBRATE_IMPOSTOR_MIN_RMS_RATIO * genuine_rms:
            # a barely-audible impostor scores HIGH (hub effect: quiet audio
            # looks like room tone, room tone looks like everything)
            print(f"INVALID impostor take: their level ({impostor_rms:.3f}) is far below "
                  f"yours ({genuine_rms:.3f}). They must speak as close and loud as you "
                  "did. Ignoring it - re-run calibrate --impostor properly.")
        else:
            impostor = scores_for_recording(verifier, vad, impostor_recording)
            impostor = np.asarray(impostor) if impostor else None

    if impostor is not None:
        ceiling = float(impostor.mean() + IMPOSTOR_CEILING_SIGMA * impostor.std())
        gap = floor - ceiling
        print(
            f"Impostor: n={len(impostor)}  mean={impostor.mean():.3f}  "
            f"sd={impostor.std():.3f}  ceiling={ceiling:.3f}  gap={gap:.3f}"
        )
        if args.posture == "balanced":
            # equal-error crossing point: accepts some per-window error both
            # ways and lets the gate's confirmation/hysteresis machinery
            # suppress the transients — for tight-margin mics
            g_mean, g_sd = float(genuine.mean()), float(genuine.std()) + 1e-6
            i_mean, i_sd = float(impostor.mean()), float(impostor.std()) + 1e-6
            open_threshold = (g_mean * i_sd + i_mean * g_sd) / (g_sd + i_sd)
            close_threshold = min(open_threshold - 0.02, i_mean + 0.5 * i_sd)
        else:
            if gap <= 0:
                print("WARNING: distributions overlap - fail-closed thresholds will "
                      "reject you often. Try --posture balanced, or --badge-only.")
            open_threshold = ceiling + OPEN_GAP_POSITION * max(gap, 0.0)
            close_threshold = ceiling + CLOSE_GAP_POSITION * max(gap, 0.0)
    else:
        open_threshold = floor
        close_threshold = open_threshold - _close_gap(genuine, open_threshold)

    open_threshold = round(open_threshold, 3)
    close_threshold = round(min(close_threshold, open_threshold - 1e-3), 3)
    print(f"Estimated FRR at open: {float(np.mean(genuine < open_threshold)):.1%}")
    if impostor is not None:
        print(f"Estimated FAR at open: {float(np.mean(impostor >= open_threshold)):.1%}   "
              f"stay-open FAR at close: {float(np.mean(impostor >= close_threshold)):.1%}")

    recent_close_threshold = _calibrate_recent(verifier, vad, recording[half:])
    prof.save_thresholds(open_threshold, close_threshold, recent_close_threshold)
    print(f"open={open_threshold}  close={close_threshold}  "
          f"recent_close={recent_close_threshold}  "
          f"written to {prof.directory / 'profile.json'}.")


def _refine_print(prof, verifier, vad: VoiceActivity, half: np.ndarray) -> bool:
    """Fold this session's voice into the enrollment set (Teams-style passive
    enrollment) — scored later against the OTHER half, so no self-inflation."""
    dense = vad.voiced_only(half)
    chunk = int(3.0 * config.SAMPLE_RATE)
    chunks = [
        dense[start : start + chunk]
        for start in range(0, len(dense) - chunk + 1, chunk // 2)
    ]
    if len(chunks) < 2:
        return False
    try:
        new_rows = np.stack([verifier.embed(c) for c in chunks])
    except NotImplementedError:
        return False  # eagle profiles are opaque
    from truemic.profile import EMBEDDING_FILE
    from truemic.verify import enrollment_matrix

    updated = np.vstack([enrollment_matrix(prof.embedding), enrollment_matrix(new_rows)])
    np.save(prof.directory / EMBEDDING_FILE, updated)
    print(f"Voiceprint refined: +{len(new_rows)} chunks from this session "
          f"({len(updated)} total).")
    return True


def _calibrate_recent(verifier, vad: VoiceActivity, recording: np.ndarray) -> float | None:
    """The 1 s close-path window scores on its own, much lower scale than the
    3 s window — it needs its own fail-closed floor or it mutes the real
    speaker constantly (measured live: window 0.95 vs recent 0.6)."""
    window = int(config.RECENT_WINDOW_SECONDS * config.SAMPLE_RATE)
    hop = int(config.DECIDE_EVERY_SECONDS * config.SAMPLE_RATE)
    scores = []
    for start in range(0, len(recording) - window + 1, hop):
        w = recording[start : start + window]
        if vad.voiced_fraction(w) < config.MIN_VOICED_FRACTION:
            continue
        score = verifier.score(w)
        if score is not None:
            scores.append(score)
    if len(scores) < 5:
        print("Too few voiced 1s windows to calibrate the recent check - disabled.")
        return None
    scores = np.asarray(scores)
    recent_close = round(float(scores.mean() - IMPOSTOR_CEILING_SIGMA * scores.std()), 3)
    print(f"Recent (1s): n={len(scores)}  mean={scores.mean():.3f}  "
          f"sd={scores.std():.3f}  recent_close={recent_close}")
    return recent_close


def _close_gap(genuine: np.ndarray, open_threshold: float) -> float:
    z_scale = abs(open_threshold) > 1.5  # AS-Norm scores live on a wider scale
    lo, hi = (0.3, 1.2) if z_scale else (0.03, 0.12)
    return float(np.clip(0.75 * genuine.std(), lo, hi))


def scores_for_recording(verifier, vad: VoiceActivity, recording: np.ndarray) -> list[float]:
    window = int(config.WINDOW_SECONDS * config.SAMPLE_RATE)
    hop = int(config.DECIDE_EVERY_SECONDS * config.SAMPLE_RATE)
    scores = []
    for start in range(0, len(recording) - hop + 1, hop):
        verifier.feed(recording[start : start + hop])
        end = start + hop
        if end < window:
            continue
        w = recording[end - window : end]
        if vad.voiced_fraction(w) < config.MIN_VOICED_FRACTION:
            continue
        score = verifier.score(w)
        if score is not None:
            scores.append(score)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the TrueMic thresholds on-site")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--in", dest="input", default=None, help="input device name substring")
    parser.add_argument("--dir", dest="profile_dir", default=config.DEFAULT_PROFILE_DIR)
    parser.add_argument(
        "--impostor", action="store_true",
        help="also record the demo teammate; thresholds go inside the score gap",
    )
    parser.add_argument(
        "--posture", choices=["failclosed", "balanced"], default="failclosed",
        help="failclosed: impostor never passes, you may be muted often on tight "
        "mics; balanced: equal-error thresholds, gate dynamics suppress flickers",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
