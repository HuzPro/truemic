"""Enroll a speaker: record ~40 s, build the voiceprint, generate keys.

The embedding is multi-conditioned (clean + noise + synthetic reverb per
chunk, outliers rejected) so a mic or room change between enrollment and use
degrades gracefully — the failure mode CrystalSound documents.

Run `python -m truemic.calibrate` afterwards — the default thresholds are
placeholders, not a calibration.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from truemic import config
from truemic import profile as profile_store
from truemic.audio import add_noise_at_snr, add_synthetic_reverb, record
from truemic.devices import find_device
from truemic.profile import EAGLE_PROFILE_FILE, EMBEDDING_FILE
from truemic.verify import (
    VoiceActivity,
    _campp_session,
    campp_embed,
    ecapa_embed,
    load_ecapa,
    require_access_key,
)

ENROLL_CHUNK_SECONDS = 3.0
MIN_USABLE_CHUNKS = 3
OUTLIER_SIGMA = 2.0
ENROLL_PROMPT = (
    "Read anything aloud — speak as you normally would in terms of tone, "
    "speed, and loudness"
)


def main() -> None:
    args = parse_args()
    engine = resolve_engine(args.engine)
    directory = Path(args.profile_dir)

    if args.reuse:
        rate, wav = wavfile.read(directory / "enroll.wav")
        assert rate == config.SAMPLE_RATE
        recording = (wav / 32768.0).astype(np.float32)
        print(f"Rebuilding the voiceprint from the existing {len(recording) / rate:.0f}s "
              "enrollment recording...")
    else:
        device = find_device(args.input, "input")
        recording = record(
            args.seconds,
            device,
            f"Enrolling {args.name!r} with {engine}. {ENROLL_PROMPT}, "
            f"for {args.seconds:.0f}s...",
        )

    default_threshold = {
        "eagle": config.DEFAULT_THRESHOLD_EAGLE,
        "ecapa": config.DEFAULT_THRESHOLD_ECAPA,
        "campp": config.DEFAULT_THRESHOLD_CAMPP,
    }[engine]
    prof = profile_store.create(directory, args.name, engine, default_threshold)
    wavfile.write(
        directory / "enroll.wav",
        config.SAMPLE_RATE,
        (np.clip(recording, -1, 1) * 32767).astype(np.int16),
    )

    vad = VoiceActivity()
    if engine == "eagle":
        enroll_eagle(recording, directory)
    else:
        enroll_embedding_engine(engine, recording, directory, vad)

    from truemic.micprint import fingerprint_from_recording

    mic_print = fingerprint_from_recording(recording, vad)
    if mic_print is not None:
        np.save(directory / profile_store.MIC_PRINT_FILE, mic_print)
        print("Microphone fingerprint captured from the quiet frames.")
    else:
        print("Warning: too few quiet frames to fingerprint the mic (pause between "
              "sentences next time); mic guard disabled.")

    print(f"Enrolled {args.name!r} ({engine}) into {directory}.")
    print("Now calibrate on this room and mic:  python -m truemic.calibrate --impostor")


def enroll_embedding_engine(
    engine: str, recording: np.ndarray, directory: Path, vad: VoiceActivity
) -> None:
    if engine == "campp":
        session = _campp_session()
        embed = lambda wav: campp_embed(session, wav)
    else:
        model = load_ecapa()
        embed = lambda wav: ecapa_embed(model, wav)

    dense = vad.voiced_only(recording)
    print(f"{len(dense) / config.SAMPLE_RATE:.0f}s of voiced audio "
          f"in the {len(recording) / config.SAMPLE_RATE:.0f}s recording.")
    chunk = int(ENROLL_CHUNK_SECONDS * config.SAMPLE_RATE)
    hop = chunk // 2  # overlapping chunks: more delivery coverage from the same audio
    voiced_chunks = [
        dense[start : start + chunk] for start in range(0, len(dense) - chunk + 1, hop)
    ]
    if not voiced_chunks:
        raise SystemExit("No voiced audio in the recording — check the mic and try again.")
    if len(voiced_chunks) < MIN_USABLE_CHUNKS:
        print(f"Warning: only {len(voiced_chunks)} voiced chunks; enrollment may be weak.")
    held_out = voiced_chunks.pop() if len(voiced_chunks) > 2 else voiced_chunks[-1]

    rng = np.random.default_rng(0)
    rows = []
    for c in voiced_chunks:
        # average the conditions WITHIN a chunk: robustness without leaving
        # pure-noise/reverb rows for an impostor to cherry-pick under top-K
        variants = np.stack(
            [embed(c), embed(add_noise_at_snr(c, 10.0, rng)), embed(add_synthetic_reverb(c, rng))]
        ).mean(axis=0)
        rows.append(variants / np.linalg.norm(variants))
    embeddings = _reject_outliers(np.stack(rows))
    np.save(directory / EMBEDDING_FILE, embeddings)  # a matrix: scored top-K at runtime
    print(f"Voiceprint set of {len(embeddings)} chunk embeddings "
          f"(3 conditions averaged per chunk, outliers dropped).")
    _noise_self_test(directory, engine, held_out)


def _noise_self_test(directory: Path, engine: str, held_out: np.ndarray) -> None:
    """RELATIVE test: the speaker's own held-out audio must beat white noise
    by a clear margin. (Absolute noise scores are useless — this embedding
    model puts ANY audio at ~0.7 cosine to any print, a hub effect.)"""
    if engine == "eagle":
        return
    from truemic.verify import CamppVerifier, EcapaVerifier

    embedding = np.load(directory / EMBEDDING_FILE)
    verifier = (
        CamppVerifier(embedding, None) if engine == "campp" else EcapaVerifier(embedding)
    )
    noise = np.random.default_rng(0).standard_normal(3 * config.SAMPLE_RATE)
    noise_score = verifier.score(noise.astype(np.float32) * 0.01)
    own_score = verifier.score(held_out)
    margin = own_score - noise_score
    if margin < config.ENROLL_SELF_TEST_MARGIN:
        print(f"WARNING: your held-out audio ({own_score:.2f}) barely beats white noise "
              f"({noise_score:.2f}) against this print - the recording carries little "
              "identity. Re-record: closer, louder, continuous speech.")
    else:
        print(f"Self-test passed: you {own_score:.2f} vs noise {noise_score:.2f} "
              f"(margin {margin:+.2f}).")


def _reject_outliers(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0)
    mean /= np.linalg.norm(mean)
    cosines = embeddings @ mean
    floor = cosines.mean() - OUTLIER_SIGMA * cosines.std()
    kept = embeddings[cosines >= floor]
    return kept if len(kept) >= MIN_USABLE_CHUNKS else embeddings


def enroll_eagle(recording: np.ndarray, directory: Path) -> None:
    import pveagle

    profiler = pveagle.create_profiler(access_key=require_access_key())
    pcm = (np.clip(recording, -1, 1) * 32767).astype(np.int16)
    frame = profiler.frame_length
    percentage = 0.0
    for start in range(0, len(pcm) - frame + 1, frame):
        percentage = profiler.enroll(pcm[start : start + frame].tolist())
    print(f"\nEagle enrollment: {percentage:.0f}%")
    if percentage < 100.0:
        profiler.delete()
        raise SystemExit("Eagle needs more speech — re-run with a larger --seconds.")
    speaker = profiler.export()
    (directory / EAGLE_PROFILE_FILE).write_bytes(speaker.to_bytes())
    profiler.delete()


def resolve_engine(engine: str) -> str:
    if engine != "auto":
        return engine
    return "campp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll the TrueMic speaker")
    parser.add_argument("--name", required=True)
    parser.add_argument("--engine", choices=["auto", "campp", "ecapa", "eagle"], default="auto")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--in", dest="input", default=None, help="input device name substring")
    parser.add_argument("--dir", dest="profile_dir", default=config.DEFAULT_PROFILE_DIR)
    parser.add_argument(
        "--reuse", action="store_true",
        help="rebuild the voiceprint from the profile's existing enroll.wav (no recording)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
