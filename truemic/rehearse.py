"""Offline rehearsal of the demo's kill-shot beat with real voices.

Two LibriSpeech test-clean speakers play the enrolled user (A) and the
teammate (B). The timeline A -> B -> A runs through the real pipeline objects
(streaming VAD, CAM++ scoring, gate decisions, delay-line gate, watermark,
receiver detection) sample-for-sample, then reports what the audience would
see: impostor leakage (must be 0), genuine speech lost, badge timeline.

    python -m truemic.rehearse [--tar models/test-clean.tar.gz]
"""

import argparse
import io
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from truemic import config
from truemic.gate import DelayGate, GateDecision
from truemic.tokens import current_slot, token_for_slot, verify_token
from truemic.watermark import PHASE_PERIOD, Watermarker, WatermarkReader, majority_token

KEY = b"\x21" * 32
ENROLL_SECONDS = 40.0
SEGMENTS = (("A", 20.0), ("B", 15.0), ("A", 10.0))
AUDIBLE_RMS = 0.01  # above the comfort-noise floor
PIPELINE_DELAY = config.WM_CHUNK_SECONDS + config.GATE_DELAY_SECONDS


def load_two_speakers(tar_path: Path) -> dict[str, np.ndarray]:
    import soundfile

    needed = ENROLL_SECONDS + sum(sec for who, sec in SEGMENTS if who == "A") + 5
    per_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                continue
            speaker = member.name.split("/")[2]
            if len(per_speaker) >= 2 and speaker not in per_speaker:
                have = {
                    s: sum(len(w) for w in ws) / config.SAMPLE_RATE
                    for s, ws in per_speaker.items()
                }
                if all(v > needed for v in have.values()):
                    break
                if speaker not in per_speaker:
                    continue
            wav, rate = soundfile.read(io.BytesIO(tar.extractfile(member).read()))
            assert rate == config.SAMPLE_RATE
            per_speaker[speaker].append(wav.astype(np.float32))
    speakers = sorted(per_speaker)[:2]
    return {
        label: np.concatenate(per_speaker[s])
        for label, s in zip(("A", "B"), speakers)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline demo rehearsal with real voices")
    parser.add_argument("--tar", default=config.MODELS_DIR / "test-clean.tar.gz")
    args = parser.parse_args()

    print("Loading two LibriSpeech speakers...")
    voices = load_two_speakers(Path(args.tar))
    enroll_wav = voices["A"][: int(ENROLL_SECONDS * config.SAMPLE_RATE)]
    cursor = {"A": len(enroll_wav), "B": 0}

    print("Enrolling speaker A through the real enrollment path...")
    from truemic.enroll import enroll_embedding_engine
    from truemic.profile import EMBEDDING_FILE
    from truemic.verify import CamppVerifier, VoiceActivity, load_cohort

    vad = VoiceActivity()
    with tempfile.TemporaryDirectory() as tmp:
        enroll_embedding_engine("campp", enroll_wav, Path(tmp), vad)
        embedding = np.load(Path(tmp) / EMBEDDING_FILE)
    verifier = CamppVerifier(embedding, load_cohort())

    timeline, labels = [], []
    for who, seconds in SEGMENTS:
        n = int(seconds * config.SAMPLE_RATE)
        timeline.append(voices[who][cursor[who] : cursor[who] + n])
        labels.extend([who] * n)
        cursor[who] += n
    timeline = np.concatenate(timeline)
    labels = np.array(labels)

    print("Running the sender pipeline (VAD, scoring, gate, watermark)...")
    vad_rt = VoiceActivity()
    watermarker = Watermarker()
    gate = DelayGate(
        delay_samples=int(config.GATE_DELAY_SECONDS * config.SAMPLE_RATE),
        fade_samples=int(config.CROSSFADE_MS / 1000 * config.SAMPLE_RATE),
    )
    decision = GateDecision(
        open_threshold=config.DEFAULT_THRESHOLD_ASNORM,
        close_threshold=config.DEFAULT_THRESHOLD_ASNORM - 1.0,
        silence_timeout=config.SILENCE_TIMEOUT_SECONDS,
        open_confirmations=config.OPEN_CONFIRMATIONS,
    )
    calibrate_thresholds(decision, verifier, vad, voices["A"], enroll_wav)

    ring = np.zeros(0, dtype=np.float32)
    wm_pending = np.zeros(0, dtype=np.float32)
    emitted = []
    chunk_samples = int(config.WM_CHUNK_SECONDS * config.SAMPLE_RATE)
    window_samples = int(config.WINDOW_SECONDS * config.SAMPLE_RATE)
    recent_samples = int(config.RECENT_WINDOW_SECONDS * config.SAMPLE_RATE)
    decide_every = int(config.DECIDE_EVERY_SECONDS * config.SAMPLE_RATE)
    next_decide = window_samples
    token = token_for_slot(KEY, current_slot(__import__("time").time(), config.TOKEN_SLOT_SECONDS))
    gate_log = []

    for start in range(0, len(timeline) - config.BLOCK_SAMPLES + 1, config.BLOCK_SAMPLES):
        block = timeline[start : start + config.BLOCK_SAMPLES]
        ring = np.concatenate([ring, block])[-window_samples:]
        vad_rt.feed(block)
        wm_pending = np.concatenate([wm_pending, block])
        if len(wm_pending) >= chunk_samples:
            chunk, wm_pending = wm_pending[:chunk_samples], wm_pending[chunk_samples:]
            if vad_rt.recent_voiced_fraction(config.WM_CHUNK_SECONDS) > 0:
                chunk = watermarker.embed(chunk, token)
            emitted.append(gate.process(chunk))
        if start + config.BLOCK_SAMPLES >= next_decide:
            next_decide += decide_every
            voiced = vad_rt.recent_voiced_fraction(config.WINDOW_SECONDS)
            window_score = (
                verifier.score(ring) if voiced >= config.MIN_VOICED_FRACTION else None
            )
            recent_score = None
            if window_score is not None:
                recent_score = verifier.score(ring[-recent_samples:])
            now = start / config.SAMPLE_RATE
            allow = decision.update(window_score, recent_score, now)
            gate.set_allow(allow)
            gate_log.append((now, labels[start], window_score, allow))

    output = np.concatenate(emitted)
    report_gate(gate_log)
    report_audio(output, timeline, labels)
    report_receiver(output, token)


def calibrate_thresholds(decision, verifier, vad, voice_a, enroll_wav) -> None:
    """30 s of held-out genuine speech, calibrate.py's no-impostor rule."""
    held_out = voice_a[len(enroll_wav) : len(enroll_wav) + 30 * config.SAMPLE_RATE]
    from truemic.calibrate import _calibrate_recent, scores_for_recording

    scores = np.asarray(scores_for_recording(verifier, vad, held_out))
    open_threshold = float(scores.mean() - 1.65 * scores.std())
    decision.open_threshold = open_threshold
    decision.close_threshold = open_threshold - float(
        np.clip(0.75 * scores.std(), 0.03, 0.12)
    )
    decision.recent_close_threshold = _calibrate_recent(verifier, vad, held_out)
    print(f"Calibrated: open={decision.open_threshold:.2f} "
          f"close={decision.close_threshold:.2f} "
          f"recent_close={decision.recent_close_threshold} "
          f"(genuine mean={scores.mean():.2f} sd={scores.std():.2f})")


def report_gate(gate_log) -> None:
    for who in ("A", "B"):
        scores = [s for _, label, s, _ in gate_log if label == who and s is not None]
        if scores:
            print(f"speaker {who}: decision scores mean={np.mean(scores):+.2f} "
                  f"min={np.min(scores):+.2f} max={np.max(scores):+.2f} n={len(scores)}")
    flips = [(t, label, allow) for (t, label, _, allow) in gate_log]
    changes = [
        f"{t:5.1f}s {'OPEN' if allow else 'MUTE'} (speaking: {label})"
        for i, (t, label, allow) in enumerate(flips)
        if i == 0 or allow != flips[i - 1][2]
    ]
    print("gate transitions: " + "; ".join(changes))


def report_audio(output, timeline, labels) -> None:
    delay = int(PIPELINE_DELAY * config.SAMPLE_RATE)
    frame = 480
    leaked = lost = a_frames = b_frames = 0
    for start in range(0, len(output) - frame, frame):
        src = start - delay
        if src < 0:
            continue
        out_audible = (
            float(np.sqrt(np.mean(np.square(output[start : start + frame])))) > AUDIBLE_RMS
        )
        in_audible = (
            float(np.sqrt(np.mean(np.square(timeline[src : src + frame])))) > AUDIBLE_RMS
        )
        if not in_audible:
            continue  # silence lost is not speech lost
        if labels[src] == "B":
            b_frames += 1
            leaked += out_audible
        else:
            a_frames += 1
            lost += not out_audible
    print(f"IMPOSTOR SPEECH LEAKED: {leaked * frame / config.SAMPLE_RATE:.2f}s "
          f"of {b_frames * frame / config.SAMPLE_RATE:.1f}s spoken "
          f"| genuine speech lost: {lost * frame / config.SAMPLE_RATE:.1f}s "
          f"of {a_frames * frame / config.SAMPLE_RATE:.1f}s spoken "
          "(includes the one-time open latency)")


def report_receiver(output, token) -> None:
    reader = WatermarkReader()
    window = int(config.DETECT_WINDOW_SECONDS * config.SAMPLE_RATE) + PHASE_PERIOD
    hop = int(config.DETECT_HOP_SECONDS * config.SAMPLE_RATE)
    delay = PIPELINE_DELAY
    states = []
    for start in range(0, len(output) - window + 1, hop):
        score, soft = reader.read_locked(output[start : start + window])
        ok = score >= config.WM_DETECT_THRESHOLD and majority_token([soft]) == token
        states.append((start / config.SAMPLE_RATE - delay, ok))
    spans = []
    for t, ok in states:
        if not spans or spans[-1][2] != ok:
            spans.append([t, t, ok])
        else:
            spans[-1][1] = t
    print("badge timeline (input time): " + "; ".join(
        f"{'GREEN' if ok else 'grey':>5} {a:5.1f}-{b:5.1f}s" for a, b, ok in spans))


if __name__ == "__main__":
    main()
