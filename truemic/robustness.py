"""Measure watermark + token survival through call-stack channel models.

The research predicts Opus-class codecs eat AudioSeal at alpha=1. This makes
it a number instead of a fear: for each channel x alpha, embed a token,
run the channel, then detect exactly like the receiver does (RMS-normalized
1 s windows, soft-vote across ticks).

    python -m truemic.robustness [--seconds 8] [--alphas 1,3,5] [--wav file.wav]
"""

import argparse
import io

import numpy as np
import soxr

from truemic import config
from truemic.audio import add_noise_at_snr, add_synthetic_reverb, synthetic_speech
from truemic.tokens import token_for_slot
from truemic.watermark import Watermarker, WatermarkReader, majority_token

KEY = b"\x42" * 32
TOKEN = token_for_slot(KEY, 12345)


def codec_roundtrip(wav: np.ndarray, codec: str, bitrate: int, container: str) -> np.ndarray:
    import av

    rate = 48_000
    upsampled = soxr.resample(wav, config.SAMPLE_RATE, rate).astype(np.float32)

    encoded = io.BytesIO()
    out = av.open(encoded, "w", format=container)
    stream = out.add_stream(codec, rate=rate)
    stream.bit_rate = bitrate
    frame = av.AudioFrame.from_ndarray(upsampled[None], format="fltp", layout="mono")
    frame.sample_rate = rate
    for packet in stream.encode(frame):
        out.mux(packet)
    for packet in stream.encode(None):
        out.mux(packet)
    out.close()

    encoded.seek(0)
    decoded_chunks = []
    inp = av.open(encoded)
    for frame in inp.decode(audio=0):
        decoded_chunks.append(frame.to_ndarray().reshape(-1))
    inp.close()
    decoded = np.concatenate(decoded_chunks).astype(np.float32)
    return soxr.resample(decoded, rate, config.SAMPLE_RATE).astype(np.float32)


def channels() -> dict:
    rng = np.random.default_rng(7)
    return {
        "clean": lambda w: w,
        "resample 16-48-16": lambda w: soxr.resample(
            soxr.resample(w, 16_000, 48_000), 48_000, 16_000
        ).astype(np.float32),
        "opus 24k": lambda w: codec_roundtrip(w, "libopus", 24_000, "ogg"),
        "opus 32k": lambda w: codec_roundtrip(w, "libopus", 32_000, "ogg"),
        "opus 64k": lambda w: codec_roundtrip(w, "libopus", 64_000, "ogg"),
        "aac 64k": lambda w: codec_roundtrip(w, "aac", 64_000, "adts"),
        "air gap (reverb+20dB SNR)": lambda w: add_noise_at_snr(
            add_synthetic_reverb(w, rng), 20.0, rng
        ),
        "air gap noisy (10dB SNR)": lambda w: add_noise_at_snr(
            add_synthetic_reverb(w, rng), 10.0, rng
        ),
    }


def measure(reader: WatermarkReader, received: np.ndarray) -> tuple[float, float, bool]:
    from truemic.watermark import PHASE_PERIOD

    reader.unlock()
    window = int(config.DETECT_WINDOW_SECONDS * config.SAMPLE_RATE) + PHASE_PERIOD
    hop = int(config.DETECT_HOP_SECONDS * config.SAMPLE_RATE)
    scores, exact, softs = [], [], []
    for start in range(0, len(received) - window + 1, hop):
        score, soft = reader.read_locked(received[start : start + window])
        scores.append(score)
        softs.append(soft)
        exact.append(majority_token([soft]) == TOKEN)
    voted = majority_token(softs[-config.TOKEN_VOTE_TICKS :]) == TOKEN
    return float(np.mean(scores)), float(np.mean(exact)), voted


def main() -> None:
    parser = argparse.ArgumentParser(description="Watermark channel-survival table")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--alphas", default="1,3,5")
    parser.add_argument("--wav", default=None, help="16 kHz mono wav to use instead of synthetic")
    args = parser.parse_args()

    if args.wav:
        from scipy.io import wavfile

        rate, wav = wavfile.read(args.wav)
        assert rate == config.SAMPLE_RATE, f"need 16 kHz, got {rate}"
        audio = (wav / 32768.0).astype(np.float32)[: int(args.seconds * rate)]
    else:
        audio = synthetic_speech(args.seconds)

    reader = WatermarkReader()
    watermarker = Watermarker()
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"{'channel':<28} " + " ".join(f"{'a=' + str(a):>16}" for a in alphas))
    print(f"{'':<28} " + " ".join(f"{'score/tick%/vote':>16}" for _ in alphas))
    for name, channel in channels().items():
        cells = []
        for alpha in alphas:
            watermarker.reset(alpha=alpha)
            marked = watermarker.embed(audio, TOKEN)
            score, tick_rate, voted = measure(reader, channel(marked))
            cells.append(f"{score:4.2f}/{tick_rate:4.0%}/{'Y' if voted else 'n'}")
        print(f"{name:<28} " + " ".join(f"{c:>16}" for c in cells))
    print("\nscore = mean detection score (threshold "
          f"{config.WM_DETECT_THRESHOLD}), tick% = single-tick exact-token rate, "
          "vote = soft majority over last 3 ticks correct")


if __name__ == "__main__":
    main()
