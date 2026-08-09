"""TrueMic receiver: listen → AudioSeal detect → token check → signed attestation → badge.

Point --in at "CABLE Output" for the same-machine demo, or at the laptop mic
for the air-gap beat (phone speaker playing verified audio held up to it).
"""

import argparse
import base64
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd
from cryptography.hazmat.primitives import serialization

from truemic import config
from truemic import profile as profile_store
from truemic.devices import find_device
from truemic.tokens import verify_token
from truemic.watermark import PHASE_PERIOD, WatermarkReader, majority_token

BADGE_PAGE = Path(__file__).resolve().parent / "badge.html"


class ReceiverState:
    def __init__(self, speaker_name: str):
        self._lock = threading.Lock()
        self._speaker_name = speaker_name
        self._wm_probability = 0.0
        self._verified_at: float | None = None
        self._attestation: dict | None = None
        self._reason = "listening"
        self._was_verified = False
        self._changed_at: float | None = None

    def update(
        self,
        wm_probability: float,
        verified_at: float | None,
        attestation: dict | None,
        reason: str,
    ):
        with self._lock:
            self._wm_probability = wm_probability
            self._reason = reason
            if verified_at is not None:
                self._verified_at = verified_at
                self._attestation = attestation

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            age = None if self._verified_at is None else now - self._verified_at
            verified = age is not None and age < config.BADGE_STALE_SECONDS
            if verified != self._was_verified:
                self._was_verified = verified
                self._changed_at = now
            reason = self._reason
            if verified:
                reason = "live token verified"
            elif reason == "live token verified":
                reason = "last verification is stale — awaiting fresh audio"
            return {
                "speaker": self._speaker_name,
                "verified": verified,
                "wm_present": self._wm_probability >= config.WM_DETECT_THRESHOLD,
                "ever_verified": self._verified_at is not None,
                "reason": reason,
                "seconds_since_change": (
                    None if self._changed_at is None else round(now - self._changed_at, 1)
                ),
                "wm_probability": round(self._wm_probability, 3),
                "seconds_since_verified": None if age is None else round(age, 1),
                "attestation": self._attestation,
            }


class SyncedRing:
    """Callback-fed ring that can slice a window ending at an exact absolute
    sample index — capture never blocks on inference, and detection windows
    stay on the embedder's phase grid (hop is a multiple of PHASE_PERIOD)."""

    def __init__(self, capacity: int):
        self._buffer = np.zeros(0, dtype=np.float32)
        self._capacity = capacity
        self._lock = threading.Lock()
        self.total = 0

    def append(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buffer = np.concatenate([self._buffer, samples])[-self._capacity :]
            self.total += len(samples)

    def window_ending_at(self, target_total: int, length: int) -> np.ndarray | None:
        with self._lock:
            behind = self.total - target_total
            if behind < 0 or self.total - length - behind < 0:
                return None
            end = len(self._buffer) - behind
            if end - length < 0:
                return None
            return self._buffer[end - length : end].copy()


def detect_loop(state: ReceiverState, prof: profile_store.Profile, in_device: int | None):
    reader = WatermarkReader()
    config.apply_torch_threads(config.RECEIVER_TORCH_THREADS)
    import gc

    gc.collect()
    gc.freeze()
    try:
        import psutil

        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    signing_key = prof.signing_key
    public_key = base64.b64encode(
        signing_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()
    window_len = int(config.DETECT_WINDOW_SECONDS * config.SAMPLE_RATE) + PHASE_PERIOD
    hop = int(config.DETECT_HOP_SECONDS * config.SAMPLE_RATE)
    ring = SyncedRing(capacity=window_len + 4 * hop)
    votes: deque[np.ndarray] = deque(maxlen=config.TOKEN_VOTE_TICKS)

    def capture(indata, frames, time_info, status):
        ring.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=config.BLOCK_SAMPLES,
        device=in_device,
        callback=capture,
    ):
        target = window_len
        while True:
            while ring.total < target:
                time.sleep(0.05)
            # detection slower than the hop skips whole hops — phase preserved
            while ring.total - target > 2 * hop:
                target += hop
            window = ring.window_ending_at(target, window_len)
            target += hop
            if window is None:
                continue
            probability, soft_bits = reader.read_locked(window)
            now = time.time()
            slot = None
            if probability >= config.WM_DETECT_THRESHOLD:
                votes.append(soft_bits)
                # the current tick decodes near-perfectly on a healthy chain
                # (measured); the pooled vote is only a fallback — pooling
                # FIRST mixes bits across token rotations and pauses and then
                # matches neither token
                slot = verify_token(
                    prof.token_key,
                    majority_token([soft_bits]),
                    now,
                    config.TOKEN_SLOT_SECONDS,
                    config.TOKEN_TOLERANCE,
                    config.TOKEN_MAX_BIT_ERRORS,
                )
                if slot is None and len(votes) > 1:
                    slot = verify_token(
                        prof.token_key,
                        majority_token(list(votes)),
                        now,
                        config.TOKEN_SLOT_SECONDS,
                        config.TOKEN_TOLERANCE,
                    )
            else:
                votes.clear()
            if slot is not None:
                reason = "live token verified"
            elif probability >= config.WM_DETECT_THRESHOLD:
                reason = "watermark present but token stale or wrong — possible replay"
            else:
                reason = "no watermark in audio — speech was never verified at the mic"
            attestation = None
            if slot is not None:
                payload = {
                    "speaker": prof.name,
                    "engine": prof.engine,
                    "wm_confidence": round(probability, 3),
                    "token_slot": slot,
                    "verified_at": round(now, 2),
                }
                body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
                attestation = {
                    **payload,
                    "signature": base64.b64encode(signing_key.sign(body)).decode(),
                    "public_key": public_key,
                }
            state.update(probability, now if slot is not None else None, attestation, reason)


class BadgeHandler(BaseHTTPRequestHandler):
    state: ReceiverState  # set before serving

    def do_GET(self):
        if self.path == "/":
            body = BADGE_PAGE.read_bytes()
            self._respond(200, "text/html; charset=utf-8", body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    payload = json.dumps(self.state.snapshot())
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.3)
            except (ConnectionError, BrokenPipeError, OSError):
                pass
        elif self.path == "/attest":
            attestation = self.state.snapshot()["attestation"]
            body = json.dumps(attestation or {"error": "no attestation yet"}, indent=2).encode()
            self._respond(200, "application/json", body)
        else:
            self._respond(404, "text/plain", b"not found")

    def _respond(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main() -> None:
    args = parse_args()
    config.apply_profile("low-power" if args.low_power else "default")
    prof = profile_store.load(Path(args.profile_dir))
    in_device = find_device(args.input, "input")
    state = ReceiverState(prof.name)

    threading.Thread(
        target=detect_loop, args=(state, prof, in_device), daemon=True
    ).start()

    BadgeHandler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BadgeHandler)
    print(f"Badge page: http://127.0.0.1:{args.port}  (listening on {args.input or 'default mic'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TrueMic receiver + verified-human badge")
    parser.add_argument(
        "--in", dest="input", default=None,
        help="input device name substring ('CABLE Output' for same-machine demo)",
    )
    parser.add_argument("--port", type=int, default=config.BADGE_PORT)
    parser.add_argument("--dir", dest="profile_dir", default=config.DEFAULT_PROFILE_DIR)
    parser.add_argument(
        "--low-power", action="store_true",
        help="settings for 2-core laptops (longer detect window/hop, capped threads)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
