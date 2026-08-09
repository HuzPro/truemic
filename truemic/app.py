"""TrueMic as a product: one command, browser onboarding, live protection.

    python -m truemic.app

Flow: pick mic (live meter) -> read the passage 3 ways -> automatic voiceprint
+ thresholds -> optional impostor pass -> live dashboard with the gate running.
"""

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import sounddevice as sd

from truemic import config
from truemic import profile as profile_store

APP_PAGE = Path(__file__).resolve().parent / "onboard.html"
PORT = 8770

RECORDINGS: dict[str, np.ndarray] = {}
ENROLL_STATS: dict[str, float] = {}  # honest PRE-refine genuine stats for the impostor step
LEVEL = {"value": 0.0, "active": False, "elapsed": 0.0, "seconds": 0.0}
GATE = {"live": None, "in_device": None}
MODELS = {}


def models():
    if not MODELS:
        from truemic.verify import VoiceActivity

        MODELS["vad"] = VoiceActivity()
    return MODELS


def record_with_level(seconds: float, device: int | None) -> np.ndarray:
    blocks: list[np.ndarray] = []
    LEVEL.update(active=True, elapsed=0.0, seconds=seconds)

    def callback(indata, frames, time_info, status):
        block = indata[:, 0].copy()
        blocks.append(block)
        LEVEL["value"] = float(np.abs(block).max())
        LEVEL["elapsed"] = sum(len(b) for b in blocks) / config.SAMPLE_RATE

    with sd.InputStream(
        samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=config.BLOCK_SAMPLES, device=device, callback=callback,
    ):
        total = int(seconds * config.SAMPLE_RATE)
        while sum(len(b) for b in blocks) < total:
            sd.sleep(100)
    LEVEL.update(active=False, value=0.0)
    return np.concatenate(blocks)[: int(seconds * config.SAMPLE_RATE)]


def api_miccheck() -> list[dict]:
    from truemic.miccheck import VIRTUAL

    candidates, seen = [], set()
    for index, device in enumerate(sd.query_devices()):
        name = device["name"]
        if device["max_input_channels"] == 0 or any(v in name for v in VIRTUAL):
            continue
        if name in seen and not any(k in name for k in ("Headset", "Hands")):
            continue
        seen.add(name)
        candidates.append((index, name))

    buffers: dict[int, list] = {}
    streams = []
    for index, name in candidates:
        buffers[index] = []

        def callback(indata, frames, time_info, status, idx=index):
            buffers[idx].append(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                device=index, callback=callback,
            )
            stream.start()
            streams.append(stream)
        except Exception:
            buffers.pop(index)
    sd.sleep(5000)
    for stream in streams:
        stream.stop()
        stream.close()

    from scipy.signal import welch

    rows = []
    for index, name in candidates:
        if index not in buffers or not buffers[index]:
            continue
        wav = np.concatenate(buffers[index])
        rms = float(np.sqrt(np.mean(wav**2)))
        if rms < 1e-5:
            continue
        f, p = welch(wav, config.SAMPLE_RATE, nperseg=2048)
        high = float(p[f >= 2000].sum() / p.sum())
        rows.append(
            {
                "index": index,
                "name": name,
                "rms": round(rms, 4),
                "high": round(high, 3),
                "good": bool(rms > 0.015 and high > 0.04),
            }
        )
    rows.sort(key=lambda r: (r["good"], r["rms"]), reverse=True)
    return rows


def api_enroll(name: str) -> dict:
    from scipy.io import wavfile

    from truemic.calibrate import _calibrate_recent, _refine_print, scores_for_recording
    from truemic.enroll import enroll_embedding_engine
    from truemic.micprint import fingerprint_from_recording
    from truemic.verify import CamppVerifier, create_verifier

    vad = models()["vad"]
    passes = [RECORDINGS[k] for k in ("pass1", "pass2", "pass3") if k in RECORDINGS]
    if len(passes) < 3:
        return {"error": "not all passes recorded"}
    directory = config.DEFAULT_PROFILE_DIR
    prof = profile_store.create(directory, name, "campp", config.DEFAULT_THRESHOLD_CAMPP)
    full = np.concatenate(passes)
    wavfile.write(
        directory / "enroll.wav", config.SAMPLE_RATE,
        (np.clip(full, -1, 1) * 32767).astype(np.int16),
    )

    enroll_embedding_engine("campp", np.concatenate(passes[:2]), directory, vad)
    prof = profile_store.load(directory)
    verifier = create_verifier(prof)

    genuine = np.asarray(scores_for_recording(verifier, vad, passes[2]))
    if len(genuine) < 4:
        return {"error": "too little voiced audio in the third pass - read it fully"}
    g_mean, g_sd = float(genuine.mean()), float(genuine.std())
    ENROLL_STATS.update(g_mean=g_mean, g_sd=g_sd)
    # margins from measured same-channel impostor deltas (~0.08 below genuine)
    open_t = round(g_mean - 0.045, 3)
    close_t = round(g_mean - 0.065, 3)
    recent_close = _calibrate_recent(verifier, vad, passes[2])
    prof.save_thresholds(open_t, close_t, recent_close)
    _refine_print(prof, verifier, vad, passes[2])

    mic_print = fingerprint_from_recording(full, vad)
    if mic_print is not None:
        np.save(directory / profile_store.MIC_PRINT_FILE, mic_print)

    noise = np.random.default_rng(0).standard_normal(3 * config.SAMPLE_RATE)
    noise_score = create_verifier(profile_store.load(directory)).score(
        noise.astype(np.float32) * 0.01
    )
    margin = g_mean - noise_score
    return {
        "genuine_mean": round(g_mean, 3),
        "genuine_sd": round(g_sd, 3),
        "open": open_t,
        "close": close_t,
        "noise_margin": round(margin, 3),
        "ok": bool(margin >= config.ENROLL_SELF_TEST_MARGIN),
    }


def api_impostor() -> dict:
    from truemic.calibrate import scores_for_recording
    from truemic.verify import create_verifier

    vad = models()["vad"]
    prof = profile_store.load()
    verifier = create_verifier(prof)
    recording = RECORDINGS.get("impostor")
    if recording is None:
        return {"error": "no impostor recording"}
    genuine_rms = float(np.sqrt(np.mean(RECORDINGS["pass1"] ** 2)))
    impostor_rms = float(np.sqrt(np.mean(recording**2)))
    if impostor_rms < config.CALIBRATE_IMPOSTOR_MIN_RMS_RATIO * genuine_rms:
        return {"error": "impostor too quiet - they must speak as close and loud as you"}
    scores = np.asarray(scores_for_recording(verifier, vad, recording))
    if len(scores) < 4:
        return {"error": "not enough impostor speech"}
    i_mean, i_sd = float(scores.mean()), float(scores.std()) + 1e-6
    # pre-refine stats stashed at enrollment: rescoring pass3 here would be
    # self-inflated (its chunks are inside the refined print by now)
    if "g_mean" not in ENROLL_STATS:
        return {"error": "enrollment stats not in this session - redo setup first"}
    g_mean = ENROLL_STATS["g_mean"]
    g_sd = ENROLL_STATS["g_sd"] + 1e-6
    separation = g_mean - i_mean
    # never wedge the flow: measure, report, and only retune when the data
    # clearly supports it — enrollment thresholds are already near-optimal
    if separation >= 0.05:
        open_t = round((g_mean * i_sd + i_mean * g_sd) / (g_sd + i_sd), 3)
        close_t = round(min(open_t - 0.02, i_mean + 0.5 * i_sd), 3)
        prof.save_thresholds(open_t, close_t, prof.recent_close_threshold)
        verdict = "tuned"
    elif separation >= 0.02:
        open_t, close_t, verdict = prof.open_threshold, prof.close_threshold, "thin"
    else:
        open_t, close_t, verdict = prof.open_threshold, prof.close_threshold, "none"
    return {
        "verdict": verdict,
        "impostor_mean": round(i_mean, 3),
        "open": open_t,
        "close": close_t,
        "separation": round(separation, 3),
    }


def api_start(monitor_only: bool) -> dict:
    from truemic.devices import find_device
    from truemic.pipeline import LiveGate
    from truemic.verify import VoiceActivity, create_verifier
    from truemic.watermark import Watermarker

    if GATE["live"] is not None:
        return {"error": "already running"}
    GATE["live"] = "starting"  # claim the slot before slow work: no double-start race
    try:
        prof = profile_store.load()
        out_device = None
        mode = "monitor"
        if not monitor_only:
            try:
                out_device = find_device(config.DEFAULT_OUT_DEVICE, "output")
                mode = "virtual-mic"
            except LookupError:
                pass
        gate = LiveGate(
            profile=prof,
            verifier=create_verifier(prof),
            vad=VoiceActivity(),
            watermarker=Watermarker() if out_device is not None else None,
            in_device=GATE["in_device"],
            out_device=out_device,
        )
        gate.start()
        GATE["live"] = gate
        return {"mode": mode, "open_threshold": prof.open_threshold}
    except (Exception, SystemExit):
        GATE["live"] = None
        raise


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", APP_PAGE.read_bytes())
        elif self.path == "/api/level":
            self._json(LEVEL)
        elif self.path == "/api/state":
            live = GATE["live"]
            self._json(
                live.state()
                if live is not None and live != "starting"
                else {"running": live == "starting", "starting": live == "starting"}
            )
        elif self.path == "/api/status":
            try:
                prof = profile_store.load()
                self._json({"enrolled": True, "name": prof.name,
                            "open": prof.open_threshold})
            except FileNotFoundError:
                self._json({"enrolled": False})
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/api/miccheck":
                self._json({"rows": api_miccheck()})
            elif self.path == "/api/record":
                GATE["in_device"] = body.get("device", GATE["in_device"])
                wav = record_with_level(float(body["seconds"]), GATE["in_device"])
                RECORDINGS[body["label"]] = wav
                self._json({"rms": round(float(np.sqrt(np.mean(wav**2))), 4)})
            elif self.path == "/api/enroll":
                self._json(api_enroll(body.get("name", "You")))
            elif self.path == "/api/impostor":
                self._json(api_impostor())
            elif self.path == "/api/start":
                self._json(api_start(bool(body.get("monitor_only", False))))
            elif self.path == "/api/stop":
                if GATE["live"] is not None and GATE["live"] != "starting":
                    GATE["live"].stop()
                    GATE["live"] = None
                self._json({"stopped": True})
            else:
                self._send(404, "text/plain", b"not found")
        except (Exception, SystemExit) as error:  # surface, don't crash the flow
            self._json({"error": str(error) or "failed"})

    def _json(self, data: dict):
        self._send(200, "application/json", json.dumps(data).encode())

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def preload_models() -> None:
    """First-run downloads (CAM++ 29MB, AudioSeal ~90MB, silero) happen here in
    the background instead of freezing a button mid-wizard."""
    try:
        models()
        from truemic.verify import _campp_session

        _campp_session()
        from truemic.watermark import Watermarker

        Watermarker()
        print("models warmed")
    except Exception as error:
        print(f"model preload: {error}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=preload_models, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"TrueMic app: {url}")
    threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if GATE["live"]:
            GATE["live"].stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
