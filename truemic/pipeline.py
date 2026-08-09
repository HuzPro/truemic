"""The live gate as a reusable engine: capture -> VAD -> verify -> watermark ->
delay-line gate -> output. The CLI sender and the app dashboard both drive it."""

import queue
import threading
import time

import numpy as np
import sounddevice as sd

from truemic import config
from truemic.audio import RollingWindow, SampleQueue
from truemic.gate import DelayGate, GateDecision
from truemic.micprint import RollingMicPrint
from truemic.tokens import current_slot, token_for_slot

OUTPUT_PREFILL_SECONDS = 1.0  # jitter slack for flaky input devices


class LiveGate:
    def __init__(
        self,
        profile,
        verifier,
        vad,
        watermarker,
        in_device: int | None,
        out_device: int | None,
        out_rate: int = config.DEFAULT_OUT_RATE,
        badge_only: bool = False,
        strict_mic: bool = False,
    ):
        self._profile = profile
        self._verifier = verifier
        self._vad = vad
        self._watermarker = watermarker
        self._in_device = in_device
        self._out_device = out_device
        self._out_rate = out_rate
        self._badge_only = badge_only
        self._strict_mic = strict_mic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = {
            "running": False,
            "open": False,
            "window_score": None,
            "recent_score": None,
            "voiced": 0.0,
            "wm_load": 0.0,
            "muted_voiced_seconds": 0.0,
            "mic_alarm": False,
            "injected": False,
            "backlog": 0,
            "drops_seconds": 0.0,
        }

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def start(self) -> None:
        self._stop.clear()
        # running=True from the caller's perspective immediately — stream-open
        # takes seconds and a poll in that window must not read "dead"
        self._set(running=True)
        self._thread = threading.Thread(target=self._run_guarded, daemon=True)
        self._thread.start()

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as error:  # a dead stream must say WHY on the dashboard
            self._set(running=False, error=str(error))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._set(running=False)

    def _run(self) -> None:
        prof = self._profile
        gate = DelayGate(
            delay_samples=int(config.GATE_DELAY_SECONDS * config.SAMPLE_RATE),
            fade_samples=int(config.CROSSFADE_MS / 1000 * config.SAMPLE_RATE),
        )
        decision = GateDecision(
            open_threshold=prof.open_threshold,
            close_threshold=prof.close_threshold,
            silence_timeout=config.SILENCE_TIMEOUT_SECONDS,
            open_confirmations=config.OPEN_CONFIRMATIONS,
            recent_close_threshold=prof.recent_close_threshold,
        )
        ring = RollingWindow(config.WINDOW_SECONDS)
        mic_guard = RollingMicPrint(prof.mic_print)
        in_queue: queue.Queue[np.ndarray] = queue.Queue()
        out_queue = SampleQueue()
        out_queue.put(np.zeros(int(OUTPUT_PREFILL_SECONDS * self._out_rate), dtype=np.float32))
        comfort_rng = np.random.default_rng()
        comfort_amp = 10 ** (config.COMFORT_NOISE_DBFS / 20)

        def in_callback(indata, frames, time_info, status):
            in_queue.put(indata[:, 0].copy())

        def out_callback(outdata, frames, time_info, status):
            outdata[:, 0] = out_queue.pop(frames)

        wm_pending = np.zeros(0, dtype=np.float32)
        chunk_samples = int(config.WM_CHUNK_SECONDS * config.SAMPLE_RATE)
        recent_samples = int(config.RECENT_WINDOW_SECONDS * config.SAMPLE_RATE)
        last_decide = time.monotonic() + config.DECIDE_EVERY_SECONDS / 2
        embed_ema = 0.0
        muted_voiced = 0.0

        streams = [
            sd.InputStream(
                samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=config.BLOCK_SAMPLES, device=self._in_device,
                callback=in_callback,
            )
        ]
        if self._out_device is not None:
            out_block = round(config.BLOCK_SAMPLES * self._out_rate / config.SAMPLE_RATE)
            streams.append(
                sd.OutputStream(
                    samplerate=self._out_rate, channels=1, dtype="float32",
                    blocksize=out_block, device=self._out_device, callback=out_callback,
                )
            )
        for stream in streams:
            stream.start()
        try:
            while not self._stop.is_set():
                try:
                    block = in_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                ring.append(block)
                for frame, voiced in self._vad.feed(block):
                    mic_guard.update(frame, voiced)
                self._verifier.feed(block)
                wm_pending = np.concatenate([wm_pending, block])

                if len(wm_pending) >= chunk_samples:
                    chunk, wm_pending = wm_pending[:chunk_samples], wm_pending[chunk_samples:]
                    allow = decision.is_open
                    chunk_has_voice = (
                        self._vad.recent_voiced_fraction(config.WM_CHUNK_SECONDS) > 0
                    )
                    if (
                        self._watermarker is not None
                        and chunk_has_voice
                        and (not self._badge_only or allow)
                    ):
                        started = time.perf_counter()
                        slot = current_slot(time.time(), config.TOKEN_SLOT_SECONDS)
                        chunk = self._watermarker.embed(
                            chunk, token_for_slot(prof.token_key, slot)
                        )
                        embed_ema = 0.8 * embed_ema + 0.2 * (time.perf_counter() - started)
                    if self._badge_only:
                        out = chunk
                    else:
                        out = gate.process(chunk)
                        out = out + (
                            comfort_amp
                            * (1.0 - gate.gain)
                            * comfort_rng.standard_normal(len(out)).astype(np.float32)
                        )
                    if self._out_device is not None:
                        if self._out_rate != config.SAMPLE_RATE:
                            import soxr

                            out = soxr.resample(
                                out, config.SAMPLE_RATE, self._out_rate
                            ).astype(np.float32)
                        out_queue.put(out)

                now = time.monotonic()
                if now - last_decide >= config.DECIDE_EVERY_SECONDS and ring.full:
                    last_decide = now
                    window = ring.window()
                    voiced = self._vad.recent_voiced_fraction(config.WINDOW_SECONDS)
                    window_score = (
                        self._verifier.score(window)
                        if voiced >= config.MIN_VOICED_FRACTION
                        else None
                    )
                    recent_score = None
                    if window_score is not None:
                        recent_voiced = self._vad.recent_voiced_fraction(
                            config.RECENT_WINDOW_SECONDS
                        )
                        if recent_voiced >= config.MIN_VOICED_FRACTION:
                            recent_score = self._verifier.score(window[-recent_samples:])
                    allow = decision.update(window_score, recent_score, now)
                    mic_match = mic_guard.match()
                    injected = mic_guard.zero_fraction() >= config.MIC_ZERO_FRACTION_ALARM
                    mic_alarm = injected or (
                        mic_match is not None and mic_match < config.MIC_MATCH_THRESHOLD
                    )
                    if self._strict_mic and mic_alarm:
                        allow = False
                    gate.set_allow(allow)
                    if not allow and voiced >= config.MIN_VOICED_FRACTION:
                        muted_voiced += config.DECIDE_EVERY_SECONDS
                    self._set(
                        open=allow,
                        window_score=window_score,
                        recent_score=recent_score,
                        voiced=round(voiced, 2),
                        wm_load=round(embed_ema / config.WM_CHUNK_SECONDS, 2),
                        muted_voiced_seconds=round(muted_voiced, 1),
                        mic_alarm=mic_alarm,
                        injected=injected,
                        backlog=in_queue.qsize(),
                        drops_seconds=round(out_queue.padded_samples / self._out_rate, 1),
                    )
        finally:
            for stream in streams:
                stream.stop()
                stream.close()
            self._set(running=False)
