"""Rate every real input device at once: speak for ~6 s, get a ranked table.

Bandwidth is what speaker verification lives on — an enrollment mic needs
energy above 2 kHz. Run this on any machine before enrolling.
"""

import sys
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import welch

from truemic import config

VIRTUAL = ("CABLE", "Voicemeeter", "Steam", "Sound Mapper", "Wave", "Primary Sound")
SECONDS = 6.0

_LINUX_ALIAS_NAMES = frozenset({"sysdefault", "default", "pulse", "pipewire"})
_LINUX_BACKEND_PREF = ("pipewire", "pulse", "default", "sysdefault")


def _linux_deduplicate(candidates):
    if sys.platform != "linux":
        return candidates
    aliases = {n for _, n in candidates if n.lower() in _LINUX_ALIAS_NAMES}
    if not aliases:
        return candidates
    chosen_name = next(
        (n for pref in _LINUX_BACKEND_PREF
         for _, n in candidates if n.lower() == pref),
        None,
    )
    return [(i, n) for i, n in candidates
            if n.lower() not in _LINUX_ALIAS_NAMES or n == chosen_name]


def main() -> None:
    candidates = []
    seen = set()
    for index, device in enumerate(sd.query_devices()):
        name = device["name"]
        if device["max_input_channels"] == 0 or any(v in name for v in VIRTUAL):
            continue
        # Bluetooth headset endpoints share mangled names — try every one
        if name in seen and not any(k in name for k in ("Headset", "Hands")):
            continue
        seen.add(name)
        candidates.append((index, name))

    candidates = _linux_deduplicate(candidates)

    streams, buffers = [], {}
    for index, name in candidates:
        buffers[index] = []

        def callback(indata, frames, time_info, status, idx=index):
            buffers[idx].append(indata[:, 0].copy())

        opened = threading.Event()
        error_holder: list = []

        def open_stream(idx=index, cb=callback):
            try:
                stream = sd.InputStream(
                    samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                    device=idx, callback=cb,
                )
                stream.start()
                streams.append(stream)
            except Exception as e:
                error_holder.append(e)
            finally:
                opened.set()

        threading.Thread(target=open_stream, daemon=True).start()
        if not opened.wait(timeout=3.0) or error_holder:
            buffers.pop(index, None)
            continue



    live = [(i, n) for i, n in candidates if i in buffers]
    print(f"Recording from {len(live)} devices — SPEAK NORMALLY for {SECONDS:.0f} seconds...")
    sd.sleep(int(SECONDS * 1000))
    for stream in streams:
        stream.stop()
        stream.close()

    print(f"\n{'rms':>8} {'>2kHz':>6} {'verdict':<10} device")
    results = []
    for index, name in live:
        if not buffers[index]:
            continue
        wav = np.concatenate(buffers[index])
        rms = float(np.sqrt(np.mean(wav**2)))
        if rms < 1e-5:
            results.append((0.0, rms, "silent", name))
            continue
        f, p = welch(wav, config.SAMPLE_RATE, nperseg=2048)
        high = float(p[f >= 2000].sum() / p.sum())
        verdict = (
            "GOOD" if rms > 0.003 and high > 0.05
            else "narrowband" if rms > 0.003
            else "too quiet"
        )
        results.append((high if rms > 0.003 else 0.0, rms, verdict, name))
    for high, rms, verdict, name in sorted(results, reverse=True):
        print(f"{rms:8.4f} {high:6.0%} {verdict:<10} {name}")
    print("\nEnroll on the best GOOD device: python -m truemic.enroll --name ... --in \"<substring>\"")


if __name__ == "__main__":
    main()
