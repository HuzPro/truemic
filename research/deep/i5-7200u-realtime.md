# Running TrueMic in real time on an Intel i5-7200U (Kaby Lake-U, 2C/4T, 15 W)

Deep-dive research + measurement, 2026-08-08. Target: the demo laptop (i5-7200U, Windows 10/11,
~8 GB RAM). Dev box used for all measurements: **AMD Ryzen 5 7600** (Zen 4, 6C/12T, Windows 11,
Python 3.12, `torch 2.9.1+rocm7.2.1`, `audioseal 0.2.0`, `silero-vad 6.2.1`, `speechbrain 1.1.0`,
`onnxruntime 1.24.4`).

---

## 0. Bottom line

Three findings dominate everything else, and two of them are bugs already present in the code:

1. **`import silero_vad` calls `torch.set_num_threads(1)` process-wide, at import time.** TrueMic
   therefore runs ECAPA single-threaded on every machine today. Measured penalty on the dev box:
   **87.9 ms → 529.0 ms** for one 2.5 s ECAPA window — a 6.0x regression, not the 2x you would
   expect from losing a thread. On the i5-7200U that is the difference between 22 % and 130 % of
   the machine. One line fixes it.
2. **PyTorch's OpenMP pool busy-waits for 200 ms after every parallel region** (default
   `KMP_BLOCKTIME`). Measured: **188 ms of CPU burned during 2 s of idle** after a single conv,
   ~9.4 % of a core, purely spinning. With inference every 0.5–1.0 s the spin window covers most
   of the duty cycle. `OMP_WAIT_POLICY=PASSIVE` removes it **with no loss of speed**
   (30.1 ms vs 30.6 ms on the same conv).
3. **The sender does ECAPA on the same thread that feeds the output stream.** A 220 ms ECAPA call
   on the i5 stalls watermark→gate→`out_queue` for 220 ms, while the output callback drains
   `out_queue` in real time. `SampleQueue.pop` zero-fills on underrun, so this is an audible hole,
   not a warning.

With those three fixed and the recommended low-power config (§9), the sender costs **~40 % of a
2-core i5-7200U** and sender+receiver+browser costs **~62 %**, leaving 38 % headroom. Adding Zoom
*with video* to the same laptop does not fit; the receiver and badge browser should move to the
second laptop the demo already needs for the air-gap beat.

---

## 1. Method, and how dev-box numbers become i5-7200U numbers

Published per-model CPU benchmarks for this exact stack essentially do not exist (see §7), so I
measured every stage directly and scaled. Raw scripts and JSON live in the session scratchpad;
all timings below are medians of 8–15 runs after 4–6 warmup iterations.

### 1.1 Three independent scaling estimates, which agree

**(a) PassMark single-thread.** i5-7200U = **1729**; Ryzen 5 7600 = **3908**. Ratio **2.26x**.
Sources: [i5-7200U](https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i5-7200U+%40+2.50GHz&id=2865),
[R5 7600](https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+5+7600&id=5172).

**(b) Peak AVX2 FP32 throughput.** Both microarchitectures retire **2 × 256-bit FMA per cycle per
core** = 8 lanes × 2 flops × 2 units = **32 FP32 FLOP/cycle/core**. Kaby Lake inherits Skylake's
two FMA units and supports AVX2+FMA
([spec](https://www.cpu-monkey.com/en/cpu-intel_core_i5_7200u)); Zen 4 matches it at 256-bit.
So the gap is purely clock:

- R5 7600 under a 2-thread AVX2 load: ~5.0 GHz → 5.0 × 32 = **160 GFLOPS/core**, 320 for two cores.
- i5-7200U under sustained load: **2.5 GHz** (base). Multiple field reports put a loaded 7200U at
  2.48–2.50 GHz, dropping to base once it reaches ~80 °C
  ([1](https://blog.silverpc.hu/2025/10/19/the-mobile-challenge-can-you-really-overclock-an-i5-7200u-laptop-cpu/),
  [2](https://community.intel.com/t5/Processors/i5-7200U-stuck-at-0-38Ghz-Miix-720-12IKB/td-p/522115)).
  → 2.5 × 32 = **80 GFLOPS/core**, 160 for two cores.

Ratio **2.0x**.

**(c) Model-specific FLOP check.** I counted MACs with forward hooks on every `Conv1d`,
`ConvTranspose1d`, `Linear` and `LSTM`:

| model | measured cost |
|---|---|
| AudioSeal generator (`audioseal_wm_16bits`, **14,675,873** params) | 0.42 GMAC per 0.5 s → **0.84 GFLOP per second of audio** |
| SpeechBrain ECAPA-TDNN (**20,767,552** params) | 4.71 GMAC → **9.41 GFLOP per 2.5 s window** |
| AudioSeal detector (`audioseal_detector_16bits`) | **8,647,298** params (MAC count unreliable — see §7.4) |
| silero-VAD v6 | **462,594** params |

ECAPA at 95 ms on 2 Zen 4 cores = 9.41 / 0.095 = **99 GFLOPS achieved = 31 % of the 320 GFLOPS
peak** — i.e. it is genuinely FLOP-bound, so peak-throughput scaling is valid for it. Applying the
same 31 % efficiency to the i5's 160 GFLOPS two-core peak gives 50 GFLOPS → 9.41 / 50 =
**188 ms**. Method (a) predicts 95 × 2.26 = 215 ms. The two agree within 15 %.

> The AudioSeal generator achieves only 0.42 GFLOP / 0.0347 s = **12 GFLOPS = 3.7 % of peak** — it
> is *overhead*-bound (many small convs, `weight_norm`, an LSTM), not FLOP-bound. Overhead-bound
> code tracks scalar single-thread performance, i.e. the PassMark ratio, more closely than the
> clock ratio.

**Adopted factor: ×2.3 nominal, ×2.6 for a thermally soaked machine.** Every i5-7200U figure
below is a scaled prediction, clearly labelled as such. No i5-7200U was available to test on.

### 1.2 Measured dev-box timings (median wall ms)

| operation | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| AudioSeal embed, 0.25 s chunk | 46.5 | 24.0 | 22.1 |
| AudioSeal embed, 0.5 s chunk | 90–101 | 34.5–35.2 | 29.6–31.0 |
| AudioSeal embed, 1.0 s chunk | 129–175 | 56.4–60.9 | 41.6–45.6 |
| AudioSeal embed, 2.0 s chunk | 208 | 76.7 | 59.3 |
| AudioSeal detect, 1.0 s window | 77–89 | 38.6–43.9 | 33.6 |
| AudioSeal detect, 1.5 s window | 88–113 | 43.3–45.9 | 37.2 |
| **ECAPA, 2.5 s window** | **522–564** | **81.2–102** | 64.3 |
| ECAPA, 2.0 s window | 456 | 73.5 | — |
| silero-VAD **TorchScript**, 0.5 s hop (15 × 512) | 11.1–12.6 | 11.6 | — |
| silero-VAD **ONNX**, 0.5 s hop (15 × 512) | 2.6–3.2 | 3.4 | — |

### 1.3 Scaled predictions for the i5-7200U (2 threads = the whole machine)

| operation | i5-7200U, ×2.3 | soaked, ×2.6 |
|---|---|---|
| ECAPA, 2.5 s window, 2 threads | **220 ms** | 250 ms |
| ECAPA, 2.5 s window, **1 thread** | **1210 ms** | 1370 ms |
| ECAPA, 2.0 s window, 2 threads | 170 ms | 190 ms |
| AudioSeal embed 0.5 s, 2 threads | 80 ms | 91 ms |
| AudioSeal embed 1.0 s, 2 threads | 135 ms | 152 ms |
| AudioSeal embed 2.0 s, 2 threads | 176 ms | 199 ms |
| AudioSeal detect 1.5 s, 2 threads | 105 ms | 119 ms |
| AudioSeal detect 1.5 s, 1 thread | 261 ms | 295 ms |
| silero ONNX, 0.5 s hop | 7 ms | 8 ms |
| silero TorchScript, 0.5 s hop | 28 ms | 32 ms |

The 1-thread ECAPA row is the current shipping behaviour. **1210 ms of work per 500 ms decide
interval = 242 % of the machine.** TrueMic as configured today cannot run on this laptop.

---

## 2. Finding 1 — `silero_vad` silently pins torch to one thread

`silero_vad/model.py`, line 3:

```python
from .utils_vad import init_jit_model, OnnxWrapper
import torch
torch.set_num_threads(1)          # <-- executes on import, process-wide
```

`truemic/verify.py::VoiceActivity.__init__` does `from silero_vad import load_silero_vad`, and
`sender.py` constructs `VoiceActivity()` *after* `create_verifier(prof)`. So ECAPA is loaded with
the right thread count and then scored with one thread forever after.

Isolated repro (`b4.py`), same process, same model object:

```
mode: none         torch threads: 2
  ecapa BEFORE:  87.74 ms      ecapa AFTER:  85.47 ms
mode: ort_default  torch threads: 2
  ecapa BEFORE:  87.86 ms
  ORT session created;  torch threads: 1     <-- silero import did this
  ecapa AFTER: 529.03 ms
  ecapa AFTER + re-set_num_threads(2):  83.37 ms
```

Two things worth internalising:

- The penalty is **6.0x, not 2x.** Going from 2 threads to 1 does not merely halve throughput; it
  drops oneDNN onto a different (much worse) code path for ECAPA's grouped/dilated `Conv1d` stack.
  AudioSeal shows the same shape at smaller magnitude (2.8x for a 2→1 thread change).
- It is **recoverable**: calling `torch.set_num_threads(2)` again restores full speed.

**Fix:** in `sender.py::main()` and `receiver.py::main()`, after all models are constructed:

```python
import torch
torch.set_num_threads(2)   # AFTER VoiceActivity(); silero's import resets this to 1
```

Placing it in `truemic/__init__.py` does **not** work — the silero import happens later and wins.

---

## 3. Finding 2 — the OpenMP pool burns a core doing nothing

PyTorch's OpenMP runtime keeps worker threads spinning after a parallel region, for
`KMP_BLOCKTIME` milliseconds (default **200 ms**), so the next region starts without a thread
wake-up. The PyTorch tuning guide recommends 0 or 1 for inference workloads
([Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)).

Measured (`b6.py`) — CPU burned by the process during 2 s of `time.sleep()` immediately after one
`Conv1d` call, plus the cost of that same call:

| setting | CPU burned while idle 2 s | conv1d median |
|---|---|---|
| default | **188 ms (9.4 % of a core)** | 30.6 ms |
| `OMP_WAIT_POLICY=PASSIVE` | **0 ms** | 30.1 ms |
| `KMP_BLOCKTIME=0` | **0 ms** | 30.7 ms |

**Free.** No measurable slowdown, and on a 2-core machine the spinning thread is directly
contending with PortAudio's callback thread. Must be set **before `import torch`**, so it belongs
next to the existing `NO_TORCH_COMPILE` line in `truemic/__init__.py`:

```python
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
```

Note the spin also corrupts naive `time.process_time()` budgeting — an early measurement pass
reported 4183 ms of "CPU" for a 564 ms single-threaded ECAPA call. Budget from wall time × threads,
not from process CPU time, unless spinning is disabled.

---

## 4. Torch CPU tuning on 2C/4T

### 4.1 `torch.set_num_threads` — use 2 (physical cores), not 4

PyTorch defaults to the **physical** core count, and both the PyTorch and Intel guidance is to
avoid logical cores because hyperthread siblings contend for the same execution resources
([Grokking PyTorch Intel CPU performance](https://docs.pytorch.org/tutorials/intermediate/torchserve_with_ipex.html),
[CPU threading and TorchScript inference](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference)).
For conv workloads on a 2C/4T part that means **2**.

Dev-box scaling of ECAPA (repeated sweep, stable across reps):

```
threads=1: 549.7 / 526.6 ms
threads=2:  81.9 /  81.2 ms      <-- the cliff is between 1 and 2
threads=3:  69.8 /  69.6 ms
threads=4:  64.4 /  64.3 ms
threads=6:  57.8 /  56.5 ms
```

Diminishing returns past 2 are steep even on real cores (2→4 buys 21 %). On the i5 the extra two
"cores" are hyperthreads, which would buy less than that while starving the audio thread. **2.**

A tempting alternative — run torch at 1 thread so one core stays free for audio — is a bad trade
here specifically *because* of the 6x single-thread cliff. Keep 2 threads and buy audio robustness
with buffering (§6.4) and process priority (§6.1) instead.

### 4.2 ECAPA stage breakdown — the feature extractor is free

At 2 threads, on a 2.5 s window:

```
compute_features (fbank):   1.33 ms
mean_var_norm:              0.06 ms
embedding_model (TDNN):    79.52 ms   <-- 98 %
full encode_batch:         81.19 ms
```

There is nothing to win in preprocessing. All ECAPA optimisation must target the TDNN itself
(shorter window, fewer decisions, a smaller model, or int8).

### 4.3 `set_flush_denormal` — keep it, but do not expect a win

Denormals are much slower than normal arithmetic on x86 and PyTorch exposes
`torch.set_flush_denormal(True)` to flush them
([docs](https://docs.pytorch.org/docs/stable/generated/torch.set_flush_denormal.html)); reports of
~2x on audio inference exist
([paulw.tokyo](https://paulw.tokyo/denormal-number-at-inference-in-pytorch/)).

**Measured here: no effect.** 1 thread, FTZ on vs off — AudioSeal embed 0.5 s 90.4 vs 81.8 ms,
ECAPA 2.5 s 522.3 vs 533.4 ms. Both within run-to-run noise, and the sign flips between rows.

Caveat: my inputs were random noise, which never decays into denormal range. A real mic feed
contains long near-silent stretches feeding a network with an LSTM and residual tails, which is
exactly where denormals appear. The call costs one line and cannot hurt, so **keep it**, but do
not count it in the budget.

### 4.4 `inference_mode` vs `no_grad` — no difference, do not bother

`inference_mode` skips view tracking and version-counter bumps
([docs](https://docs.pytorch.org/docs/2.12/generated/torch.autograd.grad_mode.inference_mode.html)),
which matters for many small ops. Measured on ECAPA 2.5 s at 2 threads: `no_grad` **82.69 ms**,
`inference_mode` **83.19 ms**. Identical. The existing `torch.no_grad()` calls are fine; changing
them is churn.

### 4.5 `channels_last` — not applicable

`channels_last` is the NHWC memory format for **4-D** tensors, and the oneDNN win comes from
avoiding NCHW↔blocked reorders for 2-D convolutions
([Intel](https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-vision-models-with-channels-last-on-cpu.html),
[oneDNN integration plan](https://gist.github.com/mingfeima/595f63e5dd2ac6f87fdb47df4ffe4772)).
Every model in this pipeline is `Conv1d` on 3-D tensors, for which PyTorch has no `channels_last`
format. Nothing to do.

### 4.6 AudioSeal chunk size — real, sublinear savings

Watermarking cost is far from proportional to chunk length (2 threads, dev box):

| chunk | wall | cost as % of real time |
|---|---|---|
| 0.25 s | 24.0 ms | 9.6 % |
| 0.5 s | 34.7 ms | 6.9 % |
| 1.0 s | 60.9 ms | 6.1 % |
| 2.0 s | 76.7 ms | 3.8 % |

Fixed per-call overhead is roughly 15 ms. Going 0.5 → 1.0 s buys ~12 % of the watermark budget and
costs +0.5 s of end-to-end latency; 0.5 → 2.0 s buys 45 % and costs +1.5 s. Given the README's
2.1 s latency budget, **1.0 s is the right stop**.

### 4.7 AudioSeal streaming API is not a CPU win — skip it

AudioSeal 0.2 added `model.streaming(batch_size=...)` with a convolutional cache, plus
`audioseal_wm_streaming` / `audioseal_detector_streaming` checkpoints
([repo](https://github.com/facebookresearch/audioseal)). It looks like an obvious optimisation.
It is not:

```
threads=2, 0.5 s chunk, non-streaming generator: 34.73 ms
threads=2, 0.5 s chunk, streaming generator:     36.88 ms   (identical param count, 14,675,873)
```

The cache buys **continuity across chunk boundaries**, not throughput. Do not spend hackathon
hours on this refactor for performance reasons.

---

## 5. ONNX Runtime on 2C/4T

### 5.1 The one place it is already a large, free win: silero-VAD

`load_silero_vad(onnx=True)` is a one-word change and measured **3.5–4.3x faster**:

| VAD backend | 0.5 s hop (15 × 512 frames) | per frame |
|---|---|---|
| TorchScript | 11.1–12.6 ms | 0.72–0.75 ms |
| ONNX | 2.6–3.2 ms | 0.17 ms |

This matches upstream's own claim that ONNX "may run up to 4-5x faster"
([silero-vad README](https://github.com/snakers4/silero-vad)); their headline "<1 ms per 30 ms
chunk on a single CPU thread" is confirmed on both backends here. On the i5 this is 28 ms → 7 ms
per decision, i.e. ~2 % of the machine recovered, and it removes a torch parallel region from the
hot path entirely. `silero-vad` already depends on `onnxruntime>=1.16.1` for this path and 1.24.4
is installed.

### 5.2 Recommended session options for a 2-core box

From the ONNX Runtime
[thread management docs](https://onnxruntime.ai/docs/performance/tune-performance/threading.html):

- `intra_op_num_threads`: default 0 = physical core count *with* automatic affinitization; setting
  it explicitly disables affinity. For a small model on a 2-core machine, **1** is right — the
  model is too small to parallelise usefully and you want the other core free.
- `inter_op_num_threads`: only relevant under `ORT_PARALLEL`. Leave sequential.
- `execution_mode`: `ORT_SEQUENTIAL` (default). `ORT_PARALLEL` helps only models with many
  independent branches, and "could also hurt performance on some models without many branches".
- `graph_optimization_level`: `ORT_ENABLE_ALL`.
- **Spinning**: `session.intra_op.allow_spinning=0` stops ORT threads busy-waiting — the same
  problem as §3. The docs' preferred middle ground is time-bounded spin
  (`spin_duration_us=1000`, `spin_backoff_max=8`, "the most consistent best performer"). With
  `intra_op_num_threads=1` there is no pool to spin, so this is moot for silero.

silero's own wrapper (`silero_vad/utils_vad.py`) already sets `intra_op_num_threads = 1` and
`inter_op_num_threads = 1` and pins `CPUExecutionProvider`. **Nothing to tune — just pass
`onnx=True`.**

### 5.3 ONNX-ifying ECAPA or AudioSeal: attractive, untested, not a hackathon-day job

ORT-vs-eager gains on CPU are real but model-dependent — large models see 1.4–3.4x, tiny ones can
see ~1.0x
([ORT blog](https://onnxruntime.ai/blogs/accelerating-phi-2),
[case study](https://medium.com/@naokig/from-2s-to-600ms-pytorch-vs-onnx-runtime-5fb2ef14e4f0)).
ECAPA is the interesting target because it is FLOP-bound and dense, and because ORT's int8 path
uses `VPMADDUBSW` on AVX2 hardware
([quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)) —
plausibly 2–3x on top of the graph win, which would take ECAPA from 220 ms to well under 100 ms on
the i5.

**But:** my `torch.onnx.export` attempts for both ECAPA and the AudioSeal generator failed in-process
with `ImportError: Lazy import of LazyModule(... speechbrain.integrations.k2_fsa ...)` — an
interaction between the exporter's module scan and SpeechBrain 1.1's lazy-module machinery. It
would need to be exported from a clean process without SpeechBrain imported, then re-validated for
score drift against the calibrated threshold. There is also **no pre-converted ONNX
`spkrec-ecapa-voxceleb` on the Hub**; WeSpeaker publishes ONNX speaker models
([toolkit](https://ar5iv.labs.arxiv.org/html/2210.17016)) but swapping models means re-enrolling
and re-calibrating.

Verdict: **do not attempt on demo day.** The cheap levers in §9 already get you to 38 % headroom.

---

## 6. Windows real-time audio hygiene for this app

### 6.1 Process priority — ABOVE_NORMAL, not HIGH

```python
import os, psutil
psutil.Process(os.getpid()).nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
```

`psutil` is not currently in `requirements.txt` but is installed. Microsoft explicitly warns to
"use extreme care with the high-priority class, because such an application can use nearly all
available CPU time"
([SetPriorityClass](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass)) —
on **two** cores, HIGH means the shell and the browser stop responding, and `AccessDenied` is
possible. ABOVE_NORMAL is enough to win against a browser and Zoom without that risk.

### 6.2 MMCSS — PortAudio may already do it, and you can reach it from Python if needed

WASAPI assigns the MMCSS task **"Pro Audio"** to exclusive-mode transport threads with a device
period under 10 ms
([Core Audio docs](https://github.com/MicrosoftDocs/win32/blob/docs/desktop-src/CoreAudio/exclusive-mode-streams.md),
[MMCSS](https://learn.microsoft.com/en-us/windows/win32/procthread/multimedia-class-scheduler-service)),
and PortAudio's WASAPI backend sets thread priority automatically — "Audio" for shared mode,
"ProAudio" for exclusive, overridable via `paWinWasapiThreadPriority`
([pa_win_wasapi.h](https://www.portaudio.com/docs/v19-doxydocs/pa__win__wasapi_8h_source.html)).
The **WMME** backend does not use MMCSS but does raise the processing thread to
`THREAD_PRIORITY_HIGHEST`
([PortAudio WMME/DS/WDMKS](https://deepwiki.com/PortAudio/portaudio/3.1.2-directsound-wmme-and-wdmks-implementations)).

Since your Python callback *runs on* PortAudio's callback thread, you can register that thread with
MMCSS yourself, once, from inside the callback:

```python
import ctypes
_mmcss_done = False
def in_callback(indata, frames, time_info, status):
    global _mmcss_done
    if not _mmcss_done:
        idx = ctypes.c_ulong(0)
        ctypes.windll.avrt.AvSetMmThreadCharacteristicsW("Pro Audio", ctypes.byref(idx))
        _mmcss_done = True
    ...
```

`AvSetMmThreadCharacteristics` must be called from the thread that gets the boost, and the task
name must match a subkey of
`HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Multimedia\SystemProfile\Tasks`
([docs](https://learn.microsoft.com/en-us/windows/win32/api/avrt/nf-avrt-avsetmmthreadcharacteristicsa)).
**Optional.** Try it only if dropouts persist after §6.4.

### 6.3 Host API: you are on MME and probably should stay there

Empirically, on this Windows 11 box `sd.default.device` resolves to **host API 0 = MME**, and
`truemic/devices.py::find_device` returns the *first* substring match in enumeration order — which
is always an MME device, because MME enumerates first:

```
0 MME              default_in=1  default_out=14
1 DirectSound      ...
2 WASAPI           ...
3 WDM-KS           ...
--- sd.default.device --- [1, 14]        # both MME

  1 api=0 sr=44100 lat_lo=0.0900   Microphone (WO Mic Device)          # MME:     90 ms
 27 api=1 sr=44100 lat_lo=0.1200   Microphone (High Definition Audio)  # DSound: 120 ms
 50 api=2 sr=48000 lat_lo=0.0020   Voicemeeter VAIO3 Input             # WASAPI:   2 ms
```

Trade-offs
([PortAudio latency notes](https://portaudio.com/docs/latency.html),
[Audacity API comparison](https://www.flykantech.com/pages/audacity-mme-windows-directsound-and-windows-wasapi)):

- **MME** — broadest compatibility, highest latency (~90 ms here). On a *weak* machine that extra
  buffering is protective, and TrueMic already spends 2.1 s of latency; 90 ms is noise.
- **WASAPI shared** — ~2 ms, lowest CPU. But the mixer runs at 48 kHz, so opening a **16 kHz**
  stream requires `sd.WasapiSettings(auto_convert=True)`, which "is in particular required for
  streams not matching the system mixer sample rate" and **has no effect in exclusive mode**
  ([sounddevice platform-specific settings](https://python-sounddevice.readthedocs.io/en/latest/api/platform-specific-settings.html)).
  Without it, opening 16 kHz on WASAPI shared simply fails.
- **WASAPI exclusive / WDM-KS** — lowest latency, but they take the device away from every other
  app. Zoom and the badge demo need the device shared. **Do not use.**

**Recommendation: keep MME.** It is what you are already getting, it is the most forgiving, and
90 ms of device buffering on a thermally-throttled dual core is a feature.

Two MME gotchas worth knowing: MME truncates device names to 31 characters (visible above:
`"Microphone (High Definition Aud"`), so long `--in`/`--out` substrings will silently skip MME
devices and land on a DirectSound one; and `find_device` has no host-API filter, so a rename or a
new virtual device can change which API you get without any error.

### 6.4 The output starvation bug, and the free fix

`sender.py` runs one loop that does watermark → gate → `out_queue.put()` **and**, every
`DECIDE_EVERY_SECONDS`, `verifier.score(window)`. `out_queue` is drained by the output callback in
real time, and `SampleQueue.pop` zero-fills short reads:

```python
if len(taken) < n:
    taken = np.concatenate([taken, np.zeros(n - len(taken), dtype=np.float32)])
```

So a stall becomes silence, silently. How much slack is there? The loop emits one
`WM_CHUNK_SECONDS` block per ~16 iterations, so `out_queue` holds between 0 and 0.5 s of audio,
averaging ~0.25 s. A 220 ms ECAPA call on the i5 (250 ms soaked) eats essentially all of it, every
decision.

**Free fix:** prefill `out_queue` with silence at startup. This costs *zero* perceptual latency
because `DelayGate._fifo` already starts as 1.5 s of zeros and `GateDecision._open` starts
`False` — the first 1.5 s of output is silence no matter what:

```python
out_queue.put(np.zeros(int(0.4 * out_rate), dtype=np.float32))
```

400 ms covers a soaked-machine ECAPA call with margin.

**Structural fix (do this if a prefill is not enough):** move `vad.voiced_fraction()` +
`verifier.score()` into a worker thread that publishes the latest verdict to the main loop.
PyTorch releases the GIL during ops, so the watermark path keeps feeding `out_queue` throughout.
Roughly 15 lines. The prefill is one line — try it first.

### 6.5 Sample rate: `--out-rate 16000` is the wrong default for VB-CABLE

VB-CABLE runs internally at a **fixed** rate selectable from **44.1 / 48 / 88.2 / 96 / 176.4 /
192 kHz** — 16 kHz is not an option
([reference manual](https://vb-audio.com/Cable/VBCABLE_ReferenceManual.pdf)). Feeding it 16 kHz
guarantees a hidden resample somewhere in MME/the Windows mixer/the cable itself, which is exactly
the crackling failure mode the README already documents.

**Set the demo up at 48 kHz everywhere**: VB-CABLE control panel internal SR = 48000, both `CABLE
Input` and `CABLE Output` set to 48000 Hz in Windows Sound, and run
`python -m truemic.sender --out-rate 48000`. soxr handles 16 k → 48 k for ~0.3 ms per second of
audio (< 0.1 % of the budget) with a far better kernel than the OS mixer's.

### 6.6 GC, timer resolution, blocksize

- **GC.** Running the collector pauses *all* Python threads in the process, including the audio
  callback. After models are loaded (a large, permanently-live object graph), do
  `gc.collect(); gc.freeze()` — `gc.freeze()` moves long-lived objects into a permanent generation
  so the collector stops rescanning them
  ([gc docs](https://docs.python.org/3/library/gc.html),
  [Close engineering](https://making.close.com/posts/taming-the-python-gc/)). Two lines, no
  behaviour change. Do **not** `gc.disable()` — torch and the HTTP server create cycles, and a
  long-running demo would leak.
- **Timer resolution.** `ctypes.windll.winmm.timeBeginPeriod(1)` raises the scheduler tick to 1 ms.
  Since Windows 10 2004 this is per-process rather than global
  ([timeBeginPeriod](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod)).
  TrueMic's loops block on `queue.get()` and `stream.read()` rather than sleeping, so the direct
  benefit is small; MME's internal timed waits do benefit. **Low priority, and it costs battery.**
- **Blocksize.** `BLOCK_SAMPLES = 480` (30 ms) is fine. Do not shrink it — smaller blocks mean more
  Python callback invocations per second, each taking the GIL. Do not grow it either; MME's own
  ~90 ms of buffering already provides the jitter absorption, and the real risk is `out_queue`
  starvation (§6.4), which blocksize does not address.
- **Power plan.** Plug the laptop in and select High performance. Core parking, deep C-states and
  constant frequency changes are a well-known source of DPC latency spikes and audio dropouts, and
  Balanced enables all three
  ([DAW Zone](https://dawzone.com/how-to-fix-audio-dropouts-and-dpc-latency-in-windows),
  [TechCult](https://techcult.com/how-to-fix-high-dpc-latency-on-windows/)). `powercfg /setactive
  SCHEME_MIN`, and set minimum processor state to 100 % on AC.

---

## 7. Published performance anchors (and why there are so few)

### 7.1 silero-VAD — confirmed

Upstream claims "one audio chunk (30+ ms) takes less than 1 ms to be processed on a single CPU
thread" and that ONNX can be 4–5x faster
([README](https://github.com/snakers4/silero-vad)). Measured here: 0.72 ms/frame TorchScript,
0.17 ms/frame ONNX — both under 1 ms, ratio 4.2x. A community experiment reached ~600x real time
with ONNX on one thread and got 5x out of TorchScript by batching the stateless layers
([discussion #408](https://github.com/snakers4/silero-vad/discussions/408)). The VAD is **not** a
problem on any of these machines; it is ~1 % of the budget.

### 7.2 ECAPA-TDNN — no usable published CPU number

The 1024-channel ECAPA config is ~20.8 M parameters (confirmed exactly here: 20,767,552), and the
literature consistently describes it as "much higher computational complexity and slower inference
speed … inadequate for scenarios with demanding inference rate and limited computational
resources" ([CAM++](https://arxiv.org/html/2303.00332v3)). One survey reports 69.43 ms inference
([MDPI](https://www.mdpi.com/2076-3417/14/4/1329)) but does not state the CPU, thread count, or
utterance length, which makes it unusable as an anchor. Hence the FLOP-based derivation in §1.1.

### 7.3 AudioSeal — no published CPU RTF

The paper reports generation "14x faster than WavMark" and detection "up to two orders of magnitude
faster" than prior work, benchmarked on a single Nvidia Quadro GP100 over 1–10 s segments
([paper](https://arxiv.org/pdf/2401.17264), [repo](https://github.com/facebookresearch/audioseal)).
No CPU figures at all. Measured here: **14,675,873** generator params, **8,647,298** detector
params, **0.84 GFLOP per second of audio** for the generator.

### 7.4 A number I do not trust

My MAC counter reported 42.4 GFLOP/s of audio for the detector — 50x the generator — which cannot
be reconciled with its measured 43 ms for a 1.5 s window (that would require ~1.5 TFLOPS from two
Zen 4 cores whose peak is 0.32 TFLOPS). The hook almost certainly double-counts nested modules or
mis-handles `ConvTranspose1d` output lengths in the detector's head. **Budget the detector from
measured wall time only** (§1.2), not from that FLOP count.

---

## 8. Thermal behaviour: bursty vs smooth scheduling

A 15 W U-class part enforces **PL1** (sustained) and **PL2** (burst), with PL2 permitted for a
window **Tau**; once Tau expires the part falls back to PL1, and OEMs frequently set PL1 at or
below 15 W with a short Tau
([XDA](https://www.xda-developers.com/understading-cpu-pl1-pl2-power-limits/),
[AnandTech](https://at-web1.www.anandtech.com/show/13591/the-intel-core-i9-9900k-at-95w-fixing-the-power-for-sff)).
Crucially, **PL2 is enforced as a moving average over Tau, so low-power moments give budget back**.

The recommended config asks for ~355 ms of two-core work per 1000 ms — a **~35 % duty cycle**.
Spread over Tau (tens of seconds), that is nowhere near PL1. So:

> **Bursty and smooth schedules are thermally identical at this duty cycle.** Do not restructure
> the pipeline for throttling reasons.

What *does* differ is **scheduling jitter**. Under the current config, `DECIDE_EVERY_SECONDS`
(0.5) and `WM_CHUNK_SECONDS` (0.5) are equal, so the ECAPA call and a watermark call land on the
**same tick** — a ~355 ms window in which both cores are saturated and the audio callback has
nowhere to run. Fixing that is nearly free:

- Keep both at 1.0 s but **phase-offset them by 0.5 s**, so the two ~200 ms bursts never overlap.
  Peak instantaneous demand halves; total work is unchanged.
- Combined with the `out_queue` prefill (§6.4), the longest stall the output path can see drops
  from ~355 ms to ~220 ms against 400 ms of buffer.

One genuine thermal caveat: a laptop that has been running the demo (plus a browser, plus screen
share) for 20 minutes *will* be at PL1 and near base clock. That is why the ×2.6 "soaked" column
exists in §1.3, and why the budget targets ≥ 30 % headroom rather than ≥ 10 %.

---

## 9. Two-process reality and the recommended low-power configuration

### 9.1 Recommended config for the i5-7200U

| knob | current | recommended (i5-7200U) | why |
|---|---|---|---|
| `torch.set_num_threads` | *effectively 1* (bug) | **2**, set after `VoiceActivity()` | §2 — 6x |
| `OMP_WAIT_POLICY` | unset | **`PASSIVE`** | §3 — frees ~9 % of a core, free |
| `DECIDE_EVERY_SECONDS` | 0.5 | **1.0** | halves the dominant cost |
| `WINDOW_SECONDS` | 2.5 | **2.5 (unchanged)** | research floor is 2 s; do not go below |
| `WM_CHUNK_SECONDS` | 0.5 | **1.0** | §4.6; +0.5 s latency |
| watermark/decide phase | aligned | **offset by 0.5 s** | §8 — halves peak burst |
| VAD backend | TorchScript | **`load_silero_vad(onnx=True)`** | §5.1 — 4x |
| VAD subsampling | 15 frames per 0.5 s hop | **15 frames per 1.0 s hop** (follows decide) | halves an already-small cost |
| `DETECT_WINDOW_SECONDS` | 1.5 | **1.5 (unchanged)** | needed for detection reliability |
| `DETECT_HOP_SECONDS` | 0.75 | **1.0** | badge staleness is 2.5 s; hop only sets granularity |
| receiver `set_num_threads` | default | **1** | stay out of the sender's way |
| `--out-rate` | 16000 | **48000** | §6.5 — VB-CABLE cannot do 16 k |
| `out_queue` prefill | none | **0.4 s of silence** | §6.4 — zero perceptual cost |
| process priority | normal | **ABOVE_NORMAL** (both processes) | §6.1 |
| power plan | whatever | **High performance, on AC** | §6.6 |

End-to-end latency under this config: 1.0 s watermark chunk + 1.5 s gate + ~0.1 s device buffers
≈ **2.6 s** (up from 2.1 s). Acceptable; call it out in the demo rather than being surprised.

### 9.2 CPU budget on the i5-7200U

All figures are *percent of the whole 2-core machine*, using the ×2.3-scaled predictions from
§1.3. Torch ops run at 2 threads, so wall time ≈ machine occupancy.

**Sender**

| item | cost per interval | interval | % of machine |
|---|---|---|---|
| ECAPA, 2.5 s window, 2 threads | 220 ms | 1.0 s | **22.0 %** |
| AudioSeal embed, 1.0 s chunk, 2 threads | 135 ms | 1.0 s | **13.5 %** |
| silero-VAD ONNX, 1.0 s hop | 7 ms | 1.0 s | **0.7 %** |
| soxr 16 k → 48 k | ~0.7 ms | 1.0 s | < 0.1 % |
| Python glue, numpy copies, PortAudio callbacks | — | — | ~3 % |
| OpenMP spin | 0 (with `PASSIVE`) | — | **0 %** |
| **sender total** | | | **≈ 39 %** |

**Receiver** (same machine, `set_num_threads(1)`)

| item | cost per interval | interval | % of machine |
|---|---|---|---|
| AudioSeal detect, 1.5 s window, 1 thread | 261 ms | 1.0 s | 26 % of one core = **13 %** |
| SSE server, HTTP, Ed25519 signing, Python | — | — | ~3 % |
| **receiver total** | | | **≈ 16 %** |

**Totals**

| scenario | total | headroom |
|---|---|---|
| sender only | 39 % | 61 % |
| sender + receiver | 55 % | **45 %** |
| sender + receiver + badge browser (~7 %) | 62 % | **38 %** ✅ |
| + Zoom, audio only (~20 %) | 82 % | 18 % ⚠️ |
| + Zoom **with video** (40–70 % on a dual core) | **> 100 %** | ❌ |

Thermally soaked (×2.6), sender + receiver + browser rises to ~70 %, still 30 % headroom.

Zoom's own guidance is a dual-core 2 GHz minimum but a **quad-core recommendation** for screen
share, and dual-core machines are documented to fall to ~5 fps when sharing
([Zoom system requirements](https://www.imss.caltech.edu/services/voice-mobile-conference-calls/zoom/system-requirements),
[Caltech IMSS](https://www.imss.caltech.edu/services/voice-mobile-conference-calls/zoom/system-requirements)).
That headroom is being spent whether TrueMic wants it or not.

### 9.3 What to sacrifice, in order

1. **Move the receiver + badge browser to the second laptop.** Frees ~23 %. The demo already
   requires a second device for the air-gap beat, and `profile/` is already documented as copyable.
   This is the single best structural decision and costs nothing.
2. **`DECIDE_EVERY_SECONDS` 0.5 → 1.0.** Frees ~22 %. The 2.5 s window is unchanged, so this
   changes *how often* you re-decide, not *what* you decide on — the research constraint is on
   window length, not decision rate.
3. **`WM_CHUNK_SECONDS` 0.5 → 1.0.** Frees ~3 % and halves the number of parallel-region entries.
4. **VAD to ONNX.** Frees ~2 %.
5. **`DETECT_HOP_SECONDS` 0.75 → 1.0 (or 1.25).** Frees 4–8 % if the receiver stays local.
6. **Zoom video off** (audio-only call, or run the far end on a phone).
7. **`--engine eagle`.** Picovoice Eagle is a native C engine explicitly targeted at CPU-only and
   embedded deployment
   ([product page](https://picovoice.ai/products/voice/speaker-recognition/)) and would replace the
   22 % ECAPA line with something close to zero. **But it validates its AccessKey over the
   internet**, which the project's own HANDOFF already flags as a pre-demo checklist item. Treat
   Eagle as an *upside*, never as the plan: budget for ECAPA, and if Eagle authenticates on the
   venue's network, enjoy the 22 %.
8. **Last resort: `WINDOW_SECONDS` 2.5 → 2.0.** Frees ~5 %. Contradicts the project's own
   non-negotiable ("1 s windows are unreliable", 2–3 s required) and 2.0 s is the edge of that
   range. Do not go below 2.0 under any circumstances.

Deliberately **not** on this list: `--no-watermark` (it is the demo), ONNX-ifying ECAPA (§5.3),
the AudioSeal streaming refactor (§4.7), WASAPI exclusive mode (§6.3).

---

## 10. Caveats and what was not verified

- **No i5-7200U was tested.** Every i5 figure is a ×2.3 (or ×2.6) scaling of a Ryzen 5 7600
  measurement, justified by three independent methods that agree within 15 % (§1.1). Re-measure on
  the actual laptop before demo day — `truemic.sender` already prints `wm_load`, which is a direct
  read of one budget line.
- **The dev box runs `torch 2.9.1+rocm7.2.1`.** The demo laptop will have a stock CPU wheel. Both
  use oneDNN and OpenMP for CPU convolution, so the *direction* of every finding holds, but
  absolute times could shift ±20 %. The two bugs in §2 and §3 are build-independent.
- **`set_flush_denormal` showed no measurable effect** on random-noise input; it may still matter
  on real speech with long silences (§4.3). Keep it; do not budget for it.
- **The AudioSeal detector FLOP count is wrong** (§7.4). Detector budgeting uses measured wall time.
- **ONNX export of ECAPA and AudioSeal was not achieved** (§5.3) — blocked by a SpeechBrain 1.1
  lazy-import interaction with `torch.onnx.export`. The int8-on-AVX2 upside remains unquantified.
- **Zoom's CPU cost is quoted from vendor guidance, not measured** on a 7200U. If the demo depends
  on Zoom running locally, measure it first — it is the single largest uncertainty in §9.2.
- **`psutil` is not in `requirements.txt`** even though §6.1 uses it (it is installed on the dev
  box). Add it if you take the priority recommendation.
