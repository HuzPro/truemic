# AudioSeal internals, CallShield protocol, and watermark survival through real call stacks

Deep-dive research for TrueMic. Every substantive claim carries a URL. Where a claim comes from
running code on this machine rather than a source, it is labelled **[measured]** and the script is
`research/deep/` companion `exp.py` (reproduced in Appendix A).

Scope of the current TrueMic implementation this is written against:
`truemic/watermark.py` (base 16-bit generator, per-0.5 s `get_watermark`, manual `wav + watermark`),
`truemic/tokens.py` (16-bit HMAC token, Hamming ≤ 1 tolerance, ±1 slot),
`truemic/config.py` (`WM_CHUNK_SECONDS=0.5`, `TOKEN_SLOT_SECONDS=4.0`, `DETECT_WINDOW_SECONDS=1.5`,
`DETECT_HOP_SECONDS=0.75`, `WM_DETECT_THRESHOLD=0.5`), `truemic/receiver.py`.

---

## 1. AudioSeal architecture and what the paper actually claims

Paper: *Proactive Detection of Voice Cloning with Localized Watermarking*, San Roman, Fernandez,
Elsahar, Défossez, Furon, Tran — ICML 2024. <https://arxiv.org/abs/2401.17264>
Code: <https://github.com/facebookresearch/audioseal> (MIT for code **and weights** since 2024-04-02,
per the repo README <https://github.com/facebookresearch/audioseal/blob/main/README.md>).

### 1.1 Generator

EnCodec/SEANet-derived encoder–decoder operating directly on the waveform. The encoder is
"convolutional blocks and an LSTM"; the decoder mirrors it with transposed convolutions
(<https://arxiv.org/abs/2401.17264>, Fig. 4). The generator emits a **residual** `δ ∈ ℝ^T`, and the
watermarked signal is `s_w = s + δ`. In code, `AudioSealWM.forward` returns `x + alpha * wm`
(<https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/models.py>).

Concrete hyper-parameters from the shipped model card
(<https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/cards/audioseal_wm_16bits.yaml>):

| field | value |
|---|---|
| `dimension` | 128 |
| `n_filters` | 32 |
| `ratios` | `[8, 5, 4, 2]` → **total stride 320** (20 ms latent frame at 16 kHz) |
| `lstm` | 2 layers |
| `norm` | `weight_norm` |
| `causal` | **false** for `audioseal_wm_16bits`; **true** for `audioseal_wm_streaming` |
| `nbits` | 16 |

Two operational consequences of stride 320 that matter for TrueMic:

* The natural quantum of audio for this model is **320 samples = 20 ms**. Chunk sizes that are
  multiples of 320 avoid ragged frame counts. TrueMic's 0.5 s = 8000 samples = 25 × 320 ✔.
* The message is injected at the **latent** rate (50 Hz), not per sample.

### 1.2 How the 16-bit message is embedded

`MsgProcessor` (`src/audioseal/models.py`) is an `nn.Embedding(2*nbits, hidden_size)`. For bit *k*
with value *b*, index `2k + b` is looked up; the 16 resulting vectors are **summed** and the single
resulting `h`-dim vector is **broadcast-added to every latent frame**:

```python
indices = (2 * torch.arange(msg.shape[-1]) + msg).long()   # b x k
msg_aux = self.msg_processor(indices).sum(dim=-2)          # b x h
hidden = hidden + msg_aux.unsqueeze(-1).repeat(1, 1, hidden.shape[2])
```

So the payload is a **constant DC offset in latent space, repeated at 50 Hz for the whole chunk**.
This is the single most important structural fact for TrueMic: the message is *repetition-coded* by
construction. Every 20 ms frame carries the full 16 bits. That is why a short detection window can
still recover the payload, and why continuously re-embedding the same token (what TrueMic does) is
much more robust than the "embed once over a whole file, then crop" scenario most third-party
robustness tables measure.

### 1.3 Detector

`AudioSealDetector` = `SEANetEncoderKeepDimension` + a `Conv1d(output_dim, 2 + nbits, 1)` head, i.e.
**one `2 + 16 = 18`-vector per input sample**. Two heads:

```python
result = self.detector(x)                       # b x 18 x T
result[:, :2, :] = softmax(result[:, :2, :], 1)  # presence
message = sigmoid(result[:, 2:, :].mean(dim=-1)) # payload
```

(<https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/models.py>)

Then `detect_watermark` computes

```python
detect_prob = count_nonzero(result[:, 1, :] > detection_threshold) / T
message     = message > message_threshold
```

Read that carefully — it changes how TrueMic should interpret its own numbers:

* **`detection_threshold` (default 0.5) is a *per-sample* threshold.** The returned "probability" is
  the **fraction of samples** whose presence logit exceeds it. It is a coverage statistic, not a
  confidence. A window that is half watermarked and half not returns ≈ 0.5.
* **`message_threshold` (default 0.5) is applied to bit probabilities**, and the bit probabilities
  come from a **mean over the whole window** followed by a sigmoid. Two different payloads inside one
  window are *averaged in logit space*, not resolved. This is exactly the token-slot straddle case in
  TrueMic's receiver (see §7.3).
* `decode_message` returns the **soft** bit probabilities. TrueMic currently throws that away with
  `(message >= 0.5)`. The distance `|p − 0.5|` per bit is free per-bit confidence and is the raw
  material for soft-decision voting (§8.4).

### 1.4 The TF-loudness perceptual loss

The signal is split into `B` non-overlapping frequency bands and windowed in time; for each
time–frequency cell the loudness difference `l_bw = Loudness(δ_bw) − Loudness(s_bw)` is computed
under ITU-R BS.1770-4, and the loss is `Σ softmax(l)_bw · l_bw` — a soft-max weighting so the
optimiser spends its budget on the cells where the watermark is *most* audible rather than
over-hiding cells that are already inaudible (<https://arxiv.org/abs/2401.17264>, §3.4).

Practical read-through: the watermark's energy is *shaped to the signal*. Silence gets almost no
watermark. **Watermarking non-speech (silence, VAD-rejected audio) is wasted compute and, per
issue #52, can even introduce audible "beeping" in silent regions**
(<https://github.com/facebookresearch/audioseal/issues/52>).

### 1.5 Detection reliability vs window length; minimum duration for 16-bit decode

The paper does not publish a duration/bit-accuracy curve. The repo's own
`examples/Getting_started.ipynb` does, in stored cell outputs, using the **streaming** pair
(<https://github.com/facebookresearch/audioseal/blob/main/examples/Getting_started.ipynb>):

| decoded chunk | duration @16 kHz | detection score | 16-bit message accuracy |
|---:|---:|---:|---:|
| 100 samples | 6.3 ms | 0.870 | 0.625 |
| 1 000 samples | **62.5 ms** | 0.993 | **1.000** |
| 16 000 samples | 1.0 s | 1.000 | 1.000 |
| 48 000 samples | 3.0 s | 1.000 | 1.000 |
| 64 000 samples | 4.0 s | 1.000 | 1.000 |

On clean audio the streaming model needs roughly **60–100 ms** for an exact 16-bit decode, not
seconds. This is the headline result for TrueMic's token channel: the 1.5 s detection window is
~20× longer than the clean-audio minimum, and that surplus is available to spend on codec/noise
margin or on faster token rotation — not something to be conservative about.

For contrast, the same notebook's non-streaming section reports a healthy but non-saturated bit
posterior on a full clip: `message_prob` ≈ `[0.23, 0.23, 0.22, 0.80, 0.78, 0.21, 0.18, 0.24, 0.30,
0.80, 0.22, 0.79, 0.79, 0.76, 0.26, 0.78]`. Margins of ~0.28 from the 0.5 decision line, not 0.49.
Bits are recoverable but not saturated — degradation eats that margin quickly.

Independent practitioner data (provcheck, 2026-07-01) puts the *whole-file-embed-then-crop* floor
much higher: "trim to ≥ 5 s window ✓; 3–5 s ~; < 3 s ✗"
(<https://github.com/CreativeMayhemLtd/provcheck/blob/main/docs/audio-watermark-survival-range.md>).
That table is not in conflict with the notebook — it measures a *different* scenario (one payload
spread over a long file, then cropped, decoded once). TrueMic re-embeds the same token continuously,
which is the notebook's regime.

### 1.6 Detection speed claim

"detection up to two orders of magnitude faster… notably 485× faster when there is no watermark"
(<https://github.com/facebookresearch/audioseal/blob/main/README.md>,
<https://arxiv.org/abs/2401.17264>) — because detection is a single forward pass with no
synchronisation search, unlike WavMark's brute-force alignment.

---

## 2. The v0.2 streaming API

`0.2.0` (2024-12-09 per <https://github.com/facebookresearch/audioseal/blob/main/CHANGELOG.md>)
added:

* new checkpoints `audioseal_wm_streaming` / `audioseal_detector_streaming`, **trained in causal
  mode** (maintainer confirmation: <https://github.com/facebookresearch/audioseal/issues/96>);
* a refactor for TorchScript compatibility;
* **deprecation of internal resampling** — `sample_rate=` is now a no-op and the caller must supply
  16 kHz (see `SAMPLE_RATE_WARN` in `models.py`).

### 2.1 Exact usage

```python
model = AudioSeal.load_generator("audioseal_wm_streaming").eval()

with model.streaming(batch_size=1):            # enables the conv cache
    for chunk in chunks:                        # (B, C, T) tensors
        wm_chunk = model(chunk, message=msg, alpha=1.0)
```

`AudioSealWM.streaming` is a `@contextmanager` that simply forwards to `self.encoder.streaming()`
and raises `NotImplementedError` on a non-streaming checkpoint or Python < 3.10
(<https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/models.py>). The detector has
**no** `streaming()` method — `audioseal_detector_streaming` is just a causal checkpoint you feed a
buffer to. `einops` is required; Python ≥ 3.10 selects the Moshi SEANet implementation instead of the
AudioCraft one (`src/audioseal/builder.py`).

### 2.2 What the cache actually does

`RawStreamingConv1d.forward` keeps the tail of the previous input and prepends it to the next chunk,
emitting only fully-formed frames
(<https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/libs/moshi/modules/streaming.py>):

```python
previous = self._streaming_state.previous
if previous is not None:
    input = torch.cat([previous, input], dim=-1)
num_frames = max(0, int(math.floor((T - kernel) / stride) + 1))
self._streaming_state.previous = input[..., num_frames * stride:]
```

`RawStreamingConvTranspose1d` keeps the analogous `partial` overlap-add tail. Net effect: the chunked
stream is **numerically the continuation of one long causal pass**, not a sequence of independent
passes. That is precisely the defect in TrueMic's current `Watermarker.embed`.

### 2.3 Does it beat chunked embedding? Yes, and the failure mode is audible

An independent production port measured the chunk-boundary artefact of naive chunked embedding
directly: "chunking the generator produces audible discontinuities at chunk boundaries (per L∞ test
in survey: **boundary 0.044 vs interior 0.0045, 10× gap**). Detector chunking is safe (boundary
0.001 vs interior 0.00002, both tiny)."
(<https://github.com/CreativeMayhemLtd/provcheck/blob/main/scripts/export-audioseal.py>)

So: **embedding must be continuous; detecting may be chunked.** TrueMic currently does the opposite
of the safe combination on the embed side.

Two further points:

* Streaming conv emits only complete frames, so `wm.shape[-1]` can be **shorter than the input chunk**
  on the first call while the cache warms. `get_watermark` does `self.decoder(hidden)[..., :length]`,
  so a short return is possible and `x + alpha*wm` would then be a shape error. Guard by trimming the
  input to `wm.shape[-1]` (what §7.1's snippet does) — or by discarding the first chunk.
* Keep chunk sizes a multiple of 320 samples so frame counts stay integral.

---

## 3. Robustness through real call stacks — the biggest demo risk

### 3.1 What the AudioSeal paper reports (Table 3)

Accuracy / TPR / FPR, threshold **tuned per attack** to maximise accuracy
(<https://arxiv.org/abs/2401.17264> §5; confirmed in issue #89 discussion,
<https://github.com/facebookresearch/audioseal/issues/89>):

| edit (eval parameters) | AudioSeal acc / TPR / FPR | AUC |
|---|---|---|
| none | 1.00 / 1.00 / 0.00 | 1.00 |
| bandpass 500–5000 Hz | 1.00 / 1.00 / 0.00 | 1.00 |
| **highpass 1500 Hz** | **0.61 / 0.82 / 0.60** | 0.61 |
| lowpass 500 Hz | 0.99 / 0.99 / 0.00 | 0.99 |
| boost ×10 | 1.00 / 1.00 / 0.00 | 1.00 |
| duck ×0.1 | 1.00 / 1.00 / 0.00 | 1.00 |
| echo 0.5 s / 0.5 gain | 1.00 / 1.00 / 0.00 | 1.00 |
| pink noise σ=0.1 | 1.00 / 1.00 / 0.00 | 1.00 |
| white noise σ=0.05 | 0.91 / 0.86 / 0.04 | 0.95 |
| speed 1.25× | 0.99 / 0.99 / 0.00 | 1.00 |
| smoothing (window 40) | 0.99 / 0.99 / 0.00 | 1.00 |
| **resample via 32 kHz** | **1.00 / 1.00 / 0.00** | 1.00 |
| AAC 64 kbps | 1.00 / 1.00 / 0.00 | 1.00 |
| MP3 32 kbps | 1.00 / 1.00 / 0.00 | 1.00 |
| EnCodec 24 kHz, nq=16 | 0.98 / 0.98 / 0.01 | 1.00 |
| **average** | 0.96 / 0.98 / 0.04 | 0.97 |

Three cautions before anyone quotes this table at a judge:

1. **It is detection only.** The paper publishes no bit-accuracy-versus-edit curve. The closest proxy
   is Table 4 (attribution): 1 000 clips of 5 s, 100 random 16-bit messages, flagged at FPR 10⁻³,
   then matched by minimum Hamming distance — **averaged over the edits**. With a *single* candidate
   message, AudioSeal attributes correctly only **68.2 %** of the time (WavMark 58.4 %), falling to
   56.4 % against 10 000 candidates. TrueMic's badge depends on the **message**, and the message is
   dramatically weaker than the presence bit.
   Upstream's own CI encodes the same expectation: `tests/test_models.py::test_detector` watermarks a
   clip, decodes it whole, and asserts only that **more than 20 of 32 bits** match (2 × 16-bit batch),
   with the comment "Due to non-deterministic decoding, messages are not always the same as message"
   (<https://github.com/facebookresearch/audioseal/blob/main/tests/test_models.py>). The maintainers
   do not consider an exact 16-bit round-trip guaranteed on the base checkpoint.
2. **Thresholds were tuned per attack.** Real deployment has one fixed threshold.
3. **Highpass at 1500 Hz breaks it** (FPR 0.60 — the detector is essentially guessing). The watermark
   energy therefore lives substantially **below 1.5 kHz**, which is exactly the band that
   speech codecs model *parametrically* rather than waveform-accurately.

### 3.2 Independent benchmarks say Opus kills it

**AudioMarkBench** (NeurIPS 2024 Datasets & Benchmarks,
<https://arxiv.org/abs/2406.06979>, <https://github.com/moyangkuo/AudioMarkBench>), 5-second clips at
16 kHz, thresholds chosen for FPR < 0.01 and FNR < 0.01 (`AudioSeal τ=0.15`):

> "EnCodec, SoundStream, and Opus achieve very high FNRs, indicating that those perturbations can
> remove watermarks from watermarked audios while preserving the audio quality."

and the general conclusion: "current audio watermarking methods are not robust to no-box removal
perturbations that are unseen during adversarial training." AudioSeal *is* reported robust to time
stretch, low-pass, high-pass and echo in that benchmark.

**StreamMark** (<https://arxiv.org/abs/2604.11917>), Table 1, Test Set A:

| method | SNR (dB) | PESQ | crop 70 % | MP3 8 kbps | **Opus 60 ms** |
|---|---:|---:|---:|---:|---:|
| AudioSeal | 25.41 | 4.30 | 100 % | 85 % | **57 %** |
| Timbre | 24.14 | 3.70 | 99 % | 79 % | 99 % |
| StreamMark | 24.16 | 4.20 | 99 % | 87 % | 99 % |

57 % is near coin-flip. StreamMark's stated motivation is that Opus "is prevalent in real-time
communication platforms like WebRTC" and that AudioSeal does not survive it.

**Practitioner range table** (provcheck, 2026-07-01, AudioSeal run at **alpha = 3.0**,
<https://github.com/CreativeMayhemLtd/provcheck/blob/main/docs/audio-watermark-survival-range.md>):

| condition | AudioSeal @ alpha 3.0 |
|---|---|
| MP3 ≥ 128 kbps | ✓ |
| AAC 192 kbps | ✓ · AAC 128 kbps ~ · AAC 96 kbps ~ |
| Opus 128 kbps | ✓ · Opus 96 kbps ✓ · **Opus 64 kbps ~** |
| **Speech codecs (Speex, AMR, GSM)** | **✗** |
| loudness ±12 dB | ✓ |
| tempo ±5 % | ✓ · ±10 % ~ |
| pitch ±semitone | ~ |
| small-room reverb | ✓ · concert-hall ~ |
| trim ≥ 5 s | ✓ · 3–5 s ~ · < 3 s ✗ |
| **speaker → mic re-recording, clean room** | **~ (degraded)** |
| **speaker → mic re-recording, noisy** | **✗** |

The same repo's codec-survival investigation gives the alpha lever explicitly (15 s clip):

| alpha | pristine | AAC 192 k | MP3 192 k | payload recovery |
|---:|---:|---:|---:|---|
| **1.0** (upstream default) | **0.671** | **0.369** | 0.689 | borderline; below their 0.70 threshold |
| 3.0 | 0.999 | 0.999 | 0.999 | clean on PCM/MP3, **one bit-flip through AAC** |
| 5.0 | 0.998 | 0.999 | 0.999 | clean on all three |

(<https://github.com/CreativeMayhemLtd/provcheck/blob/main/docs/v0.5.2-codec-survival/README.md>) —
they shipped `alpha = 3.0` as their default and recommend 5.0 where exact payload recovery matters.
Their proposed fix for the residual bit-flip is exactly what §8.4 recommends for TrueMic:
"add a brand-ID consistency check that **votes across multiple decoded payloads from the same
stream**."

### 3.3 Will it survive Zoom / Teams / Meet?

The call app is not one attack, it is a stack. In order:

1. **Windows APO / "audio enhancements"** on the endpoint (EQ, loudness equalisation, voice
   clarity), implemented as in-process COM Audio Processing Objects
   (<https://www.ninjaone.com/blog/enable-or-disable-audio-enhancements-in-windows-11/>). AudioSeal
   issue #50 is a direct, reproduced report: the same watermarked file recorded through OBS on
   Windows 11 with enhancements **on** gave score **0.20** and a wrong message; with enhancements
   **off**, score rose to **0.90** but the message was **still wrong**
   (<https://github.com/facebookresearch/audioseal/issues/50>). Maintainer response: "I don't see any
   way of solving this without fine-tuning the model with these augmentations."
2. **The client's own APM** — high-pass, AEC, noise suppression, AGC. Zoom's "Original Sound"
   bypasses "all noise suppression and echo cancellation"; the "Original Sound for Musicians" /
   high-fidelity music mode additionally raises the codec to 48 kHz and 96 kbps mono / 192 kbps
   stereo and disables post-processing
   (<https://support.zoom.us/hc/en-us/articles/115003279466-Preserve-original-sound>,
   <https://theglobalconservatory.com/pages/zoom-audio-settings-for-high-quality-music-lessons>).
3. **The codec.** Google Meet negotiates Opus with DTX, in-band FEC and RED
   (<http://www.rtcbits.com/2022/06/webrtc-google-meet.html>); typical conferencing speech runs
   **24–64 kbps** (<https://opus-codec.org/>). Microsoft Teams uses **Satin**, a proprietary
   AI/parametric low-bitrate codec, in 2-way calls
   (<https://techcommunity.microsoft.com/blog/microsoftteamsblog/satin-microsoft%E2%80%99s-latest-ai-powered-audio-codec-for-real-time-communications/2141382>,
   <https://bloggeek.me/lyra-satin-webrtc-voice-codecs/>) — a *generative* codec, i.e. it
   resynthesises speech rather than coding the waveform, which is the worst possible channel for an
   additive residual watermark.

**Verdict: do not build the demo on the assumption that the badge survives a real Zoom/Teams/Meet
call.** The evidence is convergent and negative: AudioMarkBench says Opus removes it, StreamMark
measures 57 %, provcheck marks Opus 64 kbps as degraded even at alpha 3.0, the paper's own numbers
show the watermark lives in the band speech codecs treat parametrically, and Teams doesn't even use
a waveform codec. CallShield's authors reached the same conclusion and rebuilt the watermarker at
8 kHz with 40 ms causal frames rather than trying to push AudioSeal through a phone channel
(<https://arxiv.org/abs/2601.09327>).

### 3.4 Mitigations, in order of cost/benefit for a one-day build

| mitigation | cost | effect |
|---|---|---|
| **Demo over the local loopback** (receiver on `CABLE Output`) | zero | removes the codec entirely; this is what `receiver.py --in "CABLE Output"` already does |
| **Raise `alpha` to 3.0** | one multiply | 0.671 → 0.999 pristine confidence in the only published sweep; the cheapest single robustness win available |
| **Switch to the streaming generator** | ~10 lines | continuous watermark, no boundary artefacts, better short-window decode |
| **Majority-vote the token across detection ticks** | ~15 lines | converts per-tick bit errors into a stable badge (§8.4) |
| **Zoom "Original Sound" / high-fidelity music mode** | a checkbox | disables NS/AEC/post-processing and raises the codec bitrate |
| **Disable Windows audio enhancements on both endpoints** | a checkbox | issue #50: 0.20 → 0.90 detection |
| Keep everything at 16 kHz end-to-end, resample with `soxr` only where forced | already done | the paper's resample row is 1.00; sample-rate mismatch is a bigger risk than resampling itself |
| Fine-tuning the model with codec augmentation | days | the only real fix; out of scope |

For the "air-gap" demo beat (phone speaker → laptop mic), the honest expectation is
**"~ degraded, clean room only"** per the provcheck table. Rehearse it; have the loopback demo as the
one you promise.

**If a codec-robust carrier is ever needed**, the published Opus-survivors are Timbre Watermarking
(frequency-domain, explicit repeated-embedding strategy, NDSS 2024, code + weights at
<https://github.com/TimbreWatermarking/TimbreWatermarking>, 99 % under Opus 60 ms in StreamMark's
Table 1) and StreamMark itself. Both are research repos, not `pip install` — a swap is a
post-hackathon item, and the architecture should keep the watermark behind the existing
`Watermarker`/`WatermarkReader` adapter pair so the swap stays a two-class change.

---

## 4. ONNX exportability

Two independent public exports exist. AudioSeal itself is ONNX-friendly *because* 0.2 refactored for
TorchScript: both `AudioSealWM` and `AudioSealDetector` implement `__prepare_scriptable__`, which
walks the module tree and strips the `weight_norm` forward-pre-hooks
(<https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/models.py>).

**(a) `thewh1teagle/audioseal-onnx`** — `onnxruntime`-only, no PyTorch at runtime, **INT8 dynamic
quantisation, ~34 MB of models**, claims "10 min detected in ~5 s on CPU" (≈120× real time) and
survival through "MP3 (320k–64k), AAC, resampling, volume changes, and trimming"
(<https://github.com/thewh1teagle/audioseal-onnx>). Its export recipe
(<https://github.com/thewh1teagle/audioseal-onnx/blob/main/scripts/export.py>):

```python
generator.normalizer = None
generator = generator.__prepare_scriptable__()   # strips weight_norm
detector  = detector.__prepare_scriptable__()
torch.onnx.export(..., opset_version=17, dynamo=False,
                  dynamic_axes={"audio": {0: "batch", 2: "frames"}, ...})
quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
```

Note it also sets `os.environ["TORCHDYNAMO_DISABLE"]="1"` and wraps the generator so the
`sample_rate` argument does not get traced.

**(b) `CreativeMayhemLtd/provcheck`** — exports for the Rust `tract` runtime and documents the
pitfalls precisely
(<https://github.com/CreativeMayhemLtd/provcheck/blob/main/scripts/export-audioseal.py>):

* **In-place slice assignment must be inlined.** `AudioSealDetector.forward` does
  `result[:, :2, :] = softmax(...)`; their wrapper computes `presence` and `message` as separate
  tensors instead.
* **The `NormalizationProcessor` must be skipped** — "it's None on the 16-bit config and uses
  fold/unfold ops tract doesn't handle anyway." (Confirmed: `normalizer` defaults to `False` in
  `AudioSealWMConfig`/`AudioSealDetectorConfig` and **no shipped card sets it**, so the released
  checkpoints run without it. It is also guarded by `not torch.jit.is_scripting()` in `models.py`,
  so scripted/exported graphs would silently drop it anyway.)
* **Dynamic length is a problem for `tract`, not for ONNX Runtime.** tract 0.21 cannot resolve the
  symbolic Pad expression the exporter emits for variable-length SEANet input
  ("Undetermined symbol in expression: 16003 + -1*<Sym0>"), so they bake a fixed 10 s input.
  `thewh1teagle`'s ORT build uses `dynamic_axes` successfully — **if you export for onnxruntime,
  dynamic frames work**.
* The message input must be `int64` for the `nn.Embedding` lookup.

**Expected speedup on a weak 2-core laptop.** No published AudioSeal-specific PyTorch-vs-ORT number
exists. Realistic expectation: ORT FP32 gives roughly 1.5–2.5× over eager PyTorch on CPU for a
conv-dominated graph (less Python/dispatch overhead, better fusion); INT8 dynamic quantisation adds
up to ~2–3× more on AVX512-VNNI hardware, up to a documented 6× for FP32→INT8 on ORT
(<https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7>).
Caveat: **ONNX LSTM/GRU ops are not quantisable**
(<https://github.com/microsoft/onnxruntime/issues/5747>), and AudioSeal has 2 LSTM layers in both
encoder and decoder, so INT8 only helps the conv stack — expect the low end of that range.

**Cheaper than ONNX: TorchScript.** Upstream CI already proves both models script cleanly and that a
scripted generator/detector round-trips a watermark
(<https://github.com/facebookresearch/audioseal/blob/main/tests/test_models.py>, `test_jit`):

```python
scripted = torch.jit.script(AudioSeal.load_generator("audioseal_wm_16bits").eval())
wm_audio = scripted(audio, alpha=0.8)
```

No export pipeline, no second runtime, no INT8 accuracy question — and `__prepare_scriptable__`
strips `weight_norm` automatically during scripting. Note the scripted graph **drops the
`NormalizationProcessor`** (both call sites are guarded by `not torch.jit.is_scripting()`), which is
a no-op for the shipped checkpoints since they don't have one anyway. Whether `torch.jit.script`
also works on the *streaming* checkpoints is untested upstream — the streaming path uses
`@contextmanager` and dataclass state, so verify before relying on it.

**Recommendation for a one-day build:** ONNX is a *fallback*, not a plan. Do it only if the measured
embed load (§7.4) exceeds budget, and try `torch.jit.script` and `torch.set_num_threads(1..2)` first.
Context for "is it slow?": issue #73 reports 14–18 s to watermark a 4–5 s clip on an unspecified CPU
setup with no maintainer reply (<https://github.com/facebookresearch/audioseal/issues/73>), so the
`wm_load` figure `sender.py` already prints is the right thing to watch on the demo machine.

---

## 5. CallShield — what is worth copying

*CallShield: Secure Caller Authentication over Real-Time Audio Channels*, arXiv 2601.09327
(<https://arxiv.org/abs/2601.09327>, <https://arxiv.org/html/2601.09327v1>).

### 5.1 The system

Audio-layer caller authentication with **no transcription, no internet, no trusted infrastructure**;
pairwise pre-shared symmetric keys among trusted contacts. Reported 99.2 % authentication success on
clean audio, > 95 % under common distortions, 0 % false acceptance without the key, and an average
**63 s** to authenticate including up to three retransmissions.

### 5.2 Watermark layer (RTAW)

Adapted from AudioSeal's generator/detector but rebuilt for the channel:

* **8 kHz narrowband**, **40 ms frames (T = 320 samples)**, **1 bit per frame → 25 bps**.
* Causal: 1-D causal conv, 2 residual blocks, depthwise-separable convs, 1 LSTM in the encoder;
  mirrored transposed-conv decoder; **attention-based** watermark embedding module conditioned on the
  speech (rather than AudioSeal's broadcast-add).
* **`alpha ∈ [0.6, 1.0]`, default 0.6.** PESQ > 4.2, STOI > 0.94 at α = 0.6 (4.44 → 4.25 PESQ going
  from α = 0.6 to α = 1.0).
* Bit accuracy 98 % at α = 1.0, 92 % at α = 0.6 on clean speech; > 91 % across 80 % distortion
  coverage. Full distortion table (α = 0.6 / α = 1.0): white noise 80 % 0.762/0.889, pink noise 80 %
  0.924/0.980, echo 80 % 0.912/0.971, lowpass 80 % 0.879/0.961, highpass 80 % 0.906/0.978, bandpass
  80 % 0.835/0.930, ducking 80 % 0.917/0.981.
* Latency: embed ≈ 12.06 ms, decode ≈ 4.23 ms per bit; end-to-end budget < 200 ms to avoid audible
  echo.
* Explicit reason they did not use AudioSeal directly: it "operates on 1 second 16 kHz clips and
  requires offline processing"; existing methods "typically require access to the entire or a large
  segment of audio stream (≥ 1 second)". **Note this criticism predates AudioSeal 0.2's causal
  streaming checkpoints.**

### 5.3 Protocol and framing

Four phases, each frame = **15-bit sync preamble + BCH-coded payload**:

| phase | payload | ECC | coded bits | frame |
|---|---|---|---|---|
| Start | start beacon, e.g. `10110010` (8 b) | BCH m=5, t=5 | 31 | ≈ 46 b |
| Challenge | 64-bit nonce `N` ‖ 64-bit session id `S` | BCH m=9, t=55 | 511 | ≈ 526 b |
| Response | `MAC = Truncate₁₂₈(HMAC-SHA256(K, N̂‖Ŝ))` | BCH m=9, t=55 | 511 | ≈ 526 b |
| Finish | confirmation beacon | BCH m=5, t=5 | 31 | ≈ 46 b |

* **Sync preamble:** 15 candidate patterns evaluated; the winner is `[0,1,0,1,0] × 3` — 100 % sync
  detection on clean audio, 99 % under distortion. High-transition-density patterns beat uniform
  runs. Detection is a **sliding-window search** for the fixed pattern.
* **Frame synchrony:** strictly causal components, T = 320 samples per segment, real-time inference
  with no dependence on future context.
* **ARQ:** stop-and-wait, **selective** retransmission of only the failed segment (not a full
  restart), max 3 attempts. Two strategies compared: constant α, and **adaptive α that increments by
  0.1 after each failure up to 1.0**.
* **Replay:** "To prevent replay, CallShield uses a 64-bit challenge nonce embedded during each
  session. The MAC is computed over this fresh challenge, ensuring that responses from previous calls
  are invalid." They also admit the limitation: "our MAC-based authentication scheme relies on
  deterministic keyed hashes, which are vulnerable to replay under certain active adversaries if
  session uniqueness is not enforced."
* Stated future work: burst-error channel models (Gilbert–Elliott) instead of the assumed BSC;
  probabilistic primitives for forward security.

### 5.4 What TrueMic should and should not copy

**Copy:**

1. **Adaptive strength on failure.** CallShield's α-escalation is the same idea as provcheck's
   `alpha = 3.0` default, arrived at independently. TrueMic should expose `WM_ALPHA` and default it
   above 1.0.
2. **Vote instead of trusting one decode.** CallShield gets this from BCH; provcheck proposes it
   explicitly for AudioSeal payloads. TrueMic gets it nearly free (§8.4).
3. **The freshness argument for the pitch.** CallShield's replay defence is a per-session nonce.
   TrueMic's rotating HMAC slot + staleness rejection is the same primitive in a
   sender-broadcast (no back-channel) shape — a strictly weaker but strictly cheaper story, and one
   worth stating precisely rather than overclaiming.

**Do not copy:**

4. **BCH on the 16 bits.** CallShield needs ECC because its payload is *serialised over time* — 128
   bits at 25 bps is 5 s of one-shot channel with no repetition. AudioSeal's payload is
   **already repetition-coded at 50 Hz** (§1.2), so the same 16 bits arrive ~75 times in a 1.5 s
   window. Spending bits of a 16-bit payload on BCH parity would *shrink* the token space (e.g. BCH
   (15,7,t=2) leaves 7 bits ⇒ 128 tokens ⇒ trivially guessable) while adding nothing the detector's
   own logit-averaging doesn't already do. **Keep 16 raw bits; do the error correction across
   detection ticks, not inside the payload.**
5. **A handshake.** 63 s to authenticate is a non-starter for a 4-beat demo, and TrueMic has an
   out-of-band channel (the verifier endpoint over HTTPS) that CallShield deliberately does without.
6. **A sync preamble.** AudioSeal needs no alignment search — that is the headline architectural win
   over WavMark (§1.6). Don't reintroduce the problem.

---

## 6. Detector behaviour on watermark-free audio

* Repo notebook, clean unwatermarked speech: detection score **0.0036**
  (<https://github.com/facebookresearch/audioseal/blob/main/examples/Getting_started.ipynb>, cell 15).
* AudioMarkBench, 5 s clips: `AudioSeal τ = 0.15` gives FPR < 0.01 **and** FNR < 0.01; "both FPR and
  FNR of AudioSeal are almost 0" (<https://arxiv.org/abs/2406.06979>).
* Every public "false positives abound" report against AudioSeal has been **user error, not detector
  error**: issues #14, #17 and #24 were all people running the detector on audio they had never
  watermarked and expecting synthetic-speech detection. Maintainer: "watermarking methods are
  *active*. The detector will only detect the watermark in audios where you embed the watermark."
  (<https://github.com/facebookresearch/audioseal/issues/24>,
  <https://github.com/facebookresearch/audioseal/issues/14>)
* **The message is never a detector of anything on its own.** On unwatermarked audio
  `detect_watermark` still returns 16 bits — "the message will be a random tensor if the detector
  detects no watermarking" (docstring in `models.py`; README repeats it). TrueMic already gates on
  `probability >= WM_DETECT_THRESHOLD` before checking the token — that ordering is correct and must
  stay.
* Threshold practice: `detection_threshold` is per-sample; keep it at the trained default **0.5**.
  Tune the **aggregate** score threshold (TrueMic's `WM_DETECT_THRESHOLD`) instead, and AudioMarkBench's
  τ = 0.15 is the published operating point with FPR < 0.01. TrueMic's 0.5 is conservative; it
  demands that half the samples in the window be individually flagged, which is a coverage
  requirement, not a confidence requirement — and it is exactly what fails when a window straddles
  the boundary between gated-open and gated-muted audio.
* `message_threshold` should stay 0.5 — but prefer reading the **soft** probabilities from the
  low-level `detector(x)` call and keeping the margins (§8.4).

---

## 7. Findings specific to the current TrueMic code

### 7.1 `Watermarker.embed` uses the wrong checkpoint and the wrong call pattern

```python
self._model = AudioSeal.load_generator("audioseal_wm_16bits")   # non-causal
...
watermark = self._model.get_watermark(wav, sample_rate=..., message=message)  # fresh state each 0.5 s
```

Each 0.5 s chunk is a fully independent, **non-causal** forward pass: the LSTM state resets, the
convolution padding restarts, and the decoder's `trim_right_ratio` boundary lands mid-speech every
500 ms. That is precisely the configuration measured at a **10× L∞ discontinuity at chunk
boundaries** by provcheck (§2.3). Minimal fix:

```python
class Watermarker:
    def __init__(self, device="cpu", alpha=config.WM_ALPHA):
        self._model = AudioSeal.load_generator("audioseal_wm_streaming").to(device).eval()
        self._alpha = alpha
        self._ctx = self._model.streaming(batch_size=1)
        self._ctx.__enter__()          # or hold it open around the sender's main loop

    def embed(self, chunk, token):
        wav = torch.from_numpy(chunk.astype(np.float32))[None, None, :]
        msg = torch.tensor([_token_to_bits(token)], dtype=torch.int32)
        with torch.no_grad():
            wm = self._model.get_watermark(wav, message=msg)
        n = wm.shape[-1]                                    # may be < len(chunk) while cache warms
        return (wav[..., :n] + self._alpha * wm).squeeze().numpy()
```

and load `audioseal_detector_streaming` in `WatermarkReader` to match the causal checkpoint.
Drop `sample_rate=` — it is a deprecated no-op in 0.2 and logs a warning.

### 7.2 `alpha` is not exposed

`watermark.py` does `wav + watermark`, i.e. hard-codes `alpha = 1.0` — the value provcheck measured
at **0.671 pristine confidence** and 0.369 through AAC. There is no knob to trade imperceptibility
for robustness, which is the single highest-leverage runtime parameter this system has and belongs
in `config.py` next to the similarity threshold, for the same reason: it is a property of the room
and the channel, not of the code.

### 7.3 The token-slot straddle is real and quantifiable

`DETECT_WINDOW_SECONDS = 1.5` against `TOKEN_SLOT_SECONDS = 4.0`. A slot boundary falls inside the
window `1.5 / 4.0 = 37.5 %` of the time. Because `decode_message` **averages logits over the whole
window** (§1.3), a straddling window returns a blend of the two tokens. A 2:1 split still resolves to
the majority token, but near a 50/50 split roughly half the differing bits become coin flips — far
more than `TOKEN_MAX_BIT_ERRORS = 1` absorbs. Ballpark: windows within ±0.15 s of a 50/50 split
occur ~7.5 % of the time, i.e. roughly one detection tick in thirteen is structurally doomed
regardless of channel quality. Three independent fixes, cheapest first:

* raise `TOKEN_SLOT_SECONDS` (8 s halves the straddle rate);
* shorten `DETECT_WINDOW_SECONDS` (the notebook says 0.5 s is ~24× the clean-audio minimum, §1.5);
* majority-vote across ticks so a single bad tick cannot grey the badge (§8.4) —
  `BADGE_STALE_SECONDS = 2.5` with `DETECT_HOP_SECONDS = 0.75` already gives ~3 ticks of slack, so
  this is mostly already handled; the vote makes it explicit and lets the threshold tighten.

### 7.4 Watermarking silence is wasted

`sender.py` embeds every 0.5 s chunk unconditionally, including silence, while the VAD result is
already computed one branch away. Per §1.4 the perceptual loss shapes watermark energy to the signal,
so silent chunks get near-zero watermark anyway — the compute is spent for nothing, and issue #52
reports audible "beeping" when watermarking silence
(<https://github.com/facebookresearch/audioseal/issues/52>). Gating the embed on
`voiced_fraction > 0` would cut average embed load roughly in half in a normal conversation.

### 7.5 The detector's level sensitivity is unmanaged

`NormalizationProcessor.loudness_normalization` (target RMS 0.1, gain clamped to [1, 10]) exists in
`models.py` but is **disabled on every shipped checkpoint** (`normalizer` defaults to `False`;
no card sets it). So the released detector does **no** input-level normalisation. Quiet received
audio therefore hits the detector at whatever level the call app produced. A one-line RMS
normalisation of the receiver's window to ≈ 0.1 before `detect_watermark` costs nothing and removes a
whole class of "it worked on my machine" failure. (The paper's Table 3 shows detection is robust to
±10× gain *when the model sees it*, so this is about matching the training distribution, not about
clipping.)

---

## 8. Action items for TrueMic

Ordered by (impact × cheapness). Each is a concrete change.

1. **Switch both ends to the causal streaming checkpoints.** `audioseal_wm_streaming` +
   `audioseal_detector_streaming`, and hold `model.streaming(batch_size=1)` open across the sender's
   whole session instead of re-entering per chunk. Requires `pip install audioseal>=0.2` on Python
   ≥ 3.10 with `einops` (both already satisfied). Trim the input to `wm.shape[-1]` before adding.
   Fixes the 10× chunk-boundary discontinuity and matches the model the 62.5 ms decode number was
   measured on.
2. **Add `WM_ALPHA = 3.0` to `config.py`** and apply it as `wav + WM_ALPHA * watermark`. Expose it on
   the CLI (`--alpha`) and calibrate it on-site the same way as the similarity threshold. Published
   sweep: 1.0 → 0.671 pristine / 0.369 AAC; 3.0 → 0.999 / 0.999; 5.0 → clean payload everywhere.
   Listen once at 3.0 on the real mic; back off to 2.0 if audible.
3. **Drop `sample_rate=` from both `get_watermark` and `detect_watermark`.** It is a deprecated no-op
   in 0.2 that emits a warning on every call; the caller is now responsible for feeding 16 kHz.
4. **Majority-vote the token across detection ticks in `receiver.py`.** Keep a short deque of the
   last N ticks' *soft* bit probabilities from the low-level `detector(x)` call (which returns
   `sigmoid(mean logits)`, not thresholded bits), sum `p − 0.5` per bit across ticks, then threshold.
   Two or three ticks of soft accumulation converts an isolated 3-bit-error tick into a correct
   token, and lets `TOKEN_MAX_BIT_ERRORS` go back to 0 (tightening the forgery margin from
   ~0.08 % to ~0.005 %).
5. **Raise `TOKEN_SLOT_SECONDS` to 8.0 and drop `DETECT_WINDOW_SECONDS` to 1.0.** Straddle rate falls
   from 37.5 % to 12.5 %, and 1.0 s is still ~16× the clean-audio minimum decode length. Keep
   `TOKEN_TOLERANCE = 1` (now ±8 s of clock/pipeline slack, comfortably more than the 1.5 s gate
   delay plus detect window).
6. **Only watermark voiced chunks.** `sender.py` already computes `voiced_fraction`; skip
   `watermarker.embed` when the chunk is silent. Halves embed CPU in conversation and avoids the
   silence-beeping artefact.
7. **RMS-normalise the receiver's detection window to ≈ 0.1 before `detect_watermark`.** The shipped
   detector has its `NormalizationProcessor` disabled, so nothing else will do it.
8. **Lower `WM_DETECT_THRESHOLD` from 0.5 to 0.2.** AudioSeal's aggregate score is the *fraction of
   samples* flagged, not a confidence; AudioMarkBench's published FPR < 0.01 operating point is
   τ = 0.15. Keep `detection_threshold=0.5` (the per-sample argument) at its trained default. This is
   what stops the badge greying every time a window straddles a gate open/close transition.
9. **Demo over VB-CABLE loopback, not through a call app.** `receiver.py --in "CABLE Output"` already
   does this. If a call app must be in the shot, use Zoom with **Original Sound / high-fidelity music
   mode** enabled and Windows "audio enhancements" disabled on both endpoints, and expect Teams
   (Satin, a generative codec) to fail outright.
10. **Do not add BCH/Hamming ECC inside the 16 bits.** The payload is already repetition-coded at
    50 Hz by the architecture; parity bits would shrink the token space (BCH(15,7) leaves 128 tokens)
    for no gain. Do the error correction across ticks (item 4).
11. **Reserve ONNX as a fallback.** If embed load is too high, the recipe is:
    `generator.normalizer = None; model.__prepare_scriptable__()`, inline the detector's in-place
    softmax slice, `opset_version=17, dynamo=False`, `dynamic_axes` on the frames axis, then
    `quantize_dynamic(..., QuantType.QInt8)`. Expect 1.5–2.5× from ORT FP32 and less than the usual
    INT8 multiplier because ONNX cannot quantise LSTM. Try `torch.set_num_threads(2)` first.
12. **Pitch discipline.** Never say the watermark survives arbitrary transport. The defensible line is
    the one the evidence supports: *the token is a freshness proof carried in the audio, demonstrated
    end-to-end on a real virtual-mic path; codec-robust embedding is a fine-tuning problem, and the
    published fix (CallShield / StreamMark) is a narrowband causal re-train, not a different
    protocol.* That is a roadmap item, not a hole.

---

## Appendix A — measured on this machine

See `research/deep/audioseal-measurements.md` for the raw output of the measurement script (embed
strategy comparison, bit accuracy vs window length, alpha sweep, false-positive distribution, CPU
timings, and an ffmpeg codec/telephony-band survival sweep).

## Appendix B — source index

- AudioSeal paper — <https://arxiv.org/abs/2401.17264>
- AudioSeal repo / README — <https://github.com/facebookresearch/audioseal>
- `models.py` (generator, detector, MsgProcessor, NormalizationProcessor, streaming ctx) — <https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/models.py>
- `builder.py` (normalizer default False) — <https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/builder.py>
- model cards (stride 320, causal flags) — <https://github.com/facebookresearch/audioseal/tree/main/src/audioseal/cards>
- Moshi streaming conv cache — <https://github.com/facebookresearch/audioseal/blob/main/src/audioseal/libs/moshi/modules/streaming.py>
- `Getting_started.ipynb` (chunk-size vs message accuracy table) — <https://github.com/facebookresearch/audioseal/blob/main/examples/Getting_started.ipynb>
- CHANGELOG (0.2 streaming, TorchScript refactor, resampling deprecation) — <https://github.com/facebookresearch/audioseal/blob/main/CHANGELOG.md>
- Streaming PR #98 — <https://github.com/facebookresearch/audioseal/pull/98>
- Issue #96 (causal streaming models announced) — <https://github.com/facebookresearch/audioseal/issues/96>
- Issue #50 (AAC 64 k + Windows audio enhancements break the message) — <https://github.com/facebookresearch/audioseal/issues/50>
- Issue #52 (message decode is alignment-sensitive; silence beeping) — <https://github.com/facebookresearch/audioseal/issues/52>
- Issue #89 (Table 3 used per-attack thresholds; default 0.5) — <https://github.com/facebookresearch/audioseal/issues/89>
- Issues #14 / #17 / #24 (false-positive reports were all user error) — <https://github.com/facebookresearch/audioseal/issues/14>
- Issue #27 (EnCodec variant mismatch kills detection) — <https://github.com/facebookresearch/audioseal/issues/27>
- AudioMarkBench — <https://arxiv.org/abs/2406.06979>, <https://github.com/moyangkuo/AudioMarkBench>
- StreamMark — <https://arxiv.org/abs/2604.11917>
- CallShield — <https://arxiv.org/abs/2601.09327>, <https://arxiv.org/html/2601.09327v1>
- provcheck survival range table — <https://github.com/CreativeMayhemLtd/provcheck/blob/main/docs/audio-watermark-survival-range.md>
- provcheck codec-survival + alpha sweep — <https://github.com/CreativeMayhemLtd/provcheck/blob/main/docs/v0.5.2-codec-survival/README.md>
- provcheck ONNX export notes — <https://github.com/CreativeMayhemLtd/provcheck/blob/main/scripts/export-audioseal.py>
- audioseal-onnx (INT8 ORT build) — <https://github.com/thewh1teagle/audioseal-onnx>
- Zoom Original Sound — <https://support.zoom.us/hc/en-us/articles/115003279466-Preserve-original-sound>
- Zoom high-fidelity music mode — <https://theglobalconservatory.com/pages/zoom-audio-settings-for-high-quality-music-lessons>
- Google Meet Opus/DTX/FEC/RED — <http://www.rtcbits.com/2022/06/webrtc-google-meet.html>
- Microsoft Satin codec — <https://techcommunity.microsoft.com/blog/microsoftteamsblog/satin-microsoft%E2%80%99s-latest-ai-powered-audio-codec-for-real-time-communications/2141382>
- Opus bitrate range — <https://opus-codec.org/>
- Windows 11 audio enhancements / APOs — <https://www.ninjaone.com/blog/enable-or-disable-audio-enhancements-in-windows-11/>
- ONNX Runtime INT8 speedups — <https://medium.com/microsoftazure/faster-and-smaller-quantized-nlp-with-hugging-face-and-onnx-runtime-ec5525473bb7>
- ONNX cannot quantise LSTM — <https://github.com/microsoft/onnxruntime/issues/5747>
