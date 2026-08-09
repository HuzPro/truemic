# Deep research — CPU speaker verification for TrueMic

Scope: best CPU speaker-verification setup for (a) a fast desktop and (b) an Intel i5-7200U
laptop (Kaby Lake, 2C/4T, AVX2, no AVX-512), plus the score-processing tricks the field
actually uses. Every substantive claim is cited. Numbers marked **[measured]** were produced
on this machine during the research (methodology in §1).

**Bottom line:** replace SpeechBrain ECAPA (PyTorch fp32) with **WeSpeaker CAM++ (VoxCeleb, LM)
in ONNX Runtime**, keep 80-dim Kaldi fbank + per-utterance CMN, stop rescanning the VAD over the
whole window, average enrollment embeddings (already done) and add a small fixed impostor cohort
for adaptive score normalisation plus a 3-window median. Do **not** int8-quantise the model.

---

## 1. Measurement methodology (for the [measured] numbers)

- Host: **AMD Ryzen 5 7600** (Zen 4, 6C/12T, AVX-512), Windows 11, Python 3.12,
  `onnxruntime` 1.24.4 CPU EP, `torch` as installed in this repo's environment.
- ONNX models pulled from the sherpa-onnx speaker-recognition release
  (<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>).
- Input: random float32 fbank of shape `(1, 248, 80)` = a **2.5 s window** at 25 ms/10 ms with
  `snip_edges=True`. Median of 20 runs after 3 warm-ups; `inter_op_num_threads=1`,
  `graph_optimization_level=ORT_ENABLE_ALL`.
- Scaling to the i5-7200U is an **estimate**, not a measurement: PassMark single-thread
  **3,908** (Ryzen 5 7600, <https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+5+7600&id=5172>)
  vs **1,729** (i5-7200U, 2C/4T, 15 W,
  <https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i5-7200U+%40+2.50GHz&id=2865>) → 2.26×,
  and ONNX Runtime's MLAS kernels use AVX-512 on Zen 4 but only AVX2 on Kaby Lake. Use **3–4×**
  as the projection band, and remember a 15 W part will throttle under sustained load.

---

## 2. Model options: accuracy, cost, and what you can actually download

### 2.1 Published accuracy and complexity

WeSpeaker's own VoxCeleb2 recipe table (train VoxCeleb2-dev, test VoxCeleb1-O/E/H cleaned) —
<https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md>:

| Model | Params | Scoring | vox1-O | vox1-E | vox1-H |
|---|---|---|---|---|---|
| ResNet34-TSTP-emb256 | 6.63M | Cosine | 0.867 | 1.049 | 1.959 |
| | | AS-Norm | 0.787 | 0.964 | 1.726 |
| | | LM+AS-Norm | 0.723 | 0.867 | 1.532 |
| | | LM+AS-Norm+QMF | 0.659 | 0.821 | 1.437 |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.19M | Cosine | 1.069 | 1.209 | 2.310 |
| | | LM+AS-Norm | 0.782 | 1.005 | 1.824 |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M | Cosine | 0.856 | 1.072 | 2.059 |
| | | LM+AS-Norm+QMF | 0.707 | 0.894 | 1.615 |
| **CAM++** | **7.18M** | Cosine | **0.803** | 0.932 | 1.860 |
| | | AS-Norm | **0.718** | 0.879 | 1.735 |
| | | LM+AS-Norm | **0.659** | 0.803 | 1.569 |

CAM++ paper complexity table (single-thread CPU RTF, arXiv 2303.00332,
<https://arxiv.org/abs/2303.00332>, readable copy <https://ar5iv.labs.arxiv.org/html/2303.00332>):

| Model | Params (M) | FLOPs (G) | RTF (1 thread CPU) | vox1-O EER |
|---|---|---|---|---|
| ECAPA-TDNN | 14.66 | 3.96 | 0.033 | 0.89% |
| ResNet34 | 6.70 | 6.84 | 0.032 | 0.97% |
| **CAM++** | **7.18** | **1.72** | **0.013** | **0.73%** |

The paper's claim — "more than twice the inference speed of both ResNet34 and ECAPA-TDNN" and
"51% fewer parameters and 18% lower EER than ECAPA-TDNN in VoxCeleb-O" — reproduces almost
exactly in my measurements below (CAM++ RTF 0.012 single-thread vs ResNet34 0.031).

**Short-utterance behaviour** is the number that matters for a 2–3 s window. ERes2NetV2
(Interspeech 2024, <https://arxiv.org/html/2406.02167v1>, abstract
<https://arxiv.org/abs/2406.02167>) reports VoxCeleb1-O EER by trial duration:

| Duration | ERes2NetV2 (17.8M, 12.6G FLOPs) | ERes2Net | ECAPA-TDNN | ResNet34 |
|---|---|---|---|---|
| full | 0.61% | 0.84% | — | — |
| 3 s | 0.98% | 1.89% | 1.27% | 1.47% |
| 2 s | 1.48% | 3.28% | 1.95% | 2.12% |

Two things to take from this: (a) a 2 s window roughly **doubles** EER versus full utterances for
every architecture — this is the quantitative backing for TrueMic's "decide on 2–3 s, never 1 s"
rule; (b) at short durations ECAPA and ResNet34 are within ~10% of each other, so the choice
between them is a compute decision, not an accuracy decision.

3D-Speaker's model zoo (<https://github.com/modelscope/3D-Speaker/blob/main/README.md>) lists
CAM++ 7.2M @ 0.65% vox1-O, ERes2Net-base 6.61M @ 0.84%, ERes2NetV2 17.8M @ 0.61%,
ECAPA-TDNN 20.8M @ 0.86%, ResNet34 6.34M @ 1.05%.

NVIDIA TitaNet: TitaNet-L is 25.3M params @ 0.68% vox1-O
(<https://huggingface.co/nvidia/speakerverification_en_titanet_large>); TitaNet-S is ~6.4M
(TitaNet paper, <https://arxiv.org/pdf/2110.04410>). The small variant has no widely published
VoxCeleb EER; treat it as unvalidated.

ReDimNet2 is the current efficiency frontier if you want to look past CAM++:
B0 = 1.1M params / 0.33 GMACs / 1.04% vox1-O, B1 = 2.1M / 0.56 GMACs / 0.78%,
B2 = 3.6M / 0.95 GMACs / 0.57% (<https://github.com/PalabraAI/redimnet2>, original ReDimNet
<https://arxiv.org/pdf/2407.18223>). No turnkey ONNX in the sherpa/wespeaker distribution
channels, so it is a post-hackathon upgrade, not a today choice.

### 2.2 What actually ships as a downloadable ONNX file

Trap worth knowing: **`pyannote/wespeaker-voxceleb-resnet34-LM` on HuggingFace is *not* ONNX** —
its only weight file is `pytorch_model.bin` (26.6 MB)
(<https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/main>). The ONNX lives in
the WeSpeaker org (`voxceleb_resnet34_LM.onnx`, 26.5 MB,
<https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/tree/main>) and, most
conveniently, in the sherpa-onnx release, which hosts a single flat list of ONNX files
(<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>):

- `wespeaker_en_voxceleb_CAM++.onnx` (29.3 MB), `wespeaker_en_voxceleb_CAM++_LM.onnx` (29.3 MB)
- `wespeaker_en_voxceleb_resnet34.onnx` / `_LM.onnx` (26.5 MB)
- `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` (29.6 MB),
  `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` (28.3 MB),
  `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` (28.3 MB, 192-dim, trained on
  ~200k speakers across VoxCeleb + CNCeleb + 3D-Speaker)
- `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` (26.5 MB)
- `nemo_en_titanet_small.onnx` (40.3 MB), `nemo_en_titanet_large.onnx` (101.4 MB)

**No `.int8.onnx` variants exist in that release** (verified against the GitHub release asset
list) — which is consistent with §3.2 below.

### 2.3 [measured] Cost on real hardware

2.5 s window (248 frames × 80), median ms per embedding, Ryzen 5 7600:

| Model | file | 1 thread | 2 threads | 4 threads | RTF @2 thr |
|---|---|---|---|---|---|
| **WeSpeaker CAM++ LM (vox)** | 29.3 MB | 29.8 | **20.6** | 17.0 | **0.008** |
| 3D-Speaker CAM++ zh_en advanced | 28.3 MB | 29.4 | 20.3 | 17.1 | 0.008 |
| NeMo TitaNet-small | 40.3 MB | 27.1 | 18.5 | 14.8 | 0.007 |
| WeSpeaker ResNet34 LM (vox) | 26.5 MB | 78.0 | 42.3 | 29.0 | 0.017 |
| 3D-Speaker ERes2Net (vox) | 26.5 MB | 74.5 | 41.8 | 29.1 | 0.017 |
| **SpeechBrain ECAPA (torch fp32)** | — | **496** | **81.6** | 63 | 0.033 |

Supporting costs, same host:

| Component | cost |
|---|---|
| `torchaudio.compliance.kaldi.fbank`, 2.5 s, 1 thread | **1.9 ms** (≈6% of a CAM++ inference) |
| silero-VAD (torch), one 512-sample frame | **0.75 ms** |
| silero-VAD, full 2.5 s window rescan (78 frames) | **58 ms** |
| ONNX session load (CAM++, cold) | 561 ms |

CAM++ cost vs window length [measured, 2 threads]: 1.5 s → 13.6 ms, 2.0 s → 17.3 ms,
2.5 s → 19.5 ms, 3.0 s → 22.8 ms, 4.0 s → 28.2 ms. **Going from a 2.5 s to a 3.0 s window costs
3 ms.** There is no CPU argument for a short window; the argument for a short window is only
gate latency.

### 2.4 Projection to the i5-7200U (2 physical cores, so 2 threads is the ceiling)

Applying the 3–4× band from §1, per decision (one every 0.5 s):

| Stack | est. ms/decision on i5-7200U | % of the 500 ms budget |
|---|---|---|
| CAM++ ONNX (2 thr) | 60 – 85 | 12 – 17% |
| ResNet34 / ERes2Net ONNX (2 thr) | 125 – 170 | 25 – 34% |
| SpeechBrain ECAPA torch (2 thr) | 245 – 330 | 49 – 66% |
| **plus** current silero full-window rescan | 175 – 235 | 35 – 47% |

**The current TrueMic implementation (ECAPA torch + per-decision VAD rescan) needs ~85–110% of
the wall clock on that laptop before audio I/O, watermarking, or Python overhead.** It will not
hold. CAM++ ONNX + streaming VAD lands at roughly 15–20% and leaves headroom for AudioSeal.

Cross-check on the published side: WeSpeaker's own ONNX runtime benchmark (Intel Xeon Platinum
8160, single thread, 80k samples) reports RTF 0.0184 for ECAPA-TDNN c512 and **0.0607 for
ResNet-34**, with vox1-O EER 0.814% for the ResNet-34 ONNX and a note that "ONNX and PyTorch
versions show identical performance"
(<https://github.com/wenet-e2e/wespeaker/blob/master/runtime/onnxruntime/README.md>). My
single-thread ResNet34 number (0.031 RTF on a much newer core) is the same order.

---

## 3. Runtime: ONNX Runtime vs PyTorch, and the quantisation trap

### 3.1 ONNX Runtime vs PyTorch eager

For these small conv nets on few cores, the ONNX win is large and comes mostly from
PyTorch's poor low-thread-count behaviour, not from graph tricks:

- [measured] ECAPA torch fp32 at 1 thread: **496 ms** for a 2.5 s window (reproducible across
  repeated interleaved runs: 496 / 515 ms at 1 thread vs 81 ms at 2 threads). PyTorch's
  single-thread path for this model on this build is pathological; ONNX CAM++ at 1 thread is
  **17× faster**, and even ONNX ResNet34 at 1 thread beats torch-ECAPA at 2 threads.
- Same-architecture comparison is roughly 4× at 2 threads (ECAPA torch 81.6 ms vs CAM++ ONNX
  20.6 ms), of which ~2.3× is the FLOP difference (3.96 G vs 1.72 G) and the rest is runtime.

Counter-evidence worth knowing so you don't over-claim: an optimisation study of the DIART
diarization pipeline found ONNX export of `pyannote/embedding` made latency *worse*
(0.0909 s = 139.6% of the PyTorch baseline) on a Xeon Gold 5215
(<https://arxiv.org/html/2408.02341v1>). Their hypothesis was that the exported graph could not
use the hardware optimisations the PyTorch path did. So "ONNX is faster" is not a law — but for
the wespeaker/3D-Speaker exports, which are produced and benchmarked by the model authors for
exactly this purpose, it holds, and my measurements confirm it.

Threading guidance: set `intra_op_num_threads` to the **physical** core count (2 on the
i5-7200U), `inter_op_num_threads = 1`, and sweep rather than assume
(<https://onnxruntime.ai/docs/performance/tune-performance/threading.html>). Note the
diminishing return already at 4 threads on the desktop (20.6 → 17.0 ms), i.e. this model does
not parallelise well; per-core speed matters more than core count.

### 3.2 int8 dynamic quantisation — do not do this

ONNX Runtime's own guidance: *"it is recommended to use dynamic quantization for RNNs and
transformer-based models, and static quantization for CNN models"*, and the compute win requires
*"x86-64 with VNNI"*, with explicit warning that older hardware "may see performance degradation
due to quantization overhead"
(<https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>). Kaby Lake has
no VNNI.

[measured] `quantize_dynamic(QInt8)` on `wespeaker_en_voxceleb_CAM++_LM.onnx`:

| | size | 1 thread | 2 threads |
|---|---|---|---|
| fp32 | 29.3 MB | 41.8 ms | 35.7 ms |
| int8 dynamic | 8.4 MB | **689 ms** | **307 ms** |

**16–23× slower.** (The fp32 figures in this particular run are higher than §2.3 because the
quantisation pass ran in the same process; the *ratio* is the point.) Embedding agreement between
the two models on identical input was cos = 0.93 — i.e. also a meaningful accuracy shift, though
that specific number came from synthetic input and should not be read as an EER estimate.

What the literature says about the *accuracy* of quantised speaker embedders, for completeness:
4-bit quantisation-aware training of ResNet34 is near-lossless (vox1-H 1.899% vs 1.850%
full-precision, 7.7× smaller), while 1-bit costs a lot (3.146%)
(<https://arxiv.org/html/2406.05359v2>); the DIART study saw int8 leave DER essentially unchanged
(49.7261 → 49.7276) with a 7% latency gain on a Xeon (<https://arxiv.org/html/2408.02341v1>).
So quantisation *can* be accuracy-safe — but the speed win needs VNNI and, for a CNN, static
calibration. Neither is worth a hackathon hour when fp32 CAM++ already fits in 15% of the budget.

---

## 4. Feature extraction: the exact recipe the WeSpeaker ONNX models expect

### 4.1 What the exported graph takes

`export_onnx.py` exports input `feats` with shape `(B, T, 80)` and dynamic axes `{0: 'B', 1: 'T'}`,
output `embs`
(<https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/export_onnx.py>). So the model
consumes **fbank features, not raw audio** — you own the front end. It also supports an optional
`--mean_vec` that subtracts a fixed vector *from the embedding* (not the features); the shipped
models use zeros.

### 4.2 The exact recipe

From WeSpeaker's training pipeline
(<https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/dataset/processor.py>) and its
inference CLI (<https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/speaker.py>):

```python
waveform = waveform * (1 << 15)          # float [-1,1] -> int16 range
mat = kaldi.fbank(waveform,              # torchaudio.compliance.kaldi
                  num_mel_bins=80,
                  frame_length=25,       # ms
                  frame_shift=10,        # ms
                  dither=1.0,            # TRAINING only; use 0.0 at inference
                  sample_frequency=16000,
                  window_type='hamming', # note: torchaudio's default is 'povey'
                  use_energy=False)
mat = mat - mat.mean(dim=0, keepdim=True)   # CMN: norm_mean=True, norm_var=False
```

The CLI's line is literally `if cmn: feat = feat - torch.mean(feat, dim=0)` with `cmn=True` by
default. Training config confirms `num_mel_bins: 80, frame_shift: 10, frame_length: 25,
dither: 1.0` and `embed_dim: 512` for CAM++
(<https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/conf/campplus.yaml>);
CMVN was moved out of the CPU dataset pipeline into the train/extract steps
(<https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/dataset/dataset.py>).

Beware a cross-implementation inconsistency: sherpa-onnx only applies CMN when the ONNX metadata
carries `feature_normalize_type == "global-mean"`, defaulting to empty/no-CMN
(<https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h>),
and its wespeaker metadata script does not set that key
(<https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wespeaker/add_meta_data.py>) — it only
sets `normalize_samples = 0`, meaning "samples are expected in [-32768, 32767]". Follow
**WeSpeaker's own** recipe (CMN on), and above all use the *same* recipe for enrollment and for
scoring.

### 4.3 [measured] How much each choice actually matters

Feeding a CAM++ ONNX identical synthetic signals through variant front ends (cosine between the
resulting length-normalised 512-dim embeddings):

| Variation | cosine |
|---|---|
| int16 scaling vs float [-1,1] scaling, **CMN on** | 0.996 |
| hamming vs povey window, CMN on | 0.997 |
| gain +6 dB, CMN on | 1.000 |
| **CMN on vs CMN off** | **0.532** |
| same signal + white noise at −20 dB, CMN on | 0.903 |

Read: **CMN is the only front-end choice that can silently destroy your system** — omitting it
does not "slightly degrade", it produces a different embedding space. The `<<15` scaling and the
window type are within noise once CMN is applied (CMN removes the constant log-offset that the
scale introduces), but keep them anyway to match the reference. Caveat: these were harmonic
synthetic signals, so treat them as *sensitivity* probes, not speaker-discrimination results.

### 4.4 Cost, and doing it without torchaudio

Feature extraction is **1.9 ms per 2.5 s window [measured]** — about 6% of a CAM++ inference and
1.5% of an ECAPA-torch inference. It is never the bottleneck; do not micro-optimise it.

Since `torchaudio` is already a TrueMic dependency, `torchaudio.compliance.kaldi.fbank` is the
zero-work answer (docs:
<https://docs.pytorch.org/audio/stable/generated/torchaudio.compliance.kaldi.fbank.html>).
If you later drop torch entirely (worth doing — it also removes ~0.5 s of import time and a few
hundred MB), the drop-in is **`pip install kaldi-native-fbank`**, the same C++ extractor
sherpa-onnx uses, tested bit-exact against Kaldi and with Windows wheels
(<https://github.com/csukuangfj/kaldi-native-fbank>). Hand-rolling fbank in numpy/librosa is the
one path to avoid: the failure modes are Kaldi's Povey/Hamming window definition, pre-emphasis,
snip_edges frame counting, the mel bank's exact edge frequencies, `dither` adding
non-determinism, and `use_energy`/`remove_dc_offset` defaults — five ways to be silently 0.5
cosine away from the model's training distribution, exactly as the CMN row above shows.

---

## 5. Score processing: what the field actually does

### 5.1 Adaptive score normalisation (AS-Norm)

The normalised score uses cohort statistics on both sides:
`S_as = ½ · [ (S − mean_e)/std_e + (S − mean_t)/std_t ]`, where the means/stds come from the
**top-N** scoring impostors in a cohort. In ECAPA-TDNN's original paper the *"imposter cohort
consists of the speaker-wise averages of the length-normalized embeddings of all training
utterances"*, with cohort size **1000 for VoxCeleb** but only **50 for the cross-dataset VoxSRC
2019 evaluation** (<https://ar5iv.labs.arxiv.org/html/2005.07143>, ISCA copy
<https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf>). Other systems use
300–600 (<https://arxiv.org/pdf/2209.10147>, <https://arxiv.org/pdf/2308.08766>).

Measured benefit in WeSpeaker's own recipe (cosine → AS-Norm, same model): CAM++
0.803 → 0.718 (−10.6% rel.), ResNet34 0.867 → 0.787 (−9.2% rel.), ECAPA c512 1.069 → 0.957
(−10.5% rel.).

**Is a cheap version worth it for one enrolled speaker?** Yes, but not for the EER: the
enrollment side is a constant, so `mean_e`/`std_e` are a fixed affine transform that changes
nothing about ranking. The value is entirely in the **test side** — per-window, subtracting the
mean and dividing by the std of that window's scores against a small fixed cohort removes the
window-level nuisance variation (loudness, phonetic content, channel, background) that makes a
*fixed cosine threshold* drift. That is precisely why the cross-dataset (mismatched) condition in
the ECAPA paper still used a cohort — and only needed 50. For TrueMic the cohort is ~50–200
precomputed 512-dim vectors = 100–400 KB, and per window it is one 200×512 matvec, i.e.
**tens of microseconds** against a 20 ms inference. Cost-free relative to the benefit of a
threshold that survives a different room. A trainable variant exists (TAS-Norm,
<https://arxiv.org/abs/2504.04512>) but is irrelevant here.

### 5.2 Temporal smoothing

The field's practice for continuous/streaming verification is score integration over a running
window, not per-frame decisions: frame-level scores are smoothed with a moving average over a
window to produce an utterance-level score, and running decision windows are used to impose
continuity across transitions (<https://arxiv.org/pdf/2102.01760>,
<https://dl.gi.de/bitstreams/86109004-2cff-463a-8851-2a0117a711c9/download>). Exponential
smoothing, weighted exponential smoothing, moving average and Kalman filtering have all been
compared for exactly the "probability stability vs delay" trade-off in streaming classifiers.

For TrueMic the important structural fact is that **consecutive windows already overlap 80%**
(2.5 s window, 0.5 s hop), so the raw score series is heavily autocorrelated: an EMA on top adds
lag without adding much independent information, whereas a short **median of the last 3 scores**
kills exactly the failure mode that matters — a single bad window (a cough, a plosive, a
crossfade artifact) flipping the gate. Median over mean, because the risk is outliers, not noise.
Combined with the existing hysteresis margin this gives a two-stage smoother with ~1 s of
additional confirmation latency — which the 1.5 s delay line already absorbs for free.

### 5.3 Multi-embedding enrollment: mean, not max

The standard is **average the length-normalised enrollment embeddings, then re-normalise**.
Evidence:

- sherpa-onnx's `SpeakerEmbeddingManager`, when given several embeddings for one speaker, sums
  and normalises them (the mean division is skipped because normalisation follows), then scores
  with a plain cosine dot product against a single stored row
  (<https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/speaker-embedding-manager.cc>).
- The AS-Norm cohorts in the ECAPA paper are themselves "speaker-wise averages of the
  length-normalized embeddings" (<https://ar5iv.labs.arxiv.org/html/2005.07143>).
- The classical study of PLDA multi-enrollment scoring found embedding averaging systematically
  outperformed score averaging and by-the-book scoring for i-vectors, with length normalisation
  important throughout
  (<https://www.sciencedirect.com/science/article/abs/pii/S1051200414001377>). Only with modern
  large-margin embeddings *and* a proper PLDA/PSDA back-end does by-the-book scoring beat
  averaging (<https://arxiv.org/pdf/2302.09523>) — irrelevant with a cosine back-end.

So **TrueMic's `enroll_ecapa` is already doing the right thing** (mean of per-chunk embeddings,
re-normalised). `max` over per-chunk cosines is *not* standard and is actively risky: max is an
optimistic estimator, so it inflates impostor scores as much as genuine ones and widens the
overlap. Keep the mean. The only refinement worth considering: also keep the per-chunk
embeddings so you can *reject* enrollment chunks whose cosine to the mean is an outlier (a cough,
a second voice in the room), then re-average — cheap enrollment hygiene, not a scoring change.

### 5.4 Quality-weighted scoring (QMF)

This is a real and standard technique. Quality Measure Function calibration adds side information
— typically log enrollment duration, log test duration, and embedding magnitude — into the score
calibration transform. IDLab's VoxSRC-20 system reports **+8% EER / +6% MinDCF relative** from
quality-aware calibration with duration and imposter-mean QMFs
(<https://arxiv.org/pdf/2010.11255>), and WeSpeaker's table above shows LM+AS-Norm →
LM+AS-Norm+QMF taking ResNet34 from 0.723 to 0.659 (−8.9% rel.). The underlying idea traces to
duration-conditioned calibration (<https://ieeexplore.ieee.org/document/6584746/>).

Fitting a real QMF needs labelled trials you won't have. The hackathon-grade version is the part
that carries most of the benefit: **the amount of actual speech in the window is a quality
measure, so use it as a gate rather than a weight.** TrueMic already computes `voiced_fraction`;
requiring `MIN_VOICED_FRACTION` before scoring is a crude QMF, and raising the requirement for
*entering* the verified state versus *staying* in it is a duration-quality asymmetry in the right
direction.

### 5.5 What the open-source systems do, summarised

| System | Back-end | Enrollment | Normalisation |
|---|---|---|---|
| sherpa-onnx speaker ID | cosine, threshold as min score cutoff, `Verify()` returns `score < threshold → false` | sum + L2-normalise multiple embeddings | optional feature CMN via ONNX metadata |
| WeSpeaker CLI | cosine | single or averaged | fbank + CMN (`cmn=True` default) |
| WeSpeaker recipes | cosine + AS-Norm (+QMF) | speaker-wise averaged cohort | AS-Norm |
| NeMo speaker recognition | cosine similarity or neural classifier back-end (<https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/intro.html>) | averaged | — |

Nobody in this list uses max-of-cosines. Everyone length-normalises.

---

## 6. Channel / microphone mismatch

The problem is real and well documented: in far-field and cross-device conditions the
"mismatch between enrollment and testing utterances generally exists due to different recording
environments", and it is one of the two headline challenges alongside short utterances
(<https://arxiv.org/pdf/2005.08046>).

What measurably helps, ordered by cost:

1. **Enroll on the mic you will demo on.** The cheapest and largest win, and it costs nothing.
   Everything below is a fallback for when that is impossible.
2. **CMN over the scoring window** — already in the recipe (§4.2). Cepstral mean subtraction is
   the classical channel-compensation tool precisely because a linear channel is an additive
   constant in the log-spectral domain; the survey of channel compensation lists CMS, feature
   warping, CMVN and RASTA as the feature-domain family
   (<https://www.researchgate.net/publication/256303732_Speaker_verification_in_sensor_and_acoustic_environment_mismatch_conditions>).
   Note this is *why* the CMN row in §4.3 shows scale invariance: a constant gain is a constant
   log offset and CMN eats it.
3. **Multi-condition enrollment augmentation.** Augmenting *enrollment* speech with MUSAN-style
   additive noise and RIR reverberation, then averaging the resulting embeddings, is proven
   effective at reducing enroll/test mismatch
   (<https://arxiv.org/pdf/2005.08046>, <https://arxiv.org/pdf/2008.03521>,
   <https://arxiv.org/abs/2007.08004>). You do not need MUSAN for a cheap DSP version: white/pink
   noise at 10–20 dB SNR plus a synthetic exponentially-decaying-noise room impulse
   (RT60 ≈ 0.2–0.4 s) convolved with the enrollment audio costs ~20 lines of numpy/scipy. My
   [measured] −20 dB-noise probe (cosine 0.903 for an *identical* source signal) shows how much
   room a noise-only perturbation moves the embedding — which is the same distance the threshold
   has to tolerate.
4. **AS-Norm** (§5.1) — the *scoring*-side answer to the same problem, and the one the challenge
   systems reach for on cross-dataset conditions.
5. **Multi-condition enrollment in the plain sense:** record enrollment at two mic distances and
   two speaking levels and average all chunks. Free, no DSP.

What does **not** help at inference-only time: variance normalisation (`norm_var=True`) — the
models were trained with `norm_mean=True, norm_var=False`, so turning it on is a train/test
mismatch you introduce yourself.

---

## 7. Threshold from a 30 s + 30 s on-site calibration

### 7.1 The statistical problem with the current approach

`calibrate.py` slides a 2.5 s window with a 0.5 s hop over 30 s and takes the 5th percentile of
the resulting genuine scores. That yields ~56 windows — but consecutive windows share 80% of
their audio, so the **effective independent sample count is roughly 30 s / 2.5 s ≈ 12**. A p5
estimate from ~12 effective samples has enormous variance; it is essentially "the minimum of the
sample", and the minimum of a small sample is the least stable statistic you can pick. It will be
too low when the speaker happened to be consistent and too high when one window caught a pause.

A parametric estimate uses all the data instead of one order statistic. Cosine scores from a
fixed enrollment over a homogeneous session are close enough to Gaussian for this purpose:

```
t_genuine_floor = mean(genuine) - k * std(genuine)      # k = 1.65 -> ~5% FRR, k = 2.33 -> ~1%
```

`std` is estimated from all 56 windows (correlation inflates its uncertainty, but far less than
it wrecks a percentile), and `k` is an explicit, tunable false-reject posture rather than an
accident of sample size.

### 7.2 Using the impostor recording properly

The current midpoint rule `(p5_genuine + p95_impostor) / 2` treats both distributions as equally
trustworthy and equally wide. The field's operating-point language is the DET curve: the
threshold trades FAR against FRR, EER is where they cross, and picking a point requires a stated
cost (<https://www.sciencedirect.com/topics/computer-science/equal-error-rate>). For TrueMic the
costs are wildly asymmetric — a false accept leaks an impostor's voice with a "Verified Human"
badge on a demo stage; a false reject mutes a word. So:

- Compute both a **FRR-targeted** floor (`mean_g − k·std_g`) and a **FAR-targeted** ceiling
  (`mean_i + k'·std_i`, k' ≈ 2–3).
- If ceiling < floor there is separation: place the threshold in the gap but biased away from the
  impostor tail — `t = ceiling + 0.6·(floor − ceiling)` sits 60% of the way from the impostor
  ceiling toward the genuine floor. (The current unweighted midpoint gives the impostor
  distribution equal say despite it being the more expensive error.)
- If ceiling ≥ floor, the distributions overlap and no threshold works — keep the existing
  warning, and say so loudly rather than emitting a number.
- Report the estimated FRR and FAR at the chosen threshold, from the Gaussian fits. A calibration
  tool that prints "threshold 0.42 → est. 3% of your words muted, est. 0.5% impostor windows
  passed" is a *far* better demo asset than one printing a bare number.

This is a poor-man's linear calibration; the principled version is logistic-regression calibration
to log-likelihood ratios as in the BOSARIS toolkit, where the score-to-LLR map is affine
`x(s) = a·s + b` fitted by cross-entropy, and where the documented practice under limited
development data is to train and apply on the same scores
(<https://www.isca-archive.org/interspeech_2013/leeuwen13_interspeech.pdf>,
<https://arxiv.org/pdf/1304.1199>). With 30 s per class, the Gaussian two-parameter fit is the
right amount of machinery.

### 7.3 Hysteresis headroom

`HYSTERESIS_MARGIN = 0.08` is currently a bare constant, unrelated to the measured spread. It
should be **derived from the genuine score spread**, because that is what it exists to absorb:

```
HYSTERESIS = clamp(0.75 * std(genuine), 0.03, 0.12)
```

A margin much smaller than the genuine std chatters; much larger than it means the gate is
effectively latched and a real impostor gets extra seconds. Tying it to the calibration output
makes it self-tuning on a noisy stage. With the 3-window median from §5.2 already suppressing
single-window outliers, the margin can sit at the lower end of that range.

---

## 8. Concrete recommendations for TrueMic

Ordered by (value ÷ effort).

1. **Swap the ECAPA fallback for WeSpeaker CAM++ LM in ONNX Runtime.** `wespeaker_en_voxceleb_CAM++_LM.onnx`,
   29 MB, 512-dim, 0.659% vox1-O with LM+AS-Norm. Measured 20.6 ms per 2.5 s window at 2 threads
   (vs 81.6 ms for ECAPA torch), projected 60–85 ms on the i5-7200U. `onnxruntime` is already
   installed. Feature front end: `torchaudio.compliance.kaldi.fbank` per §4.2 + CMN, 1.9 ms.
2. **Stop rescanning silero over the whole window.** 58 ms per decision → 0.75 ms per incoming
   frame with a rolling voiced-frame counter. Bigger saving than the model swap on the laptop.
   Use the ONNX silero build for another 2–3× (189 µs per 31.25 ms chunk, single thread —
   <https://github.com/snakers4/silero-vad/wiki/Performance-Metrics>).
3. **Never int8-quantise.** Measured 16–23× slower on this class of model; ORT itself recommends
   static (not dynamic) quantisation for CNNs and requires VNNI for the compute win, which Kaby
   Lake lacks.
4. **Keep mean-of-chunks enrollment** (already correct); add outlier rejection of enrollment
   chunks whose cosine to the running mean is > 2σ low, then re-average.
5. **Add a fixed impostor cohort + one-sided AS-Norm.** 50–200 precomputed embeddings shipped as
   a `.npy`; per window compute `z = (s − mean(top-N cohort scores)) / std(top-N)` and threshold
   on `z`. Microseconds of cost; makes the threshold portable across rooms and mics — the single
   biggest risk to an on-site demo.
6. **Median-of-3 on consecutive window scores** before the hysteresis comparison. Kills
   single-window flips; the 1.5 s delay line already hides the added confirmation latency.
7. **Rewrite threshold estimation as a Gaussian fit** (`mean ± k·std`) instead of p5/p95
   percentiles, print estimated FRR/FAR at the chosen threshold, and derive `HYSTERESIS_MARGIN`
   from `std(genuine)` instead of hard-coding 0.08.
8. **Widen the window to 3.0 s if latency allows.** Costs 3 ms more per decision; the published
   short-duration curves put 3 s at ~2/3 the EER of 2 s for every architecture.
9. **Cheap multi-condition enrollment** if you cannot enroll on the demo mic: augment the 40 s
   enrollment with (a) +10 dB and +20 dB SNR white noise and (b) one synthetic RT60 ≈ 0.3 s
   impulse, embed each condition's chunks, average everything.
10. **Do not chase TitaNet-S or ERes2NetV2 today.** TitaNet-small is fast (18.5 ms) but has no
    published VoxCeleb EER; ERes2Net/ResNet34 are 2× the cost of CAM++ for no short-duration
    advantage. ReDimNet2-B1 (2.1M / 0.56 GMACs / 0.78% vox1-O) is the right post-hackathon
    upgrade if ONNX exports appear.

---

## 9. Sources

**Models / accuracy**
- WeSpeaker VoxCeleb v2 recipe results — https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md
- CAM++ paper — https://arxiv.org/abs/2303.00332 · https://ar5iv.labs.arxiv.org/html/2303.00332
- ERes2NetV2 (Interspeech 2024) — https://arxiv.org/abs/2406.02167 · https://arxiv.org/html/2406.02167v1 · https://www.isca-archive.org/interspeech_2024/chen24l_interspeech.html
- 3D-Speaker model zoo — https://github.com/modelscope/3D-Speaker/blob/main/README.md
- ECAPA-TDNN paper — https://ar5iv.labs.arxiv.org/html/2005.07143 · https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf
- SpeechBrain ECAPA card (0.80% vox1-test) — https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- TitaNet — https://arxiv.org/pdf/2110.04410 · https://huggingface.co/nvidia/speakerverification_en_titanet_large
- ReDimNet / ReDimNet2 — https://arxiv.org/pdf/2407.18223 · https://github.com/PalabraAI/redimnet2

**Weights / runtime**
- sherpa-onnx speaker-recognition model release — https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
- Wespeaker HF org (ONNX) — https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/tree/main
- pyannote HF copy (PyTorch only, no ONNX) — https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/tree/main
- WeSpeaker ONNX runtime + RTF benchmarks — https://github.com/wenet-e2e/wespeaker/blob/master/runtime/onnxruntime/README.md
- ONNX export script — https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/export_onnx.py
- ORT quantization guidance — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- ORT threading — https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- DIART optimisation study (ONNX/int8 on Xeon) — https://arxiv.org/html/2408.02341v1
- Quantisation of speaker-verification nets (4-bit near-lossless) — https://arxiv.org/html/2406.05359v2

**Front end**
- WeSpeaker feature processor — https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/dataset/processor.py
- WeSpeaker dataset pipeline (CMVN moved out) — https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/dataset/dataset.py
- WeSpeaker CLI (cmn default) — https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/cli/speaker.py
- CAM++ training conf — https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/conf/campplus.yaml
- sherpa-onnx feature/CMN handling — https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h · https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/wespeaker/add_meta_data.py
- torchaudio kaldi fbank — https://docs.pytorch.org/audio/stable/generated/torchaudio.compliance.kaldi.fbank.html
- kaldi-native-fbank (torch-free, bit-exact) — https://github.com/csukuangfj/kaldi-native-fbank

**Scoring**
- sherpa-onnx embedding manager (mean + cosine) — https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/speaker-embedding-manager.cc
- Multi-enrollment PLDA scoring variants — https://www.sciencedirect.com/science/article/abs/pii/S1051200414001377
- Modern back-ends vs embedding averaging — https://arxiv.org/pdf/2302.09523
- Trainable AS-Norm — https://arxiv.org/abs/2504.04512
- Cohort sizes in challenge systems — https://arxiv.org/pdf/2209.10147 · https://arxiv.org/pdf/2308.08766
- IDLab VoxSRC-20 quality-aware calibration (QMF) — https://arxiv.org/pdf/2010.11255
- Duration QMF calibration — https://ieeexplore.ieee.org/document/6584746/
- Robust back-end / calibration — https://arxiv.org/pdf/2102.01760
- BOSARIS-style LLR calibration — https://www.isca-archive.org/interspeech_2013/leeuwen13_interspeech.pdf · https://arxiv.org/pdf/1304.1199
- NeMo speaker recognition back-ends — https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/intro.html
- Continuous verification score smoothing — https://dl.gi.de/bitstreams/86109004-2cff-463a-8851-2a0117a711c9/download

**Mismatch / robustness**
- Interspeech 2020 Far-Field SV Challenge — https://arxiv.org/pdf/2005.08046
- NPU far-field system (enrollment augmentation) — https://arxiv.org/pdf/2008.03521
- Enrollment augmentation for text-dependent SV — https://arxiv.org/abs/2007.08004
- Channel compensation survey — https://www.researchgate.net/publication/256303732_Speaker_verification_in_sensor_and_acoustic_environment_mismatch_conditions

**Hardware**
- Silero VAD performance metrics — https://github.com/snakers4/silero-vad/wiki/Performance-Metrics
- PassMark i5-7200U — https://www.cpubenchmark.net/cpu.php?cpu=Intel+Core+i5-7200U+%40+2.50GHz&id=2865
- PassMark Ryzen 5 7600 — https://www.cpubenchmark.net/cpu.php?cpu=AMD+Ryzen+5+7600&id=5172
