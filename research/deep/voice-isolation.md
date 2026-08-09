# Deep dive: how shipped "enrolled-voice mic isolation" actually works, and what TrueMic should copy

Research date: 2026-08-08. Every substantive claim below carries a URL. Where a number
comes from a vendor doc rather than a paper, that is flagged — vendors publish selectively.

---

## 0. Verdict up front

Four things fell out of the literature that change how TrueMic should be built:

1. **Nobody who ships this hard-mutes.** Teams, Zoom, Krisp and CrystalSound all implement
   enrolled-voice isolation as a *continuous time-frequency mask* inside a speech-enhancement
   model. There is no binary gate anywhere in the shipped world. The gate is TrueMic's
   differentiator and also its biggest UX liability.
2. **The whole field is tuned asymmetrically *against* over-suppression.** Google's
   VoiceFilter-Lite literally trains with a 10× penalty on removing target speech vs. leaving
   interference in ([asymmetric loss, α = 10](https://arxiv.org/abs/2009.04323)). Microsoft
   invented a metric (TSOS, "seconds of target speaker removed per half hour") purely to police
   it. TrueMic deliberately inverts this. That inversion is defensible — but it must be a
   *named, tunable posture*, not an accident, because every shipped product concluded the other
   way after contact with users.
3. **TrueMic's architecture is exactly Google's "SC" baseline** — an independently trained VAD
   plus an independently trained speaker verifier, combined at score level. Google measured this
   and it was the *worst* of the four architectures they tried
   ([Personal VAD, Odyssey 2020](https://www.isca-archive.org/odyssey_2020/ding20_odyssey.pdf)).
   The good news: the reason it was worst does not apply to TrueMic (see §A5), so the cheap
   architecture stands. The bad news: there are no public pretrained personal-VAD weights, so
   the better architecture is not reachable in a hackathon. Correct call to cut it.
4. **Channel/mic mismatch is handled by everyone the same way: constrain the channel, then
   multi-condition the enrollment.** CrystalSound mandates the same mic. Zoom assumes a headset
   and silently downgrades to plain noise removal if you aren't wearing one. Microsoft's DNS
   challenges deliberately ship *noisy and reverberant* enrollment clips to force robustness.

---

## Part A — How each product and paper actually works

### A1. Microsoft Teams Voice Isolation (the pDNS lineage)

**What it is, officially.** "Voice isolation reduces noise and uses the personalized deep voice
quality enhancement artificial intelligence (AI) model to separate the user's voice from other
sounds and voices" and it "relies on the voice profile, stored on the user's local device, to
remove any sounds or voices during a call or meeting that don't match the profile."
(<https://learn.microsoft.com/en-us/microsoftteams/voice-isolation>)

Note the verb: *remove sounds*, not *mute the stream*. This is mask-based enhancement, not gating.

**Enrollment — two paths.**

- *Manual / express*: Settings → Recognition → Create voice profile; pick your microphone and
  read the provided text, **~30 seconds**, in "a relatively quiet environment."
  (<https://support.microsoft.com/office/voice-isolation-in-microsoft-teams-calls-and-meetings-a9756ea9-4cec-44c4-aefb-6f5d17c89427>)
- *Passive*: the client captures and updates the voice profile from in-meeting speech;
  "generating a voice profile using in-meeting speech typically takes a couple of meetings."
  (<https://learn.microsoft.com/en-us/microsoftteams/rooms/voice-and-face-recognition>)

Passive enrollment is the interesting one: it is inherently **multi-condition** (many meetings,
many mics, many rooms) and inherently **continuously refreshed**. That is Microsoft's answer to
channel mismatch, and it costs the user nothing.

**Profile lifecycle (channel-drift mitigation in disguise).** Same doc: "When a user also uses
Voice Isolation on their device, a local copy of the voice profile is stored encrypted. This
copy of the voice profile expires after **14 days** and is replaced with a new copy of the voice
profile." Profiles unused for one year are deleted; profiles are removed within 90 days of
account deletion. Enrollment is localized into **25 language variants**.

**Admin surface** (tells you what the product treats as separable concerns):
`Set-CsTeamsMeetingPolicy -VoiceIsolation`, `Set-CsTeamsAIPolicy -EnrollVoice`,
`Set-CsTeamsAIPolicy -PassiveVoiceEnrollment`. Enrollment and use are separate switches.

**The false-reject escape hatch.** The user-facing doc describes a notification that appears when
"someone nearby is detected (e.g., you and a colleague are sharing your device for the meeting)"
so the user can turn isolation off. Microsoft's answer to "the gate might eat someone who should
be heard" is a *UI affordance*, not a smarter model. Also present: a hard failure mode banner —
"Your device is running too many apps and can't support voice isolation."

**The underlying research.** Teams' feature is the productisation of Microsoft's personalized
deep noise suppression (pDNS) line, run as tracks of the DNS Challenge:

| Edition | Enrollment given | Embedding extractor | Baseline model | Latency rule |
|---|---|---|---|---|
| ICASSP 2021 | 2 minutes of the speaker | (participant choice) | NSNet2 (non-personalized) | total algorithmic latency incl. frame T + stride Ts + lookahead **≤ 40 ms**; must process a frame in less than the stride time on an Intel Core i5 quad-core @ 2.4 GHz ([ar5iv](https://ar5iv.labs.arxiv.org/html/2009.06122)) |
| ICASSP 2022 | **2.5 minutes**, clean | **RawNet2** trained on VoxCeleb2 (wideband); input downsampled to 16 kHz for embedding extraction even though enhancement runs at 48 kHz | **pDCCRN** (personalized DCCRN), modified for 48 kHz | as 2021 ([arXiv 2202.13288](https://arxiv.org/pdf/2202.13288)) |
| ICASSP 2023 | **10–30 s**, deliberately "noise-free or noisy and with or without reverberation" | ECAPA-TDNN embedding code provided | — | tightened (see challenge site) ([arXiv 2303.11510](https://arxiv.org/pdf/2303.11510)) |

Two quotes from the 2022 paper worth carrying into TrueMic verbatim:

> "speaker models are sensitive to false-alarms in speech activity detection (SAD); clean speech
> can be used for obtaining accurate SAD labels resulting in speaker-discriminative embeddings."

— i.e. VAD-before-embedding is not optional. TrueMic already does this in `verify.py` /
`enroll.py`; this is the citation for why.

> "Speaker adaptation is expected to work well using multi-conditioned data; clean speech can be
> used for generating reverberant and noisy multi-condition enrollment data for speaker
> adaptation."

— i.e. Microsoft's own stated recipe is *record clean once, then synthesise the noisy/reverberant
variants yourself*. That is a 15-line numpy job for TrueMic (§D6).

The 2023 edition goes further and makes multi-condition enrollment a scored property:

> "Each test clip in both tracks has enrollment speech with 30s duration. The enrollment speech
> can be noise-free or noisy and with or without reverberation. This facilitates multi-condition
> enrollment of primary talkers which serves as a measure of robustness for personalized models."

The 2023 paper also notes that when preparing 5 s enrollment segments they "ensure all long pause
(>0.2s)" are handled — enrollment segment selection strips long pauses.

**Why W Acc / TSOS exist.** ICASSP 2022 added word accuracy as an objective metric because
"several models from past challenges had noticeable W Acc degradation resulting from
over-suppression of noise and/or speech distortions." The field's dominant failure mode is
*eating the target speaker*, and they built metrics to hunt it.

### A2. Zoom Personalized Audio Isolation

(<https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0074698>)

- **Passive by default**: after you enable it, the client "attempt[s] to create an automatic
  voiceprint of your voice" during your first meeting. No duration mandated.
- **Optional enhanced recording**: read a Zoom-supplied script; best practice is to speak "as you
  normally would in terms of tone, speed, and loudness for **at least 30 seconds**, and reduce or
  remove sources of background noise."
- **Storage split**: the automatic voiceprint stays on-device; the reference recording is stored
  by Zoom servers. Both user-deletable.
- **The channel assumption is explicit**: "Without a voiceprint recording: Personalized audio
  isolation helps in an open environment where background talkers may be heard, and the meeting
  participant uses a headset mic." If a non-headset microphone is detected and there is no
  voiceprint recording, the product **silently downgrades to standard noise removal**.

That last behaviour is the single most transferable product lesson in this whole document: when
the channel is outside what the model was calibrated for, ship a *weaker but safe* mode rather
than a strong mode that misfires.

### A3. Krisp BVC and VIVA — the "no enrollment at all" school

Krisp is the interesting negative result. Their Background Voice Cancellation removes *all*
background voices and it explicitly does **not** enrol anyone:

> "The technology does not require user voice enrollment or training on user voice data."
> (<https://sdk-docs.krisp.ai/docs/krisp-rtc-bvc-nc>)

Published engineering parameters:

| Property | Value | Source |
|---|---|---|
| Processing frame | "an audio stream of any frame size is supported, Krisp SDK processing is performed using **10 ms frames**" | BVC/NC doc |
| Working sample rate | **32 kHz**; anything above 8 kHz is resampled up to 32 kHz | BVC/NC doc |
| CPU | "varies depending on the platform within the range of **5–13 %**" | BVC/NC doc |
| Device requirement | headset/earbuds; best with "wired USB headsets with a boom microphone"; BT/AirPods OK | BVC/NC doc |
| `krisp-bvc-o-pro-v3` | 32 kHz, **14.7 MB** | [model guide](https://sdk-docs.krisp.ai/docs/rtc-model-guide-bvc-nc) |
| `krisp-bvc-o-lite-v2` | 16 kHz, **8.6 MB**, "USB/jack headsets only" | model guide |
| NC models | 2.8–5.8 MB (16/32 kHz), 2.8 MB narrowband | model guide |
| VIVA family | `viva-tel`/`tel-lite` (telephony ≤16 kHz), `viva-pro` (close mic — "requires that the user speak close to the microphone"), `viva-ss` (far-field, noise only) | [conversational AI models](https://sdk-docs.krisp.ai/docs/models-for-conversational-ai) |

Krisp never documents *how* the primary speaker is selected, but the requirement set gives it
away: a boom mic on a headset puts the wearer 20–40 dB above anyone else in the room and gives a
near-field spectral/reverberation signature. Krisp solves "whose voice is this" with **physics
instead of biometrics**. The ICASSP 2023 DNS paper says the same thing out loud: "Acoustic
properties of headset scenarios can be leveraged in developing models for suppressing neighboring
talkers without enrollment speech."

This matters for TrueMic's demo. The kill-shot beat ("teammate talks into the SAME mic") is
precisely the scenario where the physics shortcut fails and biometrics is the *only* thing that
can work — worth saying to judges, because it is why Krisp cannot do TrueMic's demo.

**Krisp Voice Security (23 June 2026)** — the closest security-framed product. "Agent Voice
Verification continuously confirms that the person on each agent call matches the enrolled
agent's voice profile, detecting unauthorized substitution in remote and distributed operations."
It **alerts**; it does not gate.
(<https://krisp.ai/blog/krisp-anounces-voice-security-speech-analytics/>,
<https://www.businesswire.com/news/home/20260623168168/en/Krisp-Launches-Voice-Security-and-Speech-Analytics-to-Give-Contact-Centers-Full-Governance-of-the-Voice-Channel>)

### A4. CrystalSound "My Voice Only"

Marketing-grade docs only, but three claims are load-bearing
(<https://www.crystalsound.ai/blog/post/my-voice-only-the-best-crystalsound-feature-you-need-to-try-now>):

- Registration: "click on the start button and speak clearly and loudly", in "a quiet place",
  "no one else speaks during registration". No duration published.
- The pitch is spatial, not biometric: it claims to "create a **virtual boundary of 2 meters
  radius** around your microphone that isolates your voice from any other sounds or speakers
  outside of it." Same physics shortcut as Krisp, dressed as personalisation.
- **The channel warning**: "Use the same microphone for both registering and using your voice, as
  different microphones can affect how our AI algorithm processes your voice."
- Honest failure disclosure: "This feature works in most situations, but there might be some
  cases where you can still hear some faint voices in the background" — i.e. the suppression
  floor is *not* silence, by design.

### A5. Google Personal VAD 1.0 and 2.0 — architecturally the closest thing to TrueMic

**Personal VAD** ([Odyssey 2020](https://www.isca-archive.org/odyssey_2020/ding20_odyssey.pdf),
[arXiv 1908.04284](https://arxiv.org/abs/1908.04284)). A frame-level 3-class classifier: `ns`
(non-speech), `tss` (target speaker speech), `ntss` (non-target speaker speech). Purpose is
identical to TrueMic's gate: "gating the inputs to a streaming on-device speech recognition
system, such that it only triggers for the target user."

Four architectures, and the naming matters because **TrueMic is SC**:

| Arch | What it is | Runtime cost | tss AP (MTR) | mAP (MTR) |
|---|---|---|---|---|
| **SC** score combination | pretrained standard VAD **plus** pretrained speaker verifier, combined at score level: `z_tss = s_t · z_s`, `z_ntss = (1−s_t) · z_s` | 4.88 M (SV) + 0.06 M (VAD) | 0.777 | 0.801 |
| ST score-conditioned | concat frame-level cosine score to the 40-d features, retrain a 41-d-input VAD | 4.88 M + 0.06 M | 0.905 | 0.901 |
| **ET embedding-conditioned** | concat the 256-d d-vector to the 40-d features (296-d input), retrain — **no SV model at runtime** | **0.13 M** | 0.878 | 0.883 |
| SET | both score and embedding | 4.88 M + 0.13 M | 0.938 | 0.928 |
| ET + weighted pairwise loss | ET, but with the loss below | 0.13 M | 0.916 | 0.912 |

Model config: 40-dim log-Mel filterbank energies, **25 ms window / 10 ms step**; **2-layer LSTM
with 64 cells** plus a 64-unit fully connected layer; d-vector is 256-d; quantized to 8-bit
integers → **130 K parameters, 130 KB on disk**.

**Weighted pairwise loss.** Confusing `ns` with `ntss` is harmless (both get discarded), so they
downweight that pair: `w<tss,ns> = w<tss,ntss> = 1`, and sweep `w<ns,ntss>`. Best is **0.1**;
going to 0.05 or 0.01 *degrades* — "it is insufficient to simply treat personal VAD task as a
binary classification problem (target speaker speech v.s. other)."

**Why SC lost, and why that does not condemn TrueMic.** The paper's two stated reasons:

> "First, it is running a window-based speaker verification model at a frame level without any
> adaptation, and such inconsistency could cause significant performance degradation. […] Second,
> this architecture requires running a speaker verification system at runtime, which can be
> expensive."

TrueMic does *not* commit the first sin — `sender.py` scores a **2.5 s window** with a
window-trained model (ECAPA/Eagle) and only decides every 0.5 s, exactly the operating point the
model was trained for. And the second sin (runtime cost) is irrelevant on a laptop. So TrueMic's
SC architecture is the *right lazy choice*; the paper's result is an argument about on-device
phones, not about correctness.

**The false-reject economics, stated by Google in one line:**

> "false rejecting speech leads to deletion errors in ASR transcriptions (**a few milliseconds of
> missed audio could remove an entire word**)."

This is the quantitative core of HANDOFF risk #3. Quote it to judges.

**Personal VAD 2.0** ([Interspeech 2022, arXiv 2204.03793](https://arxiv.org/abs/2204.03793)) —
three upgrades worth knowing:

- **FiLM beats concatenation.** `FiLM(h) = γ(e_target)·h + β(e_target)`; the speaker embedding
  produces per-channel scale and shift rather than being glued onto the input. A "speaker
  pre-net" (2 Conformer layers) extracts frame-level embeddings, cosine-compares them to the
  enrolled embedding, and FiLM-modulates using **the similarity score**.
- **Enrollment-less fallback baked into training**: with probability `p₀ = 0.2` the target
  embedding is replaced by a zero vector and `ntss` labels are rewritten to `tss` — so the same
  network degrades gracefully into a plain VAD when there is no profile. (This is the training-time
  version of Zoom's "downgrade to standard noise removal".)
- Size: 4 Conformer layers, dim 64, 8 heads, causal 7×7 conv, 31 frames left context; features are
  128-d log-Mel with **32 ms window / 10 ms shift**; **2.8 MB unquantized → 1.0 MB at 8-bit**.

**Short enrollment** ([ICASSP 2026, arXiv 2601.12769](https://arxiv.org/abs/2601.12769)):
adaptive speaker-embedding self-augmentation — fuse the enrollment embedding with keyframe
embeddings taken from the *live mixed* speech, refine iteratively; after five iterative updates
performance "matched full-length enrollment performance." This is the online-adaptation idea
Microsoft implements as passive enrollment.

### A6. VoiceFilter-Lite — the over-suppression playbook

([Interspeech 2020](https://www.isca-archive.org/interspeech_2020/wang20z_interspeech.pdf),
[arXiv 2009.04323](https://arxiv.org/abs/2009.04323))

Architecture: frame-by-frame frontend on stacked filterbanks; the d-vector is passed through 1D
CNN layers then **frame-wise concatenated**, then **3 LSTM layers × 512 nodes**, then a fully
connected sigmoid mask layer. 8-bit dynamic-range quantization. Sizes 6.8 MB (FFT magnitude),
5.8 MB (filterbank), **2.2 MB** (final stacked-filterbank adaptive model).

Two mechanisms TrueMic should steal wholesale:

**(1) Asymmetric loss.** They observed that "most of the degradation in Word Error Rate (WER) is
from **false deletions**, which indicates significant over-suppression." So:

```
g_asym(x, α) = x        if x ≤ 0
             = α · x    if x > 0        with α = 10
L_asym = Σ_t Σ_f ( g_asym( S_clean(t,f) − S_enh(t,f), α ) )²
```

Removing target speech is penalised **ten times** as hard as leaving interference in.

**(2) Adaptive suppression strength — a continuously-valued, EMA-smoothed gate depth.**

```
S_out(t) = w · S_enh(t) + (1 − w) · S_in(t)          # w = 0 → filtering off, w = 1 → full
w(t) = β · w(t−1) + (1 − β) · ( a · f_adapt(S_in(t)) + b )
```

`f_adapt ∈ [0,1]` is a second binary head predicting "is this frame overlapped speech", trained
with hinge loss (2 feed-forward layers × 64). `β` is a moving-average coefficient "to make the
suppression more smooth."

Selected WER (%) from Table 1 (LibriSpeech RNN-T; lower is better):

| Config | Clean | Non-speech noise (additive) | Speech noise (additive) |
|---|---|---|---|
| No voice filtering | 8.6 | 35.7 | 77.9 |
| Filterbank, plain L2, w=1 | 9.3 | 23.4 | 25.4 |
| Filterbank, asym L2 α=10, w=1 | **8.6** | 24.8 | 30.6 |

Read the trade honestly: the asymmetric loss buys back all of the clean-condition damage (9.3 →
8.6) at the cost of some interference-suppression (25.4 → 30.6). **That is the exact dial
TrueMic is choosing to turn the other way.** In Table 2 (realistic queries) the fixed `w = 0.3`
setting is worse on speech noise (42.1) than adaptive `w(t)` (28.8) while both preserve clean — a
direct argument for making TrueMic's gate depth *dynamic* rather than a fixed compromise.

### A7. E3Net / pDCCRN — the real-time personalized enhancement stack

([Interspeech 2022, arXiv 2204.00771](https://arxiv.org/abs/2204.00771))

- Learnable 1-D conv encoder/decoder with filter and stride equal to **20 ms window / 10 ms hop**.
- Speaker embedding (**256-d**) is **concatenated with encoder features, then projected through a
  linear layer + non-linearity** before the LSTM stack. Encoder/decoder feature dim 2048, LSTM
  hidden 256, FC hidden 1024.
- Sizes and speed (single-thread Intel Xeon @ 3.60 GHz):

| Model | Params | RTF |
|---|---|---|
| E3Net baseline (N=4 LSTM blocks) | 6.61 M | 0.065 |
| E3Net teacher (N=8) | 10.85 M | 0.122 |
| E3Net student (N=2, distilled) | 4.50 M | 0.036 |

- **TSOS** (target speaker over-suppression) "estimates the removed target speaker segments" and
  is "normalized to show the target-speaker-suppressed period in **seconds per half an hour**."
  Microsoft ships a *unit* for eaten words. TrueMic should adopt the same unit for its demo notes.

### A8. TS-VAD — the post-processing recipe everyone reuses

([arXiv 2005.07272](https://arxiv.org/pdf/2005.07272)) Target-speaker VAD for diarization,
i-vector conditioned. Its value here is one paragraph, the industry-standard smoothing recipe for
turning noisy per-frame speaker scores into segments:

> "51-tap median filtering, binarization with the threshold of 0.4, combining speech segments
> separated by pauses shorter than **0.3 s**, and deleting speech segments shorter than **0.2 s**."

Table 3 shows it is worth real accuracy: fusion with threshold-only = 34.73 / 37.52 DER;
threshold + median filter + short-segment processing = 33.56 / 36.63. Viterbi decoding over an
11-state HMM does slightly better still (32.84 / 36.02) — the cheap version (median + minimum
durations) captures most of the gain.

---

## Part B — Parameter tables

### B1. Enrollment across the field

| System | Duration | Content | Conditions | Refresh |
|---|---|---|---|---|
| Teams (manual) | ~30 s | read provided text, 25 localized languages | "relatively quiet environment", user picks the mic | local encrypted copy **expires every 14 days** and is replaced |
| Teams (passive/express) | "a couple of meetings" | spontaneous in-meeting speech | inherently multi-condition | continuous |
| Zoom (auto) | unspecified, from first meeting | spontaneous | headset assumed | continuous |
| Zoom (enhanced recording) | **≥ 30 s** | Zoom-provided script, "as you normally would in terms of tone, speed, and loudness" | reduce background noise | manual |
| DNS 2021 pDNS | 2 min | read | clean | n/a |
| DNS 2022 PDNS | 2.5 min | read | clean (DNSMOS OVRL ≥ 4.25) | n/a |
| DNS 2023 | **10–30 s** | read | **noisy/clean × reverberant/dry, deliberately** | n/a |
| CrystalSound | unspecified | "speak clearly and loudly" | quiet, **same mic as use** | manual |
| Krisp BVC | **none** | — | headset required | n/a |
| **TrueMic today** | 40 s (`enroll.py --seconds 40`), 3 s VAD-filtered chunks, mean of L2-normalised ECAPA embeddings | free reading | single condition, single mic | none |

TrueMic's 40 s sits comfortably above the field's 30 s floor. The gap is **conditions**, not duration.

### B2. Model size, latency and compute

| System | Model size | Latency budget | Notes |
|---|---|---|---|
| Personal VAD (ET) | 130 K params / **130 KB** | frame-level, 10 ms step | no SV model at runtime |
| Personal VAD 2.0 | 2.8 MB → **1.0 MB** (8-bit) | 10 ms shift | Conformer, causal |
| VoiceFilter-Lite | **2.2 MB** (8-bit) | frame-level | + noise-type head |
| E3Net student | 4.50 M params | 20 ms / 10 ms | RTF 0.036 |
| Krisp BVC pro | **14.7 MB** | 10 ms frames | 5–13 % CPU |
| DNS Challenge rule | — | **≤ 40 ms** total algorithmic latency (frame + stride + lookahead), frame processed in < stride on i5 @ 2.4 GHz | 2023 tightened further |
| **TrueMic today** | ECAPA ~20 MB + silero ~1 MB + AudioSeal | **~1.5 s gate delay + ~75–95 ms path** | — |

Context for TrueMic's 1.5 s: ITU-T G.114 says one-way delay below **150 ms** gives "essentially
transparent interactivity", **150–400 ms** is acceptable with awareness of the impact, and above
**400 ms** is unacceptable for general network planning
(<http://www.cs.columbia.edu/~andreaf/new/documents/other/T-REC-G.114-200305.pdf>). TrueMic's
outbound path is ~4× the "unacceptable" line. This is a *deliberate* trade — quality features must
be transparent, a security gate may spend latency — but it is a real conversational cost and the
demo script should acknowledge it before a judge does.

### B3. Smoothing and gating parameters actually used in the field

| Parameter | Value | Source |
|---|---|---|
| Median filter on frame scores | **51 taps**, then binarize at 0.4 | TS-VAD |
| Merge pauses shorter than | **0.3 s** | TS-VAD |
| Delete speech segments shorter than | **0.2 s** | TS-VAD |
| Silero VAD speech threshold | 0.5 | [`utils_vad.py`](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py) |
| Silero **hysteresis** (close threshold) | `neg_threshold = max(threshold − 0.15, 0.01)` — asymmetric by construction | same |
| Silero min silence before closing | **100 ms** | same |
| Silero speech padding each side | **30 ms** | same |
| Silero min speech duration | **250 ms** | same |
| Suppression-depth smoothing | EMA: `w(t) = β·w(t−1) + (1−β)·(…)` | VoiceFilter-Lite |
| Over- vs under-suppression penalty | **10 : 1** | VoiceFilter-Lite α=10 |
| Non-target/non-speech confusion weight | **0.1** (not 0, not 1) | Personal VAD WPL |

### B4. Gate shaping (audio-engineering practice, not ML)

Standard noise-gate practice for speech, for calibrating attack/hold/release/floor
(<https://audioedit.io/blog/audio-noise-gate-explained>,
<https://www.prosoundweb.com/audio-concepts-the-usage-of-gates-explained/>,
<https://www.mathworks.com/help/audio/ref/noisegate.html>):

| Parameter | Typical speech setting | Purpose |
|---|---|---|
| Attack (open) | 10–15 ms | soft start, no click |
| Hold | 30–50 ms | stops the gate chattering between words |
| Release (close) | 50–100 ms | smooth, not abrupt |
| Floor / range | −12 to −15 dB for natural speech; −20 to −30 dB for noisy locations; −60 to −80 dB for clean studio | "keeps a tiny amount of room tone alive" |

Nobody in audio engineering closes a speech gate to true silence, because dead silence reads as a
dropped call. **TrueMic cannot follow this advice directly** — see §D3.

---

## Part C — Channel mismatch

### C1. What the field does

1. **Constrain the channel.** CrystalSound: same mic for registration and use. Krisp BVC: headset,
   best with a wired USB boom mic; the 16 kHz lite model is "USB/jack headsets only". Zoom:
   headset assumed, downgrade if not. This is the cheapest and most effective mitigation and every
   shipped product uses it.
2. **Multi-condition enrollment.** DNS 2023 ships enrollment "noise-free or noisy and with or
   without reverberation … which serves as a measure of robustness". DNS 2022 states the synthesis
   recipe: take clean enrollment and *generate* reverberant/noisy variants for adaptation.
3. **Continuous re-enrollment.** Teams passive enrollment updates across meetings; the on-device
   copy is force-refreshed every 14 days. Personal-VAD-with-short-enrollment
   ([arXiv 2601.12769](https://arxiv.org/abs/2601.12769)) does the online version — fuse keyframe
   embeddings from the live stream into the enrollment embedding, iterate; five updates match
   full-length enrollment.
4. **Score normalisation.** AS-norm (adaptive symmetric score normalisation) normalises a trial
   score using the scores of an impostor cohort selected to be close to the input, specifically to
   "minimize the domain mismatch (e.g., languages, recording environments, etc.)". Recent work
   makes it trainable (TAS-norm,
   [arXiv 2504.04512](https://arxiv.org/abs/2504.04512)) or moves it to embedding level
   ([Cumani, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/cumani23_interspeech.html)).
   The important property for TrueMic: **AS-norm makes an absolute threshold portable across
   channels**, because it removes the per-utterance score offset that a different mic introduces.

### C2. Direct evidence that enrollment augmentation works

"On the effectiveness of enrollment speech augmentation for Target Speaker Extraction"
([arXiv 2409.09589](https://arxiv.org/pdf/2409.09589)) augments the *enrollment* side rather than
the mixture side:

> "We found that for both pretrained and jointly optimized speaker encoders, directly augmenting
> the enrollment speech leads to consistent performance improvement."

Augmentations tested: additive noise with **SNR ~ U(−5, 15) dB**, reverberation with a sampled
T60, SpecAugment, plus their proposed self-estimated speech augmentation. Reported improvement up
to **2.5 dB** SI-SDR on Libri2Mix. Note the SNR range goes *negative* — they augment aggressively.

### C3. Duration mismatch is its own channel problem

Short test segments hurt ECAPA measurably: EER degraded from **8.72 % to 12.8 % (+46 % relative)**
when evaluation duration dropped from 3.59 s to 2.05 s
(<https://www.researchgate.net/publication/371288593_Short-Segment_Speaker_Verification_Using_ECAPA-TDNN_with_Multi-Resolution_Encoder>).
TrueMic scores 2.5 s windows against a 40 s enrollment — a 16× duration asymmetry. This is
precisely why the on-site calibration step exists and why the threshold must never be a constant.

---

## Part D — What TrueMic should adopt

Mapping to the current code: `truemic/config.py`, `truemic/gate.py` (`DelayGate`,
`GateDecision`), `truemic/sender.py` (decision loop), `truemic/calibrate.py`, `truemic/enroll.py`.

### D1. Make the security posture an explicit, named asymmetry

Every shipped system tunes 10:1 *against* over-suppression (VoiceFilter-Lite α=10). TrueMic tunes
the other way. Fine — but encode it, don't leave it implicit:

- **Open** requires the reliable evidence: score ≥ threshold on the full 2.5 s window, for
  **2 consecutive decisions** (1.0 s of agreement).
- **Close** requires only weak evidence: a single decision below `threshold − margin`, or a short
  1.0 s sub-window below threshold.

This is the direct security-inverted analogue of the asymmetric loss, expressed as attack/release
rather than as a training objective. Ship it as `OPEN_CONFIRMATIONS = 2`, `CLOSE_CONFIRMATIONS = 1`
so the product posture is one constant away.

### D2. Close the ~0.5–1.0 s impostor leak with an asymmetric decision window

Current timing: window 2.5 s, decide every 0.5 s, gate delay 1.5 s. Trace an impostor starting at
t=0. The gate emits content from t−1.5 s. The scoring window only becomes *pure* impostor at
wall-clock 2.5 s, by which time content up to t = 1.0 s has already been emitted. In practice
mixed windows drop below threshold earlier (~2.0 s wall clock → ~0.5 s of leaked content), but
that is still half a second of impostor audio *with a valid watermark on it*. A judge replaying
the recording will find it.

Two fixes, take the second:

- Raise `GATE_DELAY_SECONDS` to 2.5 s (= `WINDOW_SECONDS`). Correct, costs another second of
  conversational latency on top of an already-unacceptable 1.5 s.
- **Add a short-window early-close**: score the last **1.0 s** of the ring as well, and close the
  gate if *either* the 2.5 s score or the 1.0 s score falls below threshold. HANDOFF is right that
  1 s windows are unreliable — but they are unreliable in the false-*reject* direction, which for
  a fail-closed gate is the safe direction. Opening still requires the reliable 2.5 s window.
  Costs one extra ECAPA forward pass per decision.

Note the current 1.5 s delay is otherwise well chosen: it puts the emitted sample near the
*centre* of the scoring window, so every gated sample is judged with both past and future context.
Keep that property (delay ≈ `WINDOW_SECONDS/2 + DECIDE_EVERY_SECONDS` = 1.75 s) if the delay is
retuned.

### D3. Keep the hard mute — and prove it, because attenuation would leak the token

Field practice says never gate speech to true silence (floor −12 to −30 dB). TrueMic must
ignore that advice, and the reason is watermark-specific: AudioSeal's training augmentations
include amplitude boosting/scaling among the "range of signal alterations"
([arXiv 2401.17264](https://arxiv.org/abs/2401.17264)), i.e. the detector is *designed* to survive
uniform gain changes. Attenuating watermarked audio by −40 dB scales signal and watermark
together, leaving the watermark-to-signal ratio unchanged — a receiver would very likely still
read a valid token off "muted" impostor audio. **A soft floor silently breaks the security
property.** Multiply by exactly zero.

To recover the naturalness that the floor was buying, add **unwatermarked comfort noise after the
gate**, in `sender.py` right before `out_queue.put`:

```python
out = gated + comfort * (1.0 - gate_gain)      # comfort ≈ -55 dBFS pink/white noise
```

Generated locally, post-gate, so it carries no token. Closed gate → quiet room tone, not a dead
line; open gate → comfort noise fades out.

**Runnable check to write** (this is the one non-trivial claim in the design):
watermark 2 s of speech, scale it by 0.01, run `WatermarkReader.read` and assert the detector
either still returns the token (proving mute is mandatory) or does not (in which case the floor
becomes available). Fifteen lines in `tests/`. Do not guess this one.

### D4. Smooth the score before thresholding

`GateDecision.update` currently thresholds a single raw score. Every system in Part A smooths:
TS-VAD uses a 51-tap median filter, VoiceFilter-Lite an EMA on suppression depth. TrueMic decides
only twice a second, so the analogue is tiny:

```python
scores = deque(maxlen=3)            # 1.5 s of decisions
smoothed = median(scores)
```

Median (not mean) — it is what TS-VAD uses and it kills a single anomalous window without dragging
the estimate the way a mean does. Apply it to the *open* path only; the close path should react to
the raw score (per D1).

### D5. Derive both thresholds from calibration, not from one threshold plus a guessed margin

`config.HYSTERESIS_MARGIN = 0.08` is a guess. Silero's own precedent is an asymmetric close
threshold of `threshold − 0.15` on a probability scale — different scale, same idea, larger
margin. `calibrate.py` already measures the two distributions it needs. Have it write both:

- `open_threshold` = midpoint of genuine p5 and impostor p95 (what it computes today), or
  `genuine_p5 − 0.05` without an impostor.
- `close_threshold` = `impostor_p95 + 0.02` when an impostor was recorded, else
  `open_threshold − 0.10`.

Now the hysteresis band is *measured on the demo mic in the demo room* rather than assumed.
Also print the observed genuine p5 − impostor p95 gap: if it is under ~0.05, the gate cannot
separate these two voices on this mic and the demo needs a different teammate or a different mic —
better to learn that at setup than on stage.

### D6. Multi-condition the enrollment embedding (15 lines, biggest robustness win per line)

Microsoft's own stated recipe: "clean speech can be used for generating reverberant and noisy
multi-condition enrollment data for speaker adaptation" (DNS 2022), and DNS 2023 makes noisy +
reverberant enrollment a scored robustness property. The TSE augmentation paper confirms
consistent gains with noise at SNR ~ U(−5, 15) dB plus sampled-T60 reverberation.

In `enroll.py`, after collecting `voiced_chunks`, embed three versions of each chunk and average
all of them:

1. the chunk as recorded,
2. the chunk + white/babble noise at ~10 dB SNR,
3. the chunk convolved with a short synthetic exponential-decay RIR (T60 ≈ 0.3 s).

Then L2-normalise the mean as today. No new dependency — `numpy` does all three. This directly
attacks the CrystalSound failure mode ("different microphones affect how the algorithm processes
your voice") and the room change between the calibration room and the demo stage.

### D7. Keep enrollment ≥ 30 s, and keep the VAD gate on the enrollment path

Already correct: 40 s recording, 3 s chunks kept only if `voiced_fraction ≥ 0.5`. The citation for
why VAD-before-embedding is mandatory is DNS 2022: "speaker models are sensitive to false-alarms
in speech activity detection; clean speech can be used for obtaining accurate SAD labels resulting
in speaker-discriminative embeddings." Add one detail from DNS 2023's enrollment prep: strip
pauses longer than **0.2 s**. Also copy Zoom's instruction verbatim into the enroll prompt —
"speak as you normally would in terms of tone, speed, and loudness" — because enrollment prosody
mismatch is a free source of false rejects.

### D8. Adopt TSOS as the demo's honesty metric

E3Net measures over-suppression in **seconds of target speaker removed per half hour**. Instrument
`sender.py` to count gated-while-voiced samples and print the same unit at shutdown. One counter.
It turns HANDOFF risk #3 from a hand-wave into a number you can put on a slide: "in 30 minutes of
my own speech the gate ate N seconds." If N is large, you learn it in rehearsal.

### D9. Ship a "downgrade" mode instead of a bad gate (product framing, near-zero code)

Zoom's behaviour — detect a non-headset mic, silently fall back to standard noise removal — and
Personal VAD 2.0's zero-embedding training path are the same idea: when confidence in the channel
is low, degrade to a weaker mode rather than misfire. TrueMic's analogue: when `calibrate.py`
reports overlapping genuine/impostor distributions, run in **watermark-only mode** (no gating, the
receiver badge reflects verification score rather than gate state). One flag, and it converts a
demo-day disaster into a graceful story.

### D10. Score normalisation — worth it only if the threshold proves unstable

If the demo has to run on a mic that wasn't the calibration mic, one-sided AS-norm is the standard
fix and costs almost nothing: keep ~20 impostor embeddings on disk, and per decision compute
`s_norm = (s − mean(cohort_scores)) / std(cohort_scores)` where `cohort_scores` are the live
window's similarities to the cohort. This cancels the per-window offset a different channel
introduces, making the calibrated threshold portable. Twenty extra dot products per decision.
**Do not build this pre-emptively** — build it only if switching mics visibly moves the score
distribution during rehearsal.

---

## Part E — What NOT to copy

- **Do not chase a fused personal-VAD model.** ET is 130 KB and better than TrueMic's
  architecture, but there are no public pretrained personal-VAD weights, the models are trained
  against specific d-vector extractors, and Personal VAD 2.0's Conformer is Google-internal.
  HANDOFF's cut is correct.
- **Do not adopt a soft attenuation floor.** See D3 — it breaks the watermark semantics.
- **Do not copy the headset/proximity shortcut.** Krisp and CrystalSound solve "whose voice is
  this" with near-field physics. It is cheaper and more robust than biometrics — and it
  categorically cannot pass TrueMic's demo beat #2 (teammate on the *same* mic). That the physics
  shortcut fails there is the argument for biometric gating; say it explicitly.
- **Do not claim latency parity.** The DNS challenges cap total algorithmic latency at 40 ms
  (20 ms in 2023). TrueMic spends 1.5 s. Frame it as a category difference (transparency feature
  vs. security control), not as a shortcoming to hide.
- **Do not treat the gate as binary at the product level.** Everyone who shipped this concluded
  the opposite. Fail-closed is right for a 4-minute demo; "tunable posture" is the honest answer
  when a judge asks what happens when it eats 0.5 % of a CEO's words.

---

## Sources

Microsoft Teams / Zoom / vendors
- <https://learn.microsoft.com/en-us/microsoftteams/voice-isolation>
- <https://learn.microsoft.com/en-us/microsoftteams/rooms/voice-and-face-recognition>
- <https://support.microsoft.com/office/voice-isolation-in-microsoft-teams-calls-and-meetings-a9756ea9-4cec-44c4-aefb-6f5d17c89427>
- <https://techcommunity.microsoft.com/blog/microsoftteamsblog/voice-isolation-in-microsoft-teams-enables-personalized-noise-suppression-for-ca/4096077>
- <https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0074698>
- <https://sdk-docs.krisp.ai/docs/krisp-rtc-bvc-nc>
- <https://sdk-docs.krisp.ai/docs/rtc-model-guide-bvc-nc>
- <https://sdk-docs.krisp.ai/docs/models-for-conversational-ai>
- <https://krisp.ai/blog/krisp-anounces-voice-security-speech-analytics/>
- <https://www.businesswire.com/news/home/20260623168168/en/Krisp-Launches-Voice-Security-and-Speech-Analytics-to-Give-Contact-Centers-Full-Governance-of-the-Voice-Channel>
- <https://www.crystalsound.ai/blog/post/my-voice-only-the-best-crystalsound-feature-you-need-to-try-now>

Papers
- Personal VAD — <https://www.isca-archive.org/odyssey_2020/ding20_odyssey.pdf> / <https://arxiv.org/abs/1908.04284> / <https://google.github.io/speaker-id/publications/PersonalVAD/>
- Personal VAD 2.0 — <https://arxiv.org/abs/2204.03793>
- Personal VAD with short enrollment (ICASSP 2026) — <https://arxiv.org/abs/2601.12769>
- VoiceFilter-Lite — <https://www.isca-archive.org/interspeech_2020/wang20z_interspeech.pdf> / <https://arxiv.org/abs/2009.04323>
- E3Net — <https://arxiv.org/abs/2204.00771>
- ICASSP 2021 DNS Challenge — <https://ar5iv.labs.arxiv.org/html/2009.06122>
- ICASSP 2022 DNS Challenge — <https://arxiv.org/pdf/2202.13288>
- ICASSP 2023 DNS Challenge — <https://arxiv.org/pdf/2303.11510>
- TS-VAD — <https://arxiv.org/pdf/2005.07272>
- Enrollment speech augmentation for TSE — <https://arxiv.org/pdf/2409.09589>
- PSE without a separate speaker embedding model — <https://www.isca-archive.org/interspeech_2024/parnamaa24_interspeech.pdf>
- Trainable adaptive score normalisation — <https://arxiv.org/abs/2504.04512>
- Adaptive score → adaptive data normalisation — <https://www.isca-archive.org/interspeech_2023/cumani23_interspeech.html>
- Short-segment ECAPA verification — <https://www.researchgate.net/publication/371288593_Short-Segment_Speaker_Verification_Using_ECAPA-TDNN_with_Multi-Resolution_Encoder>
- AudioSeal — <https://arxiv.org/abs/2401.17264>

Engineering references
- Silero VAD defaults and hysteresis — <https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py>
- Noise gate practice — <https://audioedit.io/blog/audio-noise-gate-explained>, <https://www.prosoundweb.com/audio-concepts-the-usage-of-gates-explained/>, <https://www.mathworks.com/help/audio/ref/noisegate.html>
- ITU-T G.114 one-way transmission time — <http://www.cs.columbia.edu/~andreaf/new/documents/other/T-REC-G.114-200305.pdf>
