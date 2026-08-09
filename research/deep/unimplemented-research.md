# Published-but-never-productized research TrueMic could ship

Scope: techniques validated in peer-reviewed papers / serious theses, absent from every shipping
product we surveyed, and implementable inside our existing pipeline (`truemic/`) in hours-to-days.

**Hard feasibility filter applied to everything below.** Rejected on sight: training a model from
scratch, GPU at inference, custom or non-PC hardware, >1 s added latency. Fitting a 97-feature SVM
on a hundred locally recorded clips is *not* "training from scratch" and stays in scope; retraining
a CNN+BiLSTM on 5 hours of annotated podcasts is out.

Every claim below carries a URL. Where a number is quoted it is the paper's own number, on the
paper's own data, and the caveat about how far it transfers is stated.

---

## 0. Where these plug into the code

For reference while reading the sketches:

| File | What it already does | Hook we would use |
|---|---|---|
| `truemic/verify.py` | `VoiceActivity.voiced_fraction()`, `EcapaVerifier.score()` | non-speech frames are already isolated; score is a single cosine |
| `truemic/gate.py` | `GateDecision.update(score, now)` | one place where score → allow/deny |
| `truemic/tokens.py` | `token_for_slot(key, slot)` = HMAC(key, slot)[:2] | payload derivation, 16 bits |
| `truemic/calibrate.py` | on-site 30 s threshold calibration | where any fitted coefficient would be learned |
| `truemic/receiver.py`, `badge.html` | badge rendering + staleness | UX findings land here |

---

## 1. Microphone-identity check on near-silence frames → software injection detection

**What it does.** Every microphone has a device-specific noise floor and frequency response arising
from manufacturing tolerances. Fingerprinting a recording's *silent* segments identifies which
physical microphone produced it. Applied to TrueMic: audio that did not come out of the enrolled
microphone — e.g. an ElevenLabs clone piped straight into our input via a second virtual cable —
carries the wrong noise floor and can be rejected before verification even runs.

**Evidence.**
- Buchholz, Kraetzer, Dittmann, *Microphone Classification Using Fourier Coefficients*, Information
  Hiding 2009. Feature = Fourier-coefficient histogram of **near-silence segments**; up to **93.5%
  correct classification** over 7 microphones with a 2048-point FFT and linear logistic regression.
  https://link.springer.com/chapter/10.1007/978-3-642-04431-1_17 ·
  PDF: https://omen.cs.uni-magdeburg.de/itiamsl/cms/upload/Publikationen/ckraetzer/IH09_FINAL.pdf
- Entropy-MFCC over near-silent segments as an intrinsic device-response feature reaches **~99.9%**
  identification accuracy across computer devices for both call-recording and microphone-recording
  scenarios (survey coverage in *Audio Recording Device Identification Based on Deep Learning*,
  arXiv:1602.05682). https://arxiv.org/pdf/1602.05682
- One-class variant (no impostor devices needed — exactly our situation, one enrolled mic):
  *Identifying Microphone from Noisy Recordings by Using Representative Instance One
  Class-Classification*. https://www.researchgate.net/publication/236858988
- Smartphone analogue, CCS 2014: *Do You Hear What I Hear? Fingerprinting Smart Devices Through
  Embedded Acoustic Components* — subtle fabrication imperfections in mics/speakers uniquely
  identify a device. https://dl.acm.org/doi/10.1145/2660267.2660325 ·
  PDF: https://www.cs.cmu.edu/~anupamd/paper/CCS2014.pdf
- MicPrint (MobiQuitous 2019) makes the same fingerprint deliberately spoof-resistant.
  https://dl.acm.org/doi/10.1145/3360774.3360801

**Why it is unproductized.** This entire literature is framed as *forensics* (which phone recorded
this evidence file?) or as a *privacy threat* (tracking users across sites). Nobody has inverted it
into a live consent check — "is this stream still coming from the microphone I enrolled?" That is
precisely the gap HANDOFF risk #2 names, and it is the only one of our risks that has a cheap
software answer.

**Free bonus tell.** Virtual Audio Cable's own manual states a fully virtual device produces "pure
digital silence containing zero samples only" when nothing is connected, and that shared-mode WASAPI
adds only dithering noise. https://vac.muzychenko.net/en/manual/troubles.htm — an injected stream's
pauses are often *exactly zero* or absurdly clean. A three-line check (`np.all(frame == 0)` or noise
floor below the enrolled floor by >15 dB) catches naive injection before any spectrum work.

**Implementation sketch (our stack).**
1. `enroll.py`: after the voice capture, record 5 s of room silence. Take frames where
   `VoiceActivity` says unvoiced, compute mean log-magnitude spectrum (`np.abs(np.fft.rfft(frame *
   np.hanning(2048)))`, 2048-point, averaged), store as `Profile.mic_fingerprint` plus a per-bin
   standard deviation and the mean broadband level in dB.
2. `verify.py`: a `MicFingerprint` class with `.match(window) -> float` — same spectrum over the
   window's unvoiced frames, cosine similarity against the enrolled fingerprint, plus the
   digital-silence / level checks.
3. `sender.py`: feed it into the existing decide loop; `GateDecision` gains one more AND-condition
   (or, safer, it becomes a term that raises the ASV threshold — see §9).
4. `calibrate.py`: report the live match score during the 30 s calibration so the threshold is set
   on the real mic in the real room.

**Effort:** 1.5–2 h. Pure numpy, no new dependency, no added latency (runs on frames the VAD already
rejected).

**Demo impact: HIGH.** It buys a fifth demo beat that no competitor can answer: route the clone
directly into TrueMic's input as a software device — the strongest possible attack, since it skips
the room entirely — and the badge stays grey *because the microphone is wrong*, not because a
detector guessed. It also inverts our weakest risk into a feature.

**Honest limits (say these before a judge does).** (a) It authenticates the *channel*, not liveness:
an attacker who plays the clone through a speaker at the enrolled mic still passes this check (that
is §2's job). (b) User-space can lie to user-space — a sufficiently determined attacker can replay
recorded room noise under the clone. The strong version of this lives at silicon (Cirrus Logic
patents), same as in HANDOFF. (c) It is per-mic: swapping headsets requires re-enrolling the
fingerprint. Make that an explicit "re-calibrate" action, not a silent failure.

---

## 2. Void-style loudspeaker/replay detection — 97 spectral features, no deep model

**What it does.** Distinguishes a live human voice at the mic from one replayed through *any*
loudspeaker, using only cumulative-power structure in the spectrogram. Two physical facts do the
work: loudspeakers add distortion that flattens and linearises power distribution across the audible
range; and live human voice puts disproportionately more power in low frequencies than replayed
voice.

**Evidence.** Ahmed, Kwak, Huh, Kim, Oh, Kim, *Void: A fast and light voice liveness detection
system*, **USENIX Security 2020**.
https://www.usenix.org/conference/usenixsecurity20/presentation/ahmed-muhammad ·
PDF: https://www.usenix.org/system/files/sec20-ahmed-muhammad_0.pdf
- **0.3% EER** on their own replay dataset, **11.6% EER** on ASVspoof 2017 v2.
- vs. a deep-learning baseline at 7.4% EER: **153× less memory, ~8× faster**.
- Detection rates against adjacent attacks: hidden voice command 99.7%, inaudible voice command
  100%, voice synthesis 90.2%, EQ manipulation 86.3%, replay-mixed-with-live 98.2%.
- ACM DL record: https://dl.acm.org/doi/10.5555/3489212.3489363

**Exact feature recipe** (recovered from the open reproduction, `chislab/void-voice-liveness-detection`,
https://github.com/chislab/void-voice-liveness-detection):

| Group | Dim | Computation |
|---|---|---|
| `FV_LDF` linearity-degree | 2 | quadratic polynomial fit over the normalised cumulative power curve; the fit coefficients *are* the features |
| `FV_HPF` high-power-frequency | 35 | peak selection over `FV_LFP` with threshold ω = 0.3 |
| `FV_LPC` LPCC | 12 | 12 linear-prediction cepstrum coefficients |
| `FV_LFP` low-frequency power | 48 | power spectrum split into segments, normalised, first 48 taken ×100 |

STFT: `nperseg=512`, `noverlap=384`, `nfft=2048`, `fs=16000` — **already our sample rate**.
Classifier: SVM with RBF kernel. Total 97 dims. (The reproduction notes its own EER is worse than the
paper's, so treat 0.3% as the authors' ceiling, not ours.)

**Why it is unproductized.** Void was a Samsung Research / Data61 collaboration aimed at smart
speakers; the productised descendants are all *inbound* deepfake detectors. Nobody runs a
loudspeaker-detector on the *sender* side to decide whether their own mic should be trusted.

**Implementation sketch.** `truemic/liveness.py`, ~80 lines of numpy/scipy: `scipy.signal.stft` with
the parameters above, the four feature groups, then `sklearn.svm.SVC(kernel='rbf', probability=True)`.
Training data is collected on-site in ten minutes: ~2 min of live speech + ~2 min of the same speech
replayed through a laptop/phone speaker into the same mic, sliced into 2.5 s windows — a few hundred
samples, which is comfortably enough for a 97-dim RBF SVM. **Laziest useful subset:** ship only
`FV_LDF` (2 numbers: the quadratic-fit coefficients of the cumulative power curve) with a threshold
from those same recordings — that is the paper's core physical insight and needs no classifier at
all. Start there; add the full 97 only if the 2-feature version is ambiguous.

**Effort:** 45 min for the `FV_LDF` threshold version; 3–4 h for the full 97-feature SVM including
on-site data collection.

**Demo impact: MEDIUM-HIGH**, with a *direct conflict to resolve first.*

> ⚠️ **This collides with demo beat 4.** HANDOFF beat 4 (air gap) deliberately holds a phone speaker
> playing verified audio to the laptop mic and expects the badge to still read. Void's entire job is
> to reject speaker-played audio. Pick one: either drop beat 4 and pitch replay-resistance instead
> (stronger security story, weaker "the signal lives in the audio" story), or ship Void behind a
> posture flag that is off during beat 4 and on during a dedicated "replay attack" beat. Do not ship
> it silently — a judge who notices the contradiction lands a much worse hit than a judge who watches
> you flip the switch on purpose.

---

## 3. Challenge–response nonce in the watermark payload (replaces pure time-slot HMAC)

**What it does.** Turns our freshness argument from "the token matches a recent 4 s slot" into "the
token answers a challenge this receiver issued seconds ago". A recording of verified audio then
fails not after a slot expires but the moment the session's nonce rotates, and cross-session replay
becomes structurally impossible rather than time-bounded.

**Evidence.** CallShield, *Secure Caller Authentication over Real-Time Audio Channels*,
arXiv:2601.09327 — https://arxiv.org/pdf/2601.09327 — "achieves replay resistance by computing a MAC
over a fresh challenge, ensuring that responses from previous calls are invalid... pre-recorded or
synthesized audio without access to the session's live challenge and key cannot pass verification",
using HMAC-SHA256 over a session-specific challenge. HANDOFF already names CallShield as closest
prior art; this is the one part of it we should simply adopt rather than differentiate from.

**Why it is unproductized.** CallShield is a January 2026 preprint with a 63 s handshake, no
biometric gate, no receiver UI, and no product. Nobody has shipped audio-channel challenge–response
at all. Our 16-bit AudioSeal payload plus an out-of-band HTTPS verifier lets us do the freshness
half in a fraction of CallShield's handshake budget, because we do not need to carry the whole MAC
in-band — only enough of it to bind the audio to the challenge.

**Implementation sketch.** Minimal diff to `truemic/tokens.py`:
```python
def token_for_slot(key: bytes, slot: int, nonce: bytes = b"") -> int:
    digest = hmac.new(key, nonce + slot.to_bytes(8, "big"), hashlib.sha256).digest()
    return int.from_bytes(digest[:2], "big")
```
`receiver.py` mints a random 16-byte nonce per badge session and exposes it; `sender.py` fetches it
once at start and on rotation (say every 30 s) and passes it through. Slots stay exactly as they
are, so `TOKEN_TOLERANCE` and the ±1-slot logic are untouched.

**Effort:** ~1 h, including the receiver endpoint. Zero added audio latency.

**Demo impact: HIGH.** It converts HANDOFF risk #4 ("watermarks are copyable") from a hand-wave into
a demonstrable beat: record 20 s of verified TrueMic output, replay it into a fresh receiver session,
badge stays grey. Also the cleanest possible answer to "how is this different from CallShield?" —
*we took their freshness and added the biometric gate they don't have.*

**Honest limit.** 16 bits is a token, not a proof — an attacker who can observe the current nonce and
has the key could forge; without the key, guessing succeeds at ~1/65536 per attempt (and ~0.08% with
our tolerance-1 / 1-bit-error slack, as `tokens.py` already documents). The Ed25519 attestation
remains where the cryptography actually lives.

---

## 4. Quality-aware score calibration (QMF): make the threshold a function of window quality

**What it does.** Replaces our single fixed cosine threshold with a threshold that adapts to how much
usable speech the window actually contained. Short, quiet, or half-voiced windows produce
systematically depressed similarity scores; a fixed threshold therefore mutes the genuine speaker
exactly at sentence starts and ends. QMF calibration corrects that with two fitted coefficients.

**Evidence.**
- Thienpondt, Desplanques, Demuynck, *The IDLAB VoxSRC-20 Submission: Large Margin Fine-Tuning and
  Quality-Aware Score Calibration in DNN Based Speaker Verification*, ICASSP 2021 — arXiv:2010.11255.
  https://arxiv.org/abs/2010.11255 — duration QMF alone gives **~6% relative EER / 2% MinDCF**
  improvement on average; duration + impostor-mean QMFs together **~11% EER / 3% MinDCF**. Embedding
  magnitude is reported as the *weakest* quality metric (don't bother with it).
- Ferrer, Nandwana, McLaren et al., *Quality measures based calibration with duration and noise
  dependency for speaker recognition*, Speech Communication 2015.
  https://www.sciencedirect.com/science/article/abs/pii/S0167639315000643
- *Quality measures for speaker verification with short utterances*, Digital Signal Processing 2019.
  https://www.sciencedirect.com/science/article/abs/pii/S1051200418304287
- Survey framing: Poddar, Sahidullah, Saha, *Speaker verification with short utterances: a review*,
  IET Biometrics 2018. https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-bmt.2017.0065

**Why it is unproductized.** QMF calibration is standard in NIST/VoxSRC challenge systems and absent
from every consumer voice product, because consumer products run on long enrollment-vs-utterance
comparisons where duration variance is small. Our 2.5 s rolling window with variable voiced fraction
is *exactly* the regime QMF was invented for, and nobody is running speaker verification in that
regime commercially.

**Implementation sketch.** We already compute `voiced` in `sender.py`'s decide loop and already have
a calibration script. Calibrated score:

```
s_cal = s_raw + a * (voiced_fraction - VOICED_REF) + b * (snr_db - SNR_REF)
```

`snr_db` comes free from frame energies the VAD already separates (voiced-frame RMS over
unvoiced-frame RMS). Fit `a, b` by logistic regression during `calibrate.py`'s 30 s capture, using
the genuine speaker's windows as positives and a handful of shipped cohort clips as negatives — or,
laziest, hard-code a modest `a` from the calibration scatter and skip `b`.

**Effort:** 1.5–2 h.

**Demo impact: MEDIUM visually, HIGH on reliability.** This is the difference between a gate that
clips your first syllable on stage and one that doesn't. HANDOFF risk #3 says "a gate that eats 0.5%
of a CEO's words is worse than useless" — QMF is the published answer to that sentence.

---

## 5. AS-norm: adaptive score normalisation against a small impostor cohort

**What it does.** Normalises the raw cosine score by the mean and standard deviation of that same
window's scores against a fixed cohort of impostor embeddings, restricted to the top-k most similar
cohort members. The effect is a threshold that means the same thing in a different room, on a
different mic, with a different amount of background noise — which is the single biggest thing that
can go wrong between our office test and the demo table.

**Evidence.**
- Cumani, Laface et al., *From adaptive score normalization to adaptive data normalization for
  speaker verification systems*, Interspeech 2023.
  https://www.isca-archive.org/interspeech_2023/cumani23_interspeech.html
- Trainable AS-norm (TAS-norm), ICASSP 2025, arXiv:2504.04512 — reports **4.11% relative EER** and
  **10.62% relative MinDCF** improvement *over* plain AS-norm on VoxCeleb1-O, which incidentally
  confirms plain AS-norm as the strong baseline. https://arxiv.org/abs/2504.04512
- Language-dependent AS-norm variants: arXiv:2007.07689. https://arxiv.org/pdf/2007.07689

**Why it is unproductized.** Same reason as §4 — challenge-system hygiene that never crosses into
products. Note the trainable variant is *out of scope* under our filter; plain AS-norm has no
learning step at all, which is why it is the one we take.

**Implementation sketch.** Ship ~50 ECAPA embeddings as a `.npy` next to the model (generate once
from any public speech clips or teammates — no labels needed beyond "not the enrolled speaker"). In
`EcapaVerifier.score()`: `cohort_scores = cohort @ embedding`; take the top-k (k≈20); return
`(s_raw - mean) / std`. Threshold moves to a z-score, which is far more portable than a raw cosine.
Cost: one 50×192 matmul per decision — microseconds.

**Effort:** 1–2 h including generating the cohort file.

**Demo impact: LOW visually, HIGH insurance.** Invisible when it works. Prevents the failure mode
where the threshold calibrated at 11am is wrong at 4pm in a louder room.

---

## 6. Personalised jitter/shimmer consistency — a one-class, explainable clone indicator

**What it does.** Compares the micro-prosody of the live window (F0 jitter, amplitude shimmer)
against the *enrolled speaker's own* jitter/shimmer distribution captured at enrollment. Not a
global real-vs-fake classifier — a per-speaker consistency check, which sidesteps the cross-dataset
generalisation collapse that makes global spoof detectors near-useless out of domain (34% EER,
per HANDOFF).

**Evidence.**
- *Contributions of Jitter and Shimmer in the Voice for Fake Audio Detection*, IEEE Access / IEEE
  journals 2023 — https://ieeexplore.ieee.org/document/10207023 · open copy:
  https://www.nict.go.jp/en/asean_ivo/gden8c00000004el-att/pdnf1l00000004j7.pdf — finds obvious
  differences in jitter and shimmer between fake and real speech, **shimmer especially**, with large
  dynamic variation for fake speech; combined with mel-spectrogram features and evaluated on ADD
  2022/2023.
- *Fake Audio Detection in Resource-constrained Settings using Microfeatures*, Interspeech 2021 —
  https://www.isca-archive.org/interspeech_2021/dhamyal21_interspeech.pdf
- *Robust prosody modeling for synthetic speech detection*, Speech Communication 2025 —
  https://www.sciencedirect.com/science/article/abs/pii/S0167639325000986 — argues explicitly for
  explainability-by-design: prosody models "operate under stronger constraints, making their
  behavior potentially more transparent".
- One-class framing precedent: *An Evaluation of One-Class Classification Techniques for Speaker
  Verification*. https://www.researchgate.net/publication/220637992

**Why it is unproductized.** Every commercial detector is a global binary classifier trained on
spoof corpora. Nobody ships a *per-enrolled-speaker* micro-prosody consistency check, because nobody
else has an enrollment step on the sender side — we do.

**Implementation sketch.** `pip install praat-parselmouth` (pure pip, no build, CPU, milliseconds
per window). At enrollment, compute local jitter and local shimmer over the 30 s capture and store
mean + std in `Profile`. At runtime, on voiced windows, compute the same and report a z-distance.
Surface as an indicator on the badge, never as a gate blocker.

**Effort:** 1.5–2 h. One new pip dependency (justified — reimplementing Praat's jitter is not lazy).

**Demo impact: MEDIUM.** Its value is in the Q&A: it is the one signal on stage that is both
*explainable in one sentence to a non-expert* ("the clone's pitch is too steady to be your throat")
and *personalised rather than a detector in an arms race*. Also gives the badge a second dimension
so it is not a single binary lamp (see §8).

**Honest limit.** Voice-conversion attacks that copy source prosody (RVC-style) will pass this. It is
an indicator, same disable-flag treatment as AASIST.

---

## 7. Breath-based liveness — the strongest published result here, and the least shippable today

**What it does.** Detects breath events (inhalations between phrases) and uses their presence,
rate, and placement as a liveness signal. TTS and voice-conversion systems largely do not generate
breath, and when they do, they place it wrong.

**Evidence — genuinely striking.**
- *Every Breath You Don't Take: Deepfake Speech Detection Using Breath*, arXiv:2404.15143, published
  in **ACM Digital Threats: Research and Practice 2025** (https://dl.acm.org/doi/10.1145/3754456,
  https://arxiv.org/abs/2404.15143). On 33.6 h / 232 in-the-wild samples: breath-feature + poly-SVC
  gives **AUPRC 1.00, EER 0.00, accuracy 1.00**; a plain thresholding rule still gives **0.90
  accuracy**. On the same in-the-wild samples the SOTA SSL wav2vec model gets **AUPRC 0.72, EER
  0.99** — i.e. worse than a coin flip. Breath detector itself: mel-spectrogram (128 bins, dB) + ZCR
  + RMSE, 20 ms window / 2.5 ms hop, two 1-D conv layers (16 and 8 filters) → BiLSTM → dense sigmoid,
  trained on 10 podcasts / 4 speakers / ~5 h of hand-annotated breath. Predictions every 50 ms.
- BTS-E, *Audio Deepfake Detection Using Breathing-Talking-Silence Encoder*, **ICASSP 2023** —
  https://ieeexplore.ieee.org/document/10095927 — enhances existing countermeasures by **up to 46%**.
- Corroborating that TTS systems structurally lack breath: *Frame-Wise Breath Detection with
  Self-Training*, Interspeech 2024, arXiv:2402.00288. https://arxiv.org/html/2402.00288v1

**Why it is unproductized.** It is a 2023–2025 result and it embarrasses the deep-learning
incumbents on exactly the samples they sell against. No vendor ships it.

**Why it ranks low for us anyway.** The paper's detector is a trained CNN+BiLSTM with code released
"upon publication" — at time of writing, weights are not something we can rely on having on the day.
Rebuilding it means annotating breath, which is a day, not an hour. The DSP-only fallback (ZCR +
RMSE + band energy over unvoiced frames, threshold on breath events per minute) is implementable in
~3 h but its accuracy on our mic is unvalidated, and breath rate needs ~30 s of speech to be
meaningful — which is fine as a slow-moving indicator but useless for a 2.5 s gate decision.

**Effort:** 3–5 h for an unvalidated heuristic version; ~1 day for anything defensible.

**Demo impact: MEDIUM as a slide, LOW as running code.** Recommendation: **cite it, don't build
it.** One line in the deck — "the published state of the art in clone detection is a breath
detector that beats wav2vec 1.00 vs 0.72 AUPRC; we don't need it, because we assert instead of
detect" — is worth more than three risky hours. If there is slack on day two, the DSP version is the
first thing to add.

---

## 8. Badge design from usable-security research (free, and it's the part judges stare at)

**What it does.** Fixes the failure mode that every "verified" indicator in the history of computing
has hit: users ignore positive indicators and habituate to warnings within *two* exposures.

**Evidence.**
- Schechter, Dhamija, Ozment, Fischer, *The Emperor's New Security Indicators*, **IEEE S&P 2007** —
  https://ieeexplore.ieee.org/document/4223213 · PDF: https://stuartschechter.org/papers/emperor.pdf
  — 67 bank customers. **Zero participants withheld their password when HTTPS indicators were
  removed. 23 of 25 (92%) entered it after their site-authentication image was removed.** Positive
  indicators are, empirically, near-worthless on their own.
- Anderson, Kirwan, Jenkins, Eargle, Howard, Vance, *How Polymorphic Warnings Reduce Habituation in
  the Brain*, **CHI 2015** (fMRI, n=25) — https://dl.acm.org/doi/10.1145/2702123.2702322 — visual
  processing drops sharply **after the second exposure**; warnings that *change appearance* are
  substantially more habituation-resistant.
- Porter Felt et al., *Rethinking Connection Security Indicators*, **SOUPS 2016** —
  https://www.usenix.org/system/files/conference/soups2016/soups2016-paper-porter-felt.pdf
- *Signals of Provenance: Practices & Challenges of Navigating Indicators in AI-Generated Media*,
  arXiv:2505.16057 (N=28 semi-structured interviews, 15 sighted / 13 blind-low-vision) —
  https://arxiv.org/abs/2505.16057 — participants **consistently overlooked menu-aided indicators**
  (platform AI labels) and instead relied on **content-based indicators** (title, comments);
  usability failures traced to inconsistent placement, unclear metadata, cognitive overload.
- Broader context on provenance trust: *Examining the Impact of Provenance-Enabled Media on Trust
  and Accuracy Perceptions*, CSCW 2023, arXiv:2303.12118. https://arxiv.org/pdf/2303.12118

**Design consequences for `badge.html` / `receiver.py`, all of them cheap:**

1. **Make absence loud, presence quiet.** Schechter says nobody notices a green badge. So the badge
   should be visually calm while verified and *actively alarming* on transition to unverified —
   motion, a timestamp ("UNVERIFIED since 12:04:31"), the speaker's name struck through. We already
   have `BADGE_STALE_SECONDS`; the event to design around is the *edge*, not the state.
2. **Make it visibly live (polymorphic).** Anderson's habituation result plus a security argument
   point the same way: render something that provably changes with the audio — the current token
   slot number, a live level meter, the rotating token's last 4 hex digits. A static green pill is
   both habituating *and* screenshot-forgeable; a badge that ticks with the token cannot be faked
   with a screenshot and cannot be tuned out.
3. **Put it in the content region, not a menu.** The provenance study's clearest finding. Our badge
   is already a page rather than a settings panel — keep it adjacent to the call window and never
   behind a click.
4. **Show the reason, not just the verdict.** With §1/§4/§6 running we have three named signals.
   "Unverified — voice doesn't match enrollment" reads very differently from a grey light, and it is
   what turns a demo into a product story.

**Effort:** 1–2 h of front-end.

**Demo impact: HIGH.** This is the surface judges actually evaluate, and it is the only section here
where the research is about *humans*, which nobody else in the room will have cited.

---

## 9. The ASV / anti-spoof fusion rule: cite t-DCF to justify never letting AASIST block

**What it does.** Gives a principled, citable answer to the question a sharp judge will ask: "if you
have an anti-spoof model, why isn't it in the decision?"

**Evidence.**
- Kinnunen, Lee, Delgado, Evans, Todisco, Sahidullah, Yamagishi, Reynolds, *t-DCF: a Detection Cost
  Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification*,
  Odyssey 2018 / arXiv:1804.09618. https://arxiv.org/abs/1804.09618 — the metric that made the field
  stop evaluating countermeasures in isolation, with explicit miss/false-alarm costs and priors for
  *both* subsystems. Extended treatment: arXiv:2007.05979. https://arxiv.org/pdf/2007.05979
- SASV 2022 challenge (Interspeech 2022) —
  https://www.isca-archive.org/interspeech_2022/jung22c_interspeech.pdf — baselines built on exactly
  our two models, **ECAPA-TDNN + AASIST**. Baseline1 (naive score *sum*): SASV-EER **19.31%**, a 19%
  relative reduction vs ECAPA alone (23.83%). Best challenge fusion systems reach **1.71%**.
  Baseline description: https://www.isca-archive.org/odyssey_2022/shim22_odyssey.html

**The rule to adopt.** Under t-DCF logic, a countermeasure whose out-of-domain EER is ~34% (HANDOFF,
Speech DF Arena arXiv:2509.02859) carries a false-reject cost that dominates any spoof-detection
benefit at our operating point. So: **the CM never vetoes.** It may only *raise* the ASV threshold —
an asymmetric coupling that can make the gate stricter but can never mute a genuine speaker on its
own. That is one line in `GateDecision`, and it converts "our anti-spoof is just an indicator" from
an apology into a designed, cited decision.

**Implementation sketch.** In `gate.py`, `GateDecision.update()` takes an optional `spoof_score` and
uses `effective_threshold = self.threshold + SPOOF_TIGHTENING * max(0, spoof_score - ref)`. Nothing
else changes.

**Effort:** 30 min.

**Demo impact: LOW visually, HIGH in Q&A.** Cheapest credibility on the list.

---

## 10. Pop-noise liveness — real, cheap, and probably won't fire on our hardware

**What it does.** Human breath striking the microphone diaphragm on plosives and fricatives produces
a characteristic low-frequency burst ("pop noise") that a loudspeaker cannot reproduce. Detecting it
proves a mouth was close to this microphone.

**Evidence.**
- Shiota, Villavicencio, Yamagishi, Ono, Echizen, Matsui, *Voice liveness detection algorithms based
  on pop noise caused by human breath for automatic speaker verification*, **Interspeech 2015**,
  pp. 239–243 — https://www.isca-archive.org/interspeech_2015/shiota15_interspeech.html — two
  algorithms: single-channel low-frequency detection and two-channel subtraction.
- POCO corpus, **Interspeech 2020** —
  https://www.isca-archive.org/interspeech_2020/akimoto20_interspeech.html · data:
  https://github.com/aurtg/poco — five recording settings including **5 cm, 10 cm and 20 cm**
  mouth-to-mic distances.
- *Significance of Distance on Pop Noise for Voice Liveness Detection* (2022) —
  https://link.springer.com/chapter/10.1007/978-3-031-20980-2_20 — **pop-noise energy decreases with
  distance**; plosives and fricatives carry the most pop noise and degrade the most with distance,
  while nasals and liquids carry the least and are least affected.
- Constant-Q transform variant, Computer Speech & Language 2022 —
  https://www.sciencedirect.com/science/article/abs/pii/S0885230822000547
- Bump/Morse wavelet + CNN variants — https://link.springer.com/chapter/10.1007/978-3-031-12700-7_10

**Why it is unproductized.** Ten years of Interspeech papers, zero products. It needs close-talking,
which suits headsets and phones and suits laptop arrays badly.

**Why it ranks low.** Three compounding risks on our demo hardware: (a) laptop mic at 40–60 cm is
well past the 20 cm point where the effect is already measured as degraded; (b) laptop mics and
Windows audio pipelines routinely high-pass below 100 Hz, removing exactly the band the signal lives
in; (c) it is phoneme-triggered, so it fires only on some words, making a live badge flicker. If the
demo runs on a **headset**, all three risks invert and it becomes a genuinely strong close-talk
liveness cue.

**Implementation sketch (if headset).** Band-limited energy in ~20–100 Hz on VAD-voiced frames,
normalised by broadband energy in the same frame; count frames whose ratio exceeds a threshold
calibrated during the 30 s enrollment. ~40 lines of numpy. Report as an indicator.

**Effort:** 2–3 h including on-mic validation. **Do the validation first** — 20 minutes with
`np.fft` on a recording of "puh puh puh" at the demo mic tells you whether to build it at all.

**Demo impact: LOW-MEDIUM, hardware-dependent.**

---

## 11. Free credibility: name AudioSeal's fragility before someone else does

Not an implementation — a slide, and it costs nothing.

**Evidence.** *SoK: How Robust is Audio Watermarking in Generative AI models?*, arXiv:2503.19176 —
https://arxiv.org/pdf/2503.19176 · https://sokaudiowm.github.io/ — **all** evaluated schemes,
AudioSeal included, drop **below 0.6 bit-recovery accuracy under pitch shift** while audio quality
stays high; most schemes (AudioSeal among them) are sensitive to time-stretch and cutting.
Corroborating: SyncGuard on desynchronisation attacks, arXiv:2508.17121
(https://arxiv.org/pdf/2508.17121); *The Impact of Audio Watermarking on Audio Anti-Spoofing
Countermeasures*, arXiv:2509.20736 (https://arxiv.org/abs/2509.20736).

**The line to deliver.** Because TrueMic is **fail-closed**, watermark fragility is an *availability*
problem, not a *security* problem. An attacker who pitch-shifts our audio to strip the watermark
produces audio that reads as **unverified** — they have destroyed their own credential. Every
watermark-removal attack in that SoK moves the badge to grey, never to green. Compare with
detector-based vendors, for whom the same attack moves the verdict to *wrong*.

This is the same asymmetry as the core pitch, applied to our own weakest component. Say it first.

---

## Rejected — considered and cut, with reasons (state these if asked)

| Technique | Source | Why cut |
|---|---|---|
| VoiceGesture — Doppler shift from articulatory gestures via 20 kHz probe | https://arxiv.org/pdf/2106.00859 | Needs ≥40 kHz capture (we run 16 kHz) *and* emitting ~20 kHz from the laptop speaker during a live call. Violates "no >1 s latency / no pipeline surgery" and pollutes call audio. |
| VibLive / MagLive — accelerometer / magnetometer liveness | https://arxiv.org/pdf/2404.01106 | Custom or phone-only sensors. Not on a PC mic path. |
| AuthGlass — smart-glasses multi-mic acoustic features | https://arxiv.org/pdf/2509.20799 | Custom hardware. |
| Microphone-array TDoA liveness (VoiceLive-style) | https://arxiv.org/pdf/2510.24393 | Needs a known stereo array geometry; HANDOFF explicitly forbids hard-coding device assumptions. |
| ENF (mains-hum) liveness / timestamping | https://arxiv.org/pdf/1411.7084 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11099188/ | 50/60 Hz band is high-passed away by laptop mics; needs a reference grid database and tens of seconds for a reliable estimate. Fails our latency filter and our "no external dependency at demo time" rule. |
| Trainable AS-norm (TAS-norm) | https://arxiv.org/abs/2504.04512 | Requires training learnable impostor embeddings. Plain AS-norm (§5) is the lazy 90%. |
| One-class deep spoof detectors (OC-Softmax etc.) | — | Model training from scratch. |
| Deep breath detector as shipped in arXiv:2404.15143 | https://arxiv.org/abs/2404.15143 | Weights "available upon publication"; retraining needs ~5 h of hand-annotated breath. Cited, not built (§7). |
| Target speaker extraction | — | Already cut in HANDOFF; no usable real-time pretrained weights. |
| Double-compression / codec-artefact forensics | — | Assumes a stored file with a compression history. Our signal is live PCM; there is no first compression to detect. |

---

## Recommended build order

Assuming a few hours, not a few days, and that the pipeline in `truemic/` already works:

1. **§3 nonce** (1 h) — biggest security-story gain per line of code, touches one function.
2. **§1 mic fingerprint** (2 h) — new demo beat, closes our worst risk, pure numpy.
3. **§8 badge UX** (1–2 h) — the surface being judged; research nobody else will cite.
4. **§9 fusion rule** (30 min) — free credibility in Q&A.
5. **§4 QMF + §5 AS-norm** (3 h together) — invisible, but they are why the demo doesn't fall over
   in an unfamiliar room.
6. **§2 Void `FV_LDF`** (45 min) — *only after* deciding what happens to demo beat 4.
7. **§6 jitter/shimmer** (2 h) — if there is slack and the badge needs a second dimension.
8. **§10 pop noise** — validate on the actual demo mic for 20 minutes before writing any of it.
9. **§7 breath** — cite it, don't build it.
