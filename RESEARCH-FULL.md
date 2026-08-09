# Full Research — Verbatim Agent Reports (2026-08-08)

Four parallel deep-dive research passes. Synthesis lives in `HANDOFF.md`.

---

# Report 1 — Commercial security-framed voice-gate products

16 searches, ~8 page fetches. Summary up front: **the exact concept has not shipped.** Every component exists in a shipped product somewhere, but no one has combined them at the mic-gate position with security framing.

## 1. The single closest mechanism — and it's framed as audio quality

**Microsoft Teams Voice Isolation**
- User enrolls a voice profile (~30s of read speech, stored locally). A model then "ensure[s] only your voice is transmitted through your microphone," removing "any sounds or voices during a call or meeting that don't match the profile."
- Sender-side. Real-time. Enrolled voiceprint. Gates the outgoing mic stream. **Mechanically the (a) half of the concept, shipped at Microsoft scale.**
- Framing: 100% audio-quality / "personalized noise suppression." Zero security claim, no attestation, no signal to the receiving side, no synthetic-audio detection. Admin control is a `TeamsAIPolicy` biometric-profile toggle (enabled Jan 2025), framed as privacy/consent.
- https://techcommunity.microsoft.com/blog/microsoftteamsblog/voice-isolation-in-microsoft-teams-enables-personalized-noise-suppression-for-ca/4096077 · https://learn.microsoft.com/en-us/microsoftteams/voice-isolation

**Krisp Background Voice Cancellation** — same category, no enrollment needed for core BVC ("detects and removes all other nearby human voices, keeps only the main speaker's voice"). Quality-framed.

Verdict: *the gate is a solved, commoditized audio feature. The novelty is entirely in the framing plus the attestation channel.*

## 2. The closest security-framed sender-side product

**Krisp Voice Security (Jun 2026, early access)** — the single most relevant find.
- "Krisp sits between the agent's headset and the CCaaS/UCaaS stack," on the agent's device, no telephony integration. **Exactly the sender-side mic-gate position.**
- Three modules:
  - *Agent Voice Verification* — "continuously confirms the person on each agent call matches the enrolled agent's voice profile," detecting unauthorized substitution. Sender-side, real-time, enrolled voiceprint, **explicitly security-framed**.
  - *Deepfake Detection* — but on **inbound** caller audio (receiver-side).
  - *Fraud Detection* — transcript-based social-engineering signals.
- Gaps vs. the concept: **alerts/flags, does not gate or block**. Deepfake detection points the wrong direction. No attestation to the far end. Contact-center only.
- https://krisp.ai/blog/krisp-anounces-voice-security-speech-analytics/ · https://krisp.ai/contact-center/

## 3. Sender-side attestation exists — via identity/device/face, never voice

| Product | What it does | Closeness |
|---|---|---|
| **Beyond Identity RealityCheck** (Zoom + Teams plugin) | AAL3 device-bound passkey auth; **green badge rendered in your camera layer** + participant panel so others can trust you. Continuous re-verification of user + device. | The attestation half, shipped — but attests *device + credential*, never audio. Nothing stops a verified user's mic carrying a cloned voice. |
| **Zoom + World ID "Deep Face"** (beta Apr 2026) | World App matches live Zoom frame ↔ Orb registration ↔ on-device selfie; "Verified Human" badge on tile. Deep Face Waiting Room; on-demand mid-call re-verification. | Sender-side proof-of-human, native to Zoom. **Face only — no audio path at all.** |
| **Incode Deepsight** (Dec 2025) | "Integrity layer to verify camera and device authenticity" — detects/blocks injected virtual cameras, <100ms. TikTok, Scotiabank, Nubank. | **Architectural twin of the (b) half, but for video.** IDV sessions only, no audio equivalent. |
| **PRVEN** | Biometric liveness → public proof record. | Asynchronous record, not a live stream gate. |

## 4. Deepfake-detection vendors — all receiver-side

| Company | Live calls? | Notes |
|---|---|---|
| **Pindrop** (Pulse for Meetings, GA Nov 2025) | Yes | A "Security Assistant" **bot joins the meeting**, analyzes every participant's streams. Alert-only, no gating. Claims 99% audio accuracy, <1% FP. Zoom/Teams/Webex. |
| **Reality Defender** (YC W22) | Yes | RealCall (telephony); RealMeeting Zoom/Teams plugins → Trust/Suspicious/Manipulated verdicts to the host. ValidSoft partnership (Aug 2025). Only YC company under YC's `deepfake-detection` tag. |
| **GetReal Security** (Hany Farid) | Yes | Continuous identity verification voice+video with enrolled identities — but detection runs enterprise/receiver-side, SOC alerts. 40+ IAM integrations. |
| **Scam.ai — Halo** (Aug 2026) | Yes | On-device (Snapdragon NPU) on the *viewer's* PC, flags synthetic faces. Video-first, wrong direction. |
| **Hiya** | Yes | Carrier/OEM-distributed; #1 average on HF Speech Deepfake Arena (Feb 2026). |
| **Resemble AI Detect** | Real-time capable | Detection + **watermarking** — but as content marking, not a call-time mic gate. |
| **ValidSoft** | Yes | Voice biometrics + deepfake detection + voice MFA, receiver-side. |
| **ID R&D / Mitek IDLive Voice** | Yes, ms-latency | Passive liveness (replay/synthesis/spectral artifacts). Closest *technique* to (b); always evaluating the far party. |
| **McAfee Deepfake Detector**, **Intel FakeCatcher**, **DeepMedia**, **Fraunhofer** (Aug 2026 prototype) | — | All receiver-side content/call analysis. |
| **Validia** | — | validia.ai now a stealth landing page; treat as unverified/pivoted. |

## 5. Enterprise vishing / meeting protection

Live category: Pindrop Pulse, RealMeeting, GetReal, RealityCheck, Zoom Deep Face, Scam.ai Halo, Incode. **Universal shape: observe others, score them, alert a human/SOC.** None changes what leaves *your* microphone. Adjacent: Allure Security, Nametag.

## 6. Patents — mostly Cirrus Logic

- **US11024317B2 / US11769510 — "Microphone authentication"** (priority Sep 2017): verifies audio "originated from an actual microphone rather than being artificially synthesized or injected" via Helmholtz resonance of the mic port; emits a verification signal without which voice-auth "may be configured so as not to generate a positive verification result." **Shipped-IP description of injection-detection + gating, at codec level.**
- **US10103872 / US10560256 / US11228420 / US11848753** — trusted-domain auth at the ADC boundary.
- **US12354607, US11475899, US12347440B2** (ultrasound energy in speech), **US12401744** (defensive deepfake), **US11373663** (call-audio authenticity), **US12283263 / US12424193** (audio firewall), watermark family **US11269976, US12067994, US10811018**.
- Practical read: sensor-attestation claims sit with a silicon vendor. FTO on a *software-only* gate looks better; needs counsel.

## 7. Standards / research adjacent

- **C2PA v2.3 (Dec 2025)**: live/broadcast signing at CMAF segment level; C2PA device certificates now explicitly cover **microphones** with hardware-backed attestation at cert enrollment.
- TPM-backed live-stream provenance = academic prototypes. Adaptive Security consensus: liveness detection is "helpless against digital injection… requires cryptographic attestation at the sensor and platform level." **The market wrote down the problem statement and did not ship the product.**

## Verdict table

| Component | Shipped? | By whom | Missing piece |
|---|---|---|---|
| Enrolled-voiceprint gate on outgoing mic | Yes | MS Teams Voice Isolation | Noise-suppression framing; no security claim, no far-end signal |
| Sender-side verification, security-framed, real-time | Yes | Krisp Voice Security | Alerts, doesn't gate; CC-only; no attestation |
| Detect synthetic audio injected locally | Patented, not productized | Cirrus Logic (codec) | No shipping software for PC mics |
| Sender-side badge visible to far end | Yes | RealityCheck; Zoom+World ID | Device/credential/face — never audio |
| Block virtual-device injection in-product | Yes, video | Incode Deepsight | Video only, IDV only |

Three honest risks: (1) Microsoft owns the gate mechanism + enrolled profile at platform level; adding a badge is a smaller move for them. (2) Software-only injection detection is the weak link — user-space can be lied to by user-space; credible defenses reach for silicon. (3) Everyone alert-only — probably false-positive economics; a gate that eats 0.5% of a CEO's real speech is worse than useless.

---

# Report 2 — Voice-isolation market framing

**Headline: nobody ships enrolled-speaker mic filtering as a security product.** Enrolled-voiceprint filtering is mainstream (Microsoft, Zoom, CrystalSound, neep). Security framing exists (Krisp Voice Security). They are in **different products from different vendors**, never joined. In zero cases does the far end receive any signal.

## 1. Enrolled-profile filtering (real enrollment, quality framing)

**Microsoft Teams Voice Isolation** — most rigorous enrollment stack in market.
- Enrollment: manual (~30s) or "express/passive voice enrollment" built from in-meeting speech.
- Identity-keyed: removes sounds/voices that "don't match the profile." Local copy **encrypted on-device, expiring every 14 days**; master profile in Office 365 compliance store.
- Governance: `CsTeamsAIPolicy` (`-EnrollVoice`, `-PassiveVoiceEnrollment`), `CsTeamsMeetingPolicy -VoiceIsolation`, 1-year unused-profile auto-deletion, 90-day deletion on account removal, user-only export. Unavailable GCCH/DOD.
- **Framing: unambiguously quality/productivity** ("improve the audio quality and user experience"). Security-shaped language is biometric-collection compliance, not a sold benefit. Same profile feeds speaker attribution, IntelliFrame, Copilot recaps.
- Receiver signal: **none** (indicator is local-only).

**Zoom Personalized Audio Isolation** — genuine voiceprint, zero security framing.
- Automatic voiceprint from meeting audio, optionally refined by scripted "Audio Signature" reading. On-device, never sent to Zoom, user-deletable. Framed as noise suppression, buried in Audio settings. Receiver signal: none.

**CrystalSound "My Voice Only"** (Namitech) — archetype.
- "Register Profile" in a quiet room; up to **5 profiles**; re-registration on model updates; **must use the same microphone for registration and use** — strong tell the embedding is channel-coupled, not a robust identity vector.
- Marketed as a "virtual boundary of 2 metres radius." Vendor admits leakage ("faint voices in the background").
- Win/mac/Linux; virtual-device injection into Zoom/Teams/Meet/Discord/Slack/OBS. **Entirely quality framing**; privacy = on-device processing only. No authentication/deepfake language anywhere. Receiver signal: none.

**neep** — near-clone of CrystalSound, same feature name ("My Voice Only"). $12/user/mo. Quality framing. No security claims.

## 2. Primary-speaker isolation (no enrollment, no identity)

| Product | Mechanism | Framing |
|---|---|---|
| Krisp VIVA SDK (`krisp-viva-pro-v1`, `-tel-v2`, `-tel-lite-v1`, `-ss-v1`) | dominant-talker heuristic, no enrollment | Quality / WER for voice agents |
| Krisp consumer BVC | pitch/dominance, no profile | Quality |
| ai-coustics "Quail Voice Focus" | primary-speaker, explicitly no biometric matching, ~30ms, no GPU | Quality (43% rel. WER reduction) |
| NVIDIA Broadcast | noise/echo/Studio Voice, **no profile of any kind** | Quality/creator |
| Apple Voice Isolation | near-field prioritisation, no enrollment | Quality |
| Google Meet | generic noise filter | Quality |
| IRIS Clarity | generic, no profile; 8x8 integration | Contact centre |
| ElevenLabs Voice Isolator / Cleanvoice / Adobe Enhance Speech | post-hoc cleanup, no identity | Creator |

## 3. Security-framed players (not doing mic filtering)

**Krisp Voice Security** (Jun 2026, early access) — only vendor using security framing; closest to enrolled-identity-as-security. Agent Voice Verification **flags, does not filter**. Deepfake Detection = inbound. Fraud Detection = 18 behavioural signals. Trust flows inward/upward (protects contact centre from caller, employer from agent). Krisp has all ingredients, ships them unjoined (VIVA=quality, Voice Security=security, Accent Conversion=separate).

**Picovoice** — both halves in the box, unjoined: Eagle (streaming speaker recognition, per-frame similarity scores) + Koala (generic denoising, not speaker-conditioned). Their own voice-isolator guide never distinguishes target-speaker extraction from primary-speaker isolation, never suggests Eagle+Koala.

**TSE research** is established (REAL-TSE Challenge SLT 2026, USEF-TSE, EvoTSE) but **no commercial SDK sells enrollment-conditioned TSE** — vendors ship the dominant-speaker approximation.

## 4. Receiver-visible trust: confirmed absent, universally

No product sends the far end anything. Teams indicator = speaker's own client only. Krisp verdicts = receiving agent's desktop / employer. Adjacent machinery: STIR/SHAKEN (number, SIP layer), C2PA (files, not live calls). Patents exist for watermark-based call authenticity; no shipping product.

## Closest product & what's missing

**Closest: Microsoft Teams Voice Isolation** (enrolled identity actively gates outbound mic, biometric-grade governance). Runner-up mechanism: Zoom. Runner-up framing: Krisp AVV.

Missing from Teams: (1) the security framing itself; (2) receiver-side signal; (3) fail-closed semantics (it attenuates best-effort); (4) liveness/anti-replay/anti-clone on the outbound path; (5) cross-app reach (walled garden).

**Open gap:** cross-app virtual mic that gates on an enrolled voiceprint, fails closed, includes outbound liveness/clone detection, and emits a verifiable assertion the far end can see. All four pieces ship today — in four different products from four vendors.

Key sources: Teams voice-isolation admin docs · Zoom KB0074698 · crystalsound.ai · neep.com/faq · Krisp VIVA/Voice Security/Accent posts · Picovoice Eagle docs · ai-coustics SDK · REAL-TSE Challenge.

---

# Report 3 — Technical SOTA & one-day build (14 searches + 3 papers read directly)

## TL;DR verdict

Build the pipeline. **Do not claim it stops deepfakes** — honest framing is "speaker gate + spoof *flag*": (a) AI clones pass speaker verification at 43–83% bypass rates; (b) open anti-spoof checkpoints sit ~34% EER out-of-domain, near coin-flip.

## 1. Recommended stack

| Stage | Pick | Why |
|---|---|---|
| Capture | `sounddevice` InputStream, 16 kHz mono, blocksize 480 (30 ms) | PortAudio wheel, no build step; same lib for capture AND virtual-mic output |
| VAD | `silero-vad` (ONNX, ~1 MB) or `webrtcvad` | Non-negotiable; ~1 ms/frame |
| Ring buffer | 3 s deque of float32 | decide on 2–3 s context, emit every ~0.5–1 s |
| Speaker verification | **Picovoice Eagle** (`pip install pveagle`); fallback **SpeechBrain ECAPA** | see §2 |
| Anti-spoof | **clovaai/aasist** pretrained (MIT) | see §3 |
| Output | `sounddevice` OutputStream → `"CABLE Input (VB-Audio Virtual Cable)"` | see §5 |
| Gate | delay line + 20 ms cosine crossfade | hard-muting clicks and leaks a full window |

New deps: `sounddevice`, `numpy`, `torch`(CPU) or `onnxruntime`, plus `pveagle`/`speechbrain`.

## 2. Speaker verification on CPU

**Picovoice Eagle** — only streaming API in the list; ~15-line enroll/score loop:
```python
eagle_profiler.enroll(frame)      # returns completion %, call until 100.0
eagle.process(chunk, profiles)    # [0,1] score per profile, or None if "not enough voice"
```
`None` = built-in VAD for free. `frame_length`/`sample_rate`/`min_process_samples` read off the object.
Catches: **free tier = 3 active users/month**; **AccessKey requires internet to validate** even though inference is offline — *demo WiFi is a single point of failure; validate before stage, keep ECAPA behind a flag*.

**SpeechBrain ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`): **0.80% EER VoxCeleb1-test**; ~20M params; ~30–80 ms per 2–3 s embedding on laptop CPU. **Windows landmine** — `from_hparams()` fails for non-admin (symlinks). Fix:
```python
from speechbrain.utils.fetching import LocalStrategy
clf = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}, savedir="./models/ecapa",
    local_strategy=LocalStrategy.COPY_SKIP_CACHE)
```
(speechbrain#1155, #2836)

Others: **pyannote/wespeaker-voxceleb-resnet34-LM** ships ONNX (lowest-latency no-torch path, more plumbing). **Resemblyzer** — easiest API but weakest on short windows. **WeSpeaker** toolkit Linux-oriented. **NeMo TitaNet** — huge dep tree, skip. **3D-Speaker** — Mandarin-centric flagships, skip.

**Minimum window — honest numbers:** ~1 s is below the reliable threshold.
| Window | EER (VoxCeleb1-O, strong system) |
|---|---|
| Full utterance | 0.61% |
| 3 s | 0.98% |
| 2 s | 1.48% |
3.59 s → 2.05 s raised EER 8.72% → 12.8% (+46% rel.). Resemblyzer floor ~2.63 s; 1–1.5 s undependable.
**Design consequence: decide every ~1 s ON a 2–3 s rolling window. Highest-value design choice in the build.**

## 3. Anti-spoofing on CPU

**Install `clovaai/aasist`** — MIT, pretrained AASIST (297K params) + AASIST-L (85K, 332 KB file). CPU real-time trivially. **0.83% EER ASVspoof2019 LA.**
Friction: no single-wav inference script (write ~20 lines: load config, `Model(config)`, `load_state_dict`, forward, softmax); standard recipe crops to **64,600 samples (~4.04 s)** — check `data_utils.py` before sizing windows.
Alternatives: XLSR+SLS strongest open but 300M-param wav2vec2 + fairseq (Windows install pain). Nes2Net = best upgrade path. HF one-liner classifiers = no provenance.

**Generalization — the core finding (bad):**
| Model | Avg EER (Speech DF Arena) | Pooled |
|---|---|---|
| XLSR+SLS | 13.84% | 15.68% |
| Nes2NetX | 16.11% | 17.04% |
| Wav2Vec2-AASIST | 18.02% | 19.47% |
| **AASIST** | **34.49%** | **33.16%** |
| **RawNet2** | **35.75%** | **35.66%** |
AASIST: 0.83% in-domain → ~34% cross-dataset — **40× collapse, near random**.
Corroboration: (1) XLS-R+AASIST 0.83% → **24.84% OOD** vs eight modern TTS (detectors "memorize attack-specific statistical patterns"); (2) **noise: 0.83% clean → 16.24% EER at 10 dB SNR** — a hackathon room is at/below 10 dB (RawBoost recovers to 2.55% but you're not retraining today); (3) DeePen: **time-stretch drops accuracy to 3–14%** (below 50% = worse than random), pitch-shift comparable; every commercial SaaS detector tested had ≥1 beating attack. Note: 48→16 kHz resampling is itself a mild manipulation of that family — use `soxr`, not naive decimation. (4) Deepfake-Eval-2024: open audio detectors lose 48% AUC, max 0.58. Müller et al.: in-the-wild EER up to +1000%.
**ElevenLabs specifically:** no clean published EER exists (flag: an auto-summary of DeePen claimed one; the PDF does not contain it). Directional evidence: detection likelihood notably lower when ElevenLabs is the engine (Frontiers 2025); no major benchmark contains 2024–25-gen TTS. **Assume near-chance and you won't be embarrassed on stage.**

## 4. Target speaker extraction — not in a day

**No.** BUTSpeechFIT/speakerbeam = training recipe (days). SpeakerBeam-SS (real-time, sub-20 ms) = C++ research code, **no public weights**. VoiceFilter = no canonical release. ESPnet on Windows = multi-hour adventure. TSE also inverts the latency budget (must run on every sample). Gating + smooth crossfade reads as "it removed the other voice" to an audience.

## 5. Windows audio routing

**VB-CABLE. Not close.**
- Donationware; installs `CABLE Input` (playback) + `CABLE Output` (capture). Python writes to CABLE Input; Zoom selects CABLE Output as mic. Whole integration.
- `sounddevice.query_devices()`, match substring `"CABLE Input"`, pass index — **don't hard-code indices** (they reshuffle on USB plug).
- Latency: 2048 samples → ~21.3 ms max at 48 kHz, typically ~14.2 ms. App buffers must be <682 samples — keep blocksize 480/512.
- **Sample-rate mismatch = crackling #1 cause.** Same rate in VBCABLE Control Panel + Windows Sound + streams. Models want 16 kHz, VB-CABLE defaults 44.1/48 — resample explicitly.
- **Skip WASAPI exclusive** (crash/BSOD risk per VB-Audio). Shared mode fine.
- Alternatives rejected: VoiceMeeter (config surface), open-source kernel drivers (signing/test-mode reboot — don't spend hackathon hours there).

## 6. Latency & accuracy expectations

Audio path: mic 30 ms + processing ~10 + VB-CABLE ~15 + app buffer 20–40 ≈ **75–95 ms**. Decision latency dominated by window: **2–3 s**.
**The architectural decision that matters:** optimistic gating (leaks 1–3 s of impostor audio — the flaw the audience spots) vs **delay line ~1.5 s** (verdict ready before chunk is emitted; same amount of code — a deque). Ship the delay line, tunable constant.
Accuracy: verification vs other humans on 2–3 s windows in a quiet room = demo-solid, low single-digit error. Vs a clone of the enrolled voice: **it will pass** (§7). Spoof detection vs ASVspoof-era samples = looks great; vs current ElevenLabs over a live mic in a noisy room = **near-chance**.
**Calibration knobs, not constants:** cosine threshold + VAD aggressiveness are properties of room/mic/speaker. Budget 10 min on-site: record 30 s of the real user on the real hardware, set threshold from that distribution.

## 7. The attack weakness that reframes the project

**Clones pass speaker verification.** 2026 systematic eval: ECAPA-TDNN, threshold at FAR=0.01% (compliance-grade), clones from 10–30 min of target speech:
| Cloning system | Bypass rate | Avg cosine |
|---|---|---|
| Bert-VITS2 (TTS) | **82.7%** | 0.679 |
| GPT-SoVITS (TTS) | **56.2%** | 0.598 |
| RVC (VC) | **43.1%** | 0.558 |
Genuine same-speaker similarity is 0.6–0.8 — clones land inside the band. Authors: voiceprint auth "can be easily compromised with just 10–30 minutes of target speech… readily available," single consumer GPU, <2 hours; ASV alone "unlikely to provide reliable protection."
**So: the anti-spoof module is the only defense against a clone, and it's the weakest component.** Also pre-empt: **replay laundering** (deepfake through speakers into mic shifts it off-distribution — direct work exists, arXiv 2505.14862) and **enrollment poisoning** (nothing authenticates enrollment).
**Framing advice:** "Blocks deepfakes" is falsifiable live. "Real-time speaker gating with a spoof-risk indicator, plus the measured evidence for why the second layer is hard" is the stronger submission — same code, shows you read the field. Lead with the generalization gap.

## Cut order if behind

Ship: mic → VAD → verification → gate → VB-CABLE. Anti-spoof = second independent scoring path with a one-flag disable (most likely to misbehave live, silent failure mode). TSE: don't start.

Sources: Picovoice/eagle · Eagle Python API · Hackster free-tier · speechbrain/spkrec-ecapa-voxceleb · speechbrain#1155 #2836 · pyannote/wespeaker-resnet34-LM · arXiv 1810.10884 · ERes2NetV2 Interspeech'24 · CEUR Vol-4164 paper7 · clovaai/aasist · arXiv 2110.01200 · Speech DF Arena arXiv 2509.02859 · Nes2Net · XLSR+SLS repo · arXiv 2601.02914 · DeePen arXiv 2502.20427 · Deepfake-Eval-2024 arXiv 2503.02857 · Müller Interspeech'22 · Frontiers frai.2025.1678043 · arXiv 2505.14862 · BUTSpeechFIT/speakerbeam · SpeakerBeam-SS · vb-audio.com/Cable · VB forum t=600 · python-sounddevice docs.

---

# Report 4 — Receiver-side trust / attestation

## Bottom line

Industry solved the **face/video** half and the **detection** half. Nobody ships a **sender-side positive assertion carried in the audio itself that any receiver can verify**. The one piece of exact prior art is CallShield (Jan 2026 preprint) — no receiver UI, no product, 63-second handshake. That is the gap.

## 1. C2PA — wrong layer for live calls

C2PA 2.3 (Dec 2025) added live/broadcast signing at CMAF segment level (1–10 s segments) — broadcast-shaped, one-way. Audio manifests in OGG etc.; Spotify verifying uploads; ElevenLabs/Stable Audio/AIVA on 2026 timelines. **Nothing for conferencing** — C2PA is a container format and dies at PCM decode/re-encode in any call codec. Patent US 12,244,875 fences live-streaming provenance. **Don't build on it.**

## 2. Platform identity — all-in on face, 2026

- **Zoom + World ID "Deep Face"** (launched 17 Apr 2026, beta May): Orb iris registration; live device face scan cross-referenced with the actual video frame; "Verified Human" badge; Deep Face Waiting Room; mid-call re-verification. **Closest shipped analog — camera-only, no mic equivalent.** (Spain DPA GDPR warning Feb 2026.)
- **Zoom + Pindrop** (12 Mar 2026): Pulse/Passport/Protect in **Zoom Contact Center** — continuous deepfake analysis + passive voice auth. Contact-center only; detection not proof; signal goes to the agent's fraud dashboard.
- **Teams**: no first-party detection/badge; ecosystem = Sensity plugin, IRONSCALES, UncovAI (bot participant), Pindrop, Reality Defender.
- **Webex Zero Trust E2E Identity**: certificate-based participant verification, Internal/External/**Unverified** labels — attests **certificate/account, not a human, not the speaker**.
- **Google Meet**: nothing found.

**Universal pattern: account/device attested cryptographically; human attested by face; the microphone attested by nobody.**

## 3. Startups

| Player | Does | Receiver-visible? | Audio? |
|---|---|---|---|
| Beyond Identity RealityCheck | device-bound passkey + biometric login; **green/red badge on the Zoom camera stream** + risk panel; continuous | **Yes — best shipped badge UX** | No |
| Pindrop | voice auth + synthetic detection, Zoom CC | agent dashboard | Yes |
| Reality Defender / Sensity / GetReal / Resemble | real-time detection plugins | alert, not attestation | Yes |
| Truepic | C2PA capture at camera: secure-enclave-signed, Snapdragon 8 Gen 3 TEE | yes, files | **No mic analog exists** |
| World / TFH | Orb proof-of-personhood → Zoom badge | yes | No |
| Humanity Protocol | pivoted away (Feb 2026) to palm credentials | — | No |
| Roundtable | behavioural bot score API | score | No |

**Truepic is the structural precedent confirming the gap** (enclave-signed camera capture in silicon; nobody did the mic). **RealityCheck proves the badge UX is viable** — with nothing behind it for voice.

## 4. Cryptographic approaches — the real prior art

**CallShield (arXiv 2601.09327, 14 Jan 2026)** — read this paper; closest competitor.
- QCRI et al. "First real-time, audio-layer caller authentication embedding cryptographically verifiable identity into live speech via neural watermarking."
- Custom RTAW model on AudioSeal's generator/detector; **1 bit per 40 ms at 8 kHz = 25 bps**.
- Protocol: 8-bit start beacon → 64-bit nonce + 64-bit session ID → 128-bit truncated HMAC-SHA256 → 8-bit finish. BCH ECC. Symmetric pairwise pre-shared keys, no PKI, no cloud, fully offline.
- 99.2% success clean, 95%+ under distortion. PESQ >4.2, STOI >0.94. **63 s average to authenticate.**
- Stated gaps = our opening: no receiver-notification UI at all; replay-vulnerable under active adversaries; no key distribution/rotation; no carrier story; **authenticates the device/contact, not the speaking human** — no biometric gate.

**AudioSeal (Meta)** — build material: MIT, `pip install audioseal`, **16-bit payload**, **streaming since v0.2** (`model.streaming(batch_size)`), per-sample localized detection, detector ~2 orders of magnitude faster than prior art, 16/24 kHz native, works at 48 kHz.

**SynthID audio** — unusable: detection needs Google's private keys (only SynthID Text was open-sourced). OpenAI watermarks GPT-Live voice with SynthID (Aug 2026, EU AI Act timing) with a verification API — you can't run a detector yourself. Rule out.

**SFrame / WebRTC Encoded Transform** — proper sender authentication (signs auth tags of frame groups; receivers verify against sender pubkey; Insertable Streams hook after encoder). Zero audio-quality cost — but requires controlling both endpoints and dies at any transcoding gateway/PSTN bridge.

**Hardware-attested mic capture — does not exist.** TPM/Secure Enclave/Pluton/Titan all exist; Truepic/Qualcomm proved the camera pattern; **no published research or product does enclave-signed mic capture**. Genuinely unbuilt; far beyond a hackathon.

## 5. STIR/SHAKEN

Attests that the originating carrier believes the caller may use that number (attestation A/B/C). Nothing about the human or audio. **No biometric extension found** — no FCC proceeding, no IETF draft, no vendor proposal. Structural blockers: US-centric PKI, SIP/IP only, attestation levels opaque to users.

## What is genuinely unbuilt

1. Any receiver-visible trust signal derived from the microphone (all badges are face/account).
2. A sender-side gate that **asserts** rather than a receiver-side model that guesses (detection degrades as generators improve; a proof does not).
3. A transport-independent trust signal carried in the audio (survives Zoom, Teams, PSTN, a speakerphone, a recording).
4. Receiver UX for audio attestation (CallShield explicitly didn't build it — cheapest, most visible thing to own).
5. Enclave-signed mic capture (real, hard, not a hackathon).

## Hackathon recommendation

**Voice-biometric gate → AudioSeal watermark on verified stream → browser receiver page with live badge.**
- AudioSeal: MIT, streaming, 16-bit payload, zero training. CallShield only needed a custom model for 8 kHz narrowband — laptop demo is 16/48 kHz.
- **Don't push a signature through the watermark — 16 bits is a token, not a proof.** Watermark carries a rotating token ID; receiver page trades it over HTTPS for the signed attestation blob ("Alice Chen, biometric match 0.94, verified 0.4 s ago, signed by <key>"). Watermark = binding to this audio right now; TLS + Ed25519 = the cryptography. An afternoon of work, and honest.
- Rotate token ~2 s; verifier rejects stale — a replay story CallShield lacks.

**Demo beats:** (1) green "Verified human: Alice" live badge; (2) another human, same mic/machine/account → grey in ~2 s — *account unchanged, human changed* (kills Webex E2EI and every account-based approach); (3) cloned Alice voice → stays grey, the gate never fired, no arms race entered (contrast Pindrop/Reality Defender explicitly); (4) phone speaker to laptop mic — badge reads through the air gap: transport independence in 5 seconds of stage time.

**Cut list:** no PKI, no key rotation infra, no codec robustness matrix, no Zoom SDK. The biometric gate itself is not the novel part.

**Ceilings to name out loud** (naming = rigour, hiding = naivety): watermark is copyable → replay of verified audio replays the token (rotation mitigates, doesn't solve); sender's machine is trusted → malware below the gate defeats it (CallShield concedes same); rotating tokens need the verifier online (unlike CallShield's offline pairwise design).

**Positioning line:** *STIR/SHAKEN attests the number. Zoom and World attest the face. Webex attests the certificate. Nothing attests the voice — and the voice is what people actually trust on a call.*

Key sources: c2pa.org 2.3 announcement · C2PA spec 2.4 · US12244875 · thenextweb Zoom/World · biometricupdate Zoom beta · Pindrop-Zoom globenewswire · Webex security paper · beyondidentity.com/products/realitycheck · Truepic C2PA capture · arXiv 2601.09327 (CallShield) · github.com/facebookresearch/audioseal · AudioSeal paper (Meta) · SFrame medooze writeup · openai.com content-provenance · USENIX Sec'25 AUDIO WATERMARK · arXiv 2606.15187 VoxWatermark · transnexus STIR/SHAKEN whitepaper.
