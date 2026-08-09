# TrueMic — Hackathon Handoff

Voice-authentication virtual mic with receiver-visible attestation. YC RFS angle: "Proving You're Human".
Verbatim research from the four deep-dive research passes: `RESEARCH-FULL.md`.

## Submission text (final)

**Domain:** Cybersecurity — Voice Authentication & Deepfake Defense

**Problem Statement**
AI voice cloning has broken trust in live calls — a clone takes minutes of public audio to make, beats speaker verification, and near coin-flips deepfake detectors. Existing safeguards verify the account, number, or face. Nothing verifies the voice.

**Solution (scope of work)**
A virtual mic that only passes audio matching your enrolled voiceprint — everyone else is muted before leaving the machine — and watermarks verified speech with a rotating token (Meta's AudioSeal). A receiver web page reads the watermark and shows a live "Verified Human" badge on any call app, zero platform integration. Demo: a teammate on the same machine gets blocked; an AI clone of your voice stays unverified.

**Positioning line:** STIR/SHAKEN attests the number. Zoom and World ID attest the face. Webex attests the certificate. Nothing attests the voice — and the voice is what people actually trust on a call.

## Research verdict (compressed)

**Has it been done? No.** Every component ships separately; nobody composed them:
- Enrolled-voice mic gating exists as a QUALITY feature (MS Teams Voice Isolation, Zoom Personalized Audio Isolation, CrystalSound "My Voice Only", neep). No security framing, no fail-closed, no receiver signal.
- Security framing exists but alert-only, wrong direction (Krisp Voice Security, Jun 2026: verifies enrolled agent, alerts employer, never blocks; deepfake detection points at INBOUND callers).
- Receiver-visible "verified" badges exist for face/device only (Zoom + World ID "Deep Face" Apr 2026, Beyond Identity RealityCheck, Webex E2EI certs). No mic analog anywhere.
- Closest prior art: **CallShield** (arXiv 2601.09327, Jan 2026 preprint) — crypto watermark in live call audio, 63s handshake, no biometric gate, no receiver UI, no product. Its stated gaps are our feature list.
- Injection detection at capture: Cirrus Logic patents (silicon-level), Incode Deepsight (video only). No software product for PC mics.
- All deepfake vendors (Pindrop, Reality Defender, GetReal, Hiya, Sensity, Scam.ai) are receiver-side detectors in an arms race. Alert-only, never gating.

**Thesis: assert, don't detect.** Sender-side positive assertion carried in the audio itself; a proof doesn't degrade as generators improve. Transport-independent (survives Zoom/Teams/PSTN/air-gap replay through a speaker).

## Hard numbers (do NOT claim "blocks deepfakes")

- Voice clones bypass ECAPA speaker verification at FAR=0.01%: **43–83%** (Bert-VITS2 82.7%, GPT-SoVITS 56.2%, RVC 43.1%) — arXiv 2601.02914
- AASIST anti-spoof: 0.83% EER in-domain → **~34% cross-dataset** (near coin-flip) — Speech DF Arena, arXiv 2509.02859
- Anti-spoof at 10 dB SNR (normal room noise): 0.83% → **16.24% EER**
- Trivial time-stretch drops detector accuracy to **3–14%** (worse than random) — DeePen, arXiv 2502.20427
- Reliable verification window: **2–3 s** (decide every ~1s ON a rolling 2–3s window; 1s windows are unreliable)
- Winning story: the clone fails NOT because a detector caught it, but because the gate never verified/watermarked it. We don't enter the arms race.

## Build stack (Windows, one day)

| Stage | Pick | Notes |
|---|---|---|
| Capture | `sounddevice` InputStream, 16 kHz mono, blocksize 480 | same lib for output |
| VAD | `silero-vad` (ONNX ~1MB) | non-negotiable; scoring silence = garbage scores |
| Speaker verify | Picovoice Eagle (`pip install pveagle`, streaming API) | free tier 3 users/mo; AccessKey validation NEEDS INTERNET — validate before demo; `process()` returns None when not enough voice (free VAD) |
| Fallback verify | SpeechBrain ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) | Windows fix: `local_strategy=LocalStrategy.COPY_SKIP_CACHE` (symlink bug #1155) |
| Anti-spoof | `clovaai/aasist` AASIST-L checkpoint (MIT, 85K params, 332KB) | INDICATOR ONLY, behind a disable flag; expects ~4.04s (64,600 samples) windows — check data_utils.py; write own ~20-line inference script |
| Gate | ~1.5 s delay line (deque) + 20 ms cosine crossfade | optimistic gating leaks 1–3s of impostor audio — judges will spot it |
| Watermark | AudioSeal (`pip install audioseal`, MIT, streaming since v0.2, 16-bit payload) | 16 bits = token, not proof: rotating ~2s token ID → receiver page trades token over HTTPS to verifier endpoint → signed attestation blob (Ed25519). Verifier rejects stale tokens (replay story CallShield lacks) |
| Output | VB-CABLE — write to "CABLE Input", apps select "CABLE Output" as mic | ~14–21ms latency; SAME sample rate everywhere (crackling); resample via soxr not naive; don't hard-code device indices; skip WASAPI exclusive (BSOD risk) |

- Audio path latency ~75–95 ms + 1.5 s security delay.
- Expose similarity threshold + VAD aggressiveness as constants; calibrate on-site with 30s of real speaker on real mic.
- CUT: target speaker extraction (no usable real-time pretrained weights — days not hours), PKI, key rotation, codec robustness, Zoom SDK.
- Ship order if behind: mic → VAD → verify → gate → VB-CABLE. Anti-spoof and watermark are independent add-ons.

## Demo script (4 beats)

1. **Baseline** — you talk; receiver page shows live green "Verified human: <name>" badge.
2. **Kill shot** — teammate talks into SAME mic, SAME machine, SAME account. Voice never leaves the machine; badge greys in ~2s. Account identity unchanged — only the human changed. (Kills all account/cert-based verification.)
3. **Clone** — play ElevenLabs clone of enrolled voice. Badge stays grey: gate never fired, no detector arms race entered. Contrast with Pindrop/Reality Defender explicitly.
4. **Air gap (bonus)** — phone speaker playing verified audio held to laptop mic; badge still reads. Proves signal lives in the audio, not the platform.

## Risks to name before judges do

1. Microsoft: Teams already holds the enrolled voiceprint; adding a badge is a small move for them. Moat = cross-platform + attestation channel, not the gate.
2. Software-only injection detection is the weak link — user-space can be lied to by user-space; strong version lives at silicon (Cirrus Logic patents sit there). We attest the gate's verdict, not room physics.
3. Everyone shipping chose alert-only — likely false-positive economics. Gate that eats 0.5% of a CEO's words is worse than useless. Fail-closed for demo; tunable posture for product.
4. Watermarks are copyable — recorded verified audio replays the token. Rotation + staleness rejection mitigates, doesn't solve.
5. Enrollment poisoning — nothing authenticates enrollment; enroll from a clone and the system protects the attacker.

## Key links

- AudioSeal: https://github.com/facebookresearch/audioseal
- AASIST: https://github.com/clovaai/aasist
- Eagle: https://github.com/Picovoice/eagle
- ECAPA: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- VB-CABLE: https://vb-audio.com/Cable/
- CallShield: https://arxiv.org/abs/2601.09327
- Clone-vs-ASV study: https://arxiv.org/pdf/2601.02914
- Speech DF Arena: https://arxiv.org/html/2509.02859v1
- DeePen: https://arxiv.org/pdf/2502.20427
