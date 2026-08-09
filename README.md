# TrueMic

Voice-authentication virtual mic. Only audio matching the enrolled voiceprint leaves the
machine; verified speech is watermarked with a rotating token (AudioSeal), and a receiver
page shows a live **Verified Human** badge on any call app — zero platform integration.

Pitch and prior-art verdict: `HANDOFF.md`. Original research: `RESEARCH-FULL.md`.
Deep-dive reports behind every design choice below: `research/deep/` (voice isolation,
AudioSeal/CallShield, CPU speaker verification, i5-7200U tuning, unimplemented research)
— and `research/deep/measured-corrections.md`, where our own offline measurements
overturned several report claims (read it before trusting the reports).

## Setup

1. Install [VB-CABLE](https://vb-audio.com/Cable/) (donationware, reboot after install).
   Set its control panel **and** both Windows Sound endpoints to **48000 Hz** — VB-CABLE
   cannot run at 16 kHz internally, and rate mismatches are the #1 crackling cause. The
   sender outputs 48 kHz by default (soxr-resampled).
2. `pip install -r requirements.txt` (torch/torchaudio: any 2.x build, CPU is enough).
3. First run downloads models automatically: CAM++ speaker model (29 MB), AudioSeal
   checkpoints (~90 MB), silero-VAD (bundled). Do this on hotel WiFi, not on stage.

## Quickstart (the product way)

```powershell
python -m truemic.app
```

One command. The browser opens and walks you through everything: mic pick with a
live meter, a reading passage you record three ways (natural / faster / louder — each
delivery style joins your voiceprint), automatic threshold tuning, an optional
impostor pass (a friend, or a loud video at the mic), then the live protection
dashboard: **YOUR VOICE — PASSING** / **BLOCKED — NOT YOUR VOICE**, live score bar,
blocked-seconds counter, injected-source alarm. With VB-CABLE installed it feeds the
virtual mic; without it, it runs as a monitor — which for the voice-command story is
the demo itself.

## Quickstart (CLI, for debugging)

```powershell
# 1. Enroll (~40 s; speak as you normally would; pause between sentences —
#    the quiet gaps fingerprint the microphone)
python -m truemic.enroll --name "Huz"

# 2. Calibrate on the actual room + mic, WITH the demo teammate:
python -m truemic.calibrate --impostor

# 3. Start the gate (writes verified, watermarked audio to VB-CABLE)
python -m truemic.sender

# 4. Start the receiver + badge (second laptop if the demo machine is weak; copy profile/)
python -m truemic.receiver --in "CABLE Output"
# open http://127.0.0.1:8765

# 5. In Zoom/Meet/Discord, select "CABLE Output" as the microphone.
```

`python -m truemic.devices` lists devices (match by substring, never index).
`python -m truemic.bench [--low-power]` prints per-stage cost on the current machine and
recommends a profile. On a 2-core laptop (i5-7200U class) run everything with
`--low-power` and put the receiver on the second laptop.

## Demo runbook (5 beats)

1. **Baseline** — you talk; badge goes green: *Verified human: Huz*. It visibly ticks
   (watermark %, verified-ago counter) — a static screenshot can't fake it.
2. **Kill shot** — teammate talks into the same mic, same machine, same account. Voice
   never leaves the machine (muted to comfort noise); badge slams to red in ~2 s with
   "VERIFICATION LOST". Account identity never changed — only the human did.
3. **Clone** — play an ElevenLabs clone of the enrolled voice. Two outcomes, both wins:
   biometric catches it → muted, never signed; biometric fooled (clones bypass ASV
   43–83 % of the time — say the number first) → played through a speaker it's re-recorded
   room audio and the *mic guard* flags it, and if injected digitally the sender never saw
   a real capture chain (`INJECTED?` tripwire). The honest line: **the clone fails because
   the gate never signed it, not because a detector caught it.**
4. **Replay** — replay a recording of earlier *verified* audio. Badge stays red:
   "watermark present but token stale — possible replay." Tokens rotate every 8 s; anything
   older than ~±1 slot is rejected. (Name the ceiling: within ~16–24 s a replay would
   still read — rotation narrows the window, receiver-issued nonces would close it at the
   cost of the transport independence shown in beat 5.)
5. **Air gap** — phone speaker plays verified output into the second laptop's bare mic
   (`python -m truemic.receiver`, default input, same `profile/`). **Hold the phone
   close to the mic** — near-field paths carry the full token (100% measured); across
   a reverberant room the badge honestly degrades to amber "Verified Origin"
   (watermark present, freshness unconfirmed — the same verdict as a replay, which is
   exactly right). The proof lives in the audio, not the platform.

## How it works

```
mic 16 kHz ── streaming silero-VAD (ONNX) ──┐
     │        mic fingerprint (quiet frames)│ decisions every 0.5 s
     │        CAM++ (ONNX, no-CMN front end)┴─ 3.0 s window opens (2 confirmations,
     │                                         median-smoothed, recent window must
     │                                         agree) / drops close it immediately
     ├─ AudioSeal streaming embed (rotating HMAC token, 8 s slots, alpha=3)
     └─ 1.5 s delay-line gate (hard mute + 20 ms crossfade) + comfort noise → VB-CABLE 48 kHz
receiver: capture → sample-exact PHASE LOCK (one-time 320-phase scan by bit
          crispness, then ±1 tracking) → AudioSeal detect → soft-bit majority vote
          over 3 ticks → exact token match ±1 slot → Ed25519 attestation → SSE badge
          (badge verifies the signature client-side via WebCrypto)
```

**Measured survival (the headline)**: with the phase lock and alpha=3, the token
decodes **95-100% per tick through Opus at 24-64 kbps** — the codec class Zoom uses.
AAC garbles bits (detects, amber). Dense reverb detects but can't carry bits (amber).
Full table: `research/deep/measured-corrections.md`. Caveat for stage: Zoom stacks
noise suppression/AGC on top of Opus — use Original Sound, and rehearse it.

Design decisions that came out of the deep-dive research (full citations in
`research/deep/`):

- **Fail-closed asymmetry, made explicit.** The speech-enhancement field tunes 10:1
  *against* over-suppression (VoiceFilter-Lite's asymmetric loss); a security gate inverts
  that. Opening requires two consecutive median-smoothed decisions on the reliable 3 s
  window; closing is immediate when either the 3 s or the trailing 1 s score drops. The
  short-window close is what stops an impostor riding the tail of your rolling window —
  and the 1.5 s delay line means the verdict lands before their audio ever leaves.
- **Hard mute is load-bearing.** AudioSeal survives uniform attenuation: a measured −20 dB
  "polite" floor still hands the receiver a perfect token (dies only past −32 dB, i.e.
  inaudible). `tests/test_roundtrip.py` pins this. Naturalness comes from unwatermarked
  comfort noise at −55 dBFS instead, mixed in after the gate.
- **Watermark before gate.** Muting zeroes the token with the voice. Watermarking after
  the gate would sign the silence while an impostor talks.
- **CAM++ over ECAPA.** WeSpeaker CAM++ LM in ONNX Runtime: 4× faster than
  SpeechBrain ECAPA *and* more accurate (0.659 % vs 0.782 % vox1-O EER). int8
  quantization measured 16–23× *slower* (ORT ConvInteger; Kaby Lake lacks VNNI) — don't.
- **AS-Norm: measured, then shelved.** With a realistic small cohort (40 LibriSpeech
  speakers) it *reduced* separation (d′ 6.4 raw → 3.9 normalized — top-N statistics
  too unstable). Raw cosine + on-site calibration wins at this scale; the cohort
  builder (`python -m truemic.cohort`) stays for a future hundreds-of-speakers
  version, and the code engages automatically if `models/cohort.npy` exists.
- **Enrollment is multi-conditioned** (clean + 10 dB noise + synthetic reverb per chunk,
  outliers rejected) — the mitigation for CrystalSound's documented "different mic breaks
  it" failure.
- **Mic fingerprint.** Forensic mic identification (93 % over 7 mics with plain FFT
  features) inverted into a live capture-chain check: quiet-frame spectra are enrolled and
  compared at runtime; bit-exact-zero input trips an injection alarm. Indicator by
  default; `--strict-mic` gates on it.
- **Token channel.** 16 bits is a token, not a proof: HMAC(key, 8 s slot), soft-bit
  majority vote across 3 detection ticks, then *exact* match (forgery ~0.005 %/tick). No
  ECC inside the payload — AudioSeal already repetition-codes it at 50 Hz; CallShield
  needed BCH only because its payload serializes over 5 s of channel.
- **Badge UX from usable-security research** (users ignore passive indicators; habituation
  after two exposures): presence is quiet, loss is loud (timestamped red slam), the badge
  ticks live, and it always shows *why* it isn't green.

Latency: 0.5 s watermark chunk + 1.5 s gate + ~0.1 s buffers ≈ 2.1 s (≈ 2.6 s low-power).
A DNS-challenge denoiser spends 20–40 ms — the gap is the security delay, own it: ITU-T
G.114 calls >400 ms one-way "unacceptable" for conversation, which is why this is a
verified-mic *mode*, not a default hop.

## Measured performance

Ryzen 5 7600 (dev box): sender ~17 % of one core, receiver ~5 %. Projected i5-7200U
(×2.3–2.6, from three agreeing scaling methods in `research/deep/i5-7200u-realtime.md`):
sender ~40–50 % default, ~25–30 % `--low-power`; receiver belongs on the second laptop if
Zoom runs locally. Demo-day prep for the laptop: AC power,
`powercfg /setactive SCHEME_MIN`, minimum processor state 100 %, close browsers.

## Troubleshooting

- **Crackling** — 48000 Hz everywhere (VB-CABLE panel, Windows Sound, `--out-rate`).
- **Badge flickers** — raise `BADGE_STALE_SECONDS`; check sender `wm_load` < 100 %.
- **Badge grey through a call app** — the Opus codec itself is measured-safe
  (95-100% token decode at alpha=3), so what's left is the app's own noise
  suppression/AGC: use Zoom with **Original Sound** on and audio enhancements off,
  and rehearse it. Teams (generative Satin codec) will still fail — that's a
  codec-fine-tuning roadmap item, not a protocol hole. VB-CABLE loopback remains the
  zero-risk demo path. `--alpha 4` buys margin at slight audibility cost.
- **Gate never opens** — recalibrate (`--impostor`); thresholds are properties of the
  room. Overlapping distributions? `--badge-only` never mutes and only signs verified
  speech — the graceful posture.
- **Dropouts on the laptop** — `--low-power`; then move receiver off-box; then (code
  hooks exist, see `research/deep/i5-7200u-realtime.md` items 12–13) a verify worker
  thread or MMCSS registration.
- **Eagle** — upside, never the plan: needs internet key validation
  (`PICOVOICE_ACCESS_KEY`). CAM++ is the budgeted engine.
- **speechbrain ReduceOp / torch.compile `cl not found` / silero 1-thread trap** — all
  shimmed in-code (`truemic/__init__.py`, `verify.py`); details in the vault notes.

## Deliberately not built (and why — judges ask)

- **AASIST anti-spoof**: ~34 % EER out-of-domain. The demo story doesn't need it, and per
  SASV-2022 cost analysis a spoof detector should only ever *raise* the ASV bar, never
  veto alone. Never claim "blocks deepfakes."
- **Receiver-issued nonce challenge**: closes the ≤24 s replay window but requires a
  sender↔receiver back-channel — exactly what CallShield refused, and what beat 5 shows
  off. Product roadmap, stated ceiling for now.
- **Void-style loudspeaker detection**: mutually exclusive with the air-gap beat — a
  posture choice, made deliberately.
- **Pop-noise / breath liveness**: best published in-the-wild clone detector is literally
  a breath detector (AUPRC 1.00 vs wav2vec's 0.72) — we cite it, we don't need it,
  because we assert instead of detect.
- Target-speaker extraction, PKI, codec-robustness matrix, Zoom SDK: cut per ship order.

## Tests & offline validation

```powershell
python -m pytest tests -q          # 54 tests: gate asymmetry, tokens, phase lock,
                                   # mic fingerprint, augmentation, profiles,
                                   # offline end-to-end watermark roundtrip
python -m truemic.robustness       # watermark survival table: codecs, resampling,
                                   # air-gap channels x alpha (run on any machine)
python -m truemic.rehearse         # full demo rehearsal with real LibriSpeech
                                   # voices: impostor leakage, badge timeline
                                   # (needs models/test-clean.tar.gz, 346 MB)
```

Latest rehearsal (genuine-only calibration, real voices): impostor speech leaked
0.69 s of 9.7 s spoken, badge grey for the entire impostor segment; genuine loss
dominated by the one-time open latency. `--impostor` calibration tightens both.
