# Measured corrections to the deep-dive reports (2026-08-09, this machine)

Offline experiments on the actual pipeline overturned or sharpened several report
claims. Where this file disagrees with an agent report, this file wins — these are
measurements, not literature.

## 1. The AudioSeal STREAMING pair is phase-sensitive; the paper's "no sync" claim does not transfer

`audioseal-callshield.md` §5.4 repeats the paper: "AudioSeal needs no alignment
search." True for the non-streaming pair; **false for the streaming checkpoints**
(`audioseal_wm_streaming` + `audioseal_detector_streaming`):

- Message bits decode ONLY at sample-exact alignment with the embedder's 320-sample
  frame grid: bit crispness (mean |p−0.5|·2) is ~0.75 at the true phase, ~0.05 one
  sample off. Zero usable gradient — brute-force search is required.
- Detection *score* plateaus at 1.0 over roughly ±16 samples and CANNOT be the lock
  criterion (it happily locks 16 samples off, where bits are garbage).
- Codecs shift the grid by constant sub-grid delays (Opus 32k: 215 samples).

Fix implemented in `WatermarkReader.read_locked`: one-time batched scan of all 320
phases selecting by bit crispness, then ±1 tracking per tick. `DETECT_HOP` must be a
multiple of 320 samples (0.74 s / 1.0 s) or the lock walks off every tick.

## 2. With the phase lock, the token SURVIVES Opus — the "loopback only" posture was too pessimistic

`robustness.py` (synthetic speech, 8 s, threshold semantics as shipped):

| channel | α=1 score/tick-exact | α=3 | α=5 |
|---|---|---|---|
| clean | 1.00 / 90% | 1.00 / 100% | 1.00 / 100% |
| resample 16-48-16 | 1.00 / 90% | 1.00 / 100% | 1.00 / 100% |
| opus 24k | 0.99 / 48% | 1.00 / 95% | 1.00 / 100% |
| opus 32k | 1.00 / 62% | 1.00 / 100% | 1.00 / 100% |
| opus 64k | 1.00 / 81% | 1.00 / 100% | 1.00 / 100% |
| aac 64k | 0.38 / 10% | 0.91 / 24% | 0.92 / 43% |
| air gap, reverb T60 0.3 + 20 dB SNR | 0.50 / 0% | 1.00 / 0% | 1.00 / 0% |
| near-field (direct + mild reflections) | — | 1.00 / 100% | — |

Consequences: Zoom's codec class (Opus) carries the token at the default α=3. AAC
detects but garbles bits (partial). Reverberant air paths detect perfectly but cannot
carry bits (an RIR is a filter, not a delay — sample-exact alignment cannot survive
dense multipath); **near-field playback (phone held at the mic) carries bits at
100%**, so demo beat 5 works as staged. The badge's amber "Verified Origin" state
covers the reverberant case honestly (and doubles as the replay verdict).

## 3. The sherpa-onnx CAM++ export must run WITHOUT CMN — the model card metadata, not the training recipe, is the authority

`speaker-verification-cpu.md` action 2 prescribed WeSpeaker's training recipe
including CMN ("omitting it changes the embedding by cosine 0.53"). Measured
discrimination on LibriSpeech test-clean pairs through the actual ONNX export:

| front end | same-spk cosine | cross-spk cosine | gap |
|---|---|---|---|
| CMN on, int16 | 0.867 | 0.907 | **−0.040 (collapse)** |
| **CMN off, int16** | **0.855** | **0.287** | **+0.568** |
| CMN off, float | 0.953 | 0.905 | +0.048 |

CMN on = zero discrimination (the export evidently bakes its own normalization).
The export's metadata (`normalize_samples=0`, no `feature_normalize_type`) was
right; the recipe transplanted from training was wrong. Rolling-window separation
after the fix: genuine 0.82±0.07 vs impostor 0.38±0.07, d′=6.4, no overlap.

**AS-Norm verdict (rebuilt cohort, fixed front end): shelved.** With 40 LibriSpeech
cohort speakers, one-sided AS-Norm (top-15) *reduced* d′ from 6.39 to 3.87 and blew
up impostor variance (−20.8±7.5) — small-cohort top-N statistics are unstable.
`speaker-verification-cpu.md` action 6 assumed a helpful cohort at this size; it is
not. Raw cosine + on-site calibration is the shipped configuration; the builder
stays for a future large-cohort version.

## 4. End-to-end rehearsal with real voices (truemic/rehearse.py, LibriSpeech test-clean)

A(20 s) → B impostor (15 s) → A(10 s), genuine-only calibration, full pipeline:

- impostor speech leaked: **0.69 s of 9.7 s spoken** (boundary hand-off only;
  badge grey for the entire impostor segment)
- genuine speech lost: 8.8 s of 21.3 s — dominated by the one-time open latency and
  LibriSpeech's unnatural inter-utterance gaps hitting the 3 s silence timeout;
  live continuous speech loses far less. On-site `--impostor` calibration tightens
  both numbers.
- Gate changes this motivated: opening now also requires the recent 1 s window to
  clear the open/close midpoint (kills the impostor riding the long window's tail),
  and the recent-window close needs 2 consecutive sub-close ticks (a single noisy
  1 s dip no longer mutes the real speaker: genuine loss went 62% → 41% of spoken
  audio in the stress test while leakage stayed sub-second).

## 5. Misc

- audioseal's `streaming()` context manager teardown crashes (moshi `_stop_streaming`
  → torch `isinstance` TypeError) and a half-torn cache corrupts later embeds. Use
  `model.encoder.streaming_forever(1)` + `reset_streaming()` instead (no teardown).
- Ed25519 attestation verifies client-side via WebCrypto (badge button) — canonical
  JSON reconstruction matches Python's `json.dumps(sort_keys, compact)` byte-for-byte
  after a JSON round-trip; validated with Playwright in a real browser.
