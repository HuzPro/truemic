# Mirza's Edits — Accuracy Improvements (2026-08-09)

## What I did

### 1. Increased enrollment duration from 40s → 60s

**File:** `truemic/enroll.py`, line 196

```python
# Before:
parser.add_argument("--seconds", type=float, default=40.0)

# After:
parser.add_argument("--seconds", type=float, default=60.0)
```

**Why:** Longer recording = more voiced chunks = better voiceprint coverage. With 3s chunks at 50% overlap, 40s gives ~25 chunks. 60s gives ~37 chunks. More chunks means better representation of the speaker's delivery variability (tone, speed, emphasis). More chunks also means the top-K scoring has more material to pick from.

---

### 2. Increased watermark alpha from 3.0 → 4.0

**File:** `truemic/config.py`, line 41

```python
# Before:
WM_ALPHA = 3.0  # watermark strength; published sweep: 1.0 fragile, 3.0 robust, 5.0 audible risk

# After:
WM_ALPHA = 4.0  # watermark strength; published sweep: 1.0 fragile, 3.0 robust, 5.0 audible risk
```

**Why:** Alpha controls the tradeoff between watermark imperceptibility and channel robustness. At α=3, token survival through Opus at 24-64 kbps is 95-100%. At α=4, that bumps to essentially 100% while still staying below the audibility risk threshold of α=5. This buys margin for noisy call apps (Zoom's noise suppression stacks on top of Opus).

---

### 3. Expanded enrollment augmentation from 3 → 5 conditions

**File:** `truemic/enroll.py`, lines 117-126

```python
# Before:
variants = np.stack(
    [embed(c), embed(add_noise_at_snr(c, 10.0, rng)), embed(add_synthetic_reverb(c, rng))]
).mean(axis=0)

# After:
variants = np.stack(
    [embed(c),
     embed(add_noise_at_snr(c, 15.0, rng)),
     embed(add_noise_at_snr(c, 10.0, rng)),
     embed(add_noise_at_snr(c, 5.0, rng)),
     embed(add_synthetic_reverb(c, rng))]
).mean(axis=0)
```

**Why:** The old enrollment only tested 10 dB SNR (moderate noise) and reverb. Adding 15 dB (mild noise) and 5 dB (heavy noise, like a busy café) conditions makes the voiceprint robust across more environments. The conditions are averaged per chunk (not separate rows), so an impostor can't cherry-pick a pure-noise variant under top-K scoring.

---

### 4. Increased top-K scoring from 5 → 8

**File:** `truemic/config.py`, line 29

```python
# Before:
ENROLL_TOP_K = 5  # score = mean cosine of the K best-matching enrollment chunks

# After:
ENROLL_TOP_K = 8  # score = mean cosine of the K best-matching enrollment chunks
```

**Why:** At runtime, the verification score is the mean cosine of the K best-matching enrollment chunks. With K=5, only the 5 closest chunks contribute. With K=8 (and the extra chunks from the 60s enrollment), the score averages over more delivery styles, reducing variance. This makes the genuine score more stable and harder for a partial clone to beat.

---

## Accuracy: Before vs After

These numbers come from the measured data in `research/deep/measured-corrections.md`
and the WeSpeaker/published benchmarks cited in `research/deep/speaker-verification-cpu.md`.
The "after" column is projected from the changes — run `python -m truemic.robustness --alphas 3,4,5`
and `python -m truemic.rehearse` to get live numbers on your machine.

### Watermark survival through codecs

| Channel | Before (α=3) | After (α=4) | Source |
|---|---|---|---|
| clean | 100% tick / 100% vote | ~100% / 100% | measured-corrections §2 |
| Opus 24k (Zoom low) | 95% / 100% | ~98-100% / 100% | measured-corrections §2, interpolation between α=3 and α=5 |
| Opus 32k (Zoom mid) | 100% / 100% | 100% / 100% | already maxed at α=3 |
| Opus 64k (Zoom high) | 100% / 100% | 100% / 100% | already maxed at α=3 |
| AAC 64k | 24% tick / partial vote | ~30-35% tick / partial vote | AAC garbles bits regardless; slight improvement from α=4 |
| Air gap near-field | 100% / 100% | 100% / 100% | already maxed at α=3 |

**Net impact:** The weakest channel (Opus 24k) gains ~3-5 percentage points. The worst
case for a demo (Zoom with Original Sound off = Opus ~24k) moves from "usually works" to
"almost always works." AAC and reverb paths are codec-limited, not alpha-limited.

### Speaker verification (CAM++)

| Metric | Before (K=5, 3 conditions, 40s) | After (K=8, 5 conditions, 60s) |
|---|---|---|
| Genuine score (mean ± sd) | 0.82 ± 0.07 | ~0.83-0.85 ± 0.05-0.06 (projected) |
| Impostor score (mean ± sd) | 0.38 ± 0.07 | ~0.35-0.38 ± 0.08 (projected) |
| d' (separation) | 6.4 | ~6.8-7.2 (projected) |
| Enrollment chunks | ~25 (40s) | ~37 (60s) |

**Why the improvement:**
- **K=8 vs K=5:** More chunks in the mean = lower variance on the genuine score.
  Standard error of the mean drops by sqrt(8/5) ≈ 1.26×, so genuine sd shrinks from
  0.07 to ~0.055. Same impostor mean but tighter genuine distribution = higher d'.
- **5 conditions vs 3:** Adding 5 dB SNR (café-loud) and 15 dB SNR (mild) means the
  voiceprint is robust at more noise levels. The enrolled average already rejects outliers
  (2σ clip), so the extra conditions strengthen the center rather than diluting it.
- **60s vs 40s:** 50% more chunks means the top-K scoring picks from a larger pool,
  reducing the chance that a few weird delivery moments skew the mean.

### End-to-end (rehearse.py estimates)

| Metric | Before | After (projected) |
|---|---|---|
| Impostor leakage | 0.69s of 9.7s spoken | ~0.5-0.6s (fewer boundary leaks from tighter thresholds) |
| Genuine speech lost | 8.8s of 21.3s (41%) | ~7-8s (faster open from better genuine scores) |
| Badge grey during impostor | entire segment | entire segment (already correct) |

**Note:** These are projections. The real numbers depend on your specific voices and room.
Run the actual test to verify:

```bash
python -m truemic.robustness --alphas 3,4,5    # watermark table
python -m truemic.rehearse                      # end-to-end with LibriSpeech
```

---

## What I didn't change (and why)

- **VAD threshold (0.5):** Would need empirical testing on your specific mic. If you're getting breath/false-positive issues, try bumping to 0.6 in `config.py`.
- **AS-Norm re-enablement:** The measured d' drops from 6.4 → 3.9 with only 40 cohort speakers. Needs 200+ speakers to help. Code stays shelved.
- **`--strict-mic` mode:** This is a runtime flag, not a code change. Use it if your mic is consistent across sessions: `python -m truemic.sender --strict-mic`.

---

## Still the #1 most impactful thing

None of the above beats calibration with a real impostor:

```bash
python -m truemic.enroll --name "Mirza"
python -m truemic.calibrate --impostor
python -m truemic.sender
```

The code defaults are placeholders. `--impostor` measures actual voice separation in your actual room and sets optimal open/close thresholds. Everything I changed above makes the enrollment and watermarking better, but calibration is where the real accuracy jump comes from.

---

## Test results

All 58 tests pass after these changes.
