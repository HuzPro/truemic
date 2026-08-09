import numpy as np
from scipy.signal import lfilter

from truemic.micprint import MIN_QUIET_FRAMES, RollingMicPrint, spectrum_of_frames

FRAME = 512


def colored_noise(rng, coeffs, n_frames):
    """White noise through a small FIR — a stand-in for a mic's coloration."""
    noise = rng.standard_normal(n_frames * FRAME).astype(np.float32) * 0.01
    return lfilter(coeffs, [1.0], noise).astype(np.float32).reshape(n_frames, FRAME)


MIC_A = [1.0, 0.6, 0.2]
MIC_B = [1.0, -0.7, 0.3, -0.1]


def enrolled_print(coeffs, seed=0):
    frames = colored_noise(np.random.default_rng(seed), coeffs, 100)
    return spectrum_of_frames(frames)


def run_frames(guard, coeffs, seed):
    for frame in colored_noise(np.random.default_rng(seed), coeffs, MIN_QUIET_FRAMES + 10):
        guard.update(frame, voiced=False)


class TestMicMatch:
    def test_same_mic_scores_higher_than_different_mic(self):
        same = RollingMicPrint(enrolled_print(MIC_A))
        run_frames(same, MIC_A, seed=1)
        other = RollingMicPrint(enrolled_print(MIC_A))
        run_frames(other, MIC_B, seed=1)
        assert same.match() > other.match()

    def test_same_mic_matches_closely(self):
        guard = RollingMicPrint(enrolled_print(MIC_A))
        run_frames(guard, MIC_A, seed=2)
        assert guard.match() > 0.97

    def test_no_verdict_until_enough_quiet_frames(self):
        guard = RollingMicPrint(enrolled_print(MIC_A))
        guard.update(colored_noise(np.random.default_rng(3), MIC_A, 1)[0], voiced=False)
        assert guard.match() is None

    def test_no_enrolled_print_means_no_verdict(self):
        guard = RollingMicPrint(None)
        run_frames(guard, MIC_A, seed=4)
        assert guard.match() is None

    def test_voiced_frames_do_not_pollute_the_print(self):
        guard = RollingMicPrint(enrolled_print(MIC_A))
        loud = np.sin(np.linspace(0, 300, FRAME)).astype(np.float32)
        for _ in range(50):
            guard.update(loud, voiced=True)
        assert guard.match() is None  # nothing quiet seen yet


class TestInjectionTripwire:
    def test_digital_silence_raises_the_zero_fraction(self):
        guard = RollingMicPrint(None)
        for _ in range(40):
            guard.update(np.zeros(FRAME, dtype=np.float32), voiced=False)
        assert guard.zero_fraction() == 1.0

    def test_real_mic_noise_floor_keeps_zero_fraction_low(self):
        guard = RollingMicPrint(None)
        rng = np.random.default_rng(5)
        for _ in range(40):
            guard.update(rng.standard_normal(FRAME).astype(np.float32) * 1e-4, voiced=False)
        assert guard.zero_fraction() == 0.0
