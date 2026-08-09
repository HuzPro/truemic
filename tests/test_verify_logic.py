import numpy as np
import pytest

from truemic import config
from truemic.audio import add_noise_at_snr, add_synthetic_reverb
from truemic.enroll import _reject_outliers
from truemic.verify import as_norm, enrollment_matrix, top_k_score


def unit(rng, n=512):
    v = rng.standard_normal(n).astype(np.float32)
    return v / np.linalg.norm(v)


class TestAsNorm:
    def test_score_far_above_cohort_normalizes_positive(self):
        rng = np.random.default_rng(0)
        cohort = np.stack([unit(rng) for _ in range(50)])
        embedding = unit(rng)
        assert as_norm(0.7, embedding, cohort) > 3.0

    def test_score_typical_of_cohort_normalizes_near_zero(self):
        rng = np.random.default_rng(1)
        cohort = np.stack([unit(rng) for _ in range(50)])
        embedding = unit(rng)
        top = np.sort(cohort @ embedding)[-config.AS_NORM_COHORT_TOP :]
        z = as_norm(float(top.mean()), embedding, cohort)
        assert abs(z) < 1e-6

    def test_degenerate_cohort_falls_back_to_raw_cosine(self):
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        cohort = np.tile(embedding, (50, 1))  # identical -> zero spread
        assert as_norm(0.42, embedding, cohort) == 0.42


class TestTopKScoring:
    def test_single_vector_prints_still_work(self):
        rng = np.random.default_rng(10)
        vector = unit(rng)
        matrix = enrollment_matrix(vector)
        assert matrix.shape[0] == 1
        assert top_k_score(matrix, vector) == pytest.approx(1.0)

    def test_genuine_probe_scores_far_above_impostor(self):
        rng = np.random.default_rng(11)
        base = unit(rng)
        styles = np.stack([base + 0.3 * unit(rng) for _ in range(8)])
        matrix = enrollment_matrix(styles)
        probe = styles[0] + 0.1 * unit(rng)
        probe /= np.linalg.norm(probe)
        impostor = unit(rng)
        assert float((matrix @ probe).max()) > 0.9  # the matching style is found
        assert top_k_score(matrix, probe) > top_k_score(matrix, impostor) + 0.5

    def test_impostor_matches_no_enrolled_moment(self):
        rng = np.random.default_rng(12)
        matrix = enrollment_matrix(np.stack([unit(rng) for _ in range(6)]))
        impostor = unit(rng)
        assert top_k_score(matrix, impostor) < 0.3


class TestEnrollmentOutliers:
    def test_far_off_embedding_is_dropped(self):
        rng = np.random.default_rng(2)
        base = unit(rng)
        cluster = np.stack([base + 0.01 * rng.standard_normal(512) for _ in range(9)])
        outlier = -base[None]
        kept = _reject_outliers(np.concatenate([cluster, outlier]))
        assert len(kept) == 9

    def test_tight_cluster_is_kept_whole(self):
        rng = np.random.default_rng(3)
        base = unit(rng)
        cluster = np.stack([base + 0.01 * rng.standard_normal(512) for _ in range(8)])
        assert len(_reject_outliers(cluster)) >= 7


class TestAugmentation:
    def test_noise_lands_at_the_requested_snr(self):
        rng = np.random.default_rng(4)
        signal = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
        noisy = add_noise_at_snr(signal, 10.0, rng)
        noise = noisy - signal
        snr = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
        assert snr == pytest.approx(10.0, abs=0.5)

    def test_reverb_preserves_length_and_energy(self):
        rng = np.random.default_rng(5)
        signal = rng.standard_normal(16000).astype(np.float32)
        wet = add_synthetic_reverb(signal, rng)
        assert len(wet) == len(signal)
        assert np.mean(wet**2) == pytest.approx(np.mean(signal**2), rel=0.1)
