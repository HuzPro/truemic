import numpy as np
import pytest

from truemic.gate import DelayGate, GateDecision

DELAY = 100
FADE = 20


def run_blocks(gate, blocks):
    return np.concatenate([gate.process(b) for b in blocks])


def ones_blocks(n_blocks, block_size=50):
    return [np.ones(block_size, dtype=np.float32) for _ in range(n_blocks)]


class TestDelayGate:
    def test_emits_same_number_of_samples_as_pushed(self):
        gate = DelayGate(DELAY, FADE)
        out = gate.process(np.ones(50, dtype=np.float32))
        assert len(out) == 50

    def test_audio_is_delayed_by_the_delay_line(self):
        gate = DelayGate(DELAY, FADE)
        gate.set_allow(True)
        out = run_blocks(gate, ones_blocks(10))
        assert np.allclose(out[:DELAY], 0.0)
        assert out[DELAY + FADE + 1] == pytest.approx(1.0)

    def test_denied_audio_never_leaves_the_gate(self):
        gate = DelayGate(DELAY, FADE)
        gate.set_allow(False)
        out = run_blocks(gate, ones_blocks(20))
        assert np.allclose(out, 0.0)

    def test_gate_starts_closed(self):
        gate = DelayGate(DELAY, FADE)
        out = run_blocks(gate, ones_blocks(20))
        assert np.allclose(out, 0.0)

    def test_closing_fades_out_without_a_step(self):
        gate = DelayGate(DELAY, FADE)
        gate.set_allow(True)
        run_blocks(gate, ones_blocks(10))
        gate.set_allow(False)
        out = run_blocks(gate, ones_blocks(10))
        max_jump = np.max(np.abs(np.diff(np.concatenate([[1.0], out]))))
        assert max_jump <= (np.pi / 2) / FADE + 1e-6
        assert np.allclose(out[FADE:], 0.0)

    def test_reopening_fades_in_over_the_fade_length(self):
        gate = DelayGate(DELAY, FADE)
        run_blocks(gate, ones_blocks(10))
        gate.set_allow(True)
        out = run_blocks(gate, ones_blocks(10))
        assert out[0] < 0.5
        assert out[FADE + 1] == pytest.approx(1.0)
        assert np.all(np.diff(out[: FADE + 1]) >= -1e-6)

    def test_gain_property_tracks_the_gate_state(self):
        gate = DelayGate(DELAY, FADE)
        assert gate.gain == 0.0
        gate.set_allow(True)
        run_blocks(gate, ones_blocks(5))
        assert gate.gain == 1.0


class TestGateDecision:
    def make(self, silence_timeout=3.0):
        return GateDecision(
            open_threshold=0.60,
            close_threshold=0.50,
            silence_timeout=silence_timeout,
            open_confirmations=2,
            recent_close_threshold=0.50,
        )

    def open_gate(self, d, start=0.0):
        d.update(0.7, 0.7, now=start)
        assert d.update(0.7, 0.7, now=start + 0.5) is True
        return start + 0.5

    def test_starts_closed(self):
        d = self.make()
        assert d.update(None, None, now=0.0) is False

    def test_one_good_decision_is_not_enough_to_open(self):
        d = self.make()
        assert d.update(0.9, 0.9, now=0.0) is False

    def test_opens_after_two_consecutive_good_decisions(self):
        d = self.make()
        d.update(0.7, 0.7, now=0.0)
        assert d.update(0.7, 0.7, now=0.5) is True

    def test_a_bad_score_resets_the_opening_streak(self):
        d = self.make()
        d.update(0.7, 0.7, now=0.0)
        d.update(0.3, 0.3, now=0.5)
        assert d.update(0.7, 0.7, now=1.0) is False

    def test_opening_is_blocked_while_the_recent_window_disagrees(self):
        """An impostor rides the tail of the genuine speaker's long window:
        the 3 s score still clears the bar but the last second is theirs."""
        d = self.make()
        d.update(0.7, 0.7, now=0.0)
        assert d.update(0.7, 0.4, now=0.5) is False
        assert d.update(0.7, 0.7, now=1.0) is True

    def test_closes_after_two_recent_window_drops(self):
        d = self.make()
        now = self.open_gate(d)
        assert d.update(0.7, 0.45, now=now + 0.5) is True  # one dip = noise
        assert d.update(0.7, 0.45, now=now + 1.0) is False  # two = impostor

    def test_single_recent_dip_does_not_mute_the_real_speaker(self):
        d = self.make()
        now = self.open_gate(d)
        d.update(0.7, 0.45, now=now + 0.5)
        assert d.update(0.7, 0.7, now=now + 1.0) is True

    def test_recent_path_is_inert_without_its_own_calibration(self):
        d = GateDecision(
            open_threshold=0.60, close_threshold=0.50,
            open_confirmations=2, recent_close_threshold=None,
        )
        d.update(0.9, 0.1, now=0.0)
        assert d.update(0.9, 0.1, now=0.5) is True  # opens despite low recent
        assert d.update(0.9, 0.1, now=1.0) is True  # and stays open

    def test_closes_immediately_when_long_window_drops(self):
        d = self.make()
        now = self.open_gate(d)
        assert d.update(0.45, None, now=now + 0.5) is False

    def test_holds_open_between_close_and_open_thresholds(self):
        d = self.make()
        now = self.open_gate(d)
        assert d.update(0.55, 0.55, now=now + 0.5) is True

    def test_reopening_needs_confirmations_again(self):
        d = self.make()
        now = self.open_gate(d)
        d.update(0.3, 0.3, now=now + 0.5)
        assert d.update(0.7, 0.7, now=now + 1.0) is False
        assert d.update(0.7, 0.7, now=now + 1.5) is True

    def test_holds_verdict_through_short_silence(self):
        d = self.make(silence_timeout=3.0)
        now = self.open_gate(d)
        assert d.update(None, None, now=now + 1.0) is True

    def test_closes_after_prolonged_silence(self):
        d = self.make(silence_timeout=3.0)
        now = self.open_gate(d)
        assert d.update(None, None, now=now + 3.5) is False
