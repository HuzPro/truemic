import pytest

from truemic.tokens import current_slot, token_for_slot, verify_token

KEY = b"\x01" * 32
SLOT_SECONDS = 2.0


def test_token_is_deterministic_for_same_key_and_slot():
    assert token_for_slot(KEY, 1234) == token_for_slot(KEY, 1234)


def test_token_fits_in_16_bits():
    for slot in range(50):
        assert 0 <= token_for_slot(KEY, slot) <= 0xFFFF


def test_adjacent_slots_produce_different_tokens():
    assert token_for_slot(KEY, 1000) != token_for_slot(KEY, 1001)


def test_different_keys_produce_different_tokens():
    assert token_for_slot(KEY, 1000) != token_for_slot(b"\x02" * 32, 1000)


def test_current_slot_advances_every_slot_interval():
    assert current_slot(100.0, SLOT_SECONDS) == 50
    assert current_slot(101.9, SLOT_SECONDS) == 50
    assert current_slot(102.0, SLOT_SECONDS) == 51


def test_verify_accepts_token_from_current_slot():
    now = 5000.0
    slot = current_slot(now, SLOT_SECONDS)
    token = token_for_slot(KEY, slot)
    assert verify_token(KEY, token, now, SLOT_SECONDS) == slot


def test_verify_accepts_token_from_previous_slot_within_tolerance():
    now = 5000.0
    slot = current_slot(now, SLOT_SECONDS) - 1
    token = token_for_slot(KEY, slot)
    assert verify_token(KEY, token, now, SLOT_SECONDS, tolerance=1) == slot


def test_verify_rejects_stale_token_beyond_tolerance():
    now = 5000.0
    stale_slot = current_slot(now, SLOT_SECONDS) - 5
    token = token_for_slot(KEY, stale_slot)
    assert verify_token(KEY, token, now, SLOT_SECONDS, tolerance=1) is None


def test_verify_rejects_any_bit_flip_by_default():
    now = 5000.0
    slot = current_slot(now, SLOT_SECONDS)
    genuine = token_for_slot(KEY, slot)
    assert verify_token(KEY, genuine ^ 0b1000, now, SLOT_SECONDS) is None


def test_verify_tolerates_one_bit_decode_error_when_asked():
    now = 5000.0
    slot = current_slot(now, SLOT_SECONDS)
    garbled = token_for_slot(KEY, slot) ^ 0b1000
    assert verify_token(KEY, garbled, now, SLOT_SECONDS, max_bit_errors=1) == slot


def test_verify_rejects_two_bit_errors_even_when_tolerant():
    now = 5000.0
    slot = current_slot(now, SLOT_SECONDS)
    garbled = token_for_slot(KEY, slot) ^ 0b1001
    assert verify_token(KEY, garbled, now, SLOT_SECONDS, max_bit_errors=1) is None
