"""Rotating 16-bit watermark tokens derived from a shared HMAC key.

16 bits is a token, not a proof: the watermark binds "this audio, right now"
to a slot; the Ed25519 attestation from the verifier is the cryptography.
"""

import hashlib
import hmac

TOKEN_BITS = 16


def current_slot(now: float, slot_seconds: float) -> int:
    return int(now // slot_seconds)


def token_for_slot(key: bytes, slot: int) -> int:
    digest = hmac.new(key, slot.to_bytes(8, "big"), hashlib.sha256).digest()
    return int.from_bytes(digest[:2], "big")


def verify_token(
    key: bytes,
    token: int,
    now: float,
    slot_seconds: float,
    tolerance: int = 1,
    max_bit_errors: int = 0,
) -> int | None:
    """max_bit_errors absorbs single-bit watermark decode errors near chunk
    boundaries; at 1 of 16 bits over 3 candidate slots, a random forgery still
    only passes ~0.08% of the time."""
    center = current_slot(now, slot_seconds)
    for slot in range(center - tolerance, center + tolerance + 1):
        bit_errors = (token_for_slot(key, slot) ^ token).bit_count()
        if bit_errors <= max_bit_errors:
            return slot
    return None
