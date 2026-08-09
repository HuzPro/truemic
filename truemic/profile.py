"""Enrollment profile on disk: who the speaker is, which engine verifies them,
the calibrated open/close thresholds, the shared token key, and the
attestation keypair.

The receiver reads the same directory (copy it to a second machine for a
two-laptop demo). Nothing here authenticates enrollment itself — that is a
named limitation, not an oversight.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from truemic import config

PROFILE_FILE = "profile.json"
TOKEN_KEY_FILE = "token.key"
SIGNING_KEY_FILE = "signing.key"
EAGLE_PROFILE_FILE = "eagle.profile"
EMBEDDING_FILE = "embedding.npy"
MIC_PRINT_FILE = "mic_print.npy"


@dataclass
class Profile:
    directory: Path
    name: str
    engine: str
    open_threshold: float
    close_threshold: float
    token_key: bytes
    recent_close_threshold: float | None = None  # 1 s scores have their own scale

    @property
    def signing_key(self) -> Ed25519PrivateKey:
        raw = (self.directory / SIGNING_KEY_FILE).read_bytes()
        return Ed25519PrivateKey.from_private_bytes(raw)

    @property
    def eagle_profile_bytes(self) -> bytes:
        return (self.directory / EAGLE_PROFILE_FILE).read_bytes()

    @property
    def embedding(self) -> np.ndarray:
        return np.load(self.directory / EMBEDDING_FILE)

    @property
    def mic_print(self) -> np.ndarray | None:
        path = self.directory / MIC_PRINT_FILE
        return np.load(path) if path.exists() else None

    def save_thresholds(
        self,
        open_threshold: float,
        close_threshold: float,
        recent_close_threshold: float | None = None,
    ) -> None:
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.recent_close_threshold = recent_close_threshold
        _write_json(self.directory, self)


def create(directory: Path, name: str, engine: str, open_threshold: float) -> Profile:
    import os

    directory.mkdir(parents=True, exist_ok=True)
    (directory / TOKEN_KEY_FILE).write_bytes(os.urandom(32))
    signing_key = Ed25519PrivateKey.generate()
    (directory / SIGNING_KEY_FILE).write_bytes(
        signing_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    profile = Profile(
        directory=directory,
        name=name,
        engine=engine,
        open_threshold=open_threshold,
        close_threshold=open_threshold - config.THRESHOLD_CLOSE_GAP,
        token_key=(directory / TOKEN_KEY_FILE).read_bytes(),
    )
    _write_json(directory, profile)
    return profile


def load(directory: Path = config.DEFAULT_PROFILE_DIR) -> Profile:
    data = json.loads((directory / PROFILE_FILE).read_text())
    return Profile(
        directory=directory,
        name=data["name"],
        engine=data["engine"],
        open_threshold=data["open_threshold"],
        close_threshold=data["close_threshold"],
        token_key=(directory / TOKEN_KEY_FILE).read_bytes(),
        recent_close_threshold=data.get("recent_close_threshold"),
    )


def _write_json(directory: Path, profile: Profile) -> None:
    (directory / PROFILE_FILE).write_text(
        json.dumps(
            {
                "name": profile.name,
                "engine": profile.engine,
                "open_threshold": profile.open_threshold,
                "close_threshold": profile.close_threshold,
                "recent_close_threshold": profile.recent_close_threshold,
            },
            indent=2,
        )
    )
