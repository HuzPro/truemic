"""Build models/cohort.npy for AS-Norm: one CAM++ embedding per LibriSpeech
dev-clean speaker (40 speakers, CC-BY 4.0, natively 16 kHz).

Build-time tool — run once on a machine with internet; the resulting ~80 KB
file ships with the repo and the runtime never needs the corpus.
"""

import io
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from truemic import config

DEV_CLEAN_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
UTTERANCES_PER_SPEAKER = 3
MIN_SECONDS = 4.0
CROP_SECONDS = 3.0


def main() -> None:
    import argparse

    import soundfile

    from truemic.verify import _campp_session, campp_embed

    parser = argparse.ArgumentParser(description="Build the AS-Norm cohort")
    parser.add_argument("--tar", default=None, help="existing LibriSpeech tar.gz to use")
    args = parser.parse_args()

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.tar:
        archive = Path(args.tar)
    else:
        archive = config.MODELS_DIR / "dev-clean.tar.gz"
        if not archive.exists():
            print("Downloading LibriSpeech dev-clean (337 MB, one-time)...")
            urllib.request.urlretrieve(DEV_CLEAN_URL, archive)

    session = _campp_session()
    per_speaker: dict[str, list[np.ndarray]] = defaultdict(list)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                continue
            speaker = member.name.split("/")[2]
            if len(per_speaker[speaker]) >= UTTERANCES_PER_SPEAKER:
                continue
            wav, rate = soundfile.read(io.BytesIO(tar.extractfile(member).read()))
            assert rate == config.SAMPLE_RATE
            if len(wav) < MIN_SECONDS * rate:
                continue
            crop = wav[: int(CROP_SECONDS * rate)].astype(np.float32)
            per_speaker[speaker].append(campp_embed(session, crop))
            done = sum(len(v) for v in per_speaker.values())
            print(f"\r{len(per_speaker)} speakers, {done} utterances", end="", flush=True)

    cohort = np.stack(
        [_speaker_mean(embeddings) for embeddings in per_speaker.values()]
    )
    np.save(config.COHORT_FILE, cohort)
    print(f"\nWrote {config.COHORT_FILE} with {len(cohort)} speaker embeddings.")


def _speaker_mean(embeddings: list[np.ndarray]) -> np.ndarray:
    mean = np.stack(embeddings).mean(axis=0)
    return mean / np.linalg.norm(mean)


if __name__ == "__main__":
    main()
