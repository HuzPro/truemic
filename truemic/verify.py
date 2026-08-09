"""Speaker verification engines behind one small interface, plus silero VAD.

Scoring silence produces garbage similarity scores, so the sender only asks
for a score when the VAD says the rolling window actually contains voice.

Engines: campp (WeSpeaker CAM++ LM via ONNX Runtime — default, fastest and
most accurate), ecapa (SpeechBrain, torch), eagle (Picovoice, needs key).
"""

import os

import numpy as np

from truemic import config
from truemic.profile import Profile


class VoiceActivity:
    """Streaming: feed() every capture block, query voiced fractions over
    trailing windows. voiced_fraction() stays for offline use (enroll,
    calibrate) — don't mix both styles on one instance."""

    def __init__(self):
        import torch

        threads_before = torch.get_num_threads()
        from silero_vad import load_silero_vad  # sets torch threads to 1 at import

        self._model = load_silero_vad(onnx=True)
        torch.set_num_threads(threads_before)
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames: list[tuple[int, bool]] = []  # (end sample index, voiced)
        self._samples_fed = 0

    def feed(self, block: np.ndarray) -> list[tuple[np.ndarray, bool]]:
        import torch

        frame_size = config.VAD_FRAME_SAMPLES
        self._pending = np.concatenate([self._pending, block])
        processed: list[tuple[np.ndarray, bool]] = []
        with torch.no_grad():
            while len(self._pending) >= frame_size:
                frame, self._pending = self._pending[:frame_size], self._pending[frame_size:]
                self._samples_fed += frame_size
                prob = self._model(torch.from_numpy(frame), config.SAMPLE_RATE).item()
                voiced = prob >= config.VAD_SPEECH_PROBABILITY
                self._frames.append((self._samples_fed, voiced))
                processed.append((frame, voiced))
        horizon = int(config.WINDOW_SECONDS * config.SAMPLE_RATE) + frame_size
        while self._frames and self._samples_fed - self._frames[0][0] > horizon:
            self._frames.pop(0)
        return processed

    def recent_voiced_fraction(self, seconds: float) -> float:
        span = int(seconds * config.SAMPLE_RATE)
        recent = [voiced for end, voiced in self._frames if self._samples_fed - end < span]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    def voiced_only(self, recording: np.ndarray) -> np.ndarray:
        """Concatenate just the clearly-voiced frames. Three gates, because a
        close mic fools each one alone: silero (calls breath 'speech'),
        spectral flatness (breath/hiss is flat, voiced speech is harmonic),
        and energy (breath is quiet relative to the speaker's own speech)."""
        import torch

        frame_size = config.VAD_FRAME_SAMPLES
        self._model.reset_states()
        frames, probs = [], []
        with torch.no_grad():
            for i in range(len(recording) // frame_size):
                frame = recording[i * frame_size : (i + 1) * frame_size]
                probs.append(self._model(torch.from_numpy(frame), config.SAMPLE_RATE).item())
                frames.append(frame)
        self._model.reset_states()
        if not frames:
            return np.zeros(0, dtype=np.float32)

        stack = np.stack(frames)
        spectra = np.abs(np.fft.rfft(stack, axis=1)) + 1e-10
        flatness = np.exp(np.mean(np.log(spectra), axis=1)) / spectra.mean(axis=1)
        rms = np.sqrt(np.mean(stack**2, axis=1))
        voiced = np.asarray(probs) >= config.VAD_SPEECH_PROBABILITY
        harmonic = flatness < config.ENROLL_MAX_FLATNESS
        candidates = voiced & harmonic
        if not candidates.any():
            return np.zeros(0, dtype=np.float32)
        loud = rms >= np.percentile(rms[candidates], config.ENROLL_ENERGY_PERCENTILE)
        kept = stack[candidates & loud]
        if len(kept) == 0:
            kept = stack[candidates]
        return kept.reshape(-1)

    def voiced_fraction(self, window: np.ndarray) -> float:
        import torch

        self._model.reset_states()
        frame_size = config.VAD_FRAME_SAMPLES
        n_frames = len(window) // frame_size
        if n_frames == 0:
            return 0.0
        voiced = 0
        with torch.no_grad():
            for i in range(n_frames):
                frame = window[i * frame_size : (i + 1) * frame_size]
                prob = self._model(torch.from_numpy(frame), config.SAMPLE_RATE).item()
                if prob >= config.VAD_SPEECH_PROBABILITY:
                    voiced += 1
        self._model.reset_states()
        return voiced / n_frames


class CamppVerifier:
    """WeSpeaker CAM++ LM (512-dim) in ONNX Runtime; one-sided AS-Norm when a
    cohort ships alongside the model.

    The enrollment is a MATRIX of chunk embeddings, scored as the mean of the
    top-K cosines: a speaker's delivery varies, and matching the closest
    enrolled moments beats matching a blurred average of all of them."""

    def __init__(self, enrolled: np.ndarray, cohort: np.ndarray | None = None):
        self._session = _campp_session()
        self._enrolled = enrollment_matrix(enrolled)
        self._cohort = cohort

    def feed(self, block: np.ndarray) -> None:
        pass

    def score(self, window: np.ndarray) -> float | None:
        embedding = campp_embed(self._session, window)
        cosine = top_k_score(self._enrolled, embedding)
        if self._cohort is None:
            return cosine
        return as_norm(cosine, embedding, self._cohort)

    def embed(self, wav: np.ndarray) -> np.ndarray:
        return campp_embed(self._session, wav)


class EcapaVerifier:
    """SpeechBrain ECAPA fallback (torch)."""

    def __init__(self, enrolled: np.ndarray):
        self._model = load_ecapa()
        self._enrolled = enrollment_matrix(enrolled)

    def feed(self, block: np.ndarray) -> None:
        pass

    def score(self, window: np.ndarray) -> float | None:
        return top_k_score(self._enrolled, ecapa_embed(self._model, window))

    def embed(self, wav: np.ndarray) -> np.ndarray:
        return ecapa_embed(self._model, wav)


class EagleVerifier:
    """pveagle 2.x: process(pcm, speaker_profiles) on accumulated chunks;
    returns None when there is not enough voice.

    ponytail: semantics (statefulness, None behavior) unvalidated without an
    AccessKey — smoke-test with the real key before the demo, per HANDOFF.
    """

    def __init__(self, access_key: str, profile_bytes: bytes):
        import pveagle

        self._speaker = pveagle.EagleProfile.from_bytes(profile_bytes)
        self._eagle = pveagle.create_recognizer(access_key=access_key)
        self._pending = np.zeros(0, dtype=np.int16)
        self._latest: tuple[float, float] | None = None  # (monotonic time, score)

    def feed(self, block: np.ndarray) -> None:
        import time

        pcm = (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16)
        self._pending = np.concatenate([self._pending, pcm])
        chunk_samples = max(self._eagle.min_process_samples, config.SAMPLE_RATE // 2)
        if len(self._pending) < chunk_samples:
            return
        result = self._eagle.process(self._pending.tolist(), [self._speaker])
        self._pending = np.zeros(0, dtype=np.int16)
        if result:
            self._latest = (time.monotonic(), float(result[0]))

    def score(self, window: np.ndarray) -> float | None:
        import time

        if self._latest is None:
            return None
        scored_at, score = self._latest
        if time.monotonic() - scored_at > config.WINDOW_SECONDS:
            return None
        return score

    def embed(self, wav: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Eagle profiles are opaque — no embeddings")


def enrollment_matrix(enrolled: np.ndarray) -> np.ndarray:
    """Accept both the old single-vector prints and the new chunk matrices."""
    matrix = np.atleast_2d(enrolled)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def top_k_score(matrix: np.ndarray, embedding: np.ndarray) -> float:
    cosines = matrix @ embedding
    top = np.sort(cosines)[-config.ENROLL_TOP_K :]
    return float(top.mean())


def as_norm(cosine: float, embedding: np.ndarray, cohort: np.ndarray) -> float:
    """One-sided adaptive score normalization: z-score the raw cosine against
    the top-N most similar cohort speakers. Makes a threshold calibrated in
    one room survive another."""
    cohort_scores = cohort @ embedding
    top = np.sort(cohort_scores)[-config.AS_NORM_COHORT_TOP :]
    spread = float(top.std())
    if spread < 1e-6:
        return cosine
    return (cosine - float(top.mean())) / spread


def campp_features(wav: np.ndarray) -> np.ndarray:
    """Front end for the sherpa-onnx CAM++ export: int16-range wav -> 80-dim
    kaldi fbank (hamming, no dither). NO CMN — measured on LibriSpeech pairs:
    without CMN same/cross speaker cosine is 0.86/0.29; adding CMN (WeSpeaker's
    training-recipe step) collapses it to 0.87/0.91, i.e. zero discrimination.
    The export's metadata (no feature_normalize_type) is the authority."""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    waveform = torch.from_numpy(wav.astype(np.float32) * (1 << 15))[None]
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=config.SAMPLE_RATE,
        window_type="hamming",
        use_energy=False,
    )
    return mat.numpy()[None]  # (1, T, 80)


def campp_embed(session, wav: np.ndarray) -> np.ndarray:
    embedding = session.run(None, {"feats": campp_features(wav)})[0].squeeze()
    return _normalized(embedding)


def _campp_session():
    import onnxruntime as ort

    if not config.CAMPP_MODEL_FILE.exists():
        _download_campp()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(config.CAMPP_MODEL_FILE), options, providers=["CPUExecutionProvider"]
    )


def _download_campp() -> None:
    import urllib.request

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CAM++ model (29 MB) from sherpa-onnx releases...")
    partial = config.CAMPP_MODEL_FILE.with_suffix(".onnx.tmp")
    urllib.request.urlretrieve(config.CAMPP_MODEL_URL, partial)
    partial.replace(config.CAMPP_MODEL_FILE)  # a dropped connection can't brick later runs


def load_cohort() -> np.ndarray | None:
    if config.COHORT_FILE.exists():
        return np.load(config.COHORT_FILE)
    return None


def ecapa_embed(model, wav: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        embedding = model.encode_batch(torch.from_numpy(wav)[None])
    return _normalized(embedding.squeeze().cpu().numpy())


def load_ecapa():
    _shim_torch_distributed()
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
        savedir=str(config.MODELS_DIR / "ecapa"),
        local_strategy=LocalStrategy.COPY_SKIP_CACHE,  # Windows symlink bug speechbrain#1155
    )


def create_verifier(profile: Profile):
    if profile.engine == "eagle":
        return EagleVerifier(require_access_key(), profile.eagle_profile_bytes)
    if profile.engine == "ecapa":
        return EcapaVerifier(profile.embedding)
    return CamppVerifier(profile.embedding, load_cohort())


def _shim_torch_distributed() -> None:
    """Windows ROCm torch ships without distributed; speechbrain imports
    ReduceOp unconditionally but only uses it under DDP training."""
    import torch.distributed as dist

    if not hasattr(dist, "ReduceOp"):
        class ReduceOp:
            SUM = "sum"

        dist.ReduceOp = ReduceOp


def require_access_key() -> str:
    access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
    if not access_key:
        raise RuntimeError(
            "PICOVOICE_ACCESS_KEY is not set. Eagle validates the key over the "
            "internet — check this BEFORE the demo, or use --engine campp."
        )
    return access_key


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
