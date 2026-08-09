"""Every runtime-tunable knob in one place.

SIMILARITY thresholds and VAD aggressiveness are properties of the room, mic,
and speaker — calibrate on-site with `python -m truemic.calibrate` and let the
profile override the defaults here.
"""

from pathlib import Path

SAMPLE_RATE = 16_000
BLOCK_SAMPLES = 480  # 30 ms

WINDOW_SECONDS = 3.0  # rolling verification window; 3 s is ~2/3 the EER of 2 s
RECENT_WINDOW_SECONDS = 1.0  # short close-path window; unreliable alone, safe fail-closed
DECIDE_EVERY_SECONDS = 0.5
GATE_DELAY_SECONDS = 1.5
CROSSFADE_MS = 20
SILENCE_TIMEOUT_SECONDS = 3.0
OPEN_CONFIRMATIONS = 2  # consecutive good decisions to open (VoiceFilter's 10:1 bias, inverted)
THRESHOLD_CLOSE_GAP = 0.10  # close = open - gap when calibration didn't supply one
MIN_VOICED_FRACTION = 0.4
VAD_FRAME_SAMPLES = 512  # silero's required frame at 16 kHz
VAD_SPEECH_PROBABILITY = 0.5
COMFORT_NOISE_DBFS = -55.0  # unwatermarked, post-gate; field floors are -12..-30 dB

DEFAULT_THRESHOLD_EAGLE = 0.50
DEFAULT_THRESHOLD_ECAPA = 0.40  # cosine similarity; genuine same-speaker sits ~0.6-0.8
DEFAULT_THRESHOLD_CAMPP = 0.55  # measured: genuine 0.82+-0.07, impostor 0.38+-0.07
ENROLL_TOP_K = 5  # score = mean cosine of the K best-matching enrollment chunks
ENROLL_MAX_FLATNESS = 0.3  # spectral flatness: voiced speech ~0.05-0.2, breath/hiss ~0.5+
ENROLL_ENERGY_PERCENTILE = 20  # drop the quietest 'voiced' frames (mostly breath)
ENROLL_SELF_TEST_MARGIN = 0.12  # held-out own audio must beat white noise by this much
CALIBRATE_IMPOSTOR_MIN_RMS_RATIO = 0.6  # quieter than this vs genuine = invalid impostor take
AS_NORM_COHORT_TOP = 15  # one-sided AS-Norm, engaged only if models/cohort.npy ships;
# measured d' 6.4 raw vs 3.9 with a 40-speaker cohort - needs hundreds of speakers to help

MIC_MATCH_THRESHOLD = 0.90  # cosine of quiet-frame log spectra; same mic sits ~0.97+
MIC_ZERO_FRACTION_ALARM = 0.8  # this much bit-exact zero input = injected source

WM_CHUNK_SECONDS = 0.5
WM_ALPHA = 3.0  # watermark strength; published sweep: 1.0 fragile, 3.0 robust, 5.0 audible risk
WM_TARGET_RMS = 0.1  # receiver-side level normalization (shipped detector does none)
TOKEN_SLOT_SECONDS = 8.0  # longer slots halve detect-window slot straddles
TOKEN_TOLERANCE = 1  # slots each side; covers clock skew + detection latency
TOKEN_MAX_BIT_ERRORS = 1  # single-tick grace (forgery ~0.08%/tick); the pooled-vote fallback stays exact
TOKEN_VOTE_TICKS = 3  # detection ticks pooled in the soft majority vote
WM_DETECT_THRESHOLD = 0.2  # fraction of flagged samples; benchmark FPR<1% operating point is 0.15

DETECT_WINDOW_SECONDS = 1.0
DETECT_HOP_SECONDS = 0.74  # 11840 = 37*320: hop stays on the detector's phase grid
BADGE_STALE_SECONDS = 3.5  # bridges speech pauses and single-tick phase re-locks
BADGE_PORT = 8765

DEFAULT_OUT_DEVICE = "CABLE Input"
DEFAULT_OUT_RATE = 48_000  # VB-CABLE cannot run 16 kHz internally; 48 k avoids hidden resample
DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent.parent / "profile"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

CAMPP_MODEL_FILE = MODELS_DIR / "wespeaker_en_voxceleb_CAM++_LM.onnx"
CAMPP_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B_LM.onnx"
)
COHORT_FILE = MODELS_DIR / "cohort.npy"

SENDER_TORCH_THREADS = 2  # oneDNN's 1-thread path is 6x slower, >2 buys little
RECEIVER_TORCH_THREADS = 2  # the 320-phase relock scan is batched conv - 1 thread starves it

PROFILES: dict[str, dict] = {
    "default": {},
    # 2-core U-class laptops (i5-7200U): halve the inference cadence.
    "low-power": {
        "DECIDE_EVERY_SECONDS": 1.0,
        "WM_CHUNK_SECONDS": 1.0,
        "DETECT_HOP_SECONDS": 1.0,
        "BADGE_STALE_SECONDS": 3.5,
    },
}


def apply_profile(name: str) -> None:
    for setting, value in PROFILES[name].items():
        globals()[setting] = value


def apply_torch_threads(threads: int) -> None:
    """Call after all models are loaded — model loaders fiddle with threads."""
    import torch

    torch.set_num_threads(threads)
