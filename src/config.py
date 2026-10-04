"""Central configuration for the pain-EEG decoding project.

Everything the pipeline needs to know about the dataset and the analysis
choices lives here, so the rest of the code stays clean and reproducible.

Dataset: OpenNeuro ds005284 ("26 By Biosemi"), CC0.
  - 26 healthy participants
  - 64-channel BioSemi EEG, sampled at 1024 Hz
  - 16 laser (nociceptive) stimuli per participant, delivered at a fixed
    intensity calibrated to ~7/10 subjective pain
  - Event code "condition 54" marks each laser onset
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ds005284"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TASK = "26ByBiosemi"
SFREQ_ORIG = 1024.0          # Hz, original sampling rate
LASER_CODE = "condition 54"  # event 'value' that marks a laser stimulus

# Subjects to use. Start with a subset to keep memory/time low on a 16 GB
# laptop, then scale up once the pipeline runs end-to-end.
SUBJECTS = [f"sub-{i:03d}" for i in range(1, 13)]   # sub-001 .. sub-012
ALL_SUBJECTS = [f"sub-{i:03d}" for i in range(1, 27)]

# ---------------------------------------------------------------------------
# Preprocessing choices
# ---------------------------------------------------------------------------
L_FREQ = 1.0        # high-pass (Hz) - removes slow drifts
H_FREQ = 45.0       # low-pass (Hz)  - removes muscle/line noise above gamma
NOTCH = 50.0        # mains frequency in China (dataset origin)
RESAMPLE = 256.0    # Hz, downsample after filtering to save memory

# Epoching, relative to laser onset (t = 0 s)
TMIN, TMAX = -1.0, 1.5      # seconds around each laser onset
BASELINE_WIN = (-1.0, -0.2)  # pre-stimulus window used as the "no-pain" class
RESPONSE_WIN = (0.15, 0.95)  # post-stimulus window used as the "pain" class

# ---------------------------------------------------------------------------
# Frequency bands (Hz) used for band-power features
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
