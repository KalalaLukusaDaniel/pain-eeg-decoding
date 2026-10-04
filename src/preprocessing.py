"""Minimal, reproducible preprocessing.

Kept deliberately light for a 3-day project: band-pass filter, notch out the
mains frequency, and downsample to save memory. Heavy artefact removal (ICA)
is intentionally out of scope but noted in the report as a limitation.
"""
from __future__ import annotations

import mne

from . import config as cfg


def preprocess(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Filter, notch and resample a Raw object in place; return it."""
    raw.filter(l_freq=cfg.L_FREQ, h_freq=cfg.H_FREQ, verbose="ERROR")
    # Notch the mains line (50 Hz) plus its first harmonic if still in band.
    freqs = [cfg.NOTCH]
    raw.notch_filter(freqs=freqs, verbose="ERROR")
    # Average reference is a standard, leakage-free choice for scalp EEG.
    raw.set_eeg_reference("average", verbose="ERROR")
    if cfg.RESAMPLE and cfg.RESAMPLE < raw.info["sfreq"]:
        raw.resample(cfg.RESAMPLE, verbose="ERROR")
    return raw
