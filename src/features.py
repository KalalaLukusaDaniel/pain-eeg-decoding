"""Feature extraction: band power per channel.

For each segment (channels x time) we estimate the power spectrum with Welch's
method and integrate it inside each frequency band (delta ... gamma). The
result is a compact feature vector of length n_channels * n_bands that the
classical ML models consume.
"""
from __future__ import annotations

import numpy as np
from mne.time_frequency import psd_array_welch

from . import config as cfg


def band_power_features(X: np.ndarray, sfreq: float) -> np.ndarray:
    """Convert raw segments to band-power features.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_channels, n_times)
    sfreq : float, sampling rate of X

    Returns
    -------
    feats : np.ndarray, shape (n_samples, n_channels * n_bands)
            log band power, ordered channel-major then band.
    """
    n_fft = min(X.shape[-1], int(sfreq))  # ~1 s windows, capped by segment len
    psds, freqs = psd_array_welch(
        X, sfreq=sfreq, fmin=1.0, fmax=cfg.H_FREQ,
        n_fft=n_fft, verbose="ERROR",
    )
    # psds shape: (n_samples, n_channels, n_freqs)
    feats = []
    for name, (lo, hi) in cfg.BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        # Mean power in the band, per sample/channel -> (n_samples, n_channels)
        band = psds[:, :, mask].mean(axis=-1)
        feats.append(band)
    # Stack bands -> (n_samples, n_channels, n_bands) -> flatten last two dims
    feats = np.stack(feats, axis=-1)
    feats = feats.reshape(feats.shape[0], -1)
    # Log-transform: band power is heavily right-skewed.
    feats = np.log(feats + 1e-12)
    return feats


def feature_names(ch_names) -> list[str]:
    """Human-readable names matching the columns of band_power_features."""
    return [f"{ch}_{band}" for ch in ch_names for band in cfg.BANDS]
