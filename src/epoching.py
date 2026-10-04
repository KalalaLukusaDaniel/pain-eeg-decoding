"""Turn a continuous recording into labelled epochs.

Decoding task (Option A): can we tell, from a short EEG segment, whether the
brain is *responding to a laser pain stimulus* or *at rest*?

For every laser onset we create two samples:
  - a POST-stimulus segment  -> label 1  ("pain response")
  - a PRE-stimulus segment    -> label 0  ("baseline / no pain")

Because the two come from the same trials and the same subject, the classifier
cannot cheat on slow drifts or per-subject offsets; it must find the actual
stimulus-evoked response. Subject identity is tracked as a "group" so that
train/test splits never mix a person across both sides.
"""
from __future__ import annotations

import numpy as np
import mne

from . import config as cfg


def make_epochs(raw: mne.io.BaseRaw, onsets: np.ndarray) -> mne.Epochs:
    """Build MNE Epochs centred on each laser onset."""
    sfreq = raw.info["sfreq"]
    samples = (onsets * sfreq).astype(int)
    events = np.column_stack(
        [samples, np.zeros_like(samples), np.ones_like(samples)]
    )
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"laser": 1},
        tmin=cfg.TMIN,
        tmax=cfg.TMAX,
        baseline=None,           # we handle baseline explicitly, no leakage
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
    return epochs


def response_vs_baseline(epochs: mne.Epochs):
    """Split each epoch into a post-stimulus and a pre-stimulus segment.

    Returns
    -------
    X : np.ndarray, shape (2 * n_trials, n_channels, n_times)
    y : np.ndarray, shape (2 * n_trials,)   1 = pain response, 0 = baseline
    """
    resp = epochs.copy().crop(*cfg.RESPONSE_WIN).get_data(copy=True)
    base = epochs.copy().crop(*cfg.BASELINE_WIN).get_data(copy=True)

    # Match the two windows to the same number of time samples.
    n = min(resp.shape[-1], base.shape[-1])
    resp, base = resp[..., :n], base[..., :n]

    X = np.concatenate([resp, base], axis=0)
    y = np.concatenate([np.ones(len(resp)), np.zeros(len(base))]).astype(int)
    return X, y
