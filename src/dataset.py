"""Assemble the full decoding dataset across participants.

Loops over the configured subjects, applies loading -> preprocessing ->
epoching -> feature extraction, and concatenates everything, while recording
which subject each sample came from (the "group" for leakage-free CV).
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from . import loading, preprocessing, epoching, features


def build_dataset(subjects=None, verbose=True):
    """Return X (features), Xraw (segments), y (labels), groups, ch_names, sfreq."""
    subjects = subjects or cfg.SUBJECTS
    X_feat, X_raw, y_all, groups = [], [], [], []
    ch_names, sfreq = None, None

    for subj in subjects:
        if verbose:
            print(f"[dataset] processing {subj} ...")
        raw = loading.load_raw(subj)
        raw = preprocessing.preprocess(raw)
        onsets = loading.load_laser_onsets(subj)
        epochs = epoching.make_epochs(raw, onsets)
        Xraw, y = epoching.response_vs_baseline(epochs)

        if ch_names is None:
            ch_names = epochs.ch_names
            sfreq = epochs.info["sfreq"]

        feats = features.band_power_features(Xraw, sfreq)
        X_feat.append(feats)
        X_raw.append(Xraw)
        y_all.append(y)
        groups.append(np.full(len(y), subj))

    X_feat = np.concatenate(X_feat, axis=0)
    X_raw = np.concatenate(X_raw, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    groups = np.concatenate(groups, axis=0)

    if verbose:
        print(f"[dataset] X={X_feat.shape}, y={y_all.shape}, "
              f"subjects={len(set(groups))}, pain={y_all.sum()}, "
              f"baseline={(y_all == 0).sum()}")
    return X_feat, X_raw, y_all, groups, ch_names, sfreq
