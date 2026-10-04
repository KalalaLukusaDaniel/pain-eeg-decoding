"""Load one participant's EEG recording and laser-stimulus events.

We read straight from the BIDS layout of ds005284:

    data/ds005284/sub-XXX/eeg/sub-XXX_task-26ByBiosemi_eeg.bdf
    data/ds005284/sub-XXX/eeg/sub-XXX_task-26ByBiosemi_events.tsv

The events.tsv file lists every marker with its onset in seconds; we keep
only the laser-stimulus markers (value == "condition 54").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import mne

from . import config as cfg


def subject_paths(subject: str):
    """Return (bdf_path, events_tsv_path) for a subject id like 'sub-001'."""
    eeg_dir = cfg.DATA_DIR / subject / "eeg"
    bdf = eeg_dir / f"{subject}_task-{cfg.TASK}_eeg.bdf"
    events_tsv = eeg_dir / f"{subject}_task-{cfg.TASK}_events.tsv"
    return bdf, events_tsv


def load_raw(subject: str) -> mne.io.BaseRaw:
    """Load the continuous recording for one subject as an MNE Raw object."""
    bdf, _ = subject_paths(subject)
    if not bdf.exists():
        raise FileNotFoundError(
            f"Missing EEG file for {subject}: {bdf}\n"
            "Download the dataset first (see data/DATA.md)."
        )
    raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="ERROR")

    # Keep only EEG channels (drop Status/trigger and any non-scalp channels).
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    raw.pick(picks)

    # The raw BioSemi file labels channels A1..A32, B1..B32, but the BIDS
    # channels.tsv lists the real 10-20 names (FP1, AF7, Cz, ...) in the same
    # order. Rename so topographies and band maps are interpretable.
    ch_tsv = cfg.DATA_DIR / subject / "eeg" / f"{subject}_task-{cfg.TASK}_channels.tsv"
    if ch_tsv.exists():
        names = pd.read_csv(ch_tsv, sep="\t")["name"].astype(str).tolist()
        if len(names) == len(raw.ch_names):
            raw.rename_channels(dict(zip(raw.ch_names, names)))

    # Attach a standard 10-20 montage so we can draw topographies later.
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception as err:  # pragma: no cover - montage is best-effort
        print(f"[loading] montage not set for {subject}: {err}")
    return raw


def load_laser_onsets(subject: str) -> np.ndarray:
    """Return laser-stimulus onset times (in seconds) for one subject."""
    _, events_tsv = subject_paths(subject)
    df = pd.read_csv(events_tsv, sep="\t")
    laser = df[df["value"].astype(str).str.strip() == cfg.LASER_CODE]
    onsets = laser["onset"].to_numpy(dtype=float)
    return onsets
