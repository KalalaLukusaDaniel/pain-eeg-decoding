"""01 - Exploratory analysis of one participant.

Run from the project root once at least one .bdf file is downloaded:
    python notebooks/01_exploration.py

Produces, in results/figures/:
  - raw_segment.png       a few seconds of raw EEG (several channels)
  - evoked_response.png   average EEG locked to the laser onset (the LEP)
  - topomap.png           scalp map of the evoked response around its peak
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src import loading, preprocessing, epoching

SUBJECT = "sub-001"


def main():
    raw = loading.load_raw(SUBJECT)
    raw = preprocessing.preprocess(raw)
    onsets = loading.load_laser_onsets(SUBJECT)
    print(f"{SUBJECT}: {len(onsets)} laser trials, {len(raw.ch_names)} channels, "
          f"{raw.info['sfreq']:.0f} Hz")

    # 1) Raw signal snippet
    fig = raw.copy().pick(raw.ch_names[:6]).plot(
        duration=5, start=float(onsets[0]) - 1, show=False, scalings="auto")
    fig.savefig(cfg.FIG_DIR / "raw_segment.png", dpi=150)
    plt.close(fig)

    # 2) Evoked laser response (average over trials)
    epochs = epoching.make_epochs(raw, onsets)
    epochs.apply_baseline(cfg.BASELINE_WIN)
    evoked = epochs.average()
    fig = evoked.plot(spatial_colors=True, show=False)
    fig.savefig(cfg.FIG_DIR / "evoked_response.png", dpi=150)
    plt.close(fig)

    # 3) Topography around the evoked peak
    peak_ch, peak_time = evoked.get_peak(tmin=0.1, tmax=0.6)
    fig = evoked.plot_topomap(times=[peak_time], show=False)
    fig.savefig(cfg.FIG_DIR / "topomap.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {cfg.FIG_DIR} (peak at {peak_time:.2f}s on {peak_ch})")


if __name__ == "__main__":
    main()
