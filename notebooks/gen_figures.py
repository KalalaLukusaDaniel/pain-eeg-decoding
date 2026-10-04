"""Generate the exploratory + results figures into results/figures/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config as cfg
from src import loading, preprocessing, epoching

SUBJECT = "sub-001"

raw = loading.load_raw(SUBJECT)
raw = preprocessing.preprocess(raw)
onsets = loading.load_laser_onsets(SUBJECT)
epochs = epoching.make_epochs(raw, onsets)
epochs.apply_baseline(cfg.BASELINE_WIN)
evoked = epochs.average()

# --- Fig 1: evoked laser response (the pain-evoked potential) ---
fig = evoked.plot(spatial_colors=True, gfp=True, show=False)
fig.suptitle(f"{SUBJECT}: EEG response to laser pain (avg of 16 trials)")
fig.savefig(cfg.FIG_DIR / "evoked_response.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Fig 2: topography at the N2/P2 peak ---
peak_ch, peak_time = evoked.get_peak(tmin=0.15, tmax=0.6)
fig = evoked.plot_topomap(times=[peak_time], show=False, colorbar=True)
fig.suptitle(f"Scalp map of the pain response at {peak_time:.2f}s")
fig.savefig(cfg.FIG_DIR / "topomap_peak.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Fig 3: Cz single-channel, pain window vs baseline window ---
times = epochs.times
data = epochs.get_data(copy=True)  # (trials, ch, time)
ch_idx = epochs.ch_names.index(peak_ch) if peak_ch in epochs.ch_names else 0
mean = data[:, ch_idx, :].mean(0) * 1e6
sem = data[:, ch_idx, :].std(0) / np.sqrt(len(data)) * 1e6
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.axvspan(*cfg.BASELINE_WIN, color="#9aa0a6", alpha=0.18, label="baseline window")
ax.axvspan(*cfg.RESPONSE_WIN, color="#d93025", alpha=0.12, label="response window")
ax.axvline(0, color="k", lw=0.8)
ax.plot(times, mean, color="#1a73e8", lw=1.8)
ax.fill_between(times, mean - sem, mean + sem, color="#1a73e8", alpha=0.2)
ax.set_xlabel("Time from laser onset (s)"); ax.set_ylabel(f"{peak_ch} (µV)")
ax.set_title(f"{SUBJECT}: laser-evoked response at {peak_ch}")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "cz_evoked.png", dpi=150)
plt.close(fig)

# --- Fig 4: model comparison bar chart ---
with open(cfg.RESULTS_DIR / "metrics.json") as f:
    metrics = json.load(f)
names = list(metrics.keys())
ba = [metrics[n]["balanced_accuracy"] for n in names]
order = np.argsort(ba)
names = [names[i] for i in order]; ba = [ba[i] for i in order]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
bars = ax.barh(names, ba, color="#1a73e8")
ax.axvline(0.5, color="#d93025", ls="--", lw=1, label="chance (0.5)")
for b, v in zip(bars, ba):
    ax.text(v + 0.01, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(0, 0.85); ax.set_xlabel("Balanced accuracy (LOSO)")
ax.set_title("Pain-response decoding — model comparison")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(cfg.FIG_DIR / "model_comparison.png", dpi=150)
plt.close(fig)

print("peak_ch:", peak_ch, "peak_time:", round(float(peak_time), 3))
print("Figures saved to", cfg.FIG_DIR)
print("Files:", sorted(p.name for p in cfg.FIG_DIR.glob("*.png")))
