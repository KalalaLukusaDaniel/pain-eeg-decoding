"""End-to-end classical ML baseline, evaluated with Leave-One-Subject-Out CV.

Run from the project root:
    python -m src.run_baseline
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as cfg
from . import dataset as ds
from . import models, evaluation


def plot_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["baseline", "pain"])
    ax.set_yticklabels(["baseline", "pain"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    X, Xraw, y, groups, ch_names, sfreq = ds.build_dataset()

    all_metrics = {}
    for name, pipe in models.classical_models().items():
        print(f"\n=== {name} (LOSO) ===")
        metrics, y_pred, y_prob = evaluation.loso_evaluate(pipe, X, y, groups)
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k}: {v:.3f}")
        all_metrics[name] = metrics
        plot_confusion(
            metrics["confusion_matrix"], name,
            cfg.FIG_DIR / f"confusion_{name}.png",
        )

    table = evaluation.results_table(all_metrics)
    print("\n===== MODEL COMPARISON (LOSO) =====")
    print(table.to_string(index=False))

    table.to_csv(cfg.RESULTS_DIR / "model_comparison.csv", index=False)
    with open(cfg.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved results to {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
