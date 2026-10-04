"""Train EEGNet with Leave-One-Subject-Out CV and compare to the ML baselines.

Run from the project root:
    python -m src.run_eegnet
"""
from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    recall_score, confusion_matrix,
)

from . import config as cfg
from . import dataset as ds
from . import models, evaluation

torch.manual_seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60
LR = 1e-3


def standardize(train, test):
    """Per-channel z-score, fit on train only (no leakage)."""
    mu = train.mean(axis=(0, 2), keepdims=True)
    sd = train.std(axis=(0, 2), keepdims=True) + 1e-7
    return (train - mu) / sd, (test - mu) / sd


def train_fold(Xtr, ytr, Xte):
    n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
    model = models.build_eegnet(n_ch, n_t).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr[:, None, :, :], dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    Xte_t = torch.tensor(Xte[:, None, :, :], dtype=torch.float32, device=DEVICE)

    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()
    return prob


def main():
    _, Xraw, y, groups, ch_names, sfreq = ds.build_dataset()
    print(f"[eegnet] device={DEVICE}  X={Xraw.shape}")

    logo = LeaveOneGroupOut()
    y_prob = np.zeros(len(y))
    for k, (tr, te) in enumerate(logo.split(Xraw, y, groups), 1):
        Xtr, Xte = standardize(Xraw[tr], Xraw[te])
        y_prob[te] = train_fold(Xtr, y[tr], Xte)
        print(f"  fold {k:2d}/12  test={groups[te][0]}  done")

    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "recall": recall_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }
    print("\n=== EEGNet (LOSO) ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v:.3f}")

    # Merge with existing classical results and rewrite the comparison.
    with open(cfg.RESULTS_DIR / "metrics.json") as f:
        all_metrics = json.load(f)
    all_metrics["EEGNet"] = metrics
    with open(cfg.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    table = evaluation.results_table(all_metrics)
    table.to_csv(cfg.RESULTS_DIR / "model_comparison.csv", index=False)
    print("\n===== MODEL COMPARISON (LOSO, incl. EEGNet) =====")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
