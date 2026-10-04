"""Leakage-free, participant-level evaluation.

The golden rule of this project: a participant must never appear in both the
training and the test set. We use Leave-One-Subject-Out (LOSO) cross-
validation, and every transformation (scaling) is fit on the training fold
only, inside an sklearn Pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    recall_score, confusion_matrix,
)


def loso_evaluate(pipeline, X, y, groups):
    """Run LOSO CV and return (metrics_dict, y_pred, y_prob)."""
    logo = LeaveOneGroupOut()

    y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=logo)
    try:
        y_prob = cross_val_predict(
            pipeline, X, y, groups=groups, cv=logo, method="predict_proba"
        )[:, 1]
    except Exception:
        y_prob = None

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
        "recall": recall_score(y, y_pred),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_prob)
    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    return metrics, y_pred, y_prob


def results_table(all_metrics: dict) -> pd.DataFrame:
    """Turn {model_name: metrics_dict} into a tidy comparison table."""
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "model": name,
            "balanced_accuracy": round(m.get("balanced_accuracy", float("nan")), 3),
            "f1_macro": round(m.get("f1_macro", float("nan")), 3),
            "recall": round(m.get("recall", float("nan")), 3),
            "roc_auc": round(m.get("roc_auc", float("nan")), 3)
                if "roc_auc" in m else None,
        })
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
