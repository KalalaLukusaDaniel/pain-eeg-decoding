"""Model definitions: classical ML baselines and a small EEGNet.

The classical models operate on band-power features (see features.py) and are
wrapped in a Pipeline that standardises inside each CV fold. EEGNet operates on
the raw epoch segments and is the Day-3 / Week-3 "deep learning" bonus.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from . import config as cfg


def classical_models() -> dict:
    """Return {name: sklearn Pipeline} for the ML baselines."""
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "LogisticRegression": pipe(
            LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_STATE)
        ),
        "LDA": pipe(LinearDiscriminantAnalysis()),
        "SVM-RBF": pipe(
            SVC(kernel="rbf", probability=True, random_state=cfg.RANDOM_STATE)
        ),
        "RandomForest": pipe(
            RandomForestClassifier(
                n_estimators=300, random_state=cfg.RANDOM_STATE, n_jobs=-1
            )
        ),
    }


# ---------------------------------------------------------------------------
# EEGNet (compact CNN for EEG). Imported lazily so the classical pipeline
# runs even without torch installed.
# ---------------------------------------------------------------------------
def build_eegnet(n_channels: int, n_times: int, n_classes: int = 2):
    """Return a small EEGNet-v2 style torch model.

    Reference: Lawhern et al., 2018, "EEGNet: a compact convolutional network
    for EEG-based brain-computer interfaces."
    """
    import torch.nn as nn

    F1, D, F2 = 8, 2, 16
    kern = 64

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(0.5),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                          groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(0.5),
            )
            self.classify = nn.Sequential(nn.Flatten(), nn.LazyLinear(n_classes))

        def forward(self, x):          # x: (batch, 1, channels, time)
            x = self.block1(x)
            x = self.block2(x)
            return self.classify(x)

    return EEGNet()
