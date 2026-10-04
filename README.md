# Decoding the EEG response to nociceptive (pain) stimuli

Machine-learning and deep-learning decoding of the brain's response to painful
laser stimulation from scalp EEG, evaluated with strict **leave-one-subject-out**
cross-validation.

> **Scope & honesty.** This is a compact, reproducible portfolio project, not a
> medical tool. It does **not** diagnose pain or any clinical condition.

## Question

Given a short segment of 64-channel EEG, can a model tell whether the brain is
**responding to a laser pain stimulus** versus **at rest**? Every laser trial
yields a post-stimulus segment (label = *pain response*) and a pre-stimulus
segment (label = *baseline*), so the classifier must learn the actual
stimulus-evoked response rather than any per-person quirk.

## Data

- **OpenNeuro [ds005284](https://openneuro.org/datasets/ds005284)** — "26 By
  Biosemi", License **CC0**.
- 26 healthy adults, 64-channel BioSemi EEG at 1024 Hz.
- 16 laser stimuli per participant at a fixed, individually calibrated
  intensity (~7/10 subjective pain).
- Raw signals are **not** committed here — see [`data/DATA.md`](data/DATA.md).

## Method

1. **Preprocess** — 1–45 Hz band-pass, 50 Hz notch, average reference,
   downsample to 256 Hz.
2. **Epoch** — segments around each laser onset; post-stimulus vs pre-stimulus.
3. **Features** — log band power (delta, theta, alpha, beta, gamma) per channel.
4. **Classical models** — Logistic Regression, LDA, SVM (RBF), Random Forest.
5. **Deep learning (bonus)** — a compact EEGNet on the raw segments.
6. **Evaluation** — Leave-One-Subject-Out CV; scaling fit on the train fold
   only (no leakage). Reported: balanced accuracy, macro-F1, recall, ROC-AUC,
   confusion matrix.

## Repository layout

```
data/        download instructions (never raw data)
notebooks/   01_exploration — signals, evoked response, topographies
src/         config, loading, preprocessing, epoching, features,
             models, evaluation, run_baseline
results/     figures, model_comparison.csv, metrics.json
report/      short methods & results write-up
```

## Reproduce

```bash
pip install -r requirements.txt
# 1) download the .bdf files into data/ds005284/ (see data/DATA.md)
python -m src.run_baseline          # classical ML baselines, LOSO CV
```

## Key scientific choices

- **No subject leakage** — a participant is never in both train and test.
- **Leakage-free pipeline** — all transforms fit on training data only.
- **Honest metrics** — balanced accuracy and macro-F1 lead, given class and
  subject structure.

## Reference

Lu X. *et al.* (2019). *Music Reduces Pain Unpleasantness: Evidence from an EEG
Study.* J Pain Res. (Source study for ds005284.)
