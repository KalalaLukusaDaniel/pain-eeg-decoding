# Getting the data (OpenNeuro ds005284, CC0)

The raw EEG files are **not** stored in this repository (they are large and
belong to the original authors). Download them from OpenNeuro.

- Dataset page: https://openneuro.org/datasets/ds005284/versions/1.0.0
- License: CC0 (public domain) — no account or agreement required.
- Each participant is one `.bdf` file of ~65 MB (64 channels, 1024 Hz).

## Quick start (subset used by default: sub-001 … sub-012)

Download these files and place each one at
`data/ds005284/sub-XXX/eeg/sub-XXX_task-26ByBiosemi_eeg.bdf`.

Direct links (public S3, work in any browser):

- https://s3.amazonaws.com/openneuro.org/ds005284/sub-001/eeg/sub-001_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-002/eeg/sub-002_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-003/eeg/sub-003_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-004/eeg/sub-004_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-005/eeg/sub-005_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-006/eeg/sub-006_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-007/eeg/sub-007_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-008/eeg/sub-008_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-009/eeg/sub-009_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-010/eeg/sub-010_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-011/eeg/sub-011_task-26ByBiosemi_eeg.bdf
- https://s3.amazonaws.com/openneuro.org/ds005284/sub-012/eeg/sub-012_task-26ByBiosemi_eeg.bdf

The small metadata files (`events.tsv`, `channels.tsv`, `electrodes.tsv`,
`*.json`, `participants.tsv`) are lightweight and already tracked via the
OpenNeuro GitHub mirror, so only the `.bdf` signals need downloading.

## Full dataset

To use all 26 participants, download `sub-001 … sub-026` the same way, or use
the official OpenNeuro CLI: `npx @openneuro/cli download ds005284 data/ds005284`.
