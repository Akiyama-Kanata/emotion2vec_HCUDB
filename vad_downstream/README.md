# VAD downstream data contract

This directory is for downstream experiments that use emotion2vec frame-level
features with continuous Valence/Arousal/Dominance style labels.

This README defines the data contract and tracks the current minimal
implementation. The implemented path is precomputed emotion2vec frame features
to VA/VAD regression with CCC loss. It does not yet include fine-tuning code,
dataset-specific preprocessing, or a full validation/checkpoint workflow.

Current minimal modules:

- `data.py`: loads `<prefix>.npy`, `<prefix>.lengths`, and `<prefix>.vad`.
- `model.py`: defines a padded frame-level `VADRegressionHead` and an optional
  `Emotion2vecVADModel` wrapper for waveform-to-regression experiments.
- `training.py`: provides CCC loss and a one-epoch training helper.

## Required files

Use one shared prefix for each split or dataset. For example, `train` means the
following files live next to each other:

| File | Required | Description |
|---|---:|---|
| `<prefix>.npy` | yes | Frame-level emotion2vec features concatenated over all utterances. Shape: `(total_frames, 768)`. |
| `<prefix>.lengths` | yes | One integer per line. Each value is the frame count for one utterance. |
| `<prefix>.vad` | yes | Continuous affect labels. Valence and Arousal are required; Dominance is optional. |
| `<prefix>.emo` | no | Optional categorical emotion labels in the existing IEMOCAP-style format. |

The `.npy` and `.lengths` files follow the existing `iemocap_downstream` feature
layout: utterance feature matrices are stacked along the frame axis, and
`.lengths` is used to recover utterance boundaries.

## VAD label format

The continuous label file uses tab-separated text.

Minimum VA format:

```text
utterance_id<TAB>valence<TAB>arousal
```

Optional VAD format:

```text
utterance_id<TAB>valence<TAB>arousal<TAB>dominance
```

Rules:

- `valence` and `arousal` are always required.
- `dominance` is optional. If present, it must be present for every row in the
  same file.
- Label values must be normalized to `[-1.0, 1.0]`.
- Raw 1-to-5 ratings must be normalized with `(raw - 3.0) / 2.0`.
- Missing values are not part of the first supported format.

Example:

```text
Ses01F_impro01_F000	-0.50	0.25
Ses01F_impro01_F001	0.10	0.40
```

Example with Dominance:

```text
Ses01F_impro01_F000	-0.50	0.25	-0.10
Ses01F_impro01_F001	0.10	0.40	0.20
```

## Optional categorical labels

When categorical emotion labels are needed, use the existing `.emo` format:

```text
utterance_id<TAB>class
```

Example:

```text
Ses01F_impro01_F000	ang
Ses01F_impro01_F001	neu
```

The `.emo` file is optional for VA/VAD regression. It can be used later for a
classification stage that combines continuous affect labels and categorical
emotion labels.

## Alignment requirements

The first data loader should assume:

- `.lengths` has one line per utterance.
- `.vad` has the same number of rows as `.lengths`.
- If `.emo` is used, it has the same number of rows as `.lengths` and `.vad`.
- Row order is the source of truth. Row `i` in `.vad` describes utterance `i`
  recovered from `.npy` using `.lengths`.
- `utterance_id` values in `.vad` and `.emo` must match when both files are
  present.
- The sum of all `.lengths` values equals the first dimension of `.npy`.
- Each utterance feature matrix has 768 columns.

## First data loader assumptions

The first implementation should stay small:

- Load `<prefix>.npy`, `<prefix>.lengths`, and `<prefix>.vad`.
- Return padded frame-level features, a padding mask, and normalized VA or VAD
  targets.
- Detect whether the target dimensionality is 2 or 3 from the `.vad` file.
- Treat categorical `.emo` labels as optional and leave classification-specific
  logic for a later change.

## Not implemented in this step

- VAD-assisted categorical classifier.
- emotion2vec fine-tuning.
- Full validation loop, scheduler, checkpoint saving, and experiment logging.
- WAV-path dataset or real checkpoint loading for the whole model.
- IEMOCAP mixed training or Japanese dataset preprocessing.
- Conversion scripts from raw annotation files.
- Changes to `requirements.txt`.
