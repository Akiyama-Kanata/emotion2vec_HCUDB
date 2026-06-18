# VAD downstream data contract

This directory is for downstream experiments that use emotion2vec frame-level
features with continuous Valence/Arousal/Dominance style labels.

This README defines the data contract and tracks the current minimal
implementation. The implemented path is precomputed emotion2vec frame features
to VA/VAD regression with CCC loss, validation CCC metrics, and head checkpoint
save/load. It does not yet include fine-tuning code or dataset-specific
preprocessing.

Current minimal modules:

- `data.py`: loads `<prefix>.npy`, `<prefix>.lengths`, and `<prefix>.vad`.
- `model.py`: defines a padded frame-level `VADRegressionHead` and an optional
  `Emotion2vecVADModel` wrapper for waveform-to-regression experiments.
- `training.py`: provides CCC loss, a one-epoch training helper, validation
  metrics, and head checkpoint saving.
- `train_head.py`: trains `VADRegressionHead` from `<prefix>.npy`,
  `<prefix>.lengths`, and `<prefix>.vad`.
- `inference.py`: provides WAV-to-VA/VAD JSON inference. It keeps the Stage 1
  placeholder path and can also load a real emotion2vec checkpoint when both
  `--model-dir` and `--checkpoint` are provided. It can load Stage 3 head
  checkpoints through `--head-checkpoint`. Without a head checkpoint, it only
  runs when `--allow-random-head` is set.

## WAV to VA/VAD staged plan

The WAV-to-VA/VAD path is intentionally staged.

Stage 1 is implemented as a wiring check. `vad_downstream/inference.py` accepts
a single WAV path and writes JSON with:

- `labels`
- `prediction`
- `head_checkpoint`
- `random_head`

If `--head-checkpoint` is omitted, the command fails by default. Passing
`--allow-random-head` explicitly allows an untrained random regression head and
marks the output with `"random_head": true`. These numbers are not research
results.

When `--model-dir` and `--checkpoint` are both omitted, inference uses the Stage
1 placeholder encoder.

Stage 2 is implemented as an optional real emotion2vec checkpoint path. When
both `--model-dir` and `--checkpoint` are provided, inference follows
`scripts/extract_features.py`: fairseq user module import, checkpoint loading,
`task.cfg.normalize`-controlled WAV normalization, and device selection. If only
one of the two arguments is provided, the command raises a clear `ValueError`.
The WAV contract remains 16kHz mono.

Stage 3 is implemented for head-only training. It trains `VADRegressionHead`
from `.npy/.lengths/.vad`, optionally evaluates on a validation prefix with
global CCC metrics, saves the best validation `mean_ccc` head when validation is
provided, and loads that checkpoint through `--head-checkpoint`. Stage 3
checkpoints include `target_dim`, `input_dim`, `hidden_dim`, and metadata; during
inference, `target_dim` is checked against the CLI value.

## Minimal commands

Train a head from precomputed frame features:

```bash
python -m vad_downstream.train_head \
  --train-prefix data/vad/train \
  --valid-prefix data/vad/valid \
  --output runs/vad_head.pt \
  --epochs 10 \
  --batch-size 32 \
  --device auto
```

Run WAV inference with the trained head:

```bash
python -m vad_downstream.inference \
  --wav scripts/test.wav \
  --target-dim 2 \
  --head-checkpoint runs/vad_head.pt \
  --output prediction.json \
  --device cpu
```

Add `--model-dir <MODEL_DIR> --checkpoint <CHECKPOINT>` to the inference command
when a real emotion2vec checkpoint is available. If those arguments are omitted,
the Stage 1 placeholder encoder is used for wiring checks only.

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
- Full scheduler and experiment logging.
- WAV-path dataset for training.
- IEMOCAP mixed training or Japanese dataset preprocessing.
- Conversion scripts from raw annotation files.
- Changes to `requirements.txt`.
