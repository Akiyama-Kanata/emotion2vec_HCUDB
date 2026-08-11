# VAD downstream data contract

モデル構造、勾配経路、条件付きDominance学習の詳しい図解は
[`MODEL_ARCHITECTURES_JA.md`](MODEL_ARCHITECTURES_JA.md)を参照してください。

This directory is for downstream experiments that use emotion2vec frame-level
features with continuous Valence/Arousal/Dominance style labels.

This README defines the data contract and tracks the current minimal
implementation. The implemented path is precomputed emotion2vec frame features
to VA/VAD regression with CCC loss, validation CCC metrics, and head checkpoint
save/load. It does not yet include fine-tuning code or dataset-specific
preprocessing.

Current minimal modules:

- `data.py`: loads pure VAD regression data and aligned
  `<prefix>.npy/.lengths/.vad/.emo` VAD-emotion data.
- `model.py`: defines a padded frame-level `VADRegressionHead`, a
  VAD-mediated categorical classifier, and optional emotion2vec wrappers for
  waveform-to-regression and waveform-to-VAD-mediated-classification
  experiments.
- `training.py`: provides CCC loss, a one-epoch training helper, validation
  metrics, and head checkpoint saving.
- `emotion_training.py`: provides CCC + cross-entropy loss, joint validation
  metrics, classification WA/UA/weighted F1, confusion matrix, and checkpoint
  saving for VAD-mediated classification.
- `train_head.py`: trains `VADRegressionHead` from `<prefix>.npy`,
  `<prefix>.lengths`, and `<prefix>.vad`.
- `inference.py`: provides WAV-to-VA/VAD JSON inference. It keeps the Stage 1
  placeholder path and can also load a real emotion2vec checkpoint when both
  `--model-dir` and `--checkpoint` are provided. It can load Stage 3 head
  checkpoints through `--head-checkpoint`. Without a head checkpoint, it only
  runs when `--allow-random-head` is set.
- `train_vad_emotion.py`: trains `VADMediatedEmotionClassifier` from
  `<prefix>.npy`, `<prefix>.lengths`, `<prefix>.vad`, and `<prefix>.emo`.
- `infer_vad_emotion.py`: provides WAV-to-VAD-to-emotion JSON inference with
  linear-weight contribution breakdowns.
- `train_parallel_emotion_vad.py`: trains independent categorical and V/A/D
  heads with utterance-level D masking.
- `infer_parallel_emotion_vad.py`: restores class order and Dominance status
  from a checkpoint and always emits three VAD values.

## Three supported model structures

1. `VADRegressionHead`: direct VA or VAD regression after masked pooling.
2. `VADMediatedEmotionClassifier`: predicted VAD feeds the emotion classifier;
   this remains available as the comparison model.
3. `ParallelEmotionVADClassifier`: masked pooling feeds independent emotion,
   Valence, Arousal, and Dominance heads. Emotion logits never depend on VAD,
   and every VAD head is `Linear -> ReLU -> Linear(1)`.

```bash
python -m vad_downstream.train_parallel_emotion_vad \
  --train-prefix data/vad_emotion/train \
  --valid-prefix data/vad_emotion/valid \
  --output runs/parallel_emotion_vad.pt \
  --class-labels hap sad ang dis

python -m vad_downstream.infer_parallel_emotion_vad \
  --wav scripts/test.wav \
  --classifier-checkpoint runs/parallel_emotion_vad.pt \
  --output parallel_prediction.json
```

The class order defaults to `hap sad ang dis` only when `--class-labels` is not
given. Checkpoints preserve this order, per-dimension `vad_label_counts`, and
`supervised_dimensions`.

Dominance status is `trained` when the current train split has D labels,
`untrained` for a newly initialized head without D labels, and
`retained_from_checkpoint` when VA-only continued training freezes a previously
trained D head. An `untrained` numeric output is schema-compatible only and is
not a learned Dominance estimate.

## WAV to VA/VAD staged implementation

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

Train the VAD-mediated categorical classifier:

```bash
python -m vad_downstream.train_vad_emotion \
  --train-prefix data/vad_emotion/train \
  --valid-prefix data/vad_emotion/valid \
  --output runs/vad_emotion.pt \
  --epochs 10 \
  --batch-size 32 \
  --lambda-vad 1.0 \
  --lambda-emo 1.0 \
  --device auto
```

Run WAV-to-VAD-to-emotion inference:

```bash
python -m vad_downstream.infer_vad_emotion \
  --wav scripts/test.wav \
  --classifier-checkpoint runs/vad_emotion.pt \
  --output vad_emotion_prediction.json \
  --device cpu
```

Add `--model-dir <MODEL_DIR> --checkpoint <CHECKPOINT>` when a real emotion2vec
checkpoint is available. If omitted, the Stage 1 placeholder encoder is used
only for wiring checks.

## Required files

Use one shared prefix for each split or dataset. For example, `train` means the
following files live next to each other:

| File | Required | Description |
|---|---:|---|
| `<prefix>.npy` | yes | Frame-level emotion2vec features concatenated over all utterances. Shape: `(total_frames, 768)`. |
| `<prefix>.lengths` | yes | One integer per line. Each value is the frame count for one utterance. |
| `<prefix>.vad` | yes | Continuous affect labels. Valence and Arousal are required; Dominance is optional. |
| `<prefix>.emo` | yes for VAD-mediated classification | Categorical emotion labels. Optional only for pure VA/VAD regression. |

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
- `dominance` is optional and may be present for only some rows in the same
  file for parallel training. The loader creates fixed three-dimensional
  targets and a boolean `vad_target_mask`; missing D uses a masked dummy value.
- Label values must be normalized to `[-1.0, 1.0]`.
- Raw 1-to-5 ratings must be normalized with `(raw - 3.0) / 2.0`.
- Valence and Arousal cannot be missing.

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

When categorical emotion labels are needed, use this `.emo` format:

```text
utterance_id<TAB>class
```

Example:

```text
Ses01F_impro01_F000	hap
Ses01F_impro01_F001	sad
```

For the implemented VAD-mediated classifier, the canonical class order is:

```text
hap, sad, ang, dis
```

Japanese display names are:

```text
喜び, 悲しみ, 怒り, 嫌悪
```

Unknown labels are rejected. `exc -> hap` must be handled in preprocessing if
needed; training data should not keep `exc`. Do not relabel `neu` as `dis`.

## VAD-mediated categorical classification

The selected explainable classification direction is:

```text
emotion2vec frame features
  -> masked mean pooling
  -> FNN
  -> predicted valence/arousal/dominance
  -> Linear(target_dim -> num_classes)
  -> emotion class logits
```

The final linear layer is logistic regression over the predicted VA/VAD values.
In this design, class logits depend on the predicted affect values rather than
directly on the full 768-dimensional emotion2vec feature vector. This makes it
possible to report both the predicted emotion class and the VAD values used as
the classifier input.

When training this model, the VAD regression loss should remain part of the
objective. A practical objective is `CCC loss` for predicted VA/VAD plus
`CrossEntropyLoss` for the categorical emotion label. Training only with
classification loss would make the intermediate 2D/3D vector class-discriminative
but not necessarily interpretable as VA/VAD.

The implemented objective is:

```text
loss = lambda_vad * ccc_loss(predicted_vad, target_vad)
     + lambda_emo * cross_entropy(logits, target_emotion)
```

The default is `lambda_vad = 1.0` and `lambda_emo = 1.0`.

Validation reports both sides of the task:

- VAD: valence/arousal/dominance CCC and mean CCC.
- Emotion: WA, UA, weighted F1, and confusion matrix.

For every class `c`, the explanation is exactly:

```text
logit_c = bias_c
        + weight_c,valence * predicted_valence
        + weight_c,arousal * predicted_arousal
        + weight_c,dominance * predicted_dominance
```

`infer_vad_emotion.py` writes JSON with:

- `prediction`: predicted class index, code, and Japanese name.
- `probabilities`: softmax probabilities keyed by `hap/sad/ang/dis`.
- `vad`: predicted valence/arousal/dominance used by the classifier.
- `logits`: class logits.
- `linear_weights`: class-specific bias and VAD weights.
- `contributions`: class-specific bias and `weight * VAD` terms whose
  `logit_sum` equals the corresponding logit.
- `contrast_to_runner_up`: contribution differences against the second-ranked
  class.

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

The pure VAD regression loader:

- Load `<prefix>.npy`, `<prefix>.lengths`, and `<prefix>.vad`.
- Return padded frame-level features, a padding mask, and normalized VA or VAD
  targets.
- Detect whether the target dimensionality is 2 or 3 from the `.vad` file.

The VAD-mediated classification loader:

- Loads `<prefix>.npy`, `<prefix>.lengths`, `<prefix>.vad`, and `<prefix>.emo`.
- Requires row counts to match.
- Requires `.vad` and `.emo` utterance IDs to match row by row.
- Converts `hap/sad/ang/dis` to indices `0/1/2/3`.

## Not implemented in this step

- emotion2vec fine-tuning.
- Full scheduler and experiment logging.
- WAV-path dataset for training.
- IEMOCAP mixed training or Japanese dataset preprocessing.
- Conversion scripts from raw annotation files.
- Changes to `requirements.txt`.
