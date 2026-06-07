# VAD downstream

This directory is the main path for VAD regression with cached emotion2vec
features. Predictions and labels always use this order:

```text
valence, arousal, dominance
```

## CSV format

Required:

```csv
file_path,valence,arousal,dominance
```

Optional:

```csv
split,session
```

`split` accepts `train`, `val`/`dev`/`valid`, and `test`. If no split column is
available, `train_vad.py` can split by `session` or by a fixed random seed.
VAD labels are expected in the Wagner-compatible `0..1` range. Any missing VAD
dimension is masked out of the loss.

## Feature cache

`train_vad.py` trains on `.npy` features. Each sample is expected to have a cache
file generated from `file_path` and row index by `data.feature_cache_path()`.

If cache files already exist:

```bash
py -m vad_downstream.train_vad ^
  --csv tests/fixtures/vad_dummy/vad_labels_dummy.csv ^
  --cache-dir tests/fixtures/vad_dummy/cache ^
  --epochs 1 ^
  --batch-size 2
```

If cache files are missing, pass an extractor as `module:function`:

```bash
py -m vad_downstream.train_vad ^
  --csv path/to/vad.csv ^
  --cache-dir path/to/cache ^
  --extractor my_package.extractors:extract_emotion2vec_features
```

The extractor must accept one audio path and return a `(T, 768)` or `(768,)`
array-like feature value.

## Outputs

The training script writes:

- `best_vad_regressor.pt`
- `metrics.json`

Metrics include CCC loss, per-dimension MAE, per-dimension CCC, split sizes, and
the fixed output name order.
