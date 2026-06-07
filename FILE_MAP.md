# emotion2vec workspace map

This repository keeps the original emotion2vec code and adds a cleaned VAD
regression path.

## Main path: VAD regression

| Path | Role |
|---|---|
| `vad_downstream/train_vad.py` | Main VAD training entrypoint for cached emotion2vec features |
| `vad_downstream/model.py` | `Emotion2VecVADRegressor`, output order `arousal, dominance, valence` |
| `vad_downstream/loss.py` | CCC loss with missing-label masking |
| `vad_downstream/data.py` | CSV loading, cache path creation, splitting, VAD DataLoader |
| `vad_downstream/config/default.yaml` | Reference VAD regression settings |
| `vad_downstream/README.md` | VAD CSV/cache/training usage |

## Reference implementations

| Path | Role |
|---|---|
| `upstream/` | Original emotion2vec/fairseq model and task definitions |
| `scripts/extract_features.py` | Single-WAV emotion2vec feature extraction |
| `iemocap_downstream/` | Original IEMOCAP 4-class downstream classifier |
| `iemocap_downstream/scripts/` | IEMOCAP manifest and batch feature extraction utilities |

## Fixtures and experiments

| Path | Role |
|---|---|
| `tests/fixtures/vad_dummy/` | Small VAD CSV and cached dummy features for smoke tests |
| `tests/test_vad_downstream.py` | Minimal VAD data/loss/training tests |
| `notebooks/` | Experimental notebooks kept out of the main execution path |
| `archive/vad_iemocap_two_stage/` | Old VAD-intermediate IEMOCAP classification experiment |
| `archive/plans/` | Historical planning notes |
| `archive/notebook_tools/` | One-off notebook patching tools |
| `docs/references/` | Papers, extracted paper text, and reference notes |

## Data flow

```text
VAD CSV + cached emotion2vec .npy features
  -> vad_downstream.data
  -> Emotion2VecVADRegressor
  -> vad_ccc_loss
  -> best_vad_regressor.pt + metrics.json
```
