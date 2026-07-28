# 計画: Stage 1削除・CrossEntropyのみの単段階学習への設計変更

## Context
現在の `experiment_generic.ipynb` は「Stage 1 (VA正解ラベルでVADデコーダをCCC損失学習) → Stage 2 (CrossEntropyで分類器学習)」の2段階構成。
変更後は **Stage 1を完全に削除し、CrossEntropyのみでVADデコーダ＋分類器をEnd-to-Endに最適化する**。
VAD値はVA正解ラベルに引っ張られず、感情分類タスクのために自由に学習される。

---

## 変更対象ファイル
- `vad_downstream/experiment_generic.ipynb`（Cell 2, 9, 10のみ）

---

## 変更内容

### Cell 2 (CONFIG)
`STAGE1_EPOCHS`, `STAGE1_LR` を削除する。`STAGE2_LR_FNN` は残す。

```python
# 変更前
STAGE1_EPOCHS   = 30
STAGE1_LR       = 1e-3
STAGE2_LR_FNN   = 1e-4
STAGE2_LR_CLS   = 1e-3

# 変更後
STAGE2_EPOCHS   = 30
STAGE2_LR_FNN   = 1e-4   # VADデコーダ（感情分類のために最適化）
STAGE2_LR_CLS   = 1e-3   # 分類器
```

---

### Cell 9 (関数定義)
`ccc_loss`, `stage1_loss`, `train_stage1`, `eval_stage1` を削除する。  
`train_stage2` 以降は変更なし。

---

### Cell 10 (メインループ)
Stage 1ブロック（`if has_va:` 以下のStage 1学習処理）を丸ごと削除する。  
Stage 2のオプティマイザは現行のまま（VADデコーダ+分類器の両方を最適化）。

```python
# 変更前
if has_va:
    opt1 = ...（Stage 1学習ループ）

opt2 = optim.Adam([
    {"params": model.vad_decoder.parameters(), "lr": STAGE2_LR_FNN},
    {"params": model.classifier.parameters(), "lr": STAGE2_LR_CLS},
])

# 変更後（Stage 1ブロックを削除、Stage 2はそのまま）
opt2 = optim.Adam([
    {"params": model.vad_decoder.parameters(), "lr": STAGE2_LR_FNN},
    {"params": model.classifier.parameters(), "lr": STAGE2_LR_CLS},
])
```

---

## 変更しないもの
- モデルアーキテクチャ（Cell 7）: 構造・パラメータ数は同一
- `train_stage2`, `evaluate` 関数（Cell 9）
- Cell 11の可視化: VADデコーダは引き続き学習・推論されるので散布図も動作する

---

## 検証方法
1. ノートブックを実行し、Stage 1ループが走らないことを確認
2. Stage 2の損失が下がりWA/UA/F1が出力されることを確認
3. Cell 11の散布図が正常に表示されることを確認
