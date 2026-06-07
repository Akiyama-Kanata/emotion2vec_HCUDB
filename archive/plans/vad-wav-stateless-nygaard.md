# VADデコーダ実装計画

## Context

現在の `iemocap_downstream/` は emotion2vec の特徴量（.npy）を事前抽出し、2層MLP + 平均プーリングで感情分類する。これを「VAD空間を中間表現として使う2段階学習アーキテクチャ」に拡張する。Dominanceラベルは使えないため Stage 1 は VA（2次元）CCC 損失のみ。emotion2vec の frozen/unfrozen で精度を比較する実験も行う。

---

## 確定アーキテクチャ

```
WAV (16kHz, mono)
  ↓ emotion2vec (frozen or unfrozen, 768次元)
  ↓ (B, T, 768)
Attention Pooling
  ↓ (B, 768)
FNN: Linear(768→256) → LayerNorm → ReLU → Linear(256→3) → Tanh
  ↓ (B, 3)  [Valence, Arousal, Dominance]  ← D次元はStage2で間接学習
Linear Classifier: Linear(3→N)
  ↓ (B, N)
```

### 学習方針

| Stage | 学習対象 | 損失 | 備考 |
|-------|----------|------|------|
| Stage 1 | Attention Pooling + FNN | CCC(V) + CCC(A) | Dラベルなし、損失は2次元のみ |
| Stage 2 | Attention Pooling + FNN（小LR）+ 線形分類器 | CrossEntropy | FNNは unfrozen だが LR = Stage1 × 0.1 |

---

## 実装ファイル構成

新ディレクトリ `vad_downstream/` を作成。既存 `iemocap_downstream/` は変更しない。

```
vad_downstream/
├── model.py       # AttentionPooling, VADDecoder, EmotionClassifier
├── loss.py        # ccc_loss()
├── data.py        # VADラベル対応データローダー
├── train.py       # 2段階学習パイプライン
└── config/
    └── default.yaml
```

---

## 各ファイルの実装詳細

### `vad_downstream/model.py`

```python
class AttentionPooling(nn.Module):
    # Linear(768→1) → Softmax → 重み付き和
    # padding_mask を考慮してパディング位置の重みを -inf にする

class VADDecoder(nn.Module):
    # AttentionPooling + Linear(768→256) → LayerNorm → ReLU → Linear(256→3) → Tanh

class EmotionClassifier(nn.Module):
    # VADDecoder + Linear(3→N)
    # forward() は VAD値 と ロジット の両方を返す（評価・可視化用）
```

### `vad_downstream/loss.py`

```python
def ccc_loss(pred, target):
    # Concordance Correlation Coefficient の損失 (1 - CCC)
    # pred: (B,)、target: (B,) の1次元ずつ計算し、VAとAを合算

def stage1_loss(vad_pred, va_target):
    # vad_pred: (B, 3)、va_target: (B, 2) [V, A のみ]
    # return ccc_loss(vad_pred[:, 0], va_target[:, 0]) + ccc_loss(vad_pred[:, 1], va_target[:, 1])
```

### `vad_downstream/data.py`

- 既存 `iemocap_downstream/data.py` の `load_dataset` / `SpeechDataset` / `collator` を流用
- `collator` に `va_labels` フィールドを追加（Stage 1用）
- IEMOCAPのVADアノテーションは `.emo` ファイルとは別のファイル（`EvaluationSet`）から読み込む

```python
def load_iemocap_with_va(feature_path, label_dict, va_path):
    # カテゴリラベル + Valence/Arousal の連続値を同時に読み込む
```

### `vad_downstream/train.py`

```python
def train_stage1(model, loader, optimizer, epochs):
    # VADDecoder の Attention Pooling + FNN のみ学習
    # loss = stage1_loss(vad_pred, va_target)

def train_stage2(model, loader, optimizer_fnn, optimizer_cls, epochs):
    # FNNを小LRで、線形分類器を通常LRで学習
    # loss = CrossEntropyLoss(logits, category_labels)

def run_experiment(cfg):
    # Stage1 → Stage2 → テスト評価の一連のパイプライン
```

### `vad_downstream/config/default.yaml`

```yaml
model:
  input_dim: 768
  hidden_dim: 256
  vad_dim: 3
  num_classes: 4  # hap/sad/ang/neu

emotion2vec:
  frozen: true  # false にすると fine-tune

training:
  stage1_epochs: 30
  stage2_epochs: 30
  stage1_lr: 1.0e-3
  stage2_lr_cls: 1.0e-3
  stage2_lr_fnn: 1.0e-4  # FNNの学習率をstage1の1/10

dataset:
  feat_path: ""  # .npy の特徴量パス
  va_path: ""    # VADアノテーションのパス
  batch_size: 32
```

---

## 比較実験の設計

| 条件 | emotion2vec | 学習データ | テストデータ |
|------|-------------|-----------|-------------|
| A（baseline）| frozen | IEMOCAP（英語） | IEMOCAP |
| B | fine-tune | IEMOCAP（英語） | IEMOCAP |
| C | frozen / fine-tune | IEMOCAP（英語） | 日本語コーパス |
| D | fine-tune | 日本語コーパス | 日本語コーパス |

Dominance の間接評価：条件A（frozen）vs 条件B（fine-tune）で3次元分類器の精度差を確認。`Linear(3→N)` の D次元の重み係数の大きさも分析する。

---

## 前提確認が必要な事項

1. **IEMOCAPのVAラベル形式**：現在の `.emo` ファイルにVA値が含まれるか、または別の `EvaluationSet` ファイルを参照するか
2. **日本語コーパス**：使用するデータセット名と形式（VAラベルの有無含め）
3. **emotion2vec fine-tune 時**：既存の.npy特徴量ではなく、WAVファイルからリアルタイム処理するデータローダーが必要（`Emotion2vecFeatureReader.get_feats()`を参照）

---

## 検証方法

```bash
# Stage 1 の動作確認（CCC損失が下がるか）
python vad_downstream/train.py --config-name=default +stage=1

# Stage 2 の動作確認（分類精度が上がるか）
python vad_downstream/train.py --config-name=default +stage=2

# 比較実験：frozen vs fine-tune
python vad_downstream/train.py emotion2vec.frozen=true
python vad_downstream/train.py emotion2vec.frozen=false
```

評価指標：VA-CCC（Stage 1）、WA/UA/F1（Stage 2）
