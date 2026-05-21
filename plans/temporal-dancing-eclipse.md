# 汎用感情認識ノートブック 実装プラン

## Context

既存の `vad_downstream/experiment.ipynb` はIEMOCAP専用にハードコードされており、他のデータセットで使えない。ユーザーは生WAVファイルから感情認識実験ができる汎用ノートブックを求めている。

**要件:**
- 生の音声ファイル（WAV等）から emotion2vec で特徴量抽出
- VAラベル（Valence/Arousal）がある場合は2段階学習、ない場合の対処は検討中
- 話者（セッション）ベースの Leave-One-Out 交差検証
- 最小依存・自己完結型（ローカルモジュールのインポートなし）

---

## 実装方針

新規ノートブック `vad_downstream/experiment_generic.ipynb` を作成。既存の `experiment.ipynb` は変更しない。

---

## 入力データ形式（CSV）

```csv
file_path,session,label[,valence,arousal]
audio/sample001.wav,Session1,angry,2.5,3.2
audio/sample002.wav,Session1,happy,3.8,4.1
```

| 列名 | 必須 | 説明 |
|------|------|------|
| `file_path` | ✅ | WAVファイルのパス（`AUDIO_DIR` からの相対 or 絶対） |
| `session` | ✅ | セッション/話者ID（Leave-One-Outに使用） |
| `label` | ✅ | 感情カテゴリ（`CLASS_NAMES` に含まれる文字列） |
| `valence` | ❌ | 連続値（あればStage 1実施） |
| `arousal` | ❌ | 連続値（あればStage 1実施） |

---

## ノートブック構成（セル順）

### Cell 1: インストール
```python
!pip install funasr modelscope soundfile torchaudio
```

### Cell 2: CONFIG（ユーザーが編集する唯一の場所）
```python
CSV_PATH    = "data/labels.csv"   # データCSVのパス
AUDIO_DIR   = "data/audio/"       # WAVのベースディレクトリ（相対パスの場合）
CACHE_DIR   = "cache/"            # 特徴量キャッシュ保存先
CLASS_NAMES = ["angry", "happy", "neutral", "sad"]

BATCH_SIZE      = 32
STAGE1_EPOCHS   = 30   # VAラベルなし時は無視
STAGE2_EPOCHS   = 30
STAGE1_LR       = 1e-3
STAGE2_LR_FNN   = 1e-4
STAGE2_LR_CLS   = 1e-3
```

### Cell 3: インポート
`torch`, `numpy`, `pandas`, `pathlib`, `matplotlib`（標準ライブラリのみ）

### Cell 4: emotion2vec モデルロード
`funasr.AutoModel` で `iic/emotion2vec_base` をロード

### Cell 5: CSV読み込み・検証
- `pd.read_csv(CSV_PATH)`
- `valence`/`arousal` 列の有無で `has_va` フラグを設定
- ラベルの整合性チェック（CLASS_NAMES との照合）

### Cell 6: 特徴量抽出（キャッシュ付き）
- `CACHE_DIR/{row_index}.npy` に各発話の特徴量 `(T, 768)` を保存
- 既存ファイルはスキップ（tqdmで進捗表示）

### Cell 7: モデル定義（インライン）
既存の `model.py` と同等のコードをノートブック内に直接記述:
- `AttentionPooling` — フレーム列を発話単位に集約
- `VADDecoder` — Attention + FNN → VAD値（[-1,1]）
- `EmotionClassifier` — VADDecoder + 線形分類器

### Cell 8: データセット定義（インライン）
- `SpeechDataset` — DataFrameとキャッシュディレクトリから読み込む
- `collator` — 可変長フレームのパディング処理

### Cell 9: 損失関数・学習/評価関数（インライン）
既存の `loss.py`, `train.py` と同等:
- `ccc_loss`, `stage1_loss`
- `train_stage1`, `train_stage2`, `evaluate`（WA/UA/F1計算）

### Cell 10: Leave-One-Out 交差検証ループ
```python
sessions = df["session"].unique()
for test_session in sessions:
    test_df = df[df["session"] == test_session]
    train_df = df[df["session"] != test_session]
    # DataLoader 作成 → モデル初期化 → Stage1(if has_va) → Stage2 → 評価
```

### Cell 11: 結果集計・可視化
- 全session平均 WA/UA/F1 の表示
- 混同行列（matplotlib）
- VAD散布図（`has_va` が True の場合のみ）

---

## 再利用する既存コード

| 再利用元 | 再利用内容 |
|---------|-----------|
| [model.py](vad_downstream/model.py) | `AttentionPooling`, `VADDecoder`, `EmotionClassifier` をほぼそのままコピー |
| [loss.py](vad_downstream/loss.py) | `ccc_loss`, `stage1_loss` をそのままコピー |
| [train.py](vad_downstream/train.py) | `train_stage1/2`, `evaluate`, `_weighted_f1` をコピー |
| [data.py](vad_downstream/data.py) | `SpeechDatasetVAD.collator` の実装を参考に再実装（DataFrameベースに変更） |

---

## 変更点（既存との差分）

| 項目 | 既存 | 新規 |
|------|------|------|
| データ形式 | .npy + .lengths + .emo | CSV + WAVファイル |
| 特徴量抽出 | 別途実行 | ノートブック内で自動実行 |
| データ分割 | IEMOCAPのSession構造 | CSVの `session` 列 |
| VAラベル | 必須 | オプション（なければStage 1スキップ） |
| クラス数 | 固定4 | CONFIG で指定 |
| モジュール依存 | data/model/loss/train.py | すべてインライン |

---

## 作成ファイル

- `vad_downstream/experiment_generic.ipynb` （新規作成）

---

## 検証方法

1. IEMOCAPデータで既存と同等の結果が得られるか確認
   - CSVを作成: `file_path, session, label, valence, arousal`
   - `CLASS_NAMES = ["ang", "hap", "neu", "sad"]`
   - Leave-One-Session-Out で実行
2. VAラベルなしCSVで動くか確認（Stage 1スキップ）
3. キャッシュが機能するか確認（2回目の実行で特徴量抽出をスキップ）
