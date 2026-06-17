# emotion2vec ファイル構造マップ

emotion2vecリポジトリの各ファイル・ディレクトリの役割をまとめたリファレンス。

## 論文

| ファイル | 内容 |
|---|---|
| `emotion2vec_paper_ACL2024.pdf` | 元論文 ACL 2024 Findings 採録版（正式版・引用はこちら） |

**論文タイトル**: *emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation*  
**著者**: Ma, Ziyang et al.  
**発表**: ACL 2024 Findings

---

## ディレクトリ構造概要

```
emotion2vec/
├── upstream/           # 事前学習済みモデルの定義（コア）
│   ├── models/
│   │   ├── emotion2vec.py   # モデル本体（フォワードパス・特徴抽出）
│   │   ├── base.py          # モダリティ共通エンコーダ基底クラス
│   │   ├── audio.py         # 音声モダリティ特有のエンコーダ
│   │   ├── config.py        # モデル全体のハイパーパラメータ設定
│   │   └── modules.py       # Transformerブロックなど共通モジュール
│   └── tasks/
│       └── audio_pretraining.py  # fairseqタスク定義（事前学習用）
├── iemocap_downstream/ # IEMOCAPデータセットでの下流タスク学習
│   ├── main.py              # 学習エントリポイント（5-fold CV）
│   ├── model.py             # 線形分類器（BaseModel）
│   ├── data.py              # データローダー・前処理
│   ├── utils.py             # 学習・評価ループ、指標計算
│   ├── train.sh             # 学習実行シェルスクリプト
│   ├── inference.ipynb      # 推論サンプルノートブック
│   ├── config/
│   │   └── default.yaml     # ハイドラ設定ファイル（学習パラメータ）
│   └── scripts/
│       ├── emotion2vec_extract_features.sh  # 特徴抽出シェルラッパー
│       ├── emotion2vec_speech_features.py   # バッチ特徴抽出スクリプト（TSV形式入力）
│       ├── iemocap_manifest_and_labels.sh   # IEMOCAPのmanifest生成シェルラッパー
│       └── iemocap_manifest.py              # IEMOCAPのmanifest・ラベル生成
├── vad_downstream/     # VA/VAD連続値回帰の下流タスク
│   ├── README.md            # .npy/.lengths/.vad のデータ契約
│   ├── data.py              # VAD/VA用データローダー
│   ├── model.py             # 回帰headとemotion2vec込み全体モデル
│   └── training.py          # CCC lossと最小学習ループ
├── scripts/            # 汎用特徴抽出スクリプト
│   ├── extract_features.py  # 単一WAVファイルから特徴抽出
│   ├── extract_features.sh  # 上記のシェルラッパー
│   └── test.wav             # テスト用音声サンプル
├── src/                # READMEに使用する画像リソース
│   ├── logo.png
│   ├── emotion2vec+performance.png
│   ├── emotion2vec+radar.png
│   ├── IEMOCAP.png
│   ├── Languages.png
│   ├── UMAP.png
│   └── Wechat.jpg
├── plans/              # （空）
├── README.md           # プロジェクト全体の説明・使い方
└── .gitignore
```

---

## 各ファイルの詳細

### upstream/models/emotion2vec.py
**役割**: モデル本体。`Data2VecMultiModel` クラスを定義。

- fairseqの `BaseFairseqModel` を継承
- `AudioEncoder`（音声モダリティ）を `modality_encoders['AUDIO']` に登録
- `forward()`: マスク→Transformerブロック→正規化 の順で処理
- `extract_features()`: 推論時に特徴ベクトルを取り出すインターフェース
- 出力: `{"x": 特徴テンソル, "padding_mask": ..., "layer_results": 各層出力, "mask": マスク情報}`

### upstream/models/base.py
**役割**: モダリティ非依存のエンコーダ基底クラス `ModalitySpecificEncoder` を定義。

- `local_features()`: CNNなどのローカルエンコーダで初期特徴を抽出
- `contextualized_features()`: マスキング・位置エンコーディング・Transformerによる文脈化
- `compute_mask()` / `make_maskinfo()` / `apply_mask()`: self-supervisedマスキング処理
- ALiBiバイアス（相対位置エンコーディング）の計算・キャッシュも担当
- 補助関数: `get_alibi()`, `random_masking()`, `gather_unmasked()` など

### upstream/models/audio.py
**役割**: 音声モダリティ特化のエンコーダ `AudioEncoder` と設定 `D2vAudioConfig` を定義。

- `D2vAudioConfig`: 音声CNN特徴抽出器の仕様（カーネルサイズ・ストライドなど）と畳み込み位置エンコーダの設定
- `AudioEncoder`: `ModalitySpecificEncoder` を継承し、wav2vec2と同じ `ConvFeatureExtractionModel`（CNN）でローカル特徴を抽出

### upstream/models/config.py
**役割**: モデル全体のハイパーパラメータを `Data2VecMultiConfig` データクラスで定義。

- Transformerの深さ（`depth=8`）、ヘッド数（`num_heads=12`）、埋め込み次元（`embed_dim=768`）など
- EMAの設定（`ema_decay`, `ema_end_decay`）
- 敵対的学習オプション（`adversarial_training`）、発話レベル損失タイプ（`cls_type`）

### upstream/models/modules.py
**役割**: `AltBlock`（Transformerブロック）、`Decoder1d`、`Modality`列挙型などの共通モジュール。

### upstream/tasks/audio_pretraining.py
**役割**: fairseqの事前学習タスク `Emotion2vecPretrainingTask` を定義。

- `Emotion2vecPretrainingConfig`: データパス、マルチコーパスサンプリング、正規化フラグなどを設定
- `AudioMaskingConfig`: マスキングパラメータをモデル設定から参照

---

### scripts/extract_features.py
**役割**: 単一WAVファイルから特徴ベクトルを抽出して `.npy` に保存。

- 引数: `--source_file`（入力WAV）、`--target_file`（出力NPY）、`--model_dir`、`--checkpoint_dir`、`--granularity`
- `granularity="frame"`: フレームレベル特徴（T×768）をそのまま保存
- `granularity="utterance"`: フレームを平均して1×768の発話レベル特徴を保存
- 入力条件: サンプリングレート16kHz、モノラル

### scripts/extract_features.sh
**役割**: `extract_features.py` を呼び出すシェルスクリプト（パス設定用）。

---

### iemocap_downstream/main.py
**役割**: IEMOCAPデータセットでの5-fold交差検証学習のエントリポイント。

- Leave-one-session-outの5分割CVを実行
- セッション数: `[1085, 1023, 1151, 1031, 1241]`サンプル（Session 1〜5）
- 感情クラス: `{ang: 0, hap: 1, neu: 2, sad: 3}`（4クラス分類）
- オプティマイザ: RMSprop + CyclicLR スケジューラ
- 各foldで最良の検証WAを基準にモデルを保存

### iemocap_downstream/model.py
**役割**: 線形分類器 `BaseModel` の定義。

- 構造: `Linear(768→256) → ReLU → Linear(256→4)`
- パディングマスクを考慮して平均プーリングで発話レベルの表現を計算

### iemocap_downstream/data.py
**役割**: 特徴量の読み込みとデータローダーの作成。

- `load_dataset()`: `.npy`（特徴）・`.lengths`（フレーム長）・`.emo`（ラベル）の3ファイルセットを読み込み
- `SpeechDataset`: PyTorchの `Dataset` クラス、パディングマスク付きでバッチを整形
- `train_valid_test_iemocap_dataloader()`: fold番号を指定してtrain/val/testの分割を生成

### iemocap_downstream/utils.py
**役割**: 学習・評価ループと評価指標の計算。

- `train_one_epoch()`: 1エポック分の学習ループ
- `validate_and_test()`: WA（Weighted Accuracy）・UA（Unweighted Accuracy）・Weighted F1 を計算
- `compute_unweighted_accuracy()`: クラスごとの正解率を平均（UAの計算）
- `compute_weighted_f1()`: サンプル数で重み付けしたF1スコア

### iemocap_downstream/config/default.yaml
**役割**: Hydraの学習設定ファイル。

- `feat_path`: 特徴ファイルのパス（要変更）
- `batch_size: 128`、`epoch: 100`、`lr: 5e-4`

### iemocap_downstream/train.sh
**役割**: `main.py` を呼び出すシェルスクリプト。

### iemocap_downstream/inference.ipynb
**役割**: 学習済みモデルを使った推論サンプルノートブック。

### iemocap_downstream/scripts/emotion2vec_speech_features.py
**役割**: TSV形式のマニフェストを入力として、バッチで特徴抽出を行う。

- `Emotion2vecFeatureReader`: モデルのロードと1ファイルごとの特徴抽出
- 出力: `.npy`（特徴）と `.lengths`（フレーム数）をセットで保存
- `--layer` 引数で取り出す層を指定可能（0〜11）

### iemocap_downstream/scripts/emotion2vec_extract_features.sh
**役割**: `emotion2vec_speech_features.py` のシェルラッパー（IEMOCAPのtrain分割用）。

### iemocap_downstream/scripts/iemocap_manifest.py / iemocap_manifest_and_labels.sh
**役割**: IEMOCAPの生データからfairseq形式のTSVマニフェストと感情ラベルファイル (`.emo`) を生成。

---

### vad_downstream/README.md
**役割**: VA/VAD回帰用の中間データ契約を定義。

- `<prefix>.npy`: frame-level emotion2vec特徴量 `(total_frames, 768)`
- `<prefix>.lengths`: 1発話1行のフレーム数
- `<prefix>.vad`: `utterance_id<TAB>valence<TAB>arousal` または dominance 付き
- ラベル値域は正規化済み `[-1.0, 1.0]`

### vad_downstream/data.py
**役割**: `.npy`、`.lengths`、`.vad` を読み込み、padding済みbatchへ変換。

- `load_vad_dataset()`: 特徴量、発話長、VA/VAD target、utterance_idを読み込む
- `VADSpeechDataset`: `net_input.feats`、`net_input.padding_mask`、`target` を返す

### vad_downstream/model.py
**役割**: VA/VAD連続値回帰モデルを定義。

- `VADRegressionHead`: frame-level特徴量をmasked mean poolingし、VA/VADを出力
- `Emotion2vecVADModel`: 音声波形テンソルからemotion2vec特徴抽出を経て同じheadで回帰

### vad_downstream/training.py
**役割**: VA/VAD回帰の最小学習単位を定義。

- `concordance_correlation_coefficient()`: target次元ごとのCCCを計算
- `ccc_loss()`: `1 - mean(CCC)` を学習lossとして返す
- `train_one_epoch()`: README準拠batchを1epoch学習する

---

## データの流れ（IEMOCAP評価の場合）

```
IEMOCAPデータセット（WAVファイル）
    ↓ iemocap_manifest_and_labels.sh
    ↓ （iemocap_manifest.py）
TSVマニフェスト + .emo ラベルファイル
    ↓ emotion2vec_extract_features.sh
    ↓ （emotion2vec_speech_features.py）
train.npy + train.lengths + train.emo
    ↓ main.py（5-fold CV）
    ↓ （data.py でロード → model.py で分類 → utils.py で評価）
WA / UA / Weighted F1 の結果
```

## 特徴量の仕様

| 項目 | 値 |
|---|---|
| 特徴次元 | 768次元 |
| フレームレート | 50Hz（20msごと） |
| 発話レベル特徴 | フレームレベルの平均（768次元） |
| ファイル形式 | `.npy`（NumPy配列） |
