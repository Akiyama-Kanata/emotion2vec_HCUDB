# emotion2vec リポジトリガイド

> 2026-09-08: 独立VAD実装・専用Notebook・専用テストはリポジトリ外へ移動しました。[保管先・台帳・復元手順](RETIRED_VAD.md)を参照してください。本文に残すVADの説明は保管した実装の参考情報です。

現行SERの入口はser_pipeline/とnotebooks/です。IEMOCAP実装は互換性テストとデモのため保持しています。A/B資料・結果とC/D計画はdocs/および既存の保存先を参照してください。

## 1. このリポジトリについて

このリポジトリは、音声から感情に関係する特徴表現を抽出する **emotion2vec** の公式 PyTorch コードを基盤に、下流タスクの研究実装を追加したものです。

大きく分けると、次の2層があります。

- 公式由来の機能
  - emotion2vec モデル本体
  - 音声特徴量の抽出
  - IEMOCAP における4クラス感情分類
  - FunASR 経由の emotion2vec+ 推論方法
- 現行SER実装・研究資料
  - ser_pipeline/、SER Notebook、関連テスト
  - A/Bの資料・過去結果、C/D計画
- 独立VAD研究実装
  - [リポジトリ外の保管先](RETIRED_VAD.md)へ移動済み

## 2. emotion2vec と emotion2vec+ の違い

### emotion2vec

自己教師あり学習によって、音声から感情に関係する汎用表現を抽出するモデルです。出力を分類や回帰などの下流モデルへ入力します。

- フレームレベル特徴：`(T, 768)`
- 発話レベル特徴：フレーム特徴を平均した `(768,)`
- フレームレート：約50 Hz（約20 msごと）
- 入力：16 kHz、モノラル WAV

emotion2vec 自体は基本的に特徴抽出器です。出力を使って何を予測するかは下流モデル側で決めます。

### emotion2vec+

emotion2vec を基盤として感情認識用にファインチューニングしたモデル群です。

- `emotion2vec_plus_seed`
- `emotion2vec_plus_base`
- `emotion2vec_plus_large`

FunASR から利用すると、次の9クラスについてスコアを取得できます。

```text
angry, disgusted, fearful, happy, neutral,
other, sad, surprised, unknown
```

## 3. 利用できる機能

### 3.1 音声特徴量の抽出

単一の WAV ファイルから emotion2vec 特徴を抽出できます。

```text
16 kHz mono WAV
  ↓ emotion2vec
frame-level: (T, 768)
または
utterance-level: (768,)
```

実装は `scripts/extract_features.py` にあります。

### 3.2 emotion2vec+ による9クラス感情推論

FunASR の `AutoModel` から公開モデルをロードし、感情ラベルとスコアを取得できます。公開済みモデルを使って簡単に推論したい場合の推奨経路です。

### 3.3 IEMOCAP の4クラス感情分類

IEMOCAP の標準的な5531発話を対象に、次の4クラスを分類します。

```text
angry, happy, neutral, sad
```

emotion2vec で抽出済みの768次元特徴を下流モデルへ入力します。評価方式は、IEMOCAP の5セッションを使った leave-one-session-out の5-fold cross-validation です。

評価指標は次のとおりです。

- WA：全体の分類精度
- UA：クラスごとの再現率の平均
- Weighted F1

実装は `iemocap_downstream/main.py` にあります。

### 3.4 VA/VAD の連続値回帰（保管実装の説明）

事前抽出したフレーム特徴から、感情を連続値として予測できます。

- Valence：快・不快
- Arousal：覚醒度
- Dominance：支配性・優位性

2次元の VA と3次元の VAD の両方に対応しています。

```text
emotion2vec frame features
  ↓ paddingを除外した平均プーリング
  ↓ FNN
predicted VA/VAD
```

学習損失には、予測と正解の一致度を測る CCC を使います。

```text
loss = 1 - mean(CCC)
```

### 3.5 VAD を介した説明可能な4クラス分類（保管実装の説明）

このリポジトリで追加された主要な研究実装です。

```text
emotion2vec特徴
  ↓
予測VA/VAD
  ↓
Linear(target_dim → 4 classes)
  ↓
喜び・悲しみ・怒り・嫌悪
```

クラス順は次のように固定されています。

```text
hap, sad, ang, dis
```

分類器は768次元特徴を直接分類に使わず、予測された VA/VAD のみからクラスを決定します。そのため、各感情の判定に Valence、Arousal、Dominance がどの程度寄与したかを確認できます。

推論 JSON には、次の情報が含まれます。

- 予測クラス
- 各クラスの確率
- 予測 VA/VAD
- 各クラスの logit
- 線形分類器の重みとバイアス
- `weight × VAD` の寄与
- 1位と2位のクラスの寄与差

## 4. リポジトリ構造

```text
emotion2vec/
├── upstream/
│   ├── models/                 emotion2vecモデル本体
│   └── tasks/                  fairseqの事前学習タスク
├── scripts/
│   ├── extract_features.py     単一WAVの特徴抽出
│   └── extract_features.sh     シェルラッパー
├── iemocap_downstream/
│   ├── main.py                 5-fold分類学習
│   ├── data.py                 特徴・ラベルのロード
│   ├── model.py                IEMOCAP分類モデル
│   ├── utils.py                学習・評価処理
│   ├── config/                 Hydra設定
│   └── scripts/                マニフェスト作成・特徴抽出
├── ser_pipeline/              現行SER処理
├── notebooks/                 現行研究Notebook・分析・互換デモ
├── RETIRED_VAD.md              独立VAD実装の保管案内
├── tests/                      SER・IEMOCAP互換・Notebook等のテスト
├── src/                        README・発表資料用画像
├── archive/                    過去のログ・計画・資料
├── requirements.txt
├── TESTING.md
└── README_ja.md
```

### IEMOCAP のデータフロー

```text
IEMOCAPのWAV
  ↓ マニフェスト・ラベル作成
train.tsv + train.emo
  ↓ emotion2vec特徴抽出
train.npy + train.lengths + train.emo
  ↓ 5-fold学習
WA / UA / Weighted F1
```

### VAD 下流処理のデータフロー（保管実装の説明）

```text
<prefix>.npy
<prefix>.lengths
<prefix>.vad
必要に応じて <prefix>.emo
  ↓
VADRegressionHead
または VADMediatedEmotionClassifier
  ↓
checkpoint
  ↓
WAV推論・JSON出力
```

## 5. 環境構築

このリポジトリでは、Windows ネイティブ Python ではなく、WSL/Ubuntu の専用環境が標準です。

- WSL distribution：Ubuntu
- Python：3.10
- PyTorch：1.13以上
- fairseq：0.12.2
- pip：24.1未満
- NumPy：2未満

セットアップ例：

```bash
conda create -n emotion2vec-py310 python=3.10
conda activate emotion2vec-py310

cd /mnt/c/Users/RD004/Documents/lab/emotion2vec

python -m pip install "pip<24.1"
python -m pip install -r requirements.txt
```

制限付きデータや VAD ラベルのローカルパスを設定する場合は、雛形をコピーして `.env` を作成します。

```bash
cp .env.example .env
```

`.env` の `RESTRICTED_DATA_DIR` と `VAD_CSV_PATH` を各環境に合わせて編集してください。`.env` はローカル専用であり、実際のパスや将来追加する秘密情報を Git にコミットしないでください。現状のコードは `.env` を自動では読み込まないため、必要な値は利用するシェルやツール側で環境変数として読み込む必要があります。

インポート確認：

```bash
python -c "import torch, fairseq, soundfile, hydra, numpy; print('ok')"
```

fairseq 0.12.2 と新しい Python、pip、NumPy の組み合わせには互換性問題があるため、指定バージョンを維持するのが安全です。

## 6. 動かし方

### 6.1 FunASR で emotion2vec+ を使う

最も簡単な感情推論方法です。

```bash
pip install -U funasr
```

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_plus_large",
    hub="hf",
)

result = model.generate(
    "sample.wav",
    output_dir="./outputs",
    granularity="utterance",
    extract_embedding=False,
)

print(result)
```

`hub="hf"` は Hugging Face、`hub="ms"` は ModelScope を意味します。モデルは初回実行時に自動ダウンロードされます。

### 6.2 単一 WAV の特徴を抽出する

別途 emotion2vec のチェックポイントが必要です。

```bash
python scripts/extract_features.py \
  --source_file sample.wav \
  --target_file sample_features.npy \
  --model_dir upstream \
  --checkpoint_dir /path/to/emotion2vec_checkpoint.pt \
  --granularity utterance
```

フレーム特徴が必要な場合は `--granularity frame` を指定します。

現在の `scripts/extract_features.py` は無条件に `.cuda()` を呼ぶため、実質的に CUDA 対応 GPU が必要です。入力 WAV は16 kHz・モノラルでなければなりません。

### 6.3 IEMOCAP の学習データを作る

```bash
IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
MANIFEST_PATH=/path/to/manifest

bash iemocap_downstream/scripts/iemocap_manifest_and_labels.sh \
  "$IEMOCAP_ROOT" \
  "$MANIFEST_PATH"
```

次に emotion2vec 特徴を抽出します。

```bash
bash iemocap_downstream/scripts/emotion2vec_extract_features.sh \
  /path/to/fairseq \
  /path/to/manifest \
  upstream \
  /path/to/emotion2vec_checkpoint.pt \
  /path/to/features
```

生成物：

```text
train.npy
train.lengths
train.emo
```

### 6.4 IEMOCAP 4クラス分類を学習する

```bash
cd iemocap_downstream
bash train.sh /path/to/features/train
```

既定の学習設定は、batch size 128、100 epochs、learning rate `5e-4`、seed 42です。設定は `iemocap_downstream/config/default.yaml` にあります。

評価は5セッション固定の leave-one-session-out です。各 fold では1セッションをテスト用とし、残り4セッションを80:20で学習用と検証用にランダム分割します。セッション数とセッションごとの発話数は `iemocap_downstream/main.py` にハードコードされており、現在の実装では `default.yaml` の `dataset.fold` と `dataset.test_ratio` は参照されません。

### 6.5 独立VAD実装の利用手順

VADの学習・推論CLIと専用READMEは[保管先](RETIRED_VAD.md)へ移動しました。
利用する場合は台帳と復元手順を確認してください。

## 7. VAD 用データ形式（保管実装の説明）

同じ split には同じ prefix を使います。

```text
train.npy
train.lengths
train.vad
train.emo
```

### `.npy`

全発話のフレーム特徴を連結した配列です。

```text
shape = (total_frames, 768)
```

### `.lengths`

各発話のフレーム数を1行ずつ記録します。合計値は `.npy` の第1次元と一致する必要があります。

```text
125
98
143
```

### `.vad`

タブ区切りです。

VA の場合：

```text
utterance_id<TAB>valence<TAB>arousal
```

VAD の場合：

```text
utterance_id<TAB>valence<TAB>arousal<TAB>dominance
```

値域はすべて `[-1.0, 1.0]` です。1～5の評定値は次式で正規化します。

```text
normalized = (raw - 3.0) / 2.0
```

### `.emo`

```text
utterance_id<TAB>class
```

VAD 経由分類で使用可能なラベルは次の4種類です。

```text
hap
sad
ang
dis
```

`.vad` と `.emo` は行数、行順、発話 ID が一致している必要があります。

## 8. テスト

現行の非学習テストは[TESTING.md](TESTING.md)を参照してください。
SER互換性、データ契約、Notebook構造などを確認します。学習を行う回帰テストは別途ユーザーが実行します。
VAD専用テストは[保管先](RETIRED_VAD.md)へ移動済みです。

## 9. 必要な外部データ・モデル

リポジトリを clone しただけでは、すべての学習・本番推論は実行できません。別途、目的に応じて次のものが必要です。

- emotion2vec の学習済み checkpoint
- IEMOCAP データセット
- IEMOCAP から作った manifest と感情ラベル
- VAD 回帰を行う場合は正解 VA/VAD ラベル
- VAD 経由分類では `.vad` と `.emo` の両方

`data/` には実データが同梱されていません。

## 10. 現時点の制限と注意点

### placeholder encoder

VAD 系の推論 CLI では、emotion2vec の `--model-dir` と `--checkpoint` を両方省略すると Stage 1 の placeholder encoder が使われます。片方だけを指定することはできません。placeholder encoder は配線確認用であり、出力値に研究上の意味はありません。

VA/VAD 回帰 CLI の `vad_downstream.inference` で未学習 head を使う場合は、明示的に `--allow-random-head` が必要です。説明付き感情分類 CLI の `vad_downstream.infer_vad_emotion` で未学習の分類モデルを使う場合は、`--allow-random-model` が必要です。いずれの場合も、ランダムなモデルの結果を評価値として扱ってはいけません。

### 未実装の機能

現在実装されているのは、主に事前抽出特徴から下流 head を学習する方法です。次の機能は未実装です。

- emotion2vec 本体の end-to-end fine-tuning
- WAV を直接入力する学習 Dataset
- データセット固有の VAD 前処理
- 本格的な scheduler・実験追跡
- 日本語データセットの変換処理

### IEMOCAP のクラス定義

通常の IEMOCAP 評価は次の4クラスです。

```text
ang, hap, neu, sad
```

一方、VAD 経由分類は次の4クラスです。

```text
hap, sad, ang, dis
```

`neutral` を `disgust` に置き換えてはいけません。VAD 経由分類では、IEMOCAP の実際の `dis` アノテーションを抽出する専用モードを使用します。

```bash
bash iemocap_downstream/scripts/iemocap_manifest_and_labels.sh \
  "$IEMOCAP_ROOT" \
  "$MANIFEST_PATH" \
  vad4
```

`vad4` モードは `ang`、`exc`、`hap`、`sad`、`dis` を抽出し、`exc` を `hap` に統合します。このスクリプトが生成するのはマニフェストと `.emo` ラベルです。VAD 学習に必要な `.vad` ラベルは生成されないため、同じ発話 ID と行順で別途用意する必要があります。

### GPU依存

単一 WAV の公式由来特徴抽出スクリプトは `.cuda()` を直接呼びます。CPU のみで動かす場合は修正が必要です。一方、追加された VAD 系 CLI は `--device cpu` や `--device auto` を選択できます。

### 例外処理

単一 WAV 特徴抽出スクリプトには、例外を生成するだけで再送出していない箇所があります。失敗時に分かりにくい可能性があるため、出力ファイルの生成を確認してください。

## 11. 現行研究の入口

[Python索引](PYTHON_FILES.md)、[テスト案内](TESTING.md)、docs/plans/の現行計画を参照してください。
A/Bの実装・結果、C/D計画、共通実装・資料は保持しています。独立VAD実装の利用・復元は[保管案内](RETIRED_VAD.md)を参照してください。
