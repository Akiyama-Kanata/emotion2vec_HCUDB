# emotion2vec+ large 導入・利用ガイド

この資料は、公開済みの `emotion2vec+ large` を使って、1本の音声から感情を推定するための研究室向け手順です。条件C/Dの比較実験ではなく、FunASRが提供する通常の推論方法を扱います。

公式情報：

- [emotion2vec+ large（Hugging Face）](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
- [emotion2vec GitHubリポジトリ](https://github.com/ddlBoJack/emotion2vec)
- [FunASR GitHubリポジトリ](https://github.com/modelscope/FunASR)

## 1. 何ができるか

音声ファイルを入力すると、次の9感情に対するスコアが返ります。

| 番号 | 公式ラベル | 日本語の目安 |
|---:|---|---|
| 0 | angry | 怒り |
| 1 | disgusted | 嫌悪 |
| 2 | fearful | 恐れ |
| 3 | happy | 喜び |
| 4 | neutral | 中立 |
| 5 | other | その他 |
| 6 | sad | 悲しみ |
| 7 | surprised | 驚き |
| 8 | unknown | 不明 |

スコアはモデルの推定値です。話者の主観的な感情や医学的・心理学的状態を確定するものではありません。

## 2. 推奨する音声

- WAV形式
- サンプリング周波数16 kHz
- モノラル
- 発話区間だけを含む、短い音声

異なるサンプリング周波数や圧縮音声を使う場合は、事前に16 kHz・モノラルWAVへ変換すると入力条件を統一できます。FFmpegを利用できる場合は次のように変換します。

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output_16k_mono.wav
```

## 3. 仮想環境を作る

既存の研究環境との依存関係の衝突を避けるため、専用の仮想環境を使います。

### Windows PowerShell

```powershell
py -3.11 -m venv .venv-emotion2vec
.\.venv-emotion2vec\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --upgrade funasr modelscope
```

PowerShellでスクリプト実行が拒否された場合は、そのターミナルだけ実行ポリシーを変更してから有効化します。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv-emotion2vec\Scripts\Activate.ps1
```

### Linux / WSL

```bash
python3 -m venv .venv-emotion2vec
source .venv-emotion2vec/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade funasr modelscope
```

## 4. 最小コードで推論する

次を `infer_emotion.py` として保存します。

```python
from funasr import AutoModel

model = AutoModel(
    model="iic/emotion2vec_plus_large",
    hub="hf",
    device="cpu",
)

result = model.generate(
    input="sample.wav",
    output_dir="./outputs",
    granularity="utterance",
    extract_embedding=False,
)

labels = result[0]["labels"]
scores = result[0]["scores"]

for label, score in zip(labels, scores):
    print(f"{label}: {float(score):.6f}")
```

実行します。

```bash
python infer_emotion.py
```

初回実行時はモデルが自動的にダウンロードされます。モデル本体は約2 GBあるため、完了までしばらくかかります。2回目以降は通常、保存済みキャッシュが使われます。

この資料では日本からの利用を想定してHugging Faceを選ぶ `hub="hf"` を使用しています。ModelScopeを使う場合は `hub="ms"` に変更します。

## 5. このリポジトリのサンプルを使う

このリポジトリには、入力ファイルをコマンドラインで指定できるサンプルがあります。

### CPUで実行

```powershell
python scripts/run_emotion2vec_plus_large.py "C:\path\to\sample.wav"
```

### GPUで実行

CUDA対応版PyTorchと対応GPUが利用できる環境では、次のように指定します。

```powershell
python scripts/run_emotion2vec_plus_large.py "C:\path\to\sample.wav" --device cuda:0
```

### 埋め込み特徴も取得

```powershell
python scripts/run_emotion2vec_plus_large.py "C:\path\to\sample.wav" --extract-embedding
```

実行時には全ラベルのスコア、最大スコアのラベル、FunASRの出力先が表示されます。

## 6. 結果の読み方

出力例は次の形式です。数値は音声によって変わります。

```text
scores:
  angry: 0.041000
  disgusted: 0.012000
  fearful: 0.008000
  happy: 0.721000
  neutral: 0.142000
  other: 0.030000
  sad: 0.018000
  surprised: 0.021000
  unknown: 0.007000
top emotion: happy (0.721000)
```

通常は最大スコアのラベルを推定結果として扱います。研究で使用するときは最大ラベルだけでなく、9クラスすべてのスコア、モデル名、ライブラリのバージョン、入力音声の前処理条件も保存してください。

## 7. よくある問題

### `ModuleNotFoundError: No module named 'funasr'`

仮想環境が有効になっているか確認し、次を実行します。

```bash
python -m pip install --upgrade funasr modelscope
```

### モデルを取得できない

モデル名を省略形にせず、次を正確に指定します。

```text
iic/emotion2vec_plus_large
```

ネットワーク、プロキシ、保存先の空き容量も確認してください。

### GPUで実行できない

まずCPUで動作確認します。

```powershell
python scripts/run_emotion2vec_plus_large.py "C:\path\to\sample.wav" --device cpu
```

GPUを使うには、GPUドライバー、CUDAに対応するPyTorch、対応GPUが必要です。PyTorchは利用するCUDA環境に合った公式手順で導入してください。

### ステレオ音声やサンプリング周波数が異なる

FFmpegなどで16 kHz・モノラルへ変換してから再実行します。

## 8. 今回の条件C/D実験との違い

通常利用は、1本の音声を公式の9クラスheadで分類して終了します。

条件C/D実験では、MSP-PodcastとHCUDBの対象6感情（angry、disgusted、fearful、happy、sad、surprised）から1024次元のフレーム特徴を抽出し、`official6` manifestに結び付く専用キャッシュを作ります。条件Cは公式9クラスheadを固定して評価します。条件Dは公式headから毎seed初期化し、非加重9-way cross entropyで対象6行だけをHCUDB Trainから更新します。neutral、other、unknownの3行は固定します。

C/Dの推論は9 logits全体をsoftmaxし、9クラス全体でargmaxします。neutral、other、unknownが最大になった場合も6クラスへ再分類せず、対象6クラス指標では誤分類として扱います。manifestは `python -m ser_pipeline build-manifest --label-profile official6 ...` で作成してください。既存A/B用4クラスmanifest、cache、checkpointはそのまま維持されます。

条件C/Dの実行は2冊に分かれます。まず `notebooks/03_extract_official_head_cd_features.ipynb` でofficial6 manifest、Large特徴cache、`runs/official_cd/parity.json` を生成・検証します。次に `notebooks/04_train_and_evaluate_official_head_cd.ipynb` で、入力artifact確認、条件C評価、C結果の確認と基準決定、条件Dの学習・再開、条件D評価と保存済みCとの比較、の順にセルを実行します。通常の音声感情推定だけが目的なら、この2冊を実行する必要はありません。

04 NotebookではCとDの評価出力をそれぞれ `runs/official_cd/evaluation_c` と `runs/official_cd/evaluation_d` に保存します。C評価はDのstudy summaryやcheckpointを必要とせず、MSP-Podcast Test1とHCUDB Testを各1回評価します。D評価はCを再推論せず、保存済みC summaryのhead、cache、test集合の署名が現在の入力と一致する場合だけ、3 seedのDを評価してCとの差、seed平均、標本標準偏差を計算します。確認済みフラグや数値による自動合否判定はなく、C結果確認セルとD学習セルの分離で運用します。

### 全量特徴抽出を安全に開始・再開する

初回だけ `RUN_PARITY = True` で `parity.json` を作成し、結果を確認した後はFalseへ戻します。本番は `RUN_FULL_EXTRACTION = True` の1セルで実行します。このセルは次の順序を崩しません。

1. MSP-PodcastとHCUDB1のmanifestおよびincluded音声全件を検証する。
2. 既存cacheの確定済みshard、manifest prefix、`_SUCCESS`、残件数を読み取り専用で監査し、残件分だけの容量を判定する。
3. encoderを生成して既存parity reportとの一致を確認する。
4. 各datasetのmanifest順先頭10件を一時cacheへ保存し、hash/index/offsetとmmap再読込を確認する。
5. 全datasetが成功した場合だけ、`.partial`とmeta未作成の孤立npy/indexを削除する。
6. 完成済みdatasetは`validated/skip`とし、未完datasetだけ確定済みprefixの次から抽出する。

HCUDB1は、外側の配布ディレクトリ（例: `.../HCUDB1`）を指定しても、実データを持つ内側の`HCUDB1`を一度解決し、検証・smoke・本番抽出のすべてへ同じパスを渡します。実行ログはdatasetごとに`[PRECHECK]`、`[SMOKE]`、`[RESUME]`、`[EXTRACT]`を表示します。`runs/official_cd/feature_preflight.json`にはresolved root、manifest SHA-256、検証件数、確定済み件数、残件数、回収候補、残容量が保存されます。

単一datasetをCLIから実行する場合も同じ順序です。

```powershell
python -m ser_pipeline extract-large `
  --snapshot C:\path\to\snapshot `
  --parity-report runs\official_cd\parity.json `
  --manifest runs\ser_manifests\hcudb1_official6_v1.jsonl `
  --audio-root C:\path\to\HCUDB1 `
  --cache-root runs\official_cd\cache\hcudb1 `
  --device cpu
```

metaまで確定したshardが再利用単位です。確定済みshardの欠損・hash不一致・破損は自動修復せず停止します。
