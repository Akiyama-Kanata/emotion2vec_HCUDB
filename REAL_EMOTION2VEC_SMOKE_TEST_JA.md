# 実 emotion2vec CPU 配線テスト手順

## 1. このテストの目的

この手順では、次の処理が CPU 上で最後まで動くことを確認します。

```text
scripts/test.wav
  -> 学習済み emotion2vec encoder
  -> 未学習のランダム VAD head
  -> 未学習のランダム4分類器
  -> VAD値と4分類結果を含むJSON
```

確認対象は処理の接続と実行可否です。VAD値や分類結果の精度・意味は評価しません。
下流の head と分類器は未学習なので、出力値は実行ごとに変わる可能性があります。

## 2. 現在用意されているもの

公式の学習済み checkpoint は取得済みです。

```text
Windows:
C:\Users\RD004\Documents\lab\emotion2vec\artifacts\checkpoints\emotion2vec_base.pt

WSL:
/mnt/c/Users/RD004/Documents/lab/emotion2vec/artifacts/checkpoints/emotion2vec_base.pt
```

- 配布元: <https://huggingface.co/emotion2vec/emotion2vec_base/blob/main/emotion2vec_base.pt>
- ファイルサイズ: `1,125,606,009` bytes
- SHA-256: `4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd`
- checkpoint は `.gitignore` の対象であり、Gitへコミットしません。

## 3. checkpointの確認

リポジトリのルートを開いた Windows PowerShell で実行します。

```powershell
Get-Item .\artifacts\checkpoints\emotion2vec_base.pt |
  Select-Object Length, FullName

Get-FileHash -Algorithm SHA256 `
  .\artifacts\checkpoints\emotion2vec_base.pt
```

期待する結果は次のとおりです。

```text
Length: 1125606009
SHA256: 4F14DDF7BA394BCAFDD4BFF6AE0F24AB2E4134260D4DD42C58EA791A201B02DD
```

サイズまたはhashが異なる場合、そのcheckpointを読み込まず、再取得してください。

## 4. CPU配線テストの実行

Windows PowerShellでリポジトリのルートへ移動します。

```powershell
Set-Location C:\Users\RD004\Documents\lab\emotion2vec
New-Item -ItemType Directory -Force .\outputs | Out-Null
```

続けて、次のコマンドを実行します。

```powershell
wsl -d Ubuntu `
  --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec `
  -e env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 `
  /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python `
  -m vad_downstream.infer_vad_emotion `
  --wav scripts/test.wav `
  --model-dir upstream `
  --checkpoint artifacts/checkpoints/emotion2vec_base.pt `
  --allow-random-model `
  --target-dim 3 `
  --device cpu `
  --output outputs/real_emotion2vec_smoke.json
```

`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` は、古い fairseq checkpointをPyTorch
2.12でロードするために指定しています。pickleによるロードを許可するため、公式配布元から取得してhashを確認した上記checkpointだけに使用してください。

## 5. 成功結果の確認

プロセスが終了コード `0` で終わり、次のファイルが作られていれば推論処理は完走しています。

```text
outputs/real_emotion2vec_smoke.json
```

PowerShellでJSONを確認します。

```powershell
$result = Get-Content -Raw .\outputs\real_emotion2vec_smoke.json |
  ConvertFrom-Json

$result | ConvertTo-Json -Depth 10
```

次の条件を確認してください。

- `random_model` が `true` である。
- `target_dim` が `3` である。
- `vad` に `valence`、`arousal`、`dominance` の有限値がある。
- `class_labels` が `hap`, `sad`, `ang`, `dis` である。
- `prediction.code` が上記4ラベルのいずれかである。
- `probabilities` に4クラス分の有限値がある。
- `classifier_checkpoint` が `null` である。

確率の合計は次のコマンドで確認できます。浮動小数点誤差を除き、約 `1` になれば正常です。

```powershell
($result.probabilities.PSObject.Properties.Value |
  Measure-Object -Sum).Sum
```

`--model-dir upstream` と `--checkpoint artifacts/checkpoints/emotion2vec_base.pt`
を指定しているため、このコマンドが成功すれば仮特徴抽出器ではなく実emotion2vecを通っています。

## 6. このテストでは保証しないこと

このテストでは、以下は保証されません。

- 出力されたVAD値が人間の感情評価と一致すること。
- 4分類結果が音声の正解感情と一致すること。
- HCUDB1で学習したVAD headまたは分類器が動くこと。
- 分類精度、再現率、F1などの性能。

これらを確認するには、学習用特徴・VADラベル・4感情ラベルを作成し、VAD媒介分類headを学習したcheckpointへ置き換える必要があります。現在のHCUDB1 CSVにある連続値はvalenceとarousalだけなので、学習済みheadでも3次元VADを維持する場合はdominanceラベルを別途用意する必要があります。

## 7. エラー時の確認

### `Weights only load failed` と表示される

実行コマンドに次の指定があることを確認します。

```text
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

### checkpointが見つからない

PowerShellで次を実行し、`True` になることを確認します。

```powershell
Test-Path .\artifacts\checkpoints\emotion2vec_base.pt
```

### `WSL_E_DISTRO_NOT_FOUND` と表示される

登録済みdistributionを確認します。

```powershell
wsl --list --verbose
```

このPCで確認済みのdistribution名は `Ubuntu` です。

### `ModuleNotFoundError` が表示される

指定したPython環境のimportを確認します。

```powershell
wsl -d Ubuntu `
  -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python `
  -c "import torch, fairseq, numpy, soundfile; print('imports: OK')"
```

### メモリ不足で終了する

他の大きなアプリケーションを閉じ、WSLを再起動してから再実行します。

```powershell
wsl --shutdown
```

その後、もう一度CPU配線テストを実行してください。
