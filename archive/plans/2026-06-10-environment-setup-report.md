# 2026-06-10 環境構築ログと次回作業

## 目的

emotion2vecを日本語音声感情認識向けに段階的に拡張する前に、WSL/Linux上で既存コードを再現できる環境を作る。
この段階ではVAD回帰器、VAD経由分類器、fine-tuning実装には進まない。

## ここまでに決めた方針

- 1回の作業では1つの小さい変更だけ扱う。
- まずは環境構築と依存関係の確認を完了させる。
- 実験実行環境はWindows PowerShellではなく、Ubuntu/WSL内のbashを使う。
- `base` conda環境は使わず、emotion2vec専用環境を作る。
- Pythonはfairseq互換性を優先して3.10を使う。

## 専用環境を作った理由

Ubuntu/WSL側の既存 `base` 環境は Python 3.12 系だったが、emotion2vec の既存実装で使う `fairseq==0.12.2`、`hydra-core==1.0.7`、`omegaconf==2.0.6` は古い依存関係を含む。

そのため、`base` 環境に直接入れると他の作業環境を壊す可能性があり、かつ Python 3.12 では依存解決やビルドで詰まる可能性が高いと判断した。

このリポジトリ専用に `emotion2vec-py310` を作り、Python 3.10 と `pip<24.1` を使う方針にした。以後の実験・テストは原則としてこの環境で行う。

```bash
conda activate emotion2vec-py310
cd /mnt/c/Users/RD004/Documents/lab/emotion2vec
```

## リポジトリ側の変更

`requirements.txt` を追加し、その後インストール失敗に合わせて最小修正した。

現在の主な内容:

```txt
torch>=1.13
fairseq==0.12.2
hydra-core==1.0.7
omegaconf==2.0.6
soundfile
numpy<2
npy-append-array
tqdm
scikit-learn
```

補足:

- `fairseq==0.12.2` は `omegaconf<2.1` を要求する。
- `omegaconf==2.0.6` は新しいpipでメタデータが拒否されるため、事前に `pip<24.1` へ下げる必要がある。
- `numpy<2` に固定し、古い依存との互換性を優先した。

## 実行ログ要約

### 1. PowerShell側の確認

- Windows側では `python` コマンドが見つからなかった。
- `pip` はWindowsのPython 3.13に紐づいていた。
- そのため、PowerShell上で依存関係を確認するのは研究用Linux環境の確認として不適切と判断した。

### 2. WSL/Ubuntuの確認

ユーザーがPowerShellで `wsl` を実行し、Ubuntu 22.04.2 LTSが起動することを確認した。

作業ディレクトリ:

```bash
/mnt/c/Users/RD004/Documents/lab/emotion2vec
```

### 3. 既存base環境の確認

Ubuntu内の初期状態はcondaの `base` 環境だった。

```text
/home/akiyama/miniforge/bin/python
Python 3.12.11
pip 25.1.1
```

Python 3.12は `fairseq==0.12.2` で詰まる可能性が高いため、使わない方針にした。

### 4. 専用conda環境の作成

次の環境を作成した。

```bash
conda create -n emotion2vec-py310 python=3.10 -y
conda activate emotion2vec-py310
```

確認結果:

```text
Python 3.10.20
pip 26.1.2
```

### 5. 初回インストール失敗

`python -m pip install -r requirements.txt` は失敗した。

原因:

- `pip 26.1.2` が `omegaconf 2.0.x` の古い依存メタデータを拒否した。
- `fairseq==0.12.2`、`hydra-core`、`omegaconf` の依存解決が衝突した。

対応:

- `hydra-core==1.0.7`
- `omegaconf==2.0.6`
- `numpy<2`
- `pip<24.1` を事前に入れる手順

に修正した。

### 6. 修正後インストール成功

ユーザーが以下を実行した。

```bash
python -m pip install "pip<24.1"
python -m pip install -r requirements.txt
```

結果:

- `fairseq` と `antlr4-python3-runtime` のビルド成功。
- `fairseq-0.12.2` のインストール成功。
- `torch-2.12.0`、`torchaudio-2.11.0`、`hydra-core-1.0.7`、`omegaconf-2.0.6`、`numpy-1.26.4` などがインストールされた。
- pipから `omegaconf 2.0.6` の非標準依存指定に関する警告が出たが、これは想定済み。

## 現在の状態

- WSL/Ubuntuは起動できる。
- 専用環境 `emotion2vec-py310` は作成済み。
- `requirements.txt` のインストールは成功済み。
- まだimport確認は未実施。
- まだVAD関連のREADME、データローダー、モデル、学習ループは作っていない。

## 次回の最小ステップ

次回はimport確認だけを行う。

WSL/Ubuntu内で、必ず専用環境を有効化してから実行する。

```bash
cd /mnt/c/Users/RD004/Documents/lab/emotion2vec
conda activate emotion2vec-py310
python -c "import torch, fairseq, soundfile, hydra, numpy; print('ok')"
```

期待結果:

```text
ok
```

## 次回の判定

- `ok` が出た場合:
  - 環境確認フェーズは完了。
  - 次の小変更として `vad_downstream/README.md` だけを作り、VADデータ形式を文章で固定する。

- importエラーが出た場合:
  - そのエラーだけを対象にする。
  - 次の作業は `requirements.txt` の追加修正またはインストール手順修正だけに限定する。

## まだやらないこと

- VAD回帰器の実装。
- VAD経由カテゴリ分類器の実装。
- emotion2vec本体のfine-tuning。
- IEMOCAPとの混合学習。
- 日本語データセット固有の前処理。
- README全体の大きな更新。
