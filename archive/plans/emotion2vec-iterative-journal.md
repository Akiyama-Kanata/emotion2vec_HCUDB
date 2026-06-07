# ノートブック修正計画：vad_downstream/experiment.ipynb

## Context

### 問題の背景
`vad_downstream/experiment.ipynb` を実行すると以下のエラーが発生し、動かない状態。

**エラー1（主因）**: `ModuleNotFoundError: No module named 'torchcodec'`
- torchaudio 2.9+ は `torchcodec` を必須バックエンドとして採用した
- インストール済み torchaudio: **2.11.0+cpu**
- torchcodec は未インストール
- funasr が内部で `torchaudio.load()` を呼ぶため連鎖してエラーになる

**エラー2（副因）**: `NameError: name 'model' is not defined`
- 最終セル(`3a998d48`)が上のセルをスキップして実行されると `model` が未定義

### 環境と emotion2vec スクリプトの相違点

| 項目 | emotion2vec スクリプトの前提 | ユーザー環境の実態 |
|------|-----------------------------|--------------------|
| オーディオ読み込み | soundfile（scripts/直接使用） | torchaudio が torchcodec を要求 → 失敗 |
| PyTorch | GPU推奨（CUDA対応） | **CPU専用**（2.12.0+cpu） |
| torchcodec | 不要（スクリプト自体は使わない） | torchaudio が内部で要求 |
| ffmpeg | 不要（soundfile使用） | torchaudio のフォールバックとして使うが PATH 未設定 |
| fairseq | スクリプトで直接使用 | **未インストール**（funasr 経由なら不要） |
| npy-append-array | extract_features.py で使用 | 未インストール（ノートブックは numpy.save を使うため不要） |

### 判定
- ノートブックは funasr 経由でモデルを使うため fairseq 不要
- ノートブックは `np.save` で保存するため npy-append-array 不要
- **修正対象は3箇所のみ**

---

## 修正内容

### 修正 1: セル `c958a94a` を置き換え（ffmpeg PATH + 環境診断）

**現状**: `conda install ffmpeg` をサブプロセスで実行しているが、
現在の Python プロセスの PATH には反映されない。しかも ffmpeg はすでにインストール済み。

**修正後**:
```python
import os, sys

# ffmpeg が anaconda の Library/bin にあるので PATH に追加する
_ffmpeg_dir = r"C:\Users\RD004\anaconda3\Library\bin"
if _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"PATH に追加: {_ffmpeg_dir}")

# 環境確認
import torch, torchaudio, soundfile
print(f"torch      : {torch.__version__}")
print(f"torchaudio : {torchaudio.__version__}")
print(f"soundfile  : {soundfile.__version__}")
print(f"CUDA利用可能 : {torch.cuda.is_available()} (CPUモードで実行)")
```

### 修正 2: 新規セルを `c958a94a` の直後に挿入（torchaudio.load パッチ）

**理由**: torchaudio 2.9+ は torchcodec を必須とするが未インストール。
soundfile はインストール済みなので、それで代替する。
funasr が `torchaudio.load()` を内部呼び出しするため、funasr import より前に適用必須。

**挿入するコード**:
```python
# torchaudio 2.9+ は torchcodec を必須とするが未インストール。
# soundfile（インストール済み）で torchaudio.load を上書きして回避する。
import soundfile as sf
import torch
import torchaudio as _torchaudio

def _soundfile_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                    channels_first=True, format=None, buffer_size=4096, backend=None):
    data, sr = sf.read(str(uri), dtype="float32", always_2d=True)
    # soundfile: [time, ch]  →  torchaudio 標準: [ch, time]
    tensor = torch.from_numpy(data).T
    if frame_offset > 0:
        tensor = tensor[:, frame_offset:]
    if num_frames > 0:
        tensor = tensor[:, :num_frames]
    if not channels_first:
        tensor = tensor.T
    return tensor, sr

_torchaudio.load = _soundfile_load
print("torchaudio.load → soundfile バックエンドにパッチ完了")
```

### 修正 3: セル `3a998d48` の先頭に安全チェックを追加

**現状**: カーネル再起動後に最終セルを単独実行すると `model` が未定義で NameError。

**追加コード（先頭に挿入）**:
```python
if "model" not in dir() or model is None:
    raise RuntimeError(
        "model が未定義です。セル '5. モデル構築' から順番に実行してください。"
    )
```

---

## 修正対象ファイルと対象セル

| ファイル | セル ID | 操作 |
|---------|---------|------|
| `vad_downstream/experiment.ipynb` | `c958a94a` | 置き換え（conda → PATH設定 + 環境確認） |
| `vad_downstream/experiment.ipynb` | `c958a94a` の直後 | 新規挿入（torchaudio パッチセル） |
| `vad_downstream/experiment.ipynb` | `3a998d48` | 先頭行に安全チェックを追加 |

---

## 検証方法

1. カーネルをリスタートして全セルを上から順番に実行
2. 修正1セルで `soundfile : 0.13.1` などが表示されること
3. 修正2セルで `torchaudio.load → soundfile バックエンドにパッチ完了` が表示されること
4. セル `498a6b24`（特徴量抽出）が `特徴量ファイルがすでに存在します。スキップします。` または正常に完走すること
5. セル `baa07ae0`（データ読み込み）でサンプル数・形状が表示されること
6. Stage1/Stage2 学習が CPU で正常に実行されること
7. 最終評価セルで WA / UA / F1 が表示されること
