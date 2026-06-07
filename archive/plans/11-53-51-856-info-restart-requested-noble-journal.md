# Plan: OMP Error #15 (libiomp5md.dll 二重初期化) の修正

## Context

Jupyter カーネルの再起動後、約30秒でカーネルが ExitCode: 3 で死亡する。  
原因は `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`  
つまり OpenMP ランタイムが二重にロードされている。

**根本原因**:  
ノートブックの最初のセルで `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` を設定しているが、  
これはカーネル起動後の Python コードとして実行されるため、**DLL がロードされた後に設定される可能性がある**。  
`ipykernel` 自体の起動時や numpy 等の自動インポート時に `libiomp5md.dll` が先に初期化されてしまうと、  
後からセル内で環境変数を設定しても手遅れになる。

**補足**:  
`_patch_notebook.py` はノートブックの `c958a94a` セルから `KMP_DUPLICATE_LIB_OK` 行を削除する設計になっており、  
これが実行されると上記の回避策も失われる。

## 修正方針

**二段構えで確実に対処する：**

### 修正1: `kernel.json` に環境変数を追加（主要修正）

ファイル: `c:\Users\RD004\anaconda3\share\jupyter\kernels\python3\kernel.json`

Python プロセス起動前（= DLL ロード前）に OS レベルで環境変数を設定することで、  
どのセルより先に `KMP_DUPLICATE_LIB_OK=TRUE` が有効になる。

**変更内容**:
```json
{
 "argv": ["C:/Users/RD004/anaconda3/python.exe", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
 "display_name": "Python 3 (ipykernel)",
 "language": "python",
 "metadata": {"debugger": false},
 "env": {
  "KMP_DUPLICATE_LIB_OK": "TRUE"
 }
}
```

### 修正2: `_patch_notebook.py` で KMP 行を保持（補助修正）

ファイル: `c:\Users\RD004\Documents\lab\emotion2vec\vad_downstream\_patch_notebook.py`

`new_src_1` の置き換えリストに以下の行を追加し、パッチ後もノートブックに設定が残るようにする:

```python
'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 二重ロード回避\n',
```

`import os, sys` の直後（`_ffmpeg_dir = ...` の前）に挿入する。

## 修正対象ファイル

| ファイル | 変更内容 |
|---|---|
| `c:\Users\RD004\anaconda3\share\jupyter\kernels\python3\kernel.json` | `"env"` セクションを追加 |
| `c:\Users\RD004\Documents\lab\emotion2vec\vad_downstream\_patch_notebook.py` | `new_src_1` に KMP 行を追加 |

## 検証方法

1. `kernel.json` 保存後、VS Code で Jupyter カーネルを再起動
2. ノートブックのセルを順番に実行し、カーネルが ExitCode: 3 で落ちないことを確認
3. torch / torchaudio / soundfile のバージョンが正常に表示されることを確認
