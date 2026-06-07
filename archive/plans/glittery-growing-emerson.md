# 原因診断：`ValueError: num_samples should be a positive integer value, but got num_samples=0`

## Context

ノートブック `vad_downstream/experiment.ipynb` のセル 10（DataLoader 作成）で
`ValueError` が発生し、以降のセルがすべて `notebook controller is DISPOSED` になっている。

---

## 根本原因

### 問題の連鎖

| ステップ | 状態 | 値 |
|---------|------|-----|
| ローカルに存在するセッション | `Session1/` フォルダのみ | 1085 発話 |
| `iemocap_feats.emo` 行数 | Session 1 のみ抽出済み | **1085 行** |
| `train.emo` 行数 | 全5セッション分 | 5531 行 |
| `N_SAMPLES` の設定 | 全5セッション想定 | `[1085, 1023, 1151, 1031, 1241]`（合計 5531） |
| `FOLD = 0` のテスト範囲 | `[test_start=0, test_end=1085)` | **全データがテストセット** |
| 訓練セット `tv_labels` | `labels[:0] + labels[1085:]` | **空（0 件）** |
| `n_train = int(0.8 * 0)` | → 0 | `DataLoader` に 0 を渡してクラッシュ |

### 一言でいうと

> **ローカルには Session 1 しかない（1085 件）のに、コードは全 5 セッション（5531 件）を前提として
> FOLD=0 で Session 1 をまるごとテストセットに割り当てる。
> 残りがゼロになるため DataLoader が `num_samples=0` でクラッシュする。**

---

## 修正方針（2 択）

### Option A：Session 2〜5 をダウンロードして再抽出（正式対応）

1. Session 2〜5 の WAV ファイルを `C:\Users\RD004\Documents\lab\data\iemocap\` に配置する
2. 既存の `iemocap_feats.npy` / `.lengths` / `.emo` を削除する
3. 特徴量抽出セル（セル 3）を再実行する → 5531 発話が抽出される
4. `N_SAMPLES` はそのまま使用可能

→ 5-fold CV が本来通り動作する。論文として正しい評価ができる。

### Option B：Session 1 のみで簡易 train/val/test 分割（暫定対応）

`vad_downstream/experiment.ipynb` のセル 10（DataLoader 作成部分）を以下に書き換える：

```python
# Session 1 のみのデータを 70/15/15 で分割する
from torch.utils.data import random_split

all_ds = SpeechDatasetVAD(
    data["feats"], data["sizes"], data["offsets"], data["labels"], data["va_labels"]
)

n = len(all_ds)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

train_ds, val_ds, test_ds = random_split(all_ds, [n_train, n_val, n_test])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=all_ds.collator,
                          num_workers=0, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, collate_fn=all_ds.collator,
                          num_workers=0, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, collate_fn=all_ds.collator,
                          num_workers=0, shuffle=False)

print(f"train: {n_train}  val: {n_val}  test: {n_test}")
```

→ すぐに実行できるが、セッション間の話者独立評価にならない（暫定用途のみ）。

---

## 追加ポイント：Windows の `num_workers=4` 問題

`DataLoader` で `num_workers=4` を設定しているが、Windows ではマルチプロセス DataLoader が
クラッシュしやすい。Option B のコードでは `num_workers=0` に変更済み。
Option A でも Windows 環境では `num_workers=0` 推奨。

---

## 確認手順

修正後、セル 10 を実行して以下が出ることを確認する：

```
train バッチ数: <0 より大きい数>
val   バッチ数: <0 より大きい数>
test  バッチ数: <0 より大きい数>
```
