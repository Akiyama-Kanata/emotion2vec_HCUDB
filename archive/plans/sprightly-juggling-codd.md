# 計画: VAD Downstream 最小動作版の実装

## Context

`vad_downstream/experiment.ipynb` を動かすことが目標。既存のモジュール（model.py, data.py, loss.py, train.py）はほぼ完成しているが、Windows + Jupyter 環境での DataLoader の設定ミス（`num_workers=4`）によりノートブックが止まる。最小要件の方針で「まず動く状態」にする。

**判断: 既存ノートブックを編集する**（新規作成しない）

理由:
- 4つのモジュールは正しく実装済みで再利用できる
- ノートブックの構造は正しい（バグは限定的）
- 新規作成は重複になる

---

## 問題の特定

### バグ1（必須修正）: Windows + Jupyter で DataLoader がフリーズ
- **ファイル**: `vad_downstream/data.py` の `build_dataloaders()` 行 203-221
- **原因**: `num_workers=4` は Windows の Jupyter では動かない（multiprocessing のフォーク方式の違い）
- **修正**: `num_workers=0, pin_memory=False` に変更（CPUオンリー環境では pin_memory=True も警告が出る）

### バグ2（データファイル）: `va_labels.txt` が未確認
- `load_iemocap_with_va()` が `<utterance_id> <valence> <arousal>` 形式のファイルを要求
- このファイルが存在しない場合はデータ読み込みで失敗する
- → ノートブックに TODO として明記

### バグ3（ノートブックのセル順序）: パス設定
- `FEAT_PATH`（特徴量 .npy ファイルのパス）はユーザー環境に依存
- ノートブックのセルにパス確認のチェックを追加

---

## 実装手順

### Step 1: `data.py` の修正

`build_dataloaders()` 内の DataLoader 生成を全て修正:

```python
# 変更前
num_workers=4, pin_memory=True

# 変更後  
num_workers=0, pin_memory=False
```

対象行: 203-204, 206-207, 215-216, 217-218, 220-221（全DataLoader呼び出し）

### Step 2: `experiment.ipynb` のパス設定セルに TODO を追加

セクション3（パス設定）のマークダウンセルに以下を追加:
- `FEAT_PATH`: 特徴量 .npy ファイルの場所（ユーザーが設定する）
- `VA_PATH`: `va_labels.txt` のパス（**TODO: IEMOCAPのアノテーションから作成が必要**）

### Step 3: ノートブック冒頭に TODO マークダウンセルを追加

```markdown
## TODO（未実装・要確認）

- [ ] **va_labels.txt の作成**: IEMOCAP のアノテーションファイルから
      `<utterance_id> <valence> <arousal>` 形式で生成する必要がある
- [ ] **Dominance ラベル**: 現在の loss.py は V/A のみ。D は `stage1_loss` に未追加
- [ ] **学習曲線の可視化**: Stage1/Stage2 の損失プロットは概要のみ
- [ ] **ハイパーパラメータ調整**: lr, batch_size, epochs は暫定値
- [ ] **5-fold クロスバリデーション**: 現在は fold=0 のみで動作確認
```

---

## 変更対象ファイル

| ファイル | 変更内容 |
|---------|---------|
| `vad_downstream/data.py` | `num_workers=0, pin_memory=False` に修正（5箇所） |
| `vad_downstream/experiment.ipynb` | 冒頭に TODO マークダウンセル追加、パス設定セルにコメント追加 |

---

## 動作確認手順

1. `data.py` を修正後、ノートブックのセクション3「パス設定」セルを実行
2. `FEAT_PATH` が正しければセクション4「データ読み込み」が動く
3. `va_labels.txt` が存在すればセクション5「モデル構築」→セクション6「DataLoader作成」が動く
4. エラーが出なければ Stage1 学習（セクション7）まで流す

> **注**: `va_labels.txt` が存在しない場合は Stage1 より前で止まる。その場合はノートブックの TODO に従い作成する。
