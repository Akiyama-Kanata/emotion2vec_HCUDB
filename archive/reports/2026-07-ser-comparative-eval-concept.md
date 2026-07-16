# 日本語/英語 SER 比較評価基盤 構想メモ

作成日: 2026-07-06

## この文書の目的

この文書は、HCUDB と IEMOCAP を使って日本語/英語の音声感情認識を比較評価するための構想を残すメモである。

重要な前提として、emotion2vec+ 本体はこのリポジトリで再実装しない。FunASR または公式 GitHub 実装を外部依存として利用し、モデル本体の重みは固定する。このリポジトリ側では、データ形式の統一、ラベル写像、特徴抽出結果の保存、head tuning、VAD/VA 経由評価、結果集約を担当する。

## 評価したいこと

研究上の主目的は、日本語音声感情認識の性能を上げつつ、英語音声感情認識の性能低下を確認することである。

今回の比較評価では、次の 4 条件を日本語 HCUDB と英語 IEMOCAP の両方で評価する。

1. emotion2vec+ zero-shot
2. emotion2vec+ Japanese head tuning
3. VAD/VA 経由モデル
4. VAD/VA 経由モデル Japanese head tuning

共通クラスは、現行の VAD 実装に合わせて `hap/sad/ang/dis` とする。HCUDB は VA 中心のデータとして扱い、初期実装では `target_dim=2`、つまり Valence/Arousal の 2 次元で統一する。Dominance は将来拡張とする。

## emotion2vec+ の扱い

emotion2vec+ は GitHub または FunASR から取得して利用する。リポジトリ内には本体コードや重い checkpoint をコピーしない。

このリポジトリで必要なのは、emotion2vec+ の薄い adapter である。adapter は次を担当する。

- wav から emotion2vec+ embedding を抽出する。
- zero-shot 時に FunASR の 9 クラス出力を受け取る。
- 9 クラス出力を `happy/sad/angry/disgusted -> hap/sad/ang/dis` に写像する。
- 使用した model id、checkpoint、commit hash、adapter 設定を metadata として保存する。

Japanese head tuning では emotion2vec+ 本体は更新しない。固定 embedding の上に載る分類 head だけを HCUDB train で学習する。

## データ形式

HCUDB と IEMOCAP は、いったん共通 manifest に変換する。

共通 manifest の列は次とする。

```text
utt_id,wav_path,language,split,label,valence,arousal,dominance_optional
```

`label` は `hap/sad/ang/dis` のみを採用する。対象外ラベルは暗黙に混ぜず、変換時に明示的に除外する。

IEMOCAP では `exc -> hap` は許可する。一方で、`neu -> dis` のような意味の違う置き換えは禁止する。`dis` は実際に `dis` として注釈された発話のみを使う。

HCUDB は実ファイル形式がこのリポジトリにまだ無いため、CSV/TSV から列名指定で manifest に変換できる設計にする。

## このリポジトリ側で作るもの

`ser_eval/` を新設し、比較評価に必要な CLI と共通処理を置く。

想定する構成は次の通り。

```text
ser_eval/
  manifest.py              # 共通 manifest の読み書き、検証、ラベルフィルタ
  convert_hcudb.py          # HCUDB CSV/TSV -> 共通 manifest
  convert_iemocap.py        # IEMOCAP annotation -> 共通 manifest
  export_prefix.py          # manifest -> .npy/.lengths/.emo/.vad prefix
  emotion2vec_adapter.py    # 外部 emotion2vec+ 呼び出し
  eval_zeroshot.py          # 9 class scores -> 4 class prediction
  train_direct_head.py      # fixed embedding + direct classification head
  run_experiments.py        # 4 条件 x 2 言語の実験 runner
```

既存の `vad_downstream` は、`hap/sad/ang/dis` を維持したまま、HCUDB/IEMOCAP の train/valid/test prefix を扱えるように汎用化する。

VAD/VA 経由モデルの baseline は IEMOCAP train で学習する。Japanese tuning 版は、その checkpoint を初期値として HCUDB train で追加学習する。

## 結果保存

実験 runner は、同じ設定ファイルから 4 条件 x 2 言語を実行し、`runs/ser_eval/...` 以下に結果を保存する。

最低限保存するものは次である。

- `summary.csv`
- `results.jsonl`
- confusion matrix
- class support
- checkpoint metadata
- emotion2vec+ の model id / checkpoint / commit hash
- 学習設定と評価設定の snapshot

`summary.csv` は少なくとも次の列を持つ。

```text
model_condition,train_source,eval_language,eval_dataset,accuracy,ua,weighted_f1,macro_f1,mean_ccc
```

## 指標

感情分類では、WA を `accuracy` として出す。加えて、UA、weighted F1、macro F1、confusion matrix、class support を保存する。

VAD/VA 経由モデルでは、分類指標に加えて Valence CCC、Arousal CCC、mean CCC を保存する。初期比較では Dominance は使わない。

## あなたが用意するもの

実装前に必要なのは、主に実データの場所と列情報である。

1. HCUDB の wav ディレクトリ
2. HCUDB の annotation CSV/TSV
3. HCUDB annotation 内の `utt_id`, `wav_path`, `label`, `valence`, `arousal`, `split` に対応する列名
4. HCUDB に既存 split があるか、こちらで train/valid/test を作るか
5. IEMOCAP の wav と annotation の場所
6. emotion2vec+ を FunASR 版で使うか、GitHub clone 版で使うか

この情報が決まれば、`configs/ser_eval.yaml` に集約し、変換 CLI と実験 runner から参照する。

## テスト方針

まず synthetic data で次を確認する。

- manifest のラベル写像
- 対象外ラベルの除外
- IEMOCAP の `exc -> hap`
- `neu -> dis` が起きないこと
- synthetic IEMOCAP annotation から `.emo/.vad` が生成され、ID 順序が一致すること
- direct head が小さい `.npy/.lengths/.emo/.vad` で学習・評価できること
- VAD/VA 経由 head が小さい `.npy/.lengths/.emo/.vad` で学習・評価できること
- zero-shot 評価で FunASR 呼び出し部分を mock し、9 クラス scores から 4 クラス prediction へ変換できること

最終的には、既存テストに加えて WSL の `emotion2vec-py310` 環境で次を通す。

```bash
python -m unittest discover -s tests
```

## 最初の実装順

1. `configs/ser_eval.yaml` の雛形を作る。
2. 共通 manifest の dataclass/validator を作る。
3. synthetic manifest のテストを作る。
4. HCUDB CSV/TSV 変換 CLI を作る。
5. IEMOCAP annotation 変換 CLI を作る。
6. 既存 prefix 形式への export を作る。
7. emotion2vec+ adapter を薄く追加する。
8. zero-shot 評価を mock 可能な形で実装する。
9. fixed embedding + direct head tuning を実装する。
10. VAD/VA 経由評価を HCUDB/IEMOCAP 両方に対応させる。
11. 実験 runner と `summary.csv` 出力を作る。

## 注意点

- emotion2vec+ 本体の fine-tuning とは呼ばない。今回は固定 emotion2vec+ 特徴上の head tuning である。
- `neu` を `dis` に寄せるようなラベル数合わせはしない。
- HCUDB に Dominance が無い前提で、初期比較は VA 2 次元に限定する。
- 実データがない状態では、性能値ではなく変換・学習・評価 pipeline の正しさを synthetic test で保証する。
