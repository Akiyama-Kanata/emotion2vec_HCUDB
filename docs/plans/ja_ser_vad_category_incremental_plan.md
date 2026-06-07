# 日本語音声感情認識・VAD・英語性能維持の段階的計画

## 目的

このリポジトリでは、emotion2vecを使って次の実現を目指す。

- 日本語音声感情認識の精度を上げる
- 出力として3次元VAD、`valence, arousal, dominance`、を持つ
- VADを踏まえたカテゴリ分類も行う
- 日本語性能を上げつつ、英語音声感情認識、特にIEMOCAPで比較可能な感情、の性能を保つ

ただし、現状の日本語データには `valence`, `arousal`, `emotion/category` の正解ラベルはあるが、`dominance` の正解ラベルはない。

## 現状

現状でできていること。

- `upstream/` にemotion2vec本体の元コードがある
- `scripts/extract_features.py` で単一wavからemotion2vec特徴量を抽出できる
- `vad_downstream/` に固定emotion2vec特徴量から3次元VADを回帰する導線がある
- VAD出力順は `valence, arousal, dominance` に整理されている
- 欠損ラベルmask付きCCC lossがある
- `iemocap_downstream/` にIEMOCAP 4感情分類の参照実装がある
- `archive/vad_iemocap_two_stage/` に、VADを中間表現として使う古い分類実験が残っている

現状で足りていないこと。

- 現行本線はVAD回帰のみで、カテゴリ分類が統合されていない
- 日本語データにdominance正解がないため、日本語だけでは3次元VADを完全には教師あり学習・評価できない
- 英語性能を保つための評価・学習導線がない
- 日本語と英語で比較可能な感情カテゴリのマッピングが未整理
- 実データ全体から特徴量cacheを一括作成する安定した導線が弱い
- `torch`入り環境でのテスト・smoke runが未完了

## Phase 1: 現状の土台を固める

- `torch` が使えるPython環境を用意する
- 既存テストを実行する
  - `py -m unittest discover -s tests`
- `vad_downstream/` がdummy fixtureで動くことを確認する
- 日本語データをCSV化する
  - `file_path`
  - `valence`
  - `arousal`
  - `emotion`
  - `split` または `session`
- `dominance` は日本語では欠損扱いにする
- まずは固定emotion2vec特徴量からVA回帰が動く状態を作る

## Phase 2: 日本語カテゴリ分類baselineを作る

- 日本語音声からemotion2vec特徴量cacheを作る
- VADとは別に、カテゴリ分類だけのbaselineを作る
- 評価指標を固定する
  - accuracy
  - macro F1
  - confusion matrix
- 日本語カテゴリラベルを、IEMOCAPやemotion2vec+と比較可能な形に整理する
- この段階では、VADと分類を強く結合しない

## Phase 3: VA/VADと分類を統合する

- モデル出力を2系統にする
  - `vad`: `valence, arousal, dominance`
  - `logits`: 感情カテゴリ分類
- 日本語データでは `valence` と `arousal` だけに回帰lossをかける
- `dominance` は出力するが、日本語ではloss対象外にする
- 分類器は、emotion2vec特徴量に加えてVAD予測値も使える構成にする
- 比較条件を用意する
  - 分類のみ
  - VA回帰 + 分類の同時学習
  - VAD予測値を分類入力に追加

## Phase 4: 英語性能維持を評価する

- IEMOCAP分類baselineを再現する
- 日本語学習後に、IEMOCAPで性能がどれだけ変わるか測る
- 比較対象は、日英で対応可能な感情カテゴリに絞る
- 記録する指標
  - WA / accuracy
  - UA
  - F1
  - クラス別性能
- 英語性能が落ちる場合は、段階的に対策する
  - 英語データを学習に混ぜる
  - 日本語と英語のbatch比率を調整する
  - 既存英語モデルの出力をteacherとして使う

## Phase 5: Dominanceの扱いを決める

短期方針。

- 3次元VAD出力は維持する
- 日本語では `dominance` を学習・評価しない
- 日本語のlossは `valence` と `arousal` のみにかける

中期方針。

- dominance付き英語データがあれば、それでDを保守する
- なければ既存VADモデルの出力をteacherとして使うか検討する

長期方針。

- 日本語dominanceラベルを追加するか検討する
- 疑似ラベルを使う場合は、その妥当性を別途評価する

## 実装時の前提

- 当面はemotion2vec本体を更新しない
- 固定emotion2vec特徴量の上に下流モデルを学習する
- 日本語データには `valence`, `arousal`, `emotion` がある
- 日本語データには `dominance` はない
- 英語性能維持の主な評価対象はIEMOCAP
- 実装は一度に完成形を作らず、段階ごとに検証する
