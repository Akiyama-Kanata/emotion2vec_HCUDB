# 固定 emotion2vec 下流学習実験：今後の懸念と比較条件

作成日: 2026-07-20

## 目的

この文書は、emotion2vec 系の encoder 本体を固定し、新しく実装した直接分類 head と VAD/VA 媒介分類 head だけを学習する実験について、今後解決すべき懸念と必要な比較条件を整理するものである。

対象とする encoder は次の 2 種類である。

- emotion2vec
- emotion2vec+ large

今回の範囲では、どちらの encoder も日本語・英語データで更新しない。学習対象は、このリポジトリで新しく実装した下流 head のみとする。

## 用語と主張できる範囲

emotion2vec は感情特徴を抽出する encoder であり、単体では感情クラスを出力しない。そのため、分類 head を持たない emotion2vec に分類精度を定義することはできない。

固定した encoder に分類 head を取り付け、head だけを学習して特徴表現を評価する方法は、fine-tuning ではなく、原則として `linear probing` または `frozen-feature evaluation` と呼ぶ。

今回の実験から主張できるのは、次の内容である。

- 固定した emotion2vec 特徴を使い、日本語または英語の感情をどの程度分類できるか。
- 同じ固定特徴に対して、直接分類 head と VAD/VA 媒介分類 head のどちらが有効か。
- 同じ下流条件で、emotion2vec と emotion2vec+ large のどちらの特徴が有効か。
- 英語で学習した head が、日本語へどの程度転移するか。

一方、encoder を更新しないため、次のようには主張しない。

- 「emotion2vec 自体の日本語性能が向上した」
- 「日本語学習によって emotion2vec の英語表現が劣化した」

日本語学習後に英語精度が低下した場合、今回の条件では encoder の忘却ではなく、下流 head の忘却またはデータ分布差を観測している。

## 主比較の 2 × 2 条件

日本語と英語のそれぞれについて、次の 4 条件を基本とする。

| 条件 | 固定 encoder | 学習する下流 head |
|---|---|---|
| A | emotion2vec | 直接分類 head |
| B | emotion2vec | VAD/VA → 分類 head |
| C | emotion2vec+ large | 直接分類 head |
| D | emotion2vec+ large | VAD/VA → 分類 head |

この比較から、次を切り分ける。

- A 対 B: emotion2vec 特徴に対する VAD/VA 媒介の効果
- C 対 D: emotion2vec+ large 特徴に対する VAD/VA 媒介の効果
- A 対 C: 直接分類条件での encoder の差
- B 対 D: VAD/VA 媒介条件での encoder の差

入力特徴次元が encoder 間で異なる場合、head の最初の入力層だけは変更が必要になる。隠れ層数、隠れ次元、activation、dropout、pooling、学習条件は可能な限り統一し、変更点を記録する。

## 推奨する実験順序

### 1. 英語で各 head をゼロから検証する

まず IEMOCAP などの英語データで、直接分類 head と VAD/VA 媒介分類 head をそれぞれランダム初期値から学習する。

この段階の目的は、次のとおりである。

- 新規実装した head が実データで学習可能であることを確認する。
- 固定 emotion2vec を用いた直接分類結果が、既存研究から大きく外れていないことを確認する。
- 英語内で、直接分類と VAD/VA 媒介分類を比較する。

英語で先に検証すること自体は有用だが、英語学習済み head を日本語実験の唯一の初期値にはしない。

### 2. 日本語でも各 head をゼロから学習する

日本語実験の主基準は、日本語 train split だけを使い、各 head をランダム初期値から学習する `JA-scratch` とする。

これにより、英語事前学習の影響を含まない、日本語下流学習の基準性能を得る。

### 3. 英語から日本語への転移を追加条件として調べる

英語学習済み head を日本語学習の初期値にする実験は、主基準ではなく、転移学習の追加条件として扱う。

| 条件 | 学習・評価方法 | 確認すること |
|---|---|---|
| EN-scratch | 英語でゼロから学習し、英語 test で評価 | 英語下流性能と head の妥当性 |
| JA-scratch | 日本語でゼロから学習し、日本語 test で評価 | 日本語下流性能の主基準 |
| EN-zero-shot-JA | 英語学習済み head を日本語 test へ直接適用 | 学習前の言語間転移 |
| EN→JA | 英語学習済み head を日本語で追加学習 | 英語事前学習が日本語へ与える効果 |
| EN→JA→EN-eval | 日本語追加学習後に英語 test で再評価 | 下流 head の英語忘却 |

`EN→JA` の効果は、同じ日本語 test に対する `JA-scratch` との差として判断する。`EN→JA` だけを実行すると、得られた性能が英語事前学習によるものか、日本語学習によるものかを分離できない。

単一 head で英語と日本語の両方を維持したい場合は、英語と日本語の混合学習も追加の基準とする。

## 主な懸念

### 1. 「emotion2vec そのままの精度」は存在しない

emotion2vec 単体には分類出力がないため、学習前の分類精度と学習後の分類精度を直接比較できない。ランダム初期化 head の精度を「emotion2vec そのままの精度」とみなすことにも研究上の意味はない。

固定 encoder の性能は、同じ設計の probe/head を同じ条件で学習して測定する。

### 2. 英語事前学習を必須にすると要因が混ざる

英語で head を学習してから日本語で追加学習する場合、日本語性能には英語事前学習と日本語学習の両方が影響する。必ず `JA-scratch` を置き、`EN→JA` と比較する。

### 3. 言語間の絶対精度を単純比較できない

英語と日本語で異なるデータセットを使う場合、収録条件、話者数、発話長、クラス分布、ラベル品質、課題難易度が異なる。「英語 70%、日本語 65% なので日本語に弱い」とは単純に結論づけない。

各言語内で、直接分類から VAD/VA 媒介分類への差を比較する。

```text
Delta_EN = Metric(VAD/VA, EN) - Metric(Direct, EN)
Delta_JA = Metric(VAD/VA, JA) - Metric(Direct, JA)
```

encoder 間の比較も、同じ言語、同じ split、同じ head 条件の中で行う。

### 4. 感情クラスの意味を一致させる必要がある

英語と日本語で共通クラスを定義し、意味の異なるラベルを件数合わせのために置換しない。特に `neu -> dis` のような置換は禁止する。IEMOCAP の `exc -> hap` を使う場合は、全条件で同じ規則を適用する。

共通クラスを絞ることで各クラスの件数が不足する可能性がある。train/valid/test ごとにクラス件数を保存し、少数クラスや欠損クラスを確認する。

### 5. VAD/VA ラベルの互換性を確認する必要がある

VAD/VA 媒介 head を教師あり学習するには、英語・日本語の各 train データに対応する VAD/VA 正解値が必要である。次を事前に確認する。

- 両データセットに Valence/Arousal が存在するか。
- 評価尺度と値域が一致するか。
- 発話単位のラベルであるか。
- annotator 集約方法が一致するか。
- Dominance を両方で利用できるか。

尺度が異なる場合は正規化方法を train split だけから決め、valid/test の統計量を使わない。Dominance が共通でなければ、初期比較は VA の 2 次元に限定する。

### 6. VAD/VA 媒介 head の性能差には情報ボトルネックが含まれる

直接分類 head は encoder 特徴を直接利用するが、VAD/VA 媒介 head は情報を 2 または 3 次元へ圧縮する。分類精度が低い場合、実装不良だけでなく、VAD/VA では表せないカテゴリ情報が失われた可能性がある。

分類指標だけでなく、Valence CCC、Arousal CCC、必要に応じて Dominance CCC を同時に報告する。

### 7. emotion2vec+ large の公式分類 head を主比較へ混ぜない

emotion2vec+ large の公式 9 クラス head は、大規模な疑似ラベルデータで学習されており、今回作る日本語・英語下流 head と学習条件が異なる。主比較では公式 head を外し、emotion2vec+ large も固定特徴抽出器として扱う。

公式 9 クラス出力は、同一条件の表現比較ではなく、既製モデルの zero-shot 参考値として別表にする。公式学習データの詳細が完全には公開されていないため、評価データとの重複を完全には否定できない点も制約として記載する。

### 8. 話者リークを防ぐ必要がある

train/valid/test は発話単位のランダム分割ではなく、話者またはセッションを分離する。全条件で同じ split ID を使い、utterance ID と speaker ID の重複がないことを機械的に検証する。

英語学習、日本語学習、zero-shot、追加学習のすべてで test split をモデル選択や正規化に使用しない。

### 9. head の公平性とハイパーパラメータ選択

直接分類 head と VAD/VA 媒介 head では構造と損失が異なるため、パラメータ数を完全には一致させられない可能性がある。少なくとも次を保存する。

- trainable parameter 数
- optimizer、learning rate、batch size
- epoch 数と early stopping 条件
- loss の種類と重み
- pooling、normalization、dropout
- random seed

一方の test 結果を見てハイパーパラメータを調整しない。調整は各 train/valid split の範囲に限定する。

### 10. 複数 seed と不確実性を報告する

1 回の学習結果だけで優劣を判断せず、同一 split で少なくとも 3 seed、可能なら 5 seed を実行する。Accuracy だけでなく、UA/UAR、Macro-F1、クラス別 Recall、confusion matrix、class support を保存する。

モデル間の比較では平均と標準偏差を報告し、同一 test 発話に対する paired bootstrap または McNemar 検定を検討する。

## 結果の表現例

適切な表現:

> 固定した emotion2vec 特徴を用いた日本語 SER において、VAD/VA 媒介 head は直接分類 head と比較して Macro-F1 が X ポイント変化した。

> 英語で事前学習した下流 head を日本語で追加学習した結果、日本語のみでゼロから学習した head と比較して Macro-F1 が X ポイント変化した。

避ける表現:

> 日本語データにより emotion2vec 自体の精度が上がった。

> 日本語学習により emotion2vec が英語を忘れた。

## 実験開始前の確認事項

- [ ] 英語・日本語の共通感情クラスを確定する。
- [ ] 各データセットの VAD/VA ラベル有無、値域、集約方法を確認する。
- [ ] speaker-independent な train/valid/test split を固定する。
- [ ] A～D の共通 head 設定と、encoder ごとの差分を明文化する。
- [ ] EN-scratch、JA-scratch、EN-zero-shot-JA、EN→JA の checkpoint を分けて保存する。
- [ ] 英語学習済み head からの転移を行う前に、JA-scratch を実行する。
- [ ] 公式 emotion2vec+ large head の結果を主比較ではなく参考値として分離する。
- [ ] 複数 seed、評価指標、結果保存形式を実験前に固定する。
- [ ] 使用 checkpoint、model ID、commit、特徴抽出条件を保存する。

## 既存構想メモとの関係

`archive/reports/2026-07-ser-comparative-eval-concept.md` には、英語で学習した VAD/VA head を日本語追加学習の初期値にする案が記録されている。今後は、それを唯一の日本語条件にはせず、本書の `JA-scratch` を主基準、`EN→JA` を転移学習の追加条件として扱う。

また、現在の主比較は emotion2vec と emotion2vec+ large をどちらも固定特徴抽出器として扱い、直接分類 head と VAD/VA 媒介分類 headを比較する 2 × 2 条件とする。既存構想メモを実装仕様として使用する際は、この変更を反映してから着手する。
