# emotion2vec 日本語fine-tuning/VAD媒介型分類 進捗報告メモ

作成日: 2026-07-04

## この文書の目的

この文書は、10分前後の進捗報告で話す内容をMarkdownとして整理した発表メモである。研究全体の目的、今回実装した内容、まだ評価できていないこと、次に行う実験を文章としてまとめ、後からPowerPointやMarpへ要点を抜き出せる形にする。

今回の発表では、研究全体の目標を「emotion2vecを日本語音声データでfine-tuningし、英語音声の感情認識精度を落とさずに、日本語音声の感情認識精度を上げること」と説明する。そのうえで、今回の進捗として、`emotion2vec -> VAD -> emotion` という説明可能な分類経路を実装したことを報告する。

ただし、現時点で日本語fine-tuningの性能改善や英語性能維持を実験結果として主張する段階ではない。依存入り環境でのテスト完走、実checkpointを使った推論、実データでの日本語・英語評価、VADのCCCや分類指標の算出は、今後の作業として残っている。

## 発表の要旨

本研究の大きな目的は、既存のemotion2vecを日本語音声感情認識へ適応させることである。emotion2vecは汎用的な音声感情表現を抽出できるが、日本語音声に対して十分な性能を出すには、日本語データを使ったfine-tuningが必要になる可能性がある。一方で、日本語に適応させる過程で、英語音声に対する感情認識性能が落ちると、汎用モデルとしての利点が弱くなる。したがって、本研究では「日本語性能を上げること」と「英語性能を落とさないこと」を同時に見る。

今回の進捗は、その研究全体の中で、評価と説明に使える分類経路を整えたことである。具体的には、事前抽出済みframe-level emotion2vec特徴からValence/Arousal/Dominanceを予測し、その予測VADだけをLinear層に入力して感情クラスを出すモデルを実装した。あわせて、学習CLI、推論JSON、寄与分解、データ読み込み、README、関連テストも追加した。

現時点では構文チェックと差分チェックは通っているが、PyTorch依存のテスト完走、日本語fine-tuning、実データでの日本語・英語評価は未完了である。次の到達点は、依存入り環境でテストを通し、実データで日本語SER指標、英語SER指標、VADのCCCを出すことである。

## 研究全体の目的

この研究を簡単に言うと、emotion2vecを日本語音声データでfine-tuningし、英語音声の感情認識精度を落とさずに、日本語音声の感情認識精度を上げる研究である。

目標は単に日本語データで性能を上げることだけではない。日本語に特化しすぎると、元のモデルが持っている英語音声への対応力が落ちる可能性がある。そのため、fine-tuning後のモデルは、日本語データで改善しているかだけでなく、英語データで性能低下が起きていないかも確認する必要がある。

整理すると、研究全体の評価軸は次の2つである。

1. 日本語音声感情認識の性能が上がるか。
2. 英語音声感情認識の性能を維持できるか。

今回実装したVAD媒介型分類は、この全体目標に対する一つの実験経路である。直接分類だけを見るのではなく、VADという連続感情値を経由することで、分類結果の根拠を確認しやすくする狙いがある。

## 今回の発表で言うこと

- 研究全体の目的は、日本語音声でemotion2vecをfine-tuningし、日本語SERを改善しつつ英語SERを維持することである。
- 今回の進捗は、fine-tuning結果の性能報告ではなく、VAD媒介型の分類・評価経路を実装したことである。
- 実装した経路では、emotion2vec特徴からVADを予測し、そのVADだけを使って感情分類する。
- 最終分類器は768次元のemotion2vec特徴を直接見ず、予測VADだけを見る。
- そのため、推論時に「どのVAD次元がどの感情クラスのlogitに寄与したか」を分解して出力できる。
- 現段階で性能値はまだ報告しない。実装進捗と、次に必要な日本語・英語評価の計画を報告する。

## 今回の発表で言わないこと

- 日本語fine-tuning済みとして扱わない。
- 日本語SERの向上を実験結果として扱わない。
- 英語SERの保持を実験結果として扱わない。
- VAD媒介型が直接分類より優れている、とは言わない。
- placeholder encoderやrandom modelの出力は研究結果として扱わない。
- 既存PPT由来の古いテスト成功件数や、現在のログと照合できない実験結果を使わない。

## 研究背景

音声感情認識では、発話から感情を推定する。典型的には、喜び、悲しみ、怒り、嫌悪のようなカテゴリ分類として扱われる。emotion2vecは、このような音声感情認識に使える汎用的な音声感情表現を抽出するモデルである。

しかし、汎用モデルをそのまま日本語音声に使ったときに、日本語特有の発話、韻律、データ分布に十分適応できるとは限らない。そのため、日本語音声感情データを使ってemotion2vecをfine-tuningし、日本語音声の感情認識性能を上げることが研究の中心になる。

一方で、fine-tuningには別の問題もある。日本語データに強く適応させると、元のモデルが持っていた英語音声への性能が落ちる可能性がある。これは、研究の目的から見ると望ましくない。目指すのは、日本語性能だけを上げることではなく、英語性能をできるだけ維持したまま日本語性能を上げることである。

この背景のもとで、分類結果を説明しやすくするために、Valence/Arousal/Dominanceも扱う。VADは、感情カテゴリを連続値として補助的に表すための軸である。Valenceは快・不快の方向、Arousalは覚醒度、Dominanceは支配性や主導感に対応する。同じ「怒り」に分類される音声でも、覚醒度の高低や支配性の強弱は異なる可能性がある。

## 問題設定

研究全体の問題設定は、fine-tuning前後のemotion2vecを比較することである。

```text
pretrained emotion2vec
  -> English SER / Japanese SER

Japanese fine-tuned emotion2vec
  -> English SER / Japanese SER
```

見るべき結果は、日本語SERが上がっているか、英語SERが落ちていないかである。片方だけでは不十分である。日本語だけ上がって英語が大きく落ちる場合は、汎用性を保った適応とは言いにくい。逆に、英語は維持できても日本語が改善しない場合は、日本語fine-tuningの効果が弱い。

今回実装したVAD媒介型分類は、この比較実験を行うための分類経路の一つである。直接分類の経路は次のように書ける。

```text
emotion2vec features -> emotion class
```

この経路は分類性能を出しやすい可能性がある。一方で、分類器が768次元の特徴を直接使うため、なぜその感情になったのかを説明しにくい。

今回実装した経路は次の形である。

```text
emotion2vec features -> predicted VAD -> emotion class
```

この経路では、分類器が使う情報をVADに制限する。情報を圧縮するため、直接分類より性能が下がる可能性はある。しかし、最終判断の入力がValence/Arousal/Dominanceに限定されるため、どの連続感情値が分類に効いたかを確認しやすい。

## 実装したモデル構造

実装した主な流れは次のとおりである。

```text
WAV or precomputed frame features
  -> emotion2vec frame-level features
  -> masked mean pooling
  -> FNN
  -> predicted Valence/Arousal/Dominance
  -> Linear(target_dim -> num_classes)
  -> emotion logits
  -> softmax probabilities
```

学習時には、事前抽出済みのframe-level emotion2vec特徴を主な入力として扱う。各発話のframe数は異なるため、batch化するときにpaddingし、padding maskを使って有効frameだけを平均poolingする。その後、FNNでVADを予測する。

VAD予測後の分類器はLinear層である。たとえば3次元VAD、4クラス分類の場合は `Linear(3 -> 4)` になる。ここで重要なのは、この分類器がemotion2vec特徴を直接見ない点である。分類器の入力は予測されたVADだけである。

主な実装箇所は以下である。

- [vad_downstream/model.py](../../vad_downstream/model.py): `VADRegressionHead`, `VADClassificationHead`, `VADMediatedEmotionClassifier`, `Emotion2vecVADMediatedClassifier`
- [vad_downstream/data.py](../../vad_downstream/data.py): `.npy/.lengths/.vad/.emo` の読み込み、VADと感情ラベルの整合性検証
- [vad_downstream/emotion_training.py](../../vad_downstream/emotion_training.py): VAD回帰と感情分類の複合loss、評価指標、checkpoint保存
- [vad_downstream/train_vad_emotion.py](../../vad_downstream/train_vad_emotion.py): 事前抽出特徴からVAD媒介分類器を学習するCLI
- [vad_downstream/infer_vad_emotion.py](../../vad_downstream/infer_vad_emotion.py): WAVからVADと感情分類を出し、JSONに寄与分解を保存するCLI

図としては、[src/vad_mediated_emotion_structure.svg](../../src/vad_mediated_emotion_structure.svg) が現在の実装方針に対応している。発表では縦長の図をそのまま使うより、中央の流れだけを横方向に簡略化した方が見やすい。

## 学習の考え方

VAD媒介型分類では、VAD回帰のlossと感情分類のlossを同時に使う。実装した目的関数は次の形である。

```text
loss = lambda_vad * ccc_loss(predicted_vad, target_vad)
     + lambda_emo * cross_entropy(logits, target_emotion)
```

VAD側はCCC lossを使う。CCCは予測と正解の相関だけでなく、平均や分散のずれも見るため、連続感情値の評価に向いている。感情分類側はCrossEntropyLossを使う。

ここでVAD lossを残すことが重要である。もし分類lossだけで学習すると、中間の2次元または3次元ベクトルは分類に便利な表現にはなるかもしれないが、それがValence/Arousal/Dominanceとして解釈できる保証は弱くなる。今回の方針では、中間表現をVADとして扱いたいため、VAD回帰の教師信号を併用する。

このVAD媒介型の学習は、研究全体で予定しているemotion2vec本体のfine-tuningとは区別する。今回実装した範囲は、主に事前抽出済み特徴を使ったdownstream側の分類経路であり、emotion2vec本体のfine-tuning結果を報告するものではない。

## 推論時に出す情報

推論CLIでは、予測クラスだけでなく、分類に使われたVAD値とLinear層の寄与分解をJSONに出力する。

出力する主な情報は以下である。

- `prediction`: 予測された感情クラス
- `probabilities`: 各クラスのsoftmax確率
- `vad`: 分類器の入力になった予測VAD
- `logits`: 各クラスのlogit
- `linear_weights`: Linear層のbiasとVAD各次元の重み
- `contributions`: `bias` と `weight * VAD` に分解したlogit寄与
- `contrast_to_runner_up`: 1位クラスと2位クラスの差分寄与

各クラスのlogitは次の式で表せる。

```text
logit_c = bias_c
        + weight_c,valence * predicted_valence
        + weight_c,arousal * predicted_arousal
        + weight_c,dominance * predicted_dominance
```

この式の各項をJSONで出すことで、たとえば「怒り」のlogitが高いときに、それがArousalの寄与によるものなのか、Valenceの寄与によるものなのかを確認できる。ただし、これは最終Linear層の中での寄与分解であり、音響特徴全体に対する完全な説明ではない。この点は発表で明確に線引きする。

## データ契約

VAD媒介型分類の実験データは、同じprefixを持つ複数ファイルとして扱う。

| ファイル | 役割 |
|---|---|
| `<prefix>.npy` | 全発話のframe-level emotion2vec特徴を時間方向に結合した配列。想定特徴次元は768。 |
| `<prefix>.lengths` | 各発話のframe数。`.npy` から発話境界を復元するために使う。 |
| `<prefix>.vad` | 各発話のVADラベル。Valence/Arousalは必須、Dominanceは任意。 |
| `<prefix>.emo` | 各発話の感情カテゴリラベル。VAD媒介分類では必須。 |

`.vad` と `.emo` は行順で対応させる。両方に `utterance_id` を持たせ、行数とIDが一致しない場合はエラーにする。これにより、特徴、VADラベル、感情ラベルのずれを早い段階で検出できる。

現在の標準クラス順は以下である。

```text
hap, sad, ang, dis
```

日本語表示は以下である。

```text
喜び, 悲しみ, 怒り, 嫌悪
```

`exc -> hap` は前処理で扱う想定だが、`neu -> dis` の置換はしない。`dis` を使う以上、実データで `dis` が十分あるか、fold内で極端に偏っていないかを確認する必要がある。

## 評価設計

研究全体の評価では、日本語性能と英語性能を両方見る必要がある。

まず、日本語音声感情認識では、fine-tuning前のemotion2vecと、日本語データでfine-tuningしたemotion2vecを比較する。ここで日本語SER指標が改善しているかを見る。

次に、英語音声感情認識でも同じように、fine-tuning前後の性能を比較する。ここでは、英語SER指標が大きく落ちていないかを見る。英語性能を保ったまま日本語性能が上がっていれば、研究目的に近づいたと言える。

感情分類の指標としては、WA、UA、weighted F1、confusion matrixを見る。WAは全体の正解率であり、クラス分布が偏っている場合には多数派クラスの影響を受けやすい。UAはクラスごとのrecallを平均するため、少数クラスの扱いを見るのに有効である。weighted F1はクラスごとのF1をsupportで重み付けして見る。confusion matrixは、どのクラスがどのクラスに誤分類されているかを確認するために使う。

VAD媒介型分類については、VAD回帰と感情分類を分けて見る。VAD回帰では、Valence、Arousal、DominanceそれぞれのCCCと、平均CCCを見る。Dominanceを使わない2次元設定では、ValenceとArousalのCCCを見る。

今回の評価で特に注意すべき点は、`dis` の分布である。`dis` の件数が少ない、またはfoldによって存在しない場合、4クラス分類としての評価値は不安定になる。その場合は、評価設計やクラス設計を見直す必要がある。

## 現在の検証状況

現時点で確認済みなのは、構文チェックと差分チェックである。

- `py_compile`: exit 0
- `git diff --check`: exit 0、CRLF warningのみ

一方で、以下は未完了である。

- PyTorch依存の単体テスト完走
- 実emotion2vec checkpointを使ったWAV推論
- emotion2vec本体の日本語fine-tuning
- 日本語データでの感情認識評価
- 英語データでの性能維持評価
- 実IEMOCAP/VADデータでの学習と評価
- `.emo` の `hap/sad/ang/dis` 分布確認
- `dis` のfold内分布確認

未完了の主な理由は、現在のWindows側Python環境で `numpy` や `torch` が不足しており、依存入り環境でのテストがまだ完走していないためである。また、WSLディストリビューションが未導入のため、bashスクリプトの構文チェックも未実行相当である。

発表では、この点を明確に分ける。つまり「VAD媒介型分類経路の実装は進んだ」と「日本語fine-tuningの性能結果はまだ出していない」を混同しない。

## 次にやること

次の作業は、実装を研究結果に接続するための検証である。優先順は以下の通りである。

1. 依存入りPython環境で単体テストを完走する。
2. `.emo` の `hap/sad/ang/dis` 件数を確認する。
3. 特に `dis` がfoldごとに評価可能な数だけ存在するか確認する。
4. 実 `.vad/.emo` prefixでVAD媒介型分類の学習を回す。
5. VADのCCCと分類のWA/UA/weighted F1/confusion matrixを出す。
6. 日本語fine-tuningの実験条件を固める。
7. fine-tuning前後で、日本語SERと英語SERを比較する。
8. 直接分類ベースラインとVAD媒介型分類の比較方針を決める。

この順番にする理由は、まず実装の動作保証を取り、その後でデータ分布を確認し、最後に性能評価へ進むためである。分布確認を飛ばして学習結果だけを見ると、少数クラスの扱いを誤って解釈する可能性がある。また、研究全体の主張には英語性能維持の確認が必要なので、日本語データだけで評価を終えない。

## 10分発表用の話す流れ

### 1. 現在地

本日は、emotion2vecを用いた音声感情認識の進捗として、日本語fine-tuningを目指す研究全体の位置づけと、今回実装したVAD媒介型感情分類経路について報告する。研究全体の目標は、日本語音声でemotion2vecをfine-tuningし、日本語音声の感情認識精度を上げつつ、英語音声の感情認識精度を落とさないことである。

### 2. 背景

emotion2vecは汎用的な音声感情表現を抽出できるモデルである。ただし、日本語音声の感情認識にそのまま使ったときに十分とは限らない。そのため、日本語音声データでfine-tuningして日本語性能を上げたい。一方で、日本語に適応することで英語性能が落ちると、汎用モデルとしての利点が弱くなる。そこで、日本語性能の改善と英語性能の維持を両方見る。

### 3. 今回の焦点

今回の報告は、fine-tuning結果の性能報告ではない。今回実装したのは、emotion2vec特徴からVADを予測し、そのVADだけを使って感情分類する経路である。この経路は、今後fine-tuning前後のモデルを比較するときに、分類結果を説明しやすくするための実験基盤になる。

### 4. 実装構造

実装した構造は、frame-level特徴をmasked mean poolingし、FNNでVADを予測し、そのVADをLinear層に入れて感情logitを出す形である。学習時にはCCC lossとCrossEntropyLossを足し合わせる。VAD lossを残すことで、中間表現が分類に便利なだけの値にならず、VADとして解釈できる状態を保つ。

### 5. 今回の進捗

今回追加したのは、データ読み込み、VADと感情分類の複合loss、学習CLI、推論CLI、推論JSON、寄与分解、README、関連テストである。推論JSONでは、予測クラスや確率だけでなく、VAD値、logit、Linear層の重み、bias、VAD次元ごとの寄与、1位と2位クラスの差分寄与を出力できる。

### 6. 評価設計

研究全体では、fine-tuning前後で日本語SERと英語SERを比較する。日本語側では性能が上がるかを見て、英語側では性能が落ちていないかを見る。VAD媒介型分類では、VAD側はCCC、分類側はWA、UA、weighted F1、confusion matrixを見る。

### 7. 未完了

現時点では、構文チェックと差分チェックは通っている。一方で、PyTorch依存の単体テスト、実checkpoint推論、日本語fine-tuning、実データでの日本語・英語評価は未完了である。したがって、発表では性能値を出さず、実装済み範囲と次の評価計画を分けて報告する。

### 8. 次の到達点

次は、依存入り環境でテストを完走し、`.emo` の分布を確認し、実データでVAD媒介型分類を学習・評価する。その後、日本語fine-tuningを行い、fine-tuning前後で日本語SERと英語SERを比較する。ここまで進めることで、日本語適応と英語性能維持の両方を議論できるようになる。

### 9. 締め

まとめると、研究全体の目的は、日本語音声でemotion2vecをfine-tuningし、日本語SERを改善しつつ英語SERを維持することである。今回の成果は、その最終性能ではなく、VADを経由した分類と寄与分解を行うための実装基盤である。次の到達点は、実データでVAD媒介型分類を評価し、その後fine-tuning前後の日本語・英語性能を数値化することである。

## スライド化するときの対応案

Markdown本文をスライドに落とすなら、以下の9枚程度に分けると10分に収まりやすい。

| 枚 | 見出し | スライドで見せる要点 | 図・素材 |
|---:|---|---|---|
| 1 | 現在地 | 研究全体は日本語fine-tuningによる日本語SER改善と英語SER維持。今回の進捗はVAD媒介型分類経路の実装。 | 全体フロー |
| 2 | 背景 | emotion2vecを日本語音声へ適応したいが、英語性能低下は避けたい。 | 日本語/英語の2軸図 |
| 3 | 研究方針 | fine-tuning前後で日本語SERと英語SERを比較する。 | before/after比較表 |
| 4 | VAD媒介型分類 | `features -> pooling -> FNN -> VAD -> Linear -> emotion` | [src/vad_mediated_emotion_structure.svg](../../src/vad_mediated_emotion_structure.svg) の簡略版 |
| 5 | 実装した範囲 | データ、loss、学習CLI、推論JSON、寄与分解、README、テスト。 | チェックリスト |
| 6 | 評価設計 | 日本語SER、英語SER、VAD CCC、WA/UA/F1/confusion matrix。 | 評価指標表 |
| 7 | 検証状況 | 構文チェックと差分チェックは通過。fine-tuningと実データ評価は未完了。 | 完了/未完了の2列表 |
| 8 | 次の作業 | テスト完走、分布確認、VAD媒介型評価、日本語fine-tuning、英語維持評価。 | ロードマップ |
| 9 | まとめ | 研究全体の目的と、今回できた実装基盤、次に出すべき数値を再確認。 | 冒頭フローの再掲 |

スライド本文はこの文書の段落をそのまま載せず、1枚につき1文メッセージと3から5個の短い箇条書きに圧縮する。詳細な説明は発表者ノートに回す。

## 想定質問と答え方

### Q1. この研究を一言で言うと何か

emotion2vecを日本語音声データでfine-tuningし、英語音声の感情認識精度を落とさずに、日本語音声の感情認識精度を上げる研究である。

### Q2. なぜ英語性能も見る必要があるのか

日本語だけに適応して英語性能が落ちると、元のemotion2vecが持つ汎用性を損なう可能性があるからである。本研究では、日本語への適応と英語性能維持を同時に評価する必要がある。

### Q3. VADを媒介にする利点は何か

分類器が直接768次元特徴を使わず、予測VADだけを使うため、最終分類の入力になった中間値を確認できる。さらにLinear層なので、各クラスのlogitに対するValence/Arousal/Dominanceの寄与を分解できる。

### Q4. 直接分類より性能が下がる可能性はあるか

ある。VADに情報を圧縮するため、カテゴリ分類に有用な情報が落ちる可能性がある。今回の方針は、性能だけを最大化する設計ではなく、説明しやすい経路を作る設計である。実データでは直接分類ベースラインとの比較が必要になる。

### Q5. `dis` データ分布は十分か

現時点では未確認である。`hap/sad/ang/dis` の件数、特に `dis` のfold内分布を確認してから評価設計を確定する。分布が薄い場合は、評価の信頼性やクラス設計を見直す必要がある。

### Q6. CCCをまだ報告しない理由は何か

依存入り環境でのテスト完走と、実 `.vad/.emo` prefixを使った学習・評価が未完了だからである。現段階で出せるのは実装と評価設計であり、CCCの数値は実データ実験後に報告する。

### Q7. 推論JSONの寄与分解は何を意味するか

最終Linear層の式を、biasと各VAD次元の項に分けたものである。各クラスについて `bias + weight * VAD` の和がlogitに一致するため、予測VADが分類logitにどう効いたかを確認できる。ただし、音声特徴全体に対する完全な説明ではなく、VADからlogitへの線形部分の説明である。

## 発表前チェック

- 10分で読み切れるか確認する。
- 研究全体の目的を「日本語SER改善」と「英語SER維持」の2軸で説明できているか確認する。
- 「VAD媒介型分類の実装」と「emotion2vec本体のfine-tuning」を混同していないか確認する。
- 「実装したこと」と「性能まで確認したこと」が混ざっていないか確認する。
- 既存PPT由来の古い実験結果を使っていないか確認する。
- `dis` の分布確認前に4クラス分類性能の見込みを断定していないか確認する。
- 締めの主張を「研究全体は日本語fine-tuningによる日本語SER改善と英語SER維持。今回の到達点はVAD媒介型分類の実装基盤」にそろえる。
