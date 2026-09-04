# References Log

論文・根拠参照の記録ファイル。`paper-cite` スキルにより自動追記される。

- **形式**: APA 準拠
- **エビデンス強度**: `[高]` 査読済み / `[中]` プレプリント / `[低]` 未査読
- **重複**: 同一 DOI は `(再引用)` として記録

---

## 2026-08-12 — 分類モデルのクラス集合とCSVからの自動構築

**質問/文脈**: 一般的な分類モデルでは感情クラスが固定されるのか、学習CSVからクラス集合を構築する設計が一般的かを確認した。

| 論文・資料 | 著者・提供元 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation | Ma et al. | 2024 | [高] | https://doi.org/10.18653/v1/2024.findings-acl.931 | emotion2vecは表現モデルであり、下流SERでは分類層を学習して評価する |
| CrossEntropyLoss documentation | PyTorch | 2026 | [低: 公式文書] | https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html | 通常の多クラス分類では出力logit数Cと対象クラスID範囲が対応する |
| Glossary: classes_ | scikit-learn | 2026 | [低: 公式文書] | https://scikit-learn.org/stable/glossary.html | 分類器が認識するクラス一覧と確率出力列の対応を保存する慣行 |

## 2026-08-20 — IEMOCAP・HCUDB1・DS-001感情ラベル対応表

**質問/文脈**: 3つの音声感情データセットについて、意味の差を残した感情ラベル対応表と共通クラス候補を作成した。

| 論文・資料 | 著者・提供元 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| IEMOCAP: Interactive emotional dyadic motion capture database | Busso et al. | 2008 | [高・Web書誌確認] | https://doi.org/10.1007/s10579-008-9076-6 | IEMOCAPの正式名称、データセット構成、`hap`と`exc`を統合する根拠 |
| 広島市立大学 感情音声コーパス (HCUDB) | 国立情報学研究所 | 2026 | [低: 公式データ配布文書] | https://doi.org/10.32130/src.HCUDB | HCUDB1の14名、11感情、3テイク、4,620発話という構成 |
| 感情を込めた発話データセット（DS-001） | Qlean Dataset | n.d. | [低: 公式製品仕様] | https://qleandataset.visual-bank.co.jp/lineup/ds-001 | DS-001の9感情、2強度、100名、6,800件という仕様 |

## 2026-08-21 — IEMOCAPの学習・検証・テスト分割

**質問/文脈**: IEMOCAPを単一固定splitより厳密に分割する場合の実験設計を検討した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation（再引用） | Ma et al. | 2024 | [高・Web索引確認] | https://doi.org/10.18653/v1/2024.findings-acl.931 | IEMOCAPでleave-one-session-out 5-foldを用い、各foldの学習側からvalidationを分ける評価プロトコル |
| Designing and Evaluating Speech Emotion Recognition Systems: A reality check case study with IEMOCAP | Antoniou et al. | 2023 | [中・Web要約確認] | https://arxiv.org/abs/2304.00860 | IEMOCAP研究間の分割・仮定・指標の不統一が再現性と比較可能性を損なうため、評価条件の明示が重要 |

## 2026-08-22 — IEMOCAP・HCUDB1間の分類出力設計

**質問/文脈**: 感情ラベル集合が異なる英語IEMOCAPと日本語HCUDB1を順次学習・評価する場合に、decoderの出力クラスをどう決めるべきかを検討した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| Accuracy of Automatic Cross-Corpus Emotion Labeling for Conversational Speech Corpus Commonization | Mori et al. | 2016 | [高・Web本文確認] | https://aclanthology.org/L16-1634/ | 感情コーパス間にはラベル体系の非互換性があり、カテゴリ感情のコーパス間対応は慎重に扱う必要がある |
| Cross-lingual and Multilingual Speech Emotion Recognition on English and French | Neumann & Vu | 2018 | [高・Web本文確認] | https://doi.org/10.1109/ICASSP.2018.8462162 | 言語間SERでは注釈方法や収録条件などコーパス差が結果解釈を複雑にする |
| Speech Emotion Recognition with Multi-Task Learning | Cai et al. | 2021 | [高・Web要約確認] | https://doi.org/10.21437/Interspeech.2021-1852 | IEMOCAPでhappy・angry・sad・neutralの4クラスを扱う評価設定が査読研究で使用されている |
| Handling Ambiguity in Emotion: From Out-of-Domain Detection to Distribution Estimation | Wu et al. | 2024 | [高・Web本文確認] | https://doi.org/10.18653/v1/2024.acl-long.114 | 合意不能な感情を追加クラスとして学習すると、他の感情クラスの分類性能を損なう場合がある |

## 2026-08-22 — HCUDB1の嫌悪系感情を測る副実験の位置づけ

**質問/文脈**: HCUDB1の「嫌い」を嫌悪系感情として測定可能にしつつ、IEMOCAP→HCUDB1の日英比較という主研究を損なわない実験設計を検討した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| EmoNet: A Transfer Learning Framework for Multi-Corpus Speech Emotion Recognition | Gerczuk et al. | 2021 | [中・Web本文確認] | https://arxiv.org/abs/2103.08310 | 複数SERコーパスで共有モデルとコーパス固有分類器を分ける設計例がある |
| 1st Place Solution to Odyssey Emotion Recognition Challenge Task1: Tackling Class Imbalance Problem | Chen et al. | 2024 | [高・Web本文確認] | https://doi.org/10.21437/odyssey.2024-37 | 少数クラスは学習・識別が難しく、クラス重み付けにも過学習・学習不足のトレードオフがある |

## 2026-08-22 — MSP-Podcastを嫌悪認識へ利用する可能性

**質問/文脈**: MSP-Podcastに嫌悪ラベルが含まれるか、また嫌悪認識の追加データセットとして十分な量があるかを確認した。

| 論文・資料 | 著者・提供元 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| Odyssey 2024 - Speech Emotion Recognition Challenge: Dataset, Baseline Framework, and Results | Goncalves et al. | 2024 | [高・Web本文確認] | https://doi.org/10.21437/odyssey.2024-35 | MSP-Podcastを用いた8感情分類タスクに`disgust`が明示的に含まれる |
| The ViVoLab System for the Odyssey Emotion Recognition Challenge 2024 Evaluation | Pastor et al. | 2024 | [高・Web本文確認] | https://doi.org/10.21437/odyssey.2024-39 | Challenge用MSP-Podcast v1.11では`disgust`が1,912件（全体の2.17%）で、絶対数はあるが少数クラスである |
| MSP-Podcast Corpus | Multimodal Signal Processing Laboratory, The University of Texas at Dallas | 2026 | [低: 公式コーパス仕様] | https://lab-msp.com/MSP/MSP-Podcast.html | v2.0は264,705発話・409時間で、カテゴリラベルに`disgust`を含む |

## 2026-08-22 — MSP-Podcast学習・IEMOCAP外部テスト設計

**質問/文脈**: 主学習データをMSP-Podcastとし、IEMOCAPを学習に使わない外部テストコーパスとして利用する設計と、emotion2vec特徴量の現在の処理方法を整理した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| Cross-Corpus Speech Emotion Recognition Using Semi-Supervised Transfer Non-Negative Matrix Factorization with Adaptation Regularization | Luo & Han | 2019 | [高・Web本文確認] | https://doi.org/10.21437/Interspeech.2019-2041 | 学習コーパスとテストコーパスが異なるSERはcross-corpus評価であり、両者の分布不一致が主要課題になる |

## 2026-08-22 — MSP-Podcast全感情と共通クラスの再検討

**質問/文脈**: MSP-Podcastのカテゴリ感情を網羅し、MSP-Podcast・IEMOCAP・HCUDB間で十分な評価件数を持つ共通感情をすべて比較対象にする方針を検討した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| From Pretraining to Performance: Benchmarking Self-Supervised Speech Models for Interspeech-25 SER Challenge | Uniyal & Abrol | 2025 | [高・Web本文確認] | https://doi.org/10.21437/Interspeech.2025-1283 | MSP-Podcast v1.12の8感情に対するtrain/dev別件数と、Other・No Agreementを公式8クラス評価から除外する設定 |
| The Interspeech 2025 Challenge on Speech Emotion Recognition in Naturalistic Conditions | Naini et al. | 2025 | [高・Web書誌確認] | https://doi.org/10.21437/Interspeech.2025-1972 | MSP-Podcastを用いた自然発話SER Challengeの公式な8カテゴリ評価設定 |

## 2026-08-22 — MSP-Podcast Release 1.10への実験版固定

**質問/文脈**: 利用可能なMSP-PodcastがRelease 1.10（2022-05-03）であることを受け、規模・カテゴリラベル・ラベル集約方法を版指定で確認した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| The Importance of Calibration: Rethinking Confidence and Performance of Speech Multi-label Emotion Classifiers | Chou et al. | 2023 | [高・Web本文確認] | https://doi.org/10.21437/Interspeech.2023-1113 | Release 1.10は約166時間で、primary annotationにanger・sadness・happiness・surprise・fear・disgust・contempt・neutralを含み、複数評価者の知覚ラベルを持つ |
| Minority Views Matter: Evaluating Speech Emotion Classifiers With Human Subjective Annotations by an All-Inclusive Aggregation Rule | Chou et al. | 2025 | [高・Web本文確認] | https://doi.org/10.1109/TAFFC.2024.3411290 | 多数決・相対多数決・全包含型では、評価者ラベルの集約方法と除外される曖昧発話の範囲が異なるため、実験前に集約規則を固定する必要がある |

## 2026-08-22 — emotion2vec事前学習とMSP-Podcast評価の重複リスク

**質問/文脈**: emotion2vecの自己教師あり事前学習にMSP-Podcast v1.8が含まれることが、v1.10を下流学習・評価へ使う際に何を意味するかを整理した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| The MSP-Podcast Corpus | Busso et al. | 2025 | [中・Web本文確認: プレプリント] | https://arxiv.org/abs/2509.09791 | MSP-Podcastは複数releaseを通じて発話・注釈を拡張してきたコーパスであり、版間の音声ID重複は実ファイルまたはmanifestで確認する必要がある |

## 2026-08-24 — encoder出力次元とHCUDB・MSP-Podcast分割の妥当性

**質問/文脈**: emotion2vec Baseの768次元表現と4クラスdecoder出力を区別し、HCUDB固定10/2/2話者splitおよびMSP-Podcast公式partitionが評価条件として妥当かを確認した。

emotion2vec論文、HCUDB公式配布情報、MSP-Podcast公式コーパス仕様は既存記録を再引用したため、重複行は追加しない。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| Acoustic Features and Neural Representations for Categorical Emotion Recognition from Speech | Keesing et al. | 2021 | [高・Web本文確認] | https://doi.org/10.21437/Interspeech.2021-2217 | 未知話者への一般化を評価するため、SERで話者独立cross-validationを用いる研究設計例がある |

## 2026-08-24 — MSP-Podcast Release 1.10のdisgust件数

**質問/文脈**: ローカルのRelease 1.10 `labels_consensus.csv`から、`disgust`を全metadataで2,946件、現行4クラス実験条件で2,434件（Train 1,237／Development 465／Test1 732）とsplit別に再集計した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| The Importance of Calibration: Rethinking Confidence and Performance of Speech Multi-label Emotion Classifiers（再引用） | Chou et al. | 2023 | [高・Web本文確認] | https://doi.org/10.21437/Interspeech.2023-1113 | Release 1.10のTrain／Development／Test構成と、primary emotionにdisgustが含まれることの確認。正確な件数はローカルmetadataを一次根拠として集計 |

## 2026-09-03 — キャッシュを利用する感情分類モデルの改善候補

**質問/文脈**: 「モデル改善案は何か何？」への回答。保存済みのseed 42・43・44の結果と現行実装を確認し、HCUDB追加学習の学習率、Dropout、クラス重み付き損失、Attention poolingを比較候補として整理した。今回提示した設定値は未検証の実験候補であり、精度向上を確認した値ではない。新たな学習は実行していない。

PyTorch公式CrossEntropyLoss資料とChen et al. (2024), *1st Place Solution to Odyssey Emotion Recognition Challenge Task 1: Tackling Class Imbalance Problem* は既存記録を再引用し、重複行を追加しない。前者は学習クラスごとの損失重みを設定できること、後者はクラス重み付けにも多数派・少数派間の性能のトレードオフがあることの根拠として参照した。

| 論文 | 著者 | 年 | エビデンス強度 | DOI/URL | 使用した主張 |
|------|------|----|--------------|---------|-------------|
| Dropout: A Simple Way to Prevent Neural Networks from Overfitting | Srivastava, Hinton, Krizhevsky, Sutskever, & Salakhutdinov | 2014 | [高・査読誌・出版社の公開要旨確認] | https://jmlr.org/papers/v15/srivastava14a.html | Dropoutは過学習を抑えるための手法。現行モデルのdropout=0.0から0.1・0.2を比較する提案の根拠であり、この研究条件での改善幅は不明 |
| An Attention Pooling Based Representation Learning Method for Speech Emotion Recognition | Li, Song, McLoughlin, Guo, & Dai | 2018 | [査読会議・出版社の公開要旨確認] | https://doi.org/10.21437/Interspeech.2018-1242 | 音声感情認識でAttention poolingを利用する研究例がある。既存emotion2vecフレーム特徴に適用する案は別の実験であり、論文の構成・結果を再現すると主張しない |

**書誌情報（APA）**

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html
- Li, P., Song, Y., McLoughlin, I., Guo, W., & Dai, L. (2018). An attention pooling based representation learning method for speech emotion recognition. *Proceedings of Interspeech 2018*, 3087–3091. https://doi.org/10.21437/Interspeech.2018-1242

## 2026-09-03 — MSP単体の性能を先に改善する方針

**質問/文脈**: ユーザーの「そもそも，mspの成績が低いなと考えている」を受け、HCUDB適応後のMSP性能維持よりも、MSP単体の分類性能の改善を優先するよう提案を修正した。

ローカルの `runs/ser_decoder_timing_check_20260903/formal/initial-seed-42/seed-42/before/msp_podcast/class_metrics.csv` と混同行列を確認。seed 42のMSP testはhappyが3,723/5,692件（65.4076%）、モデルの正解数は3,891件（68.3591%）で、全件happy予測との差は2.9515ポイント。クラス別再現率はanger 45.3441%、happy 84.6092%、sadness 44.0313%、disgust 25.1046%。これはクラスごとの成績差を確認する記述であり、原因が学習データの件数差だけであると断定するものではない。MSP validationによる設定選択を提案し、新たな学習・test評価は実行していない。

**再引用**: Srivastava et al. (2014) のDropout論文とChen et al. (2024) のOdysseyクラス不均衡対策論文について、出版社の公開要旨を再確認した。前者は過学習対策の根拠、後者はクラス重み付けに多数派・少数派間の性能トレードオフがあり得ることの根拠として用いる。既存の書誌情報と重複するため文献行は追加しない。今回の4クラス実験の成績をOdysseyの異なる実験条件の数値と比較していない。

## 2026-09-03 — クラス重み付き損失の比較方法と実装

**質問/文脈**: MSP単体で重みなし/ありを比較する実行経路を追加。balanced重みをtrainのincluded発話から `N / (4 * n_class)` として計算し、PyTorchの標準weighted-mean cross entropyへ渡す。validation lossと評価指標は従来の重みなし計算を維持する。公式資料の本文で式とAPI仕様を確認した。効果検証の実データ学習はユーザーが行う。

| 資料 | 著者 | 年 | 根拠 | DOI/URL | 使用した主張 |
|------|------|----|------|---------|-------------|
| compute_class_weight | scikit-learn developers | n.d. | [公式API資料・Web本文確認] | https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html | balanced方式はサンプル総数をクラス数と各クラス件数の積で割る。今回scikit-learn依存は追加せず同じ式を直接計算 |

PyTorch CrossEntropyLoss公式資料（https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html）は再引用。クラス重み指定と、mean reductionがバッチ内の正解クラス重みの和で正規化される仕様を確認した。

## 2026-09-03 — train・validation・testの成績を比較する役割と現状の不足

**質問/文脈**: train・validation・testのscoreを積極的に比較していない理由を説明した。現行コードではtrainは最適化中のバッチloss平均のみ、validationは各epochのloss・WA・UAR・macro F1・クラス別指標を保存する。従来の転移実験ではtestも評価済みだが、新しい重み比較ではtest評価を保留している。trainの分類指標が未記録なのは実装上の不足であり、testを設定選択に使わない方針とは別の理由として説明した。重みありtrain lossと重みなしvalidation lossの絶対値差から過学習の程度を判断できないこともコードに基づいて確認した。

| 資料 | 著者 | 年 | 根拠 | DOI/URL | 使用した主張 |
|------|------|----|------|---------|-------------|
| Cross-validation: evaluating estimator performance | scikit-learn developers | n.d. | [公式資料・Web本文確認] | https://scikit-learn.org/stable/modules/cross_validation.html | test成績を参照しながら設定を調整すると評価が選択の影響を受けるため、設定選択にvalidationを使い、その後にtestで評価する |
| Validation curves: plotting scores to evaluate models | scikit-learn developers | n.d. | [公式資料・Web本文確認] | https://scikit-learn.org/stable/modules/learning_curve.html | trainとvalidationの同じ指標の比較は過学習・学習不足の診断に役立つ。validationを使って選択したモデルのvalidation成績だけでは最終的な性能評価にならない |

補足: trainの確定したモデルでのscoreを得るには、保存済みcheckpointに対して評価モードで推論できる。これは追加学習を必要としないが、既に上書きされた各epochのモデルのtrain scoreを復元することはできない。学習中の予測から集計する指標と、同一checkpointを固定して集計する指標は区別する。今回この説明のために学習や実データ評価は実行していない。

## 2026-09-03 — クラス重み付け後にUARと正解率が逆方向へ変化した理由

**質問/文脈**: 「happyを出すと当たりやすかったため正解率が下がり、見せかけの成績が現実的になった」という解釈を、保存済みの重み比較結果と指標の定義から検討した。`runs/msp_class_weight_comparison/seeds-42/comparison_summary.json` と `seeds-43-44/comparison_summary.json` にある各seedのbest validation指標・混同行列を読み、クラス別指標を3seedで平均した。新たな学習・推論は実行していない。

validationの件数はanger 1,044、happy 1,808、sadness 296、disgust 452、計3,600。happyの構成比は50.22%であり、以前確認したtestの65.41%とは異なる。重みあり−なしの再現率差は順に+7.4393、−15.0627、+10.0225、+10.1032ポイント。4クラスを等しく平均するとUAR差+3.1256ポイント、構成比で重み付けすると正解率差−3.3148ポイントとなる。happyの正解率差への寄与は−7.5648ポイント、他3クラスの合計は+4.2500ポイント。F1差は順に+3.8575、−4.9993、+0.3114、+0.8327ポイントで、macro F1差は+0.000587ポイントとほぼ相殺される。

happyへの予測割合は平均47.65%から32.94%へ減少。sadnessとdisgustでは再現率が向上する一方でprecisionが低下している。評価集合と指標の計算方法は共通であり、正解率が水増しから修正されたのではなく、学習する損失を変えた結果、クラスごとの成績にトレードオフが生じたと解釈する。このvalidation結果だけから総合的な性能向上やtest性能を断定しない。

| 資料 | 著者 | 年 | 根拠 | DOI/URL | 使用した主張 |
|------|------|----|------|---------|-------------|
| balanced_accuracy_score | scikit-learn developers | n.d. | [公式API資料・Web本文確認] | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html | 各クラスのrecallの単純平均。今回の4クラスUARと同じ集計 |
| f1_score | scikit-learn developers | n.d. | [公式API資料・Web本文確認] | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html | F1はprecisionとrecallの調和平均、macroはクラス別F1の単純平均 |
| accuracy_score | scikit-learn developers | n.d. | [公式API資料・Web本文確認] | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html | 正しく分類されたサンプルの割合。単一ラベル多クラスでは各クラスrecallを正解ラベルの構成比で重み付けした値に等しい |

## 2026-09-03 — 基本の学習部分での指標表示とChatGPTでの研究相談

**質問/文脈**: ユーザーは末尾に評価節を追加する案を取りやめ、基本の学習部分で何を表示すべきかの相談を優先した。毎epochのtrain・validationを同じモデル状態・評価方法で比較し、設定とbest checkpointを確定した段階でtestを評価する構成を提案する。trainとvalidationがともに低い場合も、epoch不足と直ちに同一視せず、表現・モデル・最適化などの不足を区別する。現行実装でtrain全体を固定モデルで毎epoch評価するには追加推論が必要で、所要時間が増える。比較用lossは全splitで重みなしにそろえ、重み付き最適化lossとは分ける。今回コード・Notebookは変更せず、実データ学習・評価も実行していない。

**再引用**: scikit-learnの `learning_curve.html` と `cross_validation.html` を公式本文で再確認。trainとvalidationの比較による過学習・学習不足の診断、testを設定選択に使わない根拠に用いた。既存の書誌行との重複は省略。

**OpenAI公式資料・本文確認**: OpenAI. (n.d.). *Projects and chats*. https://learn.chatgpt.com/docs/projects 。ChatGPTのプロジェクトが会話・添付ファイル・指示・接続された情報をまとめること、通常のChatGPTプロジェクトとローカルフォルダに接続するプロジェクトの違いを確認した。相談用ChatGPTプロジェクトへ研究概要・条件・結果を渡し、決定した仕様をCodexへ戻す運用を提案する。ローカルフォルダやこの会話全体が自動共有されるとは断定せず、利用者の環境での接続有無も未確認。プロジェクト作成・ファイル送信は行っていない。

## 2026-09-03 — UAR・macro F1で学習状態を診断できる範囲

**質問/文脈**: 学習できているかの判断材料としてUARを主指標、macro F1も重視する考え方が適切かという質問。両者のtrain・validationの比較は分類性能の改善や過学習の兆候を確認する材料として妥当だが、研究上の指標の優先順位だけで診断の妥当性を説明しない。UARは各クラスの再現率を等しく平均し、macro F1はクラスごとのprecisionとrecallの両方を反映するため、今回のクラス件数が偏った分類の学習状態を調べる意味がある。

`train_one_epoch` はcross entropyを最小化し、`evaluate_loader_metrics` は確率のargmaxから予測ラベルを作ることを確認した。予測ラベルが変わらず正解クラスの確率だけが改善した場合、accuracy・UAR・macro F1が変化せずlossだけが低下することがある。この例は定義からの説明であり、今回の実測結果を示すものではない。trainとvalidationのscoreがともに低いことだけでepoch不足と断定しない。trainだけ改善してvalidationが悪化する推移は過学習を疑う材料だが、単独の最終値や一時的な変動のみで断定しない。lossをsplit間で比較する場合は計算方法をそろえ、重み付き学習lossと重みなしvalidation lossの絶対値を直接比較しない。

**再引用・公式本文確認**: scikit-learnの `balanced_accuracy_score.html`、`f1_score.html`、`learning_curve.html` およびPyTorch 2.12の `CrossEntropyLoss.html`（https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html）。既存の書誌行との重複は省略。今回の相談では学習・評価コードやNotebookを変更せず、実学習も実行していない。

## 2026-09-03 — 保存済み実験結果の集計とグラフ化

**質問/文脈**: ユーザーの依頼に基づき、既存の重み比較・転移実験・時間計測の6個のJSONから、3 seedの比較表と6種類の図を作成した。入力ファイルのSHA-256、評価集合signature、混同行列と指標の整合性、best epochの選択、10 epoch・バッチサイズ8などの比較条件を確認。時間計測用seed 42の再実行は元の結果と一致するため、独立seedとして重複集計していない。音声・特徴キャッシュ・checkpoint本体は読み込まず、追加学習・推論・test評価は実行していない。

過去のtrain分類scoreは未記録、重みありMSPのtestは未評価であることを明記した。train loss低下とvalidationの後半の推移は過学習を疑う材料として扱い、未記録のtrain scoreとの差を推定していない。元の重みなし転移実験のtestと、新しい重み付け実験のvalidationを区別した。

**再引用・公式本文確認**: scikit-learnの *Cross-validation: evaluating estimator performance*（https://scikit-learn.org/stable/modules/cross_validation.html）。保存済みのtest指標から設定を選び直す場合も、test由来の情報が選択に入る点の根拠として本文を確認した。今回の集計は記述的な整理であり、新しいモデル設定を選択していない。既存書誌行との重複は省略。

## 2026-09-03 — score主表示・loss定義・評価時の状態復元

`claim-verify`による公式資料の本文確認。比較用lossの確率からの計算式は既存コードを維持し、定義情報だけを`history_metadata`へ追加した。

| 用語・主張 | 判定 | 根拠 | 正しい表現 |
|---|---|---|---|
| `eval()`と勾配計算の無効化は別の設定 | 確認済み | [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html#evaluation-mode-nn-module-eval) | 評価時は`eval()`と`no_grad()`の両方を使う |
| クラス番号ラベルの重み付きCrossEntropyLossのmeanは対象ラベルの重み総和で正規化 | 確認済み | [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) | バッチ内の重み付き損失和 / 正解クラス重みの総和 |

確認済み2件。epoch内のバッチlossの単純平均はこのリポジトリの既存集約方法であり、split全発話の重みなし平均とは別の記録として表示する。今回のテスト環境はPyTorch 2.12.1+cu130。ユーザーが提示した2.14の公式資料を定義確認に用い、実装の数値はローカルの合成データテストで確認した。実データの学習・モデル評価は行っていない。
