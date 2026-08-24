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
