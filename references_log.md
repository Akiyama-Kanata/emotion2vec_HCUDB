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
