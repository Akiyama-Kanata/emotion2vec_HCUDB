# 実験C：結果・方法・評価妥当性の発表用まとめ

作成日: 2026-09-19  
対象: `emotion2vec+ large` 公式head固定条件（Condition C）  
用途: 研究発表・進捗報告（主結果、検証証拠、限界を分離して提示する）

## 発表の結論

> 実験Cの保存結果は、公式head・特徴cacheから全7,556発話を再評価して完全再現でき、予測JSONから独立に再集計した指標とも一致した。評価計算やデータ対応の誤りを示す証拠は見つからなかった。一方、MSP-Podcastの事前学習データ重複可能性、HCUDBのラベル近似、固定1分割という研究上の限界は残る。

| 確認対象 | 結論 | 根拠レベル |
|---|---|---|
| 保存された数値の再現性 | 全発話の再評価で予測・指標が完全一致 | 強い実証 |
| 指標計算 | Accuracy、UAR、Macro-F1、Loss、混同行列を独立再計算して一致 | 強い実証 |
| 公式推論との対応 | FunASR v1.4.15と同じ正規化・時間平均・9出力projection・softmax。基準音声の最大logit誤差0.0 | 強い実証 |
| test集合の独立性 | MSP、HCUDBとも今回のmanifest内で話者・group・音声hashのsplit間重複なし | 強い実証 |
| 科学的な一般化可能性 | 限界あり。特にMSPの事前学習重複可能性とHCUDBの固定2話者test | 留保が必要 |

「完全に間違いがない」と数学的に証明したのではなく、保存物・実装・独立再計算・公式実装を突き合わせた範囲で、評価誤りを示す不整合がないと結論づける。

## 実験Cは、公式9クラスモデルを追加学習せず評価した固定ベースラインである

```text
MSP-Podcast Test1 / HCUDB Test
  → 16 kHz・mono
  → emotion2vec+ large（固定、1024次元frame特徴）
  → 有効frameの時間平均
  → 公式Linear head（固定、9 logits）
  → softmax（9クラス）
  → argmax（9クラス）
  → 対象6感情のAccuracy / UAR / Macro-F1 / 9-way NLL
```

- 公式9クラス順は `angry / disgusted / fearful / happy / neutral / other / sad / surprised / unknown`。
- 評価対象6感情は `angry / disgusted / fearful / happy / sad / surprised`。
- 真値は対象6感情だけだが、予測は9クラス全体から選ぶ。`neutral / other / unknown` が最大なら誤分類として数える。
- Condition Cではoptimizer、学習、checkpoint選択を行わない。公式headは評価前後のstate hash一致を検査する。
- UARは6クラスのrecallの単純平均、Macro-F1は6クラスのF1の単純平均である。
- Lossは、各発話の真の公式クラスに対する9クラスnegative log-likelihoodの平均である。

この規則は「公式モデルが本来持つ9クラス判断を変えずに、対象6感情への適合度を見る」という研究質問には整合する。対象6クラス内だけのclosed-set分類能力を問う場合は別の評価規則になるため、後述の感度分析として分離する。

## 評価集合は固定され、学習・validationとtestの話者重複は検出されなかった

| 条件 | MSP-Podcast | HCUDB1 |
|---|---:|---:|
| 使用release / label profile | R1.10 / `official6` | HCUDB1 / `official6` |
| split | 公式Test1 → `test` | 固定話者splitの`test`（FG, ME） |
| test発話数 | 7,136 | 420 |
| test話者数 | 60 | 2（女性1、男性1） |
| train / validation / test話者重複 | 0 / 0 / 0 | 0 / 0 / 0 |
| testクラス件数 | 741 / 717 / 425 / 3,722 / 510 / 1,021 | 60 / 60 / 60 / 120 / 60 / 60 |
| 主な除外 | 欠損音声1,397件、重複音声62件（うちtest 31件）、Test2、対象外ラベル | 対象外の演技感情 |

クラス件数の順序は `angry / disgusted / fearful / happy / sad / surprised`。HCUDBのhappyは「狂喜・楽しい」と「余裕・嬉しい」の2演技感情を対応させるため120件、他クラスは各60件である。

HCUDB1は14名のプロ話者、4,620発話から構成される演技感情音声コーパスであり、今回の分割はtrain 10名、validation 2名、test 2名で話者を分離している。コーパス自体には他者評価感情もあるが、今回の教師ラベルは演技感情である ([NII HCUDB公式情報](https://research.nii.ac.jp/src/en/HCUDB.html); Mera et al., 2025)。

## 主結果では、HCUDBがMSPより高いUARを示した

| Dataset | N | Accuracy | UAR | Macro-F1 | 9-way Loss | 非対象3クラス予測 |
|---|---:|---:|---:|---:|---:|---:|
| MSP-Podcast Test1 | 7,136 | 30.72% | 16.57% | 22.52% | 4.8320 | 4,227件（59.23%） |
| HCUDB Test | 420 | 37.62% | 36.11% | 34.49% | 5.4745 | 33件（7.86%） |

### クラス別recall

| Dataset | angry | disgusted | fearful | happy | sad | surprised |
|---|---:|---:|---:|---:|---:|---:|
| MSP-Podcast | 11.20% | 8.09% | 4.00% | 51.13% | 24.31% | 0.69% |
| HCUDB | 75.00% | 0.00% | 21.67% | 46.67% | 28.33% | 45.00% |

発表で強調する点は次の2つに絞る。

1. MSPでは59.23%が`neutral / other / unknown`に予測され、公式9クラス判断を保つと主指標が大きく下がる。
2. HCUDBの`disgusted`は予測が0件でrecall 0%。これは集計漏れではなく、6×9混同行列と全予測行から確認できる実測結果である。

## 対象6クラス限定の参考値は、主結果と混ぜずに示す

9クラス確率のうち対象6クラスだけでargmaxした感度分析を追加した。再学習や再推論は行わず、保存済み9クラス確率から計算した参考値である。

| Dataset | 評価規則 | Accuracy | UAR | Macro-F1 |
|---|---|---:|---:|---:|
| MSP-Podcast | 主結果: 9クラスargmax | 30.72% | 16.57% | 22.52% |
| MSP-Podcast | 参考: 対象6クラスargmax | 54.62% | 35.01% | 31.50% |
| HCUDB | 主結果: 9クラスargmax | 37.62% | 36.11% | 34.49% |
| HCUDB | 参考: 対象6クラスargmax | 39.76% | 37.92% | 35.13% |

| Dataset | Accuracy差 | UAR差 | Macro-F1差 |
|---|---:|---:|---:|
| MSP-Podcast | +23.91 pt | +18.45 pt | +8.98 pt |
| HCUDB | +2.14 pt | +1.81 pt | +0.64 pt |

MSPの差が大きいため、「性能が30.72%しかない」とだけ説明するとモデルの対象6クラス内の識別力と、非対象3クラスへの予測傾向を混同する。発表では主結果を変更せず、両者を別の問いとして並べる。

## 評価誤りが見つからなかった根拠は、5段階で確認した

| 検証層 | 実施内容 | 結果 |
|---|---|---|
| 1. 公式実装とのparity | 公式`funasr.AutoModel.generate`とローカルadapterを同一音声で比較 | 最大logit誤差0.0、`passed=true` |
| 2. provenance固定 | 公式snapshot revision、checkpoint/config/tokens/head、実装4ファイルをSHA-256で固定 | 保存値と現ファイルが一致 |
| 3. 全cache再評価 | 公式headと保存済み1024次元frame cacheからMSP 7,136件、HCUDB 420件を再評価 | 両datasetで予測行・指標が保存物と完全一致 |
| 4. 独立再集計 | `predictions.json`だけからsoftmax、argmax、6×9混同行列、Accuracy、UAR、Macro-F1、Lossを再計算 | 全指標の絶対誤差0.0 |
| 5. 自動テスト | 公式ラベル契約、9-way評価、cache拒否、C/D分離、比較集合一致等 | 18件成功、実snapshot環境変数を要する1件のみskip |

追加で、全7,556予測について次を確認した。

- 発話IDはdataset内で一意で、test manifestのID集合と一致。
- 真値index、mapped emotion、original emotionはmanifestと一致。
- 保存確率は各行で合計1、保存logitから再softmaxした値と浮動小数点誤差内で一致。
- 保存予測は9クラス確率のargmaxと一致。
- `correctness`、非対象3クラス件数、混同行列、クラスsupportが一致。

主要な監査対象は次のartifactである。

- `runs/official_cd/parity.json`
- `runs/official_cd/evaluation_c/c_evaluation_summary.json`
- `runs/official_cd/evaluation_c/{msp_podcast,hcudb1}/C/{metrics,predictions}.json`
- `runs/ser_manifests/{msp_podcast,hcudb1}_official6_v1.jsonl`
- `runs/official_cd/cache/{msp_podcast,hcudb1}/cache_meta.json`

## 「評価方法が妥当」と言える範囲と、まだ言えない範囲

### 今回言えること

- 保存結果は、固定された公式head・cache・manifestから再現できる。
- 指標の式、ラベル順、予測規則、発話ID対応に内部不整合はない。
- Cは追加学習なしの固定ベースラインとして実装され、評価中にheadが変化していない。
- test集合は今回のmanifest内で話者・group・音声hashのsplit間漏洩が検出されていない。
- 9クラスargmaxは公式モデルの非対象出力を隠さない、保守的な評価である。

### 現時点では言えないこと

- **MSPを完全未知データとは呼べない。** emotion2vec論文は事前学習にMSP-Podcast v1.8を含む（Ma et al., 2024, PDF p.5）。今回のR1.10 Test1との発話単位の重複は、v1.8 metadataがローカルにないため未確認である。
- **HCUDB全体への一般化を保証しない。** testは固定した2話者420件であり、14話者cross-validationではない。
- **「嫌い→disgusted」は厳密な同義対応ではない。** 研究上の近似であり、HCUDB testの60件に影響する。
- **演技感情と知覚感情は同一ではない。** 今回は演技感情を真値にしており、16名の他者評価感情を真値にした実験ではない（Mera et al., 2025）。
- **9クラス評価と6クラス評価は別タスクである。** 感度分析の高い値を主結果へ置き換えると、実験Cの評価規則を事後変更したことになる。
- **MSPとHCUDBのloss値を単純比較しない。** 同じ式でも、データ分布と確率校正の状態が異なるため、dataset間の優劣はUAR等と誤分類内訳を併記して判断する。

## 7枚で説明する場合の構成

| # | assertion headline | 画面に置く証拠 | 話す補足 |
|---:|---|---|---|
| 1 | 実験Cの結果は全発話再評価で完全再現できた | MSP/HCUDBの主指標と「7,556件一致」 | 誤りがない絶対証明ではなく、不整合が見つからないという結論 |
| 2 | Cは公式9クラスheadを変えない固定ベースラインである | 左から右の推論フロー、固定マーク | 追加学習なし、9クラス全体でargmax |
| 3 | MSPでは非対象3クラス予測が59.23%を占める | 主指標表と非対象予測率 | 低いUARの一部は9クラス判断規則に由来 |
| 4 | HCUDBではUAR 36.11%だがdisgusted recallは0%である | クラス別recall棒グラフ | 「嫌い→disgusted」は近似対応 |
| 5 | 6クラス限定値はMSPで大きく変わる | 9-way対6-wayのペア棒グラフ | 主結果ではなく評価規則の感度分析 |
| 6 | 結果の信頼性は公式parity・全件再評価・独立再集計で支えられる | 5段階検証表 | 18テスト成功、予測・指標完全一致 |
| 7 | 計算は妥当でも、データ重複可能性とラベル近似は残る | 「言える／言えない」2列 | MSP v1.8事前学習、HCUDB固定2話者testを明示 |

画面では数値と結論だけを示し、SHA-256、artifact path、指標式、文献詳細は発表者ノートまたはbackupへ移す。

## 主張検証レポート

| 用語・主張 | 判定 | 根拠 | 本資料での表現 |
|---|---|---|---|
| emotion2vec | ✅ 確認済み | ACL 2024論文、公式実装 | `emotion2vec` |
| UAR（Unweighted Average Recall） | ✅ 確認済み | SER論文で一般的に使用。実装はクラス別recallの単純平均 | `UAR` |
| HCUDB | ✅ 確認済み | NII公式ページ、査読論文 | `広島市立大学感情音声データベース（HCUDB）` |
| 公式9クラスargmax | ⚠️ ローカルな説明語 | 標準固有名ではなく、この実験の決定規則を記述した語 | 初出で「9クラス全体の確率最大クラスを予測」と定義 |
| target6 sensitivity / 対象6クラス感度分析 | ⚠️ ローカルな説明語 | 学術的な固有手法名ではない | 「対象6クラス限定argmaxによる参考値」と記載 |

注: HCUDBの正式な展開は **Hiroshima City University** であり、北海道大学ではない。

サマリー: ✅ 確認済み3件 / ⚠️ ローカルな説明語2件 / ❌ 未確認0件。

## Source Verification Report

### Input 1: ACL Anthology / emotion2vec論文

**Date checked: 2026-09-19**

**Safety Rating: LOW RISK**  
HTTPSの学術アーカイブで、偽装ドメインや不自然なURL構造はない。

**Accuracy Assessment: SUPPORTED**  
論文本文PDFとACL書誌情報が一致し、モデルの目的、MSP-Podcast v1.8の事前学習利用、評価指標を確認した。

**Credibility Score: 10/10 — HIGH**  
査読済み会議録、著者明記、2024年、一次論文である。

**Summary**  
モデルの学術的背景、MSP-Podcast v1.8の事前学習利用、評価指標の根拠として採用する。

### Input 2: ModelScope/FunASR v1.4.15公式source

**Date checked: 2026-09-19**

**Safety Rating: LOW RISK**  
HTTPSのGitHub上にある公式organizationのタグ固定sourceである。

**Accuracy Assessment: SUPPORTED**  
sourceにwaveform normalization、frame平均、linear projection、softmaxが明記され、ローカルparityでもlogit完全一致を確認した。

**Credibility Score: 8/10 — HIGH**  
公式一次sourceで版が固定される一方、論文査読資料ではない。

**Summary**  
実行時の正規化、pooling、projection、softmaxの仕様確認に採用し、ローカルparityで補強する。

### Input 3: NII HCUDB公式ページ / J-STAGE論文

**Date checked: 2026-09-19**

**Safety Rating: LOW RISK**  
NIIとJ-STAGEの公式HTTPSドメインである。

**Accuracy Assessment: SUPPORTED**  
HCUDB1の14話者・4,620発話・演技感情と他者評価感情という説明が両sourceで一致する。

**Credibility Score: 10/10 — HIGH**  
公的配布機関と査読誌の一次資料で、著者・DOI・公開日が明記される。

**Summary**  
HCUDBの正式名称、構成、演技感情と他者評価感情の区別の根拠として採用する。

### Input 4: MSP Laboratory公式ページ / MSP-Podcast論文

**Date checked: 2026-09-19**

**Safety Rating: LOW RISK**  
研究室の公式HTTPSドメインとarXivであり、偽装を示すURL要素はない。

**Accuracy Assessment: SUPPORTED**  
Podcast由来の自然発話、知覚アノテーション、話者分割の目的を両sourceで確認した。ただし公式ページは現行v2.0中心で、今回使用したR1.10の件数はローカルmanifestを一次証拠とする。

**Credibility Score: 9/10 — HIGH**  
公式コーパス提供者の一次情報と著者プレプリント。今回のrelease固有件数はローカルR1.10 artifactで補完する。

**Summary**  
コーパスの収集・注釈・分割方針の背景に採用する。R1.10の実数はローカルmanifestを優先する。

| # | Input | Safety | Accuracy | Credibility |
|---:|---|---|---|---|
| 1 | ACL Anthology | LOW RISK | SUPPORTED | 10/10 HIGH |
| 2 | FunASR v1.4.15 source | LOW RISK | SUPPORTED | 8/10 HIGH |
| 3 | NII / J-STAGE HCUDB | LOW RISK | SUPPORTED | 10/10 HIGH |
| 4 | MSP Laboratory / MSP-Podcast paper | LOW RISK | SUPPORTED | 9/10 HIGH |

## References

- Busso, C., Lotfian, R., Sridhar, K., et al. (2025). *The MSP-Podcast Corpus* [Preprint]. arXiv. https://arxiv.org/abs/2509.09791 `[中・要確認: Web要旨のみ]`
- Ma, Z., Zheng, Z., Ye, J., Li, J., Gao, Z., Zhang, S., & Chen, X. (2024). emotion2vec: Self-supervised pre-training for speech emotion representation. *Findings of the Association for Computational Linguistics: ACL 2024*, 15747–15760. https://doi.org/10.18653/v1/2024.findings-acl.931 `[高・PDF確認済み]`
- Mera, K., Kurosawa, Y., Nakayama, M., & Takezawa, T. (2025). 感情ラベルと演技方法の違いを考慮した演技感情音声データベースHCUDBの構築. *知能と情報, 37*(4), 725–733. https://doi.org/10.3156/jsoft.37.4_725 `[高・要確認: Web要旨のみ]`
- ModelScope/FunASR. (2026). *Emotion2vec implementation, v1.4.15*. https://github.com/modelscope/FunASR/blob/v1.4.15/funasr/models/emotion2vec/model.py `[公式source]`
- National Institute of Informatics. (n.d.). *Hiroshima City University Japanese Emotional Speech Corpus (HCUDB).* https://research.nii.ac.jp/src/en/HCUDB.html `[公式コーパス資料]`
- MSP Laboratory. (n.d.). *MSP-Podcast corpus.* https://www.lab-msp.com/MSP/MSP-Podcast.html `[公式コーパス資料]`
