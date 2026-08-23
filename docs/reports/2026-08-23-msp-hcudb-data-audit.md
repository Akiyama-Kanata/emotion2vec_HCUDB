# MSP-Podcast→HCUDB feature/decoder分離 — データ監査・preflight記録

監査日: 2026-08-23（Asia/Tokyo）

## 結論

mapping対象件数、話者数、公式split、HCUDB固定split、IEMOCAP全Sessionは実metadataで再確認できた。HCUDBとIEMOCAPの対象音声は欠損0件である。一方、MSP-Podcast R1.10の`Audio/`は空であり、対象25,985件すべてが欠損しているため、strict manifest作成、MSP全件容量・時間見積り、正式抽出・学習の開始ゲートは不合格のままである。

## 監査結果

| dataset | metadata | 主mapping対象 | 既知話者 | 音声状態 |
|---|---:|---:|---:|---|
| MSP-Podcast R1.10 | 104,267 | 25,985 | 1,432 | 対象25,985件すべて欠損 |
| HCUDB1 | 4,620 | 2,100 | 14 | metadata/WAV相互欠損0 |
| IEMOCAP | 10,039 | 3,825 | 10 | 対象音声欠損0 |

IEMOCAPのAppleDouble `._*` 1,819件は音声inventoryから除外した。実発話metadataと対応する有効WAVは10,039件である。

### MSP-Podcast公式partition

| source split | metadata件数 | 今回の扱い |
|---|---:|---|
| Train | 63,076 | `train`候補 |
| Development | 10,999 | `validation`候補 |
| Test1 | 16,903 | `test`候補 |
| Test2 | 13,289 | 全件監査のみ。manifestでは除外 |

4クラスmapping、Test2除外、`SpkrID=Unknown`除外を同時に適用した対象件数が25,985件である。既知話者についてTrain/Development/Test1間の重複は0件である。

### HCUDB固定話者split

version: `hcudb1_speaker_split_v1`

| split | speaker |
|---|---|
| train | `FA, FB, FD, FH, FI, FL, MC, MJ, MM, MN` |
| validation | `FF, MK` |
| test | `FG, ME` |

各話者330発話、各演技感情30発話であり、採用する5演技感情から1話者150発話、全14話者で2,100発話となる。「嫌い→disgust」だけを近似対応として記録する。

### IEMOCAP

Session 1–5のmetadata 10,039件を外部`test`へまとめる。4クラスmapping後は3,825件で、`disgust` supportは2件である。decoder出力と確率は4クラスのまま保存し、4クラス記述評価と、真値`disgust`を除く3クラス主集計を分ける。3クラス主集計でも`disgust`予測は誤りとして扱い、確率は再正規化しない。

## 実音声1件benchmark

実行結果の機械可読版: `docs/reports/2026-08-23-hcudb1-feature-benchmark.json`

| 項目 | 結果 |
|---|---:|
| 音声 | `FA-01-01-1.wav` |
| 元形式 | mono / 48 kHz / 1.2587 s |
| 前処理 | `scipy.signal.resample_poly`で16 kHz化 |
| device | CPU |
| checkpoint読込 | 66.271 s |
| 前処理 | 1.905 s（初回importを含む） |
| 特徴抽出 | 0.417 s |
| 抽出real-time factor | 0.331 |
| 特徴shape | 62 × 768 |
| 特徴容量 | 190,464 bytes（151,322 bytes/音声秒） |
| layer契約 | `final_after_encoder_norm` |

この1件はcheckpoint load・48→16 kHz変換・最終特徴shape・有限float32を確認するpreflightであり、MSP全件の代表速度を保証する値ではない。MSP対象音声の総durationを現在計算できないため、MSP全件の所要時間・cache容量推定には使用しない。

使用したBase checkpointは1,125,606,009 bytes、SHA-256は`4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd`である。確認時のCドライブ空きは96,816,496,640 bytes（90.17 GiB）だった。ただしMSP推定cache容量が未算出なので、空き容量の最終合否判定は行わない。

## 事前学習重複リスクの主張検証

| 主張 | 判定 | 根拠・扱い |
|---|---|---|
| emotion2vecの事前学習データにMSP-Podcast v1.8が含まれる | ✅ 確認済み | emotion2vec論文Table 1。全評価のlimitationへ付記する |
| MSP-Podcast R1.8がR1.10の完全な部分集合である | ❌ 未確認 | R1.8 metadata未配置。包含関係を主張せず`unverified`とする |

一次資料: [Ma et al., emotion2vec, Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.931/)

## 正式実行ゲート

- [x] 3 mapping version、4クラス順、HCUDB固定splitをGit管理下で固定
- [x] Test2監査と今回範囲からの除外
- [x] Unknown話者を主対象から除外
- [x] HCUDB/IEMOCAP対象音声欠損0
- [x] integer layer拒否と`final_after_encoder_norm`契約
- [x] shard中断復旧・hash・mmap・metadata不一致テスト
- [x] MSP親/HCUDB子のID・親SHA・resume分離を合成E2Eで確認
- [x] before/after評価集合signature一致を合成E2Eで確認
- [x] HCUDB実音声1件benchmark
- [ ] MSP対象25,985音声の配置とmetadata一致
- [ ] MSP対象音声の総durationに基づく時間・容量見積り
- [ ] 推定cache+20%を含むディスク安全域確認
- [x] 全101テストと両Notebook demoの最終再実行
- [ ] 対象manifest hash、予定時間、必要容量、出力先の提示
- [ ] ユーザーの明示的な正式開始指示

したがって、実装・合成E2E・1件benchmarkまで完了可能だが、正式な全件抽出と学習は開始しない。
