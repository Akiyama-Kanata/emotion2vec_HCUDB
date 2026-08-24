# emotion2vec研究 現状整理・進捗報告書

> **履歴資料:** 本報告書は2026-08-21時点のスナップショットである。2026-08-22に主学習データ、ラベル、実験経路が変更されたため、現行判断には[MSP-Podcast→HCUDB SER研究 現行実施計画](../plans/2026-08-22-msp-hcudb-feature-decoder-plan.md)を使用する。

作成日: 2026-08-21（Asia/Tokyo）  
対象HEAD: `8f5612fcd4578145e8171b38ff27079dfe171386`  
ブランチ: `test`（`origin/test`より1コミット先行）

## 1. 総合判定

現在地は、**研究目的・段階計画・Base向け下流モデルの試作・自動テスト基盤・データセット間ラベル調査がそろい、必須実験Aの実装と実データ準備へ入る直前**である。

ソフトウェアの基礎状態は良好である。2026-08-21に標準WSL環境で全76テストを再実行し、`Ran 76 tests in 8.550s`、`OK`を確認した。一方、研究の中心であるIEMOCAP学習後からHCUDB追加学習後への日英性能変化は、1 seedも正式実行されていない。したがって、現時点で「日本語性能が向上した」「英語性能を維持した」「BaseまたはLargeが優れる」とは主張できない。

現在の段階を一文で表すと、**実験用プロトタイプは成立しているが、研究仮説を判定する実データ成果は未取得**である。

## 2. 研究目的

現行計画の主目的は、条件Aでemotion2vec Baseエンコーダーを固定し、同じBase用decoderについて次の2時点を比較することである。

1. IEMOCAPで学習した直後
2. そのcheckpointを親としてHCUDBで追加学習した後

比較対象は英語IEMOCAPと日本語HCUDBの性能であり、条件Aを必須実験とする。条件AはIEMOCAPのsession独立5-fold・各fold 3 seed、合計15系列で行う。英語維持の暫定基準は、各fold内で3 seedの`追加学習後 − 追加学習前`を平均し、そのfold平均を5-foldで平均したmacro F1差とUA差が、いずれも`-2.0`ポイント以上であることとする。この2.0ポイントは一般規格ではなく、本研究内で結果取得前に固定する判定基準である。

条件B〜Dは主目的達成後の拡張であり、Aの未完了中は着手しない。

| 条件 | 役割 | encoder / head | 現在の状態 |
|---|---|---|---|
| A | 必須・主目的 | emotion2vec Base固定＋新規decoder、IEMOCAP 5-fold×3 seed | 分割方針確定、実装準備前半、正式実験未着手 |
| B | encoder規模の拡張比較 | emotion2vec+ large固定＋Large特徴次元に合わせた別decoder、Aと同じfold・seed | 未着手 |
| C | 公式分類headの参考評価 | emotion2vec+ large公式9クラスhead | 未着手 |
| D | 公式`proj`追加学習との比較 | emotion2vec+ large公式`proj`のみ追加学習 | 未着手 |

条件A内では、IEMOCAP学習からHCUDB追加学習へ同じBase用decoderを引き継ぐ。これに対して条件Aと条件Bのdecoderは、encoderの出力次元に応じて入力層の`in_features`と重みshapeが異なる別decoderである。A/B比較ではdecoder自体を「同一」とせず、入力層より後段の構成と学習・評価条件を対応させる。

2026-08-21のユーザー判断により、従来の固定1 split・3 seedは廃止し、IEMOCAPの5-fold・各fold 3 seedを条件Aの必須設計として採用した。HCUDBは14話者を固定10/2/2へ分割し、全IEMOCAP fold・seedで同じ話者splitを使う。

現行計画の単一基準は[日英SER研究の段階実施工数計画](../plans/2026-08-11-ja-en-ser-revised-effort-plan.md)である。

## 3. 現行設計

### 3.1 必須実験Aの処理系列

| Fold | IEMOCAP Train | IEMOCAP Validation | IEMOCAP Test |
|---:|---|---|---|
| 1 | Session 3・4・5 | Session 2 | Session 1 |
| 2 | Session 4・5・1 | Session 3 | Session 2 |
| 3 | Session 5・1・2 | Session 4 | Session 3 |
| 4 | Session 1・2・3 | Session 5 | Session 4 |
| 5 | Session 2・3・4 | Session 1 | Session 5 |

各fold・seedの処理系列は次のとおりである。

```text
IEMOCAP train / validation
  -> 固定emotion2vec Base特徴
  -> Base用decoder学習
  -> IEMOCAP test評価（追加学習前）
  -> 固定HCUDB train / validation（10 / 2話者）で同じdecoderを追加学習
  -> 同じIEMOCAP testを再評価（追加学習後）
  -> 固定HCUDB test（2話者）を評価
```

- encoderはoptimizerへ渡さず固定する。
- IEMOCAPは上表のsession独立5-foldを使い、各foldを3 seedで実行する。
- HCUDBは話者が重複しない固定train 10 / validation 2 / test 2話者を使用する。話者割当は結果取得前にmanifestへ固定する。
- checkpointには`encoder_id`、`input_dim`、ラベル順、fold、seed、IEMOCAP split、HCUDB話者split、`training_stage`、`parent_checkpoint`を保存する計画である。
- 指標はmacro F1、UA、accuracy / WA、クラス別F1、confusion matrixとする。集計はfold内3 seed、その後5-foldの順で行い、15系列を独立標本として扱わない。

### 3.2 実装済みのモデル系統

| 系統 | 実装 | 位置づけ |
|---|---|---|
| IEMOCAP 4クラス分類 | frame特徴のpooling後にFNN分類 | 既存のBase用入口。現行Aの12クラス契約へは未拡張 |
| 直接感情分類＋並列V/A/D | 感情headとValence・Arousal・Dominanceの独立head | 主経路の試作部品。可変クラス数とcheckpoint継続の一部を実装済み |
| VAD媒介型分類 | 予測VADを感情分類器の入力にする | 比較・探索用。条件Aの主設計ではない |
| VAD / VA回帰 | CCC lossによる連続値回帰 | 連続感情値を扱う補助経路 |

`ParallelEmotionVADClassifier`や「VAD媒介型」は、このリポジトリ内の設計名として扱う。一般に確立されたモデル名として卒論本文へ記載しない。

## 4. 工程別ステータス

| 工程・成果物 | 判定 | 根拠 | 次に必要なこと |
|---|---|---|---|
| 研究目的・A→B→C→Dの実施順 | 完了 | 2026-08-11現行計画 | ラベル設計の解釈だけを実装前に固定する |
| 自動テスト基盤 | 完了 | 2026-08-21に76件すべて成功 | `TESTING.md`の期待件数と最新検証を更新する |
| Base checkpoint | 存在・配線確認済み | `artifacts/checkpoints/emotion2vec_base.pt`、1,125,606,009 bytes | 正式な特徴cache作成とencoder固定検証 |
| IEMOCAP 4クラス学習Notebook | 実装済み・デモ検証済み | `notebooks/iemocap_base_downstream_training.ipynb`、デモ実行Notebook | 12クラス契約、5-fold×3-seed正式実験へ拡張 |
| IEMOCAPの実データ経路 | 部分実装・本日未確認 | private mode、manifest生成、特徴抽出関数が存在 | 4つの必須環境変数を設定し、Session 1〜5を再検証 |
| HCUDB注釈 | ローカル確認済み | `.env`の`VAD_CSV_PATH`が4,620行CSVを指す | 音声rootを設定し、注釈との一致を再検証 |
| HCUDB音声 | 過去計画では存在、本日未確認 | 現在の`RESTRICTED_DATA_DIR`は未設定 | 4,620 WAVの存在・形式・話者を現在環境で再確認 |
| データセット間ラベル対応 | 調査完了 | 2026-08-20対応表 | 採用契約を確定し、version付き変換コードを作る |
| A用12クラス出力 | 未実装 | IEMOCAP側は`ang/hap/neu/sad`固定、VADデータ層の既定も4クラス | unionラベル、`xxx`除外、`exc`評価時統合を実装・テスト |
| IEMOCAP 5-fold | 部品実装済み | session分割、validation/test分離、単一seedの5-foldデモが存在 | 5-fold×3 seedとHCUDB継続学習を一体化し、実Session 1〜5で検証 |
| HCUDB固定10/2/2話者split | 方針確定・manifest未保存 | 汎用speaker split処理が存在 | 結果取得前に話者割当を生成・検証・保存 |
| 継続checkpoint経路 | 部分実装 | 並列型に初期checkpoint読込がある | Aの共通decoder、親子metadata、不一致拒否をE2E化 |
| Base特徴cache | デモcacheのみ | `runs/`内に仮特徴cacheは存在 | IEMOCAP / HCUDB実音声から研究用Base特徴を抽出 |
| 条件A・先行1系列 | 未着手 | 正式metrics / checkpointなし | Fold 5・seed 42を最後まで実行 |
| 条件A・15系列集計 | 未着手 | fold内seed集計・5-fold集計なし | 先行系列修正後に残り14系列を実行 |
| 条件B〜D | 未着手 | 現行計画どおり | A完了後に個別再見積もり |
| DS-001分布確認 | 未完了 | 仕様とラベルだけ確認、メタデータ未入手 | 感情・強度・話者別件数を集計 |

## 5. 現在の成果物

### 5.1 研究計画・調査文書

- [日英SER研究の段階実施工数計画](../plans/2026-08-11-ja-en-ser-revised-effort-plan.md): 現行計画の単一基準。
- [IEMOCAP・HCUDB1・DS-001 感情ラベル対応表](2026-08-20-emotion-label-correspondence.md): 直接対応・近似対応・対応なしと、4 / 7クラス候補を整理。
- [2026-08-02進捗・完成度報告](2026-08-02-current-progress-completion-report.md): 履歴資料。現在の計画判断には使用しない。
- `references_log.md`: データセット・関連研究の参照履歴。

### 5.2 コード・Notebook・テスト

- `iemocap_downstream/`: IEMOCAP 4クラス分類、session split、学習・評価・checkpoint処理。
- `vad_downstream/`: VA/VAD回帰、VAD媒介型、直接分類＋並列V/A/D、データ検証、学習・推論CLI。
- `notebooks/`: データ分析、IEMOCAP学習、VADモデル確認、Wagner系実験準備。
- `tests/`: 76件。データ整合、モデル形状、loss、checkpoint、推論、Notebook構造・デモ実行を検証。

### 5.3 実行生成物

| 生成物 | 解釈 |
|---|---|
| `artifacts/checkpoints/emotion2vec_base.pt` | 実Base encoder checkpoint。正式実験の入力資産 |
| `outputs/real_emotion2vec_smoke.json` | 実encoder＋random modelの配線確認。性能結果ではない |
| `runs/notebooks/audio_to_emotion_vad/*` | 合成音声・仮特徴によるデモ。研究結果ではない |
| `runs/iemocap_base_downstream/five_fold/*` | `demo_synthetic_fixed_features`による5-foldデモ。平均値をIEMOCAP性能として引用不可 |

`artifacts/checkpoints/`、`outputs/`、`runs/`は`.gitignore`対象であり、Gitだけでは再現用生成物を保全できない。正式実験では、成果物の保存場所・manifest・commit hash・checkpoint hashを別途固定する必要がある。

## 6. 2026-08-02以降に進んだこと

1. WSL環境の`pandas`問題を解消し、65テスト成功を確認した。
2. IEMOCAP学習Notebookのvalidation選択、test分離、5-foldデモ、checkpoint再読込を強化した。
3. テスト数が76件へ増え、2026-08-11と2026-08-21の両方で全件成功した。
4. 条件Aを必須とし、B〜Dを順番に後置する現行計画へ整理した。
5. IEMOCAP・HCUDB1・DS-001のラベルと件数を比較し、直接対応と近似対応を分離した。
6. 最新のラベル対応文書を作成した。
7. 2026-08-21に固定1 splitを廃止し、IEMOCAP 5-fold×3 seedとHCUDB固定10/2/2話者splitを条件Aへ採用した。

主実験のmetrics取得はこの期間にも進んでいない。進展は、主として**計画の固定、試作コードの信頼性向上、データ契約の調査**である。

## 7. 主要課題・リスク

### 7.1 ラベル設計の解釈が未確定

現行計画は「日英ラベルの和集合12クラスで学習し、共通4感情で主比較する」と定める。一方、2026-08-20対応表は「`anger / happy / sadness / neutral`の実用4クラスを主実験の第一候補」と記載する。共通4感情の表記と出力順は2026-08-22にこの並びへ固定された。

この2つは、実用4クラスを**評価対象だけ**とするなら両立するが、学習出力も4クラスへ縮小するなら両立しない。現状は変換コードがないため、実装前に次を明文化する必要がある。

- 学習出力はunion 12のままか。
- 実用4クラスは評価時フィルタ・統合だけか。
- HCUDBの`冷静 -> neutral`、喜び2カテゴリの`happy`統合を学習前に行うか。
- マッピング版をどの成果物へ保存するか。

現行計画を優先する暫定解釈は、**union 12で学習し、実用4クラスは主評価用の対応規則として使う**ことである。変更する場合は現行計画を改訂する。

### 7.2 実データの現在接続が未完了

- IEMOCAP private modeに必要な`IEMOCAP_ROOT`、`IEMOCAP_WORK_DIR`、`EMOTION2VEC_CHECKPOINT`、`EMOTION2VEC_USER_DIR`は現在の環境で未設定である。
- HCUDB注釈CSVは4,620行を確認できるが、`RESTRICTED_DATA_DIR`が空であり、本日時点で4,620 WAVを再検証できていない。
- したがって、過去計画の「HCUDB WAVは存在」と「IEMOCAPはSession 1のみ確認」は履歴として保持し、正式学習前に現在の状態を再確認する。

### 7.3 条件Aの契約がコードへ未反映

- IEMOCAP読込は4クラス固定で、`exc`を準備時に`hap`へ統合する。現行計画の「評価時のみ統合」と一致しない。
- `vad_downstream/data.py`と`vad_downstream/notebook_pipeline.py`には768次元固定が残る。
- `encoder_id`・抽出条件・cache・checkpoint間の完全な一致検証は未完成である。
- union 12、`xxx`除外、未知ラベル拒否、IEMOCAP 5-fold×3 seed、HCUDB固定10/2/2話者split、親子checkpointを一続きで検証するE2Eテストがない。

### 7.4 記録と再現性

- `TESTING.md`は期待値65件、最新検証2026-08-02の失敗のままで、現在の76件成功と不一致である。
- `runs/`など正式成果物候補がGit管理外である。
- 実験ID、encoder ID、特徴次元、データsplit、seed、commit hash、mapping versionを一つのmanifestへ集約する仕組みが未完成である。

### 7.5 Git作業状態

- `test`ブランチは`origin/test`より1コミット先行している。
- `archive/logs/2026-08-20-work-log.md`と`archive/logs/next-chat-handoff.md`がuntrackedである。
- `.pytest_cache/`とユーザー側global ignoreへのアクセス警告が出るが、Git状態取得と全76テスト成功は妨げていない。

## 8. 優先順位つき次工程

### P0: 実装前の契約固定

1. 現行計画を優先し、union 12を学習出力、実用4クラスを主評価対象とするかを確定する。
2. 元ラベル、学習ラベル、評価時統合、除外ラベル、`mapping_version`を1つの仕様表に固定する。

### P1: データ接続と学習開始ゲート

3. IEMOCAPの4つのprivate環境変数を設定し、Session 1〜5のWAV・ラベル・件数を検証する。
4. HCUDB音声rootを設定し、4,620注釈とのファイル一致、音声形式、話者数、感情数を再検証する。
5. 利用容量を見積もり、特徴抽出後も20GB以上の空きを残せることを確認する。

### P2: 条件Aのコード完成

6. unionラベル変換、`xxx`除外、評価時`exc -> happy`、IEMOCAP 5-fold、HCUDB固定10/2/2話者splitを実装する。
7. fold・seedごとの共通decoderとIEMOCAP→HCUDB継続checkpointを実装し、metadata不一致を学習前に拒否する。
8. 既存76件にA固有テストを追加し、全件成功を学習開始条件とする。

### P3: 正式実験A

9. Base実音声1件で特徴抽出時間・shape・有限値・容量を測る。
10. IEMOCAPとHCUDBのBase特徴cacheを作成し、manifestを保存する。
11. Fold 5・seed 42でIEMOCAP学習→HCUDB追加学習→日英前後評価を最後まで通す。
12. 修正後に残り14系列を実行し、fold内3-seed集計、5-fold集計、対応差・図・英語維持判定を保存する。

### P4: 拡張実験

13. A完了後に実測工数を使ってBを再見積もりする。B完了後にC、C完了後にDへ進む。

## 9. 次の最小ステップ

次の最小ステップは、**現行計画のunion 12と、最新ラベル対応表の実用4クラスを一つの学習・評価契約へ統合すること**である。

暫定的には次の3層に分けると、既存計画を変えずに実装できる。

```text
original_emotion  : データセット固有の元ラベル
training_emotion  : union 12のdecoder出力ラベル
evaluation_emotion: 共通4感情への評価時変換
```

この契約を確定した後、IEMOCAP 5-fold、HCUDB固定10/2/2話者split、checkpoint継続経路を合成データでE2Eテストする。その時点で初めて、正式な特徴抽出へ進める。

## 10. 主張検証レポート

| 用語・主張 | 判定 | 根拠 | 報告書での扱い |
|---|---|---|---|
| emotion2vec | 確認済み | [ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.931/) | 正式名称を使用 |
| emotion2vec+ large | 確認済み | [公式実装のModel Card](https://github.com/ddlBoJack/emotion2vec) | 公式モデル名として使用 |
| IEMOCAP | 確認済み | [USC公開論文](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf) | 正式データセット名として使用 |
| HCUDB / HCUDB1 | 確認済み | [NII公式配布情報](https://www.nii.ac.jp/dsc/idr/speech/submit/HCUDB.html) | 「広島市立大学 感情音声コーパス」と記載 |
| leave-one-session-out 5-fold | 確認済み | [emotion2vec論文](https://aclanthology.org/2024.findings-acl.931/) | IEMOCAPのsession独立評価方法として使用 |
| union 12 | プロジェクト固有 | 現行計画内のラベル設計 | 学術標準用語として扱わない |
| VAD媒介型、並列Emotion-VAD | プロジェクト固有 | リポジトリ内のモデル設計 | 一般に確立されたモデル名と断定しない |

確認済み5件、プロジェクト固有2件、未確認の外部技術用語0件。複合造語を新しい標準モデル名としては使用していない。

## 11. 今回の確認範囲

### 実行した確認

- `git status --short --untracked-files=all`
- Gitブランチ、HEAD、直近12コミット
- 現行計画、直近ログ、既存進捗報告、ラベル対応表
- 主要コード、設定、Notebook、テスト名、生成物一覧
- `.env`の値を表示しない形で設定有無・対象ファイルの存在確認
- 標準WSL環境で`python -m unittest discover -s tests`
- 技術用語・データセット名の公式・学術情報確認

### 未実行・未確認

- IEMOCAP Session 1〜5の現在の実ファイル確認
- HCUDB 4,620 WAVの現在の実ファイル確認
- DS-001メタデータの実集計
- 実emotion2vec特徴の新規抽出
- 正式なモデル学習・推論・性能比較
- ignored checkpoint群の全metadata監査
