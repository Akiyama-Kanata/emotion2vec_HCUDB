# 日英SER研究の段階実施工数計画

> **現行計画の単一基準:** 本文書を本研究の唯一の現行計画とする。過去の作業ログ、進捗報告、READMEに含まれる作業順序や「次回作業」の記述は履歴情報であり、現行の判断には使用しない。

## 概要

- 研究の主目的は、エンコーダーを固定し、IEMOCAP学習後とHCUDB追加学習後の日英性能を比較することである。この主目的は条件Aで検証する。
- 条件Aを研究の必須実験とし、Aの完了後にB、Bの完了後にC、Cの完了後にDへ進む。B〜Dは主目的の一般性と比較の厚みを増す拡張実験とする。
- 条件Aは固定1 split・3 seedで実施する。BとDも実施時は同じ3 seedを使い、Cは学習せず固定評価する。5-foldはA完了後も必須化せず、全条件とは別の拡張に回す。
- Aの学習コード準備完了までは2〜4時間、実特徴抽出を含む正式な学習開始までは累計2.5〜5時間＋特徴抽出待ち、Aの1 seed完了までは累計6〜12時間＋計算待ち、3 seedと集計までの完了は累計10〜16時間＋計算待ちを標準見積もりとする。
- B〜Dの工数はAの実測を使って段階ごとに再見積もりする。以前の全作業を新規実装として計上した見積もりは廃止する。
- +largeのCPU特徴抽出時間は未計測のため、A完了後にBへ進む段階で実音声1件をベンチマークし、B以降の夜間実行日数を更新する。

## 実験条件と実施順

| 順序 | 条件 | 位置づけ | 分類部分 | 学習 |
|---:|---|---|---|---|
| 1 | A | 主目的を検証する必須実験 | emotion2vec＋新規decoder | IEMOCAP学習→HCUDB追加学習 |
| 2 | B | encoder規模を変えた優先拡張 | emotion2vec+ large＋Aと同一設計のdecoder | IEMOCAP学習→HCUDB追加学習 |
| 3 | C | 公式分類headによる参考評価 | +large公式9クラス分類head | 学習せず日英評価 |
| 4 | D | 公式`proj`の追加学習との拡張比較 | +large公式`proj` | 公式重みからHCUDBでhead-only追加学習 |

- Aの1 seedを最後まで通して修正した後、残り2 seedを実行し、前後差・図・再現情報まで集計してからBへ進む。
- Aが計算待ちであることだけを理由にB〜Dの実装を並行開始しない。Aの失敗や否定的結果も記録し、条件を変更して成功結果へ置き換えない。

- IEMOCAPはSession 5をtest、Session 1をvalidation、残りをtrainとする。HCUDBは話者が重複しない固定train/validation/test分割を使う。
- A・Bの出力は、両データセットの名前付き感情の和集合12クラスに固定する。IEMOCAPの`xxx`は感情名ではないため除外する。
- 学習時は全利用可能クラスを保持する。主比較はangry・happy・neutral・sadの4感情とし、IEMOCAPの`excited`は評価時のみhappyへ統合する。
- 共通4感情のtestサンプルだけを評価対象にするが、モデルのlogitはマスクしない。非共通クラスを予測した場合は誤りとして数える。
- DではHCUDBの「落ち着き」を公式`other`、「焦り」を`unknown`へ事前対応し、9感情すべてを学習に使用する。
- 英語維持の暫定基準は、HCUDB追加学習後の3-seed平均macro F1とUAが、追加学習前からそれぞれ2.0ポイントを超えて低下しないこととする。これは標準規格ではなく、本研究で結果取得前に固定する判定基準とする。

## 現在の資産と不足

| 対象 | 2026-08-11時点の確認結果 | Aでの扱い |
|---|---|---|
| 自動テスト | 76件成功を作業ログで確認済み | 変更後も全件成功を維持する |
| emotion2vec Base checkpoint | `artifacts/checkpoints/emotion2vec_base.pt`に存在 | Aの固定encoderとして使用する |
| IEMOCAP WAV | 現在確認できるのはSession 1配下の3,638 WAV | ユーザーがSession 2〜5を利用可能にした後、Codexが全sessionを検証する |
| IEMOCAP既存特徴 | Session 1の1,085件・4クラス・768次元だけが存在 | 配線確認には再利用できるが、Aの固定split・union 12学習結果には使用しない |
| IEMOCAP正解ラベル | 10,039行、`xxx`を含む元ラベルCSVが存在 | Codexが`xxx`除外とunion 12対応を実装する |
| HCUDB注釈 | 4,620行、14話者、11種類の演技感情を確認済み | Codexが列対応と固定話者splitを作る |
| HCUDB WAV | 4,620件すべて存在し、注釈の音声ファイル名と一致 | ユーザーによる追加準備は不要 |
| HCUDB特徴 | 研究用Base特徴cacheは未作成 | CodexがBase特徴を抽出して検証する |

IEMOCAPのSession 2〜5が利用可能になるまで、Codexはラベル契約、読込、共通decoder、checkpoint継続経路、合成データテストを先行して実装できる。実データ学習だけをデータ準備完了まで保留する。

## 作業分担

| 担当 | 作業 | 目的・完了の判断 |
|---|---|---|
| ユーザー | IEMOCAP Session 2〜5をローカルで利用可能にし、保存場所が既知の場所と異なる場合だけ絶対パスを伝える | Session 2〜4をtrain、Session 1をvalidation、Session 5をtestとして読める状態にする |
| ユーザー | 長時間の特徴抽出・学習を開始するときにPCを稼働可能にする | 計算待ち中の中断を避ける。CSV編集、コード修正、コマンド実行は不要 |
| Codex | union 12ラベル対応、`xxx`除外、評価時`excited`統合を実装・テストする | 日英で同じ出力契約を使用できる |
| Codex | IEMOCAP固定splitとHCUDB固定話者splitを生成・検証・保存する | session・話者漏洩、空split、未知ラベルがない |
| Codex | Aで共通使用するdecoderとcheckpoint継続経路を実装する | IEMOCAP checkpointを同じdecoderのままHCUDBへ引き継げる |
| Codex | Base特徴のベンチマーク、抽出、cache検証を実行する | 768次元・有限値・encoder識別一致を確認できる |
| Codex | 1 seed先行、残り2 seed、追加学習前後の日英評価と集計を実行する | Aの完了条件を満たす成果物を保存できる |

## 実装と成果物

- 新規パイプラインを最初から作らず、既存の実装を次のように再利用する。
  - `iemocap_downstream/notebook_pipeline.py`の固定session split、validation選択、test評価、checkpoint metadataを拡張する。
  - `vad_downstream/notebook_pipeline.py`の注釈検証、話者分割、特徴cache、可変クラス数datasetをHCUDBへ使用する。
  - `vad_downstream/train_parallel_emotion_vad.py`の初期checkpoint読込と継続学習処理を共通decoder経路へ再利用する。
  - `vad_downstream/parallel_training.py`のmacro F1、UA、confusion matrix、checkpoint保存を再利用する。
- AではBase特徴へ`encoder_id`、実測`input_dim`、checkpoint/revision、抽出粒度を渡し、cache manifestとcheckpointで一致を検証する。Bへ進む際に同じ契約をLargeへ適用する。
- A・Bは入力層の`in_features`だけを変え、以降のdecoder構造、unionラベル、optimizer、学習率、seed、split、モデル選択規則を共通化する。
- IEMOCAP checkpointを親としてHCUDB追加学習を開始し、`training_stage`、`parent_checkpoint`、seed、日英splitをmetadataへ保存する。
- Cは公式分類結果だけを保存する。Dは事前抽出した+large特徴を使い、公式`proj`以外をoptimizerへ渡さない。
- A・B・Dでは、該当する追加学習前後のmacro F1、UA、accuracy/WA、クラス別F1、confusion matrix、3-seed平均・標準偏差・対応するseed差を保存する。Cでは固定評価の同じ指標と予測結果を保存する。

## Aの直近工程と工数

| 順序 | Codexが行う作業 | 目的 | 実働見積もり | 開始条件・成果物 |
|---:|---|---|---:|---|
| 1 | union 12ラベル変換とIEMOCAP読込の一般化 | 現在の4クラス固定を外し、日英で同じ出力を使う | 1〜2h | データ準備と並行可。ラベル変換テスト |
| 2 | 共通decoder、継続checkpoint、固定splitの短縮テスト | IEMOCAPからHCUDBへ同じdecoderを引き継ぐ | 1〜2h | データ準備と並行可。合成データE2Eテスト |
|  | **学習コード準備完了まで** |  | **累計2〜4h** | **Session 2〜5の準備を待たずに完了可能** |
| 3 | 実音声1件のBase特徴抽出ベンチマークと全件抽出 | 学習入力を作り、計算時間と容量を確定する | 0.5〜1h＋計算待ち | shape・有限値・時間・容量記録、cache |
|  | **正式な学習開始まで** |  | **累計2.5〜5h＋特徴抽出待ち** | **Session 1〜5と実特徴が利用可能で、学習開始条件を満たす** |
| 4 | Aの1 seed先行実験、HCUDB追加学習、前後評価、修正 | Aを最小単位で最後まで通す | 累計6〜12h＋計算待ち | 1 seedの日英前後指標とcheckpoint |
| 5 | 残り2 seedと集計 | 再現性と英語維持判定を得る | 累計10〜16h＋計算待ち | 3-seed平均・標準偏差・対応差・図 |

- 上表の時間は2026-08-11に確認した既存実装を再利用する前提の実働時間である。データコピー、CPU特徴抽出、学習の待ち時間は含めない。
- Base特徴は一度抽出してAの全seedで再利用する。各抽出・学習の開始前に、予想終了時刻と必要空き容量を提示する。
- B〜Dを実施しない場合でも、Aの完了条件を満たせば研究の主目的は達成したものとする。
- BはA完了後にLargeの1 WAVベンチマークを行って工数を見積もる。CはB完了後、DはC完了後に個別見積もりを作る。

## テストゲートと完了条件

### Aの学習開始条件

- 現在の76テストを全件成功させ、追加したunion 12ラベル、固定split、checkpoint継続、Baseの768次元特徴、次元不一致、cache識別不一致テストも成功させる。非768次元の成功テストはB開始前へ回す。
- IEMOCAP Session 1〜5のWAV・ラベルが利用可能で、Session 2〜4=train、Session 1=validation、Session 5=testの各件数がゼロでないことを確認する。
- train/validation/test間の話者・session重複、`xxx`混入、未知ラベル、ゼロ件splitを学習前に拒否する。
- Base encoderをoptimizerへ渡さず、学習前後でencoderのparameter hashが不変であることを確認する。
- IEMOCAP checkpointとHCUDB追加学習設定の`encoder_id`、`input_dim`、ラベル、splitが一致し、不一致時は学習開始前に拒否する。
- Base特徴cacheの予測容量を算出し、現在の空き容量から20GB以上を残せない場合は全件抽出を開始しない。

### Aの完了条件

- AのIEMOCAP学習済みcheckpointと、それを親にしたHCUDB追加学習checkpointが3 seed分存在する。
- 追加学習前後の日英macro F1、UA、accuracy/WA、クラス別F1、confusion matrixを同じ固定splitで保存する。
- 3-seed平均・標準偏差・対応するseed差、英語維持判定、再現用metadataを保存する。
- 成功・否定的・不確定のいずれの結果でも、事前条件を変更せず結果として記録する。
- 以上が揃った時点で研究の主目的を達成とし、Bへ進める。B〜Dの未実施はAの完了判定を妨げない。

### B〜Dの開始・完了条件

- B開始前に非768次元・次元不一致・cache識別不一致をテストし、Base/Large decoderで入力層以外のparameter名・shapeが一致することを確認する。
- +largeの20秒以下の実音声1件をbatch size 1で処理し、特徴shape・有限値・処理時間・peak RAMを記録する。peak RAMが6.5GiBを超える、または20秒音声が10分以内に完了しない場合は、現PCでの全件抽出を非実用と判定してB以降を再見積もりする。
- Cではparameter更新がゼロであることを確認する。Dでは公式`proj`だけが変化し、encoderのparameter hashが不変であることを確認する。
- B、C、Dはそれぞれの表・指標・予測・metadataを保存した時点で個別に完了とする。A〜Dがすべて揃った状態は「全拡張実験完了」と呼び、研究の主目的達成とは区別する。

## 前提

- HCUDB1の4,620 WAVと注釈はローカルに存在し、対応を確認済みである。
- IEMOCAPは2026-08-11時点でSession 1だけを確認済みであり、ユーザーがSession 2〜5を用意する。用意後はCodexが件数、ラベル、session、音声形式を再検証する。
- 現PCはCore i7-1260P、WSL RAM約7.6GiB、CUDAなし、空き容量約103GBである。
- IEMOCAPの希少クラスは学習対象に残すが、話者分割を崩してまでtrainへ移動せず、主性能主張には使用しない。
- VAD媒介型、encoderの部分・全層学習、5-fold交差検証はAの10〜16時間に含めない。
- A完了前はB〜Dの実装・環境構築・実験を開始しない。条件間で再利用できる設計をAに採用しても、それ自体をB着手とは扱わない。
