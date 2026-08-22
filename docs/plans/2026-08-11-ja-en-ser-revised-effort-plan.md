# 日英SER研究の段階実施工数計画

> **2026-08-22に廃止:** 本文書はIEMOCAP主学習、HCUDB追加学習、旧ラベル設計を前提とする履歴資料である。現行判断には使用しない。新しい単一基準は[MSP-Podcast→HCUDB SER研究 現行実施計画](2026-08-22-msp-hcudb-feature-decoder-plan.md)とする。

> **主な変更:** 主学習をMSP-Podcast Release 1.10、外部英語testをIEMOCAP、出力を`anger / happy / sadness / disgust`へ変更し、特徴抽出とdecoder学習を別工程・別Notebookへ分離した。

> **実験条件の状態（2026-08-21）:** 未確定なのは感情ラベルの扱いだけである。学習・評価に使うラベル集合、`xxx`の扱い、`excited`の統合、日英の対応関係、Dのラベル対応を決定してから正式実験を開始する。条件A〜D、データ分割、seed、decoder構成、評価指標・集計方法、判定基準、実施順は現行条件として確定している。

> **共通4感情の出力表記（2026-08-22決定）:** 正規化後のクラス名と出力順は`anger / happy / sadness / neutral`に固定する。元データのラベル名・略号は上書きせず、変換前ラベルとして保持する。

## 概要

- 研究の主目的は、エンコーダーを固定し、IEMOCAP学習後とHCUDB追加学習後の日英性能を比較することである。この主目的は条件Aで検証する。
- 条件Aを研究の必須実験とし、Aの完了後にB、Bの完了後にC、Cの完了後にDへ進む。B〜Dは主目的の一般性と比較の厚みを増す拡張実験とする。
- 条件AはIEMOCAPのsession独立5-fold・各fold 3 seed（42、43、44）で実施し、合計15系列を評価する。各foldでは1 sessionをtest、循環順で次の1 sessionをvalidation、残り3 sessionをtrainとする。Bも同じ5-fold・3 seedを使う。Cは学習せず固定評価し、DはHCUDBの固定話者splitで3 seedを使う。
- Aの学習コード準備完了までは2〜4時間、実特徴抽出を含む正式な学習開始までは累計2.5〜5時間＋特徴抽出待ちという従来見積もりを暫定維持する。正式学習系列は15系列であり、最初の1系列完了後に残り14系列を再見積もりする。特徴抽出はfold・seed間で再利用する。
- B〜Dの工数はAの実測を使って段階ごとに再見積もりする。以前の全作業を新規実装として計上した見積もりは廃止する。
- +largeのCPU特徴抽出時間は未計測のため、A完了後にBへ進む段階で実音声1件をベンチマークし、B以降の夜間実行日数を更新する。

## 実験条件と実施順

この節のうち、感情ラベルに関する箇所だけが未確定である。それ以外は現行の確定条件とする。

| 順序 | 条件 | 位置づけ | 分類部分 | 学習 |
|---:|---|---|---|---|
| 1 | A | 主目的を検証する必須実験 | emotion2vec＋新規decoder | IEMOCAP学習→HCUDB追加学習 |
| 2 | B | encoder規模を変えた優先拡張 | emotion2vec+ large＋Large特徴次元に合わせた別decoder | IEMOCAP学習→HCUDB追加学習 |
| 3 | C | 公式分類headによる参考評価 | +large公式9クラス分類head | 学習せず日英評価 |
| 4 | D | 公式`proj`の追加学習との拡張比較 | +large公式`proj` | 公式重みからHCUDBでhead-only追加学習 |

- Aは既存実装に近いFold 5・seed 42の1系列を最初に最後まで通して修正し、その後に残り4 foldと各foldの残りseedを実行する。15系列の前後差・図・再現情報まで集計してからBへ進む。
- Aが計算待ちであることだけを理由にB〜Dの実装を並行開始しない。Aの失敗や否定的結果も記録し、条件を変更して成功結果へ置き換えない。

- IEMOCAPは次のsession独立5-foldを使用する。各sessionはtestとvalidationにそれぞれ1回だけ使い、同じfoldのtrain / validation / test間でsessionを重複させない。

| Fold | Train | Validation | Test |
|---:|---|---|---|
| 1 | Session 3・4・5 | Session 2 | Session 1 |
| 2 | Session 4・5・1 | Session 3 | Session 2 |
| 3 | Session 5・1・2 | Session 4 | Session 3 |
| 4 | Session 1・2・3 | Session 5 | Session 4 |
| 5 | Session 2・3・4 | Session 1 | Session 5 |

- HCUDBは14話者をtrain 10話者、validation 2話者、test 2話者へ固定し、同一話者を複数splitへ入れない。話者割当は結果を見る前に確定してmanifestへ保存し、IEMOCAPの全fold・seedで同じHCUDB splitを使う。
- A・Bのseedは42、43、44に固定し、fold、seed、split manifestを結果取得前に実験設定へ記録する。

### 感情ラベルの扱い（未定）

- A・Bの出力候補は、両データセットの名前付き感情の和集合12クラスとする。ラベル集合と`xxx`の扱いは未確定であり、実装前に再検討する。
- 学習時に全利用可能クラスを保持し、`anger / happy / sadness / neutral`の4感情を主比較として、IEMOCAPの`excited`を評価時のみ`happy`へ統合する案を置く。クラス名と出力順は確定済みだが、学習対象と統合時点は未確定である。
- 共通4感情のtestサンプルだけを評価対象とし、モデルのlogitをマスクせず、非共通クラスの予測を誤りと数える案を置く。この評価処理も未確定である。
- DではHCUDBの「落ち着き」を公式`other`、「焦り」を`unknown`へ事前対応し、9感情すべてを学習に使用する案を置く。対応関係と学習対象は未確定である。

採用した評価ラベル集合について、各fold・seedで`追加学習後 − 追加学習前`のmacro F1差とUA差を求める。fold内で3 seedを平均し、そのfold平均を5-foldで平均した値が、macro F1とUAの両方で`-2.0`ポイント以上なら英語性能を維持したと判定する。これは標準規格ではなく、本研究で結果取得前に固定する判定基準とする。

## 現在の資産と不足

| 対象 | 2026-08-11時点の確認結果 | Aでの扱い |
|---|---|---|
| 自動テスト | 76件成功を作業ログで確認済み | 変更後も全件成功を維持する |
| emotion2vec Base checkpoint | `artifacts/checkpoints/emotion2vec_base.pt`に存在 | Aの固定encoderとして使用する |
| IEMOCAP WAV | 現在確認できるのはSession 1配下の3,638 WAV | ユーザーがSession 2〜5を利用可能にした後、Codexが全sessionを検証する |
| IEMOCAP既存特徴 | Session 1の1,085件・4クラス・768次元だけが存在 | 配線確認には再利用できるが、確定後のラベル設計と合わない場合は正式結果に使用しない |
| IEMOCAP正解ラベル | 10,039行、`xxx`を含む元ラベルCSVが存在 | `xxx`とラベル集合の扱いを条件検討で決める |
| HCUDB注釈 | 4,620行、14話者、11種類の演技感情を確認済み | ラベル列の対応を決め、固定10/2/2話者splitを作る |
| HCUDB WAV | 4,620件すべて存在し、注釈の音声ファイル名と一致 | ユーザーによる追加準備は不要 |
| HCUDB特徴 | 研究用Base特徴cacheは未作成 | CodexがBase特徴を抽出して検証する |

IEMOCAPのSession 2〜5が利用可能になるまで、Codexはデータ読込、5-fold、固定話者split、checkpoint継続経路、合成データテストを先行して実装できる。decoderの出力次元とラベル変換は、感情ラベルの扱いを決定してから確定する。実データ学習はラベル決定とデータ準備の両方が完了するまで保留する。

## 作業分担

| 担当 | 作業 | 目的・完了の判断 |
|---|---|---|
| ユーザー | IEMOCAP Session 2〜5をローカルで利用可能にし、保存場所が既知の場所と異なる場合だけ絶対パスを伝える | Session 1〜5の全foldを生成できる状態にする |
| ユーザー | 長時間の特徴抽出・学習を開始するときにPCを稼働可能にする | 計算待ち中の中断を避ける。CSV編集、コード修正、コマンド実行は不要 |
| Codex | 確定したラベル集合、除外・統合規則を実装・テストする | 日英で確定済みの出力契約を使用できる |
| Codex | IEMOCAP 5-foldとHCUDB固定10/2/2話者splitを生成・検証・保存する | 各IEMOCAP sessionがtest・validationに1回ずつ現れ、session・話者漏洩、空split、未知ラベルがない |
| Codex | 決定したラベル数に対応するdecoderとcheckpoint継続経路を実装する | IEMOCAP checkpointを同じdecoderのままHCUDBへ引き継げる |
| Codex | Base特徴のベンチマーク、抽出、cache検証を実行する | 768次元・有限値・encoder識別一致を確認できる |
| Codex | Fold 5・seed 42を先行し、残り14系列、追加学習前後の日英評価と階層集計を実行する | Aの完了条件を満たす成果物を保存できる |

## 実装と成果物

ラベル数とラベル対応に依存する箇所だけは、感情ラベルの扱いを決定後に更新する。

- 新規パイプラインを最初から作らず、既存の実装を次のように再利用する。
  - `iemocap_downstream/notebook_pipeline.py`の固定session split、validation選択、test評価、checkpoint metadataを拡張する。
  - `vad_downstream/notebook_pipeline.py`の注釈検証、話者分割、特徴cache、可変クラス数datasetをHCUDBへ使用する。
  - `vad_downstream/train_parallel_emotion_vad.py`の初期checkpoint読込と継続学習処理を共通decoder経路へ再利用する。
  - `vad_downstream/parallel_training.py`のmacro F1、UA、confusion matrix、checkpoint保存を再利用する。
- AではBase特徴へ`encoder_id`、実測`input_dim`、checkpoint/revision、抽出粒度を渡し、cache manifestとcheckpointで一致を検証する。Bへ進む際に同じ契約をLargeへ適用する。
- A用decoderとB用decoderは、encoderの出力次元に応じて入力層の`in_features`と重みshapeが異なる別decoderとする。比較条件をそろえるため、入力層より後段の層構成、optimizer、学習率、seed、split、モデル選択規則を対応させる。出力層は決定した感情ラベル数に合わせる。
- 各IEMOCAP fold・seedのcheckpointを親としてHCUDB追加学習を開始し、`training_stage`、`parent_checkpoint`、fold、seed、IEMOCAP split、HCUDB話者splitをmetadataへ保存する。
- Cは公式分類結果だけを保存する。Dは事前抽出した+large特徴を使い、公式`proj`以外をoptimizerへ渡さない。
- A・Bでは、fold・seedごとの追加学習前後のmacro F1、UA、accuracy/WA、クラス別F1、confusion matrix、対応差を保存する。集計は各fold内の3 seed、その後5-foldの順で行い、15系列を独立標本として扱わない。Dは固定HCUDB話者splitの3 seed、Cは固定評価の同じ指標と予測結果を保存する。クラス別出力の対象だけは、感情ラベル決定後に確定する。

## 実験開始までの残作業

| 順序 | 残作業 | 現時点の状態 | 実験へ進める完了条件 |
|---:|---|---|---|
| 1 | 感情ラベルの扱いを決定 | ラベル集合、`xxx`、`excited`、日英対応、Dの対応だけが未確定 | 採用案と不採用案、決定理由、ラベル対応表、学習対象、評価対象、統合・除外規則を保存する |
| 2 | 条件Aの実装完了 | ラベル依存部分以外は先行実装可能 | 決定したラベル処理、5-fold × 3-seed実行、checkpoint継続、metadata保存、階層集計を実装する |
| 3 | 合成データE2Eテスト | 実装と同時に追加する | IEMOCAP学習→checkpoint保存→HCUDB追加学習→前後評価→fold内・fold間集計が短縮設定で最後まで通る |
| 4 | 実データの接続 | HCUDBは接続可能、IEMOCAPはSession 2〜5待ち | `IEMOCAP_ROOT`、`IEMOCAP_WORK_DIR`、`EMOTION2VEC_CHECKPOINT`、`EMOTION2VEC_USER_DIR`とHCUDB音声rootが実在パスを指し、全音声・注釈を読める |
| 5 | 学習前検証 | 実装・データ接続後に実施 | split漏洩なし、全fold非空、未知ラベルなし、encoder hash不変、特徴shape・有限値・ID一致、全テスト成功、抽出後も20GB以上の空き容量を満たす |
| 6 | 実音声ベンチマークとcache作成 | 未実施 | 実音声1件で所要時間・shape・有限値を確認し、manifest付きBase特徴cacheを作成・検証する |
| 7 | Fold 5・seed 42の先行実験 | 未実施 | IEMOCAP学習、HCUDB追加学習、日英前後評価、2段階checkpoint、metadataが1系列分そろう |
| 8 | 正式な残り14系列 | 先行実験の確認後に開始 | 先行実験後にコード・条件を変更しない状態で実行し、15系列を階層集計できる |

実験開始までの最短順序は、感情ラベル決定→条件A実装→合成データE2E→実データ接続→実音声1件ベンチマーク→特徴cache作成→Fold 5・seed 42先行実験→残り14系列とする。ラベル決定前は、ラベルに依存しない共通部の整備と試験を先行できる。先行実験後にコード、ラベル対応、split、seed、モデル選択規則、主要ハイパーパラメータのいずれかを変更した場合、その先行結果は正式集計へ含めず、設定固定後に同系列を再実行する。正式実験へ移る時点で、設定ファイル、split manifest、ラベル対応表、使用checkpointの識別情報、コードのcommit hashまたは同等の版識別子を保存する。

## Aの直近工程と工数

| 順序 | Codexが行う作業 | 目的 | 実働見積もり | 開始条件・成果物 |
|---:|---|---|---:|---|
| 1 | 決定した感情ラベル変換とIEMOCAP読込の一般化 | 現在の4クラス固定を外し、決定した出力契約を日英で使う | 1〜2h | ラベル決定後に着手。ラベル変換テスト |
| 2 | 共通decoder、継続checkpoint、IEMOCAP 5-fold、HCUDB固定話者splitの短縮テスト | IEMOCAPからHCUDBへ同じdecoderを引き継ぐ | 1〜2h | データ準備と並行可。合成データE2Eテスト |
|  | **学習コード準備完了まで** |  | **累計2〜4h** | **Session 2〜5の準備を待たずに完了可能** |
| 3 | 実音声1件のBase特徴抽出ベンチマークと全件抽出 | 学習入力を作り、計算時間と容量を確定する | 0.5〜1h＋計算待ち | shape・有限値・時間・容量記録、cache |
|  | **正式な学習開始まで** |  | **累計2.5〜5h＋特徴抽出待ち** | **Session 1〜5と実特徴が利用可能で、学習開始条件を満たす** |
| 4 | Fold 5・seed 42の先行実験、HCUDB追加学習、前後評価、修正 | Aを最小系列で最後まで通す | 累計6〜12h＋計算待ち | 1系列の日英前後指標と2段階checkpoint |
| 5 | 残り14系列と階層集計 | seed変動とsession変動を分離して英語維持判定を得る | 先行系列の実測後に再見積もり | fold内3-seed集計、5-fold集計、対応差・図 |

- 上表の先行系列までの時間は2026-08-11に確認した既存実装を再利用する前提の暫定値である。データコピー、CPU特徴抽出、学習の待ち時間は含めない。残り14系列は先行系列の実測時間・失敗率・保存容量から再見積もりする。
- Base特徴は一度抽出してAの全seedで再利用する。各抽出・学習の開始前に、予想終了時刻と必要空き容量を提示する。
- B〜Dを実施しない場合でも、Aの完了条件を満たせば研究の主目的は達成したものとする。
- BはA完了後にLargeの1 WAVベンチマークを行って工数を見積もる。CはB完了後、DはC完了後に個別見積もりを作る。

## テストゲートと完了条件

### Aの学習開始条件

- 感情ラベルの扱いが決定され、ラベル集合、`xxx`、`excited`、日英対応、Dの対応、学習対象、評価対象、統合・除外規則がラベル対応表へ保存されている。
- seed 42・43・44、IEMOCAP 5-fold manifest、HCUDB 10/2/2話者split manifest、モデル選択規則、主要ハイパーパラメータが結果取得前に保存されている。
- `IEMOCAP_ROOT`、`IEMOCAP_WORK_DIR`、`EMOTION2VEC_CHECKPOINT`、`EMOTION2VEC_USER_DIR`とHCUDB音声rootが実在パスを指し、実行設定から一意に追跡できる。
- 現在の76テストを全件成功させ、決定した感情ラベル処理、IEMOCAP 5-fold、HCUDB固定話者split、checkpoint継続、Baseの768次元特徴、次元不一致、cache識別不一致テストも成功させる。非768次元の成功テストはB開始前へ回す。
- IEMOCAP Session 1〜5のWAV・ラベルが利用可能で、上表の全5-foldについてtrain / validation / testの各件数がゼロでないことを確認する。
- HCUDBの14話者がtrain 10、validation 2、test 2へ一意に割り当てられ、話者集合が重複せず、全splitの件数がゼロでないことを確認する。
- train/validation/test間の話者・session重複、決定したラベル規則への違反、未知ラベル、ゼロ件splitを学習前に拒否する。
- Base encoderをoptimizerへ渡さず、学習前後でencoderのparameter hashが不変であることを確認する。
- IEMOCAP checkpointとHCUDB追加学習設定の`encoder_id`、`input_dim`、ラベル、splitが一致し、不一致時は学習開始前に拒否する。
- Base特徴cacheの予測容量を算出し、現在の空き容量から20GB以上を残せない場合は全件抽出を開始しない。
- 合成データE2Eテストと実音声1件の特徴抽出ベンチマークを完了し、実行設定、split manifest、ラベル対応表、checkpoint識別情報、コード版識別子を保存する。

### Aの完了条件

- AのIEMOCAP学習済みcheckpointと、それを親にしたHCUDB追加学習checkpointが5-fold × 3 seedの15組存在する。
- 各fold・seedについて、追加学習前後の日英macro F1、UA、accuracy/WA、クラス別F1、confusion matrixを、同じIEMOCAP test sessionと同じHCUDB test話者で保存する。
- fold内3-seed平均・標準偏差、5-fold平均・標準偏差、各fold・seedの対応差、英語維持判定、再現用metadataを保存する。seedはfold内変動として扱い、15系列を独立標本として検定しない。
- 成功・否定的・不確定のいずれの結果でも、事前条件を変更せず結果として記録する。
- 以上が揃った時点で研究の主目的を達成とし、Bへ進める。B〜Dの未実施はAの完了判定を妨げない。

### B〜Dの開始・完了条件

- B開始前に非768次元・次元不一致・cache識別不一致をテストする。Base用とLarge用は別decoderであることを前提に、入力層より後段のparameter名・shapeが対応し、比較対象外の設計差が混入していないことを確認する。BもAと同じIEMOCAP 5-fold、HCUDB話者split、3 seedを使う。
- +largeの20秒以下の実音声1件をbatch size 1で処理し、特徴shape・有限値・処理時間・peak RAMを記録する。peak RAMが6.5GiBを超える、または20秒音声が10分以内に完了しない場合は、現PCでの全件抽出を非実用と判定してB以降を再見積もりする。
- Cではparameter更新がゼロであることを確認する。Dでは公式`proj`だけが変化し、encoderのparameter hashが不変であることを確認する。
- B、C、Dはそれぞれの表・指標・予測・metadataを保存した時点で個別に完了とする。A〜Dがすべて揃った状態は「全拡張実験完了」と呼び、研究の主目的達成とは区別する。

## 前提

- HCUDB1の4,620 WAVと注釈はローカルに存在し、対応を確認済みである。
- IEMOCAPは2026-08-11時点でSession 1だけを確認済みであり、ユーザーがSession 2〜5を用意する。用意後はCodexが件数、ラベル、session、音声形式を再検証する。
- 現PCはCore i7-1260P、WSL RAM約7.6GiB、CUDAなし、空き容量約103GBである。
- IEMOCAPの希少クラスを学習・評価へ含めるかは、感情ラベルの扱いとして決定する。どの決定でもsession分割は崩さない。
- VAD媒介型、encoderの部分・全層学習、HCUDB側の話者cross-validation、IEMOCAPとHCUDBのfold全組合せはAに含めない。
- A完了前はB〜Dの実装・環境構築・実験を開始しない。条件間で再利用できる設計をAに採用しても、それ自体をB着手とは扱わない。
