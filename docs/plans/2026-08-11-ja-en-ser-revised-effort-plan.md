# 日英SER研究の段階実施工数計画

> **現行計画の単一基準:** 本文書を本研究の唯一の現行計画とする。過去の作業ログ、進捗報告、READMEに含まれる作業順序や「次回作業」の記述は履歴情報であり、現行の判断には使用しない。

## 概要

- 研究の主目的は、エンコーダーを固定し、IEMOCAP学習後とHCUDB追加学習後の日英性能を比較することである。この主目的は条件Aで検証する。
- 条件Aを研究の必須実験とし、Aの完了後にB、Bの完了後にC、Cの完了後にDへ進む。B〜Dは主目的の一般性と比較の厚みを増す拡張実験とする。
- 条件Aは固定1 split・3 seedで実施する。BとDも実施時は同じ3 seedを使い、Cは学習せず固定評価する。5-foldはA完了後も必須化せず、全条件とは別の拡張に回す。
- A完了までの標準工数は58時間、2026-08-11時点の残工数は53時間とする。A〜Dをすべて実施する場合は12週間、週約9〜10時間、実働合計108時間を標準見積もりとする。
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

## 実装と成果物

- AではBase特徴へ`encoder_id`、実測`input_dim`、checkpoint/revision、抽出粒度を渡し、cache manifestとcheckpointで一致を検証する。Bへ進む際に同じ契約をLargeへ適用する。
- A・Bは入力層の`in_features`だけを変え、以降のdecoder構造、unionラベル、optimizer、学習率、seed、split、モデル選択規則を共通化する。
- IEMOCAP checkpointを親としてHCUDB追加学習を開始し、`training_stage`、`parent_checkpoint`、seed、日英splitをmetadataへ保存する。
- Cは公式分類結果だけを保存する。Dは事前抽出した+large特徴を使い、公式`proj`以外をoptimizerへ渡さない。
- A・B・Dでは、該当する追加学習前後のmacro F1、UA、accuracy/WA、クラス別F1、confusion matrix、3-seed平均・標準偏差・対応するseed差を保存する。Cでは固定評価の同じ指標と予測結果を保存する。

## 工数と段階工程

| 段階 | 作業 | 標準工数 | 状態・開始条件 |
|---|---|---:|---|
| A | Notebook差分保全・76テストの基準更新 | 5h | 2026-08-11完了 |
| A | ラベル対応、固定split、評価規則の実装 | 12h | 次に着手 |
| A | HCUDB/IEMOCAPのmanifest・カラム整理 | 12h | ラベル契約確定後 |
| A | Base用cache識別・metadata・事前検証 | 6h | manifest整理後 |
| A | IEMOCAP→HCUDB継続学習経路 | 8h | 学習前検査成功後 |
| A | 1 seed先行実験と修正 | 6h | 継続学習経路完成後 |
| A | 残り2 seedの実行・監視 | 4h | 1 seed成功後 |
| A | 前後差、図、再現情報、英語維持判定 | 3h | 3 seed完了後 |
| A | 再実行・障害対応バッファ | 2h | 必要時 |
|  | **A小計（研究の主目的達成）** | **58h** | **残り53h** |
| B | 非768次元、共通cache契約、Large対応 | 6h | A完了後 |
| B | FunASR/+large環境、1 WAVベンチマーク、特徴抽出準備 | 8h | Large対応後 |
| B | Bの継続学習経路、3 seed、A/B比較、バッファ | 16h | ベンチマーク合格後 |
|  | **B追加小計** | **30h** | **累計88h** |
| C | 公式9クラスheadの固定日英評価 | 4h | B完了後 |
|  | **C追加小計** | **4h** | **累計92h** |
| D | 公式`proj`学習経路、3 seed、比較、バッファ | 16h | C完了後 |
|  | **D追加小計** | **16h** | **累計108h** |

- 工数は実働時間であり、CPU特徴抽出と学習の計算待ちは含めない。各段階の1 seed終了時に実測時間から当該段階の残工数を更新する。
- Base特徴とLarge特徴はそれぞれ一度抽出し、同一encoder・抽出契約の全seedで再利用する。
- B〜Dを実施しない場合でも、Aの完了条件を満たせば研究の主目的は達成したものとする。

## テストゲートと完了条件

### Aの学習開始条件

- 現在の76テストを全件成功させ、Baseの768次元特徴、次元不一致、cache識別不一致を追加テストする。非768次元の成功テストはB開始前へ回す。
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

- HCUDB1は4,620 WAV、IEMOCAPは3,638 WAVがローカルに存在する。
- 現PCはCore i7-1260P、WSL RAM約7.6GiB、CUDAなし、空き容量約103GBである。
- IEMOCAPの希少クラスは学習対象に残すが、話者分割を崩してまでtrainへ移動せず、主性能主張には使用しない。
- VAD媒介型、encoderの部分・全層学習、5-fold交差検証は今回の108時間に含めない。
- A完了前はB〜Dの実装・環境構築・実験を開始しない。条件間で再利用できる設計をAに採用しても、それ自体をB着手とは扱わない。
