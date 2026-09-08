# emotion2vec+ large C/D実験 現行計画・実装仕様

作成日: 2026-08-22（Asia/Tokyo）
改訂日: 2026-09-07（Asia/Tokyo）

> **現行計画の単一基準:** 第1〜11節と第17〜22節をC/D実験の現行指示とする。2026-09-07に提示された「emotion2vec+ large C/D実験 計画・実装仕様」を反映した。第12〜16節は当時の実施記録として保持し、固定split・固定除外契約を引き継ぐ。旧decoder・IEMOCAP評価・旧クラス順はC/Dへ適用しない。

> **今回の作業は文書のみ。** ローカルコードを調査して計画を具体化したが、コード・API・Notebook・実行用configは変更していない。checkpoint取得・推論・特徴抽出・学習・本評価は未実施。新規ファイル名、schema名、API案は将来実装の予定であり、実装済みを意味しない。

実装計画の再開時は[Plan mode入力用プロンプト](2026-08-22-plan-mode-implementation-prompt.md)を使用する。

## 1. 目的とC/Dの定義

emotion2vec+ largeの固定encoder表現に対し、公式線形分類層 `proj` をHCUDBへ適応することで、HCUDBとMSP-Podcastの共通4感情の分類性能がどう変化するかを調べる。

| 条件 | 初期状態 | 学習 | 評価 |
|---|---|---|---|
| C | 公式checkpointのencoderと9出力proj | 追加学習なし | 共通4 logitsによる4-class評価 |
| D | Cと同じ公式checkpoint・同じproj初期値 | encoder固定、HCUDB trainによる4-logit CEでprojのみ適応 | Cと同じ4-class評価 |

C/Dでcheckpoint、encoder、test utterance集合、ラベル対応、前処理、特徴抽出、pooling、評価規則を一致させる。差はHCUDBによるproj適応の有無だけにする。Cは公式9-way分類器の9-way性能を評価する条件ではない。

研究上の比較単位は **A ↔ B、C ↔ D**。以前のA/Bとの絶対性能の直接比較を目的にC/Dを変更しない。A/Bの定義を今回再定義せず、既存Base特徴cacheと新規4-class decoderはC/Dへ流用しない。A/Bの完了はC/Dの開始条件にしない。

## 2. 共通クラスと9→4 logit選択

C/Dの統一クラス順を **anger / disgust / happy / sadness** に固定する。ローカルの `ser_pipeline/config/mappings.v1.json` で確認した元ラベル対応を引き継ぐ。

| C/D index | 統一クラス | 公式出力名 | MSP元ラベル | HCUDB元ラベル |
|---|---|---|---|---|
| 0 | anger | angry | A | 怒り |
| 1 | disgust | disgusted | D | 嫌い |
| 2 | happy | happy | H | 狂喜・楽しい、余裕・嬉しい |
| 3 | sadness | sad | S | 憂鬱・悲しい |

`original_emotion`、`mapped_emotion`、`mapping_version`と近似対応フラグを保持する。「嫌い→disgust」は厳密な同義でなく近似対応としてversion管理する。

READMEの9出力順は `angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown`。この順なら選択indexは `[0, 1, 3, 6]` だが、**実checkpoint/configで検証するまで確定値として実行に使わない**。公式9出力順とC/Dの4クラス順は別フィールドに保存する。

C/D共通処理:

```text
logits9 = official_proj(official_utterance_features)
logits4 = logits9[:, verified_selected_indices]
prediction = argmax(logits4)
probabilities4 = softmax(logits4)
D loss = cross_entropy(logits4, target4)
```

- 9-way softmax後に5クラスを削除する方式を標準処理にしない。
- fearful / neutral / other / surprised / unknownは9出力として保持し、4-class予測候補とlossから除外する。
- 対象外logitが9出力中最大でも、発話を除外せず共通4 logits内で予測する。
- 主評価にOUTSIDE列や9-way正解率を追加しない。
- 同値最大値は固定4クラス順で最初のindexを採用し、contractへ記録する。

**旧indexとの互換性:** 現行コードの `LABEL_ORDER` は `anger / happy / sadness / disgust`。既存manifest/cacheの整数 `class_index` をC/D targetとして直読しない。元manifestのラベル・versionを検証後、`mapped_emotion` からC/D indexを生成する。元manifestの内容・SHAを保持し、C/Dのラベルviewとsignatureを別に保存する。global定数の一括変更で旧結果の意味を変えない。

## 3. データセット・固定split・除外契約

| データセット | C/Dでの用途 | 固定条件 |
|---|---|---|
| MSP-Podcast R1.10 | 両条件のtestのみ。Dの学習・validationには使わない | 事前固定したTest1利用可能部分集合 |
| HCUDB1 | Dのtrain・validation、両条件のtest | 話者非重複の10 / 2 / 2 split |

HCUDBは `hcudb1_speaker_split_v1` を維持する。

- train: `FA, FB, FD, FH, FI, FL, MC, MJ, MM, MN`
- validation: `FF, MK`
- test: `FG, ME`

学習seedを変えてもsplitを再生成しない。C/Dで同じHCUDB test発話を使う。

MSPの公式Train / Development / Test1割当てと既知話者条件を保持し、Test2・SpkrID=Unknownを対象外とする。C/D用にMSP train/validationの学習や全件cacheを必須としない。第16節の欠損・0バイト1,128件、うちTest1 330件の固定除外契約を引き継ぎ、復旧音声を自動採用しない。

現行コードには音声重複の除外契約もある。使用予定の既存manifestと承認済み除外契約のSHAを照合し、重複除外が既に採用されている場合はその集合を含めて固定する。歴史的な24,857件という全split件数だけから現在のtest集合を再構成しない。manifest実体とsignatureが未確認なら本評価を開始しない。結果に応じた新たな除外・再splitは行わない。

MSP結果は「MSP-Podcast R1.10の事前固定したTest1利用可能部分集合上の性能」と記す。既存研究記録のemotion2vec事前学習におけるMSP v1.8使用と、今回のLarge checkpoint・具体的test発話の重複確認を区別し、「完全未知英語データ」と表現しない。

## 4. 公式checkpoint・前処理・parity test

公式モデルID候補はREADMEにある `iic/emotion2vec_plus_large`。公式FunASRのAutoModel経路を調べ、hub、解決されたmodel revision、実ファイル、SHA-256、依存versionを固定する。取得済みlocal snapshotを明示して再読込できる設計とし、可変のlatest識別子だけで再現しない。

- 実configとモデル実体から9出力順、projの所在・weight/bias・入力次元・出力数を確認する。
- headの不一致やキー欠損をrandom初期化・緩いstate読込で埋めない。
- 音声読込、sample rate、resampling、mono処理、waveform normalization、encoder出力、余分なtoken、padding、temporal pooling、proj入力を公式経路と照合する。
- 既存fairseq Base経路、768次元、final_after_encoder_normをLargeへ推測で適用しない。
- 公式経路のsoftmax前の9 logitsを取得する。公開APIがscoresだけを返す場合は、公式forward/projを計測する。確率の対数を生logitsとして代用しない。

本番C前に10〜20発話で以下を比較する。発話IDと選定規則は性能を見る前に固定し、両データセット、短長の違い、元sample rateの違いを含める。

```text
公式FunASR直接推論
vs
自前前処理 → 同一encoder → cache書込・再読込 → 公式と同じ集約 → 同一proj
```

waveform、encoder output、pooling、proj入力、9 logits、選択4 logits、4-way argmaxを確認する。max absolute difference、mean absolute difference、argmax一致率、device/dtype、batch/padding条件、比較対象SHAを `parity_report.json` に保存する予定とする。

atol/rtolは実行環境を確認して事前固定する。4-way argmax一致率100%と全9 logitsの許容誤差内一致を本評価の必須条件とする。不一致時は原因を特定し、結果に合わせた閾値緩和や不一致発話の削除で通過させない。前処理・revision・dtype等を変えた場合は再検証する。

## 5. Large専用cache

公式proj入力に必要な特徴を新規抽出する。既存の再開可能shard、index、hash検査、mmapを再利用する設計とする。まず公式encoderのフレーム表現を保存し、公式と同じpoolingを共通関数で行う案を採る。実モデルが別の入力形式を要求する場合はparity確認に基づいて設計を確定する。

予定schemaは `ser_large_feature_cache_v1`。最低限以下を記録する。

- model identifier、hub、revision、checkpoint SHA-256、公式実装・依存version
- preprocessing version、sample rate、normalization設定、resampling実装
- encoder出力位置、token/padding処理、pooling方法、feature dimension、dtype
- dataset、split、utterance ID、source audio identity/SHA-256、shape、shard/offset
- source manifest SHA、元mapping/split version、固定除外・重複除外signature
- extraction code version、git commit、cache schema version、cache ID
- 対応するparity reportの参照・SHA・合格状態

不完全shard、欠損、重複ID、非有限値、0フレーム、次元不一致、音声hashやmanifest不一致は拒否する。metadata不足をBase既定値で補完しない。完全性検査とparityを通過した同じLarge cacheをC/DとDの3 seedで使用する。

## 6. C評価とD学習

### C

checkpointとcontractを検証し、model全体をeval状態にする。optimizerを作らず、勾配なしで共通4-class評価を行う。両testの結果・個別予測・9 logits・再現情報を保存し、両方の保存完了をD正式実行の開始条件にする。評価前後のstate不変を検証する。

### D

毎seed、Cと同じ公式proj初期値から開始する。別seedの学習済みheadを親にしない。encoderはrequires_grad=False、eval、勾配なしで固定し、optimizerへ渡すparameterをprojだけに限定する。cache学習ではencoderを学習loopに載せない。

lossは選択4 logitsへのCross Entropyとする。初期実装案はクラス重みなし・label smoothingなし。別設定を導入する場合もHCUDB validationだけで選び、設定と探索範囲を保存する。MSPとHCUDB test、Cのtest性能を設定選択に使わない。

projは9-output Linearのまま保持し、対象外5行のweightとbiasが不変であることを保証する。**lossの勾配が0という確認だけで更新不変を保証しない。** 既存TrainingConfigのweight_decay=1e-4とAdamWをそのまま流用しない。単純な初期実装案は `AdamW(proj.parameters(), weight_decay=0)` と新規optimizer状態を使用し、対象外5行の勾配・weight/bias・optimizer状態を検証する。Dの適合checkpointからのresumeだけを許可する。

対象外行にweight decayや過去のmomentumが作用する構成は拒否する。必要なら4行のみの明示更新へ変更するが、まず上記の単純な実装を検証する。

| 技術主張の検証（claim-verify、2026-09-07） | 判定 | 根拠・計画への反映 |
|---|---|---|
| AdamWには勾配更新とは別のweight decay処理がある | 確認済み | [PyTorch公式AdamW仕様](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)の更新式を確認 |
| 9行一体のparameterで対象外行のCE勾配を0にしても、非ゼロweight decayによる変化は防げない | 上記更新式からの帰結 | step後のweight/bias不変をテストする |

model selectionはHCUDB validationのUAR最大→macro F1最大→4-logit CE最小の順を初期案とし、完全同値なら先のepochを保持する。validation lossは全発話の非加重CE平均として定義し、旧保存確率からのclip付きlossとの違いを混同しない。lr、batch size、最大epoch、early stoppingのpatience/min_delta、scheduler有無は本評価前に設定へ具体化する。test実行時点で選択checkpointを固定する。

## 7. 評価指標・個別予測・再現情報

共通4クラスすべてを固定順で集計する。accuracy（WA）、macro F1、UAR（4クラスrecall平均）、クラス別precision/recall/F1/supportを保存する。ゼロ除算・support 0のクラス指標は0とし、macroの分母は4のままとする。空testは拒否する。

混同行列は4×4、行=true、列=prediction、順序はanger / disgust / happy / sadness。OUTSIDE列は作らない。値は0–1で保存し、パーセント表示時のD−Cはpercentage pointとして表記する。

発話単位の最低保存項目:

- utterance ID、dataset、speaker ID、元ラベル、統一4-classラベル・index
- 生9 logits（公式順）、選択4 logits（C/D順）、4-way probabilities
- 4-way predicted label/index、correctness、mapping/contractへの参照

9出力の追加診断は主評価と分ける。保存する9-way確率を生logitsの代わりにしない。任意の3-class sensitivity analysisはanger / happy / sadnessの対象発話と3 logitsを使用する別contract・別出力とし、主4-class結果を書き換えない。

各runにrun ID、C/D、seed（Cは学習seedなし）、git commit・dirty状態/差分識別、config、親公式checkpoint hash、選択D checkpoint hash、manifest hash、train/val/test signature、mapping version、evaluation contract version/hash、cache version/ID、optimizer・lr・batch size・epoch・early stopping・選択規則、最終metrics・個別予測を保存する。

## 8. 実装前調査で確認した既存構成

2026-09-07にローカルコード・configを読んで確認した。以下は実験の動作確認ではない。

| 既存ファイル | 現状 | C/Dでの利用・必要変更 |
|---|---|---|
| ser_pipeline/readers.py, splits.py, exclusions.py, duplicates.py | データ読込・固定split・欠損/重複除外 | 共通基盤を利用。採用manifest/除外SHAを先に確認 |
| ser_pipeline/manifest.py | JSONL、hash、元ラベル、旧index検証 | 元検証を維持、C/D label view/signatureを追加 |
| ser_pipeline/contracts.py, config/mappings.v1.json | 旧順anger / happy / sadness / disgustに固定 | 旧定数を維持してC/D contractを追加 |
| ser_pipeline/audio.py | soundfile、mono要求、scipy resample_poly、16 kHz | hash/検査を利用。Large前処理としての採用は公式照合後 |
| ser_pipeline/features.py | fairseq Base adapter、既定768次元、frame特徴 | shard抽出基盤を拡張。LargeのFunASR読込は独立adapter |
| ser_pipeline/cache.py | shard/index、mmap、signature、完全性検査 | Large schema・前処理/pooling/parity情報を追加 |
| ser_pipeline/model.py | ReLUを含む2層BaseModel | C/D headに使用しない。旧経路を維持 |
| ser_pipeline/training.py | BaseModel構築、MSP親→HCUDB子、AdamW | seed/device等を再利用。公式head学習は別入口を追加 |
| ser_pipeline/evaluation.py | 旧LABEL_ORDERに依存、確率中心の保存 | class order明示の共通評価、9/4 logits保存を追加 |
| ser_pipeline/checkpoints.py | BaseModel signature、msp_train/hcudb_continue stage | 公式9-output head schema・親/resume検証を追加 |
| ser_pipeline/study.py | transfer study、署名照合・集計 | C固定値→D 3 seedの入口を追加 |
| ser_pipeline/cli.py, notebook_api.py, scripts/build_ser_notebooks.py | Notebook 01/02とdecoder経路 | C/parity/D/比較入口とNotebook境界を追加 |
| scripts/summarize_ser_saved_results.py | 旧saved-resultパス・weighting比較に依存 | C/D contract照合と集計を追加、旧結果と混在させない |

artifacts/checkpoints/で確認できたのはemotion2vec_base.pt。今回の調査範囲でLarge checkpoint実体・config・公式実装は確認できていない。READMEのclass orderは実checkpoint照合済みの証拠ではない。Largeの入力次元・前処理・pooling・正確な出力indexを確定済みと記さない。

## 9. Notebook・実行境界

- notebooks/01_extract_emotion2vec_features.ipynb: データ設定、manifest/contract検証、Large特徴抽出、parity/容量/進捗表示。optimizerや学習loopを置かない。
- notebooks/02_train_and_evaluate_decoder.ipynb: 検証済みcacheと公式projからC評価・保存→D学習/validation選択→両test評価を呼ぶ。音声・encoder処理を置かない。
- 集計は保存済み結果から実行する。notebooks/03_summarize_results.ipynbは必要な場合だけ追加し、学習・再抽出を行わない。
- 長時間処理本体はCLI/helperへ置き、Notebookは設定・呼出し・表示を担当する。
- 既存NotebookはC/D対応済みとみなさない。旧IEMOCAP Notebookはデモ・回帰対象として保持する。
- msp_unavailable_label_audit.ipynbは既存の復元候補確認が済むまで再生成しない。
- 実データmanifest生成、実checkpoint推論、parity、抽出、benchmark、合成/実データでoptimizer stepを行うテスト、学習・評価はユーザーが実行する従来の分担を維持する。今回これらは実行しない。

## 10. 実装・実験の順序

詳細な変更予定と検証は第17〜22節による。

1. 既存変更を保護し、実モデルmetadata・固定manifest・除外契約を確認する。
2. C/D contract、旧indexからの明示変換、9→4選択と非学習テストを実装する。
3. Large公式adapter、前処理/特徴cache、公式推論とのparity testを実装する。
4. ユーザーが10〜20発話parityとLarge実音声benchmarkを実行。未合格なら本評価へ進まない。
5. 必要なLarge cacheと容量を検証し、C共通評価・保存を実行する。
6. D限定学習・freeze/行不変・resume検証を通し、seed 42→43→44を実行する。
7. HCUDB validationで選択済みのD checkpointを両testで評価し、contract一致を確認してD−Cを集計する。

Dコードの実装・テストはC本評価完了前にも設計できるが、D正式学習はC両test保存を開始条件とする。test結果に基づいて設定を変えない。長時間実行前に時間・必要容量（+20%の余裕）を見積もり、疎通/正式出力を分ける。Base実測時間をLarge実測値として扱わない。

## 11. 研究上の解釈と今回の範囲

Dの変化は固定表現に対する線形分類層をHCUDBへ適応した効果として解釈する。「encoderが日本語感情表現を新たに学習した」とは述べない。HCUDBとMSPは言語に加えて演技/自然発話、録音条件、話者、注釈方法、感情分類が異なるため、「純粋な日本語language adaptation」と定義しない。

主実験はHCUDB target-domainへのhead-level adaptationと、MSP上の性能への影響の観測とする。「嫌い→disgust」の近似対応と、MSPの事前学習データとの重複に関する制限を報告する。

今回編集するのは本文書とPlan mode入力用プロンプトだけ。下記履歴は保存し、計画上の未実装項目を完了へ変更しない。

## 12. 2026-08-23確定追補

> 第12〜16節は履歴である。当時の記述を保持するが、現行指示は2026-09-07改訂を優先する。

本節は上記の未確定表現を置き換える確定契約である。新しい計画ファイルは作成せず、本文書を引き続き単一の現行計画とする。

- MSP-Podcastは公式`Train → train`、`Development → validation`、`Test1 → test`を使用する。Test2 13,289件は全件監査するが、今回の学習・評価・cache対象から除外する。
- `SpkrID=Unknown`は主manifestのincluded行から除外する。既知話者のTrain/Development/Test1間重複は0件である。
- HCUDBは`hcudb1_speaker_split_v1`を使用する。train=`FA, FB, FD, FH, FI, FL, MC, MJ, MM, MN`、validation=`FF, MK`、test=`FG, ME`である。
- IEMOCAPはSession 1–5をまとめた外部testとし、4クラス出力を維持する。4クラス記述評価に加えて3クラス主集計を保存するが、確率を再正規化しない。
- `--layer`は`final`だけを受け付け、cache metadataには`final_after_encoder_norm`を保存する。整数層は拒否する。
- manifest schemaは`ser_manifest_v1`、feature cacheは`ser_feature_cache_v1`、decoder checkpointは`ser_decoder_checkpoint_v1`とする。
- emotion2vec事前学習にMSP-Podcast v1.8が使われたことは論文Table 1で確認済みのlimitationとする。R1.8とR1.10の包含関係はmetadata不在のため`unverified`のままとする。

## 13. 実装状況（2026-08-23）

- [x] worktree保護、dirty prompt SHA-256記録、実装前76テスト成功
- [x] version付きmapping、共通reader、manifest、strict split/leakage検証CLI
- [x] 48→16 kHz変換、mono/有限値検査、再開可能shard、partial復旧、hash、mmap reader
- [x] dataset非依存`BaseModel`、旧import/state dict互換re-export
- [x] validation UAR→macro F1→lossのmodel選択
- [x] 0–1 metrics、4×4混同行列、クラス別指標、4確率CSV/JSON保存
- [x] MSP親/HCUDB子のstage・親ID・親SHAとparent/resume分離
- [x] 3 dataset before/afterのmanifest/utterance集合signature検証
- [x] Notebook 01/02 builder、既定の長時間実行フラグfalse、静的境界テスト
- [x] CPU合成E2E（1 epoch、seed 42）
- [x] HCUDB実音声1件のBase checkpoint benchmark
- [x] 既存・新規を含む101テストとNotebook 01/02 demoの最終成功
- [ ] MSP音声配置後のstrict manifestと全対象duration集計
- [ ] MSP全件時間・容量見積り、+20%容量ゲート、正式実行承認

詳細監査とbenchmark値は`docs/reports/2026-08-23-msp-hcudb-data-audit.md`に記録する。MSP `Audio/`が空であるため、正式な全件抽出・学習は引き続き開始しない。

## 14. 実行状況追補（2026-08-24）

- [x] HCUDB1のstrict manifestを作成（全4,620行、現行4クラス対象2,100件、欠損0）
- [x] HCUDB1対象2,100発話のemotion2vec Base特徴を全件抽出
- [x] train 1,500 / validation 300 / test 300、4 shard、412.02 MiB、partial 0件を確認
- [x] cache ID `fdbaf28f74b94d3f`、manifest SHA-256 `1ff09b60be9d83d42c0ee2203c1a655d218f3070a978303000e40e4fbc3faf46`で独立再検証成功
- [ ] MSP全件特徴抽出と親decoder学習
- [ ] MSP親checkpointからのHCUDB継続学習と正式評価

HCUDB側は実音声の特徴cache準備まで完了した。これは実験準備上の進捗であり、decoder性能や追加学習効果の結果ではない。

## 15. 2026-08-30 4クラス下流学習開始準備の追補

本節は、今回の一括研究経路、実行環境、seed実行順について、上記のIEMOCAP評価および一括3 seed実行の記述を置き換える。IEMOCAPのreader、cache、単独評価機能は保守対象として残すが、今回の`run_transfer_study()`、Notebook 01/02、正式集計には含めない。

- 正式評価対象は`msp_podcast`と`hcudb1`の2データセットだけとする。追加学習前後で同一のMSP-Podcast test集合とHCUDB test集合を評価する。
- 標準WSL環境ではCUDAを利用できないため、ユーザー判断に基づき実データ1 epoch疎通と正式学習は`device='cpu'`で実行する。
- HCUDB manifest/cacheは準備済みとのユーザー申告を前提とし、学習直前に現行validatorで完全性を再確認する。Codexは実データの所在や内容を参照しない。
- MSP-Podcastは実データを現行4クラスと公式splitに沿って整理し、`audit-data`、`build-manifest`、`validate-manifest`、1件CPU特徴抽出benchmark、必要容量+20%の容量ゲート、全特徴cache検証をユーザーが順に実行する。
- `msp_podcast_unavailable_wav_filenames.txt`は参考情報にとどめ、manifestの除外条件には使用しない。現在欠損している音声と0バイトWAVの正確な1,128件だけを承認済み除外契約で固定し、残る24,857件をstrict manifestの採用対象とする。0バイト以外の対象音声にデコード失敗などがあれば自動除外せず停止する。
- 実データ1 epoch疎通はseed 42だけで行い、MSP親学習、HCUDB継続学習、両データセットの追加学習前後評価までを確認する。この出力は`smoke/`へ隔離し、正式集計には含めない。
- 疎通後、CPU時間と学習履歴を基に正式epoch数を正の整数として固定する。未設定の場合、正式実行を拒否する。
- 正式学習はまずseed 42だけを実行する。親子checkpoint ID・SHA-256、両評価集合signature、seed、各cache ID、設定値を確認した後、確認フラグを立ててseed 43・44を別出力で実行する。
- 学習開始条件は、MSP/HCUDB両cacheの完全検証、実音声1件benchmark、必要容量+20%の容量ゲート、正式epoch数の固定、疎通／正式出力先の分離がすべて成立することとする。
- 特徴抽出、benchmark、合成または実データで`train_decoder`を呼ぶテスト、実学習はユーザーが実行する。Codexが実行するのはmapping、split、manifest、cache、Notebook境界など、学習を伴わない検査だけとする。
- `msp_unavailable_label_audit.ipynb`は復元候補の確認が済むまで再生成しない。Notebook builderの通常対象から外し、明示選択時だけ生成またはJSON内容検査の対象にする。

## 16. 2026-08-31 MSP-Podcast利用不能音声1,128件の固定除外契約

本節は、MSP-Podcast対象音声を欠損0件まで再取得してから開始するという従来条件を置き換える。公式split、4クラスmapping、既知話者条件は変更しない。

- metadata上の4クラス対象25,985件のうち、現在欠損している音声と0バイトWAVの1,128件だけを`msp_missing_audio_exclusions_v1`として固定除外し、正式採用数を24,857件とする。
- 固定除外の元ラベル内訳は`A 416 / H 576 / S 107 / D 29`、公式split内訳は`Train 579 / Development 219 / Test1 330`とする。利用不能がランダムであるとは主張しない。
- 公式`Train / Development / Test1`の割当ては組み替えない。MSP-Podcastの主testは、公式Test1から固定330件を除いた利用可能部分集合として扱う。
- 将来報告するMSP-Podcast評価値は、完全な公式Test1全体ではなく、事前に固定した利用可能部分集合に対する結果であることを明記する。
- 添付1,128件の候補一覧自体は除外条件に使わない。metadataと現行inventoryから再計算し、0バイトWAV 254件は入手不能な欠損音声として扱う。
- 除外契約はファイル名順で保存し、utterance ID、元ラベル、4クラス変換後ラベル、公式split、除外理由、固定件数内訳、正規化SHA-256を含める。件数・内訳・metadata・現在の欠損集合・承認SHAのどれかが一致しなければmanifestを作成しない。
- 契約対象音声が後日復旧しても自動採用しない。採用方針を変える場合は新versionの除外契約として再生成・再承認する。
- manifest build reportに除外契約SHA、内訳、最終採用24,857件を保存する。cache metadataと評価集合signatureにも同じ契約signatureを伝播し、正式成果物のprovenanceへ契約JSON、manifest SHA-256、cache IDを同梱する。
- 実データmanifest作成、特徴抽出、benchmark、cache生成、学習、評価はユーザーが実行する。Codexは除外契約・manifest・cache metadata・評価signature・Notebook境界などの非学習テストだけを実行する。

## 17. 2026-09-07 C/D仕様の確定と変更予定

初回の「対象外予測の扱いを後で決める」という計画を置き換える。現行指示は9 logits→共通4 logits選択→4-way予測/softmax、Dは同じ4 logitsへのCE、4×4混同行列、OUTSIDEなし。旧index変換、対象外行不変、公式logit parity、version付きcontractを必須とする。

第12〜16節のチェック済み項目は過去のBase/decoder経路の実績であり、Large C/Dの実装完了を示さない。既存ファイルはその場で編集し、Git履歴で戻す。バックアップ/版違いPythonファイルは作らない。新規module先頭に役割を示すmodule docstringを置く。

| 予定 | ファイル | 担当する実装 |
|---|---|---|
| 新規 | ser_pipeline/official_head.py | FunASR Large adapter、公式metadata/head読込、pooling、9 logits、9→4選択、公式経路の計測。fairseq Base adapterと異なるruntimeの役割 |
| 新規 | ser_pipeline/config/evaluation_contract.v1.json | 第18節の実行contract。未確認placeholderを含む正式実行を拒否 |
| 変更 | ser_pipeline/contracts.py | C/D contract読込/検証、mapping view、canonical hash。旧LABEL_ORDERは維持 |
| 変更 | ser_pipeline/manifest.py, evaluation.py | 元manifestとC/D label view、集合署名、ラベル順指定metrics、4×4行列、9/4 logits・元ラベル保存 |
| 変更 | ser_pipeline/features.py, audio.py, cache.py | Large adapter接続、前処理分岐、再開shard、新schema/parity検証 |
| 変更 | ser_pipeline/training.py, checkpoints.py | 公式proj専用train入口、4-logit CE、freeze、非対象行不変、D checkpoint/optimizer/resume |
| 変更 | ser_pipeline/study.py, cli.py, notebook_api.py | C/parity/D/比較入口、C保存ゲート、3 seed管理、設定確定チェック |
| 変更 | scripts/build_ser_notebooks.py とNotebook 01/02 | 設定・呼出し・表示と抽出/学習境界。builderを正として必要なNotebookだけ再生成 |
| 変更 | scripts/summarize_ser_saved_results.py | C/D集計入口、schema/signature検証、D−C、mean/SD、CSV/Markdown |
| 新規テスト | tests/test_ser_official_head.py | 9→4、初期化、CE勾配/step後行不変、公式parityを区分して検証 |
| 新規テスト | tests/test_ser_cd_contract.py | contract、旧index変換、signature拒否、C保存ゲート、seed集計 |
| 既存テスト拡張 | tests/test_ser_cache.py, test_ser_cache_reuse.py, test_ser_mappings.py, test_ser_notebook_boundaries.py | 新旧schema区別、再利用、旧mapping互換、Notebook境界 |

BaseModelを公式headへ作り替えず、旧decoder公開入口を維持する。ser_decoder_checkpoint_v1へ公式headを偽装して保存しない。予定schemaは `ser_official_head_checkpoint_v1` と `ser_cd_evaluation_result_v1`。新規module数を増やすための分割は行わない。

## 18. evaluation_contract_v1 のschema案

以下は仕様であり、実行用JSONは今回まだ作らない。未確認フィールドが残ればdraftとし、C/D実行を拒否する。version文字列だけで一致判定せず、正規化JSON全体のSHA-256を保存する。

| フィールド群 | 必須内容 |
|---|---|
| 識別 | schema/version、status、canonical SHA-256（hash自身をhash対象から除く） |
| 統一ラベル | class_order=[anger, disgust, happy, sadness]、class-to-index |
| mapping | MSP/HCUDB元ラベル→統一ラベル、version、近似対応、旧index読替え規則 |
| 公式モデル | model ID、hub、resolved revision、checkpoint/config hash、proj weight/bias構造 |
| 公式出力 | 検証済み全9 label order、選択4 names/indices、確認元metadata |
| 評価 | select-before-softmax、4-way argmax、同値規則、非有限値拒否、OUTSIDEなし |
| 指標 | accuracy、全4クラスmacro F1/UAR、class-wise指標、zero_division=0、support 0、空集合拒否 |
| 混同行列 | rows=true、columns=prediction、固定4クラス順 |
| データ | dataset/release、元manifest SHA、split version、各split ID集合signature、固定除外・重複除外signature |
| C/Dラベルview | dataset/split/ID/元ラベル/統一ラベル/C/D index/audio hashを正規順でhash |
| 特徴 | preprocessing version、sample rate/normalization/resampling、pooling、dimension/dtype、cache schema/ID、parity report SHA |
| 再現 | 公式推論・抽出・評価コードversion、依存version、parity許容誤差 |

HCUDB train/validation/testとMSP testの署名を固定する。MSP train/validationは学習用に要求しない。split間の話者・発話・既存音声重複契約を検査する。dataset+utterance IDの一意性と元ラベル/音声の対応まで確認する。

条件、seed、適応後head hash、optimizer履歴、metricsはrun metadataへ置く。共通contractのcheckpoint hashは親公式checkpointとする。Dのhead変化だけで共通contractが不一致になるschemaにしない。

## 19. run管理・保存形式・比較report

予定出力構造（今回作成しない）:

```text
runs/ser_cd/<study_id>/
  evaluation_contract.json
  provenance/                 # 固定manifest・除外契約・公式metadata・設定
  parity/parity_report.json
  C/<run_id>/
    run_metadata.json
    msp_podcast/{metrics.json,predictions.jsonl,confusion_matrix.csv}
    hcudb1/{metrics.json,predictions.jsonl,confusion_matrix.csv}
  D/seed-42/<run_id>/          # seed-43、seed-44も同構造
    config.json
    run_metadata.json
    best.pt
    last.pt
    history.json
    msp_podcast/{metrics.json,predictions.jsonl,confusion_matrix.csv}
    hcudb1/{metrics.json,predictions.jsonl,confusion_matrix.csv}
  comparison/{summary.json,metrics.csv,report.md}
```

best.ptはHCUDB validation選択用、last.ptは同一D runのresume用。9行全体のproj state、親公式checkpoint hash、encoder/cache識別、contract hash、seed、optimizer/RNG/scheduler状態、epoch、選択履歴を保存する。encoderは親参照で復元可能な形式にする。別seed・別contract・旧decoderからのresumeは拒否する。

Cは1点、Dは42/43/44の各結果とmean、sample standard deviation（ddof=1）、seed数を保存する。欠落seedを0で補完しない。3 seed未完了ならpartialと明記し、正式集計と区別する。各D seedと同じCとの差をデータセット別・クラス別に計算し、D−Cのmean/SD・混同行列差も保存する。

比較前に共通contract全体、親checkpoint、encoder/cache、両test署名、ラベル/音声対応を照合し、不一致なら拒否する。個別予測はdataset+utterance IDで結合し、行番号に依存しない。reportは保存済みmetrics/predictionsから再生成し、学習・再推論を行わない。

## 20. test strategy

以下は将来実装・実行するテストであり、今回は未実施。

| 検証 | 内容・完了条件 | 実行区分 |
|---|---|---|
| metadata/contract | 9クラス重複・欠損・proj次元不一致・未確定index・schema不足拒否 | 非学習 |
| 9→4 | neutral等最大でも4-class予測。4-way softmaxの参照値一致・確率和1 | 非学習 |
| 旧index | 旧happy=1/disgust=3をC/D happy=2/disgust=1へラベル経由で変換 | 非学習 |
| CE/行不変 | 4-logit CE参照計算一致、非対象5行grad=0、複数step後weight/bias完全一致、対象行更新 | ユーザー実行の小規模optimizerテスト |
| encoder固定 | optimizerがprojのみ、encoder gradなし、parameters/buffersが前後不変 | 実モデル/optimizer検証はユーザー |
| 再開 | D初期値=C、各seed独立、resume後の非対象行/state不変、別seed/contract拒否 | optimizerを伴う部分はユーザー |
| 共通評価 | 手作り4×4行列、zero division/support 0、同値、9/4 logits・元ラベル保存整合 | 非学習 |
| cache | Base拒否、Large metadata/hash不足拒否、partial、ID/音声hash不一致拒否 | 合成cacheで非学習 |
| parity | 公式vs書込/再読込cache、10〜20発話、全9 logits許容差・4-way argmax一致 | ユーザー、C前必須 |
| signature/report | 同件数の別ID・ラベル変更・除外SHA変更拒否、D−Cとddof=1手計算照合 | 非学習 |
| run境界 | C両test保存前の正式D拒否、MSP/testによる選択拒否、seed管理・partial | 非学習の設定・保存物検証 |
| Notebook/回帰 | 抽出/学習境界、旧decoder/schema/mapping互換 | 非学習部分。既存学習E2Eはユーザー |

optimizerテストは対象外行不変という重要要件の検証に限定する。既存全件テストを自動実行せず、学習や実データ利用の有無で実行区分を確認する。

## 21. 未確認事項と開始前チェックリスト

| 未確認事項 | 確認元 | 確定期限 |
|---|---|---|
| Large revision/hash/9出力順/proj構造 | 実snapshot/config/公式実装 | contract final化前 |
| normalization/resampling/token/pooling/入力次元 | 同上と中間値比較 | cache設計確定・parity前 |
| 採用manifest・除外/重複契約SHA | 既存正式manifestとprovenance | subset固定前 |
| parity atol/rtol・device/dtype・10〜20発話 | 公式経路・環境、性能非依存の選定規則 | parity実行前 |
| lr/batch/最大epoch/patience/min_delta/scheduler | HCUDB train/validationのみ | 正式設定固定前。test参照禁止 |
| Large時間・容量 | 実音声benchmarkと対象件数 | 全件抽出・正式実行前 |

### C開始前

- [ ] 実checkpoint/configの9出力順と選択indexを検証。
- [ ] 共通クラス順、元mapping、旧index変換、固定split/manifest署名を確定。
- [ ] evaluation_contract_v1の未確認フィールドを解消しhashを固定。
- [ ] 公式前処理・Large専用feature/cacheを照合。
- [ ] 10〜20発話の公式logit parityに合格。
- [ ] 4-logit選択・共通評価・manifest/signature consistencyのテストに合格。
- [ ] 必要cacheの完全性、容量、正式出力先を確認。

### D開始前

- [ ] C両testのmetrics・個別予測・9 logits・再現情報を保存。
- [ ] 同じ公式proj初期値からの開始・encoder固定を検証。
- [ ] 4-logit CE、非対象5行gradとstep後weight/bias不変、resumeを検証。
- [ ] HCUDB validationのみの設定/選択規則、epoch・early stoppingを固定。
- [ ] seed 42/43/44の独立出力、同じsplit/cache/contractを確認。

### C/D比較前

- [ ] 選択済みD checkpointを固定し、両testを共通規則で評価。
- [ ] contract hash、test集合・元ラベル・音声対応、親checkpoint/cache一致。
- [ ] D seed別結果・mean/SD、D−C、クラス別指標、4×4行列、個別9 logitsを保存。
- [ ] 3 seed完了状況、近似mapping、MSP subset・重複制限、解釈をreportへ記載。

## 22. 今回の文書改訂の確認

対象は本文書と既存Plan mode入力用プロンプト。既存の未コミット変更を基に現行部分を更新し、第12〜16節の履歴を保持する。コード・config・Notebook・checkpoint・cache・既存結果は変更しない。文書整合性・参照ファイル・差分だけを確認し、将来テストを実行済みと扱わない。
