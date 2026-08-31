# MSP-Podcast→HCUDB SER研究 現行実施計画

作成日: 2026-08-22（Asia/Tokyo）

> **現行計画の単一基準:** 2026-08-22以降は本文書を本研究の現行計画とする。以前のIEMOCAP主学習・`anger / happy / sadness / neutral`・union 12クラスを前提とする計画は履歴資料としてのみ参照する。

## 1. 確定した研究設計

- emotion2vec encoderは固定し、音声からの特徴抽出とdecoder学習を完全に分離する。
- 主な英語学習データはMSP-Podcast Release 1.10とする。
- MSP-Podcastで学習したdecoderをHCUDBで追加学習し、追加学習前後の英語・日本語性能を比較する。
- IEMOCAPは主学習に使わず、学習後の外部英語testデータセットとして使う。
- decoderの出力は4クラスとし、表記・順序を次に固定する。

```text
anger / happy / sadness / disgust
```

- 元ラベルは上書きせず、`original_emotion`として保持する。共通ラベルは`mapped_emotion`、変換規則の版は`mapping_version`として保存する。
- HCUDBの「嫌い」を`disgust`へ対応させる場合は、直接同義ではない近似対応として研究資料に明記する。
- emotion2vecの事前学習にMSP-Podcast v1.8が含まれる点をデータ重複上の制約として記録する。Release 1.10を下流学習に使うこと自体は禁止しないが、完全に未知の英語データだけを使った評価とは主張しない。

## 2. データセットの役割

| データセット | 役割 | 正式評価の対象 | 補足 |
|---|---|---|---|
| MSP-Podcast Release 1.10 | 英語decoderの主学習・validation・test | 4クラスすべて | Release付属metadataの全感情、件数、話者、公式splitを実装前に再集計する |
| HCUDB | 日本語の追加学習・validation・test | 4クラスすべて | 話者非重複の固定splitを使い、「嫌い→disgust」は近似対応として扱う |
| IEMOCAP | 外部英語test | `anger / happy / sadness`を主な定量比較 | `disgust`は2件しか確認されていないため、予測確認は行うが主性能の結論には使わない |

IEMOCAP評価でもdecoderの出力次元は4のまま変えない。主定量評価では十分なsupportがある3クラスを対象にし、`disgust`を出力した場合は誤りとして数える。4クラス混同行列と`disgust`の個別予測は別途保存し、support付きの記述的結果として報告する。

## 3. 実験系列

```text
各データセットの音声・metadataを検証
  -> ラベル変換・split manifestを固定
  -> 固定emotion2vecで特徴量を一度だけ抽出
  -> 特徴cacheを検証・凍結
  -> MSP-Podcast特徴でdecoderを学習
  -> MSP test・HCUDB test・IEMOCAPを追加学習前評価
  -> 同じdecoder checkpointをHCUDBで追加学習
  -> 同じ評価集合を追加学習後評価
  -> seed間集計と追加学習前後の対応差を保存
```

- MSP-PodcastはRelease 1.10に公式splitが含まれる場合、それを優先する。含まれない場合だけ、話者漏洩のない固定splitを結果取得前に作る。
- HCUDBは14話者をtrain 10、validation 2、test 2へ固定し、同一話者を複数splitへ入れない。
- seedは`42 / 43 / 44`の3つを使用する。
- 旧計画のIEMOCAP session 5-foldは、IEMOCAPを学習データから外したため現行主実験には適用しない。
- MSP-Podcastの公式split仕様を確認するまでは、5-foldを新たに仮定しない。

## 4. Notebookと実行コードの分離

### 4.1 特徴抽出側

新規に`notebooks/01_extract_emotion2vec_features.ipynb`を用意する。

- MSP-Podcast、HCUDB、IEMOCAPの接続状況を確認する。
- 全元ラベルと件数を集計してから4クラスを抽出する。
- ラベル変換表とsplit manifestを生成・検証する。
- emotion2vec特徴抽出CLIを呼び出す。
- 抽出件数、特徴次元、フレーム長、有限値、ID対応、encoder識別情報を検証する。
- decoderの作成、optimizer、epoch loopは置かない。

長時間処理の本体はNotebookセルへ直接埋め込まず、再開可能なCLI/helperとして実装する。Notebookは設定、実行指示、進捗確認、検証結果の表示を担当する。

### 4.2 decoder学習・評価側

新規に`notebooks/02_train_and_evaluate_decoder.ipynb`を用意する。

- 検証済み特徴cacheとmanifestだけを入力にする。
- WAV、fairseq、emotion2vec checkpointを読み込まない。
- MSP-Podcast学習、3 seed実行、checkpoint保存を行う。
- MSP checkpointからHCUDB追加学習を行う。
- 追加学習前後で同一test集合を評価する。
- 指標、混同行列、個別予測、checkpoint metadataを保存する。

### 4.3 結果集計側

必要になった時点で`notebooks/03_summarize_results.ipynb`を追加する。

- 3 seedの平均・標準偏差を集計する。
- 追加学習後−追加学習前の対応差をデータセット別・クラス別に集計する。
- 表と図を再生成できる形で保存する。
- 学習や特徴抽出は行わない。

## 5. 特徴cacheの契約

- 大規模データを単一の巨大な連結`.npy`だけへ保存しない。
- dataset / split / shard単位で保存し、中断後に未完了shardから再開できるようにする。
- 発話IDから特徴ファイルまたはshard内offsetを一意に引けるindexを持つ。
- `np.load(..., mmap_mode="r")`または同等の遅延読込を可能にする。
- manifestには最低限、dataset、release、split、utterance ID、speaker ID、元ラベル、変換後ラベル、音声識別情報、特徴shapeを保存する。
- cache metadataには、encoder名、checkpoint hash、実際に使用したlayer、特徴次元、抽出コード版、抽出日時、mapping version、manifest hashを保存する。
- 一部欠損、重複ID、非有限値、0フレーム、次元不一致、manifest不一致を検出した場合は学習を拒否する。
- cache完成後は3 seedと追加学習前後で同じ特徴を再利用し、seedごとに再抽出しない。

## 6. 既存資産の扱い

### 再利用する候補

- `iemocap_downstream/model.py`の`BaseModel`
- padding、collate、masked mean pooling
- device選択とseed固定
- checkpoint保存・再読込
- macro F1、UAR、accuracy、クラス別F1、confusion matrix
- 特徴shape・有限値・ID整合性の検証処理
- HCUDBの話者split関連処理

### 一般化または置換する箇所

- `iemocap_downstream/notebook_pipeline.py`の`CLASS_LABELS`とSession 1–5固定処理
- IEMOCAP専用のmanifest・label reader
- 単一連結`.npy`を前提とするfeature bundle loader
- `iemocap_downstream/scripts/emotion2vec_speech_features.py`の破壊的な再作成、再開不能、shard非対応
- 指定可能だが抽出処理へ反映されていない`--layer`の扱い
- 1つのNotebookで特徴抽出とdecoder学習を連続実行する構成

既存の`notebooks/iemocap_base_downstream_training.ipynb`はデモ・回帰確認用として残し、現行研究用Notebookへ上書きしない。

## 7. 評価と保存物

各seed・追加学習段階について次を保存する。

- accuracy / WA
- UAR
- macro F1
- クラス別precision / recall / F1 / support
- 4クラスconfusion matrix
- 発話単位の正解ラベル・予測ラベル・4クラス確率
- MSP学習checkpointとHCUDB追加学習checkpoint
- 親checkpoint、seed、split、label order、mapping version、cache ID、コード版を含むmetadata

MSP-PodcastとHCUDBは4クラスすべてを正式比較する。IEMOCAPは3クラス主評価と4クラス記述評価を分離し、`disgust`の2件から一般的な精度を主張しない。

## 8. 実装前に行うこと

1. MSP-Podcast Release 1.10のローカル配置と利用可能ファイルを確認する。
2. MSP-Podcastの全感情ラベル、各件数、話者数、split、Release 1.8との包含関係をmetadataから集計する。
3. MSP-Podcast、HCUDB、IEMOCAPの4クラス変換表をversion付きで確定する。
4. HCUDBとIEMOCAPの現在の音声root・metadata接続を再確認する。
5. MSP公式splitの有無を確認し、主実験のsplit契約を確定する。
6. 予測特徴容量、空き容量、1音声の抽出時間を見積もる方法を実装計画へ入れる。
7. 既存Notebook・module・testsの再利用範囲をファイル単位で確定する。

## 9. 実装順

1. dataset非依存のラベル・manifest schemaを定義する。
2. 3データセットのreaderと4クラスmappingを実装する。
3. split生成・検証とリーク検査を実装する。
4. 再開可能・shard対応の特徴抽出CLIとcache indexを実装する。
5. 特徴cache validatorとmetadata照合を実装する。
6. 特徴抽出Notebookを作成する。
7. dataset非依存のdecoder学習・評価処理を実装する。
8. MSP→HCUDB継続checkpointと前後評価を実装する。
9. decoder学習・評価Notebookを作成する。
10. 合成fixtureで抽出後cache→MSP学習→HCUDB追加学習→3データセット評価を短縮E2Eテストする。
11. 既存テストと新規テストを全件成功させる。
12. 実音声1件で特徴抽出をベンチマークする。
13. 容量・時間・出力契約を報告し、ユーザーの開始指示を待つ。

## 10. 今回はまだ行わないこと

- MSP-Podcast、HCUDB、IEMOCAPの全件特徴抽出
- decoderの正式学習
- 3 seed一括実行
- HCUDB追加学習
- 正式な性能値の作成
- BaseとLargeの比較、公式9クラスhead、VAD媒介型などの拡張実験

## 11. 学習開始前ゲート

- 4クラスの表記・順序が全manifest、decoder、checkpointで一致する。
- 全元ラベルの採用・除外理由と件数が保存されている。
- train / validation / testの話者・発話重複がない。
- 特徴cacheが再実行せず読み込め、shape・有限値・ID・hash検証を通る。
- encoderがdecoder学習経路から完全に分離されている。
- decoder Notebook単独実行時にWAVもemotion2vecも要求しない。
- checkpointからHCUDB追加学習を再開でき、親子関係を追跡できる。
- 合成データE2Eと全自動テストが成功する。
- 実音声1件のベンチマークと全件の時間・容量見積もりがある。
- 長時間処理を開始する直前に、対象、予想時間、必要容量をユーザーへ提示する。

## 12. 2026-08-23確定追補

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
- `msp_podcast_unavailable_wav_filenames.txt`は参考情報にとどめ、manifestの除外条件には使用しない。現在不足している正確な874件だけを承認済み除外契約で固定し、残る25,111件をstrict manifestの採用対象とする。存在する対象音声にデコード失敗などがあれば自動除外せず停止する。
- 実データ1 epoch疎通はseed 42だけで行い、MSP親学習、HCUDB継続学習、両データセットの追加学習前後評価までを確認する。この出力は`smoke/`へ隔離し、正式集計には含めない。
- 疎通後、CPU時間と学習履歴を基に正式epoch数を正の整数として固定する。未設定の場合、正式実行を拒否する。
- 正式学習はまずseed 42だけを実行する。親子checkpoint ID・SHA-256、両評価集合signature、seed、各cache ID、設定値を確認した後、確認フラグを立ててseed 43・44を別出力で実行する。
- 学習開始条件は、MSP/HCUDB両cacheの完全検証、実音声1件benchmark、必要容量+20%の容量ゲート、正式epoch数の固定、疎通／正式出力先の分離がすべて成立することとする。
- 特徴抽出、benchmark、合成または実データで`train_decoder`を呼ぶテスト、実学習はユーザーが実行する。Codexが実行するのはmapping、split、manifest、cache、Notebook境界など、学習を伴わない検査だけとする。
- `msp_unavailable_label_audit.ipynb`は復元候補の確認が済むまで再生成しない。Notebook builderの通常対象から外し、明示選択時だけ生成またはJSON内容検査の対象にする。

## 16. 2026-08-30 MSP-Podcast不足874件の固定除外契約

本節は、MSP-Podcast対象音声を欠損0件まで再取得してから開始するという従来条件を置き換える。公式split、4クラスmapping、既知話者条件は変更しない。

- metadata上の4クラス対象25,985件のうち、現在不足している874件だけを`msp_missing_audio_exclusions_v1`として固定除外し、正式採用数を25,111件とする。
- 固定除外の元ラベル内訳は`A 378 / H 392 / S 80 / D 24`、公式split内訳は`Train 520 / Development 210 / Test1 144`とする。angerの欠損率は7.63%であり、欠損がランダムであるとは主張しない。
- 公式`Train / Development / Test1`の割当ては組み替えない。MSP-Podcastの主testは、公式Test1から固定144件を除いた利用可能部分集合として扱う。
- 将来報告するMSP-Podcast評価値は、完全な公式Test1全体ではなく、事前に固定した利用可能部分集合に対する結果であることを明記する。
- 添付1,128件の候補一覧は除外条件に使わない。その一覧に含まれていても現在存在する254件は、metadata上の対象条件を満たす限り通常どおり採用・デコード・音声metadata・SHA-256検証を行う。
- 除外契約はファイル名順で保存し、utterance ID、元ラベル、4クラス変換後ラベル、公式split、除外理由、固定件数内訳、正規化SHA-256を含める。件数・内訳・metadata・現在の欠損集合・承認SHAのどれかが一致しなければmanifestを作成しない。
- 契約対象音声が後日復旧しても自動採用しない。採用方針を変える場合は新versionの除外契約として再生成・再承認する。
- manifest build reportに除外契約SHA、内訳、最終採用25,111件を保存する。cache metadataと評価集合signatureにも同じ契約signatureを伝播し、正式成果物のprovenanceへ契約JSON、manifest SHA-256、cache IDを同梱する。
- 実データmanifest作成、特徴抽出、benchmark、cache生成、学習、評価はユーザーが実行する。Codexは除外契約・manifest・cache metadata・評価signature・Notebook境界などの非学習テストだけを実行する。
