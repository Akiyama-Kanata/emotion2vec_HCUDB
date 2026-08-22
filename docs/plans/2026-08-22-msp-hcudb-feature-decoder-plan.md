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

