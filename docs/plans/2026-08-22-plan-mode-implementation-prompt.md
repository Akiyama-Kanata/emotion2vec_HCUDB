# Plan mode入力用プロンプト

以下をPlan modeへそのまま貼り付けて使用する。

---

このリポジトリの音声感情認識研究を、`docs/plans/2026-08-22-msp-hcudb-feature-decoder-plan.md`を唯一の現行計画として実装できる状態へ整理してください。まずリポジトリ、既存Notebook、module、tests、dirty worktreeを読み、既存変更を保護したうえで、実装計画を作成してください。この段階では全件特徴抽出や正式学習などの長時間処理を開始しないでください。

確定済みの研究条件:

- emotion2vec encoderは固定する。
- 特徴抽出とdecoder学習を完全に分離する。
- 主学習はMSP-Podcast Release 1.10、追加学習はHCUDB、外部英語testはIEMOCAPとする。
- decoder出力の表記・順序は`anger / happy / sadness / disgust`の4クラスに固定する。
- MSP-PodcastとHCUDBでは4クラスすべてを正式評価する。
- IEMOCAPでは十分な件数がある`anger / happy / sadness`を主定量評価し、`disgust`は予測を保存・確認するが、2件だけなので一般的な性能結論には使わない。
- IEMOCAP評価時もdecoder出力は4クラスのままとし、`disgust`予測を隠したり3クラスへ再正規化したりしない。
- HCUDBの「嫌い→disgust」は近似対応として明記し、元ラベルを保持する。
- seedは`42 / 43 / 44`とする。
- IEMOCAPを学習に使わないため、旧IEMOCAP 5-foldは現行主実験へ持ち込まない。
- MSP-PodcastはRelease付属の公式splitがあれば優先し、なければ話者漏洩のない固定splitを結果取得前に作る。
- emotion2vecの事前学習にMSP-Podcast v1.8が含まれる制約を研究上のlimitationとして記録する。

計画に必ず含める作業:

- MSP-Podcast Release 1.10のローカル配置、metadata、音声、全感情ラベル、各件数、話者、公式splitの有無を確認する。
- Release 1.8と1.10の包含・追加範囲を、手元のrelease metadataで確認可能な範囲まで記録する。
- MSP-Podcast、HCUDB、IEMOCAPそれぞれについて、元ラベル→4クラスのversion付きmappingと除外規則を定義する。
- `original_emotion`、`mapped_emotion`、`mapping_version`を保存する共通manifest schemaを設計する。
- train / validation / test間の話者・発話重複を拒否する検証を設計する。
- 大規模な単一`.npy`への連結を避け、dataset / split / shard単位、途中再開可能、mmapまたは遅延読込可能な特徴cacheを設計する。
- cacheにencoder名、checkpoint hash、実使用layer、特徴次元、抽出コード版、manifest hashを保存する。
- 指定した`--layer`が実際のemotion2vec抽出へ反映されるようにするか、未対応なら引数を拒否して誤解を防ぐ。
- `notebooks/01_extract_emotion2vec_features.ipynb`を特徴抽出・cache検証専用として設計する。decoder学習処理を入れない。
- `notebooks/02_train_and_evaluate_decoder.ipynb`をcache読込・decoder学習・継続学習・評価専用として設計する。WAV、fairseq、emotion2vec checkpointを読み込ませない。
- 長時間抽出本体はNotebookへ直書きせず、再開可能なCLI/helperとして設計する。
- `iemocap_downstream/model.py`の`BaseModel`、padding、masked mean pooling、metrics、checkpoint処理など、再利用可能な既存資産を特定する。
- IEMOCAPの4クラス名・Session 1–5・単一連結feature bundleへ依存する部分をdataset非依存に一般化する。
- MSP学習checkpoint→HCUDB追加学習checkpointの親子metadataと互換性検証を設計する。
- 追加学習前後で同一のMSP、HCUDB、IEMOCAP評価集合を使う。
- accuracy / WA、UAR、macro F1、クラス別precision / recall / F1 / support、4クラスconfusion matrix、発話単位の4クラス確率を保存する。
- IEMOCAPの3クラス主評価と4クラス記述評価を分離し、`disgust`のsupportを必ず表示する。
- 合成fixtureで、manifest→cache検証→MSP decoder学習→checkpoint保存→HCUDB追加学習→3データセット前後評価までを短時間E2Eテストする。
- 既存テストを壊さず、mapping、split漏洩、cache破損、metadata不一致、checkpoint継続、Notebook責務分離のテストを追加する。
- 実音声1件の抽出ベンチマーク、全件の時間・容量見積もり、正式実行前ゲートを計画する。

既存Notebookの扱い:

- `notebooks/iemocap_base_downstream_training.ipynb`はデモ・回帰確認用として残し、現行研究用に上書きしない。
- `notebooks/audio_to_emotion_vad.ipynb`と`notebooks/vad_wagner_emotion2vec_experiment.ipynb`は主4クラス実験の正式経路にしない。
- 再利用可能な処理はNotebook間でコピーせず、共通moduleへ移す。

Plan modeの出力要件:

- 実装順を依存関係順に、検証可能な小さい単位へ分ける。
- 各ステップに、変更予定ファイル、再利用する既存コード、追加テスト、完了条件を付ける。
- 既存のユーザー変更と競合しそうなファイルを明示する。
- 不明点はローカルファイルから先に調べる。研究設計を変える必要がある未解決事項だけを質問する。
- 全件特徴抽出・正式学習は、コード、テスト、1件ベンチマーク、時間・容量報告がそろい、ユーザーが開始を指示するまで実行しない。
- Base/Large比較、公式9クラスhead、VAD媒介型などは今回の実装範囲外とする。

最終的に、すぐ実装へ移せる具体的な計画と、実装後に学習開始可否を判定できるチェックリストを提示してください。

---
