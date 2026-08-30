# Python ファイル索引

このリポジトリでは、既存ファイルの編集時に別名のバックアップを作らず、
Git の履歴を復元手段として使います。`*_old.py`、`*_backup.py`、`*_v2.py`
などは作成しません。

調査時点の Python ファイルに同一内容の複製はありません。似た名前の
ファイルは、入力形式、モデル構造、学習、推論、テストなどの役割が異なります。
ただし `archive/` は現行処理から外した過去コードであり、実行には不要です。

## ディレクトリの区分

| ディレクトリ | 区分 | 用途 |
|---|---|---|
| `upstream/` | 現行・外部由来 | emotion2vec 本体と fairseq 事前学習タスク |
| `ser_pipeline/` | 現行 | データセット横断のカテゴリ感情分類パイプライン |
| `iemocap_downstream/` | 現行・互換 | IEMOCAP ベースラインとノートブック用処理 |
| `vad_downstream/` | 現行 | 連続感情値とカテゴリ感情を扱う下流処理 |
| `scripts/` | 補助 | 特徴抽出とノートブック生成 |
| `tests/` | 検証 | 現行コードと生成ノートブックの回帰テスト |
| `archive/` | 過去コード | 現行実行系では参照しない保存物 |

## `upstream/`: emotion2vec 本体

- `upstream/models/emotion2vec.py`: emotion2vec 本体モデルと特徴抽出 API。
- `upstream/models/audio.py`: 音声用 CNN 特徴抽出器とエンコーダ。
- `upstream/models/base.py`: モダリティ共通エンコーダとマスク処理。
- `upstream/models/config.py`: モデル全体の設定データクラス。
- `upstream/models/modules.py`: Transformer、位置表現、デコーダなどの部品。
- `upstream/tasks/audio_pretraining.py`: fairseq 用の事前学習タスクとデータ読込。

## `ser_pipeline/`: カテゴリ感情分類

- `ser_pipeline/__init__.py`: 公開 API とパッケージ説明。
- `ser_pipeline/__main__.py`: `python -m ser_pipeline` の入口。
- `ser_pipeline/cli.py`: マニフェスト作成、特徴抽出、実験の CLI。
- `ser_pipeline/contracts.py`: ラベル対応、スキーマ版、固定値の契約。
- `ser_pipeline/readers.py`: MSP-Podcast、HCUDB1、IEMOCAP のメタデータ読込。
- `ser_pipeline/manifest.py`: JSONL マニフェストの作成、監査、検証。
- `ser_pipeline/splits.py`: データ分割と話者・音声リークの検査。
- `ser_pipeline/audio.py`: 音声検査と 16 kHz・モノラルへの前処理。
- `ser_pipeline/features.py`: emotion2vec 特徴量の再開可能な抽出。
- `ser_pipeline/cache.py`: 分割特徴キャッシュの保存、検証、読込。
- `ser_pipeline/model.py`: カテゴリ感情分類用デコーダ。
- `ser_pipeline/training.py`: デコーダの学習、検証、選択。
- `ser_pipeline/evaluation.py`: WA、UA、F1 などの評価と結果保存。
- `ser_pipeline/checkpoints.py`: 親・再開チェックポイントの整合性検査。
- `ser_pipeline/study.py`: MSP から HCUDB へ移す比較実験の進行管理。
- `ser_pipeline/notebook_api.py`: 研究ノートブックから呼ぶ小さな統合 API。
- `ser_pipeline/preflight.py`: 本実行前の時間・容量見積りと短時間試験。

## `iemocap_downstream/`: IEMOCAP ベースライン

- `iemocap_downstream/__init__.py`: パッケージ説明。
- `iemocap_downstream/data.py`: 抽出済み特徴量と感情ラベルの読込・分割。
- `iemocap_downstream/model.py`: 共通 SER デコーダへの互換入口。
- `iemocap_downstream/utils.py`: 旧ベースラインの学習・評価補助。
- `iemocap_downstream/main.py`: 旧ベースラインの5分割学習 CLI。
- `iemocap_downstream/notebook_pipeline.py`: IEMOCAP 実験ノートブックの再利用処理。
- `iemocap_downstream/scripts/iemocap_manifest.py`: 生 IEMOCAP から音声マニフェストを作成。
- `iemocap_downstream/scripts/emotion2vec_speech_features.py`: マニフェストから特徴量を抽出。
- `iemocap_downstream/scripts/csv_to_labels.py`: CSV メタデータをラベルファイルへ変換。

## `vad_downstream/`: 連続感情値・カテゴリ感情

- `vad_downstream/__init__.py`: パッケージ説明。
- `vad_downstream/data.py`: 2種類の入力契約（CSV＋個別キャッシュ、連結済み特徴＋長さ）の読込。
- `vad_downstream/model.py`: 回帰、VAD経由分類、並列分類・回帰のモデル定義。
- `vad_downstream/loss.py`: 欠損ラベルをマスクできる CCC 損失。CSV 入力の学習で使用。
- `vad_downstream/train_vad.py`: CSV の音声パスから特徴をキャッシュして回帰器を学習。
- `vad_downstream/train_head.py`: 連結済み特徴から回帰ヘッドだけを学習。
- `vad_downstream/training.py`: `train_head.py` 用の学習・評価・保存処理。
- `vad_downstream/inference.py`: WAV から連続感情値を推論して JSON 出力。
- `vad_downstream/train_vad_emotion.py`: 推定した連続感情値を経由するカテゴリ分類器を学習。
- `vad_downstream/emotion_training.py`: VAD経由分類の損失、評価、保存処理。
- `vad_downstream/infer_vad_emotion.py`: VAD経由分類と寄与度を JSON 出力。
- `vad_downstream/train_parallel_emotion_vad.py`: 独立した感情分類ヘッドと V/A/D 回帰ヘッドを学習。
- `vad_downstream/parallel_training.py`: 並列モデルの損失、評価、保存処理。
- `vad_downstream/infer_parallel_emotion_vad.py`: 並列モデルのカテゴリ感情と V/A/D を推論。
- `vad_downstream/notebook_pipeline.py`: 音声から感情・VADを扱うノートブック用処理。

`train_vad.py` と `train_head.py` はバックアップ関係ではありません。前者は
CSV と音声パスを入力にして特徴抽出・キャッシュも扱い、後者は既に連結済みの
`<prefix>.npy/.lengths/.vad` を入力にしてヘッドだけを学習します。

## `scripts/`: 生成・抽出補助

- `scripts/extract_features.py`: fairseq チェックポイントから特徴量を抽出。
- `scripts/build_iemocap_base_notebook.py`: IEMOCAP ベース実験ノートブックを生成。
- `scripts/build_ser_notebooks.py`: 分割された SER 研究ノートブックを生成。

## `tests/`: 回帰テスト

- `tests/test_emotion2vec_feature_device.py`: 特徴抽出時の CPU/GPU 配置を検証。
- `tests/test_iemocap_notebook_pipeline.py`: IEMOCAP ノートブック処理を検証。
- `tests/test_notebook_pipeline.py`: VAD ノートブック処理を検証。
- `tests/test_parallel_emotion_vad.py`: 並列カテゴリ感情・VAD経路を検証。
- `tests/test_ser_cache.py`: SER 特徴キャッシュと事前検査を検証。
- `tests/test_ser_decoder.py`: SER デコーダ、評価、チェックポイント互換性を検証。
- `tests/test_ser_e2e.py`: SER パイプライン全体を小規模データで検証。
- `tests/test_ser_manifest.py`: マニフェスト生成と監査を検証。
- `tests/test_ser_mappings.py`: データセット別ラベル対応を検証。
- `tests/test_ser_notebook_boundaries.py`: 生成ノートブックの責務分離を検証。
- `tests/test_ser_splits.py`: 固定分割とリーク検査を検証。
- `tests/test_vad_downstream.py`: CSV 入力の VAD 回帰経路を検証。
- `tests/test_vad_downstream_data.py`: 連結済み VAD データ読込を検証。
- `tests/test_vad_downstream_model.py`: VAD モデル構造と出力形状を検証。
- `tests/test_vad_downstream_training.py`: VAD 回帰の学習・評価・保存を検証。
- `tests/test_vad_downstream_train_head.py`: 回帰ヘッド学習 CLI を検証。
- `tests/test_vad_downstream_inference.py`: WAV から VAD への推論を検証。
- `tests/test_vad_downstream_emotion_training.py`: VAD経由分類の学習・指標を検証。
- `tests/test_vad_downstream_train_vad_emotion.py`: VAD経由分類の学習 CLI を検証。
- `tests/test_vad_downstream_infer_vad_emotion.py`: VAD経由分類の推論 JSON を検証。
- `tests/execute_demo_notebook.py`: VAD デモノートブックを隔離環境で実行。
- `tests/execute_iemocap_base_demo_notebook.py`: IEMOCAP デモを実行し個人情報混入を検査。
- `tests/execute_ser_demo_notebooks.py`: SER デモノートブックを短時間設定で実行。

## `archive/`: 現行処理では不要

- `archive/vad_iemocap_two_stage/model.py`: 旧2段階 VAD経由分類モデル。
- `archive/vad_iemocap_two_stage/loss.py`: 旧2段階モデルの損失。
- `archive/vad_iemocap_two_stage/train.py`: 旧2段階モデルの学習処理。
- `archive/notebook_tools/_patch_notebook.py`: 特定の旧ノートブックを直接書き換えた一回限りの補助。

これらは実行時の import 先ではありません。Git 履歴があるため、今後は同種の
退避ファイルを新設せず、削除するかどうかは履歴保存方針を決めたうえで扱います。
