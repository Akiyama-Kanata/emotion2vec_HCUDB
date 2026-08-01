# 研究方針修正時点の完成度レポート

## 基準と対象

本レポートは2026-08-01時点のコード、Notebook、テスト、実行生成物を根拠に、固定emotion2vec特徴を用いた日英SER研究の現在地を記録する。完成度は百分率で表さず、次の判定語だけを用いる。

| 判定 | 意味 |
|---|---|
| 実装済み | 対応するコード経路が存在する |
| デモ検証済み | 合成データまたは代替実装で一連の動作と生成物を確認できる |
| 部分実装 | 必要な構成の一部だけが存在する |
| 未検証 | 実装または入口はあるが、対象の実環境・実モデル・実データで確認されていない |
| 未着手 | 対象の実験または結果取得に着手した根拠がない |

「デモ検証済み」は研究結果を意味しない。特に合成音声、決定論的な仮特徴抽出器、random modelから得た数値は配線確認だけに用い、HCUDBやIEMOCAPに対する性能値として扱わない。

## 主要判定

| 対象 | 判定 | 根拠 | 現時点の解釈 |
|---|---|---|---|
| Base向け直接感情分類＋並列VA/Dデコーダー | 実装済み | `vad_downstream/model.py`の`ParallelEmotionVADClassifier`と`Emotion2vecParallelEmotionVADClassifier`、`vad_downstream/parallel_training.py`、`vad_downstream/train_parallel_emotion_vad.py`、`vad_downstream/infer_parallel_emotion_vad.py` | 768次元入力を既定とする直接分類headと独立V/A/D head、学習、checkpoint、推論経路がある。 |
| Base向け直接感情分類＋並列VA/Dデコーダー | デモ検証済み | `notebooks/audio_to_emotion_vad.ipynb`、`runs/notebooks/audio_to_emotion_vad.demo.executed.ipynb`、`runs/notebooks/audio_to_emotion_vad/model.pt`、`runs/notebooks/audio_to_emotion_vad/test_metrics.json`、`runs/notebooks/audio_to_emotion_vad/evaluation_graphs.png` | 合成音声と`DEMO_FAKE_ENCODER_NOT_FOR_RESEARCH`による一連の学習・評価生成物がある。研究性能の根拠にはしない。 |
| VAD媒介型 | 実装済み | `vad_downstream/model.py`の`VADMediatedEmotionClassifier`と`Emotion2vecVADMediatedClassifier`、`vad_downstream/train_vad_emotion.py`、`vad_downstream/infer_vad_emotion.py`、`tests/test_vad_downstream_train_vad_emotion.py`、`tests/test_vad_downstream_infer_vad_emotion.py` | 実装はあるが、研究上は時間に余裕がある場合の探索条件である。 |
| 実emotion2vecエンコーダーによる一連の学習・評価 | 未検証 | `vad_downstream/model.py`には`freeze_encoder=True`の波形入力wrapperがあり、`notebooks/audio_to_emotion_vad.ipynb`には実特徴抽出の分岐がある一方、現存する実行生成物はデモモード | 実モデル、実音声、固定エンコーダー、デコーダー学習、評価までを連続して完了した証拠はない。 |
| emotion2vec+ large学習パイプライン | 未着手 | 本計画には比較方針を定義したが、`vad_downstream/data.py`と`vad_downstream/notebook_pipeline.py`の`FEATURE_DIM`、`iemocap_downstream/main.py`のモデル生成などは768次元固定 | 比較方針だけがあり、Large特徴を用いた学習・評価入口は未対応である。 |
| Base / Large共通デコーダーの部品 | 部分実装 | `vad_downstream/model.py`の各headは`input_dim`引数を持つが、データ読込、Notebook、設定、IEMOCAP経路には768次元固定が残る | モデル単体は入力次元を受け取れるが、パイプライン全体をエンコーダーごとに切り替えられない。 |
| 自動テスト | 実装済み | `tests/test_*.py`の`test_`メソッドを静的に数えると65件。並列型、VAD媒介型、データ、学習、推論、Notebook補助処理を含む | テスト資産は存在する。 |
| 現環境での自動テスト結果 | 未検証 | `py -m unittest discover -s tests --list-tests`は利用可能なPythonを見つけられず終了した。テスト群は`torch`と`soundfile`をimportし、現環境には両依存を利用できるPython環境がない | 65件を実行できていない。成功扱いにしない。 |
| デモNotebook実行補助スクリプト | 部分実装 | `tests/execute_demo_notebook.py`は`inference_results.csv`を必須生成物とするが、現行`notebooks/audio_to_emotion_vad.ipynb`は別WAV推論を実行しないと明記し、同CSVも現存しない | 依存環境を整えた後の再実行前に、Notebookと補助スクリプトの生成物契約をそろえる必要がある。 |
| HCUDB実データによる日本語SER結果 | 未着手 | リポジトリ内にHCUDB実験のmetrics、checkpoint、比較表がない | 日本語性能向上を示す研究結果は未取得である。 |
| IEMOCAPによる英語性能維持結果 | 未着手 | `iemocap_downstream/`に参照実装はあるが、今回の共通条件による評価生成物がない | 英語性能維持は未確認である。 |
| Base対Large結果 | 未着手 | Large側の共通学習入口と比較生成物がない | 研究の主比較結果はまだ存在しない。 |

## Large対応上の具体的な障害

以下の768次元固定を、単なるモデル既定値とパイプライン上の制約に分けて解消する必要がある。

- データ読込: `vad_downstream/data.py`の`FEATURE_DIM = 768`
- Notebook補助処理: `vad_downstream/notebook_pipeline.py`の`FEATURE_DIM = 768`と768次元の仮特徴抽出器
- Notebook: `notebooks/vad_model_check.ipynb`の`INPUT_DIM = 768`、`notebooks/audio_to_emotion_vad.ipynb`の768次元前提
- IEMOCAP学習: `iemocap_downstream/main.py`の`BaseModel(input_dim=768, ...)`
- IEMOCAP推論Notebook: `iemocap_downstream/inference.ipynb`の768次元モデル・入力
- 設定と既定値: `vad_downstream/config/default.yaml`、`vad_downstream/model.py`、各推論コードの768次元fallback

`vad_downstream/model.py`のデコーダーはすでに`input_dim`を受け取れるため、次段階では外部インターフェースを壊さず、データ層と実験入口からエンコーダー固有の次元を一貫して渡す設計が中心になる。

## デモ結果と研究結果の境界

### デモ検証済みといえる範囲

- 合成音声の作成
- 話者分割とtrain-only正規化
- 仮特徴のcache
- 直接感情分類＋並列VA/Dデコーダーの学習
- checkpoint、学習履歴、test metrics、評価図の生成
- Dominance正解がない場合の`untrained`表示と警告

### 研究結果として未取得の範囲

- 実emotion2vec Base特徴によるHCUDB性能
- 実emotion2vec+ large特徴によるHCUDB性能
- IEMOCAPでの英語性能維持
- 同一条件のBase対Large比較
- 複数seedでの再現性と統計的比較

デモの`test_metrics.json`、評価図、推論CSVに数値が含まれていても、仮特徴と合成音声に基づくため研究成果へ転記しない。`--allow-random-head`または`--allow-random-model`による出力も同様である。

## 次の最小作業

次の最小作業は「特徴次元の可変化とBase / Large共通デコーダー実験入口の設計」とする。

完了判定は、BaseとLargeについてデコーダーの`input_dim`だけを変更でき、隠れ層以降の構造、データ分割、感情ラベル、学習条件、評価指標を同一設定から生成できる設計が文書化され、対応テストが定義された状態とする。実データと依存環境がない現時点では、新しい性能値を生成しない。
