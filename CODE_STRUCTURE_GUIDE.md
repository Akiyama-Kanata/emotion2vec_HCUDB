# emotion2vec 感情認識モデル構造・コード対応ガイド

本ドキュメントは、音声感情認識モデル（VAD媒介型感情分類モデル）の処理の流れと、それを実装しているソースコードの具体的な位置をマッピングしたガイドです。

---

## 1. 感情認識モデルの処理フローとコードの対応

ユーザーの認識されている処理フロー：
`emotion2vec ➔ pooling ➔ FNN ➔ valence, arousal, dominance ➔ linear ➔ 推定感情の分類`

この流れとコードの構成に**齟齬や矛盾は一切ありません。** 以下に、各処理ステップに対応する具体的なクラス名、メソッド名、およびファイルへのリンクを示します。

### ① 特徴抽出（`emotion2vec`）
* **役割**: 音声波形（WAV）を入力とし、20msごとの感情的な音響特徴量（768次元）をフレーム単位で抽出します。
* **コード箇所**:
  * モデル本体: [upstream/models/emotion2vec.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/upstream/models/emotion2vec.py) の `Data2VecMultiModel` クラス。
  * 特徴抽出メソッド: [extract_features メソッド](file:///c:/Users/RD004/Documents/lab/emotion2vec/upstream/models/emotion2vec.py#L177)
  * VADモデル側の呼び出し口: [Emotion2vecVADModel.extract_frame_features](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py#L49)
* **データの形**: `[Batch, Time, 768]` （時間軸方向にフレームが並んでいます）

### ② 時間方向の圧縮（`pooling`）
* **役割**: 音声の長さ（フレーム数）が発話ごとに異なるため、時間軸方向に平均化して固定長の「1つの発話ベクトル」に変換します。
* **コード箇所**:
  * [vad_downstream/model.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py) 内の [masked_mean_pooling 関数](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py#L217)
  * パディング（無音部分の埋め合わせ）を無視して正しく平均をとる設計になっています。
* **データの形**: `[Batch, 768]`

### ③ 感情空間への投影（`FNN ➔ valence, arousal, dominance`）
* **役割**: プーリングされた 768次元の情報を、ニューラルネットワーク（FNN）を通して整理し、感情の3次元座標（VAD値）へ変換します。
* **コード箇所**:
  * [vad_downstream/model.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py) 内の [VADRegressionHead クラス](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py#L77)
  * **FNN（中間層＋活性化関数）の定義**:
    * `self.pre_net = nn.Linear(768, 256)` （768次元から256次元に圧縮）
    * `self.activate = nn.ReLU()` （非線形活性化）
    * `self.post_net = nn.Linear(256, 3)` （256次元からVADの3次元に変換）
* **データの形**: `[Batch, 3]` （各サンプルが [Valence, Arousal, Dominance] の3つの数値になります）

### ④ 感情クラス判定（`linear ➔ 推定感情の分類`）
* **役割**: 抽出した 3次元の VAD 値を受け取り、最終的な感情カテゴリ（喜び、悲しみ、怒りなど）の確率スコア（Logits）に変換します。
* **コード箇所**:
  * [vad_downstream/model.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py) 内の [VADClassificationHead クラス](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/model.py#L117)
  * **線形層の定義**:
    * `self.linear = nn.Linear(3, num_classes)` （3次元からクラス数（例: 4）へ変換）
* **データの形**: `[Batch, num_classes]` （最もスコアが高いクラスが予測感情になります）

---

## 2. その他の重要ファイルのガイド

モデルの学習や推論の流れを追うための追加ガイドです。

### 🔄 学習・誤差計算処理
* **[vad_downstream/train_vad_emotion.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/train_vad_emotion.py)**
  * VAD媒介型感情分類器の学習を実行するコマンドラインスクリプトです。データの読み込みからエポックのループまでを制御します。
* **[vad_downstream/emotion_training.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/emotion_training.py)**
  * **誤差計算**: [compute_vad_emotion_loss 関数](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/emotion_training.py#L17)
    * `lambda_vad * VADのCCC損失` ＋ `lambda_emo * 感情分類のCrossEntropy損失` を合算して全体の Loss とします。
  * **1エポックの学習**: [train_one_epoch 関数](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/emotion_training.py#L43)
  * **評価処理**: [evaluate 関数](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/emotion_training.py#L95)
    * 予測値（VADおよび感情クラス）を正解ラベルと比較し、WA (Weighted Accuracy)、UA (Unweighted Accuracy)、CCC などの評価指標を計算します。
* **[vad_downstream/training.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/training.py)**
  * VAD回帰に特化した CCC (Concordance Correlation Coefficient) の計算ロジックが定義されています。

### 📈 データ読み込み処理
* **[vad_downstream/data.py](file:///c:/Users/RD004/Documents/lab/emotion2vec/vad_downstream/data.py)**
  * 特徴データ（`.npy`）、発話の長さ（`.lengths`）、VAD値（`.vad`）、感情ラベル（`.emo`）をパディング付きでバッチ化して PyTorch に供給するデータローダーです。
