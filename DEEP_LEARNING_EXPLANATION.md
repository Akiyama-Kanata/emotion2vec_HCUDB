# emotion2vecリポジトリの学習構造まとめ

## 全体像

このリポジトリは、大きく分けて次の3つの部分で構成されています。

1. `upstream/`  
   emotion2vec本体。音声から感情に関係する特徴量を抽出するモデル。

2. `iemocap_downstream/`  
   emotion2vec特徴量を使って、IEMOCAPの4クラス感情分類を行うモデル。

3. `vad_downstream/`  
   emotion2vec特徴量を使って、Valence / Arousal / Dominance の連続値を予測する回帰モデル。

基本的な流れは次の通りです。

```text
音声 waveform
  ↓
emotion2vec 本体
  ↓
frame-level features: [時間, 768]
  ↓
下流モデル
  ↓
感情分類 または VA/VAD回帰
```

## モデルは何層か

emotion2vec本体は、主に以下の構成です。

| 部分 | 内容 |
|---|---|
| CNN特徴抽出器 | 7層 |
| audio prenet | Transformer風ブロック 4層 |
| メインTransformer | 8層 |
| 特徴次元 | 768 |
| Attention head数 | 12 |

そのため、Transformer風ブロックとしては `4 + 8 = 12層` と見るのが自然です。

一方、下流モデルはかなり小さいです。

### IEMOCAP分類モデル

```text
Linear(768 → 256)
  ↓
ReLU
  ↓
平均pooling
  ↓
Linear(256 → 4)
```

4クラスは以下です。

```text
angry / happy / neutral / sad
```

### VA/VAD回帰モデル

```text
平均pooling
  ↓
Linear(768 → hidden_dim)
  ↓
ReLU
  ↓
Linear(hidden_dim → 2 or 3)
```

出力は次のどちらかです。

```text
VA:  valence, arousal
VAD: valence, arousal, dominance
```

## 活性化関数

活性化関数は、ニューラルネットワークに非線形性を与える関数です。

線形層だけを何層重ねても、結局は1つの線形変換と同じような表現しかできません。  
そこで、`ReLU` や `GELU` のような活性化関数を挟むことで、複雑な関係を学習できるようにします。

このリポジトリでは主に以下が使われています。

| 場所 | 活性化関数 |
|---|---|
| IEMOCAP分類モデル | ReLU |
| VA/VAD回帰head | ReLU |
| emotion2vec本体 | GELU |

分類モデルの最後には `Softmax` は明示されていません。  
これは、PyTorchの `CrossEntropyLoss` が内部で `log_softmax` 相当の処理を行うためです。

## 損失関数

### IEMOCAP分類

IEMOCAPの4クラス分類では、損失関数として `CrossEntropyLoss` が使われています。

```text
予測logits と 正解ラベル を比較する分類用の損失
```

正解クラスのスコアが高いほど損失は小さくなります。

### VA/VAD回帰

VA/VAD回帰では、`CCC loss` が使われています。

CCC は Concordance Correlation Coefficient の略です。  
単なる誤差だけでなく、以下をまとめて評価します。

- 予測と正解の相関
- 平均値のズレ
- 分散のズレ

損失は次の形です。

```text
loss = 1 - mean(CCC)
```

完全に一致すると、

```text
CCC = 1
loss = 0
```

になります。

## 勾配の扱い

学習処理は、典型的なPyTorchの流れです。

```python
optimizer.zero_grad()
prediction = model(...)
loss = criterion(prediction, target)
loss.backward()
optimizer.step()
```

それぞれの意味は次の通りです。

| 処理 | 意味 |
|---|---|
| `optimizer.zero_grad()` | 前のbatchの勾配を消す |
| `model(...)` | 予測を出す |
| `criterion(...)` | 損失を計算する |
| `loss.backward()` | 誤差逆伝播で勾配を計算する |
| `optimizer.step()` | 勾配を使って重みを更新する |

VA/VADモデルでは、デフォルトで emotion2vec本体を固定します。

```text
freeze_encoder=True
```

つまり、emotion2vec本体の重みは更新せず、後ろに付けた回帰headだけを学習します。

このリポジトリ内では、明示的な gradient clipping は見当たりません。

## 主なハイパーパラメータ

### emotion2vec本体

| パラメータ | 値 |
|---|---:|
| `embed_dim` | 768 |
| `depth` | 8 |
| `num_heads` | 12 |
| `mlp_ratio` | 4 |
| `encoder_dropout` | 0.1 |
| `average_top_k_layers` | 8 |
| `mask_prob` | 0.7 |
| `mask_length` | 5 |

### IEMOCAP分類

| パラメータ | 値 |
|---|---:|
| `batch_size` | 128 |
| `epoch` | 100 |
| `lr` | 5e-4 |
| optimizer | RMSprop |
| scheduler | CyclicLR |
| 出力クラス数 | 4 |

### VA/VAD回帰

| パラメータ | 値 |
|---|---:|
| `epochs` | 10 |
| `batch_size` | 32 |
| `lr` | 1e-3 |
| `weight_decay` | 0.0 |
| `hidden_dim` | 256 |
| optimizer | AdamW |
| `target_dim` | 2 or 3 |

## まとめ

このリポジトリの基本思想は、次のようにまとめられます。

```text
大きなemotion2vec本体で音声特徴を抽出し、
小さな下流モデルで分類や回帰を行う
```

初心者が読むなら、まずは以下の順番がおすすめです。

1. `vad_downstream/model.py`  
   小さな回帰モデルの構造がわかりやすい。

2. `vad_downstream/training.py`  
   損失関数、勾配計算、重み更新の流れが見やすい。

3. `iemocap_downstream/model.py`  
   分類モデルの基本構造がわかる。

4. `upstream/models/`  
   emotion2vec本体のTransformer構造を確認できる。
