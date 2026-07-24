# emotion2vec 下流モデル構造

この文書では、固定した emotion2vec 特徴を利用する3種類の下流モデルを
比較する。今回追加した主モデルは「3. 並列感情分類・条件付きVADモデル」
であり、既存の2モデルは比較実験用として維持する。

## 共通の入力

学習時は、事前抽出した768次元のフレーム特徴を使用する。

```text
features:     Tensor[B, T, 768]
padding_mask: BoolTensor[B, T]
```

- `B`: バッチサイズ
- `T`: バッチ内でpaddingされたフレーム数
- `padding_mask=True`: paddingされた無効フレーム
- emotion2vec本体は固定し、下流ヘッドだけを学習する

すべてのモデルは最初にmasked mean poolingを行う。

```text
pooled[b] = 有効フレームの features[b] の平均
pooled: Tensor[B, 768]
```

padding部分は平均値に含まれない。全フレームがpaddingの発話はエラーになる。

## 3モデルの比較

| モデル | VAD出力 | 感情分類 | 分類とVADの依存関係 | 主な用途 |
|---|---:|---:|---|---|
| 直接VA/VAD回帰 | 2次元または3次元 | なし | － | VAD回帰の基準モデル |
| VAD経由感情分類 | 2次元または3次元 | あり | 分類が予測VADに依存 | 解釈可能性を重視した比較モデル |
| 並列感情分類・条件付きVAD | 常に3次元 | あり | 分類とV/A/Dが互いに独立 | 今回追加した主モデル |

## 1. 直接VA/VAD回帰モデル

実装クラスは `VADRegressionHead` である。

```text
emotion2vec特徴 [B, T, 768]
        │
        ▼
masked mean pooling
        │ [B, 768]
        ▼
Linear(768 → hidden_dim)
        ▼
ReLU
        ▼
Linear(hidden_dim → target_dim)
        │
        ▼
VAまたはVAD [B, target_dim]
```

`target_dim=2`ではValence/Arousal、`target_dim=3`では
Valence/Arousal/Dominanceをまとめて回帰する。

## 2. VAD経由感情分類モデル

実装クラスは `VADMediatedEmotionClassifier` である。

```text
emotion2vec特徴 [B, T, 768]
        │
        ▼
masked mean pooling
        │
        ▼
VADRegressionHead
        │
        ▼
予測VA/VAD [B, target_dim]
        │
        ▼
Linear(target_dim → num_classes)
        │
        ▼
感情logits [B, num_classes]
```

分類logitsは予測VADだけから計算される。そのため、分類損失の勾配も
分類層からVAD回帰ヘッドへ流れる。

```text
CrossEntropyLoss
        │
        ├─ 分類層を更新
        └─ VAD回帰ヘッドも更新
```

予測したVADが分類根拠になる一方、分類性能がVAD表現に影響する構造である。

## 3. 並列感情分類・条件付きVADモデル

実装クラスは `ParallelEmotionVADClassifier` である。

```text
emotion2vec特徴 [B, T, 768]
        │
        ▼
masked mean pooling
        │ [B, 768]
        ├─────────────────────────────────────────┐
        │                 │           │           │
        ▼                 ▼           ▼           ▼
感情分類ヘッド       Valenceヘッド  Arousalヘッド Dominanceヘッド
Linear               Linear         Linear        Linear
  ↓ ReLU               ↓ ReLU         ↓ ReLU        ↓ ReLU
Linear(C)            Linear(1)      Linear(1)     Linear(1)
        │                 │           │           │
        ▼                 ▼           ▼           ▼
logits [B,C]          V [B,1]      A [B,1]      D [B,1]
                          └───────────┴───────────┘
                                      │ concat
                                      ▼
                                  VAD [B,3]
```

公開出力は常に次の形式になる。

```python
{
    "logits": Tensor[B, num_classes],
    "vad": Tensor[B, 3],  # Valence, Arousal, Dominance
}
```

### ヘッドを独立させる理由

分類ヘッドとV/A/Dの各ヘッドはパラメータを共有しない。共有するのは、
学習対象ではない入力のpooled emotion2vec特徴だけである。

```text
CrossEntropyLoss ──→ 感情分類ヘッドだけを更新
Valence CCC loss ──→ Valenceヘッドだけを更新
Arousal CCC loss ──→ Arousalヘッドだけを更新
Dominance CCC loss → Dominanceヘッドだけを更新
```

この構造により、次の分離が保証される。

- 分類結果は予測VADに依存しない
- 分類損失からV/A/Dヘッドへ勾配が流れない
- VAD損失から分類ヘッドへ勾配が流れない
- V/A学習によってDヘッドが間接更新されない

## 教師データとマスク

`.vad`ファイルは発話ごとにVAまたはVADを記述でき、同一ファイル内で
混在できる。

```text
utt_va<TAB>valence<TAB>arousal
utt_vad<TAB>valence<TAB>arousal<TAB>dominance
```

ローダーは内部表現を常に3次元に揃える。

```python
{
    "emotion_target": LongTensor[B],
    "vad_target": FloatTensor[B, 3],
    "vad_target_mask": BoolTensor[B, 3],
}
```

例:

```text
元データ: utt_va   0.1   0.2
target:            [0.1, 0.2, 0.0]
mask:              [True, True, False]

元データ: utt_vad  0.1   0.2  -0.1
target:            [0.1, 0.2, -0.1]
mask:              [True, True, True]
```

D欠損時の`0.0`はダミー値であり、損失や評価には使用されない。
ValenceとArousalは全発話で必須である。

## 条件付きDominance学習

Dominanceヘッドの扱いは、train split全体のD教師数で決まる。

```text
train splitのD教師数
        │
        ├─ 1件以上
        │    ├─ Dヘッドをoptimizerへ含める
        │    └─ 各バッチのD教師が2件以上のときだけD CCC lossを計算
        │
        └─ 0件
             ├─ Dヘッドをoptimizerから除外
             └─ Dヘッドの全パラメータを完全に固定
```

D教師が1件だけのバッチでは、CCCを安定して計算できないためD lossを
スキップする。AdamWのweight decayによる変化も防ぐため、そのバッチでは
D勾配を`None`にしてからoptimizerを更新する。

## 損失関数

V/Aは全サンプル、DはマスクされたD教師付きサンプルだけでCCC lossを
計算する。有効な各次元のlossを平均したものが`masked_ccc_loss`である。

```text
total_loss
  = lambda_vad × masked_ccc_loss
  + lambda_emo × CrossEntropyLoss
```

バッチ内に有効なD CCC lossがない場合、V/A lossだけが
`masked_ccc_loss`へ含まれる。

## Dominance状態

checkpointと推論JSONにはD出力の解釈を示す状態を保存する。

| 状態 | 意味 | Dヘッドの扱い |
|---|---|---|
| `trained` | 現在のtrain splitにD教師が存在した | D教師付き発話から学習 |
| `untrained` | 新規モデルにD教師が存在しなかった | ランダム初期値のまま固定 |
| `retained_from_checkpoint` | D学習済みcheckpointをVAのみで追加学習した | 過去のDパラメータを固定保持 |

`untrained`でも出力schemaを一定にするためD値を数値として返す。ただし、
学習済みのDominance推定値ではないため、推論JSONへ警告を含める。

## Checkpoint

並列モデルのcheckpointには、少なくとも次の情報を保存する。

```text
model_type: parallel_emotion_vad
model_state_dict
input_dim / hidden_dim / num_classes
class_labels / class_names_ja
vad_label_counts
supervised_dimensions
dominance_status
lambda_vad / lambda_emo
metadata
```

`class_labels`は学習時の指定順で保存される。推論時はcheckpointから
クラス数と順序を復元するため、CLI側で再指定する必要はない。

## 評価指標

感情分類について次を出力する。

- WA
- UA
- weighted F1
- macro F1
- confusion matrix
- クラス別recall、F1、support

VADは教師が存在する次元だけCCCを計算する。評価データにD教師が
存在しない場合、`dominance_ccc`は`null`になる。

## 単一WAV推論

WAV推論では、音声から固定emotion2vec特徴を抽出して並列ヘッドへ入力する。

```text
16 kHz mono WAV
      │
      ▼
固定 emotion2vec encoder
      │ [1, T, 768]
      ▼
ParallelEmotionVADClassifier
      │
      ├─ 予測クラス
      ├─ 全クラスprobabilities
      ├─ 全クラスlogits
      └─ 状態付きValence/Arousal/Dominance
```

VAD部分のJSON例:

```json
{
  "vad": {
    "valence": {"value": 0.1, "status": "trained"},
    "arousal": {"value": 0.2, "status": "trained"},
    "dominance": {"value": -0.1, "status": "untrained"}
  },
  "warning": "Dominance is emitted numerically but its head has no supervised training and is not a learned dominance estimate."
}
```

## 実装対応表

| 機能 | ファイル |
|---|---|
| モデル構造 | `vad_downstream/model.py` |
| データロードとcollator | `vad_downstream/data.py` |
| masked CCC・評価・checkpoint | `vad_downstream/parallel_training.py` |
| 事前抽出特徴からの学習CLI | `vad_downstream/train_parallel_emotion_vad.py` |
| 単一WAV推論CLI | `vad_downstream/infer_parallel_emotion_vad.py` |
| モデル・勾配・状態テスト | `tests/test_parallel_emotion_vad.py` |
