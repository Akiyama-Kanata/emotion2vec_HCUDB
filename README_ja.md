<div align="center">
    <h1>
    EMOTION2VEC
    </h1>
    <p>
    <b><em>emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation</em></b><br>
    の特徴抽出と下流モデル学習のための公式 PyTorch コード
    </p>
    <p>
    <img src="src/logo.png" alt="emotion2vec Logo" style="width: 200px; height: 200px;">
    </p>
    <p>
    </p>
    <a href="https://github.com/ddlBoJack/emotion2vec"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/ddlBoJack/emotion2vec"><img src="https://img.shields.io/badge/Python-3.8+-orange" alt="version"></a>
    <a href="https://github.com/ddlBoJack/emotion2vec"><img src="https://img.shields.io/badge/PyTorch-1.13+-brightgreen" alt="python"></a>
    <a href="https://github.com/ddlBoJack/emotion2vec"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="mit"></a>
</div>

<p align="center">
  <a href="README.md">English</a> | <a href="README_ja.md">日本語</a>
</p>

# ニュース
- [2024年10月] 🔧 FunASR インターフェースの使い方を更新し、ソース選択に対応しました。中国本土のユーザーは `"ms"` または `"modelscope"`、その他の海外ユーザーは `"hf"` または `"huggingface"` を指定してください。**スムーズに利用を始めるには FunASR インターフェースの利用を推奨します。**
- [2024年6月] 🔧 emotion2vec+ のバグを修正しました。最新コードを再度取得してください。
- [2024年5月] 🔥 9 クラス感情に対応した音声感情認識の基盤モデル **emotion2vec+** を [Model Scope](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) と [Hugging Face](https://huggingface.co/emotion2vec) で公開しました。高性能な SER 向けモデルとして、emotion2vec+ の各モデル(seed、base、large)を確認してください。**2024年1月リリースではなく、このリリースの利用を推奨します。**
- [2024年1月] emotion2vec から反復的にファインチューニングした 9 クラス感情認識モデルを [modelscope](https://www.modelscope.cn/models/iic/emotion2vec_base_finetuned/summary) と [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) で公開しました。
- [2024年1月] **emotion2vec** が [modelscope](https://www.modelscope.cn/models/iic/emotion2vec_base/summary) と [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) に統合されました。
- [2023年12月] [論文](https://arxiv.org/abs/2312.15185)を公開し、emotion2vec の [WeChat グループ](./src/Wechat.jpg)を作成しました。
- [2023年11月] emotion2vec のコード、チェックポイント、抽出済み特徴量を公開しました。

# モデルカード
GitHub リポジトリ: [emotion2vec](https://github.com/ddlBoJack/emotion2vec)

|Model|⭐Model Scope|🤗Hugging Face|Fine-tuning Data (Hours)|
|:---:|:-------------:|:-----------:|:-------------:|
|emotion2vec|[Link](https://www.modelscope.cn/models/iic/emotion2vec_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_base)|/|
|emotion2vec+ seed|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_seed/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)|201|
|emotion2vec+ base|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_base)|4788|
|emotion2vec+ large|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_large)|42526|

# 概要

- [emotion2vec+: 音声感情認識の基盤モデル](#emotion2vec-音声感情認識の基盤モデル)
  - [ガイド](#ガイド)
  - [データエンジニアリング](#データエンジニアリング)
  - [性能](#性能)
  - [チェックポイントを使った推論](#チェックポイントを使った推論)
    - [FunASR からインストール](#funasr-からインストール)
- [emotion2vec: 汎用音声感情表現モデル](#emotion2vec-汎用音声感情表現モデル)
  - [ガイド](#ガイド-1)
  - [性能](#性能-1)
    - [IEMOCAP での性能](#iemocap-での性能)
    - [その他の言語での性能](#その他の言語での性能)
    - [その他の音声感情タスクでの性能](#その他の音声感情タスクでの性能)
  - [可視化](#可視化)
  - [特徴量抽出](#特徴量抽出)
    - [抽出済み特徴量のダウンロード](#抽出済み特徴量のダウンロード)
    - [自分のデータセットから特徴量を抽出](#自分のデータセットから特徴量を抽出)
      - [ソースコードからインストール](#ソースコードからインストール)
      - [FunASR からインストール](#funasr-からインストール-1)
  - [下流モデルの学習](#下流モデルの学習)
  - [コントリビューター](#コントリビューター)
  - [引用](#引用)

# emotion2vec+: 音声感情認識の基盤モデル

## ガイド
emotion2vec+ は、音声感情認識(SER)のための基盤モデルシリーズです。私たちは、音声感情認識の分野における「whisper」のようなモデルを学習することを目指しています。データ駆動の手法によって、言語や録音環境の影響を克服し、汎用的で堅牢な感情認識能力を実現します。emotion2vec+ の性能は、Hugging Face で多くダウンロードされている他のオープンソースモデルを大きく上回ります。

![](./src/emotion2vec+radar.png)

## データエンジニアリング
emotion2vec+ には 3 つのバージョンがあり、それぞれ前身モデルのデータから派生しています。音声感情表現に焦点を当てたモデルが必要な場合は、[emotion2vec: 汎用音声感情表現モデル](#emotion2vec-汎用音声感情表現モデル)を参照してください。

- emotion2vec+ seed: [EmoBox](https://github.com/emo-box/EmoBox) の学術的な音声感情データでファインチューニング
- emotion2vec+ base: フィルタリング済みの大規模疑似ラベルデータでファインチューニングし、base サイズモデル(約 90M)を取得
- emotion2vec+ large: フィルタリング済みの大規模疑似ラベルデータでファインチューニングし、large サイズモデル(約 300M)を取得

反復プロセスを下図に示します。最終的に、16 万時間の音声感情データのうち 4 万時間を用いて emotion2vec+ large モデルを学習しています。データエンジニアリングの詳細は後日公開予定です。

## 性能

[EmoBox](https://github.com/emo-box/EmoBox) における 4 クラス主要感情での性能です(ファインチューニングなし)。モデル性能の詳細は後日公開予定です。

![](./src/emotion2vec+performance.png)

## チェックポイントを使った推論

### FunASR からインストール
1. funasr をインストールします。

```bash
pip install -U funasr
```

2. コードを実行します。

```python
'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''

from funasr import AutoModel

# model="iic/emotion2vec_base"
# model="iic/emotion2vec_base_finetuned"
# model="iic/emotion2vec_plus_seed"
# model="iic/emotion2vec_plus_base"
model_id = "iic/emotion2vec_plus_large"

model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

wav_file = f"{model.model_path}/example/test.wav"
rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False)
print(rec_result)
```

モデルは自動的にダウンロードされます。

FunASR は wav.scp(kaldi 形式)でのファイルリスト入力をサポートしています。

```text
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```

詳細は [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) を参照してください。

# emotion2vec: 汎用音声感情表現モデル

## ガイド

emotion2vec は、初の汎用音声感情表現モデルです。自己教師あり事前学習により、emotion2vec はさまざまなタスク、言語、シナリオをまたいで感情表現を抽出できます。

## 性能

### IEMOCAP での性能
emotion2vec は、主流の IEMOCAP データセットにおいて、線形層のみで SOTA を達成しています。詳細は論文を参照してください。

![](./src/IEMOCAP.png)

### その他の言語での性能
emotion2vec は、複数の言語(中国語、フランス語、ドイツ語、イタリア語など)において、SOTA の SSL モデルと比較して SOTA を達成しています。詳細は論文を参照してください。

![](./src/Languages.png)

### その他の音声感情タスクでの性能
詳細は論文を参照してください。

## 可視化
IEMOCAP データセット上で学習済み特徴量を UMAP 可視化したものです。<span style="color:red;">赤</span>系と <span style="color:blue;">青</span>系は、それぞれ低覚醒と高覚醒の感情クラスを表します。詳細は論文を参照してください。

![](./src/UMAP.png)

## 特徴量抽出

### 抽出済み特徴量のダウンロード
代表的な感情データセット IEMOCAP の抽出済み特徴量を提供しています。特徴量は emotion2vec の最終層から抽出されています。特徴量は `.npy` 形式で保存されており、抽出された特徴量のサンプルレートは 50Hz です。発話レベルの特徴量は、フレームレベル特徴量の平均によって計算しています。

- frame-level: [Google Drive](https://drive.google.com/file/d/1JdQzwDJJEdKZcqSC1TXETvFZ7VpUvLEX/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1FtCwhUwhONaeEos4nLYFWw?pwd=zb3p) (password: zb3p)
- utterance-level: [Google Drive](https://drive.google.com/file/d/1jJVfoEKC8yjwj39F__8jIQayd5PBO0WD/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1AsJHacD6a5h27YJiCSee4w?pwd=qu3u) (password: qu3u)

多様な下流タスクのため、すべての wav ファイルは元データセットから抽出されています。標準的な 5531 発話、4 感情分類で学習したい場合は、`iemocap_downstream` フォルダーを参照してください。

### 自分のデータセットから特徴量を抽出

#### ソースコードからインストール
最小環境要件は `python>=3.8` と `torch>=1.13` です。テスト環境は `python=3.8` と `torch=2.01` です。

1. リポジトリを clone します。

```bash
pip install fairseq
git clone https://github.com/ddlBoJack/emotion2vec.git
```

2. emotion2vec のチェックポイントを以下からダウンロードします。

- [Google Drive](https://drive.google.com/file/d/10L4CEoEyt6mQrqdblDgDSfZETYvA9c2T/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/15zqmNTYa0mkEwlIom7DO3g?pwd=b9fq) (password: b9fq)
- [modelscope](https://www.modelscope.cn/models/damo/emotion2vec_base/summary): `git clone https://www.modelscope.cn/damo/emotion2vec_base.git`

3. `scripts/extract_features.sh` を修正して実行します。

#### FunASR からインストール
1. funasr をインストールします。

```bash
pip install -U funasr
```

2. コードを実行します。

```python
'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''

from funasr import AutoModel

model_id = "iic/emotion2vec_base"
model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

wav_file = f"{model.model_path}/example/test.wav"
rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance")
print(rec_result)
```

モデルは自動的にダウンロードされます。

FunASR は wav.scp(kaldi 形式)でのファイルリスト入力をサポートしています。

```text
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```

詳細は [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) を参照してください。

## 下流モデルの学習
IEMOCAP データセット用の学習スクリプトを `iemocap_downstream` フォルダーで提供しています。スクリプトを変更することで、他のデータセット上で下流モデルを学習できます。

## コントリビューター

| Institution | Contribution |
|:------|:-----|
| [Shanghai Jiao Tong University](https://www.seiee.sjtu.edu.cn/) | Researchers; Computing power; Data collection; |
| [Fudan University](https://istbi.fudan.edu.cn/) | Researchers |
| [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/chinese/index.html) | Researchers |
| [Alibaba Group](https://www.alibaba.com/) | Researchers; Computing power; Data host; Model host; |
| [Peng Cheng Laboratory](https://data-starcloud.pcl.ac.cn/) | Researchers |

## 引用
emotion2vec のコードや論文が役に立った場合は、以下を引用してください。

```text
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={Proc. ACL 2024 Findings},
  year={2024}
}
```
