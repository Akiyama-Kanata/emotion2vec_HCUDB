<div align="center">
    <h1>
    EMOTION2VEC
    </h1>
    <p>
    Official PyTorch code for extracting features and training downstream models with <br>
    <b><em>emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation</em></b>
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

# Repository focus

This workspace keeps the original emotion2vec implementation, and now uses it
primarily as a feature extractor for **VAD regression**:

- input labels: `file_path, valence, arousal, dominance, split, session`
- feature input: cached emotion2vec `.npy` frame features
- main entrypoint: `vad_downstream/train_vad.py`
- output order: `valence, arousal, dominance`

The original IEMOCAP downstream classifier remains under `iemocap_downstream/`
as a reference implementation. Old VAD-as-intermediate-classification experiments
were moved to `archive/vad_iemocap_two_stage/`.

# リポジトリの焦点

このワークスペースは元の emotion2vec 実装を保持しつつ、現在は主に
**VAD 回帰**のための特徴抽出器として利用します。

- 入力ラベル: `file_path, valence, arousal, dominance, split, session`
- 特徴量入力: キャッシュ済みの emotion2vec `.npy` フレーム特徴量
- メインエントリポイント: `vad_downstream/train_vad.py`
- 出力順: `valence, arousal, dominance`

元の IEMOCAP 下流分類器は、参照実装として `iemocap_downstream/` に残しています。
VAD を中間表現として使っていた古い分類実験は
`archive/vad_iemocap_two_stage/` に移動しました。

# News
- [Oct. 2024] 🔧 We update the usage in the FunASR interface with source selection. "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users. **We recommend using FunASR interface for a smooth landing.**
- [Jun. 2024] 🔧 We fix a bug in emotion2vec+. Please re-pull the latest code. 
- [May. 2024] 🔥 Speech emotion recognition foundation model: **emotion2vec+**, with 9-class emotions has been released on [Model Scope](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) and [Hugging Face](https://huggingface.co/emotion2vec). Check out a series of emotion2vec+ (seed, base, large) models for SER with high performance **(We recommend this release instead of the Jan. 2024 release)**. 
- [Jan. 2024] 9-class emotion recognition model with iterative fine-tuning from emotion2vec has been released in [modelscope](https://www.modelscope.cn/models/iic/emotion2vec_base_finetuned/summary) and [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec). 
- [Jan. 2024] **emotion2vec** has been integrated into [modelscope](https://www.modelscope.cn/models/iic/emotion2vec_base/summary) and [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec).  
- [Dec. 2023] We release the [paper](https://arxiv.org/abs/2312.15185), and create a [WeChat group](./src/Wechat.jpg) for emotion2vec. 
- [Nov. 2023] We release code, checkpoints, and extracted features for emotion2vec. 

# Model Card
GitHub Repo: [emotion2vec](https://github.com/ddlBoJack/emotion2vec)
|Model|⭐Model Scope|🤗Hugging Face|Fine-tuning Data (Hours)|
|:---:|:-------------:|:-----------:|:-------------:|
|emotion2vec|[Link](https://www.modelscope.cn/models/iic/emotion2vec_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_base)|/|
|emotion2vec+ seed|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_seed/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)|201|
|emotion2vec+ base|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_base)|4788|
|emotion2vec+ large|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_large)|42526|

# Overview

- [emotion2vec+: speech emotion recognition foundation model](#emotion2vec-speech-emotion-recognition-foundation-model)
  - [Guides](#guides)
  - [Data Engineering](#data-engineering)
  - [Performance](#performance)
  - [Inference with checkpoints](#inference-with-checkpoints)
    - [Install from FunASR](#install-from-funasr)
- [emotion2vec: universal speech emotion representation model](#emotion2vec-universal-speech-emotion-representation-model)
  - [Guides](#guides-1)
  - [Performance](#performance-1)
    - [Performance on IEMOCAP](#performance-on-iemocap)
    - [Performance on other languages](#performance-on-other-languages)
    - [Performance on other speech emotion tasks](#performance-on-other-speech-emotion-tasks)
  - [Visualization](#visualization)
  - [Extract features](#extract-features)
    - [Download extracted features](#download-extracted-features)
    - [Extract features from your dataset](#extract-features-from-your-dataset)
      - [Install from the source code](#install-from-the-source-code)
      - [Install from FunASR](#install-from-funasr-1)
  - [Training your downstream model](#training-your-downstream-model)
  - [Contributors](#contributors)
  - [Citation](#citation)

# emotion2vec+: speech emotion recognition foundation model

## Guides
emotion2vec+ is a series of foundational models for speech emotion recognition (SER). We aim to train a "whisper" in the field of speech emotion recognition, overcoming the effects of language and recording environments through data-driven methods to achieve universal, robust emotion recognition capabilities. The performance of emotion2vec+ significantly exceeds other highly downloaded open-source models on Hugging Face.

![](./src/emotion2vec+radar.png)

## Data Engineering
We offer 3 versions of emotion2vec+, each derived from the data of its predecessor. If you need a model focusing on spech emotion representation, refer to [emotion2vec: universal speech emotion representation model](#emotion2vec-universal-speech-emotion-representation-model).

- emotion2vec+ seed: Fine-tuned with academic speech emotion data from [EmoBox](https://github.com/emo-box/EmoBox)
- emotion2vec+ base: Fine-tuned with filtered large-scale pseudo-labeled data to obtain the base size model (~90M)
- emotion2vec+ large: Fine-tuned with filtered large-scale pseudo-labeled data to obtain the large size model (~300M)

The iteration process is illustrated below, culminating in the training of the emotion2vec+ large model with 40k out of 160k hours of speech emotion data. Details of data engineering will be announced later. 

## Performance

Performance on [EmoBox](https://github.com/emo-box/EmoBox) for 4-class primary emotions (without fine-tuning). Details of model performance will be announced later. 

![](./src/emotion2vec+performance.png)

## Inference with checkpoints

### Install from FunASR
1. install funasr
```bash
pip install -U funasr
```

2. run the code.
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
The model will be downloaded automatically.

FunASR support file list input in wav.scp (kaldi style):
```
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```
Refer to [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) for more details.


# emotion2vec: universal speech emotion representation model

## Guides

emotion2vec is the first universal speech emotion representation model. Through self-supervised pre-training, emotion2vec has the ability to extract emotion representation across different tasks, languages, and scenarios.

## Performance
### Performance on IEMOCAP
emotion2vec achieves SOTA with only linear layers on the mainstream IEMOCAP dataset. Refer to the paper for more details.
![](./src/IEMOCAP.png)

### Performance on other languages
emotion2vec achieves SOTA compared with SOTA SSL models on multiple languages (Mandarin, French, German, Italian, etc.). Refer to the paper for more details.
![](./src/Languages.png)

### Performance on other speech emotion tasks
Refer to the paper for more details.

## Visualization
UMAP visualizations of learned features on the IEMOCAP dataset. <span style="color:red;">Red</span> and <span style="color:blue;">Blue</span> tones mean low and high arousal emotional classes, respectively.  Refer to the paper for more details. 
![](./src/UMAP.png)

## Extract features
### Download extracted features
We provide the extracted features of popular emotion dataset IEMOCAP. The features are extracted from the last layer of emotion2vec. The features are stored in `.npy` format and the sample rate of the extracted features is 50Hz. The utterance-level features are computed by averaging the frame-level features.
- frame-level: [Google Drive](https://drive.google.com/file/d/1JdQzwDJJEdKZcqSC1TXETvFZ7VpUvLEX/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1FtCwhUwhONaeEos4nLYFWw?pwd=zb3p) (password: zb3p)
- utterance-level: [Google Drive](https://drive.google.com/file/d/1jJVfoEKC8yjwj39F__8jIQayd5PBO0WD/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1AsJHacD6a5h27YJiCSee4w?pwd=qu3u) (password: qu3u)

All wav files are extracted from the original dataset for diverse downstream tasks. If want to train with standard 5531 utterances for 4 emotions classification, please refer to the `iemocap_downstream` folder.

### Extract features from your dataset
#### Install from the source code
The minimum environment requirements are `python>=3.8` and `torch>=1.13`. Our testing environments are `python=3.8` and `torch=2.01`.
1. git clone repos.
```bash
pip install fairseq
git clone https://github.com/ddlBoJack/emotion2vec.git
```

2. download emotion2vec checkpoint from:
- [Google Drive](https://drive.google.com/file/d/10L4CEoEyt6mQrqdblDgDSfZETYvA9c2T/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/15zqmNTYa0mkEwlIom7DO3g?pwd=b9fq) (password: b9fq)
- [modelscope](https://www.modelscope.cn/models/damo/emotion2vec_base/summary): `git clone https://www.modelscope.cn/damo/emotion2vec_base.git`

3. modify and run `scripts/extract_features.sh`

#### Install from FunASR
1. install funasr
```bash
pip install -U funasr
```

2. run the code.
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
The model will be downloaded automatically.

FunASR support file list input in wav.scp (kaldi style):
```
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```
Refer to [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) for more details.

## Training your downstream model
We provide training scripts for IEMOCAP dataset in the `iemocap_downstream` folder. You can modify the scripts to train your downstream model on other datasets.

## Contributors
|  Institution | Contribution |
|:------|:-----|
| [Shanghai Jiao Tong University](https://www.seiee.sjtu.edu.cn/) | Researchers; Computing power; Data collection; |
| [Fudan University](https://istbi.fudan.edu.cn/) | Researchers |
| [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/chinese/index.html) | Researchers |
| [Alibaba Group](https://www.alibaba.com/) | Researchers; Computing power; Data host; Model host; |
| [Peng Cheng Laboratory](https://data-starcloud.pcl.ac.cn/) | Researchers |

## Citation
If you find our emotion2vec code and paper useful, please kindly cite:
```
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={Proc. ACL 2024 Findings},
  year={2024}
}
```

# 日本語訳

## ニュース
- [2024年10月] FunASR インターフェースの利用方法を更新し、ソース選択に対応しました。中国本土のユーザーは `"ms"` または `"modelscope"`、それ以外の海外ユーザーは `"hf"` または `"huggingface"` を指定できます。スムーズに利用するには FunASR インターフェースを推奨します。
- [2024年6月] emotion2vec+ の不具合を修正しました。最新コードを再取得してください。
- [2024年5月] 9 クラス感情に対応した音声感情認識基盤モデル **emotion2vec+** を [Model Scope](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary) と [Hugging Face](https://huggingface.co/emotion2vec) で公開しました。高性能な SER 向け emotion2vec+ 系列、seed、base、large を確認してください。2024年1月リリースではなく、このリリースの利用を推奨します。
- [2024年1月] emotion2vec から反復的にファインチューニングした 9 クラス感情認識モデルを [modelscope](https://www.modelscope.cn/models/iic/emotion2vec_base_finetuned/summary) と [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) で公開しました。
- [2024年1月] **emotion2vec** が [modelscope](https://www.modelscope.cn/models/iic/emotion2vec_base/summary) と [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) に統合されました。
- [2023年12月] [論文](https://arxiv.org/abs/2312.15185)を公開し、emotion2vec 用の [WeChat グループ](./src/Wechat.jpg)を作成しました。
- [2023年11月] emotion2vec のコード、チェックポイント、抽出済み特徴量を公開しました。

## モデルカード
GitHub リポジトリ: [emotion2vec](https://github.com/ddlBoJack/emotion2vec)

| モデル | ⭐Model Scope | 🤗Hugging Face | ファインチューニングデータ（時間） |
|:---:|:-------------:|:-----------:|:-------------:|
|emotion2vec|[Link](https://www.modelscope.cn/models/iic/emotion2vec_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_base)|/|
|emotion2vec+ seed|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_seed/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)|201|
|emotion2vec+ base|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_base/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_base)|4788|
|emotion2vec+ large|[Link](https://modelscope.cn/models/iic/emotion2vec_plus_large/summary)|[Link](https://huggingface.co/emotion2vec/emotion2vec_plus_large)|42526|

## 概要

- [emotion2vec+: 音声感情認識の基盤モデル](#emotion2vec-音声感情認識の基盤モデル)
  - ガイド
  - データエンジニアリング
  - 性能
  - チェックポイントを使った推論
- [emotion2vec: 汎用音声感情表現モデル](#emotion2vec-汎用音声感情表現モデル)
  - ガイド
  - 性能
  - 可視化
  - 特徴抽出
  - 下流モデルの学習
  - コントリビューター
  - 引用

## emotion2vec+: 音声感情認識の基盤モデル

### ガイド
emotion2vec+ は音声感情認識、SER、のための基盤モデル系列です。音声感情認識分野における「whisper」のようなモデルを目指し、データ駆動の方法によって言語や録音環境の影響を抑え、汎用的で堅牢な感情認識能力を実現します。emotion2vec+ の性能は、Hugging Face で多くダウンロードされている他のオープンソースモデルを大きく上回ります。

### データエンジニアリング
emotion2vec+ には 3 つのバージョンがあり、それぞれ前段のデータをもとに構築されています。音声感情表現に焦点を当てたモデルが必要な場合は、emotion2vec の汎用音声感情表現モデルの説明を参照してください。

- emotion2vec+ seed: [EmoBox](https://github.com/emo-box/EmoBox) の学術的な音声感情データでファインチューニング
- emotion2vec+ base: フィルタ済みの大規模疑似ラベルデータでファインチューニングした base サイズモデル、約 90M
- emotion2vec+ large: フィルタ済みの大規模疑似ラベルデータでファインチューニングした large サイズモデル、約 300M

この反復プロセスにより、16 万時間の音声感情データのうち 4 万時間を使って emotion2vec+ large モデルを学習しています。データエンジニアリングの詳細は後日公開予定です。

### 性能
[EmoBox](https://github.com/emo-box/EmoBox) における 4 クラス主要感情の性能です、ファインチューニングなし。モデル性能の詳細は後日公開予定です。

### チェックポイントを使った推論
FunASR から利用する場合は、まず `pip install -U funasr` を実行し、上部の Python コード例を実行します。モデルは自動的にダウンロードされます。

FunASR は kaldi 形式の `wav.scp` によるファイルリスト入力にも対応しています。
```text
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```
詳細は [FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/main/examples/industrial_data_pretraining/emotion2vec) を参照してください。

## emotion2vec: 汎用音声感情表現モデル

### ガイド
emotion2vec は、最初の汎用音声感情表現モデルです。自己教師あり事前学習により、異なるタスク、言語、シナリオをまたいで感情表現を抽出できます。

### 性能

#### IEMOCAP での性能
emotion2vec は、主要な IEMOCAP データセットにおいて、線形層のみで SOTA を達成しています。詳細は論文を参照してください。

#### 他言語での性能
emotion2vec は、マンダリン、フランス語、ドイツ語、イタリア語など複数言語で、SOTA SSL モデルと比較して SOTA を達成しています。詳細は論文を参照してください。

#### その他の音声感情タスクでの性能
詳細は論文を参照してください。

### 可視化
IEMOCAP データセット上で学習済み特徴量を UMAP 可視化したものです。<span style="color:red;">赤</span>系と <span style="color:blue;">青</span>系は、それぞれ低 arousal と高 arousal の感情クラスを表します。詳細は論文を参照してください。

### 特徴抽出

#### 抽出済み特徴量のダウンロード
代表的な感情データセット IEMOCAP の抽出済み特徴量を提供しています。特徴量は emotion2vec の最終層から抽出されています。形式は `.npy`、抽出特徴量のサンプルレートは 50Hz です。発話単位の特徴量は、フレーム単位特徴量の平均で計算しています。

- フレーム単位: [Google Drive](https://drive.google.com/file/d/1JdQzwDJJEdKZcqSC1TXETvFZ7VpUvLEX/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1FtCwhUwhONaeEos4nLYFWw?pwd=zb3p) (password: zb3p)
- 発話単位: [Google Drive](https://drive.google.com/file/d/1jJVfoEKC8yjwj39F__8jIQayd5PBO0WD/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1AsJHacD6a5h27YJiCSee4w?pwd=qu3u) (password: qu3u)

すべての wav ファイルは、多様な下流タスクのために元データセットから抽出されています。標準の 5531 発話を使って 4 感情分類を学習したい場合は、`iemocap_downstream` フォルダを参照してください。

#### 自分のデータセットから特徴抽出する
ソースコードから使う場合の最小環境要件は `python>=3.8` と `torch>=1.13` です。テスト環境は `python=3.8` と `torch=2.01` です。

1. リポジトリを clone します。
```bash
pip install fairseq
git clone https://github.com/ddlBoJack/emotion2vec.git
```

2. emotion2vec チェックポイントを以下からダウンロードします。
- [Google Drive](https://drive.google.com/file/d/10L4CEoEyt6mQrqdblDgDSfZETYvA9c2T/view?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/15zqmNTYa0mkEwlIom7DO3g?pwd=b9fq) (password: b9fq)
- [modelscope](https://www.modelscope.cn/models/damo/emotion2vec_base/summary): `git clone https://www.modelscope.cn/damo/emotion2vec_base.git`

3. `scripts/extract_features.sh` を修正して実行します。

FunASR から使う場合は、`pip install -U funasr` を実行し、上部の Python コード例を実行します。モデルは自動的にダウンロードされます。

### 下流モデルの学習
IEMOCAP データセット用の学習スクリプトを `iemocap_downstream` フォルダで提供しています。必要に応じてスクリプトを修正し、他のデータセットで下流モデルを学習できます。

### コントリビューター
|  機関 | 貢献 |
|:------|:-----|
| [Shanghai Jiao Tong University](https://www.seiee.sjtu.edu.cn/) | 研究者、計算資源、データ収集 |
| [Fudan University](https://istbi.fudan.edu.cn/) | 研究者 |
| [The Chinese University of Hong Kong](https://www.cuhk.edu.hk/chinese/index.html) | 研究者 |
| [Alibaba Group](https://www.alibaba.com/) | 研究者、計算資源、データホスト、モデルホスト |
| [Peng Cheng Laboratory](https://data-starcloud.pcl.ac.cn/) | 研究者 |

### 引用
emotion2vec のコードや論文が役に立った場合は、上記の BibTeX を引用してください。
