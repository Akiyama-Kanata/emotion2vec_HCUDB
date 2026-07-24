# emotion2vec学習用デスクトップPC要件調査

確認日: 2026-07-23（日本時間）  
対象: このリポジトリ、`emotion2vec_base`、HCUDB1を中心とする話者独立cross-validation  
結論の前提: Ubuntuネイティブ、原則1 GPU、入力は16 kHzへ変換、最大20秒、mixed precision、batch 1から検証

## 1. 結論

HCUDB向けに`emotion2vec_base`の上位1～2 Transformer層だけを部分微調整する主用途には、**RTX PRO 4500 Blackwell 32GB、Ryzen 9 9950X、RAM 128GB、4TB NVMe、850～1000W PSU、Ubuntuネイティブ**を第一推奨とする。32GBは16GBの「動作する可能性がある境界」と24GBの「実用域」より余裕があり、長さ20秒・batch 1の購入前試験で15～20%の空きを確保しやすい。RTX PRO 4500は32GB ECC、200W、アクティブ冷却であるため、長時間の14-fold運用にも向く。

ただし、現在のリポジトリはencoder固定のhead-only学習までしか実装していない。上位1～2層微調整、全層微調整、SSL継続事前学習のVRAM値と時間は、いずれもこのリポジトリでの実測値ではない。PC購入前に部分微調整用の最小実装を作り、候補GPUと同容量のクラウドGPUまたは借用機で100 step以上測ることを購入判断の必須条件とする。

代替案は次の通り。

- **互換性・費用優先:** 中古RTX 3090 24GB。既存のAmpere向けPyTorch/fairseq環境を維持しやすく、GPU単価は最安。ただしECCなし、350W、個体履歴とGDDR6X温度・保証の確認が必要。
- **速度優先:** RTX 5090 32GB。理論演算性能と1,792GB/sの帯域が強いが、ECCなし、575W、国内価格高騰、Blackwell向け環境移行、ケース・電源・騒音の負担が大きい。
- **VRAM・長期運用優先:** RTX 6000 Ada 48GB。既存環境との互換性、ECC、48GBを同時に得る。現時点ではRTX PRO 5000 Blackwell 48GBと価格が逆転しているため、速度と将来性を優先するならPRO 5000、移行リスクを避けるなら6000 Adaを選ぶ。

**24GBで「必ず足りる」、32GBで「base全層が必ず動く」、48GBで「論文のSSL条件を単GPU再現できる」とは結論しない。** 音声長、attention実装、保存する中間活性、optimizer、AMP、gradient accumulationでpeak VRAMが変わるためである。

## 2. 数値の区分

| 区分 | 意味 |
|---|---|
| 公式実測 | 論文またはメーカーが明記した測定条件・仕様 |
| checkpoint由来 | ローカルの公式配布checkpointを直接調べた値 |
| 算術推定 | 明記値から単位変換・乗算した値 |
| 未計測の目安 | このリポジトリでは未実測。購入前ベンチマークで置き換える値 |

本書のVRAMレンジとHCUDB所要時間は、特記がない限り「未計測の目安」である。GPUの公称TFLOPSや帯域から学習時間を直接比例計算していない。

## 3. 現在のリポジトリでできること

### 3.1 実装済み

- 事前抽出した50Hz・768次元特徴を読み、分類・VA/VAD headを学習する。
- WAVからencoderを呼ぶラッパーは、既定で全encoderパラメータを`requires_grad=False`にし、`torch.no_grad()`内で特徴を抽出する（`vad_downstream/model.py:16, 34, 50`）。
- head-only学習CLIはencoder自体をロードせず、`.npy`特徴を使う。

したがって、現状の学習負荷は「93.79Mパラメータのencoder学習」ではなく、小さなheadと特徴batchの負荷である。GPUなしでも成立する。

### 3.2 未実装

- baseの上位1層または2層だけをunfreezeする学習経路
- base全層微調整
- emotion2vec+ largeの部分・全層微調整
- emotion2vecのSSL継続事前学習を再現する実行設定
- FSDP、ZeRO/DeepSpeed、FlashAttention、activation checkpointing
- このプロジェクト固有のDDP/`torchrun`起動設定

fairseq自体には分散学習機能があるが、このリポジトリのコード検索では上記の省メモリ実装や起動設定は見つからなかった。部分微調整の測定には、unfreeze対象、optimizer対象、下位層の`no_grad`境界、AMP、padding/croppingを明示する追加実装が必要である。

## 4. baseモデル、論文条件、公開checkpoint

### 4.1 ローカルcheckpointから確定した値

`artifacts/checkpoints/emotion2vec_base.pt`を、tensor本体を展開せずpickleメタデータとstorage shapeから調べた。

| 項目 | 値 | 区分 |
|---|---:|---|
| model parameter | 93,790,988（93.79M） | checkpoint由来 |
| checkpoint file | 1,125,606,009 bytes = 1.048GiB | checkpoint由来 |
| SHA-256 | `4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd` | ローカル実測 |
| FP32 model weights相当 | 357.8MiB | 算術推定 |
| FP16/BF16 model weights相当 | 178.9MiB | 算術推定 |
| backbone | prenet 4層 + shared blocks 8層 = Transformer計12層 | checkpoint由来 |
| hidden / heads | 768 / 12 | checkpoint由来 |
| sampling / maximum | 16,000Hz / 320,000 samples = 20秒 | checkpoint由来・算術推定 |
| `clone_batch` | 8 | checkpoint由来 |
| teacher target | 上位8層平均 | checkpoint由来 |
| EMA | 0.999 → 0.99999、`skip_ema=false` | checkpoint由来 |
| training checkpoint optimizer | Adamの`exp_avg`と`exp_avg_sq`、各93,790,988要素 | checkpoint由来 |
| checkpointの`max_tokens` | 600,000 | checkpoint由来 |
| checkpointの`update_freq` | 6 | checkpoint由来 |

1.048GiBの大半は、約357.8MiBのFP32 model weightsと約715.6MiBのAdam 2状態で説明できる。ファイルサイズを「推論時のVRAM」や「モデル単体サイズ」と同一視してはならない。

12層という数は、設定の`depth=8`だけを見ると誤解しやすい。公開実装ではaudio modalityの`prenet_depth=4`とshared `depth=8`が連なる。論文Appendixも12-layer Transformer、768次元、FFN 3072、12 headsと記載する。

### 4.2 論文の実測条件

論文のSSL pre-training条件は次の通り。

- 262時間・169,053発話の英語感情音声
- 4×NVIDIA A10 Tensor Core GPU（A10は各24GB）
- update frequency 4で16 GPU相当のeffective batchを模擬
- 100 epoch、1 epoch約37分
- dynamic batch、maximum tokens 1,000,000
- Adam、学習率`7.5e-5`、weight decay `1e-2`
- teacherは上位8 block平均、EMAは0.999から0.99999

100×37分 = 3,700分 = **約61.7時間**は算術値であり、論文が「61.7時間」と直接報告した値ではない。37分も丸められた値なので、61.7に再現性を示す精度はない。

また、これはcold startだけの代表条件ではない。論文はLibriSpeech 960時間でpre-train済みのdata2vecまたはdata2vec 2.0を初期値として比較し、data2vec 2.0初期化が最良だったと報告する。公開`emotion2vec_base`を用いる学習は、LS-960からの継続学習を経た重みを起点にする。cold start結果は別のablationであり、同じものとして扱わない。

論文のmaximum tokens 1,000,000、update frequency 4と、取得済みcheckpointの600,000、6、`clone_batch=8`は別の記録である。公開checkpointが論文実験の全条件をそのまま保存したと仮定しない。

参照: [ACL Anthology論文](https://aclanthology.org/2024.findings-acl.931/)、[公開base設定](https://huggingface.co/emotion2vec/emotion2vec_base/blob/5e05b4938a819f0e267f187ad94557b6f925e044/config.yaml)

### 4.3 emotion2vec+ large

公式model cardはplus largeを約300Mパラメータ、40k/160k時間のpseudo-labeled感情音声でfine-tuneしたモデルとして説明する。公開設定はhidden 1024、16 heads、audio prenet 4 + shared 8、`clone_batch=12`である。これは93.79M baseの単純な同容量モデルではなく、部分微調整でもparameter・activation・checkpoint負荷が上がる。

参照: [emotion2vec+ large model card](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/main/README.md)、[large config](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/main/config.yaml)

## 5. 負荷区分別のPC要件

以下のGPU VRAMと時間は、head-only以外は未計測の目安である。時間はHCUDB1 14話者leave-one-speaker-out、最大100 epochを想定するが、early stopping、実発話長、head、validation頻度で大きく変わる。

| 負荷 | GPU VRAM | CPU | RAM | SSD | PSU / 冷却 | OS | 概算時間 |
|---|---|---|---|---|---|---|---|
| 現行repo head-only | CPU可。GPUなら4～8GBで十分な可能性が高い | 8～16 core | 32GB最低、64GB推奨 | 1～2TB | 650～750W、通常の空冷 | Windows可、Ubuntu推奨 | 特徴抽出済みなら1 fold 数分～数十分、14 fold 数時間以内が目安 |
| base上位1層 | 16GB境界、24GB推奨、32GB余裕 | 12～16 core | 64GB最低、128GB推奨 | 2～4TB | 850～1000W、前面吸気の強いケース | Ubuntu native | 14 foldでおおむね2～7日を初期予算。実測で置換 |
| base上位2層 | 16GBは不確実、24GB実用、32GB推奨 | 16 core | 128GB推奨 | 4TB | 同上 | Ubuntu native | 14 foldで3～10日を初期予算。実測で置換 |
| base全層 | 24GBはbatch 1・AMP等で試行候補、32GB下限候補、48GB推奨 | 16 core | 128GB | 4TB以上 | 1000W級、長時間冷却 | Ubuntu native | 部分微調整の約2～4倍を計画。未計測 |
| plus large上位層 | 24GBは厳しい可能性、32GB下限候補、48GB推奨 | 16～32 core | 128GB | 4TB以上 | 1000W級 | Ubuntu native | base部分微調整の約2～4倍を計画。未計測 |
| base SSL継続事前学習 | 公式実績4×24GB。単一24GB非保証。単GPU48～96GBを検証候補 | 24～64 core | 128～256GB | 4TB最低、8TB推奨 | 1200～1600Wまたは200V、workstation冷却 | Ubuntu + NCCL | 公式4×A10で約61.7h相当。単GPU時間は外挿禁止 |

「16GB境界」は、20秒・batch 1・mixed precision・上位層のみgradient保持という条件で試す価値がある、という意味である。可否保証ではない。24GBを主用途の実用推奨、32GBを再試行やbatch調整の余裕込みの推奨とする。

### 時間を安全に見積もる式

購入前試験で、warm-up後のmedian `step_seconds`、1 epochのtraining step数、validation/save時間を測る。

```text
1 fold時間 ≈ step_seconds × train_steps_per_epoch × 実行epoch数
             + validation時間 + checkpoint時間

14 fold時間 ≈ 各foldの1 epoch実測時間 × 実行epoch数 の合計
```

100 stepだけの値より、実データで1 epochを通した値の方がDataLoader、padding、validationを含む。foldごとにtest話者の発話長分布が違うため、最長・中央値・合計の3点を報告する。

## 6. VRAMの読み方

93.79M baseのmodel weightはFP32でも約358MiBにすぎない。学習時のVRAMを支配し得るのは、時間軸1000 frame（20秒×50Hz）に対するattention、各層の保存activation、gradient、optimizer state、temporary workspace、CUDA contextである。

- head-onlyではencoder activationをbackward用に保存しないため軽い。
- 上位1～2層微調整では、下位層を完全に凍結し、上位層からだけgraphを作れば全層より軽い。
- optimizerへ凍結parameterを渡さない。上位1層は概ね7M、2層は概ね14M trainable parameterの規模である。
- SSLはstudentに加えてteacher/EMA、mask clone、decoder、複数lossを持ち、supervised fine-tuningとは別の負荷である。
- PyTorchの`memory_allocated`だけでなく`max_memory_reserved`と`nvidia-smi`のprocess使用量を記録する。

DDPでは各process/GPUがmodel、gradient、通常のoptimizer stateを基本的に保持する。2×24GBを使っても、1 sampleが48GBの連続VRAMを使えるようにはならない。PyTorch公式もDDPでは各processがmodel replicaを持ち、FSDPはparameter・gradient・optimizer stateをshardすると説明している。現状repoにFSDP/ZeROはないため、GPU枚数を増やす目的は主にthroughputであり、単GPUOOMの自動解消ではない。

参照: [PyTorch FSDP tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html)

## 7. ストレージ

50Hz・768次元・float32特徴は次の大きさになる。

```text
50 × 768 × 4 = 153,600 bytes/秒
1時間 = 約0.515GiB
262時間 = 約134.9GiB
```

HCUDB1は14話者、4,620発話、48kHz・16bit・monoである。実データのdurationは未取得なので、全発話が上限20秒という意図的に保守的な上限を示す。

| HCUDB1、4,620×20秒上限 | 容量 |
|---|---:|
| 元48kHz/16bit/mono WAV payload | 8.26GiB |
| 16kHz/16bit/mono変換 | 2.75GiB |
| 16kHz float32 waveform cache | 5.51GiB |
| 50Hz×768 float32特徴 | 13.22GiB |

実際は短い定型発話なので、この上限より小さいはずである。受領後に全WAVのduration合計と最大値を測って置き換える。

foldは同じ音声・特徴へのindexとして保持し、14倍コピーしない。保存容量を消費するのは、fold別のbest/last/periodic checkpoint、optimizer、logである。現在のfull training checkpointは約1.05GiBなので、14 fold×best/last 2世代なら約29.4GiB、5世代なら約73.4GiBである。large、EMA別保存、複数seed、複数unfreeze条件を含めると数百GiBになる。

推奨する4TB SSDの配分例:

- 0.5TB: OS、環境、package/cache
- 0.5TB: HCUDB原本・16k変換・manifest・特徴
- 1.0TB: checkpoints、14 fold、seed、実験比較
- 1.0TB: scratch、temporary、将来のlarge
- 1.0TB: 空き・wear leveling・予備

SSDはOS用とdata/checkpoint用を物理的に2本へ分けると、DataLoaderとcheckpoint書込の干渉、再インストール時の事故を減らせる。論文262時間規模を保存する場合は、元音声、約135GiB/世代のfloat32特徴、複数前処理、checkpoint、backupを考え、4TBを最低、8TBを推奨する。

HCUDB1の一次情報: [NII IDR HCUDB申請ページ](https://www.nii.ac.jp/dsc/idr/speech/submit/HCUDB.html)。研究目的限定、誓約書同意が必要で、48kHz・16bit・mono、14話者、4,620発話である。

## 8. GPU比較と国内価格

価格は税込の国内掲載最安または売買相場を2026-07-23に確認したスナップショットで、在庫・保証・送料は購入直前に再確認する。professional GPUは販売店への問い合わせ価格を含む。

| GPU | VRAM / ECC | 最大電力 | 国内価格の観測 | 判断 |
|---|---:|---:|---:|---|
| 中古RTX 3090 | 24GB / なし | 350W | 中古相場 約16.5～24.6万円 | 最安。Ampere互換性。個体リスクと電力が弱点 |
| RTX PRO 4000 Blackwell | 24GB ECC | 140W | 約35.0万円 | 新品保証、低電力、1-slot。Blackwell環境移行が必要 |
| RTX PRO 4500 Blackwell | 32GB ECC | 200W | 約62.2万円 | 本調査の第一推奨。容量・安定・電力の均衡 |
| RTX 5090 | 32GB / なし | 575W | 約69.0万円から | 最速候補だが高発熱・高騒音・ECCなし |
| RTX 6000 Ada | 48GB ECC | 300W | 約123.2万円 | 既存環境互換性と48GB。世代は旧い |
| RTX PRO 5000 Blackwell | 48GB ECC | 300W | 約111.8万円 | 48GBなら6000 Adaより安い観測。migrationを許容すれば有力 |
| RTX PRO 6000 Blackwell Max-Q | 96GB ECC | 300W | 約205.5万円 | 100Vデスクトップで扱いやすい96GB。高価 |
| RTX PRO 6000 Blackwell WS | 96GB ECC | 600W | 約222.6万円 | 最大速度側。電源・排熱・回路要件が重い |

仕様の一次資料:

- [RTX 3090: 24GB、350W](https://www.nvidia.com/ja-jp/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
- [RTX PRO 4000: 24GB ECC、140W、active single-slot](https://www.nvidia.com/ja-jp/products/workstations/professional-desktop-gpus/rtx-pro-4000/)
- [RTX PRO 4500: 32GB ECC、896GB/s、200W、active dual-slot](https://www.nvidia.com/ja-jp/products/workstations/professional-desktop-gpus/rtx-pro-4500/)
- [RTX 5090: 32GB、1,792GB/s、575W、minimum 1000W PSU](https://www.nvidia.com/ja-jp/geforce/graphics-cards/50-series/rtx-5090/)
- [RTX 6000 Ada: 48GB ECC、300W、active dual-slot](https://www.nvidia.com/ja-jp/products/workstations/rtx-6000/)
- [RTX PRO 5000: 48/72GB ECC、1,344GB/s、300W](https://www.nvidia.com/ja-jp/products/workstations/professional-desktop-gpus/rtx-pro-5000/)
- [RTX PRO 6000: 96GB ECC、WS 600W](https://www.nvidia.com/ja-jp/products/workstations/professional-desktop-gpus/rtx-pro-6000/)

価格参照:

- [RTX 3090中古相場、2026-07-22更新](https://price-rank.com/p/9362/market_price)
- [RTX PRO 4000国内価格](https://kakaku.com/item/K0001682019/)
- [RTX PRO 4500国内価格](https://kakaku.com/item/K0001682018/)
- [RTX 5090国内一覧](https://kakaku.com/pc/videocard/itemlist.aspx?pdf_Spec103=500&pdf_Spec109=1)
- [RTX 6000 Ada価格履歴](https://kakaku.com/item/K0001562419/pricehistory/)
- [RTX PRO 5000 48GB国内価格](https://kakaku.com/item/K0001682017/)
- [RTX PRO 6000 Max-Q 96GB国内価格](https://kakaku.com/item/K0001682016/)
- [RTX PRO Blackwell国内価格記事](https://news.kakaku.com/prdnews/cd%3Dpc/ctcd%3D0550/id%3D147740/)

### 8.1 3090とPRO 4000

3090は費用対VRAMが非常に高い。既存Ampere binaryで動かしやすく、部分微調整の購入前検証機として合理的である。購入時は、負荷試験、VRAM error、hotspot/memory junction、ファン、補助電源端子、分解歴、保証、元マイニング用途を確認する。

PRO 4000はECC、新品保証、140W、1-slot、active冷却が長時間運用に有利で、カード寸法と電源の問題が小さい。一方、同じ24GBに中古3090の約2倍のGPU代を払う。さらにBlackwell対応PyTorchへの移行試験が必要なので、「差せば今のfairseq環境がそのまま動く」とは扱わない。

### 8.2 PRO 4500と5090

PRO 4500は32GB ECC、200W、active dual-slot。5090は32GB非ECC、575Wで、メモリ帯域は1,792GB/s。速度を最優先し、電源・冷却・騒音を許容するなら5090、再現実験の長時間安定性、ECC、研究室の100V回路、総運用コストを重視するならPRO 4500が適する。

2026-07-23の観測価格差は約6.8万円に縮まっているため、PRO 4500は「安いから」ではなく、ECCと200Wを買う選択である。5090の実学習速度倍率はこのモデルで未計測であり、公称AI TOPSから断定しない。

### 8.3 48～96GB

base全層、large、SSL実験を同じPCで行うなら48GB以上の価値が出る。現時点の観測ではPRO 5000 48GBがRTX 6000 Adaより約11万円安い。ただし6000 Adaは既存環境との互換性が高く、PRO 5000はBlackwell移行が必要である。

96GBは単GPUで大きな実験を試せるが、base上位1～2層だけが目的なら過剰投資である。RTX PRO 6000 Max-Q 300Wは一般的な100V workstationへ収めやすい。600W版はGPUだけで600Wであり、CPU・PSU損失・storageを含むwall powerと排熱を別設計にする。

## 9. 3段階の構成

総額はGPUの確認価格と、同時点の国内BTO・パーツ市場を基にした税込の算術目安である。購入時は同一販売店の正式見積を取る。

### 9.1 最小構成

| 部品 | 推奨 |
|---|---|
| GPU | 中古RTX 3090 24GB、または新品RTX PRO 4000 Blackwell 24GB |
| CPU | Ryzen 9 7900/9900X級、12 core以上 |
| RAM | DDR5 64GB（2 DIMM、128GBへ増設可能にする） |
| SSD | NVMe 2TB、空きM.2 slotを残す |
| PSU | 3090は1000W、PRO 4000は750～850W、80 PLUS Gold以上 |
| 冷却 | 大型towerまたは280/360mm AIO、front mesh full/mid tower |
| OS | 3090はUbuntu 22.04系の既存環境、PRO 4000はmodern Ubuntu環境を別作成 |

概算総額:

- 中古3090: **43～58万円**
- PRO 4000新品: **60～75万円**

国内BTOの参考として、Core Ultra 9、64GB、2TB、PRO 4000搭載機が71.48万円で販売された。[PC Watchの国内BTO記事](https://pc.watch.impress.co.jp/docs/news/2066990.html)

最小構成は24GBなので、base上位2層・20秒の合格試験を通した個体だけ採用する。SSD 2TBはHCUDBには足りるが、largeや複数seedを始めたら4TBを追加する。

### 9.2 推奨構成

| 部品 | 第一推奨 |
|---|---|
| GPU | RTX PRO 4500 Blackwell 32GB ECC |
| CPU | Ryzen 9 9950X（16 core / 32 thread、170W） |
| motherboard | X870E級、GPU x16、M.2を3本以上、2.5/10GbE |
| RAM | DDR5 128GB（2×64GBを優先。4 DIMM時の速度低下を確認） |
| SSD | 4TB NVMe TLC、可能ならOS 1～2TB + data 4TB |
| PSU | 850～1000W、80 PLUS Gold/Platinum、余裕ある12V系 |
| CPU冷却 | 高性能dual-towerまたは360mm AIO |
| ケース | GPU長270mm以上、2-slot、front mesh、140mm fan複数 |
| OS | Ubuntu 24.04 host + 検証済みcontainer/env |

概算総額: **100～125万円**。Ryzen 9 9950Xは16/32、最大5.7GHz、170Wで、2026年7月の国内最安は約9.1万円だった。[AMD仕様](https://www.amd.com/en/products/processors/desktops/ryzen.html)、[国内価格履歴](https://kakaku.com/item/K0001630330/pricehistory/)

速度優先の差し替えはRTX 5090 32GB、1200W ATX 3.1 PSU、より大きいケースとする。概算総額は**108～135万円**。GPUを長時間100%で回すため、安価な1000Wぎりぎりではなく1200Wを採る。NVIDIAのminimum system powerは1000Wである。

### 9.3 大容量構成

単GPU:

| 目的 | GPU | RAM / SSD | PSU | 概算総額 |
|---|---|---|---|---:|
| 互換性優先48GB | RTX 6000 Ada 48GB | 128GB / 4TB | 1000W | 160～190万円 |
| 速度・将来性48GB | RTX PRO 5000 Blackwell 48GB | 128～256GB / 4～8TB | 1000W | 150～185万円 |
| 単GPU最大容量 | RTX PRO 6000 Blackwell Max-Q 96GB | 256GB / 8TB | 1200W | 250～310万円 |
| 単GPU最大速度側 | RTX PRO 6000 Blackwell WS 96GB | 256GB / 8TB | 1600W級、200V検討 | 280～350万円 |

複数GPUではRyzen/AM5を使わず、Threadripper PRO + WRX90を基本にする。AMDはWRX90で8-channel memoryと最大128 PCIe 5.0 lanesを提供しており、複数のx16 GPUとNVMeに適する。[AMD Threadripper PRO説明](https://www.amd.com/en/blogs/2025/amd-introduces-new-zen-5-based-ryzen-threadripper-pro.html)

2 GPU以上の概算は**350万円以上**。カード間隔、各slotの実lane、12V-2x6 cableの曲げ、電源回路、重量支持、NCCL P2P topologyをBTO事業者に図面で確認する。DDPのVRAMは合算されないため、「2×24GBを48GBの代わりにする」目的では買わない。

## 10. ソフトウェア互換性

### 10.1 Ampere/Adaで既存環境を維持

このrepoは`fairseq==0.12.2`、Hydra 1.0.7、OmegaConf 2.0.6を固定し、Python 3.8～3.10を案内している。RTX 3090、A10、RTX 6000 Adaなら、現在動作確認したPython 3.10系・PyTorch/CUDA組合せをlockfile/container imageとして凍結しやすい。

実行環境を再構築する前に、Python、PyTorch、CUDA runtime、driver、fairseq commit/package、pip freeze、checkpoint hash、1 WAV smoke testを保存する。`torch>=1.13`という下限だけでは再現環境にならない。

### 10.2 Blackwellへ移行

PyTorch 2.7はBlackwell supportとCUDA 12.8 wheelを導入した。したがって2.7 + CUDA 12.8はBlackwellの実用的な最低線である。ただし2026年現在のPyTorch 2.12ではCUDA 12.8 wheelが廃止予定で、BlackwellはCUDA 13.0+がstable側になっている。新規構築では「2.7/cu128で固定する互換環境」と「current PyTorch/CUDA 13環境」の双方を候補にし、このrepoのtestと実checkpoint loadを比較する。

参照: [PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/)、[PyTorch 2.12 release](https://pytorch.org/blog/pytorch-2-12-release-blog/)

fairseqの公式repoは2026-03-20にarchiveされread-onlyになった。0.12.2は最新PyTorch、Python、NumPy、Hydraを前提に保守されていない。Blackwell化はGPU交換ではなく、少なくとも次を含むmigration projectである。

1. Python/fairseq buildの修正
2. checkpoint load互換性の検証
3. custom `upstream` moduleのunit/smoke test
4. feature一致の許容誤差試験
5. AMP/BF16 backward試験
6. 100 step save/resume試験

参照: [archived fairseq repository](https://github.com/facebookresearch/fairseq)

### 10.3 Ubuntuを推奨する理由

PyTorch distributedはLinuxがstable、Windowsはprototypeで、WindowsはNCCLをサポートしない。multi-GPU GPU trainingはNCCLが最良性能と公式文書にある。単GPUhead-onlyならWindowsでもよいが、部分微調整から先はUbuntu nativeを標準にする。

参照: [PyTorch distributed backends](https://docs.pytorch.org/docs/stable/distributed.html)

## 11. 電源、冷却、ケース、騒音

日本の一般的な100V・15Aコンセントは理論上1,500Wまでである。[Panasonic FAQ](https://jpn.faq.panasonic.com/app/answers/detail/a_id/80575/p/4268)

- 5090機はwall power、瞬間負荷、PSU効率を考え、同じbranchで暖房・電子レンジ等を使わない。1200W ATX 3.1 PSU、専用コンセント、正しい12V-2x6挿入を推奨する。
- PRO 4500 200Wは、9950Xと組み合わせても100V研究室で扱いやすい。
- PRO 6000 WS 600Wとhigh-core Threadripperは、system全体が100V/15Aの余裕を圧迫する。200V専用回路またはMax-Q 300Wを検討する。
- 連続学習ではGPU hotspot、VRAM温度、CPU package、SSD温度、fan RPM、wall power、室温をlogする。
- consumer 3～4 slot GPUは隣接slotと吸気を塞ぐ。カタログのGPU長・高さだけでなく、電源connectorを曲げない側面余白を確認する。
- 5090の575Wは部屋への約575Wの熱源でもある。CPU等を加えると小型暖房器具相当になり、夏季の空調費と騒音が研究室運用条件になる。

論文で使われたA10は24GB・150Wだがpassive coolingで、NVIDIAはsystem airflowを必要とすると明記する。サーバー風洞を持たない一般デスクトップへ単体で載せる案は推奨しない。[A10 product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/A10-Product-Brief.pdf)

## 12. 購入前ベンチマーク

### 12.1 事前条件

1. HCUDBの誓約書・利用規約を確認する。許可が明確になるまで、音声をクラウドや外部事業者へuploadしない。
2. HCUDB1全WAVについてsample rate、channel、bit depth、durationをmanifestへ出す。
3. 最長WAVと、durationのp50/p90/p99を特定する。
4. 48kHz原本を保持したまま、別directoryへ16kHz mono PCMを作る。変換後の件数、duration差、clipping、hashを検証する。
5. 20秒を超える実音声があれば、crop/split/pad方針を実験protocolとして固定する。

### 12.2 実装すべき最小benchmark

- 公式base checkpointをloadする。
- 全parameterをfreezeし、shared `blocks[-1:]`、次に`blocks[-2:]`だけをunfreezeする。
- optimizerへtrainable parameterだけを渡す。
- 16kHz waveformからend-to-endでforward/backwardする。事前抽出特徴を入力してはいけない。
- batch 1、実データ最長、FP16とBF16を別runにする。
- 20 warm-up step後、100 step以上を測る。
- deterministicな小さなhead/lossを使い、loss finite、gradient finite、更新対象と固定対象をassertする。

### 12.3 記録項目

| 項目 | 測定方法 |
|---|---|
| peak VRAM allocated/reserved | `torch.cuda.reset_peak_memory_stats()`後のmax値 |
| GPU全process使用量 | `nvidia-smi` |
| step時間 | CUDA synchronize後、p50/p90/p99 |
| DataLoader時間 | load/resample/collateをforwardと分離 |
| host RAM peak | process RSSとsystem available |
| power / temperature | `nvidia-smi dmon`等 |
| checkpoint | save時間、file size、load時間 |
| resume | 同一seedで次step loss/parameter更新が許容誤差内 |

### 12.4 合格条件

- 100 measured stepをOOM、NaN/Inf、driver resetなしで完走
- 推奨GPUでは`max_memory_reserved`とprocess VRAMの大きい方が物理VRAMの80～85%以下、すなわち15～20%空き
- GPU温度・VRAM温度・clockが長時間で悪化し続けない
- host available RAMが最低16GB残る
- checkpoint save/resume成功
- 固定層のparameterが不変、unfreeze層にfinite gradientがある
- 最長音声で合格後、batchと音声長を段階的に増やしOOM境界を記録

16GB候補で動いても、使用率が95%なら購入合格にしない。allocator fragmentation、validation、別process、長いsample、driver更新でOOMになる余地がないためである。

### 12.5 14-fold外挿

100 step試験後に、少なくとも1 foldでtrain/validation/checkpointを含む1 epochを通す。各foldの発話数と総durationからstep数を計算し、14 foldを外挿する。100 epoch固定だけでなく、予定するearly stoppingのmedian/p90 epochも報告する。

クラウドGPUで試す場合は、合成20秒音声でまずVRAM境界を確認できる。実HCUDBのuploadが許されない場合、データ形状試験は合成音声、最終throughput試験はオンプレミスで分ける。

## 13. 最終購入判断

1. 部分微調整実装を先に用意する。
2. 24GBと32GB相当で、上位1層・2層、最長音声、batch 1、BF16/FP16を測る。
3. 24GBで20%近い余裕が出るなら、互換性優先は中古3090、新品・低電力優先はPRO 4000。
4. 24GBの余裕が不足、または今後base全層を試すならPRO 4500 32GBを採用する。
5. benchmarkで32GB使用量が27GBを超える、largeを主目的にする、SSLを実験範囲へ入れる場合は48GB以上へ上げる。

現時点の主用途だけで判断すれば、**RTX PRO 4500 32GB + Ryzen 9 9950X + RAM 128GB + 4TB NVMe**が最も説明可能な1台である。5090は同容量で速さを買う選択、3090は保証と電力を犠牲に費用を下げる選択、48～96GB professional GPUは研究範囲をlarge/SSLまで広げる選択である。
