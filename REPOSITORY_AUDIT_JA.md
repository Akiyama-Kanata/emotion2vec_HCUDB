# emotion2vec リポジトリ監査レポート

調査日: 2026-07-16
対象: `REPOSITORY_GUIDE_JA.md`、環境設定、IEMOCAP 下流処理、VAD 下流処理、テスト  
目的: 初めて利用する人が、安全かつ再現可能な方法で実験できるかを実装品質の観点から確認する

## 1. 結論

このリポジトリは、事前抽出した emotion2vec 特徴から VAD 回帰や感情分類を試すための**研究プロトタイプ**としては構造が比較的分かりやすく、VAD 系には入力検証と単体テストも用意されています。一方、現状のまま第三者が環境を再構築し、得られた評価値を研究結果として比較するには不足があります。

残る最優先の問題は次の4点です。

1. `requirements.txt` が環境を固定しておらず、古い fairseq と任意の新しい PyTorch を組み合わせ得るため、再現性がない。
2. IEMOCAP の検証データが発話単位のランダム分割で、同じ話者が学習と検証の両方に入る可能性がある。また、セッション件数と並び順をコードが暗黙に仮定している。
3. 実emotion2vec encoderのCPU動作は確認できたが、学習済みVAD経由感情分類headがなく、実データでの分類性能は未検証である。
4. VAD 経由分類の `weight × VAD` は線形分類器の logit を分解した値であり、因果的な説明や、VAD が音声判断の真の理由であることを証明するものではない。

`REPOSITORY_GUIDE_JA.md` はこれらの一部を「制限」として正しく説明しています。ただし、注意書きがあることと、実験の再現性・妥当性が確保されていることは別です。

### 1.1 このPCでの実行方針と現在地（2026-07-16更新）

このPCは Intel Iris Xe のみで NVIDIA CUDA を利用できないため、**WSL2 Ubuntu上でCPUを使い、emotion2vec encoderを凍結して感情/VAD headだけを学習する**構成を標準とする。Pythonのプロジェクト専用環境を用意すること自体は一般的であり、他プロジェクトとの依存衝突を避けるためにも維持する。ただし、利用者が長い環境依存コマンドを毎回入力しないよう、環境構築・単体テスト・E2Eテストはスクリプトへ集約する。

既存のWSL専用環境 `/home/akiyama/miniforge/envs/emotion2vec-py310/bin/python` は利用可能で、主要importと単体テスト48件の成功を確認した。現在確認できた主要版は PyTorch `2.12.0+cu130`、NumPy `1.26.4` である。さらに、この環境をCPU指定で使用し、公式 `emotion2vec_base.pt` と `scripts/test.wav` から分類JSONまでのスモークテストが成功した。これは「現在の環境で実encoderをロードして推論経路を通せる」証拠ではあるが、`requirements.txt` から同じ環境を再構築できる証拠ではない。また、インストール済みPyTorchはCUDA buildであり、Iris XeではCUDAを利用していないため、最終的な標準環境はCPU版PyTorchを含む再構築可能な定義へ固定する必要がある。

公式の学習済みencoder checkpointは `artifacts/checkpoints/emotion2vec_base.pt` に取得済みで、ファイルサイズ `1,125,606,009` bytes、SHA-256 `4f14ddf7ba394bcafdd4bff6ae0f24ab2e4134260d4dd42c58ea791a201b02dd` を確認した。`REAL_EMOTION2VEC_SMOKE_TEST_JA.md` の手順により、実encoderと未学習のランダムVAD・分類headを使うCPU配線テストは完了している。意味のある感情分類結果を得る完全なE2Eテストには、引き続き学習済みVAD経由感情分類head checkpointが必要である。

### 1.2 このPCで今後行うこと

次の順序で進める。既存環境を先に壊さず、実モデルがCPUで動くことを確認してから、その動作環境を再構築可能な形へ固定する。

1. **既存WSL環境を維持する**
   - `/home/akiyama/miniforge/envs/emotion2vec-py310` を当面の基準環境として残す。
   - 主要importと単体テスト48件は成功済みであり、新しい環境へ直ちに置き換えない。
2. **現在の環境を記録する**
   - Python、pip、PyTorch、fairseq、NumPyを含む直接・推移依存の一覧を保存する。
   - OS、CPU実行、git revision、取得日時も一緒に記録する。
3. **実モデルを用意する**
   - 公式のemotion2vec encoder checkpointは取得・hash確認済み。
   - 学習後のVAD経由感情分類head checkpointを用意する。
   - encoder checkpointの出典・サイズ・SHA-256はスモークテスト手順に記録済み。今後のhead checkpointにも同等の台帳情報を残す。
4. **特徴抽出のCPU対応を完成させる**
   - 特徴抽出コードの無条件な `.cuda()` を `.to(device)` へ変更する。
   - `--device auto/cpu/cuda` を追加し、このPCでは `--device cpu` を使用する。
5. **実音声E2Eテストを完成させる**
   - 実emotion2vec encoderとランダムheadを使うCPU配線テストは完了済み。
   - `scripts/test.wav` を実emotion2vec encoderと学習済み分類headへ入力するテストを追加する。
   - VAD、`hap/sad/ang/dis` の分類、クラス確率、有限値、確率和、ラベル集合、checkpoint hashを自動検証する。
6. **動作確認済み環境を固定する**
   - Python 3.10、CPU版PyTorch、fairseq 0.12.2、NumPy、全推移依存を固定する。
   - `environment.yml` とlock/constraintsを用意し、新規環境から同じテスト結果を再現する。
7. **操作を1コマンド化する**
   - 環境構築、単体テスト、CPU E2Eテストの入口となるスクリプトを用意する。
   - WSL distribution名、個人名、Python絶対パスを設定へ集約し、通常利用者から隠す。

現在の確認状況は次のとおりである。

- [x] 既存WSL専用環境を確認した。
- [x] PyTorch `2.12.0+cu130`、NumPy `1.26.4`、fairseqのimportに成功した。
- [x] 単体テスト48件が成功した。
- [x] このPCではIntel Iris Xeを計算用GPUとして扱わず、CPU運用とする方針を決めた。
- [x] 公式emotion2vec checkpointを取得し、サイズとSHA-256を確認した。
- [x] 実emotion2vec checkpointをCPUで読み込み、実音声から分類JSONまでの配線テストを成功させた。
- [ ] 学習済みVAD経由感情分類headを用意する。
- [ ] 学習済みheadを使う実音声CPU E2Eテストを成功させる。
- [ ] 動作確認済み依存を固定し、新規環境から再構築する。

## 2. モデル構造を初心者向けに整理

### 2.1 実際の処理

```text
16 kHz mono WAV
  │
  ├─ emotion2vec encoder
  │    各時刻を768次元ベクトルへ変換: [B, T, 768]
  │
  ├─ paddingを除外した時間平均
  │    発話全体を1個の768次元ベクトルへ圧縮: [B, 768]
  │
  ├─ Linear(768 → 256) + ReLU + Linear(256 → 2または3)
  │    VA/VADを予測: [B, D]
  │
  └─ Linear(D → 4)
       VAD経由分類の場合だけ、4感情のlogitを出力
```

- emotion2vec は音声を感情関連の特徴列に変換する大きな事前学習済みモデルである。
- `VADRegressionHead` は特徴列を平均し、小さな全結合ネットワークで VA/VAD を予測する（`vad_downstream/model.py:77-114`）。
- `VADClassificationHead` は予測 VAD だけを入力する線形分類器である（同 `:117-139`）。768次元特徴から感情クラスへ直接つながる経路はない。
- WAV 推論時の encoder は既定で凍結され、`torch.no_grad()` 内で実行される（同 `:32-52`）。事前抽出特徴を使う学習 CLI は encoder 自体をロードせず、head だけを学習する。

### 2.2 この構造の長所

- 分類器への入力を2～3次元に限定しているため、予測の計算経路を追跡しやすい。
- padding を除いた平均プーリングが実装され、全フレームが padding の入力も拒否する。
- 回帰と分類を同時学習でき、checkpoint に次元数、クラス順、ハイパーパラメータ、履歴を保存する。

### 2.3 この構造だけでは保証されないこと

- 平均プーリングでは、感情が発話のどの時点で変化したか、語尾だけ強くなったか、といった時間順序が失われる。同じフレーム集合を異なる順序に並べても head の入力は同じになる。
- `logit_c = bias_c + Σ(weight_c,d × VAD_d)` なので、出力された寄与度は**その線形計算の正確な内訳**ではある。しかし、VAD予測自体が音声特徴から学習された潜在値であるため、「人間の意味での valence が原因で怒りと判断した」とまでは言えない。
- 分類損失の勾配も VAD head に流れる。`lambda_emo` が強い場合、3つの出力が正解VADに忠実な量というより、4クラス分類に便利な3次元符号へ変化し得る。VAD CCC、分類性能、損失重み別の比較が必要である。
- end-to-end fine-tuning は未実装で、encoder が対象データへ適応する効果は評価できない。

## 3. 問題一覧

重要度は、**重大**（漏えい・評価の信用を直接損なう）、**高**（再現や主要結果を大きく左右する）、**中**（運用・解釈上の明確な弱点）、**低**（保守性や分かりやすさの問題）とした。

### 3.1 環境構築と安全性

| 重要度 | 問題 | 根拠 | 改善案 |
|---|---|---|---|
| 解消 | `.env` が Git 追跡対象 | 2026-07-14に追跡対象から外し、`.gitignore` と値を空にした `.env.example` を導入した。履歴上も秘密情報らしい値は検出されなかった。 | 現在の対策を維持し、実値を含む `.env` を再追加しない。 |
| 高 | 依存環境が固定されない | `requirements.txt:4-12` は `torch>=1.13`、`numpy<2`、バージョンなしの `soundfile` 等を許す。推移依存も固定されない。pip 公式も再現可能なインストールには直接・推移依存の固定を推奨している。 | 検証済みの Python、PyTorch、CUDA/CPU、全依存を lock/constraints ファイルへ固定し、CPU版とCUDA版を分ける。OS・GPUドライバ条件も記録する。 |
| 高 | 古い fairseq に対して PyTorch の上限がない | fairseq 0.12.2 は2022年公開で、PyPIの配布 wheel は CPython 3.6～3.8向けのみ。リポジトリは2026-03-20に archive 済みである。一方 `torch>=1.13` は将来版まで許す。 | 実際に通った組み合わせを厳密に固定する。例としてガイドが想定する Python 3.10 / pip 24.0 / NumPy 1.26.4 に加え、PyTorch の完全な版とCUDA buildも固定してCIで再構築する。 |
| 高 | PyTorch/CUDA の導入方法が曖昧 | PyTorch 1.13でも CPU、CUDA 11.6、CUDA 11.7は異なる配布物である。単なる `pip install -r requirements.txt` では使用するCUDA buildが明示されない。 | NVIDIA driver、CUDA runtime、PyTorch wheel indexを含むコマンドを記載する。CPU確認用とGPU本番用の2系統を用意する。 |
| 中 | `pip<24.1` が手順依存 | requirements自身ではpipを制御できず、先に手作業でdowngレードする必要がある。新規利用者が見落とすとfairseqのbuild/metadataで失敗し得る。 | bootstrap scriptまたはconda環境定義にpip版を含め、クリーン環境からのCIを追加する。 |
| 中 | `numpy<2` は安全側だが完全固定ではない | NumPy 2.0 はABIを破壊するため上限には根拠がある。しかし1.x内の任意版を許すため、同一環境は再現しない。 | 検証済みの `numpy==1.26.4` をlock側で固定する。NumPy公式も2.0で1.x向けバイナリとの非互換が起きると説明している。 |
| 中 | WSLと個人環境への依存 | `TESTING.md:7,20` が distribution名 `Ubuntu` と `/home/akiyama/...` を固定する。別ユーザー・別distributionではそのまま動かない。Docker/native Windowsも対象外。 | `environment.yml` またはコンテナを用意し、ユーザー名に依存しない `python -m ...` を標準にする。WSL distributionと環境パスは変数化する。 |
| 中 | `.gitignore` の生成物除外が部分的 | `.env`、`/artifacts/checkpoints/`、`/outputs/` は除外済み。一方、Hydra出力、学習ログ、`.pytest_cache`、大規模特徴量などの包括的な方針はまだない。 | 残る生成物を用途別に除外し、必要な小規模fixtureだけを明示的に管理する。 |
| 中 | 外部資産の取得と同一性確認が部分的 | 公式emotion2vec checkpointは配布元、期待ファイル名、サイズ、SHA-256、確認コマンドを `REAL_EMOTION2VEC_SMOKE_TEST_JA.md` に記録済み。一方、IEMOCAP、VADラベル、将来の分類headについては利用条件、hash、版、前処理が1か所に揃っていない。 | 資産台帳を作り、未整理の外部データと学習済みheadにも出典・ライセンス・SHA-256・前処理コマンド・期待件数を記録する。IEMOCAPは配布条件上、自動ダウンロードではなく配置検証を用意する。 |

#### 対応状況（2026-07-16更新）

重大項目「`.env` が Git 追跡対象」には対応済みである。

- `.env` を Git の追跡対象から外し、ルートの `.gitignore` に追加した。ローカルの `.env` と既存設定値は保持している。
- 実値を含まない `.env.example` を追加し、`RESTRICTED_DATA_DIR` と `VAD_CSV_PATH` を空値で定義した。
- `REPOSITORY_GUIDE_JA.md` に、雛形のコピー手順、実値をコミットしない注意、現状コードは `.env` を自動読込しないことを記載した。
- `git check-ignore .env`、追跡ファイル一覧、雛形の空値を確認した。履歴上も秘密情報らしい値は検出されなかったため、履歴書き換えと資格情報ローテーションは行っていない。
- 制限付き実行ではユーザー登録のWSLが見えなかったが、許可付きで既存の `Ubuntu` と専用Pythonを使用できた。主要importと単体テスト48件は成功した。今回の変更は設定管理と文書だけで、Python APIやCLIの挙動は変更していない。

2026-07-16時点では、`.gitignore` に `/artifacts/checkpoints/` と `/outputs/` も追加され、取得済みcheckpointとスモークテストJSONが誤ってGitへ入る経路は抑えられた。ただし、特徴量、Hydra出力、学習ログ、cacheの包括的な除外は未対応であり、項目全体としては継続課題である。

外部根拠:

- [fairseq 0.12.2（PyPI）](https://pypi.org/project/fairseq/0.12.2/)
- [fairseq GitHub（archive済み）](https://github.com/facebookresearch/fairseq)
- [PyTorch 1.13 のCPU/CUDA別インストール](https://docs.pytorch.org/get-started/previous-versions/)
- [pip: Repeatable Installs](https://pip.pypa.io/en/stable/topics/repeatable-installs/)
- [NumPy 2.0 release notes](https://numpy.org/doc/stable/release/2.0.0-notes.html)

### 3.2 特徴抽出と推論

| 重要度 | 問題 | 根拠 | 改善案 |
|---|---|---|---|
| 高 | 単一WAV特徴抽出が失敗を成功のように終了し得る | `scripts/extract_features.py:55-66` の裸の `except:` は、例外オブジェクトを作るだけでraiseもログ出力もしない。出力がなくても終了コード0になり得る。 | 捕捉対象を限定し、元例外を付けて再送出する。出力は一時ファイルへ保存後に置換し、成功時にshapeと保存先を表示する。 |
| 高 | 公式由来の特徴抽出はGPUを強制 | `scripts/extract_features.py:42,51` と `iemocap_downstream/scripts/emotion2vec_speech_features.py:51,68` が無条件に `.cuda()` を呼ぶ。後者は大量処理中の再開機構もない。 | `--device auto/cpu/cuda` を共通化し、modelとtensorを `.to(device)` へ移す。CPU smoke testを追加する。 |
| 中 | 入力形式検証に `assert` を使用 | `scripts/extract_features.py:47-48` 等。Pythonを最適化モードで実行するとassertは無効化できる。またWAV以外では `wav` が未定義のまま進む。 | 拡張子・存在・sample rate・channel・dtype・空音声を明示的な例外で検証し、必要ならresample/downmix方針を選択可能にする。 |
| 中 | checkpoint読込の信頼境界が部分的にしか説明されない | `REAL_EMOTION2VEC_SMOKE_TEST_JA.md` は公式配布元、SHA-256、`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` によるpickle読込リスクを明記している。一方、他のcheckpoint経路すべてに同じ出典・schema検証が適用されるわけではない。 | 許可した配布元とhashの台帳を全checkpointへ広げ、対応可能なPyTorch版ではweight-only読込を検討する。checkpoint schema/versionも検証する。 |
| 低 | placeholder経路が本推論と同じCLIにある | VAD推論は引数を省略すると配線確認用encoderを利用できる。random head/modelには明示フラグが必要な点は良いが、生成JSONが実験成果へ混入する余地は残る。 | placeholderを独立したsmoke-testコマンドへ分離し、出力に大きな警告とmodel provenanceを必須化する。 |

#### CPUスモークテスト対応状況（2026-07-16）

- `artifacts/checkpoints/emotion2vec_base.pt` のサイズとSHA-256を実測し、公式配布物の期待値と一致した。
- `vad_downstream.infer_vad_emotion` は `--device cpu`、実checkpoint、`scripts/test.wav` で完走し、`outputs/real_emotion2vec_smoke.json` を生成した。
- JSONの `random_model=true`、`target_dim=3`、VAD 3値、4クラス、`classifier_checkpoint=null`、確率和 `1.00000004470348` を確認した。
- この成功は実encoderのロードと推論配線を保証するが、ランダムheadのVAD値・クラス予測・確率に感情推定上の意味はない。
- `scripts/extract_features.py` とIEMOCAP一括特徴抽出に残る `.cuda()` 強制は、この推論CLIの成功によって解消されたわけではない。

### 3.3 IEMOCAP のデータ分割と評価

| 重要度 | 問題 | 根拠 | 改善案 |
|---|---|---|---|
| 重大 | 検証分割で話者リークし得る | テストはセッション単位だが、残り4セッションは `random_split` で発話単位に80:20分割される（`iemocap_downstream/data.py:178-195`）。各IEMOCAPセッションは男女1名ずつの対話なので、同じ話者の発話がtrainとvalidationへ入る可能性が高い。model selectionが話者固有特徴へ過適合し得る。 | 検証も話者またはセッション単位にする。外側test session、内側validation sessionのnested LOSO等を採用し、分割IDを保存する。 |
| 高 | セッション件数と順序をハードコード | `iemocap_downstream/main.py:32-47` は `[1085,1023,1151,1031,1241]` と、特徴がSession 1→5順で連続することを仮定する。抽出条件や欠損で件数が変わると、別セッションがtestへ混ざる。 | utterance IDからsession/speakerを解析してsplitを生成し、全IDの重複・欠損・期待クラスを検証する。位置ではなくIDをcheckpointへ保存する。 |
| 高 | 設定ファイルが実際の挙動を表さない | `dataset.test_ratio`、`dataset.fold`、`optimization.weight_decay`、`label_smooth`、`lr_scheduler` は定義されるが学習処理で参照されない（`default.yaml:7-19`）。optimizerはweight decayなし、scheduler値もコード固定（`main.py:60-61`）。 | 使用する設定だけに整理し、全ハイパーパラメータを設定から渡す。未知・未使用キーを起動時にエラーまたは警告にする。 |
| 高 | `eval_is_test=True` はtestをvalidationにも使用 | `data.py:165-176` はtest datasetをvalidationとして使い、同じtestで最良epochを選んだ後に報告する。これは独立した最終評価ではない。 | このモードを研究評価から削除するか、デバッグ専用と明記し、結果出力に `invalid_for_final_evaluation` を残す。 |
| 中 | クラス不均衡への学習対策と報告が限定的 | `CrossEntropyLoss()` は重みなし（`main.py:62`）。UAは出すが、fold別support、混同行列、複数seedの分散は標準出力されない。 | fold別support・混同行列・per-class recall/F1と平均±標準偏差を保存する。class weight/samplerは検証実験で比較する。 |
| 中 | 再現性設定が不完全 | `torch.manual_seed` のみで、Python/NumPy、DataLoader worker、決定論的algorithmは明示しない。PyTorch公式もseedだけではplatformやreleaseをまたぐ完全再現を保証しない。 | 全RNG、DataLoader generator/worker、deterministic設定を統一し、環境情報とsplitを成果物に保存する。複数seedで結論の頑健性を報告する。 |
| 中 | checkpoint選択がWAだけ | 最良epochはvalidation WAで決まり、主要指標としてUAを重視する実験とは選択基準がずれる可能性がある。 | 事前にprimary metricを決め、WA/UA/F1のどれで選んだかをcheckpointに保存する。 |

IEMOCAPが5組・10名の話者から成る根拠は、[USCのIEMOCAP原論文](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf)を参照した。

### 3.4 VAD 学習と説明可能性

| 重要度 | 問題 | 根拠 | 改善案 |
|---|---|---|---|
| 高 | train/valid/test作成と話者分離をCLIが保証しない | `train_head.py` と `train_vad_emotion.py` は別prefixを受け取るだけで、ID重複、話者重複、test splitを検証しない。validは省略可能で、その場合は最終epochを保存する。 | データ分割manifestを標準化し、split間ID重複を拒否する。speaker/session列を持たせ、group splitを生成・保存する。独立test評価CLIを追加する。 |
| 高 | 寄与度を因果的説明と誤解しやすい | `infer_vad_emotion.py:245-267,325-350` は線形重みと予測VADの積を計算する。これはlogitの恒等的分解で、入力音声への因果効果・特徴重要度・VADの正しさを検証しない。 | 名称を「linear logit decomposition」に限定する。真値VAD置換、次元ablation、VAD shuffle、直接分類baseline、介入実験、信頼区間を追加する。 |
| 高 | 分類損失がVAD表現を変形できる | joint lossではCCCとCrossEntropyの両方が同じVAD headへ逆伝播する（`emotion_training.py:25-35`）。`lambda` によって説明軸の意味と分類性能が変わる。 | VAD head凍結版、joint版、真値VAD分類、直接768次元分類を比較する。各lambdaでCCCと分類性能を同時報告する。 |
| 中 | CCCをミニバッチごとに最適化 | `training.py:7-29` はbatch内平均・分散・共分散でCCCを計算する。小batchやラベル分散が小さいbatchでは推定が不安定で、batch構成に損失が左右される。batch size 1では共分散が0になり、有用なCCC勾配を得にくい。 | 十分なbatch sizeを検証・強制し、MSE/MAEとの複合損失、分散を考慮したsampler、batch size別ablationを行う。評価時は現状どおり全標本を結合したglobal CCCを使う。 |
| 中 | 時間平均だけで発話を表現 | `model.py:112,217-233`。局所的ピーク、順序、話者内変化を捨てる。 | mean poolingをbaselineとして明記し、attention/statistics pooling（mean+std）等と同一splitで比較する。 |
| 中 | 出力VADの値域を拘束しない | 最終層は線形で、ラベルが `[-1,1]` でも予測は範囲外になり得る。必ずしも誤りではないが、JSON利用者の想定とずれる。 | raw値を残した上で範囲外率を評価し、必要ならtanh版と比較する。黙ってclipするとCCCを歪め得るため方針を明記する。 |
| 中 | 不均衡分類に重みなしCEを使用 | `emotion_training.py:31,53`。特にIEMOCAPのdisgustは少数になり得るが、class supportに応じた警告やsamplerはない。 | splitごとのsupportを必須表示し、欠損クラスを拒否する。weighted CE/focal loss/samplerはbaselineと比較して採否を決める。 |
| 低 | scheduler・early stopping・実験追跡がない | VAD CLIはAdamWとbest checkpoint保存を持つが、scheduler、patience、TensorBoard/W&B等はない。 | 最低限CSV/JSONLでepoch指標、実行コマンド、git commit、環境、データhashを保存する。early stoppingは同じprimary metricで行う。 |

## 4. ガイドの記述との照合

### 正しく記載されているが、未解決の制限

- 特徴抽出スクリプトが `.cuda()` を直接呼ぶ。
- 単一WAV特徴抽出が例外を再送出しない。
- IEMOCAPのセッション数・件数がハードコードされ、`dataset.fold` と `test_ratio` が未使用。
- placeholder/random modelは研究上の意味を持たない。
- end-to-end fine-tuning、データセット固有VAD前処理、実験追跡等が未実装。
- 公式encoder checkpointはローカルに取得済みだがGitには含めない。学習済み分類head、IEMOCAP、VADラベルは引き続き必要で、実データは同梱されない。

### 修正または補足すべき記述

- 「PyTorch 1.13以上」は互換性保証として広すぎる。動作確認済みの完全な版とCPU/CUDA buildを示すべきである。
- 「5-fold leave-one-session-out」はtest分割については正しいが、内側validationが話者独立ではない点も併記すべきである。
- VAD寄与度は「説明」より「線形logitの分解」と表現する方が正確である。
- `TESTING.md` の期待値 `Ran 32 tests` は古い。2026-07-14時点の静的集計では `tests/` に48個の `test_*` メソッドがある。実際のdiscover件数は実行環境で確認すべきである。
- `data/` は空であるというガイドの説明と現状は一致した。
- 実encoderを使うCPU配線テストの目的と制約は `REAL_EMOTION2VEC_SMOKE_TEST_JA.md` に分離して記載されている。これは学習済みheadの性能検証ではない。

## 5. テストと検証状態

### 今回実施した確認

- 対象コード、設定、テストの静的確認: **実施済み**
- `.env` のGit追跡状態: **確認済み**（値は出力・転載していない）
- `data/` 内の実データ有無: **空であることを確認**
- 現行テストメソッドの静的集計: **48件**
- 指定されたWSL import確認: **成功**（PyTorch `2.12.0+cu130`、NumPy `1.26.4`、fairseqを含む）
- 指定されたWSL unit test: **48件成功**（`Ran 48 tests ... OK`）
- 公式checkpointを使う実encoder CPU配線テスト: **成功**（2026-07-16、ランダムVAD・分類head）
- 公式checkpointのサイズ・SHA-256: **確認済み**
- 学習済みVAD経由分類headを使うCPU E2E: **未検証**
- CUDA/GPU推論: **このPCでは対象外**（Intel Iris Xeのみ）
- IEMOCAP/VAD実データによる学習・評価: **未検証**

### WSLテストの実行方法と制約

通常のCodexサンドボックスは制限ユーザーとして動作するため、所有者ユーザーに登録されたWSL distributionとPATHが見えず、当初は `WSL_E_DISTRO_NOT_FOUND` となった。これは環境が存在しないことを意味しなかった。許可付きで次のコマンドを実行すると、既存の `Ubuntu` と専用Pythonを利用でき、48件すべて成功した。利用者自身もWindows PowerShellから同じコマンドを実行できる。

```powershell
wsl -d Ubuntu `
  --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec `
  -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python `
  -m unittest discover -s tests
```

なお現行テストは、VADデータ検証、pooling、CCC、checkpoint、CLI、寄与度計算を広く単体検証しており、追加実装の退行防止として有用である。これに加え、手動スモークテストで実fairseq checkpointから分類JSONまでの配線を確認した。一方、学習済みheadの性能、実IEMOCAPによる学習・評価、環境の新規構築、話者独立splitの妥当性を保証する統合テストではない。

## 6. 推奨する改善順

### 優先度1: 秘密情報と成果物の保護

1. [x] `.env` を追跡対象から外し、`.env.example` を導入する（2026-07-14対応）。
2. [x] `.gitignore` にcheckpointと推論出力を追加する（2026-07-16対応）。
3. [ ] `.gitignore` に特徴量、Hydra出力、学習ログ、cacheを追加する。
4. [ ] 外部データと学習済みheadの出典・hash・利用条件を台帳化する。公式encoder checkpointの配布元・サイズ・SHA-256は記録済み。

### 優先度2: 再構築可能な環境

1. [x] 既存のWSL専用環境で主要importと単体テスト48件を実行する（2026-07-14確認）。
2. [x] 公式checkpointのCPU smoke testを通す（2026-07-16確認）。
3. [ ] 現在の環境一覧を退避する。
4. [ ] Python 3.10、pip、CPU版PyTorch、NumPy、fairseq、全推移依存をlock/constraintsへ固定する。
5. [ ] `environment.yml` とbootstrapスクリプトを用意し、新規環境から1コマンドで再構築する。
6. [ ] WSL distribution名とPython絶対パスをスクリプト設定へ集約し、個人名依存を利用者から隠す。

### 優先度3: 評価リークの排除

1. IEMOCAP splitを位置・固定件数ではなくutterance ID、session、speakerから生成する。
2. validationも話者独立にし、testをmodel selectionへ使うモードを研究評価から除外する。
3. VADデータにもsplit manifestと重複検査を導入する。

### 優先度4: 特徴抽出の堅牢化

1. 裸の`except`を修正し、失敗時は非0終了にする。
2. `scripts/extract_features.py` とIEMOCAP特徴抽出の `.cuda()` を `--device auto/cpu/cuda` と `.to(device)` へ変更する。
3. [x] `scripts/test.wav` と実emotion2vec checkpointを使い、ランダムheadまでのCPU配線テストを通す（2026-07-16確認）。
4. 学習済み分類headを使い、CPUでWAV入力から分類JSONまでを検証するE2Eテストを追加する。
5. E2EではVAD/logit/確率の有限性、確率和、ラベル集合、model/checkpoint hash、非0終了を自動検証する。
6. 入力、出力shape、checkpoint schema、特徴抽出の途中再開を検証する。

### 優先度5: モデル評価と説明の検証

1. 直接768次元分類、真値VAD分類、凍結VAD head、joint学習を同一splitで比較する。
2. lambda、batch size、pooling、seedのablationを行う。
3. 「寄与度」を線形logit分解と呼び、VAD介入・shuffle・次元除去で説明の頑健性を検証する。

## 7. 最小限の合格基準

このリポジトリを第三者が研究評価に使える状態と判断する最低条件は次のとおりである。

- 新規のWSL/Ubuntuまたはコンテナから、固定依存を1コマンドで構築できる。
- CPU unit testが全件通り、実checkpointと実WAVから分類JSONまでのCPU E2E testも通る。
- 各実験にgit revision、環境、data/checkpoint hash、split ID、seedが保存される。
- train/validation/test間でutteranceとspeakerの重複がないことを機械的に検証する。
- IEMOCAPのfold別support、WA、UA、F1、混同行列と複数seedのばらつきを報告する。
- VAD分類について、VAD CCCと分類性能を同時に示し、直接分類baselineとの差を示す。
- 寄与度を因果説明として主張せず、線形分解として適切に限定する。

## 8. 総合評価

現状は「VADを中間表現にするアイデアを実装し、単体テストに加えて実emotion2vec encoderのCPU配線まで確認できた段階」である。特にVAD系の入力検証、masked mean pooling、checkpoint metadata、random modelの明示フラグ、公式checkpointのhash確認手順は良い基盤である。

ただし、環境固定、話者独立データ分割、学習済みhead、実データ統合評価が不足しているため、まだ「第三者が同じ結果を再現できる完成した研究パイプライン」とは評価できない。秘密ファイルとcheckpoint・推論出力の誤コミット対策は前進した。次はモデルを複雑化する前に、環境とsplitを固定し、実データで直接分類baselineとVAD忠実度を同じ条件で比較することが最も効果的である。
