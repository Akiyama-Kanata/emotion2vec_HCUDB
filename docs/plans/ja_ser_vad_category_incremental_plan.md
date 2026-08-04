# 固定emotion2vecを用いた日英SER比較の段階的計画

## 研究目的

研究の主目的は、エンコーダーを更新せず、固定したemotion2vec特徴を用いる下流SERシステムによって日本語音声感情認識を改善し、同時に英語音声感情認識性能を維持することである。日本語評価はHCUDB、英語評価はIEMOCAPを中心に行う。

主比較はemotion2vec Baseとemotion2vec+ largeである。両条件ともエンコーダー重みは常に固定し、学習対象は新規デコーダーだけとする。本研究はemotion2vec自体の日本語適応を目的としない。

## 主実験: Base対Large

BaseとLargeには、各エンコーダーが出力する固定特徴へ同等構造のデコーダーを接続する。

- デコーダーの`input_dim`だけを各エンコーダーの出力次元に合わせる。
- 入力層より後の隠れ層数、隠れ次元、活性化、dropout、出力層を同一にする。
- データ分割、感情ラベル、前処理、乱数seed、epoch数、optimizer、学習率、batch条件、モデル選択規則を共通化する。
- エンコーダーからの特徴抽出は勾配なしで行い、エンコーダーのパラメータをoptimizerへ渡さない。
- 公開API、CLI、checkpoint形式は今回変更しない。将来の共通実験入口では、デコーダーの`input_dim`をエンコーダーごとに設定可能にする。

### 共通データ条件

- HCUDBは話者独立のtrain / validation / test分割を用い、同一話者を複数splitへ入れない。
- IEMOCAPも話者またはsession単位で分割し、評価話者を学習へ混ぜない。
- 日英で比較する感情ラベルの対応表を事前に固定し、結果を見てから統合・除外しない。
- HCUDBの`valence`、`arousal`、`emotion`を利用する。正解のない`dominance`は日本語の損失および評価対象にしない。

### 共通評価指標

- 主指標: macro F1、UA
- 補助指標: WA / accuracy、クラス別F1、confusion matrix
- 複数seedの各結果と要約統計を保存する。
- 日本語改善と英語維持を別々に報告し、Base対Largeの差を同一条件で比較する。

「英語性能維持」の許容範囲と統計的な比較方法は、本実験開始前に固定する。実験結果を見た後で基準を変更しない。

## 探索条件: VAD媒介型

`emotion2vec特徴 -> VA/VAD予測 -> 感情カテゴリ`のVAD媒介型は、主比較ではなく、時間に余裕がある場合に確認する探索条件とする。Base対Largeの共通直接分類実験を完了するまで、VAD媒介型の追加実験を優先しない。

- VAD媒介型の分類性能は、直接分類デコーダーと分けて報告する。
- HCUDBではVとAのみ教師ありとし、Dを実測値と同等に扱わない。
- VADを使う場合もエンコーダーは固定し、学習対象はデコーダーだけとする。
- VAD媒介型の結果は、Base対Largeという研究の中心的比較の代替にしない。

## 実施段階

### Phase 1: エンコーダー別入力次元の設定と共通入口

「特徴次元可変化」はプロジェクト内の作業名であり、研究目的や一般的な標準用語ではない。この作業では、特徴の切り詰め、射影、次元圧縮、または同一実行中の次元変更を行わない。固定した各エンコーダーが出力する特徴次元を、実験開始時に`input_dim`として一度だけ設定し、Base / Largeの公平比較に必要な共通入口を作る。

#### 2026-08-03時点の障害分類

| 分類 | 対象 | 現状 | Phase 1での扱い |
|---|---|---|---|
| パイプライン制約 | `vad_downstream/data.py` | `FEATURE_DIM = 768`を連結特徴の読込時に無条件で検証する | データ読込へ`expected_input_dim`を渡し、学習開始前に全特徴を検証する |
| パイプライン制約 | `vad_downstream/notebook_pipeline.py` | cache読込・生成時の検証が`FEATURE_DIM = 768`固定である | `FeatureCache`が`encoder_id`と`expected_input_dim`を必須情報として保持する |
| パイプライン制約 | `iemocap_downstream/main.py` | `BaseModel(input_dim=768, ...)`を直書きしている | 設定の`model.input_dim`を渡す |
| 設定不足 | `iemocap_downstream/config/default.yaml` | `model.input_dim`が存在しない | 既定値768の設定項目を追加する |
| Notebook制約 | `iemocap_downstream/inference.ipynb` | モデル入力と合成特徴を768で直書きしている | 実験設定またはcheckpointの`input_dim`を使用する |
| cache識別不足 | `vad_downstream/data.py`の単体`.npy` cache | ファイル名と内容にencoder ID、期待次元、抽出契約がない | manifestに識別情報を保存し、再利用前に一致を検証する |
| checkpoint識別不足 | VAD / 直接分類のcheckpoint保存経路 | `input_dim`は概ね保存済みだが、encoder IDは任意または未保存である | 新しい共通入口ではencoder IDを必須metadataとして保存する |
| 推論fallback | `vad_downstream/infer_vad_emotion.py` | random head作成時は768固定、旧checkpointは欠損時768へfallbackする | 旧checkpoint fallbackは維持し、新しい明示設定がある場合だけ上書きする |
| 推論fallback | `vad_downstream/infer_parallel_emotion_vad.py` | 旧checkpointの`input_dim`欠損時に768へfallbackする | 後方互換の正常な既定値として維持する |
| モデル既定値 | `vad_downstream/model.py`、`iemocap_downstream/model.py` | コンストラクタの既定値は768だが、引数自体は受け取れる | 公開API互換のため維持する。Largeを拒否する原因とは分類しない |
| encoder既定値 | `upstream/models/config.py` | emotion2vec Baseの`embed_dim=768`を表す | Baseモデルの正常な定義として変更しない |

READMEや説明資料、既存テストfixtureに記録された768は、Baseの事実または既存例であり、実行時制約でない限りPhase 1の変更対象にしない。

#### 共通入口のデータフロー

共通入口は次の順序を固定する。

```text
encoder_id
  -> 選択した特徴抽出契約に対応する expected_input_dim
  -> cache manifest / 特徴配列の事前検証
  -> decoder(input_dim=expected_input_dim)
  -> checkpoint と評価生成物へ識別情報を保存
```

1. 実験設定に`encoder.id`、`encoder.input_dim`、抽出粒度、checkpointまたはmodel revisionを明示する。
2. `input_dim`はモデル名だけから暗黙推定しない。とくにemotion2vec+ largeは、raw encoder表現と外部推論APIが返す`feats`を同一の抽出契約とみなさない。採用する抽出経路を固定し、その出力次元を設定値とする。
3. cache生成時に、少なくとも`encoder_id`、`input_dim`、抽出契約、元checkpoint / revisionの識別子をmanifestへ保存する。
4. cache読込直後、デコーダー作成とoptimizer作成より前に、全特徴が2次元`[T, C]`であり、`C == expected_input_dim`で、有限値だけを含むことを検証する。
5. 検証済みの`expected_input_dim`だけをデコーダーの`input_dim`へ渡す。特徴からの自動的な切り詰め、padding、射影は行わない。
6. checkpointには`input_dim`とencoder識別情報を保存し、推論時も実encoder、cache、checkpointの組合せを再検証する。

採用する設定値は、実験開始前に実特徴1件を抽出して確定する。現行Baseの既定抽出は768次元である。[emotion2vec+ largeの公式設定](https://huggingface.co/emotion2vec/emotion2vec_plus_large/commit/7714a5c669f5711ac128c78f35732e7f9cc976ad)には`embed_dim: 1024`がある一方、[公式README](https://github.com/ddlBoJack/emotion2vec/blob/main/README.md)のFunASR利用例は返却`feats`を768次元として説明しているため、Largeの`input_dim`は使用する抽出APIを決めずに1024または768へ固定しない。Phase 1実装時に採用した抽出経路、実測shape、設定値を同時に記録する。

#### 公平比較の不変条件

Base / Large間で変更を許すのは、`encoder.id`、encoder checkpoint / revision、抽出結果の`input_dim`、およびそれらから派生するcache識別情報だけである。デコーダーは入力層の`in_features`だけを変え、隠れ層数、`hidden_dim`、活性化、dropout、出力層、loss、optimizer、学習率、batch条件、seed、split、ラベル、モデル選択規則を同一にする。

設定比較とモデル構造比較の両方をテストし、入力層以外のparameter名とshapeが一致しない場合は実験を開始しない。

#### 互換性条件

- 既存の公開コンストラクタ、CLI、設定を省略した呼出しでは768を既定値として残す。
- 既存checkpointで`input_dim`が欠ける場合は、従来どおり768として読めるようにする。
- 既存checkpointにencoder IDがない場合は`legacy / unknown`として読込可能にする。ただし、新しいBase / Large共通実験入口では混用検証ができないため、そのまま主実験へ投入しない。
- 新しいmetadataは既存keyを削除・改名せず追加する。既存checkpointのstate dict形式も変更しない。
- 厳格な整合性検証は新しい共通実験入口で必須とし、既存の低水準APIを呼ぶだけの利用は壊さない。

#### 学習開始前に失敗させる条件

- 設定した`input_dim`と実特徴の最終次元が異なる。
- 同一datasetまたは同一cache集合の特徴次元が揃っていない。
- cache manifestのencoder ID、`input_dim`、抽出契約、checkpoint / revisionが実験設定と異なる。
- 初期checkpointのencoder IDまたは`input_dim`が現在の実験設定と異なる。
- encoder IDは同じだが、異なる抽出契約のcacheを指定した。
- Base / Large設定間で、許可項目以外のデコーダーまたは学習条件が異なる。

失敗時は期待値、実測値、対象ファイルまたはcheckpointを含む`ValueError`を出し、デコーダー作成、optimizer作成、学習、cache上書きへ進まない。

#### 次回実装時のテスト仕様

1. `tests/test_vad_downstream_data.py`へ、768以外（例: 1024次元）の合成連結特徴を`expected_input_dim=1024`で読み込めるテストを追加する。
2. 同ファイルへ、設定1024に対して実特徴768を渡すと、学習開始前に対象path・期待1024・実測768を含むエラーになるテストを追加する。
3. 単体`.npy` cacheのdatasetについても、全recordの次元を事前検証し、不一致を`__getitem__`の途中まで遅延させないテストを追加する。
4. `tests/test_notebook_pipeline.py`へ、cache manifestのencoder IDまたは`input_dim`が異なる場合に既存`.npy`を再利用しないテストを追加する。識別不一致時は無言で上書きせずエラーにする。
5. Base設定とLarge設定から直接分類デコーダーを生成し、最初の`Linear.in_features`だけが異なり、以降の層型・parameter名・shape・学習ハイパーパラメータが同一であることを検証する。
6. checkpoint読込時に、現在のencoder IDまたは`input_dim`とcheckpoint metadataが異なれば拒否し、両方が一致すれば読めることを検証する。
7. 既存APIを引数なしで呼んだ場合と、旧checkpointに新metadataがない場合の768 fallbackが維持される回帰テストを残す。

次回の最初の実装対象は`vad_downstream/data.py`と`tests/test_vad_downstream_data.py`である。`_validate_features`へ期待次元を渡し、連結特徴と単体cacheの両方をデコーダー構築前に検証する。ここが通るまでNotebook、IEMOCAP、推論fallback、cache manifestへ変更範囲を広げない。

### Phase 2: HCUDB共通実験

- 共通の話者分割と感情ラベルを確定する。
- Base特徴とLarge特徴を同じsplitから作成する。
- 各条件で同一学習条件の直接感情分類デコーダーを学習する。
- macro F1、UA、WA / accuracy、クラス別F1、confusion matrixを保存する。

### Phase 3: IEMOCAP共通実験

- HCUDBと対応可能なラベルおよびIEMOCAPの話者独立分割を固定する。
- BaseとLargeに同等構造のデコーダーを接続し、HCUDB側と対応する条件で評価する。
- 日本語側の改善だけでなく、英語側の性能維持を確認する。

### Phase 4: 探索的VAD条件

- 主実験完了後、必要性と時間を再確認する。
- 直接分類、並列VA/D補助出力、VAD媒介型を区別して報告する。
- 合成データや未学習モデルの出力を研究成果へ含めない。

## 完了条件

次のすべてを満たした時点で主実験を完了とする。

- BaseとLargeのエンコーダーが固定され、デコーダーだけが学習されたことを記録で確認できる。
- 両条件のデータ分割、感情ラベル、学習条件、評価指標が共通である。
- 入力層の`input_dim`以外のデコーダー構造が同一である。
- HCUDB実データによる日本語評価結果が保存されている。
- IEMOCAP実データによる英語評価結果が保存されている。
- Base対Largeの比較表と、英語性能維持の事前基準に対する判定が作成されている。
- デモ結果と研究結果が明確に分離されている。

## 現在の最小作業

Phase 1の設計は2026-08-03に確定した。次回は設計判断を挟まず、`vad_downstream/data.py`へ期待入力次元を渡す実装と`tests/test_vad_downstream_data.py`の非768・不一致テストから開始する。コード変更と実験実行はまだ行っていない。
