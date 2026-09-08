# Plan mode入力用プロンプト — emotion2vec+ large C/D

改訂日: 2026-09-07（Asia/Tokyo）

以下をPlan modeへ貼り付けて使用する。[現行計画・実装仕様](2026-08-22-msp-hcudb-feature-decoder-plan.md)の第1〜11節・第17〜22節が唯一の現行基準である。本プロンプトは計画作成を依頼するもので、実装や実験の開始指示ではない。

---

このリポジトリでemotion2vec+ largeのC/D実験を実装するための具体的なimplementation planを作成してください。まずAGENTS.md、dirty worktree、既存コード・config・Notebook・tests・利用可能なcheckpoint metadata/manifestを調査し、既存変更を保護してください。2026-09-07の現行計画にはファイル単位の初期調査とschema案があります。現物を確認して必要な箇所を具体化してください。

この段階ではコード・API・Notebook・実行用configを変更せず、checkpoint取得、特徴抽出、学習、評価などの実験を実行しないでください。現行計画の第12〜16節の実施記録をLarge C/Dの実装済み機能として扱わないでください。

## 確定済みの実験条件

- Cはemotion2vec+ large公式checkpointのencoderと公式9-output projを追加学習せず使用する。ただし評価は共通4 logitsだけを使う4-class SER baselineであり、9-way性能の評価ではない。
- DはCと同じ公式checkpoint・同じproj初期値から開始し、encoder固定で公式projのみをHCUDB trainに適応する。新規decoderやMSP学習済みdecoderに置き換えない。
- C/Dとも **9 logits → angry/disgusted/happy/sadの4 logits選択 → 4-way argmax / softmax** を使う。Dのlossは **選択4 logitsへのCross Entropy** とする。9-way CEは禁止する。
- 統一クラス順は **anger / disgust / happy / sadness**。4×4混同行列は行=true、列=predictionでこの順に固定する。OUTSIDE列は作らない。
- 対象外5 logitsは出力・保存するが、主評価の予測候補・lossから除外する。9出力中でneutral等が最大でも発話を除外せず4 logits内で予測する。
- 9-way softmax後の5クラス削除を標準処理にしない。主評価・D学習で使う選択関数は共通化する。
- 研究上の比較単位はA ↔ B、C ↔ D。A/Bの定義を再定義せず、A/Bとの絶対性能の直接比較を成立させるためにC/Dを変更しない。A/Bの完了はC/Dの開始条件にしない。
- 以前のBase cache、新規4-class decoder、IEMOCAP/VADの旧評価規則はC/Dへ流用しない。

## 固定データ・モデル選択・解釈

- 評価対象はMSP-Podcast R1.10とHCUDB1のみ。C/Dでcheckpoint、encoder、両test集合、mapping、前処理、特徴抽出/pooling、評価規則を共通にする。
- MSPは事前固定したTest1利用可能部分集合。Train/Development/Test1の公式割当てを保持し、Test2・SpkrID=Unknownは既存契約どおり対象外。
- msp_missing_audio_exclusions_v1（1,128件、うちTest1 330件）を引き継ぐ。既存manifestで承認済み音声重複除外を使用している場合は、そのSHAと集合も固定する。件数だけで採用集合を推測せず、現在の正式manifest/provenanceを確認する。
- HCUDBはhcudb1_speaker_split_v1を維持。train=FA, FB, FD, FH, FI, FL, MC, MJ, MM, MN、validation=FF, MK、test=FG, ME。学習seedでsplitを変更しない。
- 元ラベルを保持する。MSPはA→anger、D→disgust、H→happy、S→sadness。HCUDBは怒り→anger、嫌い→disgust、狂喜・楽しい/余裕・嬉しい→happy、憂鬱・悲しい→sadness。「嫌い→disgust」は近似対応としてversion管理する。
- Dの設定・checkpoint選択・early stoppingはHCUDB validationのみ。MSPを学習/validationに使わず、HCUDB test・MSP test・Cのtest性能で設定や評価規則を調整しない。
- C両testの結果・個別予測・再現情報保存後にD正式学習を開始する。Dは42/43/44の3 seedで、各seedが同じ公式projから独立して開始する。
- Cは固定1点。Dのseed別結果・mean/SDと各D−Cを保存する。標準偏差はddof=1、欠落seedはpartialとして正式3 seed集計と区別する。
- encoderの新たな日本語学習、純粋な言語適応とは解釈しない。固定表現上の分類層をHCUDBへ適応した効果と、MSP性能への影響として扱う。
- MSPを完全未知英語データと呼ばない。emotion2vec事前学習にMSP v1.8が使われた既存記録と、今回のLarge checkpoint・特定test発話の重複確認を区別する。
- 任意の3-class sensitivity analysisはanger/happy/sadnessの別contract・別出力とし、主4-classを変更しない。

## 調査時に特に確認する落とし穴

1. 現行contracts.pyとmappings.v1.jsonの順はanger/happy/sadness/disgust。整数class_indexをC/D targetとして直読しない。元manifestを旧契約で検証した後、mapped_emotionからC/D indexを生成する。元manifest SHAとC/D label view signatureを別々に保持し、global順変更で旧結果を壊さない。
2. READMEの9出力順なら選択indexは[0,1,3,6]だが、実checkpoint/configとモデル実体で確認するまで確定値にしない。model ID候補はiic/emotion2vec_plus_large。revision/hash、proj構造・入力次元、公式前処理/poolingを推測しない。
3. features.pyのEmotion2vecEncoderはfairseq Base経路、model.pyのBaseModelはReLUを含む2層decoder。Large公式headと同一とみなさない。
4. training.pyは既定weight_decay=1e-4のAdamWを使う。4-logit CEで非対象行のgradが0でもweight decay等による更新は別に防ぐ。初期実装案はproj全体へのAdamW、weight_decay=0、新規optimizer状態。対象外5行のweight/biasとoptimizer状態を検証し、不適合なresumeを拒否する。必要なら対象4行だけの明示更新へ切り替える。
5. 公式APIのscoresを生logitsと誤認しない。softmax前のproj出力を計測する。確率の対数で生logitsを代用しない。
6. evaluation.py/checkpoints.py/study.py/集計scriptは旧クラス順・BaseModel・MSP親→HCUDB子に依存する。既存CLIやNotebookはC/D対応済みではない。

## 計画に必須の16項目

1. **再利用module:** readers/splits/manifest/exclusions/duplicates、seed/device、shard/mmap、指標計算・保存処理について、再利用と変更範囲をファイル単位で示す。
2. **新規module:** 現行計画のofficial_head.py案の役割を確認する。新規Pythonは固有のruntime/build/test役割があるものだけとし、先頭docstringを必須にする。版違い/backupファイルは作らない。
3. **公式checkpoint読込:** FunASR AutoModelの実経路、hub、固定revision/local snapshot、SHA-256、依存version、公式proj初期値の復元方法を示す。
4. **class order検証:** 実config・公式実装・proj出力数の整合を確認し、未知/欠損/重複ラベルやhead不整合を拒否する。
5. **9→4選択:** 共通選択関数の配置、target4への変換、同値argmax、4-way softmax、非有限値拒否を定義する。
6. **C評価:** optimizerなし、eval/勾配なし、state不変、両test結果保存を設計する。
7. **D学習:** 4-logit CE、HCUDB train、validation選択、seed別独立初期化、親/resume区別を設計する。初期案は非加重CE・smoothingなし、選択はUAR→macro F1→CE、完全同値は先のepoch。lr/batch/epoch/patience等の未確定値と確定方法を記す。
8. **encoder固定:** requires_grad=False、eval、optimizerから除外、parameters/buffers不変の検証を示す。
9. **proj行の検証:** 非対象5行grad=0、複数optimizer step後weight/bias不変、対象行更新、resume時の状態検証を計画する。
10. **Large cache:** ser_large_feature_cache_v1案、再開shard、ID/hash、revision・前処理・sample rate・normalization・pooling・次元・parity情報を設計する。Base cacheを拒否し、合格した同一cacheをC/Dで使用する。
11. **parity:** 10〜20発話の公式直接推論vs前処理→encoder→cache書込/再読込→集約→proj。waveformから9/4 logitsまで比較し、max/mean absolute differenceとargmax一致率を保存。atol/rtolを事前固定し、不一致解消まで本評価を止める。
12. **evaluation contract:** class order/mapping、公式9出力と4 indices、選択/確率/予測規則、指標・4×4行列方向、split/subset/preprocessing/cache/checkpointを含める。draft実行を拒否し、canonical hashで比較する。Dの適応後head hashは共通contractの外に置く。
13. **manifest/signature:** 固定test集合、source manifest SHA、ラベルview、元ラベル/音声hash、除外/重複除外SHAを照合し、不一致は比較拒否する。
14. **seed管理:** C 1点、D 42/43/44、同じsplit/cache、独立run ID/出力、C保存ゲート、best/last checkpoint、partial状態を設計する。
15. **保存形式:** metrics JSON、個別予測JSONL、4×4 CSV、9/4 logits、4確率、元/統一ラベル、speaker、correctness、再現情報を保存する。予定schemaはser_official_head_checkpoint_v1とser_cd_evaluation_result_v1。
16. **比較report:** 同一contract/test確認後、DとD−Cのseed別/mean/SD、クラス別指標と行列差、CSV/Markdownを保存済み結果から生成する。IDで対応付け、Cを独立3回として数えない。

## Notebook・テスト・実行分担

- Notebook 01は特徴抽出/cache/parity、Notebook 02は公式projのC評価とD学習/評価を呼ぶ。共通処理をNotebook間でコピーしない。旧「checkpointを読まない」制約で公式proj初期化を妨げない。
- builderを正とし、必要なNotebookのみ再生成する計画にする。既存IEMOCAP Notebookはデモ/回帰用に保持し、VAD Notebookを正式経路にしない。
- msp_unavailable_label_audit.ipynbは復元候補確認が済むまで再生成しない。
- 非学習テストと、optimizer step/実checkpoint/実音声を伴うテストを分ける。実データmanifest作成・parity・抽出・benchmark・学習/評価・optimizerテストはユーザーが実行する従来の分担を維持する。
- Large実音声benchmark、全件の時間・容量見積もり、必要容量+20%、疎通/正式出力の分離を計画する。Base benchmarkをLarge実測値とみなさない。

## Plan modeの出力要件

- 既存実装、不足実装、変更対象file、新規file、test strategy、実行順序を明示する。
- 各ステップに再利用する処理、変更予定ファイル、検証、完了条件を付ける。
- C/D研究規則の確定事項と、実checkpoint・環境・manifestからの未確認事項を区別する。4-logit選択やOUTSIDEなしを未確定に戻さない。
- ローカル情報から先に調べ、確認できないmetadata・パス・hash・index・入力次元を捏造しない。
- 既存ユーザー変更との競合を明示し、過去の実施記録を改変しない。
- C開始前、D正式学習開始前、C/D比較前の完了条件を別チェックリストとして提示する。
- この段階では計画を提示するだけで、実装・実験を開始しない。

---
