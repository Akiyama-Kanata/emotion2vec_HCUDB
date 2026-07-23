# 次チャット引き継ぎ

## 最終更新

2026-07-19

## 現在地

英語感情認識性能を維持しながら日本語性能を向上させるため、emotion2vec-base、emotion2vec+ large、下流感情分類head、VAD回帰head、本体の日本語部分fine-tuningの位置づけを検討中。

現在の主要実装は、固定したemotion2vecの事前抽出特徴からVAD回帰headと感情分類headを学習するもの。本体は更新していないため、現段階の実験は厳密にはemotion2vecのfine-tuningではなくhead training / head tuningである。

## 完了したこと

- 本体固定、部分FT、全層FTの用語を整理した。
- HCUDBで本体上位1～2層を部分FTする二段階案を整理した。
- emotion2vec+ largeはbaseとは別に部分FTする必要があると確認した。
- VADを感情分類の必須経路、並列補助タスク、比較用ボトルネックのどれとして扱うか整理した。
- 本体部分FTに必要な入力、ラベル、暫定GPU目安を整理した。

## 未完了 / 次の最小ステップ

次のどちらを研究の中心にするか決める。

1. 分類精度中心：emotion2vec特徴から感情を直接分類し、VADは並列補助出力または比較条件にする。
2. 説明可能性中心：`emotion2vec特徴 -> VAD -> 感情`を主経路として、直接分類との性能差を検証する。

決定後、比較表を確定し、各データセットの感情カテゴリ、実測VADラベル、話者分割を一覧化する。

## 重要な前提

- 本体固定＋新規head学習は「emotion2vecのfine-tuning」ではなく「下流head training」。
- 本体上位層まで更新した場合は「partial fine-tuning」と呼べる。
- emotion2vec+ largeの付属9クラス分類headへVADを単純挿入することはできない。VAD媒介型では付属headを新規headへ置き換え、native 9-class headは参考条件として残す。
- VAD教師値なしで分類lossだけから3次元中間表現を学習した場合、それを実測VADと同等に扱わない。
- HCUDBで本体を適応する場合、test話者を適応学習へ混ぜない。

## 変更ファイル

- `archive/logs/2026-07-19-work-log.md`：今回新規作成。
- `archive/logs/next-chat-handoff.md`：今回の検討内容へ更新。
- `DEEP_LEARNING_EXPLANATION.md`：既存の未コミット変更あり。今回未変更。

## 検証状況

- リポジトリ内のモデル構造と公式のemotion2vec+ largeモデル情報を確認済み。
- 学習、VRAM計測、精度評価は未実行。
- コード変更とテスト実行は未実施。

## 注意点

- base上位1～2層の部分FTはVRAM 12～16GBが最低候補、24GBが暫定推奨だが、実測前の目安。
- +large上位1～2層の部分FTは24GBが最低候補、40～48GBが暫定推奨だが、実測前の目安。
- PC購入前にクラウドGPUで最長クラスの音声を使ったforward/backwardスモークテストを行う。
- IEMOCAPなどを外部クラウドへ置く前に、利用規約と大学の研究データ管理規程を確認する。
