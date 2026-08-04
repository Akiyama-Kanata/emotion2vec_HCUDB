# 次チャット引き継ぎ

## 最終更新

2026-08-03

## 現在地

Base / Large共通実験入口のPhase 1設計を確定した。コードは未変更で、次回はデータ層の期待入力次元対応から開始できる。直前の基準状態は全65テスト成功。

## 完了したこと

- 768固定を、パイプライン制約、設定不足、cache / checkpoint識別不足、正常な後方互換fallbackへ分類した。
- 共通データフロー、学習開始前の失敗条件、公平比較の不変条件、既存API / checkpoint互換条件を計画書へ追記した。
- 非768受理、不一致拒否、入力層だけの差、cache / checkpoint混用拒否、旧768 fallbackの次回テスト仕様を固定した。
- コード、公開API、checkpoint、cache形式は変更していない。

## 未完了 / 次の最小ステップ

最初に`vad_downstream/data.py`の`_validate_features`とデータセット構築へ`expected_input_dim`を渡す。次に`tests/test_vad_downstream_data.py`へ1024次元受理と、設定1024 / 実特徴768の不一致拒否テストを追加する。このデータ層テストが通るまで他経路へ変更を広げない。

## 重要な前提

- `vad_downstream/data.py`と`vad_downstream/model.py`の既存未コミット変更は保持されている。
- エンコーダーはBase、Largeとも固定し、主比較ではデコーダー条件をそろえる。
- 「特徴次元可変化」は研究目的ではなく、エンコーダー別の固定入力次元を設定する準備作業である。切り詰め、射影、次元圧縮はしない。
- Largeの期待次元はモデル名から推測せず、採用した抽出APIの実特徴shapeと設定を一致させる。
- Base / Large間で変更できるのはencoder識別情報と入力層の`in_features`だけで、隠れ層以降と学習条件は同一にする。
- 合成音声、仮特徴、random modelの数値を研究成果として扱わない。

## 変更ファイル

- `docs/plans/ja_ser_vad_category_incremental_plan.md`
- `archive/logs/2026-08-03-work-log.md`
- `archive/logs/next-chat-handoff.md`

## 検証状況

- 今回は文書のみ変更したため、テスト未実行。
- 直前の基準状態: 全65件成功、failure 0件、error 0件。
- コード変更、失敗テスト、実験実行はなし。

## 注意点

- 新しい共通入口では、特徴・cache・checkpointのencoder ID、`input_dim`、抽出契約の不一致をoptimizer作成前に拒否する。
- 既存APIと旧checkpointの768 fallbackは壊さず、新しい実験入口だけを厳格にする。
- emotion2vec+ largeはraw encoder設定とFunASR返却`feats`で参照される次元が異なるため、抽出経路を固定してから期待値を確定する。
- WSLでテストする場合、サンドボックス内では`Wsl/Service/E_ACCESSDENIED`になるため承認実行が必要。
