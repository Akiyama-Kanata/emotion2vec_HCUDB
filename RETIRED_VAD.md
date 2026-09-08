# 独立VAD実装の保管案内

2026-09-08に、独立VAD実装・専用Notebook・専用テスト・指定された旧実装と資料をリポジトリ外へ移動しました。

- 保管先: `C:\Users\RD004\Documents\lab\emotion2vec_retired_2026-09-08`
- [ファイル台帳](docs/maintenance/vad-retirement-2026-09-08/ledger.csv): 元の相対パス・元絶対パス・保存先・サイズ・SHA-256・移動理由
- [移動／照合／復元スクリプト](docs/maintenance/vad-retirement-2026-09-08/retirement.ps1)
- [ファイル単位の移動完了記録](docs/maintenance/vad-retirement-2026-09-08/operations.jsonl)
- [非学習テスト結果](docs/maintenance/vad-retirement-2026-09-08/non-training-tests.txt)
- [最終照合結果](docs/maintenance/vad-retirement-2026-09-08/verification.json)

8月22日分の保管フォルダとは別の場所です。保管先でも元の相対パスを維持しています。台帳と復元スクリプトは保管先の直下にも保存しました。

## 残したもの

A/Bの実装・資料・過去結果、C/D計画と関連資料、SERとIEMOCAPの実装・互換性テスト・デモ、HCUDB分析Notebook、MSPダウンロード関連一式、除外記録、共通文献を保持しています。C/D計画は整理前の状態のまま保持しており、この整理で計画の実装は行っていません。

`runs/`、特徴キャッシュ、checkpoint、固定manifest、画像は現位置に保持しました。`tests/fixtures/vad_dummy/cache/`の8個の特徴キャッシュも保持対象です。このためVAD fixtureから移動したのは`vad_labels_dummy.csv`で、fixtureディレクトリ自体は残っています。

VADパッケージ配下のPythonバイトコードとNotebook自動保存も台帳に含めて移動しました。`.ipynb_checkpoints/experiment-checkpoint.ipynb`はNotebookの自動保存であり、保持対象のモデルcheckpointとは異なります。ファイルのみを移動し、移動元の空ディレクトリは残しています。

共通説明とVAD説明の混在文書は保持しました。構成案内・README・テスト案内では保管先へ誘導し、過去の実施記録・監査記録・共通文献は本文を変更していません。既存の計画書2件の未コミット変更、SVGの削除状態、未追跡画像を保持しています。コミットは作成していません。

## 照合

リポジトリ直下のPowerShellで実行します。

```powershell
& ./docs/maintenance/vad-retirement-2026-09-08/retirement.ps1 -Mode Verify
```

台帳の全102ファイルについて、移動元にファイルがないこと、保管先のサイズとSHA-256が台帳と一致することを確認します。不一致があれば直ちに停止します。

## 復元

復元が必要になったときだけ、次を実行してください。保管先への書き込み権限が必要です。

```powershell
& 'C:\Users\RD004\Documents\lab\emotion2vec_retired_2026-09-08\retirement.ps1' -Mode Restore
```

復元前に全ファイルのパス・サイズ・ハッシュを検査し、保管先から元の相対パスへファイル単位で戻します。既存ファイルへの上書き、ルート外への移動、リンク先への移動は拒否します。各ファイルの復元後にもサイズ・ハッシュを照合し、失敗時はそこで停止します。途中まで移動された場合も、元の場所に残る一致ファイルを確認して残りだけを復元できます。

復元は台帳にあるファイルに限ります。今回の案内文書の編集は自動で戻しません。復元後に案内を更新してください。作業前から存在した変更を消さないため、`git reset --hard`や一括`git restore`は使わないでください。

`docs/maintenance/vad-retirement-2026-09-08/`には作業前のGit状態、追跡ファイルのSHA-256、保持資産のサイズ・更新日時・画像等のSHA-256も保存しています。大きなruns/checkpointの照合はサイズ・更新日時によるもので、全資産のSHA-256再計算ではありません。
