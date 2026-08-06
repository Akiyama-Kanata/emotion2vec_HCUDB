"""Build the lecture-style IEMOCAP downstream experiment notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "iemocap_base_downstream_training.ipynb"


def markdown(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.strip() + "\n"}


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    markdown(
        """
# IEMOCAP Base downstream training：講義・実験Notebook

## 1. 実行順とデータの役割

上から順に、環境確認 → データ準備・特徴抽出 → 基準設定 → 試行設定 → validation比較 → 最終評価を実行します。

- **train**：分類器の重み更新にだけ使用します。
- **validation**：epochとハイパーパラメータの選択に使用します。主指標はUA、同率時はmacro F1、loss、基準設定の順です。
- **held-out test**：設定選択が完了した後、選択済みcheckpointを再読込して一度だけ評価します。

実データのパス、発話ID、個別ラベル、個別予測は表示・保存しません。Notebookに表示するのは環境の可否、集計値、学習ログ、グラフ、集計指標だけです。`demo` は合成特徴で一連の流れを確認し、`private` は利用者の非公開環境で実行します。
        """,
        "intro",
    ),
    markdown(
        """
## 2. 定義セル

Notebookには処理の実装を埋め込まず、`iemocap_downstream.notebook_pipeline` の公開関数だけを呼び出します。
        """,
        "definitions-heading",
    ),
    code(
        """
import os, sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'iemocap_downstream').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iemocap_downstream.notebook_pipeline import (
    environment_summary, evaluate_selected_experiment,
    generate_private_manifest_and_labels, load_private_feature_bundle,
    make_demo_bundle, plot_training_history, private_paths_from_env,
    run_private_feature_extraction, run_validation_experiment,
    select_best_experiment, validate_feature_bundle,
)

MODE = os.environ.get('IEMOCAP_NOTEBOOK_MODE', 'demo').lower()
if MODE not in {'demo', 'private'}:
    raise ValueError('IEMOCAP_NOTEBOOK_MODE must be demo or private')
ARTIFACT_DIR = PROJECT_ROOT / 'runs' / 'iemocap_base_downstream'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        """,
        "definitions",
    ),
    markdown("## 3. 環境確認\n\nパスそのものは表示せず、実行環境と必要ファイルの有無だけを確認します。", "environment-heading"),
    code(
        """
private_paths = private_paths_from_env() if MODE == 'private' else None
environment_summary(MODE, private_paths)
        """,
        "environment",
    ),
    markdown(
        """
## 4. データ準備・特徴抽出

`private` では、必要な場合だけ環境変数 `IEMOCAP_RUN_PREP=1` と `IEMOCAP_RUN_EXTRACT=1` を指定します。ここでは関数を呼び出すだけで、manifest生成や特徴抽出の実装詳細は扱いません。
        """,
        "data-heading",
    ),
    code(
        """
if MODE == 'demo':
    bundle = make_demo_bundle(seed=42)
    data_status = {'preparation': 'not_required', 'extraction': 'synthetic_fixed_features'}
else:
    preparation = generate_private_manifest_and_labels(private_paths) if os.environ.get('IEMOCAP_RUN_PREP') == '1' else {'status': 'using_existing_artifacts'}
    extraction = run_private_feature_extraction(private_paths, PROJECT_ROOT, 'auto') if os.environ.get('IEMOCAP_RUN_EXTRACT') == '1' else {'status': 'using_existing_features'}
    bundle = load_private_feature_bundle(private_paths.work_dir / 'train')
    data_status = {'preparation': preparation['status'], 'extraction': extraction['status']}
data_status
        """,
        "data",
    ),
    markdown("## 5. 集計値による入力検証\n\n特徴次元、有限値、総フレーム数、ラベル行数、Sessionの存在だけを表示します。", "validation-heading"),
    code("validate_feature_bundle(bundle, expected_input_dim=16 if MODE == 'demo' else 768)", "validation"),
    markdown(
        """
## 6. ハイパーパラメータ

`hp_base` は比較の基準なので変更しません。短縮動作確認は `epochs=1` のまま実行し、本実験ではこの設定セルの `epochs` を変更します。`hp_trial` は試行用ですが、公平な比較のため `seed`、`test_session`、`validation_session` は `hp_base` と同じ値を維持します。
        """,
        "hyperparameters-heading",
    ),
    code(
        """
hp_base = {
    'seed': 42,
    'device': 'auto',
    'epochs': 1,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'hidden_dim': 256,
    'dropout': 0.0,
    'patience': None,
    'test_session': 5,
    'validation_session': 1,
}
hp_base
        """,
        "hp-base",
    ),
    code(
        """
hp_trial = {
    'seed': 42,
    'device': 'auto',
    'epochs': 1,
    'batch_size': 8,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'hidden_dim': 128,
    'dropout': 0.2,
    'patience': None,
    'test_session': 5,
    'validation_session': 1,
}
hp_trial
        """,
        "hp-trial",
    ),
    markdown("## 7. 基準設定の実験\n\n次の実験セルはtrain/validationだけを使い、validation UAが最良のepochを保存します。", "base-heading"),
    code("base_result = run_validation_experiment(bundle, hp_base)", "base-run"),
    code(
        """
base_history = pd.DataFrame(base_result['history']).set_index('epoch')
display(base_history.round(4))
plot_training_history(base_result['history'], ARTIFACT_DIR / 'base_history.png')
        """,
        "base-log",
    ),
    markdown("## 8. 試行設定の実験\n\n基準設定と同じ分割・seedで、編集したハイパーパラメータを検証します。", "trial-heading"),
    code("trial_result = run_validation_experiment(bundle, hp_trial)", "trial-run"),
    code(
        """
trial_history = pd.DataFrame(trial_result['history']).set_index('epoch')
display(trial_history.round(4))
plot_training_history(trial_result['history'], ARTIFACT_DIR / 'trial_history.png')
        """,
        "trial-log",
    ),
    markdown("## 9. validation指標の比較と自動選択\n\nUA最大を優先し、同率時はmacro F1最大、validation loss最小、基準設定の順で決定します。", "selection-heading"),
    code(
        """
validation_comparison = pd.DataFrame({
    'base': base_result['validation_metrics'],
    'trial': trial_result['validation_metrics'],
}).T.loc[:, ['loss', 'wa', 'ua', 'macro_f1']]
validation_comparison.index.name = 'experiment'
display(validation_comparison.round(4))
        """,
        "comparison",
    ),
    code("selected_experiment = select_best_experiment({'base': base_result, 'trial': trial_result})", "selection"),
    code(
        """
{
    'selected_experiment': selected_experiment['name'],
    'selection_order': selected_experiment['selection_order'],
    'best_validation_metrics': selected_experiment['validation_metrics'],
}
        """,
        "selection-summary",
    ),
    markdown(
        """
## 10. 選択済みcheckpointの最終評価

ここで初めてheld-out testを評価します。選択済みcheckpointを再読込し、train / validation / testの4指標を一度だけ表示します。
        """,
        "final-heading",
    ),
    code(
        """
final_result = evaluate_selected_experiment(bundle, selected_experiment)
final_table = pd.DataFrame(final_result['split_metrics']).T.loc[['train', 'validation', 'test'], ['loss', 'wa', 'ua', 'macro_f1']]
final_table.index.name = 'split'
display(final_table.round(4))
        """,
        "final-evaluation",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
