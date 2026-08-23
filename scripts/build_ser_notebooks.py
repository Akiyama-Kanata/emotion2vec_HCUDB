"""Build the separated SER feature and decoder study notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


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


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


feature_cells = [
    markdown(
        """
# 01 — 4クラスSER特徴cacheの準備

MSP-Podcast R1.10、HCUDB1、IEMOCAPのmetadata監査、version付きmapping/split、manifest、固定encoder出力のshard cacheを確認します。正式な全件処理は明示フラグを変更するまで開始しません。
        """,
        "intro",
    ),
    code(
        """
import os, sys, subprocess
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser_pipeline.manifest import audit_dataset
from ser_pipeline.notebook_api import (
    demo_cache_summary, environment_summary, extraction_command_preview,
    mapping_summary, one_item_feature_benchmark, split_summary,
)

RUN_FULL_EXTRACTION = False
ARTIFACT_DIR = PROJECT_ROOT / 'runs' / 'ser_feature_preflight'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        """,
        "setup",
    ),
    markdown("## 1. 実行環境と固定契約", "environment-heading"),
    code("environment_summary()", "environment"),
    code("pd.DataFrame(mapping_summary())", "mapping"),
    code("split_summary()", "splits"),
    markdown(
        """
## 2. ローカルmetadata監査

環境変数でrootが設定されたデータセットだけを読み取り専用で監査します。未設定時は空の表になり、demoは継続します。
        """,
        "audit-heading",
    ),
    code(
        """
configured_roots = {
    'msp_podcast': os.environ.get('MSP_PODCAST_ROOT'),
    'hcudb1': os.environ.get('HCUDB1_ROOT'),
    'iemocap': os.environ.get('IEMOCAP_ROOT'),
}
audit_rows = [audit_dataset(name, root) for name, root in configured_roots.items() if root]
pd.DataFrame(audit_rows)
        """,
        "audit",
    ),
    markdown("## 3. 合成manifest/cacheの境界確認", "demo-heading"),
    code("demo_status = demo_cache_summary(ARTIFACT_DIR / 'demo')\ndemo_status", "demo-cache"),
    markdown("## 4. 1件preflightと容量表示", "benchmark-heading"),
    code("one_item_feature_benchmark(feature_dim=768, seconds=1.0)", "benchmark"),
    markdown(
        """
## 5. 全件コマンドのpreview

次のセルはコマンド文字列を表示するだけです。正式実行時もlayerは `final` 固定です。
        """,
        "preview-heading",
    ),
    code("full_command = extraction_command_preview()\nfull_command", "preview"),
    code(
        """
if RUN_FULL_EXTRACTION:
    raise RuntimeError('Placeholders must be replaced and the formal preflight gate approved before execution')
else:
    {'status': 'preview_only', 'command': full_command}
        """,
        "full-gate",
    ),
]


decoder_cells = [
    markdown(
        """
# 02 — MSP-Podcast→HCUDB 4クラスdecoder学習・評価

検証済みframe cacheとmanifestだけを入力にし、MSP-Podcast学習、HCUDB継続学習、IEMOCAP外部testを同じ4クラス出力で比較します。正式実行は既定で無効です。
        """,
        "intro",
    ),
    code(
        """
import os, sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser_pipeline.notebook_api import environment_summary, run_demo_transfer_study
from ser_pipeline.study import DatasetArtifacts, run_transfer_study
from ser_pipeline.training import TrainingConfig

RUN_DEMO = True
RUN_FORMAL_STUDY = False
STUDY_SEEDS = (42, 43, 44)
ARTIFACT_DIR = PROJECT_ROOT / 'runs' / 'ser_decoder_study'
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        """,
        "setup",
    ),
    markdown("## 1. cache-only実行環境", "environment-heading"),
    code("environment_summary()", "environment"),
    markdown("## 2. 合成cacheでの短時間E2E", "demo-heading"),
    code(
        """
demo_seeds = (42,)
demo_summary = run_demo_transfer_study(ARTIFACT_DIR / 'demo', seeds=demo_seeds, epochs=1) if RUN_DEMO else None
{'status': 'completed' if demo_summary else 'skipped', 'seeds': list(demo_seeds)}
        """,
        "demo-run",
    ),
    code(
        """
if demo_summary:
    demo_run = demo_summary['runs'][0]
    rows = []
    for stage_name in ('before', 'after'):
        for dataset_name, payload in demo_run[stage_name].items():
            metrics = payload['result']['metrics_4class']
            rows.append({
                'stage': stage_name,
                'dataset': dataset_name,
                'accuracy_percent': metrics['accuracy'] * 100,
                'uar_percent': metrics['uar'] * 100,
                'macro_f1_percent': metrics['macro_f1'] * 100,
            })
    demo_table = pd.DataFrame(rows)
else:
    demo_table = pd.DataFrame()
demo_table
        """,
        "demo-results",
    ),
    markdown(
        """
## 3. 正式3-seed実行ゲート

必要な6つのmanifest/cacheパスを環境変数で与え、全preflight項目の承認後にフラグを変更します。指標ファイルは0–1、ここでの表示だけ百分率です。
        """,
        "formal-heading",
    ),
    code(
        """
if RUN_FORMAL_STUDY:
    names = ('msp_podcast', 'hcudb1', 'iemocap')
    formal_artifacts = {
        name: DatasetArtifacts(
            manifest_path=Path(os.environ[f'SER_{name.upper()}_MANIFEST']),
            cache_root=Path(os.environ[f'SER_{name.upper()}_CACHE']),
        )
        for name in names
    }
    formal_summary = run_transfer_study(
        formal_artifacts,
        ARTIFACT_DIR / 'formal',
        seeds=STUDY_SEEDS,
        base_config=TrainingConfig(seed=42, device='auto'),
    )
else:
    formal_summary = {'status': 'disabled_by_default', 'seeds': list(STUDY_SEEDS)}
formal_summary
        """,
        "formal-gate",
    ),
]


for filename, cells in (
    ("01_extract_emotion2vec_features.ipynb", feature_cells),
    ("02_train_and_evaluate_decoder.ipynb", decoder_cells),
):
    output = NOTEBOOK_DIR / filename
    output.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)
