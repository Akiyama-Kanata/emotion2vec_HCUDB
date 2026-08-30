"""Build the SER study and metadata-audit notebooks."""

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


unavailable_label_audit_cells = [
    markdown(
        """
# MSP-Podcast 破損候補の感情ラベル照合

`msp_podcast_unavailable_wav_filenames.txt` と `labels_consensus.csv` をファイル名で照合し、元の感情ラベルと研究用4クラスへの対応を確認します。このNotebookは音声ファイルを検索・読み込み・再生せず、結果ファイルも書き出しません。

既定では合成データだけを使用します。実metadataを確認するときだけ、設定セルのパスを確認して `RUN_REAL_DATA = True` に変更してください。
        """,
        "intro",
    ),
    code(
        r"""
import os
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser_pipeline.contracts import LABEL_ORDER, map_emotion

# Safety gate: Falseのままなら、下の実ファイルは一切読みません。
RUN_REAL_DATA = False
RUN_WAV_CSV_AUDIT = False

# 今回指定されたDesktop版（1,128件）を既定入力にします。
UNAVAILABLE_LIST_PATH = Path(r'C:\Users\RD004\Desktop\msp_podcast_unavailable_wav_filenames.txt')

# 直接指定を優先し、未指定なら既存のMSP_PODCAST_ROOTから解決します。
_label_csv_override = os.environ.get('MSP_LABEL_CSV_PATH')
_msp_root = os.environ.get('MSP_PODCAST_ROOT')
MSP_LABEL_CSV_PATH = (
    Path(_label_csv_override)
    if _label_csv_override
    else Path(_msp_root) / 'Labels' / 'labels_consensus.csv'
    if _msp_root
    else None
)

# WAV照合ではこのフォルダ以下のファイル名だけを再帰的に列挙します。
_audio_dir_override = os.environ.get('MSP_AUDIO_DIR')
MSP_AUDIO_DIR = (
    Path(_audio_dir_override)
    if _audio_dir_override
    else Path(_msp_root) / 'Audio'
    if _msp_root
    else None
)
        """,
        "setup",
    ),
    markdown(
        """
## 1. 読み込み・照合関数

照合キーは、パス部分を除いたファイル名の前後空白を除去し、大文字小文字を区別しない形に正規化します。表示結果には入力時の表記を残します。
        """,
        "functions-heading",
    ),
    code(
        r'''
def normalize_filename(value):
    """Return a case-insensitive basename key without touching the file itself."""
    text = str(value).strip().replace('\\', '/')
    return text.rsplit('/', 1)[-1].casefold()


def prepare_unavailable_lines(lines):
    """Normalize text-list rows and report blanks and duplicate candidate names."""
    raw_lines = [str(line) for line in lines]
    nonempty = [line.strip() for line in raw_lines if line.strip()]
    frame = pd.DataFrame({'input_filename': nonempty})
    frame['match_key'] = frame['input_filename'].map(normalize_filename)
    duplicate_mask = frame.duplicated('match_key', keep=False)
    duplicate_candidates = frame.loc[duplicate_mask, ['input_filename', 'match_key']].copy()
    unique_frame = frame.drop_duplicates('match_key', keep='first').reset_index(drop=True)
    stats = {
        'raw_line_count': len(raw_lines),
        'blank_line_count': len(raw_lines) - len(nonempty),
        'nonempty_input_rows': len(nonempty),
        'unique_input_files': len(unique_frame),
        'duplicate_input_rows': len(nonempty) - len(unique_frame),
        'non_wav_input_rows': int((~frame['match_key'].str.endswith('.wav')).sum()),
    }
    return unique_frame, stats, duplicate_candidates.reset_index(drop=True)


def read_unavailable_list(path):
    """Read only the named text file; no audio directory is inspected."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    return prepare_unavailable_lines(lines)


def prepare_label_frame(frame):
    """Validate and normalize MSP label metadata without resolving audio paths."""
    required = {'FileName', 'EmoClass'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'MSP label CSV is missing columns: {sorted(missing)}')

    labels = frame.loc[:, ['FileName', 'EmoClass']].copy()
    labels['FileName'] = labels['FileName'].astype(str).str.strip()
    labels['EmoClass'] = labels['EmoClass'].astype(str).str.strip()
    labels['match_key'] = labels['FileName'].map(normalize_filename)
    if labels['match_key'].eq('').any():
        raise ValueError('MSP label CSV contains an empty FileName')

    duplicate_mask = labels.duplicated('match_key', keep=False)
    if duplicate_mask.any():
        examples = labels.loc[duplicate_mask, 'FileName'].head(10).tolist()
        raise ValueError(f'MSP label CSV has duplicate normalized filenames: {examples}')
    return labels.rename(columns={'FileName': 'csv_filename', 'EmoClass': 'original_emotion'})


def read_msp_label_csv(path):
    """Read the MSP UTF-8/BOM-compatible metadata CSV as strings."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    return prepare_label_frame(frame)
        ''',
        "loaders",
    ),
    code(
        """
def scan_wav_directory(audio_dir):
    \"\"\"List WAV filenames recursively without opening or decoding audio.\"\"\"
    root = Path(audio_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'MSP audio directory was not found: {root}')

    rows = []
    for path in root.rglob('*'):
        if path.is_file() and path.suffix.casefold() == '.wav':
            rows.append(
                {
                    'wav_filename': path.name,
                    'wav_relative_path': path.relative_to(root).as_posix(),
                    'match_key': normalize_filename(path.name),
                }
            )
    wavs = pd.DataFrame(rows, columns=['wav_filename', 'wav_relative_path', 'match_key'])
    duplicate_mask = wavs.duplicated('match_key', keep=False)
    if duplicate_mask.any():
        examples = wavs.loc[duplicate_mask, 'wav_relative_path'].head(10).tolist()
        raise ValueError(f'MSP audio directory has duplicate normalized WAV filenames: {examples}')
    return wavs.sort_values('wav_relative_path').reset_index(drop=True)


def audit_wav_csv(wavs, labels):
    \"\"\"Compare actual WAV filenames and label rows in both directions.\"\"\"
    comparison = labels.merge(
        wavs,
        how='outer',
        on='match_key',
        validate='one_to_one',
        indicator=True,
    )
    comparison['comparison_status'] = comparison['_merge'].map(
        {
            'both': 'matched',
            'left_only': 'label_without_wav',
            'right_only': 'wav_without_label',
        }
    ).astype(str)
    comparison['original_emotion'] = comparison['original_emotion'].fillna('')
    label_present = comparison['_merge'].ne('right_only')
    mapping = pd.DataFrame(
        [mapping_fields(row.original_emotion, present) for row, present in zip(
            comparison.itertuples(index=False), label_present
        )]
    )
    comparison = pd.concat([comparison.reset_index(drop=True), mapping], axis=1)
    comparison = comparison.drop(columns=['_merge', 'match_key'])

    available = comparison.loc[comparison['comparison_status'].eq('matched')].copy()
    missing_wavs = comparison.loc[
        comparison['comparison_status'].eq('label_without_wav'),
        ['csv_filename', 'original_emotion', 'mapped_emotion', 'mapping_status'],
    ].reset_index(drop=True)
    unlabeled_wavs = comparison.loc[
        comparison['comparison_status'].eq('wav_without_label'),
        ['wav_filename', 'wav_relative_path', 'comparison_status'],
    ].reset_index(drop=True)

    summary = pd.DataFrame(
        [
            {
                'wav_files_found': len(wavs),
                'label_csv_rows': len(labels),
                'matched_wav_and_label': len(available),
                'label_rows_without_wav': len(missing_wavs),
                'wav_files_without_label': len(unlabeled_wavs),
                'available_primary_4_files': int(available['included_in_primary_4'].eq(True).sum()),
            }
        ]
    )
    raw_counts = (
        available['original_emotion']
        .replace('', '<empty_label>')
        .value_counts(dropna=False)
        .rename_axis('original_emotion')
        .reset_index(name='count')
    )
    mapped_counts = (
        available.loc[available['included_in_primary_4'].eq(True), 'mapped_emotion']
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
        .rename_axis('mapped_emotion')
        .reset_index(name='count')
    )
    return summary, raw_counts, mapped_counts, comparison, missing_wavs, unlabeled_wavs
        """,
        "wav-audit-functions",
    ),
    code(
        """
def mapping_fields(original_emotion, matched):
    if not matched:
        return {
            'mapped_emotion': None,
            'included_in_primary_4': None,
            'mapping_status': 'metadata_not_found',
            'mapping_version': None,
        }
    if not original_emotion:
        return {
            'mapped_emotion': None,
            'included_in_primary_4': False,
            'mapping_status': 'empty_label',
            'mapping_version': None,
        }
    try:
        decision = map_emotion('msp_podcast', original_emotion)
    except ValueError:
        return {
            'mapped_emotion': None,
            'included_in_primary_4': False,
            'mapping_status': 'unknown_label',
            'mapping_version': None,
        }
    return {
        'mapped_emotion': decision.mapped_emotion,
        'included_in_primary_4': decision.included,
        'mapping_status': 'included_primary_4' if decision.included else 'not_in_primary_4',
        'mapping_version': decision.mapping_version,
    }


def audit_unavailable_labels(candidates, input_stats, labels):
    details = candidates.merge(labels, how='left', on='match_key', validate='one_to_one', indicator=True)
    details['metadata_match'] = details['_merge'].eq('both')
    details['original_emotion'] = details['original_emotion'].fillna('')
    mapping = pd.DataFrame(
        [mapping_fields(row.original_emotion, row.metadata_match) for row in details.itertuples(index=False)]
    )
    details = pd.concat([details.reset_index(drop=True), mapping], axis=1)
    details = details.drop(columns=['_merge', 'match_key'])

    summary_values = dict(input_stats)
    summary_values.update(
        {
            'label_csv_rows': len(labels),
            'metadata_matched_files': int(details['metadata_match'].sum()),
            'metadata_unmatched_files': int((~details['metadata_match']).sum()),
            'primary_4_files': int(details['included_in_primary_4'].eq(True).sum()),
            'outside_primary_4_files': int(details['mapping_status'].eq('not_in_primary_4').sum()),
            'empty_label_files': int(details['mapping_status'].eq('empty_label').sum()),
            'unknown_label_files': int(details['mapping_status'].eq('unknown_label').sum()),
        }
    )
    summary = pd.DataFrame([summary_values])

    raw_outcomes = details['original_emotion'].where(details['metadata_match'], '<metadata_not_found>')
    raw_outcomes = raw_outcomes.replace('', '<empty_label>')
    raw_label_counts = raw_outcomes.value_counts(dropna=False).rename_axis('original_emotion').reset_index(name='count')

    def mapped_outcome(row):
        if row.mapping_status == 'included_primary_4':
            return row.mapped_emotion
        return f'<{row.mapping_status}>'

    outcome_order = list(LABEL_ORDER) + [
        '<not_in_primary_4>', '<empty_label>', '<unknown_label>', '<metadata_not_found>'
    ]
    mapped_outcomes = details.apply(mapped_outcome, axis=1)
    mapped_label_counts = (
        mapped_outcomes.value_counts()
        .reindex(outcome_order, fill_value=0)
        .rename_axis('mapped_outcome')
        .reset_index(name='count')
    )
    unmatched = details.loc[
        ~details['metadata_match'], ['input_filename', 'mapping_status']
    ].reset_index(drop=True)
    return summary, raw_label_counts, mapped_label_counts, details, unmatched
        """,
        "audit-functions",
    ),
    markdown(
        """
## 2. 入力選択

`RUN_REAL_DATA = False` では、照合境界を確認する小さな合成例を使います。実metadataモードでも、音声ファイルにはアクセスしません。
        """,
        "input-heading",
    ),
    code(
        """
if RUN_REAL_DATA:
    if MSP_LABEL_CSV_PATH is None:
        raise ValueError(
            'Set MSP_LABEL_CSV_PATH directly, or set MSP_LABEL_CSV_PATH/MSP_PODCAST_ROOT in the environment.'
        )
    candidates, input_stats, duplicate_candidates = read_unavailable_list(UNAVAILABLE_LIST_PATH)
    labels = read_msp_label_csv(MSP_LABEL_CSV_PATH)
    data_mode = 'real_metadata'
else:
    synthetic_lines = [
        ' MSP-PODCAST_DEMO_A.wav ',
        'MSP-PODCAST_DEMO_H.wav',
        'MSP-PODCAST_DEMO_S.wav',
        'MSP-PODCAST_DEMO_D.wav',
        'MSP-PODCAST_DEMO_EXCLUDED.wav',
        'MSP-PODCAST_DEMO_UNKNOWN.wav',
        'MSP-PODCAST_DEMO_EMPTY.wav',
        'MSP-PODCAST_DEMO_DUP.wav',
        'msp-podcast_demo_dup.WAV',
        'MSP-PODCAST_DEMO_MISSING.wav',
        '',
    ]
    synthetic_labels = pd.DataFrame(
        {
            'FileName': [
                'MSP-PODCAST_DEMO_A.wav', 'MSP-PODCAST_DEMO_H.wav',
                'MSP-PODCAST_DEMO_S.wav', 'MSP-PODCAST_DEMO_D.wav',
                'MSP-PODCAST_DEMO_EXCLUDED.wav', 'MSP-PODCAST_DEMO_UNKNOWN.wav',
                'MSP-PODCAST_DEMO_EMPTY.wav', 'MSP-PODCAST_DEMO_DUP.wav',
            ],
            'EmoClass': ['A', 'H', 'S', 'D', 'C', 'Z', '', 'H'],
        }
    )
    candidates, input_stats, duplicate_candidates = prepare_unavailable_lines(synthetic_lines)
    labels = prepare_label_frame(synthetic_labels)
    data_mode = 'synthetic_only'

summary, raw_label_counts, mapped_label_counts, details, unmatched = audit_unavailable_labels(
    candidates, input_stats, labels
)
{'data_mode': data_mode, 'run_real_data': RUN_REAL_DATA}
        """,
        "select-input",
    ),
    markdown("## 3. 入力・照合サマリー", "summary-heading"),
    code(
        """
display(summary)
if not duplicate_candidates.empty:
    print('重複として1件にまとめた候補:')
    display(duplicate_candidates)
        """,
        "summary",
    ),
    markdown("## 4. 元ラベルと4クラス対応の件数", "counts-heading"),
    code(
        """
print('CSV元ラベル別件数:')
display(raw_label_counts)
print('4クラス対応・対象外・未一致別件数:')
display(mapped_label_counts)
        """,
        "counts",
    ),
    markdown("## 5. ファイル単位の照合結果", "details-heading"),
    code(
        """
detail_columns = [
    'input_filename', 'csv_filename', 'metadata_match', 'original_emotion',
    'mapped_emotion', 'included_in_primary_4', 'mapping_status', 'mapping_version',
]
display(details.loc[:, detail_columns])
        """,
        "details",
    ),
    markdown("## 6. CSVに存在しなかった候補", "unmatched-heading"),
    code(
        """
if unmatched.empty:
    print('すべての候補がラベルCSVに存在しました。')
else:
    display(unmatched)
        """,
        "unmatched",
    ),
    markdown(
        """
## 7. 実際に存在するWAVとラベルCSVの照合

設定セルで `MSP_AUDIO_DIR` と `MSP_LABEL_CSV_PATH` を指定し、`RUN_WAV_CSV_AUDIT = True` にした場合だけ実行します。WAVはファイル名を列挙するだけで、音声内容を開いたりデコードしたりしません。
        """,
        "wav-audit-heading",
    ),
    code(
        """
if RUN_WAV_CSV_AUDIT:
    if MSP_AUDIO_DIR is None:
        raise ValueError('Set MSP_AUDIO_DIR directly, or set MSP_AUDIO_DIR/MSP_PODCAST_ROOT in the environment.')
    if MSP_LABEL_CSV_PATH is None:
        raise ValueError(
            'Set MSP_LABEL_CSV_PATH directly, or set MSP_LABEL_CSV_PATH/MSP_PODCAST_ROOT in the environment.'
        )

    wav_files = scan_wav_directory(MSP_AUDIO_DIR)
    wav_labels = read_msp_label_csv(MSP_LABEL_CSV_PATH)
    (
        wav_csv_summary,
        available_raw_counts,
        available_mapped_counts,
        wav_csv_comparison,
        label_rows_without_wav,
        wav_files_without_label,
    ) = audit_wav_csv(wav_files, wav_labels)

    print('WAVとラベルCSVの照合サマリー:')
    display(wav_csv_summary)
    print('実在WAVの元ラベル別件数:')
    display(available_raw_counts)
    print('実在WAVの研究用4クラス別件数:')
    display(available_mapped_counts)
    print('CSVにはあるがWAVが見つからない行:')
    display(label_rows_without_wav)
    print('WAVはあるがCSVにラベルがないファイル:')
    display(wav_files_without_label)
else:
    print('WAVフォルダ照合は無効です。設定セルのRUN_WAV_CSV_AUDITをTrueにすると実行します。')
        """,
        "wav-audit-run",
    ),
]


for filename, cells in (
    ("01_extract_emotion2vec_features.ipynb", feature_cells),
    ("02_train_and_evaluate_decoder.ipynb", decoder_cells),
    ("msp_unavailable_label_audit.ipynb", unavailable_label_audit_cells),
):
    output = NOTEBOOK_DIR / filename
    output.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(output)
