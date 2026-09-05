"""Build or compare the SER study and metadata-audit notebooks."""

from __future__ import annotations

import argparse
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
            "kernelspec": {
                "display_name": "emotion2vec-py310",
                "language": "python",
                "name": "emotion2vec-py310",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


feature_cells = [
    markdown(
        """
# 01 — MSP-Podcast 4クラス特徴cacheの作成

MSP-Podcast R1.10の欠損監査、完全一致重複監査、承認契約、strict manifest作成、実音声1件のCPU benchmark、容量+20%判定、emotion2vec特徴抽出、cache検証を上から順に行います。

実データを読む処理はすべて既定で無効です。設定セルでパスを指定し、各段階の結果を確認してから、対応するフラグを1つずつ`True`にしてください。IEMOCAPは今回の一括研究経路には含めません。
        """,
        "intro",
    ),
    code(
        """
import json, os, sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser_pipeline.cache import validate_cache
from ser_pipeline.duplicates import (
    generate_msp_audio_duplicate_exclusion_contract,
    load_msp_audio_duplicate_audit,
    load_msp_audio_duplicate_exclusion_contract,
)
from ser_pipeline.exclusions import load_msp_missing_audio_exclusion_contract
from ser_pipeline.features import Emotion2vecEncoder, extract_feature_cache
from ser_pipeline.manifest import (
    audit_dataset, build_manifest, generate_msp_audio_duplicate_audit,
    generate_msp_missing_audio_exclusion_contract,
    load_manifest, validate_manifest,
)
from ser_pipeline.notebook_api import environment_summary, mapping_summary, split_summary
from ser_pipeline.preflight import (
    benchmark_audio_extraction, disk_capacity_gate,
    estimate_full_extraction, save_benchmark,
)

STUDY_DATASETS = ('msp_podcast', 'hcudb1')

# 実行フラグ: 最初はすべてFalseのまま、上から1段階ずつ有効化します。
RUN_MSP_AUDIT = False
RUN_MSP_GENERATE_EXCLUSION_CONTRACT = False
RUN_MSP_VERIFY_EXCLUSION_CONTRACT = False
RUN_MSP_DUPLICATE_AUDIT = False
RUN_MSP_GENERATE_DUPLICATE_EXCLUSION_CONTRACT = False
RUN_MSP_VERIFY_DUPLICATE_EXCLUSION_CONTRACT = False
RUN_MSP_BUILD_MANIFEST = False
RUN_MSP_VALIDATE_MANIFEST = False
RUN_MSP_BENCHMARK = False
RUN_MSP_CAPACITY_GATE = False
RUN_FULL_EXTRACTION = False
RUN_VALIDATE_CACHE = False

# 前段の結果をユーザーが確認した後だけTrueにします。
CONFIRM_MANIFEST_VALIDATED = False
CONFIRM_BENCHMARK_AND_CAPACITY = False

# 環境変数を使わない場合は、ここへWSLから見えるPathを直接指定できます。
MSP_ROOT = Path(os.environ['MSP_PODCAST_ROOT']) if os.environ.get('MSP_PODCAST_ROOT') else None
BENCHMARK_AUDIO = Path(os.environ['MSP_BENCHMARK_AUDIO']) if os.environ.get('MSP_BENCHMARK_AUDIO') else None

USER_DIR = PROJECT_ROOT / 'upstream'
CHECKPOINT_PATH = PROJECT_ROOT / 'artifacts' / 'checkpoints' / 'emotion2vec_base.pt'
MANIFEST_PATH = PROJECT_ROOT / 'runs' / 'ser_manifests' / 'msp_podcast_4class_v1.jsonl'
EXCLUSION_CONTRACT_PATH = PROJECT_ROOT / 'runs' / 'ser_manifests' / 'msp_missing_audio_exclusions_v1.json'
DUPLICATE_AUDIT_PATH = PROJECT_ROOT / 'runs' / 'ser_manifests' / 'msp_audio_duplicate_audit_v1.json'
DUPLICATE_CANDIDATES_CSV_PATH = PROJECT_ROOT / 'runs' / 'ser_manifests' / 'msp_audio_duplicate_candidates_v1.csv'
DUPLICATE_EXCLUSION_CONTRACT_PATH = PROJECT_ROOT / 'runs' / 'ser_manifests' / 'msp_audio_duplicate_exclusions_v1.json'
CACHE_ROOT = PROJECT_ROOT / 'runs' / 'ser_feature_cache' / 'msp_podcast_base_final_v1'
REPORT_DIR = PROJECT_ROOT / 'runs' / 'ser_feature_preflight' / 'msp_podcast'
BENCHMARK_PATH = REPORT_DIR / 'one_audio_cpu_benchmark.json'
CAPACITY_PATH = REPORT_DIR / 'capacity_estimate.json'


def require_path(value, label, *, kind):
    if value is None:
        raise ValueError(f'Set {label} in the configuration cell before execution')
    path = Path(value)
    valid = path.is_file() if kind == 'file' else path.is_dir()
    if not valid:
        raise FileNotFoundError(f'{label} was not found: {path}')
    return path


def persist_report(report, path):
    save_benchmark(report, path)
    return report
        """,
        "setup",
    ),
    markdown("## 1. 設定・実行環境・固定契約", "environment-heading"),
    code(
        """
{
    'msp_root': str(MSP_ROOT) if MSP_ROOT else None,
    'benchmark_audio': str(BENCHMARK_AUDIO) if BENCHMARK_AUDIO else None,
    'user_dir': str(USER_DIR),
    'checkpoint': str(CHECKPOINT_PATH),
    'manifest': str(MANIFEST_PATH),
    'exclusion_contract': str(EXCLUSION_CONTRACT_PATH),
    'duplicate_audit': str(DUPLICATE_AUDIT_PATH),
    'duplicate_candidates_csv': str(DUPLICATE_CANDIDATES_CSV_PATH),
    'duplicate_exclusion_contract': str(DUPLICATE_EXCLUSION_CONTRACT_PATH),
    'cache_root': str(CACHE_ROOT),
    'device': 'cpu',
    'run_flags': {
        'audit': RUN_MSP_AUDIT,
        'generate_exclusion_contract': RUN_MSP_GENERATE_EXCLUSION_CONTRACT,
        'verify_exclusion_contract': RUN_MSP_VERIFY_EXCLUSION_CONTRACT,
        'duplicate_audit': RUN_MSP_DUPLICATE_AUDIT,
        'generate_duplicate_exclusion_contract': RUN_MSP_GENERATE_DUPLICATE_EXCLUSION_CONTRACT,
        'verify_duplicate_exclusion_contract': RUN_MSP_VERIFY_DUPLICATE_EXCLUSION_CONTRACT,
        'build_manifest': RUN_MSP_BUILD_MANIFEST,
        'validate_manifest': RUN_MSP_VALIDATE_MANIFEST,
        'benchmark': RUN_MSP_BENCHMARK,
        'capacity_gate': RUN_MSP_CAPACITY_GATE,
        'full_extraction': RUN_FULL_EXTRACTION,
        'validate_cache': RUN_VALIDATE_CACHE,
    },
}
        """,
        "configuration",
    ),
    code("environment_summary()", "environment"),
    code(
        "mapping_rows = [row for row in mapping_summary() if row['dataset'] in STUDY_DATASETS]\n"
        "pd.DataFrame(mapping_rows)",
        "mapping",
    ),
    code(
        "all_split_contracts = split_summary()\n"
        "{name: all_split_contracts[name] for name in STUDY_DATASETS}",
        "splits",
    ),
    markdown(
        """
## 2. MSP-Podcast metadata・音声inventory監査

`RUN_MSP_AUDIT = True`にした場合だけ実データを読みます。0バイトWAVは入手不能な欠損音声として扱います。結果には、現在不足している対象音声の元ラベル、4クラス変換後ラベル、公式split別件数も含まれます。`missing_eligible_audio == 1128`、`zero_byte_audio_files_treated_as_missing == 254`、固定内訳、`unregistered_audio_files == 0`、現行契約の対象25,985件を確認してから次へ進みます。
        """,
        "audit-heading",
    ),
    code(
        """
if RUN_MSP_AUDIT:
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    audit_report = audit_dataset('msp_podcast', msp_root)
    persist_report(audit_report, REPORT_DIR / 'audit.json')
else:
    audit_report = {'status': 'disabled_by_default'}
audit_report
        """,
        "audit",
    ),
    markdown(
        """
### 2.1 現在不足している対象音声の感情・split内訳

添付リストではなく、metadata上の4クラス対象と現在存在するWAVを照合し、不足している対象音声だけを集計します。`missing_count`の合計が`missing_eligible_audio`と一致することを確認してください。
        """,
        "missing-label-heading",
    ),
    code(
        """
if RUN_MSP_AUDIT:
    label_pairs = (
        ('A', 'anger'),
        ('H', 'happy'),
        ('S', 'sadness'),
        ('D', 'disgust'),
    )
    missing_label_summary = pd.DataFrame([
        {
            'original_label': original_label,
            'mapped_label': mapped_label,
            'eligible_total': audit_report['eligible_mapped_label_counts'].get(mapped_label, 0),
            'available_count': audit_report['available_eligible_mapped_label_counts'].get(mapped_label, 0),
            'missing_count': audit_report['missing_eligible_original_label_counts'].get(original_label, 0),
        }
        for original_label, mapped_label in label_pairs
    ])
    missing_split_summary = pd.DataFrame([
        {
            'source_split': source_split,
            'missing_count': audit_report['missing_eligible_source_split_counts'].get(source_split, 0),
        }
        for source_split in ('Train', 'Development', 'Test1')
    ])
    print('不足対象音声の感情ラベル内訳:')
    display(missing_label_summary)
    print('不足対象音声の公式split内訳:')
    display(missing_split_summary)
else:
    print('監査は無効です。RUN_MSP_AUDIT = Trueで監査セルから実行してください。')
        """,
        "missing-label-summary",
    ),
    markdown(
        """
### 2.2 欠損ファイルの分布

欠損1,128件を、実ファイルなし／0バイト、感情ラベル、公式split、感情×splitで集計します。`missing_share_pct`は全欠損に占める構成比、`missing_rate_pct`は各区分の対象総数に対する欠損率です。
        """,
        "missing-distribution-heading",
    ),
    code(
        """
if RUN_MSP_AUDIT:
    missing_total = int(audit_report['missing_eligible_audio'])
    missing_kind_names = {
        'absent': 'ファイルなし',
        'zero_byte': '0バイト',
    }
    missing_kind_summary = pd.DataFrame([
        {
            'missing_type': missing_kind_names[kind],
            'missing_count': audit_report['missing_eligible_kind_counts'].get(kind, 0),
            'missing_share_pct': round(
                100 * audit_report['missing_eligible_kind_counts'].get(kind, 0) / missing_total,
                2,
            ) if missing_total else None,
        }
        for kind in ('absent', 'zero_byte')
    ])

    distribution_label_pairs = (
        ('A', 'anger'),
        ('H', 'happy'),
        ('S', 'sadness'),
        ('D', 'disgust'),
    )
    missing_label_distribution = pd.DataFrame([
        {
            'original_label': original_label,
            'mapped_label': mapped_label,
            'eligible_total': eligible_total,
            'missing_count': missing_count,
            'missing_share_pct': round(100 * missing_count / missing_total, 2) if missing_total else None,
            'missing_rate_pct': round(100 * missing_count / eligible_total, 2) if eligible_total else None,
        }
        for original_label, mapped_label in distribution_label_pairs
        for eligible_total, missing_count in [(
            audit_report['eligible_mapped_label_counts'].get(mapped_label, 0),
            audit_report['missing_eligible_original_label_counts'].get(original_label, 0),
        )]
    ])

    missing_split_distribution = pd.DataFrame([
        {
            'source_split': source_split,
            'eligible_total': eligible_total,
            'available_count': eligible_total - missing_count,
            'missing_count': missing_count,
            'missing_share_pct': round(100 * missing_count / missing_total, 2) if missing_total else None,
            'missing_rate_pct': round(100 * missing_count / eligible_total, 2) if eligible_total else None,
        }
        for source_split in ('Train', 'Development', 'Test1')
        for eligible_total, missing_count in [(
            audit_report['eligible_source_split_counts'].get(source_split, 0),
            audit_report['missing_eligible_source_split_counts'].get(source_split, 0),
        )]
    ])

    missing_label_split_cross = pd.DataFrame.from_dict(
        audit_report['missing_eligible_original_by_source_split_counts'],
        orient='index',
    ).reindex(
        index=('Train', 'Development', 'Test1'),
        columns=('A', 'H', 'S', 'D'),
        fill_value=0,
    ).fillna(0).astype(int)
    missing_label_split_cross['Total'] = missing_label_split_cross.sum(axis=1)
    missing_label_split_cross.loc['Total'] = missing_label_split_cross.sum(axis=0)

    print('欠損種別:')
    display(missing_kind_summary)
    print('感情ラベル別の欠損構成比・欠損率:')
    display(missing_label_distribution)
    print('公式split別の欠損構成比・欠損率:')
    display(missing_split_distribution)
    print('感情ラベル × 公式split（欠損件数）:')
    display(missing_label_split_cross)
else:
    print('監査は無効です。RUN_MSP_AUDIT = Trueで監査セルから実行してください。')
        """,
        "missing-distribution",
    ),
    markdown(
        """
## 3. 除外候補生成

`RUN_MSP_GENERATE_EXCLUSION_CONTRACT = True`にした場合だけ、添付リストを参照せず、metadata上の4クラス対象と現在のWAV inventoryから、0バイトWAVを含む不足行を再計算します。1,128件・固定内訳に一致しない場合はJSONを書きません。
        """,
        "exclusion-generation-heading",
    ),
    code(
        """
if RUN_MSP_GENERATE_EXCLUSION_CONTRACT:
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    exclusion_generation_report = generate_msp_missing_audio_exclusion_contract(
        msp_root,
        EXCLUSION_CONTRACT_PATH,
    )
    persist_report(exclusion_generation_report, REPORT_DIR / 'exclusion_contract_generation.json')
else:
    exclusion_generation_report = {'status': 'disabled_by_default'}
exclusion_generation_report
        """,
        "exclusion-generation",
    ),
    markdown(
        """
## 4. 除外契約の件数・内訳・SHA確認

`RUN_MSP_VERIFY_EXCLUSION_CONTRACT = True`にすると、1,128件、元ラベル`A 416 / H 576 / S 107 / D 29`、公式split`Train 579 / Development 219 / Test1 330`、ファイル名順、重複なし、正規化SHA-256を検証します。
        """,
        "exclusion-verification-heading",
    ),
    code(
        """
if RUN_MSP_VERIFY_EXCLUSION_CONTRACT:
    contract_path = require_path(EXCLUSION_CONTRACT_PATH, 'EXCLUSION_CONTRACT_PATH', kind='file')
    _, exclusion_verification_report = load_msp_missing_audio_exclusion_contract(contract_path)
    persist_report(exclusion_verification_report, REPORT_DIR / 'exclusion_contract_verification.json')
else:
    exclusion_verification_report = {'status': 'disabled_by_default'}
exclusion_verification_report
        """,
        "exclusion-verification",
    ),
    markdown(
        """
## 5. 承認SHA設定

上の検証結果を確認後、承認する`normalized_sha256`を64桁の小文字16進文字列として設定します。未設定のままstrict manifest作成を有効化すると必ず拒否します。
        """,
        "exclusion-approval-heading",
    ),
    code(
        """
APPROVED_MSP_EXCLUSION_SHA256 = None
# 例: APPROVED_MSP_EXCLUSION_SHA256 = '64桁の検証済みSHA-256'
        """,
        "exclusion-approval",
    ),
    markdown(
        """
## 6. MSP-Podcast完全一致重複監査

欠損音声除外契約のSHAを承認した後、`RUN_MSP_DUPLICATE_AUDIT = True`にした場合だけ、利用可能な4クラス・既知話者・`Train / Development / Test1`音声を読みます。全対象のファイルバイトSHA-256を計算し、同じサンプルレート・チャンネル数・フレーム数の候補だけを`little-endian float32`へデコードして、形状込みの波形SHA-256を計算します。リサンプリング、許容誤差、類似度閾値は使いません。

このセルは監査JSONと候補CSVだけを生成し、manifestや生WAVを変更しません。`summary`で同一split群とcross-split群を分けて確認し、候補CSVでは話者・ラベル不一致フラグも確認してください。
        """,
        "duplicate-audit-heading",
    ),
    code(
        """
if RUN_MSP_DUPLICATE_AUDIT:
    if APPROVED_MSP_EXCLUSION_SHA256 is None:
        raise RuntimeError('Approve the missing-audio exclusion SHA-256 before duplicate audit')
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    require_path(EXCLUSION_CONTRACT_PATH, 'EXCLUSION_CONTRACT_PATH', kind='file')
    duplicate_audit_report = generate_msp_audio_duplicate_audit(
        msp_root,
        DUPLICATE_AUDIT_PATH,
        DUPLICATE_CANDIDATES_CSV_PATH,
        approved_missing_audio_exclusion_contract=EXCLUSION_CONTRACT_PATH,
        expected_missing_audio_exclusion_sha256=APPROVED_MSP_EXCLUSION_SHA256,
    )
    persist_report(duplicate_audit_report, REPORT_DIR / 'duplicate_audit_generation.json')
    duplicate_candidates = pd.read_csv(DUPLICATE_CANDIDATES_CSV_PATH)
    print('完全一致重複監査サマリー:')
    display(pd.Series(duplicate_audit_report['summary'], name='count').to_frame())
    print('重複候補（候補が0件の場合は空の表です）:')
    display(duplicate_candidates)
else:
    duplicate_audit_report = {'status': 'disabled_by_default'}
duplicate_audit_report
        """,
        "duplicate-audit",
    ),
    markdown(
        """
## 7. 重複除外IDの手動判断

候補CSVをグループごとに確認し、除外すると判断した発話IDだけを明示します。ここへ書いただけではまだ除外されません。監査候補にないID、同じIDの重複指定、cross-split群を複数splitに残す判断は次の契約生成で拒否されます。同一split内の候補は、承認しないまま残すこともできます。
        """,
        "duplicate-decision-heading",
    ),
    code(
        """
MSP_APPROVED_DUPLICATE_EXCLUDE_IDS = [
    # 'MSP-PODCAST_XXXX',
]
        """,
        "duplicate-decision",
    ),
    markdown(
        """
## 8. 重複除外契約の生成・検証

`RUN_MSP_GENERATE_DUPLICATE_EXCLUSION_CONTRACT = True`にすると、監査候補と明示IDだけから`msp_audio_duplicate_exclusions_v1`を生成します。続けて`RUN_MSP_VERIFY_DUPLICATE_EXCLUSION_CONTRACT = True`にすると、監査SHA、除外レコード、理由、除外後件数、cross-split解消条件、正規化SHA-256を再検証します。
        """,
        "duplicate-contract-heading",
    ),
    code(
        """
if RUN_MSP_GENERATE_DUPLICATE_EXCLUSION_CONTRACT:
    require_path(DUPLICATE_AUDIT_PATH, 'DUPLICATE_AUDIT_PATH', kind='file')
    duplicate_contract_generation_report = generate_msp_audio_duplicate_exclusion_contract(
        DUPLICATE_AUDIT_PATH,
        MSP_APPROVED_DUPLICATE_EXCLUDE_IDS,
        DUPLICATE_EXCLUSION_CONTRACT_PATH,
    )
    persist_report(
        duplicate_contract_generation_report,
        REPORT_DIR / 'duplicate_exclusion_contract_generation.json',
    )
else:
    duplicate_contract_generation_report = {'status': 'disabled_by_default'}
duplicate_contract_generation_report
        """,
        "duplicate-contract-generation",
    ),
    code(
        """
if RUN_MSP_VERIFY_DUPLICATE_EXCLUSION_CONTRACT:
    audit_path = require_path(DUPLICATE_AUDIT_PATH, 'DUPLICATE_AUDIT_PATH', kind='file')
    duplicate_contract_path = require_path(
        DUPLICATE_EXCLUSION_CONTRACT_PATH,
        'DUPLICATE_EXCLUSION_CONTRACT_PATH',
        kind='file',
    )
    duplicate_audit_payload, duplicate_audit_verification = load_msp_audio_duplicate_audit(audit_path)
    _, duplicate_contract_verification = load_msp_audio_duplicate_exclusion_contract(
        duplicate_contract_path,
        duplicate_audit_payload,
    )
    duplicate_contract_verification_report = {
        'audit': duplicate_audit_verification,
        'exclusion_contract': duplicate_contract_verification,
    }
    persist_report(
        duplicate_contract_verification_report,
        REPORT_DIR / 'duplicate_exclusion_contract_verification.json',
    )
else:
    duplicate_contract_verification_report = {'status': 'disabled_by_default'}
duplicate_contract_verification_report
        """,
        "duplicate-contract-verification",
    ),
    markdown(
        """
## 9. 重複除外契約SHAの承認

上の検証結果と手動判断を確認後、重複除外契約の`normalized_sha256`を設定します。除外0件でも、cross-split候補が残っていないことを証明する監査連結済み契約として承認が必要です。
        """,
        "duplicate-approval-heading",
    ),
    code(
        """
APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256 = None
# 例: APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256 = '64桁の検証済みSHA-256'
        """,
        "duplicate-approval",
    ),
    markdown(
        """
## 10. strict manifest作成

`RUN_MSP_BUILD_MANIFEST = True`にすると、欠損契約・重複監査・重複除外契約と両方の承認SHAを照合します。現在の全対象音声SHAが監査JSONと一致しなければ「監査が古い」として停止し、承認済みIDだけを`msp_audio_duplicate_exclusion_approved_v1`で`included: false`にします。最終件数は欠損除外後件数から重複契約件数を引いた契約値で検証されます。
        """,
        "manifest-build-heading",
    ),
    code(
        """
if RUN_MSP_BUILD_MANIFEST:
    if APPROVED_MSP_EXCLUSION_SHA256 is None:
        raise RuntimeError('Set APPROVED_MSP_EXCLUSION_SHA256 after reviewing the exclusion contract')
    if APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256 is None:
        raise RuntimeError('Set APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256 after reviewing duplicate exclusions')
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    require_path(EXCLUSION_CONTRACT_PATH, 'EXCLUSION_CONTRACT_PATH', kind='file')
    require_path(DUPLICATE_AUDIT_PATH, 'DUPLICATE_AUDIT_PATH', kind='file')
    require_path(DUPLICATE_EXCLUSION_CONTRACT_PATH, 'DUPLICATE_EXCLUSION_CONTRACT_PATH', kind='file')
    manifest_build_report = build_manifest(
        'msp_podcast',
        msp_root,
        MANIFEST_PATH,
        strict=True,
        inspect_excluded_audio=True,
        approved_exclusion_contract=EXCLUSION_CONTRACT_PATH,
        expected_exclusion_sha256=APPROVED_MSP_EXCLUSION_SHA256,
        duplicate_audit=DUPLICATE_AUDIT_PATH,
        approved_duplicate_exclusion_contract=DUPLICATE_EXCLUSION_CONTRACT_PATH,
        expected_duplicate_exclusion_sha256=APPROVED_MSP_DUPLICATE_EXCLUSION_SHA256,
    )
    persist_report(manifest_build_report, REPORT_DIR / 'manifest_build.json')
else:
    manifest_build_report = {'status': 'disabled_by_default'}
manifest_build_report
        """,
        "manifest-build",
    ),
    markdown(
        """
## 11. manifestと実音声の完全検証

`RUN_MSP_VALIDATE_MANIFEST = True`にすると、included音声のmetadataとSHA-256を再計算します。結果が`status: ok`で、`audio.verified_audio`と`included`が一致したことを確認してください。
        """,
        "manifest-validation-heading",
    ),
    code(
        """
if RUN_MSP_VALIDATE_MANIFEST:
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    require_path(MANIFEST_PATH, 'MANIFEST_PATH', kind='file')
    manifest_validation_report = validate_manifest(MANIFEST_PATH, audio_root=msp_root)
    persist_report(manifest_validation_report, REPORT_DIR / 'manifest_validation.json')
else:
    manifest_validation_report = {'status': 'disabled_by_default'}
manifest_validation_report
        """,
        "manifest-validation",
    ),
    markdown(
        """
## 12. 実音声1件のCPU benchmark

manifestで`included: true`のWAVを`BENCHMARK_AUDIO`へ指定します。manifest検証結果を確認後、`CONFIRM_MANIFEST_VALIDATED = True`と`RUN_MSP_BENCHMARK = True`にします。
        """,
        "benchmark-heading",
    ),
    code(
        """
if RUN_MSP_BENCHMARK:
    if not CONFIRM_MANIFEST_VALIDATED:
        raise RuntimeError('Confirm the complete MSP manifest validation first')
    benchmark_audio = require_path(BENCHMARK_AUDIO, 'BENCHMARK_AUDIO', kind='file')
    require_path(USER_DIR, 'USER_DIR', kind='directory')
    require_path(CHECKPOINT_PATH, 'CHECKPOINT_PATH', kind='file')
    benchmark_report = benchmark_audio_extraction(
        benchmark_audio,
        USER_DIR,
        CHECKPOINT_PATH,
        device='cpu',
    )
    persist_report(benchmark_report, BENCHMARK_PATH)
else:
    benchmark_report = {'status': 'disabled_by_default'}
benchmark_report
        """,
        "benchmark",
    ),
    markdown(
        """
## 13. 全件所要時間・容量+20%ゲート

`RUN_MSP_CAPACITY_GATE = True`にすると、manifestの対象総時間と1件benchmarkから全件見積りを作ります。`capacity.passes`が`True`でなければ全件抽出へ進みません。
        """,
        "capacity-heading",
    ),
    code(
        """
if RUN_MSP_CAPACITY_GATE:
    if not CONFIRM_MANIFEST_VALIDATED:
        raise RuntimeError('Confirm the complete MSP manifest validation first')
    require_path(MANIFEST_PATH, 'MANIFEST_PATH', kind='file')
    require_path(BENCHMARK_PATH, 'BENCHMARK_PATH', kind='file')
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    included_rows = [row for row in load_manifest(MANIFEST_PATH) if row['included']]
    total_duration_seconds = sum(float(row['duration_seconds']) for row in included_rows)
    saved_benchmark = json.loads(BENCHMARK_PATH.read_text(encoding='utf-8'))
    estimate = estimate_full_extraction(
        total_duration_seconds,
        saved_benchmark,
        storage_margin=1.2,
    )
    capacity = disk_capacity_gate(CACHE_ROOT, estimate['required_bytes_with_margin'])
    capacity_report = {
        'included_utterances': len(included_rows),
        'estimate': estimate,
        'capacity': capacity,
    }
    persist_report(capacity_report, CAPACITY_PATH)
    if not capacity['passes']:
        raise RuntimeError('Capacity gate failed; do not start full extraction')
else:
    capacity_report = {'status': 'disabled_by_default'}
capacity_report
        """,
        "capacity-gate",
    ),
    markdown(
        """
## 14. MSP-Podcast全件特徴抽出

manifest検証、benchmark、容量判定を確認した後だけ、2つの確認フラグと`RUN_FULL_EXTRACTION`を`True`にします。deviceはCPU、layerは`final`固定です。中断した場合はcacheを削除せず、同じ設定でこのセルを再実行すると完成済みshardを再利用します。
        """,
        "extraction-heading",
    ),
    code(
        """
if RUN_FULL_EXTRACTION:
    if not CONFIRM_MANIFEST_VALIDATED:
        raise RuntimeError('Confirm the complete MSP manifest validation first')
    if not CONFIRM_BENCHMARK_AND_CAPACITY:
        raise RuntimeError('Confirm the CPU benchmark and +20% capacity gate first')
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    require_path(MANIFEST_PATH, 'MANIFEST_PATH', kind='file')
    require_path(USER_DIR, 'USER_DIR', kind='directory')
    require_path(CHECKPOINT_PATH, 'CHECKPOINT_PATH', kind='file')
    saved_capacity = json.loads(require_path(CAPACITY_PATH, 'CAPACITY_PATH', kind='file').read_text(encoding='utf-8'))
    if not saved_capacity['capacity']['passes']:
        raise RuntimeError('Saved capacity gate does not pass')
    saved_benchmark = json.loads(require_path(BENCHMARK_PATH, 'BENCHMARK_PATH', kind='file').read_text(encoding='utf-8'))
    encoder = Emotion2vecEncoder(
        USER_DIR,
        CHECKPOINT_PATH,
        layer='final',
        device='cpu',
    )
    if encoder.info.checkpoint_sha256 != saved_benchmark['encoder_checkpoint_sha256']:
        raise RuntimeError('Benchmark and extraction checkpoint SHA-256 differ')
    extraction_report = extract_feature_cache(
        MANIFEST_PATH,
        msp_root,
        CACHE_ROOT,
        encoder,
        layer='final',
        max_shard_frames=65536,
        expected_dim=768,
    )
    persist_report(extraction_report, REPORT_DIR / 'extraction_result.json')
else:
    extraction_report = {'status': 'disabled_by_default'}
extraction_report
        """,
        "full-extraction-gate",
    ),
    markdown(
        """
## 15. cache最終検証

抽出完了後に`RUN_VALIDATE_CACHE = True`として実行します。manifest対象件数、768次元、有限float32、shard/index hash、全splitの`_SUCCESS`、cache完了フラグ、checkpoint SHA-256を検証します。
        """,
        "cache-validation-heading",
    ),
    code(
        """
if RUN_VALIDATE_CACHE:
    require_path(MANIFEST_PATH, 'MANIFEST_PATH', kind='file')
    require_path(CACHE_ROOT, 'CACHE_ROOT', kind='directory')
    saved_benchmark = json.loads(require_path(BENCHMARK_PATH, 'BENCHMARK_PATH', kind='file').read_text(encoding='utf-8'))
    cache_validation = validate_cache(
        CACHE_ROOT,
        MANIFEST_PATH,
        expected_signature={
            'feature_dim': 768,
            'feature_layer': 'final_after_encoder_norm',
            'dtype': 'float32',
        },
    )
    expected_count = sum(bool(row['included']) for row in load_manifest(MANIFEST_PATH))
    if cache_validation['utterances'] != expected_count:
        raise RuntimeError('Cache utterance count does not match the manifest')
    cache_meta = json.loads((CACHE_ROOT / 'cache_meta.json').read_text(encoding='utf-8'))
    if cache_meta['encoder_checkpoint_sha256'] != saved_benchmark['encoder_checkpoint_sha256']:
        raise RuntimeError('Benchmark and cache checkpoint SHA-256 differ')
    partial_files = [str(path) for path in CACHE_ROOT.rglob('*.partial')]
    if partial_files:
        raise RuntimeError(f'Partial cache files remain: {partial_files[:5]}')
    cache_validation_report = {
        **cache_validation,
        'checkpoint_sha256': cache_meta['encoder_checkpoint_sha256'],
        'partial_files': partial_files,
    }
    persist_report(cache_validation_report, REPORT_DIR / 'cache_validation.json')
else:
    cache_validation_report = {'status': 'disabled_by_default'}
cache_validation_report
        """,
        "cache-validation",
    ),
]


decoder_cells = [
    markdown(
        """
# 02 — MSP-Podcast→HCUDB 4クラスdecoder学習・評価

検証済みframe cacheとmanifestだけを入力にし、MSP-Podcast学習とHCUDB継続学習を行います。通常の学習ではtrain・validationを扱い、設定確定後のtest評価は「7」で実行します。IEMOCAPは今回の一括研究経路には含めません。疎通・正式学習・最終評価は別の出力先を使い、実行フラグはすべて既定で無効です。
        """,
        "intro",
    ),
    markdown("## 1. cache-only実行環境", "environment-heading"),
    code(
        """
import os, sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
importlib.invalidate_caches()
for _ser_module_name in ('ser_pipeline.checkpoints', 'ser_pipeline.training', 'ser_pipeline.study', 'ser_pipeline.notebook_api'):
    importlib.reload(importlib.import_module(_ser_module_name))

from ser_pipeline.notebook_api import environment_summary, display_training_history, load_saved_summary
from ser_pipeline.study import (
    DatasetArtifacts, prepare_study_stores, require_formal_epochs,
    run_transfer_study, summarize_study, FinalEvaluationTarget, run_final_evaluations,
)
from ser_pipeline.training import TrainingConfig, TrainingMonitoringConfig

STUDY_DATASETS = ('msp_podcast', 'hcudb1')


def load_study_artifacts():
    artifacts = {}
    missing = []
    for name in STUDY_DATASETS:
        manifest_key = f'SER_{name.upper()}_MANIFEST'
        cache_key = f'SER_{name.upper()}_CACHE'
        if not os.environ.get(manifest_key):
            missing.append(manifest_key)
        if not os.environ.get(cache_key):
            missing.append(cache_key)
        exclusion_contract_path = None
        duplicate_audit_path = None
        duplicate_exclusion_contract_path = None
        if name == 'msp_podcast':
            exclusion_key = 'SER_MSP_PODCAST_EXCLUSION_CONTRACT'
            duplicate_audit_key = 'SER_MSP_PODCAST_DUPLICATE_AUDIT'
            duplicate_exclusion_key = 'SER_MSP_PODCAST_DUPLICATE_EXCLUSION_CONTRACT'
            for provenance_key in (exclusion_key, duplicate_audit_key, duplicate_exclusion_key):
                if not os.environ.get(provenance_key):
                    missing.append(provenance_key)
            if exclusion_key not in missing:
                exclusion_contract_path = Path(os.environ[exclusion_key])
            if duplicate_audit_key not in missing:
                duplicate_audit_path = Path(os.environ[duplicate_audit_key])
            if duplicate_exclusion_key not in missing:
                duplicate_exclusion_contract_path = Path(os.environ[duplicate_exclusion_key])
        if manifest_key not in missing and cache_key not in missing:
            artifacts[name] = DatasetArtifacts(
                manifest_path=Path(os.environ[manifest_key]),
                cache_root=Path(os.environ[cache_key]),
                exclusion_contract_path=exclusion_contract_path,
                duplicate_audit_path=duplicate_audit_path,
                duplicate_exclusion_contract_path=duplicate_exclusion_contract_path,
            )
    if missing:
        raise ValueError(f'Missing study artifact environment variables: {missing}')
    return artifacts


def validate_execution_gates():
    if not CONFIRM_CACHE_VALIDATION:
        raise RuntimeError('Confirm both MSP/HCUDB caches were completely validated')
    if not CONFIRM_BENCHMARK_AND_CAPACITY:
        raise RuntimeError('Confirm the one-item CPU benchmark and the +20% capacity gate')
    artifacts = load_study_artifacts()
    stores = prepare_study_stores(artifacts)
    cache_validation = {name: store.validation_report for name, store in stores.items()}
    return artifacts, cache_validation, stores
        """,
        "setup",
    ),
    code("environment_summary()", "environment"),
    markdown("## 2. 実行設定の確認", "configuration-heading"),
    code(
        """
RUN_REAL_SMOKE = False
RUN_FORMAL_SEED_42 = False
RUN_FORMAL_SEEDS_43_44 = False
FORMAL_EPOCHS = 10
CONFIRM_CACHE_VALIDATION = False
CONFIRM_BENCHMARK_AND_CAPACITY = False
CONFIRM_SMOKE_COMPLETED = False
CONFIRM_SEED_42_ARTIFACTS = False
TRAINING_MONITORING_CONFIG = TrainingMonitoringConfig(max_epoch_samples=2000, sampling_seed=0)

# 新しい学習には既存結果と別の出力先を指定します。
ARTIFACT_DIR = PROJECT_ROOT / 'runs' / 'ser_decoder_study'
TRAINING_OUTPUT_DIR = PROJECT_ROOT / 'runs' / 'ser_decoder_score_loss_20260903'

# Falseのときに読む完了summary。過去結果を再表示しても元ファイルは更新しません。
SAVED_SMOKE_SUMMARY = ARTIFACT_DIR / 'smoke' / 'study_summary.json'
SAVED_SEED_42_SUMMARY = PROJECT_ROOT / 'runs' / 'ser_decoder_timing_check_20260903' / 'formal' / 'initial-seed-42' / 'study_summary.json'
SAVED_FOLLOWUP_SUMMARY = ARTIFACT_DIR / 'formal' / 'followup-seeds-43-44' / 'study_summary.json'
        """, "training-settings",
    ),
    code(
        """
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "ser_pipeline").is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent

MANIFEST_DIR = PROJECT_ROOT / "runs" / "ser_manifests"
CACHE_DIR = PROJECT_ROOT / "runs" / "ser_feature_cache"

artifact_paths = {
    "SER_MSP_PODCAST_MANIFEST": MANIFEST_DIR / "msp_podcast_4class_v1.jsonl",
    "SER_MSP_PODCAST_CACHE": CACHE_DIR / "msp_podcast_base_final_v1",
    "SER_MSP_PODCAST_EXCLUSION_CONTRACT": MANIFEST_DIR / "msp_missing_audio_exclusions_v1.json",
    "SER_MSP_PODCAST_DUPLICATE_AUDIT": MANIFEST_DIR / "msp_audio_duplicate_audit_v1.json",
    "SER_MSP_PODCAST_DUPLICATE_EXCLUSION_CONTRACT": MANIFEST_DIR / "msp_audio_duplicate_exclusions_v1.json",
    "SER_HCUDB1_MANIFEST": MANIFEST_DIR / "hcudb1_4class_v1.jsonl",
    "SER_HCUDB1_CACHE": CACHE_DIR / "hcudb1_base_final_v1",
}

for name, path in artifact_paths.items():
    os.environ[name] = str(path)

{name: {"path": str(path), "exists": path.exists()} for name, path in artifact_paths.items()}
        """,
        "study-artifact-paths",
    ),
    code(
        """
{
    'datasets': STUDY_DATASETS,
    'device': 'cpu',
    'run_real_smoke': RUN_REAL_SMOKE,
    'run_formal_seed_42': RUN_FORMAL_SEED_42,
    'run_formal_seeds_43_44': RUN_FORMAL_SEEDS_43_44,
    'formal_epochs': FORMAL_EPOCHS,
    'training_config': TrainingConfig(device='cpu', epochs=FORMAL_EPOCHS),
    'monitoring_config': TRAINING_MONITORING_CONFIG,
    'new_output': TRAINING_OUTPUT_DIR,
    'seed_42_output_exists': (TRAINING_OUTPUT_DIR / 'formal' / 'initial-seed-42').exists(),
    'saved_summaries': [SAVED_SMOKE_SUMMARY, SAVED_SEED_42_SUMMARY, SAVED_FOLLOWUP_SUMMARY],
}
        """,
        "configuration",
    ),
    markdown(
        """
## 3. 実データ1 epoch疎通（正式集計外）

seed 42でMSP親学習→HCUDB継続学習を1 epochずつ行い、train監視値・validationを確認します。MSPの監視はクラス比を保つ固定2,000件、2,000件以下のHCUDBは全trainです。出力は`smoke/`に隔離され、正式結果には混ぜません。Falseのままなら指定summaryだけを読み、scoreと折りたたみlossを再表示します。
        """,
        "smoke-heading",
    ),
    code(
        """
if RUN_REAL_SMOKE:
    smoke_artifacts, smoke_cache_validation, smoke_stores = validate_execution_gates()
    smoke_summary = run_transfer_study(
        smoke_artifacts,
        TRAINING_OUTPUT_DIR / 'smoke',
        seeds=(42,),
        base_config=TrainingConfig(seed=42, device='cpu', epochs=1),
        monitoring_config=TRAINING_MONITORING_CONFIG,
        stores=smoke_stores,
    )
else:
    smoke_summary = load_saved_summary(SAVED_SMOKE_SUMMARY)
for run in smoke_summary.get('runs', []):
    for stage in ('parent', 'child'):
        training = run[stage]
        display_training_history(training, save_plots=RUN_REAL_SMOKE)
summarize_study(smoke_summary)
        """,
        "smoke-gate",
    ),
    markdown(
        """
## 4. 正式seed 42実行ゲート

1 epoch疎通の時間と履歴を確認し、`FORMAL_EPOCHS = 10`、seed 42で実行します。学習フラグがFalseなら指定summaryを再表示します。新しい学習の保存先は既存結果と分けます。

各epochで **train（固定2,000件・参考）** とvalidation全3,600件のUAR・macro F1、accuracy（参考）、共通のbest epochを表示します。固定集合は発話IDの安定SHA-256順位で一度だけ選び、全seed・全設定で共通です。監視はcheckpoint選択や正式結果に使いません。
このセル内に **UAR（主指標）→ Macro F1 → Accuracy（参考）** の曲線と、直下に **lossを確認：split間の比較用／最適化に使用したloss** の折りたたみを表示します。青はtrain監視、橙はvalidation、灰色破線はvalidation UAR → macro F1 → lossで選んだ共通のbest epochです。
学習終了後、best状態をtrain全15,524件で1回だけ評価し、正式なtrain結果を曲線とは別表に表示します。HCUDBは毎epochの監視自体が全1,500件なので重複評価を省略します。未記録は欠測のまま表示します。
        """,
        "formal-seed-42-heading",
    ),
    code(
        """
if RUN_FORMAL_SEED_42:
    if not CONFIRM_SMOKE_COMPLETED:
        raise RuntimeError('Confirm the real-data 1 epoch smoke run before formal seed 42')
    formal_epochs = require_formal_epochs(FORMAL_EPOCHS)
    formal_artifacts, formal_cache_validation, formal_stores = validate_execution_gates()
    formal_seed_42_summary = run_transfer_study(
        formal_artifacts,
        TRAINING_OUTPUT_DIR / 'formal' / 'initial-seed-42',
        seeds=(42,),
        base_config=TrainingConfig(seed=42, device='cpu', epochs=formal_epochs),
        monitoring_config=TRAINING_MONITORING_CONFIG,
        stores=formal_stores,
    )
else:
    formal_seed_42_summary = load_saved_summary(SAVED_SEED_42_SUMMARY)
for run in formal_seed_42_summary.get('runs', []):
    for stage in ('parent', 'child'):
        training = run[stage]
        display_training_history(training, save_plots=RUN_FORMAL_SEED_42)
summarize_study(formal_seed_42_summary)
        """,
        "formal-seed-42-gate",
    ),
    markdown(
        """
## 5. 正式seed 43・44実行ゲート

seed 42のcheckpoint、train・validationの集合情報、cache ID、設定値を確認した後だけ実行します。過去summaryではmanifest情報を確認します。出力はseed 42の正式出力と分けて保存します。

seed 42と同様に、各epochの固定train監視集合・validation全件のUAR・macro F1・accuracyを表示し、bestモデルのtrain全件結果は別表にします。
同じセル内にseed・データセットごとのscore曲線と折りたたみlossを表示します。Falseのままなら指定summaryだけを読みます。
        """,
        "formal-followup-heading",
    ),
    code(
        """
if RUN_FORMAL_SEEDS_43_44:
    if not CONFIRM_SEED_42_ARTIFACTS:
        raise RuntimeError('Confirm the formal seed 42 artifacts before seeds 43 and 44')
    formal_epochs = require_formal_epochs(FORMAL_EPOCHS)
    followup_artifacts, followup_cache_validation, followup_stores = validate_execution_gates()
    formal_followup_summary = run_transfer_study(
        followup_artifacts,
        TRAINING_OUTPUT_DIR / 'formal' / 'followup-seeds-43-44',
        seeds=(43, 44),
        base_config=TrainingConfig(seed=43, device='cpu', epochs=formal_epochs),
        monitoring_config=TRAINING_MONITORING_CONFIG,
        stores=followup_stores,
    )
else:
    formal_followup_summary = load_saved_summary(SAVED_FOLLOWUP_SUMMARY)
for run in formal_followup_summary.get('runs', []):
    for stage in ('parent', 'child'):
        training = run[stage]
        display_training_history(training, save_plots=RUN_FORMAL_SEEDS_43_44)
summarize_study(formal_followup_summary)
        """,
        "formal-followup-gate",
    ),
]


decoder_cells += [
    markdown(
        """
## 6. MSP単体：クラス重み付き損失の比較

この節の **6.1 → 6.2 → 6.3 → 6.4** だけを実行します。6.1で更新済みの学習コードを読み直します。
この節は独立しており、上のsetupや「3〜5」の学習セルを実行する必要はありません。
MSPの重みなし結果を読み込み、同じseedの初期値から重みありモデルを10 epoch学習します。
バッチサイズ8、学習率0.001、Dropout 0、データ分割と発話順序を維持します。
重みは **trainの総件数 / (4 × trainのクラス別件数)**。validationのUARを主指標に比較します。
validationのlossは従来の重みなし計算です。重みあり/なしのtrain lossは同じ尺度として比較しません。
HCUDBの学習とtest評価はこの節では実行しません。
        """, "msp-loss-heading",
    ),
    markdown("### 6.1 比較設定", "msp-loss-settings-heading"),
    code(
        """
import json, os, sys
from pathlib import Path
import pandas as pd
from IPython.display import display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
importlib.invalidate_caches()
for _msp_module_name in ('ser_pipeline.checkpoints', 'ser_pipeline.training', 'ser_pipeline.study', 'ser_pipeline.notebook_api'):
    importlib.reload(importlib.import_module(_msp_module_name))

from ser_pipeline.cache import ShardedFeatureStore
from ser_pipeline.notebook_api import display_training_history, load_saved_summary
from ser_pipeline.study import DatasetArtifacts, load_msp_comparison_baselines, run_msp_loss_comparison
from ser_pipeline.training import TrainingConfig, TrainingMonitoringConfig, training_loss_config

RUN_MSP_WEIGHTED_TRAINING = False  # 学習を開始するとき True にする
MSP_COMPARISON_SEEDS = (42,)  # 次に (43, 44) で同じ比較を行う
MSP_COMPARISON_CONFIG = TrainingConfig(device='cpu', epochs=10, batch_size=8)
MSP_MONITORING_CONFIG = TrainingMonitoringConfig(max_epoch_samples=2000, sampling_seed=0)
MSP_COMPARISON_OUTPUT = PROJECT_ROOT / 'runs' / 'msp_class_weight_comparison' / (
    'seeds-' + '-'.join(str(seed) for seed in MSP_COMPARISON_SEEDS)
)
MSP_TRAINING_OUTPUT = PROJECT_ROOT / 'runs' / 'msp_class_weight_score_loss_20260903' / (
    'seeds-' + '-'.join(str(seed) for seed in MSP_COMPARISON_SEEDS)
)
MSP_SAVED_COMPARISON_SUMMARY = MSP_COMPARISON_OUTPUT / 'comparison_summary.json'

manifest_dir = PROJECT_ROOT / 'runs' / 'ser_manifests'
def msp_input(env_key, default):
    return Path(os.environ.get(env_key) or default)

MSP_COMPARISON_ARTIFACT = DatasetArtifacts(
    manifest_path=msp_input('SER_MSP_PODCAST_MANIFEST', manifest_dir / 'msp_podcast_4class_v1.jsonl'),
    cache_root=msp_input('SER_MSP_PODCAST_CACHE', PROJECT_ROOT / 'runs' / 'ser_feature_cache' / 'msp_podcast_base_final_v1'),
    exclusion_contract_path=msp_input('SER_MSP_PODCAST_EXCLUSION_CONTRACT', manifest_dir / 'msp_missing_audio_exclusions_v1.json'),
    duplicate_audit_path=msp_input('SER_MSP_PODCAST_DUPLICATE_AUDIT', manifest_dir / 'msp_audio_duplicate_audit_v1.json'),
    duplicate_exclusion_contract_path=msp_input('SER_MSP_PODCAST_DUPLICATE_EXCLUSION_CONTRACT', manifest_dir / 'msp_audio_duplicate_exclusions_v1.json'),
)
MSP_BASELINE_SUMMARIES = [
    PROJECT_ROOT / 'runs' / 'ser_decoder_timing_check_20260903' / 'formal' / 'initial-seed-42' / 'study_summary.json',
    PROJECT_ROOT / 'runs' / 'ser_decoder_study' / 'formal' / 'followup-seeds-43-44' / 'study_summary.json',
]
print('比較seed:', MSP_COMPARISON_SEEDS, '学習を実行:', RUN_MSP_WEIGHTED_TRAINING)
print('新しい学習の保存先:', MSP_TRAINING_OUTPUT)
print('再表示するsummary:', MSP_SAVED_COMPARISON_SUMMARY)
        """, "msp-loss-settings",
    ),
    markdown(
        """
### 6.2 キャッシュ・比較元・クラス重みの確認

学習フラグがTrueのとき、学習済み比較元と設定・train/validation集合・manifest・cache・checkpointを照合します。初回のキャッシュ完全検証には数分かかります。Falseなら検証を起動せず6.3へ進みます。
同じカーネルでこのセルを再実行した場合は、入力に変更がなければ検証済みstoreを再利用します。
        """, "msp-loss-prepare-heading",
    ),
    code(
        """
if RUN_MSP_WEIGHTED_TRAINING:
    if MSP_TRAINING_OUTPUT.exists() and any(MSP_TRAINING_OUTPUT.iterdir()):
        raise ValueError('保存先に結果があります。6.1で新しい学習の保存先を設定してください。')
    if 'msp_comparison_store' not in globals():
        print('MSPキャッシュの完全検証を開始します。', flush=True)
        msp_comparison_store = ShardedFeatureStore(
            MSP_COMPARISON_ARTIFACT.cache_root, MSP_COMPARISON_ARTIFACT.manifest_path,
        )
    else:
        msp_comparison_store.require_paths(MSP_COMPARISON_ARTIFACT.cache_root, MSP_COMPARISON_ARTIFACT.manifest_path)
        msp_comparison_store.ensure_validated()
    msp_comparison_baselines = load_msp_comparison_baselines(
        MSP_BASELINE_SUMMARIES, msp_comparison_store, MSP_COMPARISON_CONFIG, MSP_COMPARISON_SEEDS,
    )
    msp_comparison_loss = training_loss_config(msp_comparison_store, 'msp_podcast', 'balanced')
    display(pd.DataFrame({
        '感情': msp_comparison_loss['label_order'],
        'train件数': msp_comparison_loss['train_class_counts'],
        '重み': msp_comparison_loss['class_weights'],
    }).round(4))
    print('比較元の照合完了。seed:', list(msp_comparison_baselines))
else:
    print('学習フラグはFalseです。6.3で保存済みsummaryを再表示します。')
        """, "msp-loss-prepare",
    ),
    markdown(
        """
### 6.3 MSPの重みあり学習を実行

各epochでtrain（固定2,000件・参考）・validation全件のUAR・macro F1、accuracy（参考）、共通のbest epochを表示します。同じセル内にscore曲線と折りたたみloss、別表にbestモデルのtrain全件結果を表示します。train監視評価時間・最後のtrain全件評価時間は分けて確認できます。
Falseなら`MSP_SAVED_COMPARISON_SUMMARY`を読み、同じ表示を行います。cache検証・学習・モデル評価は起動しません。過去の未記録train score/lossは欠測として扱います。新しく学習した場合に限り`.scores.png`と`.losses.png`を保存します。
        """, "msp-loss-run-heading",
    ),
    code(
        """
if RUN_MSP_WEIGHTED_TRAINING:
    msp_loss_comparison = run_msp_loss_comparison(
        MSP_COMPARISON_ARTIFACT, MSP_TRAINING_OUTPUT, MSP_BASELINE_SUMMARIES,
        seeds=MSP_COMPARISON_SEEDS, base_config=MSP_COMPARISON_CONFIG,
        monitoring_config=MSP_MONITORING_CONFIG, store=msp_comparison_store,
    )
else:
    msp_loss_comparison = load_saved_summary(MSP_SAVED_COMPARISON_SUMMARY)
for run in msp_loss_comparison.get('runs', []):
    display_training_history(run['baseline']['training'])
    display_training_history(run['weighted'], save_plots=RUN_MSP_WEIGHTED_TRAINING)
        """, "msp-loss-run",
    ),
    markdown(
        """
### 6.4 validation結果を比較

`none`は保存済みの重みなし結果、`balanced`は今回の重みあり結果です。
差分は **重みあり − 重みなし**。UAR・macro F1・WA（正解率）・再現率は大きいほど良い指標です。曲線とlossは6.3で確認し、この節ではvalidation比較表を表示します。
seed 42で動作を確認したら、6.1のseedを`(43, 44)`に変えて比較し、3 seedでの傾向を確認します。
        """, "msp-loss-results-heading",
    ),
    code(
        """
msp_comparison_summary_path = (MSP_TRAINING_OUTPUT / 'comparison_summary.json') if RUN_MSP_WEIGHTED_TRAINING else MSP_SAVED_COMPARISON_SUMMARY
if msp_comparison_summary_path.is_file():
    msp_loss_comparison = json.loads(msp_comparison_summary_path.read_text(encoding='utf-8'))
    print('validation結果:')
    display(pd.DataFrame(msp_loss_comparison['rows']).drop(columns=['loss'], errors='ignore').round(4))
    print('validation差分（重みあり − 重みなし）:')
    display(pd.DataFrame([
        {'seed': run['seed'], **run['validation_deltas']} for run in msp_loss_comparison['runs']
    ]).drop(columns=['loss'], errors='ignore').round(4))
    print('完了seed:', msp_loss_comparison['completed_seeds'], '/', msp_loss_comparison['requested_seeds'])
    print('比較実行時間（分）:', round(msp_loss_comparison['seconds'] / 60, 2))
    print('保存先:', msp_comparison_summary_path)
else:
    print('比較結果はまだありません。6.3の学習を完了してください。')
        """, "msp-loss-results",
    ),
]


decoder_cells += [
    markdown(
        """
## 7. 設定確定後の最終test評価

合成データ検証を確認し、train・validationによる設定選択が完了してから実行します。
`RUN_FINAL_TEST = False`のまま対象を編集し、`CONFIRM_FINAL_SETTINGS = True`で設定確定を明示します。
`FINAL_TARGETS`には表示名、保存済みbest checkpointのパス、保存済み来歴から確認したSHA-256、評価datasetを指定します。未指定なら実行できません。
転移実験では、各seedのMSP親・HCUDB子をそれぞれMSP・HCUDB testで評価する4対象を明示します。MSP単体なら確定したモデルとMSP testだけを指定します。
`FINAL_ARTIFACTS`には対象datasetの`DatasetArtifacts`を指定します。MSP単体の場合、6.1の`MSP_COMPARISON_ARTIFACT`を使えます。転移実験の場合は1・2の設定確認後に`load_study_artifacts()`を使えます。
最終評価の計画を`final_evaluation_plan.json`に先に保存し、結果は別の`final_evaluation_summary.json`へ保存します。checkpointの欠損やSHA不一致はエラーにします。結果からbestを選び直す処理はありません。
        """, "final-test-heading",
    ),
    code(
        """
from pathlib import Path
from ser_pipeline.study import FinalEvaluationTarget, run_final_evaluations

RUN_FINAL_TEST = False
CONFIRM_FINAL_SETTINGS = False
FINAL_DEVICE = 'cpu'
FINAL_BATCH_SIZE = 8
FINAL_ARTIFACTS = {}  # 例: {'msp_podcast': MSP_COMPARISON_ARTIFACT}
FINAL_TARGETS = [
    # FinalEvaluationTarget(
    #     name='MSP seed 42 確定モデル',
    #     checkpoint_path=Path('保存済みbest checkpointのパス'),
    #     expected_sha256='保存済み来歴で確認した64桁のSHA-256',
    #     dataset='msp_podcast',
    # ),
]
FINAL_OUTPUT_DIR = PROJECT_ROOT / 'runs' / 'ser_final_test_20260903'
{
    'run_final_test': RUN_FINAL_TEST, 'settings_confirmed': CONFIRM_FINAL_SETTINGS,
    'device': FINAL_DEVICE, 'batch_size': FINAL_BATCH_SIZE,
    'targets': FINAL_TARGETS, 'output': FINAL_OUTPUT_DIR,
}
        """, "final-test-settings",
    ),
    code(
        """
if RUN_FINAL_TEST:
    if not CONFIRM_FINAL_SETTINGS:
        raise RuntimeError('train・validationで設定を確定してから最終test評価を実行してください。')
    if not FINAL_TARGETS:
        raise ValueError('保存済みbest checkpointと期待SHA-256を明示的に指定してください。')
    final_test_summary = run_final_evaluations(
        FINAL_ARTIFACTS, FINAL_TARGETS, FINAL_OUTPUT_DIR,
        device=FINAL_DEVICE, batch_size=FINAL_BATCH_SIZE,
    )
    for evaluation in final_test_summary['evaluations']:
        print(evaluation['target']['name'], evaluation['target']['dataset'], {
            key: round(evaluation['result']['metrics_4class'][key], 4)
            for key in ('uar', 'macro_f1', 'wa')
        })
    print('最終test評価の保存先:', final_test_summary['summary_path'])
else:
    print('最終test評価は無効です。設定確定後に対象を明示して実行してください。')
        """, "final-test-gate",
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


NOTEBOOKS = {
    "01_extract_emotion2vec_features.ipynb": feature_cells,
    "02_train_and_evaluate_decoder.ipynb": decoder_cells,
    "msp_unavailable_label_audit.ipynb": unavailable_label_audit_cells,
}
DEFAULT_NOTEBOOKS = (
    "01_extract_emotion2vec_features.ipynb",
    "02_train_and_evaluate_decoder.ipynb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare parsed JSON content without writing any notebook",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=NOTEBOOK_DIR,
        help="directory to write or inspect (defaults to the tracked notebooks directory)",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        choices=tuple(NOTEBOOKS),
        help="notebook to process; repeat for multiple files (defaults to study notebooks only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = tuple(args.notebook) if args.notebook else DEFAULT_NOTEBOOKS
    target_dir = args.output_dir.resolve()
    generated = {name: notebook(NOTEBOOKS[name]) for name in selected}
    if args.check:
        mismatches = []
        for name, expected in generated.items():
            path = target_dir / name
            try:
                actual = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                mismatches.append(f"{name}: {exc}")
                continue
            if actual != expected:
                mismatches.append(f"{name}: JSON content differs")
        if mismatches:
            raise SystemExit("Notebook check failed:\n" + "\n".join(mismatches))
        print(f"Notebook JSON content matches ({len(selected)} files); no files written")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in generated.items():
        output = target_dir / name
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
