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
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


feature_cells = [
    markdown(
        """
# 01 — MSP-Podcast 4クラス特徴cacheの作成

MSP-Podcast R1.10の監査、strict manifest作成、実音声1件のCPU benchmark、容量+20%判定、emotion2vec特徴抽出、cache検証を上から順に行います。

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
from ser_pipeline.exclusions import load_msp_missing_audio_exclusion_contract
from ser_pipeline.features import Emotion2vecEncoder, extract_feature_cache
from ser_pipeline.manifest import (
    audit_dataset, build_manifest, generate_msp_missing_audio_exclusion_contract,
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
    'cache_root': str(CACHE_ROOT),
    'device': 'cpu',
    'run_flags': {
        'audit': RUN_MSP_AUDIT,
        'generate_exclusion_contract': RUN_MSP_GENERATE_EXCLUSION_CONTRACT,
        'verify_exclusion_contract': RUN_MSP_VERIFY_EXCLUSION_CONTRACT,
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

`RUN_MSP_AUDIT = True`にした場合だけ実データを読みます。添付の利用不能候補リストは除外条件に使用しません。結果には、現在不足している対象音声の元ラベル、4クラス変換後ラベル、公式split別件数も含まれます。`missing_eligible_audio == 874`と固定内訳、`unregistered_audio_files == 0`、現行契約の対象25,985件を確認してから次へ進みます。
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
## 3. 除外候補生成

`RUN_MSP_GENERATE_EXCLUSION_CONTRACT = True`にした場合だけ、添付リストを参照せず、metadata上の4クラス対象と現在のWAV inventoryから不足行を再計算します。874件・固定内訳に一致しない場合はJSONを書きません。
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

`RUN_MSP_VERIFY_EXCLUSION_CONTRACT = True`にすると、874件、元ラベル`A 378 / H 392 / S 80 / D 24`、公式split`Train 520 / Development 210 / Test1 144`、ファイル名順、重複なし、正規化SHA-256を検証します。
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
## 6. strict manifest作成

`RUN_MSP_BUILD_MANIFEST = True`にすると、承認SHAと除外契約を照合し、契約内874件だけを`included: false`にします。一覧外欠損、復旧済み契約対象、metadata不一致、音声デコード失敗、重複、話者漏洩、ラベル契約違反があれば停止します。
        """,
        "manifest-build-heading",
    ),
    code(
        """
if RUN_MSP_BUILD_MANIFEST:
    if APPROVED_MSP_EXCLUSION_SHA256 is None:
        raise RuntimeError('Set APPROVED_MSP_EXCLUSION_SHA256 after reviewing the exclusion contract')
    msp_root = require_path(MSP_ROOT, 'MSP_ROOT', kind='directory')
    require_path(EXCLUSION_CONTRACT_PATH, 'EXCLUSION_CONTRACT_PATH', kind='file')
    manifest_build_report = build_manifest(
        'msp_podcast',
        msp_root,
        MANIFEST_PATH,
        strict=True,
        inspect_excluded_audio=True,
        approved_exclusion_contract=EXCLUSION_CONTRACT_PATH,
        expected_exclusion_sha256=APPROVED_MSP_EXCLUSION_SHA256,
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
## 7. manifestと実音声の完全検証

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
## 8. 実音声1件のCPU benchmark

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
## 9. 全件所要時間・容量+20%ゲート

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
## 10. MSP-Podcast全件特徴抽出

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
## 11. cache最終検証

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

検証済みframe cacheとmanifestだけを入力にし、MSP-Podcast学習、HCUDB継続学習、両データセットの追加学習前後評価を行います。IEMOCAPは今回の一括研究経路には含めません。実データ1 epoch疎通と正式実行は別の出力先を使い、どちらも既定で無効です。
        """,
        "intro",
    ),
    code(
        """
import os, sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'ser_pipeline').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ser_pipeline.cache import validate_cache
from ser_pipeline.notebook_api import environment_summary
from ser_pipeline.study import DatasetArtifacts, require_formal_epochs, run_transfer_study
from ser_pipeline.training import TrainingConfig

STUDY_DATASETS = ('msp_podcast', 'hcudb1')
RUN_REAL_SMOKE = False
RUN_FORMAL_SEED_42 = False
RUN_FORMAL_SEEDS_43_44 = False
FORMAL_EPOCHS = None

# 実測・検証結果をユーザーが確認した後だけTrueにします。
CONFIRM_CACHE_VALIDATION = False
CONFIRM_BENCHMARK_AND_CAPACITY = False
CONFIRM_SMOKE_COMPLETED = False
CONFIRM_SEED_42_ARTIFACTS = False

ARTIFACT_DIR = PROJECT_ROOT / 'runs' / 'ser_decoder_study'


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
        if name == 'msp_podcast':
            exclusion_key = 'SER_MSP_PODCAST_EXCLUSION_CONTRACT'
            if not os.environ.get(exclusion_key):
                missing.append(exclusion_key)
            else:
                exclusion_contract_path = Path(os.environ[exclusion_key])
        if manifest_key not in missing and cache_key not in missing:
            artifacts[name] = DatasetArtifacts(
                manifest_path=Path(os.environ[manifest_key]),
                cache_root=Path(os.environ[cache_key]),
                exclusion_contract_path=exclusion_contract_path,
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
    cache_validation = {
        name: validate_cache(current.cache_root, current.manifest_path)
        for name, current in artifacts.items()
    }
    return artifacts, cache_validation
        """,
        "setup",
    ),
    markdown("## 1. cache-only実行環境", "environment-heading"),
    code("environment_summary()", "environment"),
    markdown("## 2. 実行設定の確認", "configuration-heading"),
    code(
        """
{
    'datasets': STUDY_DATASETS,
    'device': 'cpu',
    'run_real_smoke': RUN_REAL_SMOKE,
    'run_formal_seed_42': RUN_FORMAL_SEED_42,
    'run_formal_seeds_43_44': RUN_FORMAL_SEEDS_43_44,
    'formal_epochs': FORMAL_EPOCHS,
}
        """,
        "configuration",
    ),
    markdown(
        """
## 3. 実データ1 epoch疎通（正式集計外）

seed 42でMSP親学習→HCUDB継続学習→両データセットの前後評価を行います。出力は`smoke/`に隔離され、正式結果には混ぜません。
        """,
        "smoke-heading",
    ),
    code(
        """
if RUN_REAL_SMOKE:
    smoke_artifacts, smoke_cache_validation = validate_execution_gates()
    smoke_summary = run_transfer_study(
        smoke_artifacts,
        ARTIFACT_DIR / 'smoke',
        seeds=(42,),
        base_config=TrainingConfig(seed=42, device='cpu', epochs=1),
    )
else:
    smoke_summary = {'status': 'disabled_by_default', 'seed': 42, 'epochs': 1}
smoke_summary
        """,
        "smoke-gate",
    ),
    markdown(
        """
## 4. 正式seed 42実行ゲート

1 epoch疎通の時間と履歴を確認して`FORMAL_EPOCHS`を正の整数に固定し、先にseed 42だけを実行します。未設定のまま実行すると拒否します。
        """,
        "formal-seed-42-heading",
    ),
    code(
        """
if RUN_FORMAL_SEED_42:
    if not CONFIRM_SMOKE_COMPLETED:
        raise RuntimeError('Confirm the real-data 1 epoch smoke run before formal seed 42')
    formal_epochs = require_formal_epochs(FORMAL_EPOCHS)
    formal_artifacts, formal_cache_validation = validate_execution_gates()
    formal_seed_42_summary = run_transfer_study(
        formal_artifacts,
        ARTIFACT_DIR / 'formal' / 'initial-seed-42',
        seeds=(42,),
        base_config=TrainingConfig(seed=42, device='cpu', epochs=formal_epochs),
    )
else:
    formal_seed_42_summary = {
        'status': 'disabled_by_default', 'seed': 42, 'formal_epochs': FORMAL_EPOCHS
    }
formal_seed_42_summary
        """,
        "formal-seed-42-gate",
    ),
    markdown(
        """
## 5. 正式seed 43・44実行ゲート

seed 42のcheckpoint、評価signature、cache ID、設定値を確認した後だけ実行します。出力はseed 42の正式出力と分けて保存します。
        """,
        "formal-followup-heading",
    ),
    code(
        """
if RUN_FORMAL_SEEDS_43_44:
    if not CONFIRM_SEED_42_ARTIFACTS:
        raise RuntimeError('Confirm the formal seed 42 artifacts before seeds 43 and 44')
    formal_epochs = require_formal_epochs(FORMAL_EPOCHS)
    followup_artifacts, followup_cache_validation = validate_execution_gates()
    formal_followup_summary = run_transfer_study(
        followup_artifacts,
        ARTIFACT_DIR / 'formal' / 'followup-seeds-43-44',
        seeds=(43, 44),
        base_config=TrainingConfig(seed=43, device='cpu', epochs=formal_epochs),
    )
else:
    formal_followup_summary = {
        'status': 'disabled_by_default', 'seeds': [43, 44], 'formal_epochs': FORMAL_EPOCHS
    }
formal_followup_summary
        """,
        "formal-followup-gate",
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
