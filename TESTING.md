# Testing

This repository uses a dedicated WSL/Ubuntu Python environment for tests.
Run tests with the following command from Windows:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest discover -s tests
```

Expected result:

```text
Ran 122 tests in ...s
OK
```

Standard environment:

- WSL distribution: `Ubuntu-Recovered`
- Python: `/home/akiyama/miniforge/envs/emotion2vec-py310/bin/python`
- Python version: 3.10.20
- pip: 24.0 (`pip<24.1`)
- Key pinned packages:
  - `fairseq==0.12.2`
  - `omegaconf==2.0.6`
  - `hydra-core==1.0.7`
  - `numpy==1.26.4`

Verify imports:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -c "import torch, fairseq, soundfile, hydra, numpy; print('ok')"
```

If the environment must be recreated inside Ubuntu-Recovered, use Python 3.10:

```bash
conda create -n emotion2vec-py310 python=3.10
conda activate emotion2vec-py310
cd /mnt/c/Users/RD004/Documents/lab/emotion2vec
python -m pip install "pip<24.1"
python -m pip install -r requirements.txt
python -c "import torch, fairseq, soundfile, hydra, numpy; print('ok')"
python -m unittest discover -s tests
```

Notes:

- The Windows-side `python` or `py` command may not be available in this setup.
- In sandboxed sessions, `wsl` may be available but the Ubuntu-Recovered distribution may
  not be visible. In that case, run the command with approved execution outside
  the sandbox.
- The standard test environment for this repository is the `Ubuntu-Recovered` WSL
  distribution with
  `/home/akiyama/miniforge/envs/emotion2vec-py310/bin/python`.
- Docker and native Windows Python test setup are outside the current supported
  workflow.

## Latest verification

### MSP class-weight comparison (2026-09-03)

Notebook 02 now ends with `6. MSP単体：クラス重み付き損失の比較`.
Execute only 6.1 through 6.4. Section 6.1 reloads checkpoints, training and study
in dependency order so an already-open kernel picks up the added comparison
functions and configuration fields. Section 6.1 defaults to
`RUN_MSP_WEIGHTED_TRAINING = False` and `MSP_COMPARISON_SEEDS = (42,)`.
Set the flag to `True` to run weighted MSP training, then repeat with `(43, 44)`.
The output directory includes the selected seeds; a nonempty directory is
rejected to preserve existing results. Section 6.4 can also read saved results.

`TrainingConfig.class_weighting` accepts `none` (unchanged default) or `balanced`.
Balanced weights are `N / (4 * n_class)`, calculated solely from included
training utterances in label order `anger, happy, sadness, disgust`. Missing
training classes are rejected for balanced weighting. PyTorch cross entropy
uses its standard weighted-mean reduction (sum of weighted per-item losses
divided by the sum of the observed label weights in each batch). Training loss
is the mean of these batch losses, so weighted and unweighted training losses
must not be compared as the same objective. Validation loss and all metrics
retain the existing unweighted calculation and best-checkpoint selection.

`run_msp_loss_comparison` reuses saved **unweighted validation results**, then
trains weighted MSP models from scratch with the same seed, model, 10 epochs,
batch size 8, learning rate and batch ordering. It runs neither HCUDB training
nor test evaluation. Before training it checks baseline configuration, complete
epoch history, manifest hash, feature cache ID, checkpoint hash/signature and
agreement between summary and checkpoint validation metrics. Both runs use the
same manifest and therefore the same training/validation sets. The cache is
fully validated once and its store reused for the comparison.

Weights, training counts and loss mode are saved in both training summaries
and checkpoints. Existing unweighted checkpoints remain loadable. Resuming
with a different loss configuration is rejected; parent-model loading remains
independent of training loss because the model architecture has not changed.

Non-training regression command (optimizer updates are mocked):

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest discover -s tests -p test_ser_class_weights.py -v
```

Verified on 2026-09-03: 24 non-training tests across class weights, decoder,
cache reuse and notebook boundaries passed in 13.196 seconds. The independent
comparison settings/disabled gate were executed, and the saved real seed
42/43/44 baseline checkpoint signatures, hashes, manifest hashes and validation
metrics were checked read-only. No real training or full feature scan was run.

Before the MSP/HCUDB implementation, the baseline suite passed on 2026-08-23:

```text
Ran 76 tests in 7.431s
OK
```

The non-training SER checks that Codex may run independently are:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest \
  tests.test_ser_mappings tests.test_ser_splits tests.test_ser_manifest \
  tests.test_ser_exclusions tests.test_ser_cache tests.test_ser_notebook_boundaries
```

`tests.test_ser_e2e` calls `train_decoder` on synthetic caches. It is retained as a
user-run regression test and Codex does not execute it. The non-training study
contract/epoch gate can be checked separately by selecting only
`tests.test_ser_e2e.SerEndToEndTest.test_study_contract_and_formal_epoch_gate_do_not_train`.

Run both separated notebooks with all formal/long-running flags disabled:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python tests/execute_ser_demo_notebooks.py
```

Real feature extraction and training are not part of Codex-run verification.
They remain user-run operations gated by complete MSP/HCUDB cache validation,
the one-item CPU benchmark, the +20% capacity margin, an explicitly fixed formal
epoch count, and separated smoke/formal output directories. Formal training runs
seed 42 first; seeds 43 and 44 remain disabled until the seed 42 artifacts are
confirmed.

## Notebook 02 cache reuse and timing (2026-09-03)

The non-training optimization checks are:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest tests.test_ser_cache_reuse tests.test_ser_decoder tests.test_ser_cache tests.test_ser_notebook_boundaries
```

These use synthetic caches and initialized checkpoints. No optimizer step or
`train_decoder` call is executed. They check full-validation counts, changed and
missing files, non-finite features, read-only mmap reuse, direct-copy batch
values/labels/order, evaluation signatures, checkpoint compatibility, prediction
files, and the two-seed study with training replaced by checkpoint-only I/O.

The training regression remains **user-run**:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest tests.test_ser_e2e
```

It includes a comparison of the original per-utterance tensor-copy route and the
direct-to-batch route, asserting equal training history and checkpoint weights on
small CPU inputs. Real-data comparison and speed measurements are also user-run.

The working Notebook 02 may contain user settings, additional cells and saved
results. Boundary tests check disabled defaults generated in a temporary
directory and verify that the working execution cells match those defaults.
Do not regenerate the working notebook to clear a `--check` mismatch, or run
`tests/execute_ser_demo_notebooks.py` on a working notebook whose execution flags
are enabled. For code changes, restart the notebook kernel and rerun setup before
the desired execution cell.
The working setup was missing definitions for the existing
`CONFIRM_CACHE_VALIDATION` and `CONFIRM_BENCHMARK_AND_CAPACITY` gates. Both are
restored with the original `False` defaults; set them to the reviewed values
before a real run. The working `FORMAL_EPOCHS = 10` and other user settings are
preserved.

`prepare_study_stores()` performs the entry-gate validation and passes the same
stores to `run_transfer_study(..., stores=...)`. Each unchanged dataset has one
full validation across all selected seeds and their training/before/after stages.
Standalone `train_decoder()` and `evaluate_checkpoint()` fully validate by
default. Reuse is explicit through `store=...`; `validate=False` is retained for
compatibility but does not bypass validation. No validation proof is persisted.

Before reuse, the store checks the manifest, root metadata, success markers,
shard metadata, indices and feature files by file identity, size, modification
time and change time. A change invalidates old maps and causes revalidation;
invalid/missing inputs are rejected. This assumes inputs remain read-only during
each training/evaluation operation; it is not a concurrent-writer lock or a
defense against metadata-preserving tampering.

Timing files are saved separately from checkpoint history:

- `*_timings.json` beside training checkpoints: cache access/full validation,
  setup, each epoch's batch preparation, computation, validation, checkpoint
  saving, finalization and total. Per-epoch checkpoint saving includes best-state
  selection/copying; timing-file writes are outside that interval.
- `timings.json` in each before/after dataset directory: batch preparation,
  computation, result building, evaluation total and result-file saving.
- `study_timings.json`: full-validation count/seconds per dataset, before/after
  evaluation totals, summary saving and study-call duration. This is the final
  timing record referenced by `timings_path` in `study_summary.json`; the summary
  itself is written before its own save duration is known. Study-call duration
  excludes an entry-gate validation done before the call, which is recorded in
  the per-dataset cache-validation times.

Computation timing includes device transfers. CUDA is synchronized at timing
boundaries; the current Notebook 02 configuration uses CPU. Notebook display
uses `summarize_study()`; full returned results and all prediction files remain
available. Historical results without timing data display `None` for seconds.

This change leaves model computation, epoch/seed selection, batch size, learning
rate, batch order and evaluation rules unchanged. Skipping padded computation,
moving caches to WSL storage, changing CPU thread counts and changing loader
workers remain separate measurement steps; no speedup factor is claimed.

Verified on 2026-09-03 in the designated WSL environment: all 48 selected
non-training SER tests passed in 25.409 seconds. The training regressions and
real-data speed comparison were not executed by Codex.

The user-run MSP exclusion/manifest sequence is:

```bash
python -m ser_pipeline generate-msp-exclusion-contract \
  --root /path/to/MSP_PODCAST \
  --output runs/ser_manifests/msp_missing_audio_exclusions_v1.json
python -m ser_pipeline audit-msp-audio-duplicates \
  --root /path/to/MSP_PODCAST \
  --audit-output runs/ser_manifests/msp_audio_duplicate_audit_v1.json \
  --candidates-csv-output runs/ser_manifests/msp_audio_duplicate_candidates_v1.csv \
  --approved-missing-exclusion-contract runs/ser_manifests/msp_missing_audio_exclusions_v1.json \
  --expected-missing-exclusion-sha256 APPROVED_MISSING_64_HEX_SHA256
python -m ser_pipeline generate-msp-duplicate-exclusion-contract \
  --audit runs/ser_manifests/msp_audio_duplicate_audit_v1.json \
  --approved-id REVIEWED_UTTERANCE_ID \
  --output runs/ser_manifests/msp_audio_duplicate_exclusions_v1.json
python -m ser_pipeline build-manifest \
  --dataset msp_podcast \
  --root /path/to/MSP_PODCAST \
  --output runs/ser_manifests/msp_podcast_4class_v1.jsonl \
  --approved-exclusion-contract runs/ser_manifests/msp_missing_audio_exclusions_v1.json \
  --expected-exclusion-sha256 APPROVED_MISSING_64_HEX_SHA256 \
  --duplicate-audit runs/ser_manifests/msp_audio_duplicate_audit_v1.json \
  --approved-duplicate-exclusion-contract runs/ser_manifests/msp_audio_duplicate_exclusions_v1.json \
  --expected-duplicate-exclusion-sha256 APPROVED_DUPLICATE_64_HEX_SHA256
```

The audit and manifest commands refuse unset or mismatched approval SHAs. The
duplicate-contract command accepts `--approved-id` repeatedly; omit it only when
the reviewed audit has no unresolved cross-split group. These are real-data
operations and are not run by Codex.

Final implementation verification on 2026-08-23:

```text
Ran 101 tests in 16.030s
OK
Separated SER demo notebooks completed
```

MSP exact-duplicate audit implementation verification on 2026-09-01:

```text
Ran 122 tests in 17.821s
OK
Separated SER demo notebooks completed
```
