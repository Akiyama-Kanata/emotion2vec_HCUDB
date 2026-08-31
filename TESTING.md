# Testing

This repository uses a dedicated WSL/Ubuntu Python environment for tests.
Run tests with the following command from Windows:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest discover -s tests
```

Expected result:

```text
Ran 101 tests in ...s
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

The user-run MSP exclusion/manifest sequence is:

```bash
python -m ser_pipeline generate-msp-exclusion-contract \
  --root /path/to/MSP_PODCAST \
  --output runs/ser_manifests/msp_missing_audio_exclusions_v1.json
python -m ser_pipeline build-manifest \
  --dataset msp_podcast \
  --root /path/to/MSP_PODCAST \
  --output runs/ser_manifests/msp_podcast_4class_v1.jsonl \
  --approved-exclusion-contract runs/ser_manifests/msp_missing_audio_exclusions_v1.json \
  --expected-exclusion-sha256 APPROVED_64_HEX_SHA256
```

The second command refuses an unset or mismatched approval SHA. Both commands
are real-data operations and are not run by Codex.

Final implementation verification on 2026-08-23:

```text
Ran 101 tests in 16.030s
OK
Separated SER demo notebooks completed
```
