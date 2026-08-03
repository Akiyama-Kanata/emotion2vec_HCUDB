# Testing

This repository uses a dedicated WSL/Ubuntu Python environment for tests.
Run tests with the following command from Windows:

```bash
wsl -d Ubuntu-Recovered --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest discover -s tests
```

Expected result:

```text
Ran 65 tests in ...s
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

On 2026-08-02, the recovered environment started successfully, but the full
suite did not pass:

```text
Ran 59 tests in 3.953s
FAILED (errors=2)
Successful: 57; failures: 0; errors: 2
First failing test: test_notebook_pipeline (unittest.loader._FailedTest)
Exception tail: ModuleNotFoundError: No module named 'pandas'
```
