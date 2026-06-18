# Testing

This repository uses a dedicated WSL/Ubuntu Python environment for tests.
Run tests with the following command from Windows:

```bash
wsl -d Ubuntu --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest discover -s tests
```

Expected result:

```text
Ran 32 tests in ...s
OK
```

Standard environment:

- WSL distribution: `Ubuntu`
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
wsl -d Ubuntu --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -c "import torch, fairseq, soundfile, hydra, numpy; print('ok')"
```

If the environment must be recreated inside Ubuntu, use Python 3.10:

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
- In sandboxed sessions, `wsl` may be available but the Ubuntu distribution may
  not be visible. In that case, run the command with approved execution outside
  the sandbox.
- The standard test environment for this repository is the `Ubuntu` WSL
  distribution with
  `/home/akiyama/miniforge/envs/emotion2vec-py310/bin/python`.
- Docker and native Windows Python test setup are outside the current supported
  workflow.
