# Testing

This repository uses a dedicated WSL/Ubuntu Python environment for tests.
Run tests with the following command from Windows:

```bash
wsl -d Ubuntu --cd /mnt/c/Users/RD004/Documents/lab/emotion2vec -e /home/akiyama/miniforge/envs/emotion2vec-py310/bin/python -m unittest discover -s tests
```

Expected result:

```text
Ran 20 tests in ...s
OK
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

