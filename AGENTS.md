# Repository working rules

- Edit an existing source file in place and use Git history for rollback.
- Do not create versioned or backup copies such as `*_old.py`, `*_new.py`,
  `*_backup.py`, `*_copy.py`, or `*_v2.py` when revising code.
- Add a new Python file only when it has a distinct runtime, build, or test role.
- Every new Python module must start with a module docstring that states its role.
- Move code into `archive/` only when the user explicitly requests preservation of
  an obsolete implementation; archived code is not part of the active runtime.
