# Development oracle harness (delete when JUG is standalone)

JUG is **standalone**: no runtime dependency on libstempo, tempo2, or pytempo.
During parity development we keep optional **external oracles** here and in
marked pytest modules.

## Delete checklist

When JUG no longer needs tempo2 cross-checks:

1. Remove this file and oracle modules:
   - `jug/testing/sandbox_tempo2.py`
   - `jug/testing/tempo2_reference.py`
   - `jug/testing/tempo2_diagnostics.py` (or strip oracle backends, keep JUG-only helpers)
2. Remove pytest modules matching:
   - `tests/test_dev_oracle_*.py`
   - `tests/test_tempo2_*.py`
3. Drop `dev_oracle` / `tempo2` markers from `pyproject.toml` if unused.
4. Replace acceptance with native golden vectors or an external harness repo.

## Grep targets

```bash
rg 'dev_oracle|sandbox_tempo2|tempo2_reference|importorskip\("libstempo"\)' jug tests
pytest -m dev_oracle   # run only oracle-backed dev tests
pytest -m 'not dev_oracle'  # JUG-only CI path
```
