# Repository Guidelines

This repository ingests daily spreadsheet logs and produces dosage and habit recommendations. Use this guide to align with the existing workflows, tooling, and style expectations.

## Project Layout
- `action_optimizer/optimizer.py` – core analysis pipeline.
- `action_optimizer/autofill.py` – CLI utility that backfills spreadsheet cells using heuristics defined in the header rows.
- `action_optimizer/tests/` – `unittest` suites covering optimizer and autofill behaviour.
- `scripts/`, `pep8*.sh`, `list_*` – maintenance helpers; assume they are invoked from the repo root.
- External assets (reports, large spreadsheets) live outside the repo; treat inputs such as `input.ods` as scratch files unless told otherwise.

## Environment & Tooling
- Target Python 3.10. Bootstrap the virtualenv with `./init_virtualenv.sh` before running anything else.
- Native binaries used by the pipeline: Weka, libsvm, matplotlib backends. Make sure they are installed locally or in CI before running end-to-end flows.
- Formatting and linting run through pre-commit: `yapf`/`.style.yapf` handle formatting, `pylint` (via `pep8.sh` and `pep8-changed.sh`) enforces lint rules defined in `pylint.rc`. The `.pre-commit-config.yaml` orchestrates both; run `pre-commit run --all-files` or use the helper scripts before sending patches.

## Coding Conventions
- Python files use 4-space indentation, `snake_case` identifiers, and module-level loggers (see `optimizer.py` for patterns). Keep functions focused; large routines may need refactoring into helpers under `action_optimizer/`.
- Preserve spreadsheet semantics: when modifying autofill logic, copy formulas and cached values together so LibreOffice recalculates correctly. Never mutate metadata rows (`learn`, `predict`, `default`, etc.) without explicit direction.
- Document tricky data flows with concise comments, especially when translating spreadsheet conventions into code.

## Testing & Verification
- Run the full suite with `tox` or `./test.sh` (wrapper around `tox`). `tox` uses Python 3.9/3.10 as configured; sync the virtualenv accordingly.
- Targeted checks: `python -m unittest action_optimizer.tests.test_autofill`, `python -m unittest action_optimizer.tests.test_optimizer`.
- For spreadsheet changes, verify before/after behaviour with the standalone CLI:
  ```bash
  source .env/bin/activate
  python action_optimizer/autofill.py input.ods --output output.ods
  ```
  Compare “learn/predict/default” rows and any `last` formula columns to confirm values and formulas align with expectations.

## Collaboration Tips
- Keep large sample files out of version control unless explicitly required; reference them in notes or fixtures instead.
- When extending autofill or optimizer logic, add regression tests in `action_optimizer/tests/` mirroring the spreadsheet scenarios you are handling.
- Share follow-up commands (lint, tox, CLI) in PR summaries so reviewers can reproduce your checks quickly.
