# Contributing

pyOptoTrace is in an early extraction and validation phase. Small, test-backed changes are the
easiest to review.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ruff check src tests examples
```

## Pull requests

1. Open an issue before a large API or pipeline change.
2. Keep research data, identifiers, absolute workstation paths, and notebook outputs out of the
   repository.
3. Add or update a test for behavior changes.
4. Document array shapes, units, sign conventions, and optional dependencies.
5. Confirm that you have the right to contribute the code under the MIT License.

Public examples must use synthetic data or data whose redistribution terms are documented.
