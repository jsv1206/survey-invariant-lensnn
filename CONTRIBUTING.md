# Contributing

1. Fork and create a feature branch.
2. Install the dev environment:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[dev]"
   ```
3. Run the checks before opening a PR:
   ```bash
   ruff check src tests
   pytest --runslow          # the FULL suite -- required before a PR
   ```
   During iteration use the default instead:
   ```bash
   pytest                    # ~43s, 89 tests, 7 skipped
   ```
   The 7 skipped tests are simulator-heavy `lenstronomy` renders (`slow` marker)
   and account for **61% of the full suite's runtime**. They are gated behind
   `--runslow` so day-to-day edits do not pay ~70s for tests that only matter when
   `sim_lens.py` or `data.py`'s render path changes. Skips are always REPORTED, so
   a green run states what it covered -- but do not call a default run "all tests
   pass"; that phrase belongs to `--runslow` only. See `tests/conftest.py`.
4. Keep modules small and physicist-readable; new experiment switches should be
   constructor flags on `LitLensNet`, not hard-coded in the training loop.
