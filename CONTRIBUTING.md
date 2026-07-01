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
   pytest -q
   ```
4. Keep modules small and physicist-readable; new experiment switches should be
   constructor flags on `LitLensNet`, not hard-coded in the training loop.
