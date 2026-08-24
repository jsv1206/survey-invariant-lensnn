"""Fast-by-default test runs; the simulator-heavy tests are opt-in.

The full suite is ~114s, and **54% of that is five tests in
`test_flux_normalization.py`** (61s) plus two in `test_build_test_set_parallel.py`
(9s). Both files drive `lenstronomy` renders at real survey resolutions, which is
inherently slow and has nothing to do with the model, the losses, the metrics or
the figure code that actually changes day to day.

So they are marked `slow` and skipped unless `--runslow` is passed. Everything
else -- including the tests that pin the reblur centering fix (bug 11), the
reblur noise floor (bug 12), the decoder contract, and the Wiener baseline
fairness rules -- still runs every time, because those are the ones that catch
regressions in current work.

    pytest                  # ~40s, iteration
    pytest --runslow        # ~114s, the full 96 -- FINAL CHECK / pre-commit

**Skips are reported, never silent.** pytest prints "N skipped" in its summary,
so a green run always states what it did and did not cover. Do not convert this
into a filter that hides tests without saying so, and do not quote a run as
"all tests pass" unless it was a `--runslow` run.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="also run the simulator-heavy tests marked `slow` (full suite)")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: simulator-heavy (lenstronomy renders); skipped unless --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(
        reason="slow simulator test -- pass --runslow to include (see tests/conftest.py)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
