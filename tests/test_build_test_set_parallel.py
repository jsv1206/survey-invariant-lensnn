"""Regression test for parallelized Test Set 1/2 simulation.

build_test_set_1/2 re-simulate from scratch on every test.py/diagnose.py
invocation, single-threaded by default -- slow (~15-20 min for 200 lenses x
5 surveys observed on this sandbox). Added an optional `n_workers` that
farms out the expensive simulate_one render step to a process pool while
keeping the parameter draw (and, for Test Set 2, the noise draw) sequential,
so the result must be byte-identical regardless of n_workers -- only wall
time should change.
"""

import pytest
import torch

from survey_lens import env_setup  # noqa: F401  (must precede lenstronomy)
from survey_lens import surveys as SV
from survey_lens.data import build_test_set_1, build_test_set_2

# 9s -- builds real test sets twice (sequential vs pooled) to prove byte-identity.
# Only meaningful when data.py's render path changes. See tests/conftest.py.
pytestmark = pytest.mark.slow

FOV = 8.0


def test_build_test_set_1_parallel_matches_sequential():
    seq = build_test_set_1(SV.SURVEYS, n_lenses=6, seed=3, fov=FOV, n_workers=1)
    par = build_test_set_1(SV.SURVEYS, n_lenses=6, seed=3, fov=FOV, n_workers=3)
    for nm in SV.SURVEYS:
        for a, b in zip(seq[nm], par[nm]):
            assert torch.equal(a["g"], b["g"])
            assert torch.equal(a["f_gt"], b["f_gt"])
            assert float(a["theta_E"]) == float(b["theta_E"])


def test_build_test_set_2_parallel_matches_sequential():
    """Noise draws must stay sequential/order-preserving even when the
    expensive render step is parallelized -- otherwise the noise realization
    would depend on worker scheduling, breaking reproducibility."""
    seq = build_test_set_2(SV.SURVEYS, n_lenses=6, seed=3, fov=FOV, n_workers=1)
    par = build_test_set_2(SV.SURVEYS, n_lenses=6, seed=3, fov=FOV, n_workers=3)
    for nm in SV.SURVEYS:
        for a, b in zip(seq[nm], par[nm]):
            assert torch.equal(a["g"], b["g"])       # noisy input must match exactly
            assert torch.equal(a["f_gt"], b["f_gt"])  # clean truth must match exactly
