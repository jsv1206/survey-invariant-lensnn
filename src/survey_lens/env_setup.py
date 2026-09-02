"""Import this FIRST, before lenstronomy/astropy, in every script/kernel.

Redirects astropy/XDG/numba config+cache into the workspace (site-packages and
~ are read-only in this sandbox) and points lenstronomy at a user config that
keeps numba JIT enabled but with cache=False (the numba file-cache locator
fails under the kernel's exec-from-fd import, but cache=False avoids it while
keeping the JIT speedup).

Also pins CPU thread pools to the container's actual cgroup CPU quota. This
sandbox reports `os.cpu_count() == 127` (the host's core count) but is
cgroup-limited to far fewer -- `/sys/fs/cgroup/cpu.max` reads `3200000
100000` here, i.e. 32 cores. Python's cpu_count()/sched_getaffinity() do not
see cgroup quotas (no cpuset), so PyTorch defaults to 127 intra-op threads
and massively oversubscribes the 32-core quota: measured on a grouped
conv2d, 438ms at 127 threads vs 58-70ms at 8-32 threads (~7x). At HST image
scale this pushed a single reblur_loss forward+backward over 2s and is a
likely contributor to jobs stalling/getting killed under this quota; see
Also stubs `torchvision` if the real one is broken. In this sandbox
`torchvision==0.19.0` (nominally paired with the installed `torch
2.4.0.post301`) fails at import with `RuntimeError: operator
torchvision::nms does not exist` -- an ABI mismatch, most likely because
`.post301` is a locally patched/rebuilt torch that this stock torchvision
wheel wasn't compiled against. `lightning` (hence `data.py`/`train.py`/
`test.py`) transitively imports `torchvision` at MODULE-IMPORT time via
`torchmetrics.functional.image` (an unconditional import of its whole
submodule package, not lazy per-metric) even though survey_lens never calls
any torchmetrics image metric. Without this stub, `import lightning` -- and
therefore any of this project's training/eval scripts -- cannot even be
imported in this environment. Only activates if the real import actually
fails, so it is a no-op the moment torchvision is fixed properly.
"""
import os, sys, textwrap


def _stub_broken_torchvision():
    import types
    try:
        import torchvision  # noqa: F401
        return  # real one works -- nothing to do
    except Exception:
        pass
    for _mod in list(sys.modules):          # drop any partially-imported state
        if _mod == "torchvision" or _mod.startswith("torchvision."):
            del sys.modules[_mod]

    class _Dummy:
        """Placeholder for a torchvision symbol referenced only by
        torchmetrics' (unused, image-quality) metrics at import time."""
        def __init__(self, *a, **k):
            pass
        def __call__(self, *a, **k):
            raise RuntimeError(
                "torchvision is stubbed in this sandbox (ABI-broken install); "
                "survey_lens does not use torchmetrics image metrics.")

    def _missing(name):
        if name.startswith("_"):           # preserve normal getattr(x, dunder, default) semantics
            raise AttributeError(name)
        return _Dummy

    def _permissive_module(name):
        m = types.ModuleType(name)
        m.__path__ = []                    # so `torchvision.sub` submodule imports resolve
        m.__getattr__ = _missing           # PEP 562: satisfies `from torchvision.x import anything`
        return m

    for _name in ("torchvision", "torchvision.transforms", "torchvision.models",
                 "torchvision.transforms.functional", "torchvision.io",
                 "torchvision.ops", "torchvision.datasets", "torchvision.utils"):
        sys.modules[_name] = _permissive_module(_name)


_stub_broken_torchvision()

_WD = os.path.dirname(os.path.abspath(__file__))
# The sandbox runs python with PYTHONSAFEPATH=1, so the script/module directory
# is NOT auto-added to sys.path. Add it here so sibling modules import cleanly
# from any entry point (CLI script, kernel, notebook).
if _WD not in sys.path:
    sys.path.insert(0, _WD)
_CFG = os.path.join(_WD, ".cfg")

os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(_CFG, "xdg_config"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_CFG, "xdg_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_CFG, "numba_cache"))
for _d in [os.environ["XDG_CONFIG_HOME"], os.environ["XDG_CACHE_HOME"],
           os.environ["NUMBA_CACHE_DIR"], os.path.join(_CFG, "astropy")]:
    os.makedirs(_d, exist_ok=True)

def _cgroup_cpu_quota():
    """Cores allowed by the cgroup CPU quota, or None if unset/unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota == "max":
            return None
        return max(1, int(quota) // int(period))
    except (FileNotFoundError, ValueError):
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return max(1, quota // period) if quota > 0 else None
    except (FileNotFoundError, ValueError):
        return None


_n_cpu = _cgroup_cpu_quota() or os.cpu_count() or 1
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(_n_cpu))
try:
    import torch
    torch.set_num_threads(_n_cpu)  # env vars above only bind at first parallel-op init; this is authoritative regardless of import order
except ImportError:
    pass

_lconf_dir = os.path.join(os.environ["XDG_CONFIG_HOME"], "lenstronomy")
os.makedirs(_lconf_dir, exist_ok=True)
_lconf = os.path.join(_lconf_dir, "config.yaml")
if not os.path.exists(_lconf):
    with open(_lconf, "w") as f:
        f.write(textwrap.dedent("""\
            numba:
              enable: True
              nopython: True
              cache: False
              parallel: False
              error_model: 'numpy'
              fastmath: False
            conventions:
              sersic_major_axis: False
            """))
