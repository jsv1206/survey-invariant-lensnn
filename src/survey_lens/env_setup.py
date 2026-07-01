"""Import this FIRST, before lenstronomy/astropy, in every script/kernel.

Redirects astropy/XDG/numba config+cache into the workspace (site-packages and
~ are read-only in this sandbox) and points lenstronomy at a user config that
keeps numba JIT enabled but with cache=False (the numba file-cache locator
fails under the kernel's exec-from-fd import, but cache=False avoids it while
keeping the JIT speedup).
"""
import os, sys, textwrap

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
