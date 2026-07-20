"""Entry-point shim for the super-router.

The implementation lives in the ``routerlib`` package next to this file. This
shim keeps the historical ``python scripts/router.py '<task>'`` invocation and
the ``import scripts.router`` import surface working unchanged: it ensures the
``scripts`` directory is importable, then re-exports the full public API from
``routerlib``.
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Stdlib modules the test suite reaches through ``scripts.router`` (e.g.
# ``r.subprocess``, ``r.urllib.error.HTTPError``). Import them explicitly so the
# flat attribute surface is guaranteed regardless of package internals.
import signal  # noqa: E402,F401
import subprocess  # noqa: E402,F401
import urllib.error  # noqa: E402,F401
import urllib.request  # noqa: E402,F401

from routerlib import *  # noqa: E402,F401,F403
from routerlib import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
