"""Make the repo-root ``testing`` package importable from inside the
``approach_1`` / ``approach_2`` test suites.

pytest only adds the rootdir + each package's tests dir to ``sys.path`` by
default, so ``import testing`` from ``approach_2/tests`` would otherwise fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))