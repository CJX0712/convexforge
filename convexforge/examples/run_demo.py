"""End-to-end demo: synthetic problems -> all engines -> benchmark.json.

Run:
    python examples/run_demo.py
"""

import os
import sys

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_k] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convexforge.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["demo"]))
