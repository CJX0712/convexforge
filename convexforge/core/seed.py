"""Deterministic seeding — single entry point for the whole system.

BLAS/LAPACK threads are pinned to 1 so multithreaded reductions cannot change
summation order between runs (a real source of nondeterminism).
"""

from __future__ import annotations

import os
import random

import numpy as np

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def pin_threads() -> None:
    """Pin every BLAS/LAPACK thread env var to 1 (call before importing numpy)."""
    for key in _THREAD_KEYS:
        os.environ[key] = "1"


def set_all(seed: int) -> None:
    """Seed the Python RNG and NumPy RNG."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
