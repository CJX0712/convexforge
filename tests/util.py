"""Shared test utilities (importable by every test module)."""

from __future__ import annotations

from convexforge.core.config import Config
from convexforge.core.seed import set_all
from convexforge.data.synthetic import build_problem_set, make_problem


def small_cfg(**over: object) -> Config:
    """A deliberately tiny config so the unit suite stays fast."""
    base: dict[str, object] = {
        "seed": 7,
        "n_seeds": 1,
        "n_vars": 40,
        "n_cons": 12,
        "n_train": 2,
        "n_test": 2,
        "max_iters": 120,
        "cert_tol": 1e-10,
    }
    base.update(over)
    cfg = Config(**base)  # type: ignore[arg-type]
    cfg.validate()
    return cfg


def make_task(regime: str = "qp_pd", seed: int = 7, **over: object):
    """One reproducible problem instance."""
    cfg = small_cfg(**over)
    set_all(cfg.seed)
    return make_problem(regime, 0, seed, cfg.n_vars, cfg.n_cons)


def split_task(regime: str = "qp_pd", seed: int = 7, **over: object):
    """(train, test) problem blocks."""
    cfg = small_cfg(**over)
    return build_problem_set(regime, seed, cfg.n_vars, cfg.n_cons, cfg.n_train, cfg.n_test)
