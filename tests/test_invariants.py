"""The mathematical invariant gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from util import make_task

from convexforge.core.errors import CertificateError, NumericalError
from convexforge.data.synthetic import ground_truth_kkt, make_problem
from convexforge.eval.certificate import certificate, duality_gap_lp
from convexforge.eval.invariants import (
    assert_finite,
    assert_ground_truth_kkt,
    assert_monotone,
    assert_no_leak,
    assert_no_solution_beats_optimum,
    assert_psd_problem,
    assert_weak_duality_lp,
)
from convexforge.solvers.engines import base_engine

REGIMES = ("lp", "qp_pd", "qp_singular", "degen")


def test_psd_problem_all_regimes():
    for reg in REGIMES:
        assert_psd_problem(make_task(reg))


def test_not_psd_raises():
    prob = make_task("qp_pd")
    prob.Q[0, 0] = -5.0
    with pytest.raises(NumericalError):
        assert_psd_problem(prob)


def test_ground_truth_kkt_exact():
    for reg in REGIMES:
        assert_ground_truth_kkt(make_task(reg))


def test_certificate_vanishes_at_optimum():
    for reg in REGIMES:
        prob = make_task(reg)
        cert = certificate(prob, prob.x_star, prob.meta["y_star"])["cert"]
        assert cert < 1e-8, f"{reg}: cert at optimum = {cert}"


def test_certificate_detects_a_bad_point():
    prob = make_task("qp_pd")
    rng = np.random.default_rng(0)
    bad = prob.x_star + rng.standard_normal(prob.n) * 0.5
    assert certificate(prob, bad, prob.meta["y_star"])["cert"] > 1e-3


def test_no_solution_beats_optimum_holds():
    prob = make_task("qp_pd")
    for m in ("pen", "admm", "pdhg", "ipm"):
        r = base_engine(m).solve(prob, max_iters=120, tol=1e-10)
        assert_no_solution_beats_optimum(prob, r.x, tol=1e-5)


def test_no_solution_beats_optimum_fires_on_inflated_optimum():
    """If the claimed optimum is too high, the true optimum must be flagged."""
    prob = make_task("qp_pd")
    inflated = replace(prob, f_star=prob.f_star + 1e3)
    with pytest.raises(CertificateError):
        assert_no_solution_beats_optimum(inflated, prob.x_star, tol=1e-6)


def test_weak_duality_lp():
    prob = make_task("lp")
    r = base_engine("ipm").solve(prob, max_iters=120, tol=1e-10)
    assert_weak_duality_lp(prob, r.x, r.y, tol=1e-6)
    gap = duality_gap_lp(prob, r.x, r.y)
    assert gap > -1e-6, gap


def test_monotone_flags_a_rise():
    with pytest.raises(NumericalError):
        assert_monotone([1.0, 0.5, 0.4, 0.9])


def test_monotone_accepts_a_decrease():
    assert_monotone([5.0, 4.0, 3.9, 3.0])


def test_no_leak():
    assert_no_leak([0, 1, 2], [3, 4])


def test_no_leak_overlap_raises():
    with pytest.raises(CertificateError):
        assert_no_leak([0, 1, 2], [2, 3])


def test_finite_gate():
    assert_finite(np.ones(3), name="ok")
    with pytest.raises(NumericalError):
        assert_finite(np.array([1.0, np.nan]), name="bad")


def test_generator_is_deterministic():
    a = make_problem("qp_pd", 3, 11, 40, 12)
    b = make_problem("qp_pd", 3, 11, 40, 12)
    assert np.array_equal(a.Q, b.Q)
    assert np.array_equal(a.x_star, b.x_star)
    assert a.f_star == b.f_star


def test_ground_truth_helper_reports_zero():
    res = ground_truth_kkt(make_task("degen"))
    assert max(res.values()) < 1e-8
