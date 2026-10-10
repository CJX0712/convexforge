"""CLI smoke tests."""

from __future__ import annotations

import json

import pytest

from convexforge.cli import main
from convexforge.pipeline.selftest import run_selftest


def test_selftest_all_pass():
    res = run_selftest()
    assert res["all_pass"], [c for c in res["checks"] if not c[1]]


def test_selftest_has_ten_gates():
    res = run_selftest()
    assert len(res["checks"]) == 10


def test_cli_selftest_exit_zero():
    assert main(["selftest"]) == 0


def test_cli_demo_writes_benchmark_json(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV_N_VARS", "24")
    monkeypatch.setenv("ENV_N_CONS", "6")
    monkeypatch.setenv("ENV_MAX_ITERS", "40")
    monkeypatch.setenv("ENV_N_SEEDS", "1")
    monkeypatch.setenv("ENV_N_TRAIN", "1")
    monkeypatch.setenv("ENV_N_TEST", "1")
    out = tmp_path / "benchmark.json"
    assert main(["demo", "--out", str(out)]) == 0

    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) >= {"config", "rows", "per_method", "per_regime", "headline"}
    assert "deterministic" in data
    h = data["headline"]
    assert set(h) >= {
        "naive_pen",
        "flagship_convfuse",
        "best_component",
        "orders_vs_best_component",
        "orders_vs_naive",
        "non_inferior",
        "gate_pass",
    }
    assert h["best_component"]["method"] in {"admm", "pdhg", "ipm"}
    assert data["failure_cases"], "at least one failure case expected"
    assert data["ablation"], "ablation must be reported"


def test_cli_run_is_alias_for_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV_N_VARS", "20")
    monkeypatch.setenv("ENV_N_CONS", "5")
    monkeypatch.setenv("ENV_MAX_ITERS", "20")
    monkeypatch.setenv("ENV_N_SEEDS", "1")
    monkeypatch.setenv("ENV_N_TRAIN", "1")
    monkeypatch.setenv("ENV_N_TEST", "1")
    out = tmp_path / "b2.json"
    assert main(["run", "--out", str(out)]) == 0
    assert out.exists()


def test_no_subcommand_defaults_to_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("ENV_N_VARS", "20")
    monkeypatch.setenv("ENV_N_CONS", "5")
    monkeypatch.setenv("ENV_MAX_ITERS", "20")
    monkeypatch.setenv("ENV_N_SEEDS", "1")
    monkeypatch.setenv("ENV_N_TRAIN", "1")
    monkeypatch.setenv("ENV_N_TEST", "1")
    out = tmp_path / "b3.json"
    assert main(["--out", str(out)]) == 0
    assert out.exists()


def test_cli_rejects_unknown_subcommand():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
