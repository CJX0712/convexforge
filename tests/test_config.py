"""Configuration: ENV overrides, validation, defaults."""

from __future__ import annotations

import os

import pytest

from convexforge.core.config import Config
from convexforge.core.errors import ConfigError


def test_defaults_validate():
    Config().validate()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("ENV_SEED", "11")
    monkeypatch.setenv("ENV_MAX_ITERS", "77")
    monkeypatch.setenv("ENV_N_VARS", "50")
    cfg = Config.from_env()
    assert cfg.seed == 11
    assert cfg.max_iters == 77
    assert cfg.n_vars == 50


def test_env_bad_value_raises(monkeypatch):
    monkeypatch.setenv("ENV_SEED", "not-an-int")
    with pytest.raises(ConfigError):
        Config.from_env()


def test_n_cons_must_be_less_than_n_vars():
    cfg = Config(n_vars=10, n_cons=10)
    with pytest.raises(ConfigError):
        cfg.validate()


def test_positive_checks():
    for bad in ({"max_iters": 0}, {"n_seeds": 0}, {"n_train": 0}, {"cert_tol": 0}):
        with pytest.raises(ConfigError):
            Config(**bad).validate()


def test_env_is_not_persisted_between_calls(monkeypatch):
    monkeypatch.delenv("ENV_SEED", raising=False)
    assert Config.from_env().seed == 7
    assert "ENV_SEED" not in os.environ or os.environ.get("ENV_SEED") == ""


def test_regimes_default_cover_all_families():
    assert set(Config().regimes) == {"lp", "qp_pd", "qp_singular", "degen"}
