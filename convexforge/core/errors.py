"""Error taxonomy E100~E500."""

from __future__ import annotations

from typing import Any


class ConvexForgeError(Exception):
    """Base error. Carries a stable machine code."""

    code = "E000"

    def __init__(self, message: str, **ctx: Any) -> None:
        self.ctx = ctx
        suffix = " ".join(f"{k}={v}" for k, v in ctx.items())
        super().__init__(f"[{self.code}] {message}" + (f" ({suffix})" if suffix else ""))


class ConfigError(ConvexForgeError):
    code = "E100"


class ShapeError(ConvexForgeError):
    code = "E200"


class NumericalError(ConvexForgeError):
    code = "E300"


class CertificateError(ConvexForgeError):
    code = "E400"


class DataLeakError(ConvexForgeError):
    code = "E500"
