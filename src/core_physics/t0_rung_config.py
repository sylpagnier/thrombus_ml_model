"""Compatibility shim for legacy rung env helpers.

Active GraphSAGE deploy code only needs a gamma mode constant and a context
manager used around rollout/eval calls.
"""

from __future__ import annotations

from contextlib import contextmanager

RUNG2_GAMMA_MODE = "proxy"


@contextmanager
def t0_rung2_env():
    """No-op context kept for backwards compatibility."""
    yield

