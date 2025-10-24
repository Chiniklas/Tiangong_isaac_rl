"""Lightweight debug logging for UniGraspTransformer components."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _is_debug_enabled() -> bool:
    """Return True if verbose logging is enabled."""

    value = os.getenv("UNIGRASPTRANSFORMER_DEBUG", "").strip().lower()
    if value in {"", "0", "false", "off", "no"}:
        return False
    return True


def log_debug(message: str) -> None:
    """Print a debug message when UNIGRASPTRANSFORMER_DEBUG is enabled."""

    if _is_debug_enabled():
        print(f"[DEBUG][UniGraspTransformer] {message}")


__all__ = ["log_debug"]
