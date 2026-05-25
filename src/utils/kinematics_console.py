"""Console / tqdm policy for kinematics training (Windows-friendly logs)."""

from __future__ import annotations

import os


def kinematics_quiet_logs() -> bool:
    """Epoch + validation summary lines only (no tqdm bars)."""
    return os.environ.get("KINEMATICS_QUIET", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def kinematics_tqdm_enabled() -> bool:
    if kinematics_quiet_logs():
        return False
    raw = os.environ.get("KINEMATICS_TQDM", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def kinematics_val_progress_enabled() -> bool:
    if kinematics_quiet_logs():
        return False
    raw = os.environ.get("KINEMATICS_VAL_PROGRESS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")
