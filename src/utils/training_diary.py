"""
Append-only JSONL training diary for Phase 1 runs — machine- and human-readable context for
post-hoc analysis (e.g. planning the next run).

Each line is one JSON object: ``ts_utc``, ``run_id``, ``tier``, ``event``, plus event fields.

Disable with ``PHASE1_TRAINING_DIARY=0`` or set ``PHASE1_TRAINING_DIARY_PATH`` to a custom ``.jsonl`` file.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.paths import get_project_root, reports_dir


def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (str, int, bool)):
        return v
    return str(v)


def env_snapshot(*prefixes: str) -> Dict[str, str]:
    """Return ``os.environ`` entries whose keys start with any of ``prefixes``."""
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in prefixes):
            out[k] = v
    return dict(sorted(out.items()))


class TrainingDiary:
    """Writes ``<reports_dir>/training_diary_{tier}_{run_id}.jsonl`` unless overridden."""

    def __init__(self, tier: str, enabled: Optional[bool] = None):
        self.tier = tier
        if enabled is None:
            raw = os.environ.get("PHASE1_TRAINING_DIARY", "1").strip().lower()
            enabled = raw not in ("0", "false", "no", "off")
        self.enabled = enabled
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path: Optional[Path] = None
        if not self.enabled:
            return
        reports = reports_dir()
        custom = os.environ.get("PHASE1_TRAINING_DIARY_PATH", "").strip()
        if custom:
            self.path = Path(custom)
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.path = reports / f"training_diary_{tier}_{self.run_id}.jsonl"
        print(f"📝 Training diary (JSONL): {self.path}")

    def _write(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        row: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "tier": self.tier,
            "event": event,
        }
        row.update(_json_safe(payload))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_run_start(self, **fields: Any) -> None:
        self._write("run_start", dict(fields))

    def log_epoch_end(self, epoch: int, **fields: Any) -> None:
        self._write("epoch_end", {"epoch": epoch, **dict(fields)})

    def log_validation(self, epoch: int, scores: Mapping[str, Any], **extra: Any) -> None:
        row: Dict[str, Any] = {"epoch": epoch}
        for k, v in scores.items():
            row[f"val_{k}"] = v
        row.update(extra)
        self._write("validation", row)

    def log_event(self, event: str, **fields: Any) -> None:
        self._write(event, dict(fields))

    def log_run_end(self, **fields: Any) -> None:
        self._write("run_end", dict(fields))
