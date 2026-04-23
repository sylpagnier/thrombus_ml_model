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

from src.utils.paths import reports_training_dir


def write_t1_experiment_artifact(
    tier1_cfg: Any,
    *,
    best_rel_l2: float,
    best_val_composite_loss: float,
    best_loss: float,
    early_stopped: bool,
    n_graphs: int,
    n_train: int,
    n_val: int,
    graph_dir: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``reports/experiments/tier1_<name>_<ts>.json`` for post-run comparison.

    ``tier1_cfg`` must provide ``experiment_name`` and ``to_serializable()`` (e.g. ``Tier1TrainConfig``).
    ``best_val_composite_loss`` is the minimum validation composite (Tier 1:
    ``rel_l2_anchor + 100×continuity``); lower is better.
    """
    rep = reports_training_dir("tier1", "experiments")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = str(getattr(tier1_cfg, "experiment_name", "default"))
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]
    path = rep / f"tier1_{safe_name}_{ts}.json"
    ser = tier1_cfg.to_serializable() if callable(getattr(tier1_cfg, "to_serializable", None)) else {}
    payload: Dict[str, Any] = {
        "tier": "tier1",
        "ts_utc": ts,
        "tier1_train_config": ser,
        "metrics": {
            "best_rel_l2": best_rel_l2,
            "best_val_composite_loss": best_val_composite_loss,
            "best_loss": best_loss,
            "early_stopped": early_stopped,
        },
        "data": {
            "n_graphs": n_graphs,
            "n_train": n_train,
            "n_val": n_val,
            "graph_dir": graph_dir,
        },
        "env_tier1": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TIER1_")},
    }
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📒 Experiment artifact: {path}")
    print("🧾 History reminder: append this run's key metrics to your Tier1 training history log.")
    return path


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
    """Writes one main JSONL diary per run under a run-specific folder."""

    def __init__(self, tier: str, enabled: Optional[bool] = None):
        self.tier = tier
        if enabled is None:
            raw = os.environ.get("PHASE1_TRAINING_DIARY", "1").strip().lower()
            enabled = raw not in ("0", "false", "no", "off")
        self.enabled = enabled
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        if not self.enabled:
            return
        reports = reports_training_dir(self.tier)
        self.run_dir = reports / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        custom = os.environ.get("PHASE1_TRAINING_DIARY_PATH", "").strip()
        if custom:
            self.path = Path(custom)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.run_dir = self.path.parent
        else:
            self.path = self.run_dir / "training_diary_main.jsonl"
        os.environ["PHASE1_TRAINING_RUN_DIR"] = str(self.run_dir)
        print(f"📝 Training diary (main JSONL): {self.path}")
        print(f"📂 Training run folder: {self.run_dir}")

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
