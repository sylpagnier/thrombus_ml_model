"""
Append-only JSONL training diary for Kinematics runs — machine- and human-readable context for
post-hoc analysis (e.g. planning the next run).

Each line is one JSON object: ``ts_utc``, ``run_id``, ``phase``, ``event``, plus event fields.

Disable with ``KINEMATICS_TRAINING_DIARY=0`` or set ``KINEMATICS_TRAINING_DIARY_PATH`` to a custom ``.jsonl`` file.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.paths import reports_training_dir


def write_t1_experiment_artifact(
    kinematics_cfg: Any,
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
    """Write ``reports/experiments/kinematics_<name>_<ts>.json`` for post-run comparison.

    ``kinematics_cfg`` must provide ``experiment_name`` and ``to_serializable()`` (e.g. ``Phase1TrainConfig``).
    ``best_val_composite_loss`` is the minimum validation composite (Kinematics:
    ``rel_l2_anchor + 100×continuity``); lower is better.
    """
    rep = reports_training_dir("kinematics", "experiments")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = str(getattr(kinematics_cfg, "experiment_name", "default"))
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:80]
    path = rep / f"kinematics_{safe_name}_{ts}.json"
    ser = kinematics_cfg.to_serializable() if callable(getattr(kinematics_cfg, "to_serializable", None)) else {}
    payload: Dict[str, Any] = {
        "phase": "kinematics",
        "ts_utc": ts,
        "kinematics_train_config": ser,
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
        "env_kinematics": {k: v for k, v in sorted(os.environ.items()) if k.startswith("KINEMATICS_")},
    }
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📒 Experiment artifact: {path}")
    print("🧾 History reminder: append this run's key metrics to your Phase1 training history log.")
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


def prune_training_diary_runs(base_dir: Path, *, keep: int = 5, current_run_id: str | None = None) -> int:
    """Keep only the newest ``keep`` run folders that contain ``training_diary_main.jsonl``."""
    base = Path(base_dir)
    if keep < 1 or not base.exists():
        return 0
    candidates = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if current_run_id and child.name == current_run_id:
            continue
        diary_path = child / "training_diary_main.jsonl"
        if diary_path.exists():
            candidates.append(child)
    candidates.sort(key=lambda p: p.name, reverse=True)
    keep_old = max(0, keep - 1) if current_run_id else keep
    removed = 0
    for old_run in candidates[keep_old:]:
        try:
            shutil.rmtree(old_run)
            removed += 1
        except OSError:
            pass
    return removed


class TrainingDiary:
    """Writes one main JSONL diary per run under a run-specific folder."""

    def __init__(self, phase: str, enabled: Optional[bool] = None):
        self.phase = phase
        if enabled is None:
            raw = os.environ.get("KINEMATICS_TRAINING_DIARY", "1").strip().lower()
            enabled = raw not in ("0", "false", "no", "off")
        self.enabled = enabled
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        if not self.enabled:
            return
        reports = reports_training_dir(self.phase)
        self.run_dir = reports / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        custom = os.environ.get("KINEMATICS_TRAINING_DIARY_PATH", "").strip()
        if custom:
            self.path = Path(custom)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.run_dir = self.path.parent
        else:
            self.path = self.run_dir / "training_diary_main.jsonl"
            prune_training_diary_runs(reports, keep=5, current_run_id=self.run_id)
        os.environ["KINEMATICS_TRAINING_RUN_DIR"] = str(self.run_dir)
        print(f"📝 Training diary (main JSONL): {self.path}")
        print(f"📂 Training run folder: {self.run_dir}")

    def _write(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        row: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "phase": self.phase,
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
