"""
Training run logs for kinematics (verbose diary) and biochem (compact run log).

Kinematics: append-only ``training_diary_main.jsonl`` per run (disable with ``KINEMATICS_TRAINING_DIARY=0``).

Biochem: one ``run.jsonl`` per run (``meta`` / ``val`` / ``end`` events) plus a global ``runs_index.jsonl``
for cross-run comparison. Disable with ``BIOCHEM_TRAINING_LOG=0``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.paths import reports_training_dir

# Env keys copied into biochem ``meta`` / index rows (comparison-relevant knobs only).
_BIOCHEM_RUN_LOG_ENV_KEYS: tuple[str, ...] = (
    "BIOCHEM_RUN_NOTE",
    "BIOCHEM_PRESET",
    "BIOCHEM_LOSS_ISOLATE",
    "BIOCHEM_LOSS_DATA_ONLY",
    "BIOCHEM_COMPLEXITY_STEP",
    "BIOCHEM_TEACHER_EPOCHS",
    "BIOCHEM_EPOCHS",
    "BIOCHEM_STOP_AFTER_TEACHER",
    "BIOCHEM_MU_WALL_BYPASS_WEIGHT",
    "BIOCHEM_WALL_GATE_CURRICULUM_EPOCHS",
    "BIOCHEM_MU_WALL_GATE_POS_INIT",
    "BIOCHEM_MU_WALL_MIX_MODE",
    "BIOCHEM_MU_WALL_HEAD_ACTIVATION",
    "BIOCHEM_TBPTT_MAX_WINDOW",
    "BIOCHEM_DETACH_MACRO_STATE",
    "BIOCHEM_VAL_TIME_STRIDE",
    "BIOCHEM_TEACHER_FORCE_MIN",
    "BIOCHEM_TEACHER_MU_RATIO_MAX",
    "BIOCHEM_TEACHER_VAL_EVERY",
    "BIOCHEM_USE_SPLIT_MU_HEAD",
    "BIOCHEM_USE_DELTA_MU_HEAD",
    "BIOCHEM_MU_LOG_WALL_WEIGHT",
    "BIOCHEM_MU_LOG_HIGH_WEIGHT",
    "BIOCHEM_INIT_FROM_BEST",
    "BIOCHEM_SKIP_PRETRAIN",
)


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
    """Write ``reports/experiments/kinematics_<name>_<ts>.json`` for post-run comparison."""
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


def biochem_env_digest() -> Dict[str, str]:
    """Subset of ``BIOCHEM_*`` env vars useful when comparing runs."""
    return {k: os.environ[k] for k in _BIOCHEM_RUN_LOG_ENV_KEYS if k in os.environ}


def _prune_run_folders(
    base_dir: Path,
    *,
    marker_name: str,
    keep: int,
    current_run_id: str | None = None,
) -> int:
    base = Path(base_dir)
    if keep < 1 or not base.exists():
        return 0
    candidates = []
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if current_run_id and child.name == current_run_id:
            continue
        if (child / marker_name).exists():
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


def prune_training_diary_runs(base_dir: Path, *, keep: int = 5, current_run_id: str | None = None) -> int:
    """Keep only the newest ``keep`` kinematics run folders with ``training_diary_main.jsonl``."""
    return _prune_run_folders(
        base_dir,
        marker_name="training_diary_main.jsonl",
        keep=keep,
        current_run_id=current_run_id,
    )


class TrainingDiary:
    """Kinematics: verbose JSONL diary per run folder."""

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


class BiochemRunLogger:
    """Biochem: compact ``run.jsonl`` + global ``runs_index.jsonl`` for run comparison."""

    RUN_JSONL = "run.jsonl"
    INDEX_JSONL = "runs_index.jsonl"

    def __init__(self, enabled: Optional[bool] = None):
        if enabled is None:
            raw = os.environ.get("BIOCHEM_TRAINING_LOG", "1").strip().lower()
            enabled = raw not in ("0", "false", "no", "off")
        self.enabled = enabled
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path: Optional[Path] = None
        self.run_dir: Optional[Path] = None
        self._best_val: Dict[str, Any] = {}
        self._run_note: str = ""
        if not self.enabled:
            return
        reports = reports_training_dir("biochem")
        try:
            keep = max(1, int(os.environ.get("BIOCHEM_TRAINING_LOG_KEEP", "20")))
        except ValueError:
            keep = 20
        self.run_dir = reports / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / self.RUN_JSONL
        _prune_run_folders(reports, marker_name=self.RUN_JSONL, keep=keep, current_run_id=self.run_id)
        os.environ["KINEMATICS_TRAINING_RUN_DIR"] = str(self.run_dir)
        print(f"📊 Biochem run log: {self.path}")
        index_path = reports / self.INDEX_JSONL
        print(f"📋 Run index (append on end): {index_path}")

    def _append(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        row: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
        }
        row.update(_json_safe(payload))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_meta(self, **fields: Any) -> None:
        env = biochem_env_digest()
        run_note = (fields.pop("run_note", None) or env.get("BIOCHEM_RUN_NOTE") or "").strip()
        self._run_note = run_note
        self._append(
            "meta",
            {
                "host": socket.gethostname(),
                "run_note": run_note,
                "env": env,
                **dict(fields),
            },
        )

    def log_val(self, stage: str, epoch: int, metrics: Mapping[str, Any], **extra: Any) -> None:
        """Record one validation snapshot (teacher or corrector)."""
        row: Dict[str, Any] = {"stage": stage, "epoch": int(epoch)}
        for k, v in metrics.items():
            key = k if k.startswith("val_") else f"val_{k}"
            row[key] = v
        row.update(extra)
        self._append("val", row)
        score = row.get("val_mu_log_mae")
        if score is None:
            score = row.get("val_avg_mu_log_mae")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            prev = self._best_val.get("val_mu_log_mae")
            if prev is None or float(score) < float(prev):
                self._best_val = {k: row[k] for k in row if k.startswith("val_")}
                self._best_val["best_epoch"] = int(epoch)
                self._best_val["stage"] = stage

    def log_end(self, **fields: Any) -> None:
        if not self.enabled:
            return
        end_row = dict(fields)
        if self._best_val:
            end_row.setdefault("best_val", self._best_val)
        self._append("end", end_row)
        if self.run_dir is None:
            return
        index_on = (os.environ.get("BIOCHEM_TRAINING_LOG_INDEX", "1") or "").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if not index_on:
            return
        index_path = reports_training_dir("biochem") / self.INDEX_JSONL
        env = biochem_env_digest()
        best = self._best_val or {}
        run_note = (
            (fields.get("run_note") or self._run_note or env.get("BIOCHEM_RUN_NOTE") or "").strip()
        )
        index_row: Dict[str, Any] = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "run_note": run_note,
            "host": socket.gethostname(),
            "env": env,
            "best_epoch": best.get("best_epoch"),
            "val_mu_log_mae": best.get("val_mu_log_mae") or best.get("val_avg_mu_log_mae"),
            "val_mu_log_mae_wall": best.get("val_mu_log_mae_wall") or best.get("val_avg_mu_log_mae_wall"),
            "val_mu_log_mae_high_mu": best.get("val_mu_log_mae_high_mu")
            or best.get("val_avg_mu_log_mae_high_mu"),
            "val_mu_pearson": best.get("val_mu_pearson") or best.get("val_avg_mu_pearson"),
            "teacher_best_mu_score": fields.get("teacher_best_mu_score"),
            "best_composite": fields.get("best_composite"),
            "interrupted": bool(fields.get("interrupted", False)),
            "checkpoint_teacher": fields.get("checkpoint_teacher"),
            "checkpoint_high_mu": fields.get("checkpoint_high_mu") or fields.get("checkpoint_bio"),
        }
        index_row = _json_safe(index_row)
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(index_row, ensure_ascii=False) + "\n")
        except OSError:
            pass
