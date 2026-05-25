"""Geometry-level helpers for kinematics training (L0/L1/L2 curriculum)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.utils.anchor_mask import graph_has_anchor

_VESSEL_STEM_RE = re.compile(r"^vessel_(\d+)$", re.IGNORECASE)


def vessel_index_from_stem(stem: str) -> Optional[int]:
    m = _VESSEL_STEM_RE.match(str(stem).strip())
    return int(m.group(1)) if m else None


def read_geometry_level_from_mesh_json(mesh_input_dir: Path, stem: str) -> Optional[int]:
    """Read ``level`` from ``vessel_<id>.json`` written by ``vessel_generator``."""
    json_path = mesh_input_dir / f"{stem}.json"
    if not json_path.is_file():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        level = meta.get("level")
        if level is None:
            return None
        return int(level)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def graph_geometry_level(data: Any, *, default: int = -1) -> int:
    """Return geometry level 0/1/2 from graph attrs, else ``default`` (-1 = unknown)."""
    if hasattr(data, "geometry_level"):
        raw = getattr(data, "geometry_level")
        if torch.is_tensor(raw):
            return int(raw.view(-1)[0].item())
        return int(raw)
    return int(default)


def attach_geometry_metadata(
    data: Any,
    *,
    mesh_input_dir: Path,
    stem: Optional[str] = None,
) -> Any:
    """Set ``geometry_level`` and ``config_id`` on a graph (in-place)."""
    if stem is None:
        stem = str(getattr(data, "graph_stem", "") or "")
    idx = vessel_index_from_stem(stem)
    if idx is not None:
        data.config_id = int(idx)
    if graph_geometry_level(data, default=-1) >= 0:
        return data
    level = None
    if stem:
        level = read_geometry_level_from_mesh_json(mesh_input_dir, stem)
    if level is None:
        data.geometry_level = torch.tensor([-1], dtype=torch.int8)
    else:
        data.geometry_level = torch.tensor([int(level)], dtype=torch.int8)
    return data


@dataclass(frozen=True)
class GeometryCurriculumConfig:
    """Per-epoch sampling weights over geometry levels 0/1/2."""

    enabled: bool = True
    # foundation | ramp | l2_heavy | off
    phase: str = "auto"
    foundation_mix: Tuple[float, float, float] = (0.45, 0.45, 0.10)
    ramp_end_mix: Tuple[float, float, float] = (0.30, 0.30, 0.40)
    l2_heavy_mix: Tuple[float, float, float] = (0.15, 0.15, 0.70)
    hard_mining_start_epoch: int = 16
    hard_mining_interval: int = 4

    def resolved_phase(self, epoch: int, stage: int, stage1_end: int, stage2_end: int) -> str:
        if not self.enabled or self.phase == "off":
            return "off"
        if self.phase != "auto":
            return self.phase
        if stage == 1:
            return "foundation"
        if stage == 2:
            return "ramp"
        return "l2_heavy"

    def level_weights(
        self,
        epoch: int,
        stage: int,
        *,
        stage1_end: int,
        stage2_end: int,
    ) -> Dict[int, float]:
        phase = self.resolved_phase(epoch, stage, stage1_end, stage2_end)
        if phase == "off":
            return {0: 1.0, 1: 1.0, 2: 1.0}
        if phase == "foundation":
            return _mix_to_dict(self.foundation_mix)
        if phase == "l2_heavy":
            return _mix_to_dict(self.l2_heavy_mix)
        # ramp: linear blend foundation -> ramp_end over stage-2 window
        t0, t1 = int(stage1_end), int(stage2_end)
        alpha = 0.0 if t1 <= t0 else float(max(0.0, min(1.0, (epoch - t0) / (t1 - t0))))
        f = self.foundation_mix
        r = self.ramp_end_mix
        mix = tuple((1.0 - alpha) * f[i] + alpha * r[i] for i in range(3))
        return _mix_to_dict(mix)

    def describe(
        self,
        epoch: int,
        stage: int,
        *,
        stage1_end: int,
        stage2_end: int,
    ) -> str:
        phase = self.resolved_phase(epoch, stage, stage1_end, stage2_end)
        w = self.level_weights(epoch, stage, stage1_end=stage1_end, stage2_end=stage2_end)
        return (
            f"geometry={phase} "
            f"(L0={w[0]:.2f}, L1={w[1]:.2f}, L2={w[2]:.2f})"
        )


def _mix_to_dict(mix: Tuple[float, float, float]) -> Dict[int, float]:
    total = float(sum(mix))
    if total <= 0:
        return {0: 1.0, 1: 1.0, 2: 1.0}
    return {0: mix[0] / total, 1: mix[1] / total, 2: mix[2] / total}


def geometry_sample_weight(
    data: Any,
    level_weights: Dict[int, float],
    *,
    unknown_weight: float = 1.0,
) -> float:
    lvl = graph_geometry_level(data, default=-1)
    if lvl not in level_weights:
        return float(unknown_weight)
    return float(level_weights[lvl])


def cohort_level_counts(dataset: Sequence[Any]) -> Dict[int, int]:
    counts: Dict[int, int] = {0: 0, 1: 0, 2: 0, -1: 0}
    for d in dataset:
        lvl = graph_geometry_level(d, default=-1)
        counts[lvl] = counts.get(lvl, 0) + 1
    return counts


def warn_if_single_level_cohort(
    dataset: Sequence[Any],
    *,
    curriculum: GeometryCurriculumConfig,
    epoch: int,
    stage: int,
    stage1_end: int,
    stage2_end: int,
) -> None:
    if not curriculum.enabled:
        return
    phase = curriculum.resolved_phase(epoch, stage, stage1_end, stage2_end)
    if phase == "off":
        return
    counts = cohort_level_counts(dataset)
    known = counts[0] + counts[1] + counts[2]
    if known == 0:
        print("⚠️ Geometry curriculum: no geometry_level on graphs — run backfill or re-graph from mesh JSON.")
        return
    if phase == "foundation" and counts[0] + counts[1] == 0:
        print(
            "⚠️ Geometry curriculum foundation needs L0/L1 graphs; cohort is L2-only. "
            "Regenerate mixed vessels (--mixed-levels) or disable curriculum."
        )
    if phase == "l2_heavy" and counts[2] == 0:
        print("⚠️ Geometry l2_heavy phase but no L2 graphs in dataset.")


def split_anchor_physics_stratified(
    dataset: Sequence[Any],
    *,
    seed: int = 42,
    train_ratio: float = 0.9,
    min_val_per_level: int = 1,
) -> Dict[str, Any]:
    """90/10 split with per-level val holdout when ``geometry_level`` is known."""
    import random

    by_level: Dict[int, List[Any]] = {0: [], 1: [], 2: [], -1: []}
    for d in dataset:
        lvl = graph_geometry_level(d, default=-1)
        by_level.setdefault(lvl, []).append(d)

    rng = random.Random(seed)
    train: List[Any] = []
    val: List[Any] = []
    n_anchors = 0
    n_physics = 0

    for lvl, graphs in by_level.items():
        if not graphs:
            continue
        anchors = [d for d in graphs if graph_has_anchor(d)]
        physics = [d for d in graphs if not graph_has_anchor(d)]
        rng.shuffle(anchors)
        rng.shuffle(physics)
        split_a = int(train_ratio * len(anchors))
        split_p = int(train_ratio * len(physics))
        train_a = anchors[:split_a]
        val_a = anchors[split_a:]
        train_p = physics[:split_p]
        val_p = physics[split_p:]
        if lvl >= 0 and len(val_a) + len(val_p) < min_val_per_level:
            # Ensure at least one val graph per known level when possible
            pool = val_a + val_p + train_a + train_p
            if len(pool) > min_val_per_level:
                move = pool[-1]
                if graph_has_anchor(move) and train_a:
                    train_a = train_a[:-1]
                    val_a = val_a + [move]
                elif train_p:
                    train_p = train_p[:-1]
                    val_p = val_p + [move]
        train.extend(train_a + train_p)
        val.extend(val_a + val_p)
        n_anchors += len(train_a)
        n_physics += len(train_p)

    rng.shuffle(train)
    rng.shuffle(val)
    return {
        "train": train,
        "val": val,
        "n_anchors": n_anchors,
        "n_physics": n_physics,
    }


__all__ = [
    "GeometryCurriculumConfig",
    "attach_geometry_metadata",
    "cohort_level_counts",
    "geometry_sample_weight",
    "graph_geometry_level",
    "read_geometry_level_from_mesh_json",
    "split_anchor_physics_stratified",
    "vessel_index_from_stem",
    "warn_if_single_level_cohort",
]
