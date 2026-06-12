"""Quick grid sweep for Rung4 s0 rule knobs (CUDA).

Uses deploy-faithful ``eval_rung4_step_clot`` (nucleation-projected phi + health),
same path as ``eval_t0_rung4_step.py``.

Grids:
  legacy  - original top_frac x tau_end x gain (24 configs)
  g1      - tau_start x tau_end x top_frac x gain (default; ~180 configs)
  spread  - spread_hops x spread_decay at code defaults for other knobs
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_rung4_ladder import (  # noqa: E402
    S0_RULE_ENV_KEYS,
    eval_rung4_step_clot,
)
from src.utils.paths import get_project_root  # noqa: E402


@contextmanager
def _s0_env(cfg: dict[str, str]):
    saved = {k: os.environ.get(k) for k in S0_RULE_ENV_KEYS}
    for k, v in cfg.items():
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _grid_legacy() -> list[dict[str, str]]:
    top_fracs = ["0.08", "0.10", "0.12", "0.14"]
    tau_ends = ["0.30", "0.35", "0.38"]
    gains = ["1.15", "1.20"]
    out: list[dict[str, str]] = []
    for tf, te, g in itertools.product(top_fracs, tau_ends, gains):
        out.append({
            "T0_R4_S0_SPATIAL_TOP_FRAC": tf,
            "T0_R4_S0_ONSET_TAU_END": te,
            "T0_R4_S0_FI_MAT_GAIN": g,
        })
    return out


def _grid_g1() -> list[dict[str, str]]:
    tau_starts = ["0.04", "0.06", "0.08"]
    tau_ends = ["0.26", "0.30", "0.34", "0.38"]
    top_fracs = ["0.06", "0.08", "0.10", "0.12", "0.14"]
    gains = ["1.10", "1.15", "1.20"]
    out: list[dict[str, str]] = []
    for ts, te, tf, g in itertools.product(tau_starts, tau_ends, top_fracs, gains):
        out.append({
            "T0_R4_S0_ONSET_TAU_START": ts,
            "T0_R4_S0_ONSET_TAU_END": te,
            "T0_R4_S0_SPATIAL_TOP_FRAC": tf,
            "T0_R4_S0_FI_MAT_GAIN": g,
        })
    return out


def _grid_spread() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for hops, decay in itertools.product(["0", "1", "2"], ["0.0", "0.5", "0.85"]):
        if hops == "0" and decay != "0.0":
            continue
        out.append({
            "T0_R4_S0_SPREAD_HOPS": hops,
            "T0_R4_S0_SPREAD_DECAY": decay,
        })
    return out


def _pick_grid(name: str) -> list[dict[str, str]]:
    key = (name or "g1").strip().lower()
    if key == "legacy":
        return _grid_legacy()
    if key == "spread":
        return _grid_spread()
    if key == "g1":
        return _grid_g1()
    raise ValueError(f"unknown grid {name!r}; use legacy, g1, or spread")


def _eval_cfg(
    data,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    anchor: str,
    times: list[int],
    cfg: dict[str, str],
) -> dict:
    with _s0_env(cfg):
        ev = eval_rung4_step_clot(data, phys, bio, device, step="s0", times=times)
    return {
        "anchor": anchor,
        "cfg": cfg,
        "clot": ev["clot"],
        "rollout_health": ev["rollout_health"],
    }


def _score_row(row: dict, times: list[int]) -> float:
    t_last = times[-1]
    c53 = next(r for r in row["clot"] if r["time"] == t_last)
    c27 = next((r for r in row["clot"] if r["time"] == 27), c53)
    health = float(row.get("rollout_health", {}).get("health_score", 0.0))
    # balance final F1 vs not starting too early; prefer healthy rollouts
    return (
        float(c53["clot_f1"])
        - 0.15 * max(0.0, float(c27["clot_f1"]) - 0.35)
        + 0.05 * health
    )


def _fingerprint(row: dict, t_last: int) -> str:
    c = next(r for r in row["clot"] if r["time"] == t_last)
    h = row.get("rollout_health", {})
    return (
        f"{c['clot_f1']:.6f}|{c['pred_pos_frac']:.8f}|"
        f"{h.get('health_pass')}|{h.get('wall_carpet')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default="patient007,patient002")
    ap.add_argument("--times", default="27,53")
    ap.add_argument("--grid", default="g1", choices=("legacy", "g1", "spread"))
    ap.add_argument("--out", default="outputs/biochem/t0_r4_sstar/rules/s0_sweep.json")
    args = ap.parse_args()

    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    grid = _pick_grid(args.grid)

    device = require_cuda_device()
    root = get_project_root()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    results: list[dict] = []
    for anchor in anchors:
        graph = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
        data = torch.load(graph, map_location=device, weights_only=False)
        for cfg in grid:
            print(f"[i] {anchor} {cfg}", flush=True)
            results.append(_eval_cfg(data, phys, bio, device, anchor, times, cfg))

    t_last = times[-1]
    ranked = sorted(results, key=lambda r: _score_row(r, times), reverse=True)
    fps = {_fingerprint(r, t_last) for r in results}
    flat_plateau = len(fps) == 1

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": args.grid,
        "times": times,
        "n_configs": len(grid),
        "flat_plateau": flat_plateau,
        "unique_fingerprints": len(fps),
        "ranked": ranked[:12],
        "all": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    winner_path = out.parent / "s0_winner_env.json"
    if ranked:
        winner_path.write_text(json.dumps(ranked[0]["cfg"], indent=2), encoding="utf-8")

    print(f"[OK] {out}")
    if flat_plateau:
        print(
            "[WARN] flat plateau: all configs tie on deploy phi F1 @ "
            f"t={t_last} (unique={len(fps)}); proceed to s*_G4 gate ML",
            flush=True,
        )
    for row in ranked[:5]:
        c53 = next(r for r in row["clot"] if r["time"] == t_last)
        hp = row.get("rollout_health", {})
        print(
            f"  {row['anchor']} F1@53={c53['clot_f1']:.3f} "
            f"health={hp.get('health_pass')} cfg={row['cfg']}",
            flush=True,
        )
    if ranked:
        print(f"[i] winner env -> {winner_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
