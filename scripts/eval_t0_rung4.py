"""Rung 4 only: GT flow + pred teacher species vs Rung 2 baseline.

Usage::

    python scripts/eval_t0_rung4.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_t0_rung1234 import (  # noqa: E402
    _clot_timeline,
    _mu_timeline_rung4,
    _species_log_mae,
)
from src.core_physics.t0_mu_physics import eval_anchor_t0_mu  # noqa: E402
from src.core_physics.t0_rung_config import (  # noqa: E402
    DEFAULT_SPECIES_DUMP_DIR,
    RUNG2_GAMMA_MODE,
    resolve_default_teacher_ckpt,
    rollout_t0_pred_species_series,
    t0_rung2_env,
    t0_rung4_env,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung 4 species isolation eval")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,27,53")
    ap.add_argument("--teacher-ckpt", default="")
    ap.add_argument("--species-dump", default="")
    ap.add_argument("--teacher-time-stride", type=int, default=6)
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_rung4_eval.json")
    args = ap.parse_args()

    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    teacher = Path(args.teacher_ckpt) if args.teacher_ckpt.strip() else Path(resolve_default_teacher_ckpt())
    if not teacher.is_absolute():
        teacher = root / teacher
    if not graph.is_file() or not teacher.is_file():
        print(f"[ERR] missing graph or teacher", file=sys.stderr)
        return 1

    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(graph, map_location=device, weights_only=False)

    species_dump = Path(args.species_dump) if args.species_dump.strip() else (
        root / DEFAULT_SPECIES_DUMP_DIR / f"{args.anchor}.pt"
    )
    if not species_dump.is_absolute():
        species_dump = root / species_dump

    print(f"[i] teacher: {teacher}", flush=True)
    t_roll = time.perf_counter()
    pred_species = rollout_t0_pred_species_series(
        data,
        str(teacher),
        device,
        bio_cfg=bio,
        dumped_graph=str(species_dump) if species_dump.is_file() else None,
        time_stride=max(int(args.teacher_time_stride), 1),
    )
    print(f"[i] species ready in {time.perf_counter() - t_roll:.1f}s", flush=True)

    with t0_rung2_env():
        rung2_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode=RUNG2_GAMMA_MODE)
    rung4_mu = _mu_timeline_rung4(
        data, phys, bio, device, times, pred_species_series=pred_species, teacher_ckpt=str(teacher)
    )
    clot2 = _clot_timeline(
        data, phys, bio, device, times, gamma_mode=RUNG2_GAMMA_MODE, env_ctx=t0_rung2_env()
    )
    clot4 = _clot_timeline(
        data,
        phys,
        bio,
        device,
        times,
        gamma_mode=RUNG2_GAMMA_MODE,
        pred_species_series=pred_species,
        env_ctx=t0_rung4_env(teacher_ckpt=str(teacher)),
    )
    species_mae = _species_log_mae(pred_species, data, times, device)

    t_last = times[-1]
    c2 = next(r for r in clot2 if r["time"] == t_last)
    c4 = next(r for r in clot4 if r["time"] == t_last)
    sp = next(r for r in species_mae if r["time"] == t_last)

    gates = {
        "rung2_clot_f1_nuc_t_last": float(c2["clot_f1"]) >= 0.85,
        "rung4_species_log_mae_t_last": float(sp["species_log_mae"]) < 2.0,
        "rung4_clot_f1_nuc_t_last": float(c4["clot_f1"]) >= 0.35,
        "rung4_clot_f1_within_0p25_of_rung2": abs(float(c2["clot_f1"]) - float(c4["clot_f1"])) <= 0.25,
    }
    payload = {
        "anchor": args.anchor,
        "teacher_ckpt": str(teacher),
        "species_source": "dump" if species_dump.is_file() else f"live_stride_{int(args.teacher_time_stride)}",
        "rung2": {"mu": rung2_mu.to_dict(), "clot_nucleation": clot2},
        "rung4": {
            "mu": {"times": rung4_mu},
            "clot_nucleation": clot4,
            "species_log_mae": species_mae,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {out}")
    print(f"[i] R2 F1={c2['clot_f1']:.3f} R4 F1={c4['clot_f1']:.3f} species_mae={sp['species_log_mae']:.4f}")
    print(f"[i] gates={gates} all_pass={payload['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
