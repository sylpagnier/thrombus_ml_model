"""One-off tune rung-2 gamma proxy (no comsol_sr)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_mu_physics import eval_anchor_t0_mu

graph = Path("data/processed/graphs_biochem_anchors/patient007.pt")
rows = []
best = None
for gmode in ("max", "kinematic", "poiseuille", "max_kinematic", "graph"):
    poi_list = (None, 0.75, 0.85, 1.0) if gmode in ("max", "poiseuille") else (None,)
    for gscale in (0.7, 0.85, 1.0, 1.15, 1.3):
        for poi in poi_list:
            os.environ.pop("CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR", None)
            os.environ["CLOT_PHI_PHYSICS_GAMMA_SCALE"] = f"{gscale:g}"
            if poi is not None:
                os.environ["CLOT_PHI_PHYSICS_POISEUILLE_SCALE"] = f"{poi:g}"
            else:
                os.environ.pop("CLOT_PHI_PHYSICS_POISEUILLE_SCALE", None)
            r = eval_anchor_t0_mu(graph, times=[0, 53], gamma_mode=gmode, hard_step=True)
            t0 = next(x for x in r.times if x["time"] == 0)
            t53 = next(x for x in r.times if x["time"] == 53)
            row = {
                "gmode": gmode,
                "gscale": gscale,
                "poi": poi,
                "bulk0": t0["ratio_median_bulk"],
                "growth53": t53["ratio_median_growth"],
                "r_growth53": t53["pearson_growth"],
                "r_all53": t53["pearson_all"],
                "logmae_g53": t53["mu_log_mae_growth"],
            }
            rows.append(row)
            score = abs(row["bulk0"] - 1.0) + abs(row["growth53"] - 1.0) * 0.5 + (1 - min(row["r_growth53"], 1.0))
            if best is None or score < best[0]:
                best = (score, row)

print("[OK] best", best[1], "score", best[0])
for score, row in sorted(
    (
        abs(r["bulk0"] - 1) + abs(r["growth53"] - 1) * 0.5 + (1 - min(r["r_growth53"], 1)),
        r,
    )
    for r in rows
)[:10]:
    print(f"  score={score:.3f} {row}")
