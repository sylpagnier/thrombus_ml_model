"""Fast sweep of deploy Leg B mask configs (clot_shape smoke)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.run_mlp_clot_inject_probe import (  # noqa: E402
    _eval_anchor,
    _load_teacher,
    _mean,
)
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.inference.deploy_mu_map_env import apply_deploy_mu_map_env, clear_oracle_mu_map_env
from src.inference.biochem_teacher_loader import resolve_rollout_mu_ratio_max
from src.config import BiochemConfig
from src.utils.paths import get_project_root
import torch


CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("legacy_wall_phi", {
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
        "BIOCHEM_MLP_DEPLOY_PHI_Q": "0",
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "0",
        "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "0",
    }),
    ("growth_only", {
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "1",
    }),
    ("growth_phiq92", {
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
        "BIOCHEM_MLP_DEPLOY_PHI_Q": "0.92",
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "1",
    }),
    ("growth_excess015", {
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
        "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0.015",
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "1",
    }),
    ("canonical_v2", {
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
        "BIOCHEM_MLP_DEPLOY_PHI_Q": "0.90",
        "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0.01",
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "1",
    }),
    ("dgamma_only", {
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "1",
        "BIOCHEM_MLP_DEPLOY_PHI_Q": "0",
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "0",
        "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "1",
    }),
    ("dgamma_phiq92", {
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "1",
        "BIOCHEM_MLP_DEPLOY_PHI_Q": "0.92",
        "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
        "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "1",
    }),
]


def main() -> int:
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    os.environ["BIOCHEM_ROLLOUT_PROGRESS"] = "0"
    ckpt = root / "outputs/biochem/clot_baseline/teacher_best_high_mu.pth"
    clot = root / "outputs/biochem/clot_baseline/clot_phi_best.pth"
    graph_dir = root / "data/processed/graphs_biochem_anchors"
    anchors = ["patient003", "patient007", "patient006"]
    if not ckpt.is_file() or not clot.is_file():
        print("[ERR] missing baseline ckpts", file=sys.stderr)
        return 1

    mu_ratio = resolve_rollout_mu_ratio_max(BiochemConfig(phase="biochem"), cli_value=20.0)
    teacher, phys, bio = _load_teacher(ckpt, device, mu_ratio, fast=True)
    rows = []
    for name, overrides in CONFIGS:
        clear_oracle_mu_map_env()
        for k in list(os.environ.keys()):
            if k.startswith("BIOCHEM_MLP_") or k.startswith("BIOCHEM_MU_NEIGHBOR"):
                os.environ.pop(k, None)
        apply_deploy_mu_map_env(overrides)
        teacher.clear_clot_phi_injector()
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot))
        leg_rows = [
            _eval_anchor(
                teacher,
                "B_deploy",
                a,
                graph_dir,
                device,
                bio,
                phys,
                time_stride=5,
                fast=True,
            )
            for a in anchors
        ]
        shape = _mean(leg_rows, "clot_shape")
        recall = _mean(leg_rows, "clot_recall")
        fp_dist = _mean(leg_rows, "clot_fp_distant")
        flow = _mean(leg_rows, "flow_score")
        n_trig = _mean(leg_rows, "inject_n_region")
        row = {
            "name": name,
            "clot_shape": shape,
            "recall": recall,
            "clot_fp_distant": fp_dist,
            "flow_score": flow,
            "inject_n_region": n_trig,
            "overrides": overrides,
        }
        rows.append(row)
        print(
            f"  {name:18s} shape={shape or 0:.3f} recall={recall or 0:.3f} "
            f"fp_dist={fp_dist or 0:.0f} n_trig={n_trig or 0:.0f}",
            flush=True,
        )

    rows.sort(key=lambda r: (r.get("clot_shape") or -1.0), reverse=True)
    out = root / "outputs/biochem/mlp_clot_inject_probe/b_deploy_sweep_fast.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ranked": rows}, indent=2), encoding="utf-8")
    best = rows[0]
    print(f"[OK]  best={best['name']} clot_shape={best.get('clot_shape')} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
