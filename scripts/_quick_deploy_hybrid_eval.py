"""One-off hybrid deploy config eval (3 anchors, fast smoke)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from scripts.run_mlp_clot_inject_probe import _eval_anchor, _load_teacher, _mean
from src.config import BiochemConfig
from src.inference.biochem_teacher_loader import resolve_rollout_mu_ratio_max
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.inference.deploy_mu_map_env import apply_deploy_mu_map_env, clear_oracle_mu_map_env


CONFIGS: list[tuple[str, dict[str, str]]] = [
    (
        "phiq92_only",
        {
            "BIOCHEM_MLP_DEPLOY_PHI_Q": "0.92",
            "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "0",
            "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "0",
        },
    ),
    (
        "growth_no_req",
        {
            "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
            "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "0",
        },
    ),
    (
        "phiq92_growth_no_req",
        {
            "BIOCHEM_MLP_DEPLOY_PHI_Q": "0.92",
            "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "1",
            "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "0",
        },
    ),
    (
        "legacy_excess02",
        {
            "BIOCHEM_MLP_DEPLOY_MU_EXCESS_SI": "0.02",
            "BIOCHEM_MLP_NEIGHBOR_GROWTH_ONLY": "0",
            "BIOCHEM_MLP_DEPLOY_REQUIRE_MLP_CLOTS": "0",
        },
    ),
]


def main() -> int:
    root = _REPO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    os.environ["BIOCHEM_ROLLOUT_PROGRESS"] = "0"
    ckpt = root / "outputs/biochem/clot_baseline/teacher_best_high_mu.pth"
    clot = root / "outputs/biochem/clot_baseline/clot_phi_best.pth"
    graph_dir = root / "data/processed/graphs_biochem_anchors"
    anchors = ["patient003", "patient007", "patient006"]
    mu_ratio = resolve_rollout_mu_ratio_max(BiochemConfig(phase="biochem"), cli_value=20.0)
    teacher, phys, bio = _load_teacher(ckpt, device, mu_ratio, fast=True)

    for name, overrides in CONFIGS:
        clear_oracle_mu_map_env()
        for k in list(os.environ.keys()):
            if k.startswith("BIOCHEM_MLP_") or k.startswith("BIOCHEM_MU_NEIGHBOR"):
                os.environ.pop(k, None)
        apply_deploy_mu_map_env(overrides)
        teacher.clear_clot_phi_injector()
        attach_clot_phi_injector_to_teacher(teacher, device, str(clot))
        rows = [
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
        per = {r["anchor"]: round(r.get("clot_shape") or 0.0, 3) for r in rows}
        shape = _mean(rows, "clot_shape") or 0.0
        fp = _mean(rows, "clot_fp_distant") or 0.0
        print(
            f"  {name:22s} mean={shape:.3f} fp_dist={fp:.0f} "
            f"p003={per.get('patient003')} p007={per.get('patient007')} p006={per.get('patient006')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
