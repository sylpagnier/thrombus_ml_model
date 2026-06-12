"""Step 5c: LOAO deploy v1 frozen vs coupled closed-loop eval."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from src.inference.clot_ml_deploy_v1 import DeployV1Recipe, eval_deploy_v1_on_graph, load_deploy_v1_recipe
from src.utils.paths import get_project_root


@torch.no_grad()
def eval_loao_deploy_v1(
    recipe: DeployV1Recipe,
    *,
    device: torch.device,
    anchor_dir: Path | None = None,
    coupled: bool = False,
    sim_end_scale: float = 1.0,
) -> dict[str, Any]:
    root = get_project_root()
    adir = anchor_dir or (root / "data/processed/graphs_biochem_anchors")
    paths = sorted(adir.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors under {adir}")

    per_anchor: list[dict[str, Any]] = []
    for p in paths:
        data = torch.load(p, map_location=device, weights_only=False)
        row = eval_deploy_v1_on_graph(
            data,
            recipe,
            device=device,
            sim_end_scale=sim_end_scale,
            coupled=coupled,
            anchor=p.stem,
        )
        per_anchor.append(row)

    mean_deploy = sum(r["deploy_score"] for r in per_anchor) / len(per_anchor)
    return {
        "step": "5c",
        "coupled": coupled,
        "sim_end_scale": sim_end_scale,
        "mean_deploy": mean_deploy,
        "per_anchor": per_anchor,
        "recipe": recipe.phi_shell,
    }


@torch.no_grad()
def compare_frozen_vs_coupled(
    recipe_path: str | Path | None = None,
    *,
    device: torch.device,
    sim_end_scale: float = 1.0,
    deploy_drop_tol: float = 0.02,
) -> dict[str, Any]:
    """5c gate: coupled LOAO within deploy_drop_tol of frozen baseline."""
    recipe = load_deploy_v1_recipe(recipe_path)
    frozen = eval_loao_deploy_v1(
        recipe, device=device, coupled=False, sim_end_scale=sim_end_scale
    )
    coupled_recipe = replace(recipe, coupled=True)
    coupled = eval_loao_deploy_v1(
        coupled_recipe, device=device, coupled=True, sim_end_scale=sim_end_scale
    )
    delta = coupled["mean_deploy"] - frozen["mean_deploy"]
    # Pass if within tol OR coupled improves mean (feedback helps, not random drift).
    pass_5c = abs(delta) <= deploy_drop_tol or delta >= 0.0
    return {
        "frozen_mean_deploy": frozen["mean_deploy"],
        "coupled_mean_deploy": coupled["mean_deploy"],
        "mean_deploy_delta": delta,
        "pass_5c_gate": pass_5c,
        "deploy_drop_tol": deploy_drop_tol,
        "frozen": frozen,
        "coupled": coupled,
    }
