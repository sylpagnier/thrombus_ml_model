"""Clot baseline recipe and manifest I/O."""

from __future__ import annotations

import json
from pathlib import Path

from src.inference.clot_baseline_recipe import (
    ClotBaselineRecipe,
    default_lane_a_recipe,
    load_recipe_json,
    save_manifest,
)


def test_load_lane_a_reference_recipe():
    recipe = load_recipe_json()
    assert recipe.name == "lane_a"
    assert recipe.mu_ratio_max == 20.0
    assert recipe.pred_kine is True
    assert recipe.clot_phi_env.get("CLOT_PHI_HIDDEN") == "32"


def test_manifest_roundtrip(tmp_path: Path):
    recipe = default_lane_a_recipe()
    recipe.teacher_ckpt = "outputs/biochem/clot_baseline/teacher_best_high_mu.pth"
    path = tmp_path / "manifest.json"
    save_manifest(recipe, eval_summary={"min_f1": 0.59}, path=path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["recipe"]["name"] == recipe.name
    assert raw["eval"]["min_f1"] == 0.59


def test_recipe_resolved_paths():
    recipe = ClotBaselineRecipe(dump_anchor_dir="outputs/biochem/foo")
    paths = recipe.resolved_paths(Path("/repo"))
    assert paths["dump_anchor_dir"] == Path("/repo/outputs/biochem/foo")
