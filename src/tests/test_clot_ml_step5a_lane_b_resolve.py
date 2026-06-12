"""Step 5a lane_b manifest -> clot-phi ckpt resolution."""

from __future__ import annotations

from pathlib import Path

from src.training.clot_ml_step5a_mu_readout import resolve_lane_b_clot_phi_ckpt
from src.utils.paths import get_project_root


def test_resolve_lane_b_from_manifest_json():
    manifest = get_project_root() / "outputs/biochem/clot_baseline/manifest.json"
    if not manifest.is_file():
        return
    ckpt = resolve_lane_b_clot_phi_ckpt(str(manifest))
    assert ckpt is not None
    assert ckpt.suffix.lower() == ".pth"
    assert ckpt.is_file()


def test_resolve_lane_b_from_pth_path():
    manifest = get_project_root() / "outputs/biochem/clot_baseline/manifest.json"
    if not manifest.is_file():
        return
    from src.inference.clot_baseline_recipe import load_manifest

    recipe, _ = load_manifest(manifest)
    pth = recipe.resolved_paths().get("clot_phi_ckpt")
    if pth is None or not pth.is_file():
        return
    ckpt = resolve_lane_b_clot_phi_ckpt(str(pth))
    assert ckpt == Path(pth)
