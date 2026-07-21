"""Promote deploy Leg B baseline (physics neighbor mask, no GT in forward)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.promote_clot_baseline import _copy_if_exists, _load_eval_summary  # noqa: E402
from src.inference.clot_baseline_recipe import baseline_dir, load_recipe_json, save_manifest
from src.inference.deploy_mu_map_env import DEPLOY_MU_MAP_ENV, deploy_env_for_manifest
from src.utils.paths import get_project_root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="outputs/biochem/clot_baseline/teacher_best_high_mu.pth")
    ap.add_argument("--clot-phi", default="outputs/biochem/clot_baseline/clot_phi_best.pth")
    ap.add_argument("--scorecard-json", default="outputs/biochem/mlp_clot_inject_probe/b_deploy_baseline_fast.json")
    ap.add_argument("--recipe", default="data/reference/clot_baseline_lane_b_deploy.json")
    args = ap.parse_args()

    root = get_project_root()
    out_dir = baseline_dir()
    recipe = load_recipe_json(args.recipe)
    recipe.deploy_mu_map_env = dict(DEPLOY_MU_MAP_ENV)

    teacher_src = root / args.teacher.replace("/", "\\")
    clot_src = root / args.clot_phi.replace("/", "\\")
    if not teacher_src.is_file() or not clot_src.is_file():
        print("[ERR] missing teacher or clot-phi ckpt", file=sys.stderr)
        return 1

    teacher_dst = out_dir / "teacher_best_high_mu.pth"
    clot_dst = out_dir / "clot_phi_best.pth"
    if teacher_src.resolve() != teacher_dst.resolve():
        _copy_if_exists(teacher_src.resolve(), teacher_dst)
    if clot_src.resolve() != clot_dst.resolve():
        _copy_if_exists(clot_src.resolve(), clot_dst)

    recipe.teacher_ckpt = "outputs/biochem/clot_baseline/teacher_best_high_mu.pth"
    recipe.clot_phi_ckpt = "outputs/biochem/clot_baseline/clot_phi_best.pth"

    score_path = root / args.scorecard_json.replace("/", "\\")
    eval_summary: dict = {"lane": "B_deploy", **deploy_env_for_manifest()}
    if score_path.is_file():
        raw = json.loads(score_path.read_text(encoding="utf-8"))
        means = raw.get("means", {}).get("B_mlp_mu_map_neighbor") or raw.get("means", {})
        eval_summary["clot_shape_scorecard"] = means
        eval_summary["scorecard_json"] = str(score_path)

    save_manifest(recipe, eval_summary=eval_summary)
    print("[OK]  deploy Leg B baseline promoted (manifest uses neighbor physics mask)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
