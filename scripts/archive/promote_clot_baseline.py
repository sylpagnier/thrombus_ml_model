"""Promote Lane A (or custom) teacher + clot-phi to stable deploy paths + manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.inference.clot_baseline_recipe import (
    ClotBaselineRecipe,
    baseline_dir,
    default_lane_a_recipe,
    load_recipe_json,
    save_manifest,
)
from src.utils.paths import get_project_root


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[OK]  {dst.name} <- {src}", flush=True)
    return True


def _load_eval_summary(eval_json: Path) -> dict:
    if not eval_json.is_file():
        return {}
    rows = []
    for line in eval_json.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        return {}
    f1s = [float(r.get("f1", 0)) for r in rows if "f1" in r]
    return {
        "n_anchors": len(rows),
        "min_f1": min(f1s) if f1s else None,
        "mean_f1": sum(f1s) / len(f1s) if f1s else None,
        "eval_json": str(eval_json),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="", help="Teacher .pth (default: gnode12 mu_unlock or promoted)")
    ap.add_argument("--clot-phi", default="", help="clot_phi_best.pth")
    ap.add_argument("--dump-dir", default="", help="Dump anchor dir for manifest")
    ap.add_argument("--eval-json", default="", help="multi_anchor.jsonl from clot train")
    ap.add_argument("--recipe", default="data/reference/clot_baseline_lane_a.json", help="Recipe JSON template")
    args = ap.parse_args()

    root = get_project_root()
    out_dir = baseline_dir()
    recipe = load_recipe_json(args.recipe)

    teacher_src = Path(args.teacher) if args.teacher else None
    if teacher_src is None or not teacher_src.is_file():
        for rel in (
            "outputs/biochem/gnode10_sweep/gnode12_mu_unlock/biochem_teacher_best_high_mu.pth",
            "outputs/biochem/gnode10_sweep/gnode12_lane_a_promoted/biochem_teacher_best_high_mu.pth",
            recipe.teacher_ckpt,
        ):
            cand = root / rel if not Path(rel).is_absolute() else Path(rel)
            if cand.is_file():
                teacher_src = cand
                break
    if teacher_src is None or not teacher_src.is_file():
        raise FileNotFoundError("Teacher checkpoint not found; pass --teacher")

    clot_src = Path(args.clot_phi) if args.clot_phi else None
    if clot_src is None or not clot_src.is_file():
        leg = recipe.clot_leg
        cand = root / f"outputs/biochem/passive_species_focus_compare/{leg}/clot_phi_best.pth"
        if cand.is_file():
            clot_src = cand
        elif (root / recipe.clot_phi_ckpt).is_file():
            clot_src = root / recipe.clot_phi_ckpt
    if clot_src is None or not clot_src.is_file():
        raise FileNotFoundError("Clot-phi checkpoint not found; pass --clot-phi")

    dump_dir = args.dump_dir or recipe.dump_anchor_dir
    eval_path = Path(args.eval_json) if args.eval_json else (
        root / f"outputs/biochem/passive_species_focus_compare/{recipe.clot_leg}/multi_anchor.jsonl"
    )

    teacher_dst = out_dir / "teacher_best_high_mu.pth"
    clot_dst = out_dir / "clot_phi_best.pth"
    _copy_if_exists(teacher_src.resolve(), teacher_dst)
    _copy_if_exists(clot_src.resolve(), clot_dst)

    recipe.teacher_ckpt = str(teacher_dst.relative_to(root)).replace("\\", "/")
    recipe.clot_phi_ckpt = str(clot_dst.relative_to(root)).replace("\\", "/")
    recipe.dump_anchor_dir = dump_dir.replace("\\", "/")

    eval_summary = _load_eval_summary(eval_path if eval_path.is_file() else Path(""))
    save_manifest(recipe, eval_summary=eval_summary)
    print(f"[OK]  clot baseline promoted under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
