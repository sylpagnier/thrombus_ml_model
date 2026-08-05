"""Promote wall-gen / phase1 baseline (FS_ab_coupled stack).

Copies the winning flow-source A/B checkpoint into a stable alias, writes
data/reference/mat_wall_gen_baseline.json with typed architecture + holdout score.

This does NOT replace locked WC_v7 (compound / in-family backbone). It is the
baseline for generalization wall-gen sweeps (go_phase1_sweep_v3.ps1).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import mat_growth_leg_spec  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

DEFAULT_SRC = "outputs/biochem/eda/flow_source_ab/FS_ab_coupled/best.pth"
DEFAULT_LEG = "FS_ab_coupled"
DEFAULT_LABEL = (
    "Wall-gen baseline: deploy-faithful train (RGP-DEQ @ t=0 + local tiling), "
    "drop-xy, WC_v7 dynamics"
)
ALIAS_ROOT = "outputs/biochem/biochem_gnn/wall_gen_baseline"
LOCKED_ALIAS = "outputs/biochem/biochem_gnn/wall_gen_baseline/species/best.pth"
REFERENCE_JSON = "data/reference/mat_wall_gen_baseline.json"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _copy_ckpt(src: Path, dst: Path, *, skip_copy: bool) -> bool:
    if not src.is_file():
        print(f"[ERR] missing ckpt: {src}", file=sys.stderr)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if skip_copy and dst.is_file():
        print(f"[skip] {dst.name} exists", flush=True)
        return True
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"[OK] {_rel(dst, REPO)} <- {_rel(src, REPO)}", flush=True)
    return True


def _load_eval_metrics(eval_path: Path | None) -> dict:
    if eval_path is None or not eval_path.is_file():
        return {}
    raw = json.loads(eval_path.read_text(encoding="utf-8"))
    simple = raw.get("simple") or raw
    mean = dict(simple.get("mean") or {})
    per = dict(simple.get("per_anchor") or {})
    p020 = dict(per.get("patient020") or {})
    return {
        "eval_json": _rel(eval_path, REPO),
        "mean": {
            "deploy_clot_score": float(mean.get("deploy_clot_score") or 0.0),
            "deploy_clot_f1": float(mean.get("deploy_clot_f1") or 0.0),
            "deploy_mat_f1": float(mean.get("deploy_mat_f1") or 0.0),
        },
        "patient020": {
            "deploy_clot_score": float(
                p020.get("deploy_clot_score") or mean.get("deploy_clot_score") or 0.0
            ),
            "deploy_clot_f1": float(
                p020.get("deploy_clot_f1") or mean.get("deploy_clot_f1") or 0.0
            ),
            "deploy_mat_f1": float(
                p020.get("deploy_mat_f1") or mean.get("deploy_mat_f1") or 0.0
            ),
        },
        "protocol": {
            "holdout": "patient020",
            "deploy_faithful": True,
            "flow": "RGP-DEQ @ t=0 + local tiling (corrector coupling)",
            "no_gt_velocity": True,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="Source best.pth")
    ap.add_argument("--leg", default=DEFAULT_LEG)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument(
        "--eval-json",
        default="",
        help="Cold deploy eval JSON (prefer patient020-only holdout)",
    )
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args()

    root = get_project_root()
    src = Path(args.src)
    if not src.is_absolute():
        src = root / src

    locked = root / LOCKED_ALIAS
    alias_dir = root / ALIAS_ROOT / "species"
    if not _copy_ckpt(src, locked, skip_copy=args.skip_copy):
        return 1
    meta_src = src.with_suffix(".json")
    if meta_src.is_file():
        shutil.copy2(meta_src, alias_dir / "best.json")
        print(f"[OK] {_rel(alias_dir / 'best.json', REPO)}", flush=True)

    spec = mat_growth_leg_spec(str(args.leg))
    eval_path = None
    if args.eval_json.strip():
        eval_path = Path(args.eval_json)
        if not eval_path.is_absolute():
            eval_path = root / eval_path
        if eval_path.is_file():
            shutil.copy2(eval_path, root / ALIAS_ROOT / "eval_holdout_p020.json")

    scores = _load_eval_metrics(eval_path)
    promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "stack": "wall_gen_baseline",
        "leg": str(args.leg),
        "label": str(args.label),
        "promoted_at": promoted_at,
        "source_ckpt": _rel(src, root),
        "baseline_ckpt": LOCKED_ALIAS.replace("\\", "/"),
        "note": (
            "Phase1 / wall-gen generalization baseline from flow-source A/B. "
            "Train+deploy use RGP-DEQ base flow and local tiling (no GT velocity). "
            "Does not replace locked WC_v7 for compound/in-family warm-starts."
        ),
        "cohort": {
            "train": "patient005,patient006,patient010,patient023,patient002",
            "val": "patient020",
            "holdout": "patient020",
        },
        "config_kwargs": dict(spec.config_kwargs),
        "runtime_kwargs": dict(spec.runtime_kwargs),
        "scores": scores,
        "flow_source_ab": {
            "gt_mean_score_020_034": 0.2680,
            "kine_mean_score_020_034": 0.2534,
            "coupled_mean_score_020_034": 0.2765,
            "winner": "FS_ab_coupled",
        },
    }

    alias_manifest = root / ALIAS_ROOT / "manifest.json"
    alias_manifest.parent.mkdir(parents=True, exist_ok=True)
    alias_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {_rel(alias_manifest, root)}", flush=True)

    ref = root / REFERENCE_JSON
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {_rel(ref, root)}", flush=True)

    p020 = (scores.get("patient020") or {})
    print(
        f"[OK] wall_gen baseline={args.leg} "
        f"p020_score={float(p020.get('deploy_clot_score') or 0.0):.4f} "
        f"p020_f1={float(p020.get('deploy_clot_f1') or 0.0):.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
