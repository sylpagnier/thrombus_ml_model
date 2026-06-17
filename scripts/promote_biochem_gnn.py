"""Lock + promote biochem GNN as canonical deploy baseline."""

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

from src.biochem_gnn.config import (  # noqa: E402
    DEFAULT_KINE_CKPT,
    LEGACY_LOCKED,
    LEGACY_REFERENCE,
    PHASE_MANIFEST,
    REFERENCE_JSON,
    STACK_NAME,
    beta_ckpt_path,
    default_manifest_payload,
    global_ckpt_path,
    loao_root_path,
    rel_path,
    staging_ckpt_pick_path,
    staging_loao_eval_path,
    staging_manifest_path,
)
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _copy_ckpt(src: Path, dst: Path, *, skip_copy: bool) -> bool:
    if not src.is_file():
        print(f"[WARN] missing source ckpt: {src}", file=sys.stderr)
        return False
    if skip_copy and dst.is_file():
        print(f"[skip] {dst.name} exists", flush=True)
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"[OK] {dst.name} <- {src}", flush=True)
    return True


def _load_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_summary_from_loao(path: Path) -> dict:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        return {}
    rows = raw.get("rows") or []
    per_anchor: list[dict] = []
    for r in rows:
        sg = r.get("species_gnn") or {}
        s0 = r.get("s0_baseline") or {}
        per_anchor.append({
            "anchor": r.get("anchor"),
            "flow": r.get("flow_source"),
            "clot_f1_t_last": sg.get("clot_f1_t_last"),
            "clot_f1_t53": sg.get("clot_f1_t_last"),
            "s0_f1_t_last": s0.get("clot_f1_t_last"),
            "s0_f1_t53": s0.get("clot_f1_t_last"),
            "delta_vs_s0": r.get("delta_f1_vs_s0"),
            "health_pass": sg.get("health_pass"),
            "ckpt": r.get("ckpt"),
        })
    gt_rows = [r for r in rows if r.get("flow_source") == "gt"]
    gt_holdout = [r for r in gt_rows if not r.get("is_train_val_anchor")]
    mean_f1 = (
        sum(r["species_gnn"]["clot_f1_t_last"] for r in gt_holdout) / len(gt_holdout)
        if gt_holdout
        else None
    )
    mean_delta = (
        sum(r.get("delta_f1_vs_s0", 0.0) for r in gt_holdout) / len(gt_holdout)
        if gt_holdout
        else None
    )
    p007 = next((r for r in gt_rows if r.get("anchor") == "patient007"), None)
    return {
        "eval_json": str(path),
        "mean_holdout_clot_f1_t_last_gt": mean_f1,
        "mean_holdout_clot_f1_t53_gt": mean_f1,
        "mean_holdout_delta_vs_s0_gt": mean_delta,
        "patient007_clot_f1_t_last": (
            p007["species_gnn"]["clot_f1_t_last"] if p007 else None
        ),
        "patient007_clot_f1_t53": (
            p007["species_gnn"]["clot_f1_t_last"] if p007 else None
        ),
        "patient007_s0_f1_t_last": (
            p007["s0_baseline"]["clot_f1_t_last"] if p007 else None
        ),
        "patient007_s0_f1_t53": (
            p007["s0_baseline"]["clot_f1_t_last"] if p007 else None
        ),
        "per_anchor": per_anchor,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote biochem GNN baseline")
    ap.add_argument("--src-manifest", default="")
    ap.add_argument("--species-src", default="")
    ap.add_argument("--beta-src", default="")
    ap.add_argument("--loao-src", default="")
    ap.add_argument("--ckpt-pick", default="")
    ap.add_argument("--loao-eval", default="")
    ap.add_argument("--kine-ckpt", default="")
    ap.add_argument("--out-reference", default="")
    ap.add_argument("--skip-copy", action="store_true")
    ap.add_argument("--legacy-alias", action="store_true", default=True)
    ap.add_argument("--no-legacy-alias", action="store_false", dest="legacy_alias")
    args = ap.parse_args()

    root = get_project_root()
    out_dir = root / "outputs/biochem/biochem_gnn/locked"
    out_dir.mkdir(parents=True, exist_ok=True)
    loao_dst = out_dir / "loao"
    global_dst = out_dir / "species_gnn_best.pth"
    beta_dst = out_dir / "viscosity_beta.pth"

    species_src = root / args.species_src if args.species_src.strip() else global_ckpt_path()
    beta_src = root / args.beta_src if args.beta_src.strip() else beta_ckpt_path()
    loao_src = root / args.loao_src if args.loao_src.strip() else loao_root_path()

    if not _copy_ckpt(species_src, global_dst, skip_copy=args.skip_copy):
        return 1
    if not _copy_ckpt(beta_src, beta_dst, skip_copy=args.skip_copy):
        return 1

    src_manifest_path = (
        root / args.src_manifest if args.src_manifest.strip() else staging_manifest_path()
    )
    src_raw = _load_json(src_manifest_path) or {}
    if isinstance(src_raw.get("baseline"), dict):
        src_raw = src_raw["baseline"]

    ckpt_pick_path = (
        root / args.ckpt_pick if args.ckpt_pick.strip() else staging_ckpt_pick_path()
    )
    ckpt_pick = _load_json(ckpt_pick_path) or []
    loao_preferred: list[str] = []
    ckpt_overrides: dict[str, str] = {}
    beta_overrides = dict(src_raw.get("beta_overrides") or {})
    global_rel = rel_path(global_dst)

    for row in ckpt_pick if isinstance(ckpt_pick, list) else []:
        anc = str(row.get("anchor", ""))
        pick = str(row.get("pick", "global"))
        if pick == "loao":
            loao_preferred.append(anc)
            src_fold = loao_src / f"holdout_{anc}" / "best.pth"
            dst_fold = loao_dst / f"holdout_{anc}" / "best.pth"
            if not _copy_ckpt(src_fold, dst_fold, skip_copy=args.skip_copy):
                ckpt_overrides[anc] = global_rel
            else:
                ckpt_overrides[anc] = rel_path(dst_fold)
        else:
            ckpt_overrides[anc] = global_rel

    for anc in BIOCHEM_ANCHORS_6:
        ckpt_overrides.setdefault(anc, global_rel)

    baseline = dict(default_manifest_payload())
    baseline.update({
        "phase": PHASE_MANIFEST,
        "stack": STACK_NAME,
        "species_gnn_ckpt": global_rel,
        "viscosity_beta": rel_path(beta_dst),
        "kinematics_ckpt": args.kine_ckpt.strip() or rel_path(DEFAULT_KINE_CKPT),
        "loao_dir": rel_path(loao_dst),
        "train_val_anchor": str(src_raw.get("train_val_anchor", "patient007")),
        "loao_preferred": loao_preferred,
        "ckpt_overrides": ckpt_overrides,
        "beta_overrides": beta_overrides,
    })

    loao_eval_path = (
        root / args.loao_eval if args.loao_eval.strip() else staging_loao_eval_path()
    )
    eval_summary = _eval_summary_from_loao(loao_eval_path)

    promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reference = {
        "name": STACK_NAME,
        "version": 1,
        "promoted_at": promoted_at,
        "description": (
            "Canonical biochem_deploy stack: species_graphsage + gelation_beta + "
            "clot_trigger_physics. Flow uses frozen pmgp_deq_kine (flow_coupling not trained yet)."
        ),
        "compare_against": {
            "prior_clot_baseline": "R4.s0 inc40 rules",
            "prior_f1_patient007_t53": 0.408,
        },
        "stack": [
            "pmgp_deq_kine (PMGP-DEQ kinematics_best.pth)",
            "species_graphsage (wall-band GraphSAGE pushforward)",
            "gelation_beta.pth",
            "clot_trigger_physics (Carreau + gelation + nucleation)",
        ],
        "training_sources": {
            "species_global": rel_path(species_src),
            "viscosity_beta": rel_path(beta_src),
            "loao_folds": rel_path(loao_src),
        },
        "baseline": baseline,
        "eval": eval_summary,
    }

    runtime_manifest = out_dir / "manifest.json"
    runtime_manifest.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    ref_path = root / args.out_reference if args.out_reference.strip() else (root / REFERENCE_JSON)
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    (out_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    if args.legacy_alias:
        for legacy_ref in (LEGACY_REFERENCE, Path("data/reference/clot_deploy_gnn_baseline.json"),
                           Path("data/reference/species_gnn_deploy_baseline.json")):
            lp = root / legacy_ref
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text(json.dumps(reference, indent=2), encoding="utf-8")
        legacy_locked = root / LEGACY_LOCKED
        legacy_locked.mkdir(parents=True, exist_ok=True)
        for name in ("species_gnn_best.pth", "viscosity_beta.pth", "manifest.json", "eval_summary.json"):
            src = out_dir / name
            dst = legacy_locked / name
            if src.is_file() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)

    p007 = eval_summary.get("patient007_clot_f1_t53")
    s0 = eval_summary.get("patient007_s0_f1_t53")
    print(
        f"[OK] {STACK_NAME} locked | p007 F1={p007:.3f} vs s0={s0:.3f}"
        if p007 is not None and s0 is not None
        else f"[OK] {STACK_NAME} locked",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
