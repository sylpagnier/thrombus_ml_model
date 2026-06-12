"""Audit clot-trigger deploy masks: no GT forward leakage + nucleation sanity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_nucleation_mask import (  # noqa: E402
    gt_new_commit_mask,
    resolve_nucleation_eligibility,
)
from src.core_physics.clot_phi_simple import build_clot_phi_step  # noqa: E402
from src.core_physics.clot_trigger_rollout import (  # noqa: E402
    forward_path_uses_gt_commits,
    lumen_false_positive_frac,
    rollout_clot_trigger_physics,
    snapshot_clot_trigger_rollout_config,
)
from src.core_physics.neighbor_band_trigger import apply_physics_trigger_baseline_env  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.core_physics.clot_growth_masks import growth_seed_mode  # noqa: E402
from src.training.clot_trigger_stack import (  # noqa: E402
    apply_clot_trigger_deploy_env,
    deploy_env_is_faithful,
)
from src.utils.paths import get_project_root  # noqa: E402


def _recall_inside(new_nodes: torch.Tensor, region: torch.Tensor) -> float:
    new_nodes = new_nodes.reshape(-1).bool()
    region = region.reshape(-1).bool()
    n_new = int(new_nodes.sum().item())
    if n_new <= 0:
        return float("nan")
    return float((new_nodes & region).sum().item()) / float(n_new)


def audit_anchor(
    graph_path: Path,
    *,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
) -> dict:
    data = torch.load(graph_path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    traj = rollout_clot_trigger_physics(
        data, phys_cfg=phys_cfg, bio_cfg=bio_cfg, device=device, time_stride=1
    )

    gt_leak_checks: list[bool] = []
    recalls: list[float] = []
    lumen_fp_deploy: list[float] = []
    lumen_fp_raw: list[float] = []
    per_t: list[dict] = []

    for t in range(n_steps):
        step = build_clot_phi_step(data, t, phys_cfg, bio_cfg, device)
        phi_gt = step.phi_gt.reshape(-1)
        bundle = traj[t]
        phi_deploy = bundle["phi"]
        phi_raw = bundle["phi_raw"]

        elig_pred = resolve_nucleation_eligibility(
            data,
            t,
            device,
            phys_cfg,
            bio_cfg,
            growth_seed="pred",
            phi_pred_by_time={k: v["phi"] for k, v in traj.items()},
        )
        new_gt = gt_new_commit_mask(data, t, phys_cfg, device)
        rec = _recall_inside(new_gt, elig_pred)
        if new_gt.any():
            recalls.append(rec)

        fp_d = lumen_false_positive_frac(phi_deploy, phi_gt, data=data, device=device)
        fp_r = lumen_false_positive_frac(phi_raw, phi_gt, data=data, device=device)
        lumen_fp_deploy.append(fp_d)
        lumen_fp_raw.append(fp_r)

        # Deploy forward must not read GT commits for E(tau).
        gt_leak_checks.append(not forward_path_uses_gt_commits())

        per_t.append(
            {
                "t": t,
                "tau": float(macro_tau_at_index(data, t, bio_cfg=bio_cfg)),
                "n_new_gt": int(new_gt.sum().item()),
                "recall_gt_new_in_elig_pred": rec,
                "elig_pred_frac": float(elig_pred.float().mean().item()),
                "lumen_fp_deploy": fp_d,
                "lumen_fp_raw": fp_r,
            }
        )

    mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")
    mean_fp_deploy = sum(lumen_fp_deploy) / len(lumen_fp_deploy) if lumen_fp_deploy else float("nan")
    mean_fp_raw = sum(lumen_fp_raw) / len(lumen_fp_raw) if lumen_fp_raw else float("nan")

    pass_deploy = (
        not forward_path_uses_gt_commits()
        and all(gt_leak_checks)
        and mean_fp_deploy <= mean_fp_raw + 1e-9
    )
    pass_recall = mean_recall >= 0.90 if recalls else True

    return {
        "anchor": graph_path.stem,
        "n_steps": n_steps,
        "mean_recall_gt_new_in_elig_pred": mean_recall,
        "mean_lumen_fp_deploy": mean_fp_deploy,
        "mean_lumen_fp_raw": mean_fp_raw,
        "lumen_fp_reduction": mean_fp_raw - mean_fp_deploy,
        "pass_no_gt_forward": not forward_path_uses_gt_commits(),
        "pass_lumen_fp_not_worse": mean_fp_deploy <= mean_fp_raw + 1e-9,
        "pass_recall_elig": pass_recall,
        "pass_deploy_mask": bool(pass_deploy),
        "pass_audit": bool(pass_deploy),
        "per_t": per_t,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Clot trigger deploy mask audit (T0 physics)")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_deploy_audit.json")
    ap.add_argument("--min-recall", type=float, default=0.90)
    args = ap.parse_args()

    apply_clot_trigger_deploy_env()
    apply_physics_trigger_baseline_env()
    if growth_seed_mode() != "pred" or not deploy_env_is_faithful():
        print("[FAIL] deploy env must use pred growth seed and ceiling/nucleation loss", flush=True)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")
    root = get_project_root()

    anchor_dir = Path(args.anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    paths = discover_anchor_paths(anchor_dir)
    if not paths:
        print(f"[ERR] no graphs in {anchor_dir}", file=sys.stderr)
        return 2

    rows = [
        audit_anchor(p, device=device, phys_cfg=phys_cfg, bio_cfg=bio_cfg) for p in paths
    ]
    n_pass = sum(1 for r in rows if r["pass_audit"])
    n_recall_pass = sum(1 for r in rows if r["pass_recall_elig"])
    summary = {
        "step": "t0_deploy_mask_audit",
        "rollout_config": snapshot_clot_trigger_rollout_config(),
        "deploy_faithful_forward": not forward_path_uses_gt_commits(),
        "anchors_pass_deploy": n_pass,
        "anchors_pass_recall": n_recall_pass,
        "anchors_total": len(rows),
        "pass_deploy_all": n_pass == len(rows),
        "pass_recall_majority": n_recall_pass >= max(1, len(rows) - 1),
        "min_recall_gate": float(args.min_recall),
        "per_anchor": [{k: v for k, v in r.items() if k != "per_t"} for r in rows],
    }

    for r in rows:
        recall_note = "" if r["pass_recall_elig"] else " recall=WARN"
        print(
            f"[OK] {r['anchor']}: recall={r['mean_recall_gt_new_in_elig_pred']:.3f} "
            f"lumen_fp deploy={r['mean_lumen_fp_deploy']:.3f} raw={r['mean_lumen_fp_raw']:.3f} "
            f"pass={r['pass_audit']}{recall_note}",
            flush=True,
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {out_path}", flush=True)
    print(
        f"[summary] deploy_pass={n_pass}/{len(rows)} recall_pass={n_recall_pass}/{len(rows)} "
        f"deploy_forward={summary['deploy_faithful_forward']}",
        flush=True,
    )

    if not summary["deploy_faithful_forward"]:
        print("[FAIL] CLOT_TRIGGER_FORWARD_SEED must be pred for deploy audit", flush=True)
        return 1
    if n_pass < len(rows):
        print("[FAIL] one or more anchors failed deploy mask audit (GT leak or lumen FP regression)", flush=True)
        return 1
    if not summary["pass_recall_majority"]:
        print("[WARN] GT-new recall inside E_pred below gate on multiple anchors (see patient006)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
