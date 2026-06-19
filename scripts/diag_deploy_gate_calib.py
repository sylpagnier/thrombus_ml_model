"""Cheap decisive check: can CALIBRATION of the deployable shear gate recover the ideal-gate
FP pruning, or is the kine-wallfunc ranking genuinely too weak?

For each anchor we take the deploy stack's predicted clot, then PRUNE it by shear rank: keep only
the lowest-shear fraction q of predicted nodes. Sweep q for (a) deployable kine-wallfunc shear and
(b) ideal exact spf.sr. Report:
  f1_nogate            : q=1 (no pruning)
  f1_depl_bestq        : best q pruning by deployable shear (per-patient oracle threshold = upper bound)
  f1_depl_fixedq       : single cohort-best q applied to all (transferable / realistic)
  f1_ideal_bestq       : best q pruning by exact spf.sr (reference ceiling)
If f1_depl_fixedq >> f1_nogate and ~ f1_ideal_bestq -> calibration works, L1 viable cheap.
If f1_depl_bestq ~ f1_nogate -> deployable RANKING too weak (need better flow / learned signal).

Caches per-anchor arrays -> outputs/.../deploy_gate_calib_arrays.pt for fast re-runs.
Run: python scripts/diag_deploy_gate_calib.py
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from src.biochem_gnn import BiochemGNN, load_manifest, reference_manifest_path
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.species_pushforward_continuous import discover_biochem_anchors
from src.utils.kinematics_inference import (
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
from src.utils.paths import get_project_root
import scripts.eval_biochem_gnn_deploy_ab as ev
import scripts.s2_kine_flow_test as kft
import scripts.spfsr_lib as spfsr

ANCHOR = ROOT / "data" / "processed" / "graphs_biochem_anchors"
OUTDIR = ROOT / "outputs" / "reports" / "comsol_validation"
ARR = OUTDIR / "deploy_gate_calib_arrays.pt"
QGRID = np.round(np.arange(0.1, 1.01, 0.05), 2)


def _f1(pred, gt):
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def _prune_curve(pred, gt, shear):
    """F1 vs q: keep lowest-shear fraction q of predicted-positive nodes."""
    idx = pred.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return {float(q): 0.0 for q in QGRID}
    s = shear[idx]
    out = {}
    for q in QGRID:
        thr = torch.quantile(s, float(q))
        keep = torch.zeros_like(pred)
        keep[idx] = s <= thr
        out[float(q)] = _f1(keep, gt)
    return out


def build_arrays(anchors):
    device = require_cuda_device(); cpu = torch.device("cpu")
    manifest = load_manifest(str(reference_manifest_path()))
    phys = PhysicsConfig(phase="biochem")
    kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), cpu, phys_cfg=PhysicsConfig(phase="kinematics"))
    cfg = BiochemConfig(phase="biochem"); lss = float(cfg.lss)
    store = {}
    for a in anchors:
        dgpu = torch.load(ANCHOR / f"{a}.pt", map_location=device, weights_only=False)
        n = int(dgpu.y.shape[0]); t_main = min(200, n - 1)
        with ev.rollout_env(manifest, anchor=a, mode="deploy_frozen") as fm:
            model = BiochemGNN.from_manifest(manifest, anchor=a, device=device, flow_mode=fm)
            phi = model.rollout(dgpu).phi_by_time
        pred = (phi[t_main].reshape(-1).detach().to(cpu) > 0.5)
        d = torch.load(ANCHOR / f"{a}.pt", map_location=cpu, weights_only=False)
        gt = gt_clot_phi_at_time(d, t_main, phys, device=cpu).reshape(-1).bool()
        with torch.no_grad():
            uv = predict_kinematics(kine, d.clone())
        wf = kft.wallfunc_shear_uv(d, uv[:, 0].detach(), uv[:, 1].detach(), cpu)
        sr = (spfsr.aligned(d, cpu, a)["sr"][min(t_main, spfsr.load_raw(a, cpu)["sr"].shape[0] - 1)]
              if spfsr.has_cache(a) else None)
        store[a] = {"pred": pred, "gt": gt, "wf": wf, "sr": sr, "t_main": t_main}
        print(f"  [{a}] t={t_main} pred_pos={int(pred.sum())} gt_pos={int(gt.sum())}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    torch.save(store, ARR)
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=",".join(discover_biochem_anchors(get_project_root())))
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    if ARR.exists() and not args.rebuild:
        store = torch.load(ARR, weights_only=False); print(f"[i] loaded cached arrays {ARR}")
    else:
        print("[i] running deploy rollouts (cache miss)..."); store = build_arrays(anchors)
    anchors = [a for a in anchors if a in store]

    # per-anchor prune curves
    depl = {a: _prune_curve(store[a]["pred"], store[a]["gt"], store[a]["wf"]) for a in anchors}
    ideal = {a: (_prune_curve(store[a]["pred"], store[a]["gt"], store[a]["sr"]) if store[a]["sr"] is not None else None)
             for a in anchors}
    # cohort-best fixed q for deployable
    cohort = {float(q): float(np.mean([depl[a][float(q)] for a in anchors])) for q in QGRID}
    fixed_q = max(cohort, key=cohort.get)

    print(f"\n{'patient':<11}{'nogate':>8}{'depl_bestq':>11}{'depl@fixed':>11}{'ideal_bestq':>12}")
    rows = {}
    nog, dbq, dfq, ibq = [], [], [], []
    for a in anchors:
        f_nog = depl[a][1.0]
        f_dbq = max(depl[a].values())
        f_dfq = depl[a][fixed_q]
        f_ibq = (max(ideal[a].values()) if ideal[a] is not None else float("nan"))
        rows[a] = dict(t=store[a]["t_main"], nogate=f_nog, depl_bestq=f_dbq, depl_fixedq=f_dfq,
                       ideal_bestq=f_ibq, depl_bestq_at=max(depl[a], key=depl[a].get))
        nog.append(f_nog); dbq.append(f_dbq); dfq.append(f_dfq); ibq.append(f_ibq)
        print(f"{a:<11}{f_nog:>8.3f}{f_dbq:>11.3f}{f_dfq:>11.3f}{f_ibq:>12.3f}")
    print(f"\n[mean] nogate={np.mean(nog):.3f}  depl_bestq={np.mean(dbq):.3f}  "
          f"depl@fixed_q({fixed_q})={np.mean(dfq):.3f}  ideal_bestq={np.nanmean(ibq):.3f}")
    print("[i] depl@fixed >> nogate and ~ ideal_bestq => calibration recovers it (L1 viable).")
    print("[i] depl_bestq ~ nogate => deployable ranking too weak (need better flow / learned signal).")
    out = {"fixed_q": fixed_q, "cohort_q_curve": cohort, "per_patient": rows,
           "mean": {"nogate": float(np.mean(nog)), "depl_bestq": float(np.mean(dbq)),
                    "depl_fixedq": float(np.mean(dfq)), "ideal_bestq": float(np.nanmean(ibq))}}
    (OUTDIR / "deploy_gate_calib.json").write_text(json.dumps(out, indent=2))
    print(f"[save] {OUTDIR / 'deploy_gate_calib.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
