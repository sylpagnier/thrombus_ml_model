"""Confirm the lever: decompose the current deploy stack's clot error before building.

For each anchor (deploy_frozen, the deployable mode) at the main eval time:
  * precision / recall / F1 (reproduce deploy_ab_eval)
  * FALSE POSITIVES split: wall / near-wall(1-hop) / deep-lumen  + how many fall OUTSIDE a
    low-shear gate (ideal = exact COMSOL spf.sr<lss ; deployable = kine wallfunc<lss) => prunable
  * FALSE NEGATIVES split: wall / near-wall / deep-lumen  (deep-lumen FN = growth/label-ceiling,
    NOT gate-fixable)
  * projected F1 after ideal-gate and deployable-gate FP pruning (the L1 upper/realistic bound)

Run: python scripts/diag_deploy_error_decomp.py [--anchors a,b] [--times 53,200]
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

OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "deploy_error_decomp.json"
ANCHOR = ROOT / "data" / "processed" / "graphs_biochem_anchors"


def _bands(d):
    wall = d.mask_wall.reshape(-1).bool()
    row, col = d.edge_index
    nb = torch.zeros(d.num_nodes, dtype=torch.bool)
    nb[col[wall[row]]] = True
    near = nb & ~wall
    lumen = ~wall & ~near
    return wall, near, lumen


def _f1(pred, gt):
    tp = float((pred & gt).sum()); fp = float((pred & ~gt).sum()); fn = float((~pred & gt).sum())
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=",".join(discover_biochem_anchors(get_project_root())))
    ap.add_argument("--times", default="53,200")
    args = ap.parse_args()
    device = require_cuda_device()
    cpu = torch.device("cpu")
    manifest = load_manifest(str(reference_manifest_path()))
    phys = PhysicsConfig(phase="biochem"); cfg = BiochemConfig(phase="biochem"); lss = float(cfg.lss)
    times = [int(x) for x in args.times.split(",") if x.strip()]
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), cpu, phys_cfg=PhysicsConfig(phase="kinematics"))

    print(f"[i] lss={lss}  anchors={len(anchors)}\n")
    hdr = (f"{'patient':<11}{'prec':>6}{'rec':>6}{'f1':>6} | {'FP_wall':>8}{'FP_near':>8}{'FP_lumen':>9}"
           f" | {'FPprun_id':>10}{'FPprun_dp':>10} | {'FN_lumen%':>10} | {'f1_ideal':>9}{'f1_depl':>8}")
    print(hdr)
    rep = {"per_patient": {}}
    agg = {k: [] for k in ["f1", "f1_ideal", "f1_depl", "fn_lumen_frac"]}
    for a in anchors:
        dgpu = torch.load(ANCHOR / f"{a}.pt", map_location=device, weights_only=False)
        n_steps = int(dgpu.y.shape[0]); eval_times = sorted({max(0, min(t, n_steps - 1)) for t in times})
        with ev.rollout_env(manifest, anchor=a, mode="deploy_frozen") as fm:
            model = BiochemGNN.from_manifest(manifest, anchor=a, device=device, flow_mode=fm)
            phi = model.rollout(dgpu).phi_by_time
        t_main = eval_times[-1]
        pred = (phi[t_main].reshape(-1).detach().to(cpu) > 0.5)

        d = torch.load(ANCHOR / f"{a}.pt", map_location=cpu, weights_only=False)
        gt = gt_clot_phi_at_time(d, t_main, phys, device=cpu).reshape(-1).bool()
        wall, near, lumen = _bands(d)
        p, r, f1 = _f1(pred, gt)
        fp = pred & ~gt; fn = ~pred & gt
        nfp = float(fp.sum())
        # gates
        gate_ideal = None
        if spfsr.has_cache(a):
            sr = spfsr.aligned(d, cpu, a)["sr"]
            gate_ideal = (sr[min(t_main, sr.shape[0] - 1)] < lss)
        with torch.no_grad():
            uv = predict_kinematics(kine, d.clone())
        u, v = uv[:, 0].detach(), uv[:, 1].detach()
        gate_depl = (kft.wallfunc_shear_uv(d, u, v, cpu) < lss)

        fp_prun_id = float((fp & ~gate_ideal).sum()) / max(nfp, 1) if gate_ideal is not None else float("nan")
        fp_prun_dp = float((fp & ~gate_depl).sum()) / max(nfp, 1)
        fn_lumen_frac = float((fn & lumen).sum()) / max(float(fn.sum()), 1)
        f1_ideal = _f1(pred & (gate_ideal if gate_ideal is not None else pred), gt)[2]
        f1_depl = _f1(pred & gate_depl, gt)[2]

        rep["per_patient"][a] = dict(t=t_main, prec=p, rec=r, f1=f1,
            fp_wall=int((fp & wall).sum()), fp_near=int((fp & near).sum()), fp_lumen=int((fp & lumen).sum()),
            fp_prun_ideal=fp_prun_id, fp_prun_depl=fp_prun_dp, fn_lumen_frac=fn_lumen_frac,
            f1_ideal_gate=f1_ideal, f1_depl_gate=f1_depl)
        for k, vv in [("f1", f1), ("f1_ideal", f1_ideal), ("f1_depl", f1_depl), ("fn_lumen_frac", fn_lumen_frac)]:
            agg[k].append(vv)
        print(f"{a:<11}{p:>6.2f}{r:>6.2f}{f1:>6.2f} | {int((fp&wall).sum()):>8}{int((fp&near).sum()):>8}"
              f"{int((fp&lumen).sum()):>9} | {fp_prun_id:>10.2f}{fp_prun_dp:>10.2f} | {fn_lumen_frac:>10.2f}"
              f" | {f1_ideal:>9.2f}{f1_depl:>8.2f}")

    rep["mean"] = {k: float(np.nanmean(v)) for k, v in agg.items()}
    m = rep["mean"]
    print(f"\n[mean] f1={m['f1']:.3f}  f1_after_ideal_gate={m['f1_ideal']:.3f}  "
          f"f1_after_deployable_gate={m['f1_depl']:.3f}  FN_lumen_frac={m['fn_lumen_frac']:.3f}")
    print("[i] f1_ideal>>f1 => precision is gate-prunable (L1 worth it); high FN_lumen => growth lever needed too")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
