"""Is the deployable kine flow an adequate gate input vs GT (COMSOL) flow?

The kine RGP-DEQ predicts the steady no-clot flow for a geometry == the initial (t0)
field that sets the stagnation zones. This test: (1) how close is kine u,v to GT t0 u,v,
(2) does a kine-flow low-shear gate localize the clot as well as a GT-flow gate.
Gate scored wall-nucleate + dilate-1 vs GT clot (same as S2). t0 flow only (deployable).

Run: python scripts/s2_kine_flow_test.py
"""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SPECIES_ROLLOUT_DEPLOY_FAITHFUL", "1")
os.environ.setdefault("SPECIES_ROLLOUT_VEL_SOURCE", "kinematics")
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.utils.rheology import compute_shear_rate
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b

cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
lss = float(cfg.lss)


def wls_shear_uv(d, u, v, dev):
    u_ref = float(d.u_ref.view(-1)[0]); d_bar = float(d.d_bar.view(-1)[0])
    Gx, Gy = d.G_x.to(dev), d.G_y.to(dev)
    uu = u.reshape(-1, 1); vv = v.reshape(-1, 1)
    dudx = torch.sparse.mm(Gx, uu).squeeze(1); dudy = torch.sparse.mm(Gy, uu).squeeze(1)
    dvdx = torch.sparse.mm(Gx, vv).squeeze(1); dvdy = torch.sparse.mm(Gy, vv).squeeze(1)
    g_nd = compute_shear_rate(dudx, dudy, dvdx, dvdy, eps=1e-6)
    return g_nd * (u_ref / max(d_bar, 1e-8))


def wallfunc_shear_uv(d, u, v, dev):
    u_ref = float(d.u_ref.view(-1)[0]); d_bar = float(d.d_bar.view(-1)[0])
    sdf_phys = (d.x[:, s1b.SDF_CH].to(dev).clamp_min(0.0) * d_bar)
    ring = s1b._ring_op(d, dev)
    sdf_ring = ring(sdf_phys).clamp_min(1e-5)
    speed_phys = torch.sqrt(u ** 2 + v ** 2) * u_ref
    return ring(speed_phys) / sdf_ring


def f1(pred, gt):
    tp = float((pred & gt).sum()); p = tp / max(float(pred.sum()), 1); r = tp / max(float(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9)


def score_gate(d, sr, wall, gt):
    pred = s1b._dilate((sr < lss) & wall, d.edge_index, 1)
    return f1(pred, gt)


def main():
    from src.utils.kinematics_inference import (load_kinematics_predictor,
                                                predict_kinematics, resolve_kinematics_checkpoint)
    ckpt = resolve_kinematics_checkpoint()
    print(f"[i] kine ckpt: {ckpt}")
    model = load_kinematics_predictor(ckpt, dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    anchors = ["patient007", "patient005", "patient006", "patient010", "patient001"]
    print(f"\n{'patient':<11}{'velCorr':>9}{'velCorrW':>9}{'relL2':>8}"
          f"{'wls_gt':>8}{'wls_kine':>9}{'wf_gt':>8}{'wf_kine':>9}")
    rows = []
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = d.y.shape[0]; wall = d.mask_wall.reshape(-1).bool()
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        ug = d.y[0, :, 0].to(dev); vg = d.y[0, :, 1].to(dev)
        pred = predict_kinematics(model, d.to(dev))
        uk, vk = pred[:, 0], pred[:, 1]

        def corr(x, y, m=None):
            x = x[m] if m is not None else x; y = y[m] if m is not None else y
            x = x.detach().cpu().numpy(); y = y.detach().cpu().numpy()
            return float(np.corrcoef(np.concatenate([x, y]).reshape(2, -1))[0, 1])
        vc = np.mean([corr(uk, ug), corr(vk, vg)])
        vcw = np.mean([corr(uk, ug, wall), corr(vk, vg, wall)])
        rel = float((torch.sqrt(((uk - ug) ** 2 + (vk - vg) ** 2).sum())
                     / torch.sqrt((ug ** 2 + vg ** 2).sum() + 1e-9)))
        sr_gt = wls_shear_uv(d, ug, vg, dev); sr_kine = wls_shear_uv(d, uk, vk, dev)
        wf_gt = wallfunc_shear_uv(d, ug, vg, dev); wf_kine = wallfunc_shear_uv(d, uk, vk, dev)
        r = dict(a=a, velCorr=vc, velCorrW=vcw, relL2=rel,
                 wls_gt=score_gate(d, sr_gt, wall, gt), wls_kine=score_gate(d, sr_kine, wall, gt),
                 wf_gt=score_gate(d, wf_gt, wall, gt), wf_kine=score_gate(d, wf_kine, wall, gt))
        rows.append(r)
        print(f"{a:<11}{vc:>9.3f}{vcw:>9.3f}{rel:>8.2f}{r['wls_gt']:>8.3f}{r['wls_kine']:>9.3f}"
              f"{r['wf_gt']:>8.3f}{r['wf_kine']:>9.3f}")
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"\n{'MEAN':<11}{m('velCorr'):>9.3f}{m('velCorrW'):>9.3f}{m('relL2'):>8.2f}"
          f"{m('wls_gt'):>8.3f}{m('wls_kine'):>9.3f}{m('wf_gt'):>8.3f}{m('wf_kine'):>9.3f}")
    print(f"\n[i] if kine~GT (velCorr high, gate F1 close) -> kine flow is an adequate deployable")
    print(f"[i] gate input; remaining gap is gate FORMULATION (exact-shear ceiling p007~0.68).")


if __name__ == "__main__":
    main()
