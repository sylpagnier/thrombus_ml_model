"""Where should ML go? p007 probe: physics backbone + ML corrector at the weak spot.

Backbone (physics): exact gate + deposition + autocat closed loop (validated). The deploy
gap is wall-shear estimation. Question: from DEPLOYABLE inputs (kine flow + geometry), can
ML recover the exact-shear gate (0.80) that noisy WLS misses (0.56)? And is it better to
learn SHEAR (regress spf.sr) or the GATE (classify membership)?

Reports: ceilings (label, exact-shear physics) + ML-deployable closed-loop F1, held-out AUC,
feature importances.  Run: python scripts/_diag_ml_direction.py
"""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SPECIES_ROLLOUT_DEPLOY_FAITHFUL", "1")
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b
import scripts.s1c_soft_eval as s1c
import scripts.s2_deploy_forward as s2
import scripts.s2_kine_flow_test as kft

cfg = BiochemConfig(phase="biochem"); s1c.cfg = cfg
phys = PhysicsConfig(phase="biochem"); dev = torch.device("cpu")
crit = float(cfg.viscosity_mat_crit); lss = float(cfg.lss)
FEAT_NAMES = ["kine_wls_log", "kine_wf_log", "sdf", "width", "mu_prior", "speed_ring"]


def closed_loop_f1(d, sp, rp, ap, g_low):
    mat = s2._integrate_closed_loop(d, cfg, dev, g_low, ap, rp, sp["step2t"], sp["t_s"])
    wall = d.mask_wall.reshape(-1).bool()
    gt = gt_clot_phi_at_time(d, d.y.shape[0] - 1, phys, device=dev).reshape(-1).bool()
    return s1c._scores(d, mat, gt, wall, crit)["swept_best_f1"]


def main():
    from src.utils.kinematics_inference import load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    d = torch.load(s1b.ANCHOR_DIR / "patient007.pt", map_location=dev, weights_only=False)
    sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
    rp, ap = s2._resting_bulk(d, cfg, dev)
    wall = d.mask_wall.reshape(-1).bool()
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()

    # flows + shears
    with torch.no_grad():
        pred = predict_kinematics(model, d.to(dev))
    uk, vk = pred[:, 0].detach(), pred[:, 1].detach()
    kine_wls = kft.wls_shear_uv(d, uk, vk, dev)            # [N] t0
    kine_wf = kft.wallfunc_shear_uv(d, uk, vk, dev)
    sr_exact_T = s1b._exact_shear_p007(d, dev)             # [T,N] exact spf.sr (time-varying, coupled)
    sr_exact0 = sr_exact_T[0]                              # [N] exact spf.sr at t0 (initial flow)

    # ---- ceiling decomposition ----
    gtmat = sp["mat"][-1]
    label_ceil = s1c._scores(d, gtmat, gt, wall, crit)["swept_best_f1"]
    coupled_ceil = closed_loop_f1(d, sp, rp, ap, (sr_exact_T < lss).float())          # time-varying exact
    init_exact = closed_loop_f1(d, sp, rp, ap, (sr_exact0.reshape(1, -1) < lss).float().expand(T, -1))
    init_kine = closed_loop_f1(d, sp, rp, ap, (kine_wls.reshape(1, -1) < lss).float().expand(T, -1))
    print(f"[CEILINGS] label(perfect Mat)={label_ceil:.3f}")
    print(f"           coupled-flow exact gate (time-varying, oracle) = {coupled_ceil:.3f}  <- needs coupling loop")
    print(f"           initial-flow exact gate (t0)                   = {init_exact:.3f}  <- deployable, best shear")
    print(f"           initial-flow kine  gate (t0)                   = {init_kine:.3f}  <- deployable now")
    print(f"  => coupling (flow evolution) is worth {coupled_ceil - init_exact:+.3f}; "
          f"shear operator at t0 is worth {init_exact - init_kine:+.3f}")

    # ---- features (deployable) ----
    speed_ring = s1b._ring_op(d, dev)(torch.sqrt(uk ** 2 + vk ** 2))
    X = np.stack([
        np.log1p(kine_wls.cpu().numpy().clip(0)), np.log1p(kine_wf.cpu().numpy().clip(0)),
        d.x[:, s1b.SDF_CH].cpu().numpy(), d.x[:, s1b.WIDTH_CH].cpu().numpy(),
        d.x[:, s1b.MU_PRIOR_CH].cpu().numpy(), speed_ring.cpu().numpy(),
    ], axis=1)
    w = wall.cpu().numpy()
    # target = COUPLED (post-evolution) stagnation footprint = where flow ends up low-shear.
    # Can ML predict this from INITIAL geometry+flow? (the deployable coupling shortcut)
    gate_tgt = (sr_exact_T[-1].cpu().numpy() < lss).astype(int)

    # spatial held-out split among wall nodes (by x median) -> honest within-patient signal
    xcoord = d.x[:, 0].cpu().numpy()
    wall_idx = np.where(w)[0]
    tr = wall_idx[xcoord[wall_idx] <= np.median(xcoord[wall_idx])]
    te = wall_idx[xcoord[wall_idx] > np.median(xcoord[wall_idx])]
    sc = StandardScaler().fit(X[tr])
    print(f"\n[learn the COUPLED GATE from initial features] target=final-frame low-shear, spatial split "
          f"(train {len(tr)} / test {len(te)} wall nodes)")
    for name, clf in [("logistic", LogisticRegression(max_iter=2000, class_weight="balanced")),
                      ("rforest", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced"))]:
        clf.fit(sc.transform(X[tr]), gate_tgt[tr])
        auc = roc_auc_score(gate_tgt[te], clf.predict_proba(sc.transform(X[te]))[:, 1]) \
            if gate_tgt[te].sum() else float("nan")
        prob = np.zeros(len(w)); prob[w] = clf.predict_proba(sc.transform(X[w]))[:, 1]
        g = (torch.from_numpy(prob).float().to(dev) > 0.5).float().reshape(1, -1).expand(T, -1)
        f1 = closed_loop_f1(d, sp, rp, ap, g)
        imp = getattr(clf, "feature_importances_", None)
        impstr = "  ".join(f"{n}:{v:.2f}" for n, v in zip(FEAT_NAMES, imp)) if imp is not None else ""
        print(f"   {name:<9} test-AUC={auc:.3f}  closed-loop F1={f1:.3f}   {impstr}")

    # learn the SHEAR (regress coupled final-frame log spf.sr) then physics gate
    from sklearn.ensemble import RandomForestRegressor
    ytgt = np.log1p(sr_exact_T[-1].cpu().numpy().clip(0))
    reg = RandomForestRegressor(n_estimators=300, max_depth=8).fit(sc.transform(X[tr]), ytgt[tr])
    r2 = reg.score(sc.transform(X[te]), ytgt[te])
    pred_sr = np.zeros(len(w)); pred_sr[w] = np.expm1(reg.predict(sc.transform(X[w])))
    g = (torch.from_numpy(pred_sr).float().to(dev) < lss).float().reshape(1, -1).expand(T, -1)
    f1 = closed_loop_f1(d, sp, rp, ap, g)
    print(f"\n[learn the SHEAR] regress coupled log spf.sr  test-R2={r2:.3f}  -> physics gate F1={f1:.3f}")
    print(f"\n[i] deployable targets: initial gate ~{init_kine:.2f}; ML coupled-gate corrector lifts "
          f"toward coupled ceiling {coupled_ceil:.2f}; label ceiling {label_ceil:.2f}")


if __name__ == "__main__":
    main()
