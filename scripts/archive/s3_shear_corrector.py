"""S3: LOAO ML wall-shear / coupled-gate corrector, supervised by exported COMSOL spf.sr.

The exact coupled gate hits ~0.75 (s3_exact_gate_all) but needs the COMSOL coupled flow. Here
we learn it deployably: map DEPLOYABLE features (initial kine-flow shear + geometry) -> the
COUPLED stagnation (final-frame [spf.sr<lss]), leave-one-anchor-out, then feed the predicted
gate into the validated closed-loop deposition law and score swept-F1.

Compares per patient: corrector (deployable) vs exact-coupled ceiling vs frozen-initial.
Run: python scripts/s3_shear_corrector.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.utils.kinematics_inference import (
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b
import scripts.s1c_soft_eval as s1c
import scripts.s2_deploy_forward as s2
import scripts.s2_kine_flow_test as kft
import scripts.spfsr_lib as spfsr

OUT = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "comsol_validation" / "s3_shear_corrector.json"
COMPLETE_FRAMES = 201
FEATS = ["kine_wls_log", "kine_wf_log", "sdf", "width", "mu_prior", "speed_ring"]


def deployable_features(d, model, dev):
    with torch.no_grad():
        pred = predict_kinematics(model, d.to(dev))
    u, v = pred[:, 0].detach(), pred[:, 1].detach()
    ring = s1b._ring_op(d, dev)
    speed_ring = ring(torch.sqrt(u ** 2 + v ** 2))
    X = np.stack([
        np.log1p(kft.wls_shear_uv(d, u, v, dev).cpu().numpy().clip(0)),
        np.log1p(kft.wallfunc_shear_uv(d, u, v, dev).cpu().numpy().clip(0)),
        d.x[:, s1b.SDF_CH].cpu().numpy(), d.x[:, s1b.WIDTH_CH].cpu().numpy(),
        d.x[:, s1b.MU_PRIOR_CH].cpu().numpy(), speed_ring.cpu().numpy(),
    ], axis=1)
    return X


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); s1c.cfg = cfg
    crit, lss = float(cfg.viscosity_mat_crit), float(cfg.lss)
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    anchors = [a for a in sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
               if spfsr.has_cache(a)]
    print(f"[i] anchors={len(anchors)}  lss={lss}\n[i] building per-anchor cache...")

    C = {}
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
        rp, ap = s2._resting_bulk(d, cfg, dev)
        wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        sr = spfsr.aligned(d, dev, a)["sr"]
        X = deployable_features(d, model, dev)
        C[a] = dict(d=d, sp=sp, rp=rp, ap=ap, wall=wall, gt=gt, sr=sr, X=X, T=T,
                    sr0=sr[0].cpu().numpy(), srT=sr[-1].cpu().numpy())

    def _loao_pred(a, ycol):
        """LOAO RF regression of log1p(spf.sr) over wall nodes -> predicted spf.sr on a."""
        Xtr = np.concatenate([C[b]["X"][C[b]["wall"]] for b in anchors if b != a])
        ytr = np.concatenate([np.log1p(C[b][ycol][C[b]["wall"]].clip(0)) for b in anchors if b != a])
        sc = StandardScaler().fit(Xtr)
        rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0, n_jobs=-1)
        rf.fit(sc.transform(Xtr), ytr)
        wall = C[a]["wall"]; pred = np.full(len(wall), 1e9)
        pred[wall] = np.expm1(rf.predict(sc.transform(C[a]["X"][wall])))
        return pred

    def _auc(pred_sr, true_sr, wall):
        yt = (true_sr[wall] < lss).astype(int)
        if not (0 < yt.sum() < len(yt)):
            return float("nan")
        return roc_auc_score(yt, (-pred_sr[wall]))     # lower shear -> higher score

    print(f"{'patient':<11}{'fr':>5}{'rd_auc':>7}{'wls_auc':>8}{'wf_auc':>7}"
          f"{'corrector':>11}{'exact':>8}{'frozen':>8}{'label':>8}")
    rep = {"per_patient": {}}
    comp = {"corrector": [], "exact": [], "frozen": [], "label": [], "rd_auc": [], "wls_auc": []}
    for a in anchors:
        c = C[a]; d, sp, T = c["d"], c["sp"], c["T"]
        wall, wall_t, gt = c["wall"], c["d"].mask_wall.reshape(-1).bool(), c["gt"]
        # t0 readout transfer: does a learned local flow->shear map beat raw operators?
        pred0 = _loao_pred(a, "sr0")
        rd_auc = _auc(pred0, c["sr0"], wall)
        wls_auc = _auc(c["X"][:, 0], c["sr0"], wall)      # X[:,0]=log1p wls (monotone ok for AUC)
        wf_auc = _auc(c["X"][:, 1], c["sr0"], wall)
        # deployable corrector: predict coupled (final) spf.sr from t0 features, gate at physical lss
        predT = _loao_pred(a, "srT")
        gate = (torch.from_numpy(predT < lss).float().to(dev)).reshape(1, -1).expand(T, -1)
        f_corr = s1c._scores(d, s2._integrate_closed_loop(d, cfg, dev, gate, c["ap"], c["rp"], sp["step2t"], sp["t_s"]),
                             gt, wall_t, crit)["swept_best_f1"]
        g_ex = (c["sr"] < lss).float()
        f_ex = s1c._scores(d, s2._integrate_closed_loop(d, cfg, dev, g_ex, c["ap"], c["rp"], sp["step2t"], sp["t_s"]),
                           gt, wall_t, crit)["swept_best_f1"]
        g_fr = (c["sr"][0].reshape(1, -1) < lss).float().expand(T, -1)
        f_fr = s1c._scores(d, s2._integrate_closed_loop(d, cfg, dev, g_fr, c["ap"], c["rp"], sp["step2t"], sp["t_s"]),
                           gt, wall_t, crit)["swept_best_f1"]
        lab = s1c._scores(d, sp["mat"][-1], gt, wall_t, crit)["swept_best_f1"]
        cohort = "complete" if T >= COMPLETE_FRAMES else "early"
        rep["per_patient"][a] = dict(cohort=cohort, frames=T, rd_auc=rd_auc, wls_auc=wls_auc,
                                     wf_auc=wf_auc, corrector=f_corr, exact=f_ex, frozen=f_fr, label=lab)
        if cohort == "complete":
            for k, vv in [("corrector", f_corr), ("exact", f_ex), ("frozen", f_fr), ("label", lab),
                          ("rd_auc", rd_auc), ("wls_auc", wls_auc)]:
                comp[k].append(vv)
        print(f"{a:<11}{T:>5}{rd_auc:>7.3f}{wls_auc:>8.3f}{wf_auc:>7.3f}"
              f"{f_corr:>11.3f}{f_ex:>8.3f}{f_fr:>8.3f}{lab:>8.3f}")

    if comp["corrector"]:
        rep["complete_mean"] = {k: float(np.nanmean(v)) for k, v in comp.items()}
        m = rep["complete_mean"]
        print(f"\n[complete-cohort mean] readout_auc={m['rd_auc']:.3f} (wls {m['wls_auc']:.3f})  "
              f"corrector={m['corrector']:.3f}  exact={m['exact']:.3f}  frozen={m['frozen']:.3f}  label={m['label']:.3f}")
        print(f"[i] corrector recovers {100*(m['corrector']-m['frozen'])/max(m['exact']-m['frozen'],1e-6):.0f}% "
              f"of the exact coupling lever (frozen->exact)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
