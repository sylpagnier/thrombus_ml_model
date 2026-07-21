"""S3 WIRE: per-frame ML shear-readout corrector inside the geometry-occlusion loop.

Decision from the diagnostics:
  * exact coupled spf.sr gate -> ~0.75 (s3_exact_gate_all), but needs COMSOL's coupled flow.
  * static t0-features -> coupled-gate map only recovers 17% and fails cross-geometry
    (s3_shear_corrector): the time-EVOLUTION is the lever, not the t0 readout.
  * geometry occlusion makes the kine model divert physically (s3_coupled_loop), but a
    WLS/wallfunc readout on the diverted flow still can't read the sharpened stagnation.

So: let PHYSICS evolve the flow (geometry occlusion) and let ML read spf.sr from each evolved
flow state. The readout corrector is a LOCAL operator (flow field -> shear), trained LOAO on
occluded-flow / exact-spf.sr pairs, and should transfer across geometry.

Eval (LOAO, complete-cohort headline):
  exact        : COMSOL spf.sr gate                                  (ceiling)
  oracle+corr  : occlude with GT clot per frame, corrector readout   (readout quality, no bootstrap)
  oracle+wf    : occlude with GT clot per frame, wallfunc readout    (physics-operator baseline)
  prog+corr    : progressive produced clot, corrector readout        (real deployable)
  frozen       : initial-flow wallfunc gate                          (no coupling)
Run: python scripts/s3_corrector_loop.py [--refresh-stride 20] [--anchors p1,p2]
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.config import BiochemConfig, PhysicsConfig, NodeFeat
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.core_physics.comsol_surface_deposition import DepositionConstants
from src.data_gen.lib.graph_velocity_priors import (
    smooth_width_nd_on_edges, width_nd_to_radius_nd, mass_conserving_umax_nd)
from src.utils.kinematics_inference import (
    load_kinematics_predictor, predict_kinematics, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b
import scripts.s1c_soft_eval as s1c
import scripts.s2_deploy_forward as s2
import scripts.s2_kine_flow_test as kft
import scripts.spfsr_lib as spfsr

OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s3_corrector_loop.json"
N_SUB = 4
COMPLETE_FRAMES = 201
TRAIN_FRAMES = 5
FEATS = ["wls_log", "wf_log", "speed_log", "speed_ring", "sdf", "width"]


def _predict_uv(model, d, x, dev):
    b = d.clone(); b.x = x
    with torch.no_grad():
        pred = predict_kinematics(model, b.to(dev))
    return pred[:, 0].detach(), pred[:, 1].detach()


def _occlude(d, clot_mask, pos, wall_np, old_sdf, dev):
    """Geometry occlusion: clot -> solid. Returns occluded node features x and new sdf/width."""
    x = d.x.clone()
    if clot_mask is None or int(clot_mask.sum()) == 0:
        return x, old_sdf, d.x[:, NodeFeat.WIDTH_ND].reshape(-1).to(dev) if x.shape[1] > NodeFeat.WIDTH_ND.start else (2.0 * old_sdf)
    solid = wall_np | clot_mask.cpu().numpy()
    dist, _ = cKDTree(pos[solid]).query(pos)
    new_sdf = torch.tensor(dist, dtype=torch.float32, device=dev).clamp_min(1e-6)
    width_new = smooth_width_nd_on_edges((2.0 * new_sdf).view(-1, 1), d.edge_index, d.num_nodes).reshape(-1)
    x[:, NodeFeat.SDF] = new_sdf.view(-1, 1)
    if x.shape[1] > NodeFeat.WIDTH_ND.start:
        x[:, NodeFeat.WIDTH_ND] = width_new.view(-1, 1)
    R0 = width_nd_to_radius_nd(2.0 * old_sdf); R1 = width_nd_to_radius_nd(width_new)
    scale = (mass_conserving_umax_nd(R1) / mass_conserving_umax_nd(R0)).clamp(0.2, 5.0)
    x[:, NodeFeat.UV_PRIOR] = d.x[:, NodeFeat.UV_PRIOR] * scale.view(-1, 1)
    x[torch.tensor(solid, device=dev), NodeFeat.UV_PRIOR] = 0.0
    return x, new_sdf, width_new


def _feats(d, u, v, sdf, width, ring, dev):
    speed = torch.sqrt(u ** 2 + v ** 2)
    return np.stack([
        np.log1p(kft.wls_shear_uv(d, u, v, dev).cpu().numpy().clip(0)),
        np.log1p(kft.wallfunc_shear_uv(d, u, v, dev).cpu().numpy().clip(0)),
        np.log1p(speed.cpu().numpy().clip(0)), ring(speed).cpu().numpy(),
        sdf.cpu().numpy(), width.cpu().numpy(),
    ], axis=1)


def build_anchor(a, cfg, phys, dev, model):
    d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
    sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
    rp, ap = s2._resting_bulk(d, cfg, dev)
    gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
    wall = d.mask_wall.reshape(-1).bool()
    sr = spfsr.aligned(d, dev, a)["sr"]
    pos = d.x[:, NodeFeat.XY].cpu().numpy(); wall_np = wall.cpu().numpy()
    old_sdf = d.x[:, NodeFeat.SDF].reshape(-1).to(dev)
    ring = s1b._ring_op(d, dev)
    # training pairs: occlude with GT clot at a few frames -> (occluded-flow feats, exact spf.sr)
    Xtr, ytr = [], []
    frames = np.unique(np.linspace(int(0.3 * T), T - 1, TRAIN_FRAMES).astype(int))
    for f in frames:
        clot = gt_clot_phi_at_time(d, int(f), phys, device=dev).reshape(-1).bool()
        x, sdf, width = _occlude(d, clot, pos, wall_np, old_sdf, dev)
        u, v = _predict_uv(model, d, x, dev)
        X = _feats(d, u, v, sdf, width, ring, dev)
        Xtr.append(X[wall_np]); ytr.append(np.log1p(sr[int(f)].cpu().numpy().clip(0))[wall_np])
    return dict(d=d, sp=sp, rp=rp, ap=ap, gt=gt, wall=wall, sr=sr, T=T, pos=pos,
                wall_np=wall_np, old_sdf=old_sdf, ring=ring,
                Xtr=np.concatenate(Xtr), ytr=np.concatenate(ytr))


def _gate_from_flow(c, dev, u, v, sdf, width, *, readout, rf, scaler, lss, j):
    N = int(c["d"].num_nodes); d = c["d"]
    if readout == "exact":
        return (c["sr"][j] < lss).float()
    if readout == "wallfunc":
        return (kft.wallfunc_shear_uv(d, u, v, dev) < lss).float()
    X = _feats(d, u, v, sdf, width, c["ring"], dev)
    pred = np.full(N, 1e9)
    pred[c["wall_np"]] = np.expm1(rf.predict(scaler.transform(X[c["wall_np"]])))
    return torch.from_numpy(pred < lss).float().to(dev)


def build_gate(c, cfg, dev, model, *, readout, occ, refresh, rf=None, scaler=None):
    """Build a per-frame gate g_low[T,N] for non-produced modes (exact / oracle-occlusion /
    frozen), then score it through the *validated* s2._integrate_closed_loop."""
    d = c["d"]; T = c["T"]; N = int(d.num_nodes); lss = float(cfg.lss)
    g = torch.zeros(T, N, device=dev)
    u, v = _predict_uv(model, d, d.x.clone(), dev)
    width0 = (d.x[:, NodeFeat.WIDTH_ND].reshape(-1).to(dev) if d.x.shape[1] > NodeFeat.WIDTH_ND.start
              else 2.0 * c["old_sdf"])
    gl = _gate_from_flow(c, dev, u, v, c["old_sdf"], width0, readout=readout, rf=rf, scaler=scaler, lss=lss, j=0)
    for j in range(T):
        if readout == "exact":
            gl = (c["sr"][j] < lss).float()
        elif occ == "oracle" and (j % refresh == 0 or j == T - 1):
            clot = gt_clot_phi_at_time(d, j, PhysicsConfig(phase="biochem"), device=dev).reshape(-1).bool()
            x, sdf, width = _occlude(d, clot, c["pos"], c["wall_np"], c["old_sdf"], dev)
            u, v = _predict_uv(model, d, x, dev)
            gl = _gate_from_flow(c, dev, u, v, sdf, width, readout=readout, rf=rf, scaler=scaler, lss=lss, j=j)
        g[j] = gl
    return s2._integrate_closed_loop(d, cfg, dev, g, c["ap"], c["rp"], c["sp"]["step2t"], c["sp"]["t_s"])


def rollout_produced(c, cfg, dev, model, *, readout, refresh, rf=None, scaler=None):
    """Real deployable loop: progressive PRODUCED clot occludes the flow online (gate depends
    on running Mat). Same deposition scheme as s2._integrate_closed_loop."""
    d, sp = c["d"], c["sp"]; T = c["T"]
    k = DepositionConstants.si(cfg); Da, Minf = float(cfg.surface_damkohler), float(cfg.Minf)
    crit, lss = float(cfg.viscosity_mat_crit), float(cfg.lss)
    t_s, step2t = sp["t_s"], sp["step2t"]
    dep_const = k.k_rs * c["rp"] + k.k_as * c["ap"]
    N = int(d.num_nodes)
    u, v = _predict_uv(model, d, d.x.clone(), dev)
    width0 = (d.x[:, NodeFeat.WIDTH_ND].reshape(-1).to(dev) if d.x.shape[1] > NodeFeat.WIDTH_ND.start
              else 2.0 * c["old_sdf"])
    gl = _gate_from_flow(c, dev, u, v, c["old_sdf"], width0, readout=readout, rf=rf, scaler=scaler, lss=lss, j=0)
    M = torch.zeros(N, device=dev); Mas = torch.zeros(N, device=dev); Mat = torch.zeros(N, device=dev)
    for j in range(1, T):
        dt = float(t_s[j] - t_s[j - 1]) / N_SUB; s2t = float(step2t[j])
        for _ in range(N_SUB):
            avail = (1.0 - (M + Mas + Mat) / Minf).clamp(0.0, 1.0)
            cdep = avail * gl * dep_const; cauto = gl * (Mas / Minf) * k.k_aa * c["ap"]
            dMas = Da * s2t * cdep; dMat = Da * s2t * (cdep + cauto)
            M = M + dt * dMas; Mas = Mas + dt * dMas; Mat = Mat + dt * dMat
        if j % refresh == 0 or j == T - 1:
            x, sdf, width = _occlude(d, (Mat >= crit), c["pos"], c["wall_np"], c["old_sdf"], dev)
            u, v = _predict_uv(model, d, x, dev)
            gl = _gate_from_flow(c, dev, u, v, sdf, width, readout=readout, rf=rf, scaler=scaler, lss=lss, j=j)
    return Mat


def main():
    refresh = 20; anchors = None
    for a in sys.argv[1:]:
        if a.startswith("--refresh-stride"):
            refresh = int(a.split("=")[-1])
        elif a.startswith("--anchors"):
            anchors = a.split("=")[-1].split(",")
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); s1c.cfg = cfg
    crit = float(cfg.viscosity_mat_crit)
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    if anchors is None:
        anchors = [a for a in sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
                   if spfsr.has_cache(a)]
    print(f"[i] anchors={len(anchors)} refresh={refresh}\n[i] building per-anchor occluded-flow training pairs...")
    C = {a: build_anchor(a, cfg, phys, dev, model) for a in anchors}

    print(f"\n{'patient':<11}{'fr':>5}{'exact':>8}{'orc+corr':>9}{'orc+wf':>8}{'prog+corr':>10}{'frozen':>8}{'label':>8}")
    rep = {"refresh": refresh, "per_patient": {}}
    comp = {k: [] for k in ["exact", "orc_corr", "orc_wf", "prog_corr", "frozen", "label"]}
    for a in anchors:
        c = C[a]; d, sp = c["d"], c["sp"]; T = c["T"]; gt, wall = c["gt"], c["wall"]
        Xtr = np.concatenate([C[b]["Xtr"] for b in anchors if b != a])
        ytr = np.concatenate([C[b]["ytr"] for b in anchors if b != a])
        scaler = StandardScaler().fit(Xtr)
        rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=0, n_jobs=-1).fit(scaler.transform(Xtr), ytr)

        def sc(mat):
            return s1c._scores(d, mat, gt, wall, crit)["swept_best_f1"]
        f_ex = sc(build_gate(c, cfg, dev, model, readout="exact", occ="none", refresh=refresh))
        f_oc = sc(build_gate(c, cfg, dev, model, readout="corrector", occ="oracle", refresh=refresh, rf=rf, scaler=scaler))
        f_ow = sc(build_gate(c, cfg, dev, model, readout="wallfunc", occ="oracle", refresh=refresh))
        f_pc = sc(rollout_produced(c, cfg, dev, model, readout="corrector", refresh=refresh, rf=rf, scaler=scaler))
        f_fr = sc(build_gate(c, cfg, dev, model, readout="wallfunc", occ="none", refresh=refresh))
        lab = sc(sp["mat"][-1])
        cohort = "complete" if T >= COMPLETE_FRAMES else "early"
        rep["per_patient"][a] = dict(cohort=cohort, frames=T, exact=f_ex, orc_corr=f_oc,
                                     orc_wf=f_ow, prog_corr=f_pc, frozen=f_fr, label=lab)
        if cohort == "complete":
            for kk, vv in [("exact", f_ex), ("orc_corr", f_oc), ("orc_wf", f_ow),
                           ("prog_corr", f_pc), ("frozen", f_fr), ("label", lab)]:
                comp[kk].append(vv)
        print(f"{a:<11}{T:>5}{f_ex:>8.3f}{f_oc:>9.3f}{f_ow:>8.3f}{f_pc:>10.3f}{f_fr:>8.3f}{lab:>8.3f}")

    if comp["exact"]:
        rep["complete_mean"] = {k: float(np.mean(v)) for k, v in comp.items()}
        m = rep["complete_mean"]
        print(f"\n[complete-cohort mean] exact={m['exact']:.3f}  oracle+corr={m['orc_corr']:.3f}  "
              f"oracle+wf={m['orc_wf']:.3f}  prog+corr={m['prog_corr']:.3f}  frozen={m['frozen']:.3f}  label={m['label']:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
