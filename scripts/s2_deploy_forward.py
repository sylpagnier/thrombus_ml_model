"""S2: deployable closed-loop forward solve of the validated surface clot law.

The S2 diagnostics (scripts/s2_deploy_cascade.py, _diag_bulk_flat.py) proved the bulk
cascade does NOT need an ADR solve: GT resting platelets RP are exactly constant and
activated AP stays 0.2-1.6x of its resting IC (mildly depleted on clot, never blows up).
The deposition law uses only rp/ap/Mas -- thrombin/agonists only feed activation, which
barely fires locally. So the deployable recipe freezes rp=c_RP0, ap=ap_rest from the IC
and integrates ONLY the surface system (M, Mas, Mat) -- the COMSOL wall ODE
(biochem_wall_residual), which is parameter-free (Da=1e-4 + cfg adhesion rates):

  dMas/dt = dM/dt = Da*step2t*avail*g_low*(k_rs*rp + k_as*ap)
  dMat/dt = dMas/dt + Da*step2t*g_low*(Mas/Minf)*k_aa*ap
  avail   = 1 - (M+Mas+Mat)/Minf,   M == Mas

This is the FIRST test where Mas is *produced* (closed-loop autocat), not oracle -- so the
low-shear gate finally matters (under oracle Mas in S1/S1b it was net-harmful/masked).
No GT is used except to score; inputs are IC + geometry + flow shear (gate). Gate sources:
  wls      : on-graph WLS shear from GT flow  (oracle-flow reference)
  carreau  : analytic Carreau wall shear from geometry only (fully deployable, flow-free)
  wallfunc : near-wall wall-function shear from flow         (deployable w/ kine flow)

Scoring = S1c soft/threshold-robust (hard-F1@crit, threshold-swept best-F1, soft-Dice),
wall-nucleate + 1-hop dilate, complete vs early cohorts. Compared to the S1c oracle-Mas
headline and the species-GNN deploy baseline.

Run: python scripts/s2_deploy_forward.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.comsol_surface_deposition import DepositionConstants  # noqa: E402
from src.core_physics.t0_rung4_ladder import resting_species_log_nd  # noqa: E402
import scripts.s1b_gate_variants as s1b  # noqa: E402
import scripts.s1c_soft_eval as s1c  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s2_deploy_forward.json"
COMPLETE_FRAMES = 201
N_SUB = 4                         # sub-steps per frame (surface ODE is slow; few suffice)
GNN_BASELINE_P007 = 0.70         # biochem_deploy species-GNN deploy F1 on patient007 (AGENTS.md)


def _resting_bulk(d, cfg, dev):
    """Deployable constant rp, ap from the resting IC (no GT)."""
    sc = cfg.get_species_scales(device=dev)
    rest = torch.expm1(resting_species_log_nd(d, dev)[:, :9].clamp(-10, 8))   # nd [N,9]
    rp = rest[:, 0] * float(sc[0])      # RP working units [N]
    ap = rest[:, 1] * float(sc[1])      # AP working units [N]
    return rp, ap


def _integrate_closed_loop(d, cfg, dev, g_low, ap, rp, step2t, t_s, n_sub=N_SUB):
    """Closed-loop surface ODE; Mas produced (autocat feedback). Returns Mat_final [N]."""
    k = DepositionConstants.si(cfg)
    Da, Minf = float(cfg.surface_damkohler), float(cfg.Minf)
    N = g_low.shape[1]
    M = torch.zeros(N, device=dev); Mas = torch.zeros(N, device=dev); Mat = torch.zeros(N, device=dev)
    dep_rate_const = k.k_rs * rp + k.k_as * ap          # [N]
    T = g_low.shape[0]
    for j in range(1, T):
        dt = float(t_s[j] - t_s[j - 1]) / n_sub
        gl = g_low[j]; s2 = float(step2t[j])
        for _ in range(n_sub):
            avail = (1.0 - (M + Mas + Mat) / Minf).clamp(0.0, 1.0)
            common_dep = avail * gl * dep_rate_const                       # R_M (RP+AP adhesion)
            common_auto = gl * (Mas / Minf) * k.k_aa * ap                  # autocatalytic
            dMas = Da * s2 * common_dep
            dMat = Da * s2 * (common_dep + common_auto)
            M = M + dt * dMas; Mas = Mas + dt * dMas; Mat = Mat + dt * dMat
    return Mat


def _shear_cache(d, dev):
    """Raw shear sources reused by both the hard gates and the learned-gate features."""
    return {"wls": s1b._wls_shear(d, dev)[0],
            "carreau": s1b._carreau_shear(d, dev),
            "wallfunc": s1b._wallfunc_shear(d, dev)}


def _hard_gates(d, srs, cfg):
    """g_low[T,N] in {0,1} for each shear source (hard low-shear threshold)."""
    T = d.y.shape[0]; lss = float(cfg.lss)
    return {
        "wls": (srs["wls"] < lss).float(),
        "carreau": (srs["carreau"].reshape(1, -1).expand(T, -1) < lss).float(),
        "wallfunc": (srs["wallfunc"] < lss).float(),
    }


def _learned_gates(cache, anchors, phys, dev):
    """Leave-one-anchor-out logistic gate (soft prob in [0,1]); deployable features (S1b)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception:
        print("[warn] sklearn missing; skipping learned gate")
        return None
    feat, tgt, wmask = {}, {}, {}
    for a in anchors:
        d, sp, srs = cache[a]
        feats, wall = s1b._learned_features(d, sp, srs, dev)
        gt = gt_clot_phi_at_time(d, d.y.shape[0] - 1, phys, device=dev).reshape(-1).cpu().numpy().astype(bool)
        feat[a] = feats; tgt[a] = (gt & wall); wmask[a] = wall
    out = {}
    for a in anchors:
        Xtr = np.concatenate([feat[b][wmask[b]] for b in anchors if b != a])
        ytr = np.concatenate([tgt[b][wmask[b]] for b in anchors if b != a])
        if ytr.sum() == 0:
            out[a] = None; continue
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
        prob = np.zeros(len(wmask[a]))
        prob[wmask[a]] = clf.predict_proba(scaler.transform(feat[a][wmask[a]]))[:, 1]
        out[a] = torch.from_numpy(prob).float().to(dev)
    return out


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); crit = float(cfg.viscosity_mat_crit)
    s1c.cfg = cfg
    anchors = sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    sources = ["wls", "carreau", "wallfunc", "learned", "combo"]
    print(f"[i] anchors={len(anchors)}  crit={crit:.3g}  lss={cfg.lss}  n_sub={N_SUB}")
    print(f"[i] Da={cfg.surface_damkohler}  k_aa(SI)={DepositionConstants.si(cfg).k_aa:.4g}  "
          f"k_rs={DepositionConstants.si(cfg).k_rs:.4g}  k_as={DepositionConstants.si(cfg).k_as:.4g}\n")

    cache = {}
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        cache[a] = (d, s1b._species(d, cfg, dev), _shear_cache(d, dev))
    learned = _learned_gates(cache, anchors, phys, dev)        # LOAO soft gate (or None)

    rep = {"sources": sources, "per_patient": {}, "complete": {s: [] for s in sources},
           "early": {s: [] for s in sources}}
    hdr = f"{'patient':<11}{'cohort':>8}" + "".join(f"{s+'_sw':>11}" for s in sources)
    print(hdr)
    for a in anchors:
        d, sp, srs = cache[a]; T = sp["mat"].shape[0]
        rp, ap = _resting_bulk(d, cfg, dev)
        gates = _hard_gates(d, srs, cfg)
        if learned is not None and learned.get(a) is not None:
            # Hard threshold: under produced Mas a broad soft gate over-gels (precision
            # collapse); a crisp membership gives the learned predictor its S1b footprint.
            gates["learned"] = (learned[a] > 0.5).float().reshape(1, -1).expand(T, -1)
            # combo = union of the two deployable gates (carreau geometry OR learned);
            # carreau collapses (analytic shear never < lss) are rescued by the learned gate.
            gates["combo"] = torch.clamp(gates["carreau"] + gates["learned"], 0.0, 1.0)
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        wall = d.mask_wall.reshape(-1).bool()
        cohort = "complete" if T >= COMPLETE_FRAMES else "early"
        rep["per_patient"][a] = {"cohort": cohort, "n_frames": T, "gt_clot": int(gt.sum())}
        line = f"{a:<11}{cohort:>8}"
        for s in sources:
            if s not in gates:
                line += f"{'-':>11}"; continue
            mat_final = _integrate_closed_loop(d, cfg, dev, gates[s], ap, rp, sp["step2t"], sp["t_s"])
            sc = s1c._scores(d, mat_final, gt, wall, crit)
            sc["mat_med_clot"] = float(mat_final[gt].median()) if int(gt.sum()) else 0.0
            rep["per_patient"][a][s] = sc
            rep[cohort][s].append(sc)
            line += f"{sc['swept_best_f1']:>11.3f}"
        print(line)

    def mean(rows, k): return float(np.mean([r[k] for r in rows])) if rows else float("nan")
    print(f"\n{'== COHORT MEANS ==':<20}")
    print(f"{'cohort':<10}{'src':<10}{'hardF1':>9}{'sweptF1':>9}{'softDice':>10}")
    for c in ("complete", "early"):
        for s in sources:
            print(f"{c:<10}{s:<10}{mean(rep[c][s],'hard_f1'):>9.3f}"
                  f"{mean(rep[c][s],'swept_best_f1'):>9.3f}{mean(rep[c][s],'soft_dice'):>10.3f}")
    rep["headline_complete"] = {
        s: {"hard_f1": mean(rep["complete"][s], "hard_f1"),
            "swept_best_f1": mean(rep["complete"][s], "swept_best_f1"),
            "soft_dice": mean(rep["complete"][s], "soft_dice")} for s in sources}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"\n[save] {OUT}")
    best = max(sources, key=lambda s: rep["headline_complete"][s]["swept_best_f1"])
    h = rep["headline_complete"][best]
    print(f"[i] best deployable gate (complete): {best}  swept-F1={h['swept_best_f1']:.3f}  "
          f"hard-F1={h['hard_f1']:.3f}  soft-Dice={h['soft_dice']:.3f}")
    print(f"[i] reference: S1c oracle-Mas headline swept-F1=0.863 ; species-GNN deploy p007 F1~{GNN_BASELINE_P007}")


if __name__ == "__main__":
    main()
