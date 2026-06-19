"""S1c pre-S2 evaluation: soft/threshold-robust scoring + clot-weighted closure fit.

Addresses the pre-S2 diagnostic findings (docs/SPECIES_LEARNING_STRATEGY.md S5):
  #1 soft score  - the hard `Mat>=crit` cliff flips marginal clots (GTmat~crit) to F1=0
                   under a small reconstruction undershoot. Report a threshold-swept
                   best-F1 (footprint quality, magnitude-robust) + soft-Dice alongside it.
  #2 clot-weighted fit - the global lstsq is drowned by non-clot zeros and undershoots the
                   few high-growth nodes; weight the regression by surface platelet density
                   (Mas), which focuses the closure where deposition actually happens.

Law = deposition + autocat with NO shear gate (gate=1): the gate is net-harmful under
oracle Mas and can only be judged at S2. Cohorts: "complete" sims (full 201-frame /
30000 s solve) drive the headline; "early" sims (solver-terminated) are reported separately.

Run: python scripts/s1c_soft_eval.py
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
import scripts.s1b_gate_variants as s1b  # noqa: E402

OUT = ROOT / "outputs" / "reports" / "comsol_validation" / "s1c_soft_eval.json"
COMPLETE_FRAMES = 201            # full solve length
TAU = 0.25                       # soft-gel band (fraction of crit)


def _fit(sp, g_low, weighted):
    """Fit (c_dep,k_aa); weighted=True -> WLS weighted by Mas (surface platelet density)."""
    Da, krs, kas = cfg.surface_damkohler, cfg.k_rs, cfg.k_as
    step2t = sp["step2t"]
    dep = Da * g_low * (sp["sat"] * (krs * sp["rp"] + kas * sp["ap"])) * step2t
    auto = g_low * (sp["mas"] / sp["Minf"]) * sp["ap"] * step2t
    s2 = step2t.expand_as(sp["mat"]); msk = (s2 > 0.5) & torch.isfinite(sp["dmat"])
    A = torch.stack([dep[msk], auto[msk]], 1).cpu().numpy()
    b = sp["dmat"][msk].cpu().numpy()
    if weighted:
        w = (sp["mas"] / sp["Minf"])[msk].clamp(min=1e-4).sqrt().cpu().numpy()
        A = A * w[:, None]; b = b * w
    ok = np.all(np.isfinite(A), 1) & np.isfinite(b)
    coef, *_ = np.linalg.lstsq(A[ok], b[ok], rcond=None)
    c_dep, k_aa = float(coef[0]), float(coef[1])
    rate = torch.nan_to_num(c_dep * dep + k_aa * auto)
    t = sp["t_s"]; dt = (t[1:] - t[:-1]).reshape(-1, 1)
    incr = 0.5 * (rate[1:] + rate[:-1]) * dt
    mat_rec = sp["mat"][:1] + torch.cat(
        [torch.zeros(1, rate.shape[1]), torch.cumsum(incr, 0)], 0)
    return mat_rec[-1]


def _f1(pred, gt):
    tp = float((pred & gt).sum()); p = tp / max(float(pred.sum()), 1)
    r = tp / max(float(gt.sum()), 1)
    return 2 * p * r / max(p + r, 1e-9), p, r


def _scores(d, mat_final, gt, wall, crit):
    """hard F1@crit, threshold-swept best-F1, soft-Dice (all wall-nucleate + dilate-1)."""
    ei = d.edge_index
    # hard @ crit
    hp = s1b._dilate((mat_final >= crit) & wall, ei, 1)
    hardF1 = _f1(hp, gt)[0]
    # threshold sweep (magnitude-robust footprint quality)
    best = 0.0; best_thr = crit
    ratio = (mat_final / crit).clamp(min=0)
    for thr in np.geomspace(0.3, 3.0, 24):
        pp = s1b._dilate((ratio >= thr) & wall, ei, 1)
        f = _f1(pp, gt)[0]
        if f > best:
            best, best_thr = f, float(thr)
    # soft-Dice (no threshold): sigmoid band around crit, wall nodes
    p = torch.sigmoid((ratio - 1.0) / TAU) * wall.float()
    g = gt.float()
    dice = float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-9))
    return {"hard_f1": hardF1, "swept_best_f1": best, "swept_thr": best_thr, "soft_dice": dice}


def main():
    global cfg
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); crit = float(cfg.viscosity_mat_crit)
    anchors = sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    rep = {"complete": [], "early": [], "per_patient": {}}
    print(f"{'patient':<11}{'cohort':>9}{'old hardF1':>12}{'new hardF1':>12}"
          f"{'sweptF1':>10}{'softDice':>10}{'thr/crit':>10}")
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        wall = d.mask_wall.reshape(-1).bool()
        ones = torch.ones(T, sp["mat"].shape[1])
        mat_old = _fit(sp, ones, weighted=False)     # global lstsq (pre-fix)
        mat_new = _fit(sp, ones, weighted=True)       # clot-weighted (#2)
        old_hard = _f1(s1b._dilate((mat_old >= crit) & wall, d.edge_index, 1), gt)[0]
        sc = _scores(d, mat_new, gt, wall, crit)      # #1 soft scoring on #2 fit
        cohort = "complete" if T >= COMPLETE_FRAMES else "early"
        rep[cohort].append(sc); rep["per_patient"][a] = {"cohort": cohort, "old_hard_f1": old_hard, **sc}
        print(f"{a:<11}{cohort:>9}{old_hard:>12.3f}{sc['hard_f1']:>12.3f}"
              f"{sc['swept_best_f1']:>10.3f}{sc['soft_dice']:>10.3f}{sc['swept_thr']:>10.2f}")

    def mean(rows, k): return float(np.mean([r[k] for r in rows])) if rows else float("nan")
    print(f"\n{'COHORT':<11}{'n':>4}{'hardF1':>10}{'sweptF1':>10}{'softDice':>10}")
    for c in ("complete", "early"):
        print(f"{c:<11}{len(rep[c]):>4}{mean(rep[c],'hard_f1'):>10.3f}"
              f"{mean(rep[c],'swept_best_f1'):>10.3f}{mean(rep[c],'soft_dice'):>10.3f}")
    rep["headline_complete"] = {"swept_best_f1": mean(rep["complete"], "swept_best_f1"),
                                "soft_dice": mean(rep["complete"], "soft_dice"),
                                "hard_f1": mean(rep["complete"], "hard_f1")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"\n[save] {OUT}")
    print(f"[i] HEADLINE (complete sims) swept-F1={rep['headline_complete']['swept_best_f1']:.3f}  "
          f"soft-Dice={rep['headline_complete']['soft_dice']:.3f}")


if __name__ == "__main__":
    main()
