"""Diagnostic: exact-shear gate F1 across ALL anchors (true physics ceiling everywhere).

Uses the exported COMSOL spf.sr (time-varying, coupled) fed into the validated closed-loop
deposition law. This is the ceiling our deployable ML shear corrector should chase. Reports:
  label   = perfect Mat (hard upper bound)
  coupled = exact time-varying spf.sr gate  (the realistic physics ceiling; p007 was 0.77)
  frozen  = t0 spf.sr broadcast             (initial-flow only; p007 was ~0.49)
Run: python scripts/s3_exact_gate_all.py
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
import scripts.s1b_gate_variants as s1b
import scripts.s1c_soft_eval as s1c
import scripts.s2_deploy_forward as s2
import scripts.spfsr_lib as spfsr

OUT = Path(__file__).resolve().parents[1] / "outputs" / "reports" / "comsol_validation" / "s3_exact_gate_all.json"
COMPLETE_FRAMES = 201


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); s1c.cfg = cfg
    crit, lss = float(cfg.viscosity_mat_crit), float(cfg.lss)
    anchors = sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
    print(f"[i] lss={lss}  crit={crit:.3g}\n")
    print(f"{'patient':<12}{'frames':>7}{'label':>8}{'coupled':>9}{'frozen':>8}{'gap_couple':>11}")
    rep = {"per_patient": {}}
    comp = {"label": [], "coupled": [], "frozen": []}
    for a in anchors:
        if not spfsr.has_cache(a):
            print(f"{a:<12}  no cache"); continue
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        sp = s1b._species(d, cfg, dev); T = sp["mat"].shape[0]
        rp, ap = s2._resting_bulk(d, cfg, dev)
        gt = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        wall = d.mask_wall.reshape(-1).bool()
        sr = spfsr.aligned(d, dev, a)["sr"]                       # [T,N] exact, aligned
        g_couple = (sr < lss).float()
        g_frozen = (sr[0].reshape(1, -1) < lss).float().expand(T, -1)

        lab = s1c._scores(d, sp["mat"][-1], gt, wall, crit)["swept_best_f1"]
        f_co = s1c._scores(d, s2._integrate_closed_loop(d, cfg, dev, g_couple, ap, rp, sp["step2t"], sp["t_s"]),
                           gt, wall, crit)["swept_best_f1"]
        f_fr = s1c._scores(d, s2._integrate_closed_loop(d, cfg, dev, g_frozen, ap, rp, sp["step2t"], sp["t_s"]),
                           gt, wall, crit)["swept_best_f1"]
        cohort = "complete" if T >= COMPLETE_FRAMES else "early"
        rep["per_patient"][a] = {"cohort": cohort, "frames": T, "label": lab,
                                 "coupled": f_co, "frozen": f_fr, "gt_clot": int(gt.sum())}
        if cohort == "complete":
            comp["label"].append(lab); comp["coupled"].append(f_co); comp["frozen"].append(f_fr)
        print(f"{a:<12}{T:>7}{lab:>8.3f}{f_co:>9.3f}{f_fr:>8.3f}{f_co - f_fr:>+11.3f}")

    if comp["label"]:
        rep["complete_mean"] = {k: float(np.mean(v)) for k, v in comp.items()}
        m = rep["complete_mean"]
        print(f"\n[complete-cohort mean] label={m['label']:.3f}  coupled={m['coupled']:.3f}  "
              f"frozen={m['frozen']:.3f}  coupling lever={m['coupled'] - m['frozen']:+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
