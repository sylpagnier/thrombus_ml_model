"""S3 probe: is COUPLED spf.sr predictable from GEOMETRY (occluded by the clot), bypassing the
kine flow? Decides the architecture: if global-geometry features >> wallfunc AUC, a GNN shear
surrogate (global receptive field over mesh+clot) is the lever; if not, we need a coupled flow
model. LOAO, final-frame oracle occlusion, low-shear membership AUC on wall nodes.
Run: python scripts/s3_geom_shear_probe.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from src.config import BiochemConfig, PhysicsConfig, NodeFeat
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.utils.kinematics_inference import (
    load_kinematics_predictor, resolve_kinematics_checkpoint)
import scripts.s1b_gate_variants as s1b
import scripts.s2_kine_flow_test as kft
import scripts.spfsr_lib as spfsr
import scripts.s3_corrector_loop as cl

COMPLETE = 201


def multiscale_sdf(d, sdf, dev, hops=(1, 2, 4, 8)):
    ring = s1b._ring_op(d, dev)
    feats = [sdf.cpu().numpy()]
    cur = sdf
    last = 0
    for h in hops:
        for _ in range(h - last):
            cur = ring(cur)
        last = h
        feats.append(cur.cpu().numpy())
    return np.stack(feats, 1)


def main():
    cfg = BiochemConfig(phase="biochem"); phys = PhysicsConfig(phase="biochem")
    dev = torch.device("cpu"); lss = float(cfg.lss)
    model = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev, phys_cfg=PhysicsConfig(phase="kinematics"))
    anchors = [a for a in sorted(p.stem for p in s1b.ANCHOR_DIR.glob("patient*.pt") if "_metadata" not in p.stem)
               if spfsr.has_cache(a)]
    C = {}
    for a in anchors:
        d = torch.load(s1b.ANCHOR_DIR / f"{a}.pt", map_location=dev, weights_only=False)
        T = d.y.shape[0]; wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
        pos = d.x[:, NodeFeat.XY].cpu().numpy()
        old_sdf = d.x[:, NodeFeat.SDF].reshape(-1).to(dev)
        clot = gt_clot_phi_at_time(d, T - 1, phys, device=dev).reshape(-1).bool()
        x, sdf, width = cl._occlude(d, clot, pos, wall, old_sdf, dev)
        u, v = cl._predict_uv(model, d, x, dev)
        wf = np.log1p(kft.wallfunc_shear_uv(d, u, v, dev).cpu().numpy().clip(0))
        ms = multiscale_sdf(d, sdf, dev)
        sr = spfsr.aligned(d, dev, a)["sr"][-1].cpu().numpy()
        Xloc = np.stack([sdf.cpu().numpy(), width.cpu().numpy()], 1)
        Xglob = np.concatenate([ms, width.cpu().numpy()[:, None], pos], 1)   # multiscale geom + position
        C[a] = dict(T=T, wall=wall, wf=wf, Xloc=Xloc, Xglob=Xglob, y=(sr < lss).astype(int))

    def loao_auc(a, key):
        Xtr = np.concatenate([C[b][key][C[b]["wall"]] for b in anchors if b != a])
        ytr = np.concatenate([C[b]["y"][C[b]["wall"]] for b in anchors if b != a])
        if len(np.unique(ytr)) < 2:
            return float("nan")
        sc = StandardScaler().fit(Xtr)
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(sc.transform(Xtr), ytr)
        w = C[a]["wall"]; yt = C[a]["y"][w]
        if not (0 < yt.sum() < len(yt)):
            return float("nan")
        return roc_auc_score(yt, clf.predict_proba(sc.transform(C[a][key][w]))[:, 1])

    print(f"{'patient':<11}{'fr':>5}{'wf_auc':>8}{'geom_loc':>9}{'geom_glob':>10}")
    comp = {"wf": [], "loc": [], "glob": []}
    for a in anchors:
        w = C[a]["wall"]; yt = C[a]["y"][w]
        wf_auc = roc_auc_score(yt, -C[a]["wf"][w]) if 0 < yt.sum() < len(yt) else float("nan")
        loc = loao_auc(a, "Xloc"); glob = loao_auc(a, "Xglob")
        if C[a]["T"] >= COMPLETE:
            comp["wf"].append(wf_auc); comp["loc"].append(loc); comp["glob"].append(glob)
        print(f"{a:<11}{C[a]['T']:>5}{wf_auc:>8.3f}{loc:>9.3f}{glob:>10.3f}")
    print(f"\n[complete-cohort mean AUC] wallfunc={np.nanmean(comp['wf']):.3f}  "
          f"geom_local={np.nanmean(comp['loc']):.3f}  geom_global={np.nanmean(comp['glob']):.3f}")
    print("[i] geom_global >> wallfunc  => GNN shear surrogate (mesh+clot->spf.sr) is the lever.")


if __name__ == "__main__":
    main()
