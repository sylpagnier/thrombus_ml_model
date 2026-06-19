"""Which DEPLOYABLE shear proxy best predicts the exact low-shear gate at the wall?

Targets the COMSOL gate (spf.sr < lss) on wall nodes and ranks candidate proxies
(all available at deploy from geometry + kine flow) by AUC and rank correlation.

Run: python scripts/_diag_shear_proxies.py
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import BiochemConfig  # noqa: E402
import scripts.s1_kaa_closure_generalization as s1  # noqa: E402
import scripts._diag_shear_resolution as dr  # noqa: E402


def auc(score, target):
    """Rank AUC: P(score higher on positives)."""
    t = target.astype(bool)
    if t.sum() == 0 or (~t).sum() == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = t.sum(); n0 = (~t).sum()
    return float((ranks[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    cfg = BiochemConfig(phase="biochem"); dev = torch.device("cpu")
    d = torch.load(dr.GRAPH, map_location=dev, weights_only=False)
    d_bar = float(d.d_bar.view(-1)[0]); lss = float(cfg.lss)
    x_cm, y_cm, sr_exp, _, _ = dr.load_export()
    exp_nd = np.stack([x_cm * dr.CM_TO_M / d_bar, y_cm * dr.CM_TO_M / d_bar], axis=1)
    g_nd = d.x[:, :2].cpu().numpy()
    _, idx = cKDTree(exp_nd).query(g_nd, k=1)
    sr_exact = sr_exp[idx]                      # [N,4] exact 1/s
    b = dr.NTB - 1
    target = (sr_exact[:, b] < lss)            # gate ON = positive

    wall = d.mask_wall.reshape(-1).bool().cpu().numpy()
    x = d.x.cpu().numpy()
    # feature columns (comma-split schema): see x_channel_names
    sdf, shear_pot = x[:, 2], x[:, 3]
    u_pr, v_pr, mu_pr, wss_pr, width = x[:, 11], x[:, 12], x[:, 13], x[:, 14], x[:, 15]
    speed = np.sqrt(u_pr ** 2 + v_pr ** 2)
    speed_over_w = speed / (width + 1e-6)

    sr_wls = s1._shear_series(d, dev)[0][dr.GTIDX[b]].cpu().numpy()
    # WLS shear averaged over interior (non-wall) 1-hop neighbours (shift off no-slip wall)
    ei = d.edge_index.cpu().numpy(); src, dst = ei[0], ei[1]
    interior = ~wall
    acc = np.zeros(len(sr_wls)); cnt = np.zeros(len(sr_wls))
    for s_, t_ in ((src, dst), (dst, src)):
        m = interior[s_]
        np.add.at(acc, t_[m], sr_wls[s_[m]]); np.add.at(cnt, t_[m], 1.0)
    sr_wls_ring = np.where(cnt > 0, acc / np.maximum(cnt, 1), sr_wls)

    # exact shear at interior ring (upper bound for the 'shift off wall' idea)
    acc2 = np.zeros(len(sr_wls)); cnt2 = np.zeros(len(sr_wls))
    sre = sr_exact[:, b]
    for s_, t_ in ((src, dst), (dst, src)):
        m = interior[s_]
        np.add.at(acc2, t_[m], sre[s_[m]]); np.add.at(cnt2, t_[m], 1.0)
    sr_exact_ring = np.where(cnt2 > 0, acc2 / np.maximum(cnt2, 1), sre)

    # proxies: higher proxy -> higher shear -> gate OFF, so AUC uses -proxy
    proxies = {
        "WLS sr (wall, baseline)": sr_wls,
        "WLS sr (interior ring)": sr_wls_ring,
        "wss_prior_nd": wss_pr,
        "shear_potential": shear_pot,
        "speed_prior/width": speed_over_w,
        "speed_prior": speed,
        "mu_prior_nd (inv)": -mu_pr,
        "exact sr (interior ring)*": sr_exact_ring,
        "exact spf.sr (ceiling)*": sre,
    }
    w = wall
    pos = int(target[w].sum()); tot = int(w.sum())
    print(f"[i] wall nodes={tot}  exact low-shear (gate ON)={pos} ({pos/tot:.2%})")
    print(f"\n{'proxy':<30}{'AUC':>8}{'spearman|sr':>13}")
    sre_w = sre[w]
    rows = []
    for name, p in proxies.items():
        a = auc(-p[w], target[w])
        sp = spearmanr(p[w], sre_w).correlation
        rows.append((name, a, sp))
    for name, a, sp in sorted(rows, key=lambda r: -(r[1] if np.isfinite(r[1]) else 0)):
        star = "  <- not deployable" if name.endswith("*") else ""
        print(f"{name:<30}{a:>8.3f}{sp:>13.3f}{star}")


if __name__ == "__main__":
    main()
