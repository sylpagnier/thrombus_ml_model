r"""Comprehensive physical EDA of the COMSOL clot data (docs/WALL_MODEL_PLAN.md s10).

Grounded in the validated COMSOL law (docs/COMSOL_PHYSICS_VALIDATION.md), not in generic
feature hunting. The mechanism there is:

    J0_Mat = Da * ( [d(spf.sr,x) < sgt] * (L/gamma_m)*|d(spf.sr,x)| * common      <- separation gate, 21% of growing nodes
                  + [spf.sr < lss]                                  * common )    <- LOW-SHEAR gate, 79.7%  (DOMINANT)
    common = Sat(M)*k_rs*rp + Sat(M)*k_as*ap + (Mas/Minf)*k_aa*ap
    J0_th  = beta*phi_at*Mat*PT                                                   <- thrombin \propto Mat => autocatalysis
    mu1(Mat): hard step 1 -> 80 at Mat = 2e7 plt/cm^2                             <- the clot label IS this step

and ~90% of Mat growth is the autocatalytic (Mas/Minf)*k_aa*AP term, only ~7% fresh
deposition. So the system is a **gated, autocatalytic surface reaction with a hard
threshold readout** -- an IGNITION problem, not a steady-state one.

Two facts about our graphs make the deployable question sharp:
  * At t=0 every species channel is spatially UNIFORM (RP, AP, PT, ... all constant).
    So clot location is determined ENTIRELY by geometry + flow -- consistent with s2.6
    ("clot initiation is unseeded"). There is no chemical seed to find.
  * mu_eff = Carreau(shear_rate) * mu1(Mat), and at t=0 mu1 == 1, so **mu_eff(t=0) is a
    monotone-decreasing function of shear rate**. The dominant gate `spf.sr < lss` is
    therefore EXACTLY a threshold on t=0 viscosity -- a quantity we already carry as a
    node feature (mu_prior_nd / y[0,:,3]). This EDA tests that directly.

Answers four questions, each at the level the modelling actually needs:
  Q1 WHERE  does clot form            -- node-level, which t=0 field is the real gate
  Q2 WHEN   does it ignite            -- node/vessel onset time, and what predicts it
  Q3 HOW MUCH / HOW THICK             -- vessel burden and deep mass (the s9.15 regime var)
  Q4 IS REGIME DEPLOYABLE-PREDICTABLE -- can t=0 features route a vessel without GT

    python scripts/eda_clot_physics.py
    python scripts/eda_clot_physics.py --anchors patient039,patient040,patient043
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
ONSET_STRIDE = 5  # grade GT clot every Nth step for onset (201 steps -> 41 probes)


# ---------------------------------------------------------------- graph helpers
def hop_distance_from_wall(n_nodes: int, edge_index: np.ndarray, wall: np.ndarray, max_hop: int = 6) -> np.ndarray:
    """BFS hop distance from the wall set; unreached nodes get max_hop+1."""
    nbrs: list[list[int]] = [[] for _ in range(n_nodes)]
    for a, b in zip(edge_index[0], edge_index[1]):
        nbrs[a].append(b)
    hop = np.full(n_nodes, max_hop + 1, dtype=np.int32)
    dq = deque()
    for i in np.nonzero(wall)[0]:
        hop[i] = 0
        dq.append(i)
    while dq:
        i = dq.popleft()
        if hop[i] >= max_hop:
            continue
        for j in nbrs[i]:
            if hop[j] > hop[i] + 1:
                hop[j] = hop[i] + 1
                dq.append(j)
    return hop


def neighbor_mean(vals: np.ndarray, edge_index: np.ndarray, n: int, iters: int = 1) -> np.ndarray:
    """Iterated neighbour-mean smoothing (the hop-k aggregation s2.3 used)."""
    out = vals.astype(np.float64).copy()
    row, col = edge_index[0], edge_index[1]
    for _ in range(max(int(iters), 0)):
        s = np.zeros(n, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        np.add.at(s, row, out[col])
        np.add.at(c, row, 1.0)
        out = s / np.maximum(c, 1.0)
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(=). 0.5 = no information; distance from 0.5 is the signal."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # rank-based, O(n log n)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(cnt.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- per-anchor extraction
def analyse_anchor(anc: str, phys: PhysicsConfig, dev: torch.device) -> dict | None:
    d = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=dev, weights_only=False)
    n = int(d.num_nodes)
    n_times = int(d.y.shape[0])
    ei = d.edge_index.cpu().numpy()
    wall = d.mask_wall.bool().cpu().numpy().reshape(-1) if getattr(d, "mask_wall", None) is not None else None
    if wall is None or wall.sum() == 0:
        return None
    hop = hop_distance_from_wall(n, ei, wall)

    # ---- GT clot: final label + per-node onset (coarse time grid) ----
    t_last = n_times - 1
    gt_final = (gt_clot_phi_at_time(d, t_last, phys, device=dev).cpu().numpy().reshape(-1) > 0.5)
    probes = list(range(0, n_times, ONSET_STRIDE))
    if probes[-1] != t_last:
        probes.append(t_last)
    onset = np.full(n, np.nan)
    for t in probes:
        lab = (gt_clot_phi_at_time(d, t, phys, device=dev).cpu().numpy().reshape(-1) > 0.5)
        fresh = lab & ~np.isfinite(onset)
        onset[fresh] = t
    onset_frac = onset / max(t_last, 1)  # normalise: vessels have different T

    # ---- t=0 deployable fields (geometry + flow only; species are uniform at t=0) ----
    y0 = d.y[0].cpu().numpy()
    x = d.x.cpu().numpy()
    u, v = y0[:, 0], y0[:, 1]
    speed = np.hypot(u, v)
    mu0 = y0[:, 3]                      # Carreau(shear) at t=0 -> INVERSE shear-rate proxy
    feats = {
        "mu0": mu0,                                        # the physics gate, if the theory holds
        "log_mu0": np.log(np.maximum(mu0, 1e-9)),
        "speed": speed,
        "neg_speed": -speed,                               # sign-aligned: high = stagnant
        "speed_h1": -neighbor_mean(speed, ei, n, 1),
        "speed_h2": -neighbor_mean(speed, ei, n, 2),
        "speed_h3": -neighbor_mean(speed, ei, n, 3),
        "mu0_h2": neighbor_mean(mu0, ei, n, 2),
        "sdf": x[:, 2],
        "shear_potential": -x[:, 3],
        "width": x[:, 15],
        "width_d1": x[:, 16],
        "abs_width_d1": -np.abs(x[:, 16]),
        "width_d2": x[:, 17],
        "pressure": y0[:, 2],
        "recirc": -(u / np.maximum(speed, 1e-9)),          # u<0 (backflow) -> high
        "vmag_frac": np.abs(v) / np.maximum(speed, 1e-9),  # cross-flow fraction
    }

    wallm = wall & (hop == 0)
    res = {
        "anchor": anc, "n_nodes": n, "n_times": n_times, "n_wall": int(wallm.sum()),
        "n_clot_wall": int((gt_final & wallm).sum()),
        "n_clot_all": int(gt_final.sum()),
        "deep_mass": int((gt_final & (hop >= 2)).sum()),
        "burden": float((gt_final & wallm).sum() / max(wallm.sum(), 1)),
    }

    # ---- Q1: which t=0 field predicts WHERE (wall nodes only) ----
    pos = gt_final & wallm
    neg = (~gt_final) & wallm
    res["where_auc"] = {}
    if pos.sum() >= 3 and neg.sum() >= 3:
        for k, f in feats.items():
            res["where_auc"][k] = auc(f[pos], f[neg])

    # ---- Q2: WHEN -- vessel onset + what predicts per-node onset ----
    on_w = onset_frac[pos]
    res["onset_frac_p10"] = float(np.nanpercentile(on_w, 10)) if on_w.size else float("nan")
    res["onset_frac_med"] = float(np.nanmedian(on_w)) if on_w.size else float("nan")
    res["onset_spread"] = (
        float(np.nanpercentile(on_w, 90) - np.nanpercentile(on_w, 10)) if on_w.size > 5 else float("nan")
    )
    res["onset_rho"] = {}
    if on_w.size >= 10:
        for k, f in feats.items():
            res["onset_rho"][k] = spearman(f[pos], on_w)

    # ---- vessel-level t=0 aggregates (candidate regime predictors) ----
    wf = {k: f[wallm] for k, f in feats.items()}
    agg: dict[str, float] = {}
    for k, f in wf.items():
        if f.size == 0:
            continue
        agg[f"{k}_p10"] = float(np.percentile(f, 10))
        agg[f"{k}_med"] = float(np.median(f))
        agg[f"{k}_p90"] = float(np.percentile(f, 90))
    # stagnation VOLUME: how much of the near-wall band is slow / viscous
    band = hop <= 3
    sp_b = speed[band]
    mu_b = mu0[band]
    if sp_b.size:
        for q in (10, 25, 50):
            agg[f"band_speed_q{q}"] = float(np.percentile(sp_b, q))
        thr = np.percentile(speed[wallm], 25) if wallm.sum() else 0.0
        agg["stag_frac_band"] = float((sp_b <= thr).mean())
        agg["stag_depth"] = float(np.mean(hop[band][sp_b <= thr])) if (sp_b <= thr).any() else 0.0
        agg["mu_band_p90"] = float(np.percentile(mu_b, 90))
        agg["mu_band_med"] = float(np.median(mu_b))
    agg["wall_frac"] = float(wallm.sum() / max(n, 1))
    res["agg"] = agg

    # ---- Q-physics: is Mat growth autocatalytic (accelerating), as COMSOL says? ----
    mat = d.y[:, :, 15].cpu().numpy()          # Mat_log1p_nd
    if pos.sum() > 0:
        m = mat[:, pos].mean(axis=1)
        half = len(m) // 2
        early = float(m[1:half].mean() - m[0]) if half > 2 else float("nan")
        late = float(m[half:].mean() - m[half]) if half > 2 else float("nan")
        res["mat_early_gain"] = early
        res["mat_late_gain"] = late
        res["mat_accel_ratio"] = float(late / early) if (np.isfinite(early) and abs(early) > 1e-12) else float("nan")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="Comprehensive physical EDA of the COMSOL clot data")
    ap.add_argument("--anchors", default="", help="Comma list; default = all non-mirror anchors on disk")
    ap.add_argument("--min-clot", type=int, default=20, help="Skip anchors with fewer wall-clot nodes")
    ap.add_argument("--out", default="outputs/biochem/eda/clot_physics/eda.json")
    args = ap.parse_args()

    dev = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    if args.anchors.strip():
        anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    else:
        anchors = sorted(
            p.stem for p in ANCHOR_DIR.glob("*.pt") if "mirror" not in p.stem
        )
    print(f"[i] {len(anchors)} anchors\n")

    rows: list[dict] = []
    for i, anc in enumerate(anchors, 1):
        try:
            r = analyse_anchor(anc, phys, dev)
        except Exception as exc:
            print(f"  [{i:2d}/{len(anchors)}] {anc:>14} SKIP ({type(exc).__name__}: {exc})")
            continue
        if r is None:
            print(f"  [{i:2d}/{len(anchors)}] {anc:>14} SKIP (no wall mask)")
            continue
        tag = "" if r["n_clot_wall"] >= args.min_clot else "  (low clot, excluded from stats)"
        print(f"  [{i:2d}/{len(anchors)}] {anc:>14} wall={r['n_wall']:4d} clot={r['n_clot_wall']:4d} "
              f"deep={r['deep_mass']:4d} burden={r['burden']:.3f} onset_med={r['onset_frac_med']:.2f}{tag}")
        rows.append(r)

    rich = [r for r in rows if r["n_clot_wall"] >= args.min_clot]
    print(f"\n{'='*86}\n=== Q1. WHERE does clot form -- which t=0 field IS the gate? ===\n{'='*86}")
    print(f"Node-level AUC(clot vs non-clot) on wall nodes, over {len(rich)} clot-rich vessels.")
    print("AUC 1.0 = perfect (sign-aligned so higher feature = more clot); 0.5 = no information.\n")
    keys = sorted({k for r in rich for k in r.get("where_auc", {})})
    stats = []
    for k in keys:
        vals = np.array([r["where_auc"][k] for r in rich if k in r.get("where_auc", {})], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        stats.append((k, vals.mean(), np.median(vals), vals.std(), (vals > 0.5).mean(), vals.min(), vals.max()))
    stats.sort(key=lambda s: -abs(s[1] - 0.5))
    print(f"  {'feature':>16} {'mean':>7} {'median':>7} {'sd':>6} {'frac>0.5':>9} {'min':>7} {'max':>7}")
    for k, mu_, md, sd, fr, lo, hi in stats:
        print(f"  {k:>16} {mu_:7.3f} {md:7.3f} {sd:6.3f} {fr:9.2f} {lo:7.3f} {hi:7.3f}")

    print(f"\n{'='*86}\n=== Q2. WHEN does it ignite ===\n{'='*86}")
    om = np.array([r["onset_frac_med"] for r in rich], float)
    os_ = np.array([r["onset_spread"] for r in rich], float)
    print(f"  median onset (frac of horizon): mean={np.nanmean(om):.3f}  range=[{np.nanmin(om):.3f},{np.nanmax(om):.3f}]")
    print(f"  within-vessel onset spread p90-p10: mean={np.nanmean(os_):.3f}"
          f"  -> {'clot ignites over a WIDE window (progressive)' if np.nanmean(os_) > 0.25 else 'clot ignites NEARLY SIMULTANEOUSLY (switch-like)'}")
    acc = np.array([r.get("mat_accel_ratio", np.nan) for r in rich], float)
    acc = acc[np.isfinite(acc)]
    if acc.size:
        print(f"  Mat late/early growth ratio: median={np.median(acc):.2f}"
              f"  -> {'ACCELERATING (autocatalytic, matches COMSOL)' if np.median(acc) > 1.2 else 'not accelerating'}")
    okeys = sorted({k for r in rich for k in r.get("onset_rho", {})})
    orows = []
    for k in okeys:
        vals = np.array([r["onset_rho"][k] for r in rich if k in r.get("onset_rho", {})], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            orows.append((k, vals.mean(), (vals < 0).mean()))
    orows.sort(key=lambda s: -abs(s[1]))
    print(f"\n  Per-node Spearman(feature, onset_time) within vessel -- negative = feature predicts EARLIER clot:")
    print(f"  {'feature':>16} {'mean rho':>9} {'frac<0':>8}")
    for k, mu_, fr in orows[:8]:
        print(f"  {k:>16} {mu_:9.3f} {fr:8.2f}")

    print(f"\n{'='*86}\n=== Q3/Q4. Vessel-level: is REGIME (deep mass) predictable from t=0? ===\n{'='*86}")
    aggkeys = sorted({k for r in rich for k in r.get("agg", {})})
    targets = {
        "deep_mass": np.array([r["deep_mass"] for r in rich], float),
        "burden": np.array([r["burden"] for r in rich], float),
        "onset_frac_med": np.array([r["onset_frac_med"] for r in rich], float),
        "n_clot_wall": np.array([r["n_clot_wall"] for r in rich], float),
    }
    print(f"  Spearman(t=0 aggregate, vessel target), n={len(rich)} vessels."
          f"  |rho|>~{1.96/np.sqrt(max(len(rich)-1,2)):.2f} is nominally significant.\n")
    for tname, tvals in targets.items():
        srows = []
        for k in aggkeys:
            fv = np.array([r["agg"].get(k, np.nan) for r in rich], float)
            rho = spearman(fv, tvals)
            if np.isfinite(rho):
                srows.append((k, rho))
        srows.sort(key=lambda s: -abs(s[1]))
        print(f"  -- target: {tname}")
        for k, rho in srows[:6]:
            print(f"       {k:>22} rho={rho:+.3f}")
        print()

    out = Path(args.out)
    if not out.is_absolute():
        out = get_project_root() / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def _safe(o):
        if isinstance(o, dict):
            return {k: _safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    out.write_text(json.dumps(_safe({"anchors": rows}), indent=2), encoding="utf-8")
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
