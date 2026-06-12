"""Lower-wall clot vs non-clot diagnostic (geometry + flow separation)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_growth_masks import gt_clot_mask_at_time, resolve_ceiling_mask
from src.core_physics.clot_t0_extended_probe import build_feature_table_at_time
from src.core_physics.clot_t0_pattern_probe import (
    _binary_auc,
    _wall_mask,
    build_t0_feature_table,
    discover_anchor_paths,
)
from src.utils.paths import get_project_root


def _lower_upper_wall(data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(data.num_nodes)
    wall = _wall_mask(data, device, n)
    pos = data.x[:, :2].to(device)
    ym = pos[wall, 1].median()
    lower = wall & (pos[:, 1] <= ym)
    upper = wall & (pos[:, 1] > ym)
    return lower, upper


def _feature_rows(
    feats: dict[str, torch.Tensor],
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    for key, val in feats.items():
        if int(val.numel()) != int(pos_mask.numel()):
            continue
        v = val.reshape(-1).float()
        if int(pos_mask.sum()) < 3 or int(neg_mask.sum()) < 3:
            continue
        cm = float(v[pos_mask].mean())
        nm = float(v[neg_mask].mean())
        auc = max(_binary_auc(v, pos_mask.float()), _binary_auc(-v, pos_mask.float()))
        rows.append({"feature": key, "auc": auc, "delta": cm - nm, "clot_mean": cm, "non_mean": nm})
    rows.sort(key=lambda r: -r["auc"])
    return rows


def _rule_metrics(rule: torch.Tensor, gt: torch.Tensor, pool: torch.Tensor) -> dict[str, float]:
    tp = int((rule & gt & pool).sum())
    fp = int((rule & ~gt & pool).sum())
    fn = int((~rule & gt & pool).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "pred_frac": float(rule[pool].sum()) / max(int(pool.sum()), 1),
        "n_pred": int(rule.sum()),
    }


def probe_anchor_lower_wall(
    data,
    *,
    stem: str,
    t_out: int,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
) -> dict:
    n = int(data.num_nodes)
    lower, upper = _lower_upper_wall(data, device)
    ceiling = resolve_ceiling_mask(data, device, bio)
    gt = gt_clot_mask_at_time(data, t_out, phys, device)
    pool = lower & ceiling
    pos_m = pool & gt
    neg_m = pool & ~gt

    feats_t0 = build_t0_feature_table(data, device=device, phys_cfg=phys, bio_cfg=bio)
    feats_t = {
        k: v[0] for k, v in build_feature_table_at_time(data, t_out, device=device, phys_cfg=phys, bio_cfg=bio).items()
    }
    pos = data.x[:, :2].to(device)
    feats_t0["pos_x_nd"] = pos[:, 0]
    feats_t0["pos_y_nd"] = pos[:, 1]
    feats_t["pos_x_nd"] = pos[:, 0]
    feats_t["pos_y_nd"] = pos[:, 1]

    rows_t0 = _feature_rows(feats_t0, pos_m, neg_m)
    rows_t = _feature_rows(feats_t, pos_m, neg_m)

    rules: list[dict] = []
    if rows_t0:
        for feat in [r["feature"] for r in rows_t0[:6]]:
            v = feats_t0[feat].reshape(-1).float()
            pool_idx = pool.nonzero(as_tuple=False).reshape(-1)
            pv = v[pool_idx]
            idx = next(i for i, r in enumerate(rows_t0) if r["feature"] == feat)
            higher_risk = rows_t0[idx]["delta"] >= 0
            for q in (0.55, 0.65, 0.75):
                thr = torch.quantile(pv, q)
                flag = torch.zeros(n, dtype=torch.bool, device=device)
                pick = pv >= thr if higher_risk else pv <= thr
                flag[pool_idx[pick]] = True
                m = _rule_metrics(flag, gt, pool)
                rules.append({"rule": f"lower|{feat}|q{int(q * 100)}", **m})

    if "flux_stag" in feats_t0 and "neg_dgamma_dx" in feats_t0:
        stag = feats_t0["flux_stag"].reshape(-1).float()
        ndx = feats_t0["neg_dgamma_dx"].reshape(-1).float()
        pool_idx = pool.nonzero(as_tuple=False).reshape(-1)
        for sq in (0.60, 0.70):
            for nq in (0.60, 0.70):
                s_thr = torch.quantile(stag[pool_idx], sq)
                n_thr = torch.quantile(ndx[pool_idx], nq)
                flag = pool & (stag >= s_thr) & (ndx >= n_thr)
                m = _rule_metrics(flag, gt, pool)
                rules.append({"rule": f"lower|stag_q{int(sq * 100)}&negdx_q{int(nq * 100)}", **m})

    rules.sort(key=lambda r: (-r["f1"], -r["prec"]))
    return {
        "anchor": stem,
        "t_out": t_out,
        "n_lower_wall": int(lower.sum()),
        "n_lower_ceiling": int(pool.sum()),
        "n_lower_clot": int(pos_m.sum()),
        "n_lower_non_clot": int(neg_m.sum()),
        "n_upper_clot": int((upper & ceiling & gt).sum()),
        "top_t0": rows_t0[:12],
        "top_t": rows_t[:12],
        "best_rules": rules[:8],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Lower wall clot vs non-clot feature probe")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--t-out", type=int, default=37)
    ap.add_argument("--all-anchors", action="store_true")
    ap.add_argument(
        "--out-json",
        default="outputs/biochem/diagnostics/clot_lower_wall_clot_probe.json",
    )
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    anchor_dir = root / args.anchor_dir

    stems = [p.stem for p in discover_anchor_paths(anchor_dir)]
    if not args.all_anchors:
        stems = [args.anchor.strip()] if args.anchor.strip() else stems[:1]

    reports = []
    for stem in stems:
        path = anchor_dir / f"{stem}.pt"
        if not path.is_file():
            print(f"[WARN] skip missing {path}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        t_out = min(int(args.t_out), int(data.y.shape[0]) - 1)
        rep = probe_anchor_lower_wall(data, stem=stem, t_out=t_out, device=device, phys=phys, bio=bio)
        reports.append(rep)
        print(f"\n=== {stem} t_out={args.t_out} ===")
        print(
            f"lower wall: clot={rep['n_lower_clot']} non-clot={rep['n_lower_non_clot']} "
            f"(ceiling pool={rep['n_lower_ceiling']})"
        )
        print("\nTop t=0 features (deployable):")
        print(f"{'feature':<22} {'AUC':>6} {'delta':>10}")
        for row in rep["top_t0"][:10]:
            print(f"{row['feature']:<22} {row['auc']:>6.3f} {row['delta']:>10.3g}")
        print("\nBest lower-wall rules @ t=0 features:")
        print(f"{'rule':<40} {'F1':>6} {'prec':>6} {'rec':>6} {'pred+':>6}")
        for row in rep["best_rules"][:6]:
            print(
                f"{row['rule']:<40} {row['f1']:>6.3f} {row['prec']:>6.3f} "
                f"{row['rec']:>6.3f} {row['pred_frac']:>6.3f}"
            )

    out = root / args.out_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"t_out": args.t_out, "anchors": reports}, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
