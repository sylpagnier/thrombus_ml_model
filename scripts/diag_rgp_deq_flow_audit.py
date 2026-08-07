#!/usr/bin/env python
"""Z1 (WALL_MODEL_PLAN.md s17) -- measure the RGP-DEQ's flow accuracy WITHOUT the leak.

**The question.** `data.x[:, UV_PRIOR|MU_PRIOR]` are bit-identical to the converged clot-free
CFD field `y[0]` (s16.1), and `ginodeq.py:438-440` feeds them straight into the DEQ. So the flow
surrogate is handed the field it exists to predict, and its genuine accuracy on unseen geometry
has never been measured. Under the s17 Z2 contract -- deploy gives us **geometry + IC/BC only** --
those columns will not exist at deploy time.

**The test.** Re-run the DEQ with the prior block replaced three ways and compare against GT:

    stored    the leaked converged CFD field           (what training has always used)
    analytic  Poiseuille magnitude + potential-flow direction, from geometry+BC only (LEGAL)
    zero      prior block zeroed                        (ablation floor)

**What is reported, and why.** Field error alone is not the thing we care about -- the biochem
model consumes flow only through discriminative structure. So alongside relative L2 we report
the clot AUC of `-|u_pred|`, i.e. whether the *predicted* field still ranks clot-prone nodes.
A surrogate can be numerically mediocre and still be perfectly useful here, or be numerically
close and have lost the recirculation topology that carries the signal (s16.3/s16.4).

Usage:
    python scripts/diag_rgp_deq_flow_audit.py --anchors patient041,patient043
    python scripts/diag_rgp_deq_flow_audit.py --all --out outputs/logs/z1_flow_audit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_gen.lib.legal_priors import PRIOR_SOURCES, apply_prior_source  # noqa: E402

ANCHOR_DIR = Path("data/processed/graphs_biochem_anchors")
MAT = 15
U_COL, V_COL = 0, 1


def auc(score: torch.Tensor, label: torch.Tensor) -> float:
    s = score.double().reshape(-1)
    l = label.reshape(-1).bool()
    if int(l.sum()) < 5 or int((~l).sum()) < 5:
        return float("nan")
    r = s.argsort().argsort().double() + 1
    n1, n0 = int(l.sum()), int((~l).sum())
    return (r[l].sum().item() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wall_band(data, hops: int = 3) -> torch.Tensor:
    n = int(data.num_nodes)
    row, col = data.edge_index
    deg = torch.zeros(n)
    deg.index_add_(0, row, torch.ones(row.shape[0]))
    deg = deg.clamp(min=1.0)
    band = data.mask_wall.reshape(-1).bool().clone()
    for _ in range(hops):
        acc = torch.zeros(n)
        acc.index_add_(0, row, band[col].float())
        band = band | (acc / deg > 0)
    return band


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", default="", help="comma list; default = the s9 cohort")
    ap.add_argument("--all", action="store_true", help="every pack with >=20 clotted nodes")
    ap.add_argument("--sources", default=",".join(PRIOR_SOURCES))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.all:
        pids = [os.path.basename(f)[:-3] for f in sorted(glob.glob(str(ANCHOR_DIR / "patient*.pt")))
                if "mirror" not in f]
    elif args.anchors.strip():
        pids = [s.strip() for s in args.anchors.split(",") if s.strip()]
    else:
        pids = ["patient039", "patient040", "patient041", "patient042", "patient043", "patient044"]

    from src.utils.kinematics_inference import (
        load_kinematics_predictor,
        predict_kinematics,
        resolve_kinematics_checkpoint,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_kinematics_checkpoint()
    print(f"[i] RGP-DEQ ckpt: {ckpt}")
    print(f"[i] device: {dev}")
    model = load_kinematics_predictor(ckpt, dev)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rows: list[dict] = []

    hdr = f"{'vessel':<12}{'source':<10}{'relL2_u':>9}{'relL2_v':>9}{'cos':>7}{'AUC(-|u_pred|)':>16}{'AUC(-|u_gt|)':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for pid in pids:
        f = ANCHOR_DIR / f"{pid}.pt"
        if not f.exists():
            continue
        base = torch.load(f, map_location="cpu", weights_only=False)
        if int((base.y[-1, :, MAT] > 1e-4).sum()) < 20:
            continue
        band = wall_band(base)
        lab = (base.y[-1, :, MAT] > 1e-4)[band]
        u_gt = base.y[0, :, U_COL]
        v_gt = base.y[0, :, V_COL]
        spd_gt = torch.sqrt(u_gt * u_gt + v_gt * v_gt)
        auc_gt = auc(-spd_gt[band], lab)

        for src in sources:
            try:
                # IMPORTANT: the DEQ caches per-graph. Clearing is required or the second
                # source silently returns the first source's solve.
                for attr in ("_cache_key", "_cache_pred", "_cache_latent"):
                    if hasattr(model, attr):
                        setattr(model, attr, None)
                d = apply_prior_source(base, src).to(dev)
                pred = predict_kinematics(model, d).detach().cpu()
                u_p, v_p = pred[:, 0], pred[:, 1]
                rel_u = (torch.norm(u_p - u_gt) / torch.norm(u_gt).clamp(min=1e-12)).item()
                rel_v = (torch.norm(v_p - v_gt) / torch.norm(v_gt).clamp(min=1e-12)).item()
                gtn = torch.stack([u_gt, v_gt], 1)
                pn = torch.stack([u_p, v_p], 1)
                m = spd_gt > spd_gt.median()
                cos = torch.nn.functional.cosine_similarity(pn[m], gtn[m], dim=1).mean().item()
                spd_p = torch.sqrt(u_p * u_p + v_p * v_p)
                a = auc(-spd_p[band], lab)
                rows.append(dict(vessel=pid, source=src, rel_l2_u=rel_u, rel_l2_v=rel_v,
                                 cos=cos, auc_pred=a, auc_gt=auc_gt))
                print(f"{pid:<12}{src:<10}{rel_u:>9.3f}{rel_v:>9.3f}{cos:>7.3f}{a:>16.3f}{auc_gt:>14.3f}")
            except Exception as e:  # keep going; one bad vessel must not kill the audit
                print(f"{pid:<12}{src:<10}  ERROR: {type(e).__name__}: {str(e)[:60]}")

    print()
    for src in sources:
        sub = [r for r in rows if r["source"] == src]
        if not sub:
            continue
        print(f"  {src:<9} n={len(sub):<3} relL2_u={st.mean(r['rel_l2_u'] for r in sub):.3f}  "
              f"cos={st.mean(r['cos'] for r in sub):+.3f}  "
              f"AUC(pred)={st.mean(r['auc_pred'] for r in sub):.3f}  "
              f"AUC(gt)={st.mean(r['auc_gt'] for r in sub):.3f}")

    print("\nREAD THIS AS:")
    print("  * 'stored' is the leaked condition -- it is the upper bound, not a deployable number.")
    print("  * 'analytic' is the ONLY deployable row under the s17 Z2 contract.")
    print("  * If AUC(pred) under 'analytic' approaches AUC(gt), the surrogate preserves the")
    print("    discriminative structure and the biochem model can be built on it.")
    print("  * If it collapses toward 0.5, flow prediction is the project and biochem is downstream.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
