"""GT clot hop-distance census across biochem anchors (no model required).

Counts binary GT clot nodes by BFS hop from ``mask_wall`` at deploy eval time
(and optionally at early/mid times) to characterize where real clots live
relative to the wall.

Usage:
  python scripts/survey_gt_clot_hop_census.py
  python scripts/survey_gt_clot_hop_census.py --anchors patient001,patient007
  python scripts/survey_gt_clot_hop_census.py --mat-leg WC_v7_clot_phi_mse --max-hop 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env  # noqa: E402
from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    compute_hop_distances,
    deploy_eval_time_index,
    discover_biochem_anchors,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _hop_hist(clot: torch.Tensor, hops: torch.Tensor, max_hop: int) -> dict[str, int]:
    """Count clot-positive nodes per hop bucket; ``hop_ge`` = hops > max_hop."""
    c = clot.reshape(-1).bool()
    h = hops.reshape(-1)
    out: dict[str, int] = {}
    for k in range(0, max_hop + 1):
        out[f"hop{k}"] = int((c & (h == k)).sum().item())
    out[f"hop_gt{max_hop}"] = int((c & (h > max_hop)).sum().item())
    out["total"] = int(c.sum().item())
    out["offwall"] = int((c & (h >= 1)).sum().item())
    return out


def _mesh_hop_hist(hops: torch.Tensor, max_hop: int) -> dict[str, int]:
    """Node inventory by hop (all mesh nodes, not just clot)."""
    h = hops.reshape(-1)
    out: dict[str, int] = {}
    for k in range(0, max_hop + 1):
        out[f"hop{k}"] = int((h == k).sum().item())
    out[f"hop_gt{max_hop}"] = int((h > max_hop).sum().item())
    out["n_nodes"] = int(h.numel())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="GT clot hop-distance census")
    ap.add_argument("--anchors", default="", help="Comma list (default: all on disk)")
    ap.add_argument("--mat-leg", default="WC_v7_clot_phi_mse", help="Leg env for clot thresh consistency")
    ap.add_argument("--max-hop", type=int, default=6)
    ap.add_argument(
        "--out",
        default="outputs/biochem/offwall_model/gt_clot_hop_census.json",
    )
    ap.add_argument(
        "--times",
        default="deploy",
        help="Comma times: deploy | early | mid | late | integer indices",
    )
    args = ap.parse_args()

    root = get_project_root()
    if args.mat_leg.strip():
        apply_mat_growth_leg_env(args.mat_leg.strip(), force=True)
        print(f"[i] mat-leg env: {args.mat_leg.strip()}", flush=True)

    phys = PhysicsConfig(phase="biochem")
    device = torch.device("cpu")
    max_hop = max(int(args.max_hop), 0)

    if args.anchors.strip():
        anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    else:
        anchors = discover_biochem_anchors(root)

    time_tokens = [t.strip().lower() for t in args.times.split(",") if t.strip()]
    if not time_tokens:
        time_tokens = ["deploy"]

    per_anchor: dict[str, dict] = {}
    print(f"[i] anchors={len(anchors)} max_hop={max_hop} times={time_tokens}", flush=True)
    print(
        f"{'anchor':<12} {'t':>5} {'tot':>5} {'ow':>5} "
        + " ".join(f"{'h'+str(k):>5}" for k in range(0, max_hop + 1))
        + f" {'h>'+str(max_hop):>5}",
        flush=True,
    )

    for anc in anchors:
        path = root / "data/processed/graphs_biochem_anchors" / f"{anc}.pt"
        if not path.is_file():
            print(f"[WARN] missing {path}", flush=True)
            continue
        data = torch.load(path, map_location=device, weights_only=False)
        n = int(data.num_nodes)
        n_times = int(data.y.shape[0])
        wall = _wall_mask_from_data(data, device, n)
        hops = compute_hop_distances(data.edge_index, wall, n)
        mesh_hist = _mesh_hop_hist(hops, max_hop)

        # Resolve requested times for this graph length.
        resolved: list[tuple[str, int]] = []
        for tok in time_tokens:
            if tok in ("deploy", "last", "eval"):
                resolved.append((tok, deploy_eval_time_index(n_times)))
            elif tok == "early":
                resolved.append((tok, max(0, n_times // 4)))
            elif tok == "mid":
                resolved.append((tok, max(0, n_times // 2)))
            elif tok == "late":
                resolved.append((tok, max(0, (3 * n_times) // 4)))
            else:
                ti = int(tok)
                resolved.append((tok, max(0, min(ti, n_times - 1))))

        anc_row: dict = {
            "n_nodes": n,
            "n_times": n_times,
            "n_wall": int(wall.sum().item()),
            "mesh_hop_hist": mesh_hist,
            "times": {},
        }
        for label, ti in resolved:
            phi = gt_clot_phi_at_time(data, ti, phys, device=device)
            hist = _hop_hist(phi, hops, max_hop)
            anc_row["times"][label] = {"t_index": ti, "clot_hop_hist": hist}
            print(
                f"{anc:<12} {ti:5d} {hist['total']:5d} {hist['offwall']:5d} "
                + " ".join(f"{hist[f'hop{k}']:5d}" for k in range(0, max_hop + 1))
                + f" {hist[f'hop_gt{max_hop}']:5d}",
                flush=True,
            )
        per_anchor[anc] = anc_row

    # Cohort means at deploy (or first requested time label).
    primary = time_tokens[0]
    hop_keys = [f"hop{k}" for k in range(0, max_hop + 1)] + [f"hop_gt{max_hop}", "total", "offwall"]
    sums = {k: 0.0 for k in hop_keys}
    n_ok = 0
    for anc, row in per_anchor.items():
        tblock = row["times"].get(primary) or next(iter(row["times"].values()), None)
        if tblock is None:
            continue
        hist = tblock["clot_hop_hist"]
        for k in hop_keys:
            sums[k] += float(hist.get(k, 0))
        n_ok += 1
    mean = {k: (sums[k] / max(n_ok, 1)) for k in hop_keys}
    frac_of_clot = {
        k: (mean[k] / mean["total"]) if mean["total"] > 0 else 0.0
        for k in hop_keys
        if k not in ("total", "offwall")
    }
    frac_offwall = (mean["offwall"] / mean["total"]) if mean["total"] > 0 else 0.0

    print("\n=== COHORT MEAN (clot nodes / anchor) ===", flush=True)
    print(
        f"  total={mean['total']:.1f}  offwall={mean['offwall']:.1f} "
        f"({100.0 * frac_offwall:.1f}% of clot)  n_anchors={n_ok}",
        flush=True,
    )
    for k in range(0, max_hop + 1):
        key = f"hop{k}"
        print(
            f"  {key}: mean={mean[key]:7.2f}  "
            f"frac_of_clot={100.0 * frac_of_clot.get(key, 0.0):5.1f}%",
            flush=True,
        )
    ge = f"hop_gt{max_hop}"
    print(
        f"  {ge}: mean={mean[ge]:7.2f}  "
        f"frac_of_clot={100.0 * frac_of_clot.get(ge, 0.0):5.1f}%",
        flush=True,
    )

    # How many anchors have any clot at each hop?
    presence = {f"hop{k}": 0 for k in range(0, max_hop + 1)}
    presence[f"hop_gt{max_hop}"] = 0
    for anc, row in per_anchor.items():
        tblock = row["times"].get(primary) or next(iter(row["times"].values()), None)
        if tblock is None:
            continue
        hist = tblock["clot_hop_hist"]
        for k in range(0, max_hop + 1):
            if hist.get(f"hop{k}", 0) > 0:
                presence[f"hop{k}"] += 1
        if hist.get(f"hop_gt{max_hop}", 0) > 0:
            presence[f"hop_gt{max_hop}"] += 1

    print("\n=== ANCHORS WITH ANY GT CLOT AT HOP ===", flush=True)
    for k in range(0, max_hop + 1):
        print(f"  hop{k}: {presence[f'hop{k}']}/{n_ok}", flush=True)
    print(f"  hop>{max_hop}: {presence[f'hop_gt{max_hop}']}/{n_ok}", flush=True)

    report = {
        "mat_leg": args.mat_leg.strip(),
        "max_hop": max_hop,
        "time_tokens": time_tokens,
        "n_anchors": n_ok,
        "cohort_mean_clot_hop_hist": mean,
        "cohort_frac_of_clot": frac_of_clot,
        "cohort_offwall_frac_of_clot": frac_offwall,
        "anchors_with_any_clot_at_hop": presence,
        "per_anchor": per_anchor,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
