"""Explore GT species temporal/spatial patterns across biochem anchors.

Answers: what changes where/when, flow vs wall coupling, and which channels
a clot model must learn beyond resting IC.

Usage::

    python scripts/diagnose_species_temporal_patterns.py
    python scripts/diagnose_species_temporal_patterns.py --anchors patient007 --out outputs/biochem/diagnostics/species_temporal_p007.json
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

from src.config import BiochemConfig, BulkSpecies, PhysicsConfig, PredChannels, WallSpecies  # noqa: E402
from src.core_physics.clot_continuous_time import macro_tau_at_index  # noqa: E402
from src.core_physics.clot_phi_simple import _wall_mask_from_data  # noqa: E402
from src.core_physics.kinematics_clot_prior import clot_prior_score_flat  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_rung4_ladder import FI_SLICE_IDX, MAT_SLICE_IDX  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

SPECIES_NAMES = [s.name for s in BulkSpecies] + [s.name for s in WallSpecies]


def _list_anchors(graph_dir: Path, only: list[str] | None) -> list[Path]:
    paths = sorted(graph_dir.glob("patient*.pt"))
    if only:
        stems = {a.strip().lower() for a in only}
        paths = [p for p in paths if p.stem.lower() in stems]
    if not paths:
        raise FileNotFoundError(f"No anchors in {graph_dir}")
    return paths


def _shear_flat(data, t: int, device: torch.device) -> torch.Tensor:
    from src.core_physics.kinematics_clot_prior import shear_rate_si
    from src.core_physics.clot_phi_simple import _anchor_flow_props

    y = data.y[int(t)].to(device=device, dtype=torch.float32)
    props = _anchor_flow_props(data, device)
    return shear_rate_si(data, y[:, PredChannels.U], y[:, PredChannels.V], props).reshape(-1).clamp(min=0.0)


def _region_masks(data, t: int, phys: PhysicsConfig, device: torch.device) -> dict[str, torch.Tensor]:
    n = int(data.num_nodes)
    wall = _wall_mask_from_data(data, device, n).reshape(-1).bool()
    phi = gt_clot_phi_at_time(data, int(t), phys, device).reshape(-1)
    clot = phi >= 0.5
    growth = clot | (wall & (phi > 0.08))
    bulk = ~(wall | clot)
    return {"wall": wall, "clot": clot, "growth": growth, "bulk": bulk}


def _region_mean(series_1d: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask.reshape(-1).bool()
    if not bool(m.any().item()):
        return float("nan")
    return float(series_1d[m].mean().item())


def _time_indices(n_steps: int, max_times: int) -> list[int]:
    if max_times <= 0 or n_steps <= max_times:
        return list(range(n_steps))
    import numpy as np

    idx = np.linspace(0, n_steps - 1, num=max_times, dtype=int)
    return sorted({int(i) for i in idx.tolist()})


@torch.no_grad()
def analyze_anchor(
    path: Path,
    *,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    max_times: int,
) -> dict:
    data = torch.load(path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    times = _time_indices(n_steps, max_times)
    n = int(data.num_nodes)
    sp0 = data.y[0, :, 4:16].to(device=device, dtype=torch.float32)

    # --- resting IC structure ---
    wall0 = _wall_mask_from_data(data, device, n).reshape(-1).bool()
    ic = {
        "wall_minus_bulk_mean": {},
        "wall_fraction_elevated_fi": float(
            (sp0[wall0, FI_SLICE_IDX] > sp0[:, FI_SLICE_IDX].median() + 0.1).float().mean().item()
        ),
    }
    for ci, name in enumerate(SPECIES_NAMES):
        ic["wall_minus_bulk_mean"][name] = _region_mean(
            sp0[:, ci], wall0
        ) - _region_mean(sp0[:, ci], ~wall0)

    # --- per-time regional means + deltas ---
    timeline: list[dict] = []
    prev_sp = sp0
    for t in times:
        y_t = data.y[int(t)].to(device=device, dtype=torch.float32)
        sp = y_t[:, 4:16]
        masks = _region_masks(data, t, phys, device)
        shear = _shear_flat(data, t, device)
        tau = float(macro_tau_at_index(data, t, bio_cfg=bio))
        row: dict = {
            "time": int(t),
            "tau": tau,
            "n_clot": int(masks["clot"].sum().item()),
            "regions": {},
            "species_delta_from_t0": {},
            "d_species_dt": {},
        }
        for rname, rmask in masks.items():
            row["regions"][rname] = {
                name: _region_mean(sp[:, ci], rmask)
                for ci, name in enumerate(SPECIES_NAMES)
            }
        dstep = max(int(times[1] - times[0]) if len(times) > 1 else 1, 1)
        d_sp = (sp - prev_sp) / float(dstep)
        for ci, name in enumerate(SPECIES_NAMES):
            row["species_delta_from_t0"][name] = _region_mean(sp[:, ci] - sp0[:, ci], masks["growth"])
            row["d_species_dt"][name] = _region_mean(d_sp[:, ci], masks["growth"])
        # flow coupling in growth band
        g = masks["growth"]
        if bool(g.any().item()):
            ds_fi = d_sp[g, FI_SLICE_IDX]
            sh = shear[g]
            if ds_fi.numel() > 4:
                ds_c = ds_fi - ds_fi.mean()
                sh_c = sh - sh.mean()
                denom = ds_c.norm() * sh_c.norm()
                row["corr_dFI_dt_shear_growth"] = float((ds_c * sh_c).sum().item() / denom) if denom > 1e-12 else 0.0
            else:
                row["corr_dFI_dt_shear_growth"] = float("nan")
        timeline.append(row)
        prev_sp = sp

    # --- first activation time in clot (50% of final FI rise in growth band) ---
    fi_growth = [r["regions"]["growth"]["FI"] for r in timeline]
    fi0 = fi_growth[0] if fi_growth else float("nan")
    fi_end = fi_growth[-1] if fi_growth else float("nan")
    half = fi0 + 0.5 * (fi_end - fi0)
    t_half_fi = None
    for r in timeline:
        if r["regions"]["growth"]["FI"] >= half:
            t_half_fi = r["time"]
            break

    # --- neighbor transport proxy at mid time ---
    t_mid = times[len(times) // 2]
    sp_mid = data.y[t_mid, :, 4:16].to(device=device, dtype=torch.float32)
    sp_prev = data.y[max(t_mid - 1, 0), :, 4:16].to(device=device, dtype=torch.float32)
    delta_node = (sp_mid - sp_prev).abs().mean(dim=1)
    ei = data.edge_index.to(device)
    src, dst = ei[0], ei[1]
    nbr_mean = torch.zeros(n, device=device, dtype=torch.float32)
    deg = torch.zeros(n, device=device, dtype=torch.float32)
    nbr_mean.scatter_add_(0, dst, delta_node[src])
    deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
    nbr_mean = nbr_mean / deg.clamp(min=1.0)
    masks_mid = _region_masks(data, t_mid, phys, device)
    g = masks_mid["growth"]
    local = delta_node[g].mean().item() if bool(g.any()) else float("nan")
    nbr = nbr_mean[g].mean().item() if bool(g.any()) else float("nan")
    transport_ratio = float(nbr / max(local, 1e-12)) if local == local else float("nan")

    # --- channel ranking: total rise in clot at final time ---
    t_last = times[-1]
    masks_last = _region_masks(data, t_last, phys, device)
    c = masks_last["clot"]
    rises = {}
    if bool(c.any().item()):
        for ci, name in enumerate(SPECIES_NAMES):
            rises[name] = float((data.y[t_last, c, 4 + ci] - sp0[c, ci]).mean().item())
    ranked = sorted(rises.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "anchor": path.stem,
        "n_steps": n_steps,
        "n_nodes": n,
        "times_sampled": times,
        "ic": ic,
        "t_half_fi_growth": t_half_fi,
        "transport_vs_local_ratio_growth": transport_ratio,
        "clot_species_rise_rank": [{"species": k, "mean_delta_log_nd": v} for k, v in ranked[:6]],
        "timeline": timeline,
    }


def _summarize_all(rows: list[dict]) -> dict:
    """Cross-vessel aggregates."""
    def _mean_key(key_path: str) -> float:
        vals = []
        for r in rows:
            cur = r
            for k in key_path.split("."):
                cur = cur[k]
            if isinstance(cur, (int, float)) and cur == cur:
                vals.append(float(cur))
        return sum(vals) / max(len(vals), 1)

    half_times = [r["t_half_fi_growth"] for r in rows if r.get("t_half_fi_growth") is not None]
    return {
        "n_anchors": len(rows),
        "mean_transport_vs_local_ratio": _mean_key("transport_vs_local_ratio_growth"),
        "median_t_half_fi_growth": sorted(half_times)[len(half_times) // 2] if half_times else None,
        "anchors": [r["anchor"] for r in rows],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GT species temporal pattern diagnostic")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--anchors", default="", help="Comma list; default all patient*.pt")
    ap.add_argument("--max-times", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/biochem/diagnostics/species_temporal_patterns.json")
    args = ap.parse_args()

    root = get_project_root()
    graph_dir = root / args.graph_dir
    only = [a.strip() for a in args.anchors.split(",") if a.strip()] or None
    paths = _list_anchors(graph_dir, only)

    device = torch.device(args.device)
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    print(f"[i] anchors={[p.stem for p in paths]} max_times={args.max_times}", flush=True)
    rows = [
        analyze_anchor(p, phys=phys, bio=bio, device=device, max_times=int(args.max_times))
        for p in paths
    ]
    payload = {"summary": _summarize_all(rows), "anchors": rows}

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {out}", flush=True)

    s = payload["summary"]
    print(
        f"[i] cross-vessel median t_50pct_FI_growth={s.get('median_t_half_fi_growth')} "
        f"mean neighbor/local transport ratio={s.get('mean_transport_vs_local_ratio'):.3f}",
        flush=True,
    )
    for r in rows:
        top = r.get("clot_species_rise_rank", [])[:3]
        top_s = ", ".join(f"{x['species']}:{x['mean_delta_log_nd']:.3f}" for x in top)
        print(
            f"[i] {r['anchor']}: t_half_FI={r.get('t_half_fi_growth')} "
            f"transport_ratio={r.get('transport_vs_local_ratio_growth', float('nan')):.2f} "
            f"top_rise=[{top_s}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
