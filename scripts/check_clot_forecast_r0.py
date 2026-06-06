#!/usr/bin/env python3
"""R0: label sanity for clot forecast ladder (mu(t) -> mu(t+dt) on COMSOL anchors)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PhysicsConfig, VesselConfig, STATE_CHANNEL_MU_EFF_ND
from src.core_physics.clot_phi_simple import (
    cap_mu_eff_si,
    clot_phi_thresh_si,
    supervision_region_mask,
)
from src.utils.channel_schema import infer_missing_schema
from src.utils.paths import get_project_root


def _parse_anchors(raw: str, anchor_dir: Path) -> list[Path]:
    if raw.strip():
        stems = [s.strip() for s in raw.split(",") if s.strip()]
        return [anchor_dir / f"{s}.pt" if not s.endswith(".pt") else anchor_dir / s for s in stems]
    return sorted(anchor_dir.glob("*.pt"))


def _time_si(data, ti: int) -> float | None:
    if hasattr(data, "t_si") and torch.is_tensor(data.t_si) and data.t_si.numel() > ti:
        return float(data.t_si[ti].item())
    if hasattr(data, "t") and torch.is_tensor(data.t) and data.t.numel() > ti:
        return float(data.t[ti].item())
    return None


def analyze_anchor(path: Path, phys: PhysicsConfig, *, pair_stride: int, mu_thr: float) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    if not hasattr(data, "y") or data.y.dim() != 3:
        raise ValueError(f"{path.name}: expected data.y [T,N,C]")

    n_steps = int(data.y.shape[0])
    device = torch.device("cpu")
    rows: list[dict] = []
    clot_frac_t0 = None
    clot_frac_tfinal = None
    max_new_clot_nodes = 0
    max_dlog_mu = 0.0
    growth_events = 0

    for ti in range(0, max(n_steps - pair_stride, 0)):
        t_out = ti + pair_stride
        y_in = data.y[ti]
        y_out = data.y[t_out]
        mu_in = cap_mu_eff_si(phys.viscosity_nd_to_si(y_in[:, STATE_CHANNEL_MU_EFF_ND]))
        mu_out = cap_mu_eff_si(phys.viscosity_nd_to_si(y_out[:, STATE_CHANNEL_MU_EFF_ND]))
        region = supervision_region_mask(data, device, mu_out, phys)
        if not bool(region.any().item()):
            continue

        clot_in = (mu_in.reshape(-1) >= mu_thr) & region.reshape(-1)
        clot_out = (mu_out.reshape(-1) >= mu_thr) & region.reshape(-1)
        new_nodes = int((clot_out & ~clot_in).sum().item())
        dlog = (
            torch.log(mu_out[region].clamp(min=1e-8)) - torch.log(mu_in[region].clamp(min=1e-8))
        ).abs()
        mean_dlog = float(dlog.mean().item()) if dlog.numel() else 0.0
        max_pair_dlog = float(dlog.max().item()) if dlog.numel() else 0.0
        max_dlog_mu = max(max_dlog_mu, max_pair_dlog)
        max_new_clot_nodes = max(max_new_clot_nodes, new_nodes)
        if new_nodes > 0:
            growth_events += 1

        frac_out = float(clot_out.sum().item()) / max(float(region.sum().item()), 1.0)
        if ti == 0:
            clot_frac_t0 = float(clot_in.sum().item()) / max(float(region.sum().item()), 1.0)
        if t_out == n_steps - 1:
            clot_frac_tfinal = frac_out

        rows.append(
            {
                "t_in": ti,
                "t_out": t_out,
                "t_in_si": _time_si(data, ti),
                "t_out_si": _time_si(data, t_out),
                "region_n": int(region.sum().item()),
                "clot_frac_in": float(clot_in.sum().item()) / max(float(region.sum().item()), 1.0),
                "clot_frac_out": frac_out,
                "new_clot_nodes": new_nodes,
                "mean_abs_dlog_mu": mean_dlog,
                "max_abs_dlog_mu": max_pair_dlog,
            }
        )

    pass_growth = (clot_frac_tfinal or 0.0) > (clot_frac_t0 or 0.0) + 0.01 or growth_events >= 2
    pass_signal = max_dlog_mu >= 0.02
    status = "PASS" if (pass_growth and pass_signal and rows) else "FAIL"

    return {
        "anchor": path.stem,
        "n_steps": n_steps,
        "pair_stride": pair_stride,
        "mu_thr_si": mu_thr,
        "n_pairs": len(rows),
        "clot_frac_t0": clot_frac_t0,
        "clot_frac_tfinal": clot_frac_tfinal,
        "growth_events": growth_events,
        "max_new_clot_nodes": max_new_clot_nodes,
        "max_abs_dlog_mu": max_dlog_mu,
        "pass_growth": pass_growth,
        "pass_signal": pass_signal,
        "status": status,
        "pairs": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R0 clot forecast label sanity")
    ap.add_argument("--anchor-dir", default="", help="Override biochem anchor graph dir")
    ap.add_argument("--anchors", default="patient003,patient007,patient006")
    ap.add_argument("--pair-stride", type=int, default=1)
    ap.add_argument("--mu-thr-si", type=float, default=0.055)
    ap.add_argument(
        "--out",
        default="outputs/biochem/clot_forecast_ladder/r0_label_sanity.json",
    )
    args = ap.parse_args()
    root = get_project_root()
    if args.anchor_dir.strip():
        anchor_dir = Path(args.anchor_dir)
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir

    phys = PhysicsConfig(phase="biochem")
    mu_thr = float(args.mu_thr_si or clot_phi_thresh_si(phys))
    paths = _parse_anchors(args.anchors, anchor_dir.resolve())
    if not paths:
        print(f"[ERR] no anchors in {anchor_dir}", file=sys.stderr)
        return 1

    results: list[dict] = []
    for path in paths:
        if not path.is_file():
            print(f"[WARN] missing {path}", flush=True)
            continue
        row = analyze_anchor(path, phys, pair_stride=max(1, args.pair_stride), mu_thr=mu_thr)
        results.append(row)
        print(
            f"[i]  {row['anchor']}: pairs={row['n_pairs']} "
            f"clot t0={row['clot_frac_t0']:.3f} tfinal={row['clot_frac_tfinal']:.3f} "
            f"growth_events={row['growth_events']} max_dlog={row['max_abs_dlog_mu']:.4f} "
            f"-> {row['status']}",
            flush=True,
        )

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "rung": "R0",
        "anchor_dir": str(anchor_dir),
        "pair_stride": max(1, args.pair_stride),
        "mu_thr_si": mu_thr,
        "n_anchors": len(results),
        "n_pass": n_pass,
        "overall": "PASS" if results and n_pass == len(results) else "FAIL",
        "anchors": results,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[{'OK' if summary['overall'] == 'PASS' else 'ERR'}]  R0 {summary['overall']} "
        f"({n_pass}/{len(results)} anchors) -> {out_path}",
        flush=True,
    )
    return 0 if summary["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
