"""Compare biochem anchor graph time horizons against raw COMSOL export.

Reads raw ``cfd_results_biochem/*.txt`` headers and
current ``graphs_biochem_anchors`` to report step counts, physical time span,
and clot-FI/Mat timing.

Usage::

    python scripts/diagnose_biochem_extract_horizon.py
    python scripts/diagnose_biochem_extract_horizon.py --out outputs/biochem/diagnostics/extract_horizon_fulltime.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_rung4_ladder import FI_SLICE_IDX, MAT_SLICE_IDX  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ANCHORS_6 = (
    "patient001",
    "patient002",
    "patient003",
    "patient004",
    "patient006",
    "patient007",
)


def _parse_comsol_times(txt_path: Path) -> list[float]:
    with txt_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("% x") and "@ t=" in line:
                times: list[float] = []
                for m in re.finditer(r"t=([0-9.]+)", line):
                    t = float(m.group(1))
                    if t not in times:
                        times.append(t)
                return times
    return []


def _graph_summary(path: Path, *, phys: PhysicsConfig, device: torch.device) -> dict:
    if not path.is_file():
        return {"present": False}
    data = torch.load(path, map_location=device, weights_only=False)
    n_steps = int(data.y.shape[0])
    if hasattr(data, "t") and data.t is not None:
        t_phys = data.t.reshape(-1).detach().cpu().numpy().astype(np.float64)
    else:
        t_phys = np.arange(n_steps, dtype=np.float64)
    clot_frac: list[float] = []
    fi_growth: list[float] = []
    mat_growth: list[float] = []
    phi_last = gt_clot_phi_at_time(data, n_steps - 1, phys, device).reshape(-1)
    for ti in range(n_steps):
        phi = gt_clot_phi_at_time(data, ti, phys, device).reshape(-1)
        clot = phi >= 0.5
        clot_frac.append(float(clot.float().mean().item()))
        sp = data.y[ti, :, 4:16].to(device=device, dtype=torch.float32)
        g = phi >= 0.08
        if bool(g.any().item()):
            fi_growth.append(float(sp[g, FI_SLICE_IDX].mean().item()))
            mat_growth.append(float(sp[g, MAT_SLICE_IDX].mean().item()))
        else:
            fi_growth.append(float("nan"))
            mat_growth.append(float("nan"))
    clot_arr = np.asarray(clot_frac, dtype=np.float64)
    peak_i = int(np.nanargmax(clot_arr)) if clot_arr.size else 0
    legacy_i = min(53, n_steps - 1)
    # Jaccard vs final GT clot mask (how complete is clot at each time).
    jaccard_vs_final: list[float] = []
    for ti in range(n_steps):
        phi = gt_clot_phi_at_time(data, ti, phys, device).reshape(-1)
        c = phi >= 0.5
        f = phi_last >= 0.5
        inter = (c & f).sum().float()
        union = (c | f).sum().float().clamp(min=1.0)
        jaccard_vs_final.append(float((inter / union).item()))
    jacc_arr = np.asarray(jaccard_vs_final, dtype=np.float64)
    sample_idx = sorted(
        {
            0,
            legacy_i,
            n_steps // 4,
            n_steps // 2,
            3 * n_steps // 4,
            n_steps - 1,
            peak_i,
        }
    )
    return {
        "present": True,
        "n_steps": n_steps,
        "t_first_s": float(t_phys[0]),
        "t_last_s": float(t_phys[-1]),
        "t_span_s": float(t_phys[-1] - t_phys[0]),
        "legacy_index_53": legacy_i,
        "gt_clot_frac_at_legacy_53": float(clot_arr[legacy_i]) if legacy_i < len(clot_arr) else None,
        "gt_clot_frac_at_last": float(clot_arr[-1]) if clot_arr.size else None,
        "gt_clot_frac_peak": float(clot_arr[peak_i]) if clot_arr.size else None,
        "gt_clot_jaccard_vs_final_at_legacy_53": float(jacc_arr[legacy_i]) if legacy_i < len(jacc_arr) else None,
        "gt_clot_jaccard_vs_final_at_last": float(jacc_arr[-1]) if jacc_arr.size else None,
        "gt_clot_frac_peak_index": peak_i,
        "gt_clot_frac_peak_time_s": float(t_phys[peak_i]) if clot_arr.size else None,
        "sample_times": [
            {
                "index": int(ti),
                "t_s": float(t_phys[ti]),
                "gt_clot_frac": float(clot_arr[ti]),
                "gt_clot_jaccard_vs_final": float(jacc_arr[ti]),
                "fi_growth_mean": float(fi_growth[ti]) if fi_growth[ti] == fi_growth[ti] else None,
                "mat_growth_mean": float(mat_growth[ti]) if mat_growth[ti] == mat_growth[ti] else None,
            }
            for ti in sample_idx
        ],
    }


def _raw_export_summary(txt_dir: Path, stem: str) -> dict:
    txt = txt_dir / f"{stem}.txt"
    if not txt.is_file():
        return {"present": False}
    times = _parse_comsol_times(txt)
    if not times:
        return {"present": True, "error": "no_time_header"}
    arr = np.asarray(times, dtype=np.float64)
    return {
        "present": True,
        "n_raw_steps": int(arr.size),
        "t_last_raw_s": float(arr[-1]),
        "n_full_uncapped": int(arr.size),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Biochem extract horizon diagnostic")
    ap.add_argument("--raw-dir", default="data/processed/cfd_results_biochem")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--anchors", default=",".join(ANCHORS_6))
    ap.add_argument("--out", default="outputs/biochem/diagnostics/extract_horizon_fulltime.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    root = get_project_root()
    raw_dir = root / args.raw_dir
    graph_dir = root / args.graph_dir
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    device = torch.device(args.device)
    phys = PhysicsConfig(phase="biochem")
    _ = BiochemConfig(phase="biochem")

    rows: list[dict] = []
    for stem in anchors:
        row = {
            "anchor": stem,
            "raw_export": _raw_export_summary(raw_dir, stem),
            "graphs_current": _graph_summary(graph_dir / f"{stem}.pt", phys=phys, device=device),
        }
        raw = row["raw_export"]
        cur = row["graphs_current"]
        if raw.get("present") and cur.get("present"):
            row["delta_steps_vs_raw"] = int(cur["n_steps"]) - int(raw.get("n_raw_steps", 0))
        rows.append(row)

    summary = {
        "n_anchors": len(rows),
        "mean_n_steps_current": float(np.mean([r["graphs_current"]["n_steps"] for r in rows if r["graphs_current"].get("present")])),
        "anchors_matching_raw_steps": [
            r["anchor"]
            for r in rows
            if r.get("delta_steps_vs_raw", 1) == 0
        ],
    }

    payload = {"summary": summary, "anchors": rows}
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] {out}", flush=True)
    print(
        f"[i] mean n_steps current={summary['mean_n_steps_current']:.1f}",
        flush=True,
    )
    for r in rows:
        cur = r["graphs_current"]
        if not cur.get("present"):
            print(f"[WARN] {r['anchor']}: missing graph", flush=True)
            continue
        raw = r["raw_export"]
        print(
            f"[i] {r['anchor']}: steps {cur['n_steps']} "
            f"(raw {raw.get('n_raw_steps', '?')}) "
            f"t_last={cur['t_last_s']:.0f}s "
            f"clot_frac@53={cur.get('gt_clot_frac_at_legacy_53', float('nan')):.4f} "
            f"jacc@53={cur.get('gt_clot_jaccard_vs_final_at_legacy_53', float('nan')):.3f} "
            f"clot_frac@last={cur.get('gt_clot_frac_at_last', float('nan')):.4f} "
            f"jacc@last={cur.get('gt_clot_jaccard_vs_final_at_last', float('nan')):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
