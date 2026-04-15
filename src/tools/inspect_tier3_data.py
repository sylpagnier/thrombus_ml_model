"""
Tier 3 export/graph inspector (restored + modernized, non-interactive first).

Examples:
    python -m src.tools.inspect_tier3_data --tier tier3_patients --summary
    python -m src.tools.inspect_tier3_data --tier tier3_patients --stem vessel_001 --unit-audit
    python -m src.tools.inspect_tier3_data --tier tier3_patients --stem vessel_001 --graph-summary
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

from src.config import BiochemConfig, VesselConfig
from src.utils.paths import get_project_root


FIELD_COLUMNS = [
    "x",
    "y",
    "u",
    "v",
    "p",
    "mu_eff",
    "rp",
    "ap",
    "apr",
    "aps",
    "PT",
    "th",
    "at",
    "fg",
    "fi",
    "M",
    "Mas",
    "Mat",
]


def _resolve_export_dir(tier: str) -> Path:
    cfg = VesselConfig(tier=tier)
    return Path(cfg.output_dir)


def _resolve_graph_dir(tier: str) -> Path:
    cfg = VesselConfig(tier=tier)
    return Path(cfg.graph_output_dir)


def _domain_txt_stems(export_dir: Path) -> list[str]:
    stems = []
    for p in sorted(export_dir.glob("*.txt")):
        if p.stem.endswith(("_inlet", "_outlet", "_wall")):
            continue
        stems.append(p.stem)
    return stems


def _parse_times_from_header(domain_file: Path) -> list[float]:
    times: list[float] = []
    with open(domain_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("% x") and "@ t=" in line:
                for match in re.finditer(r"t=([0-9.]+)", line):
                    t_val = float(match.group(1))
                    if t_val not in times:
                        times.append(t_val)
                break
    return times


def _load_first_block(domain_file: Path, sample_rows: int = 100000) -> pd.DataFrame:
    df_full = pd.read_csv(domain_file, comment="%", sep=r"\s+", header=None, nrows=sample_rows)
    if df_full.shape[1] < 20:
        raise ValueError(f"Unexpected format in {domain_file.name}: got {df_full.shape[1]} columns (need >=20).")
    df = df_full.iloc[:, 2:20].copy()
    df.columns = FIELD_COLUMNS
    return df


def _boundary_mask(boundary_file: Path, tree: cKDTree, num_nodes: int, tolerance: float = 1e-6) -> np.ndarray:
    mask = np.zeros(num_nodes, dtype=bool)
    if not boundary_file.exists():
        return mask
    bdf = pd.read_csv(boundary_file, comment="%", sep=r"\s+", header=None)
    coords = np.unique(bdf.iloc[:, -2:].values, axis=0)
    dist, idx = tree.query(coords)
    valid = idx[dist < tolerance]
    mask[valid] = True
    return mask


def inspect_boundaries(stem: str, export_dir: Path) -> None:
    domain_file = export_dir / f"{stem}.txt"
    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain export: {domain_file}")

    df = _load_first_block(domain_file)
    tree = cKDTree(df[["x", "y"]].values)
    n = len(df)

    m_in = _boundary_mask(export_dir / f"{stem}_inlet.txt", tree, n)
    m_out = _boundary_mask(export_dir / f"{stem}_outlet.txt", tree, n)
    m_wall = _boundary_mask(export_dir / f"{stem}_wall.txt", tree, n)
    m_int = ~(m_in | m_out | m_wall)

    print(f"\n=== Boundary summary: {stem} ===")
    print(f"Nodes total     : {n}")
    print(f"Inlet nodes     : {int(m_in.sum())}")
    print(f"Outlet nodes    : {int(m_out.sum())}")
    print(f"Wall nodes      : {int(m_wall.sum())}")
    print(f"Interior nodes  : {int(m_int.sum())}")


def audit_units(stem: str, export_dir: Path, sample_rows: int = 50000) -> None:
    domain_file = export_dir / f"{stem}.txt"
    if not domain_file.exists():
        raise FileNotFoundError(f"Missing domain export: {domain_file}")

    df = _load_first_block(domain_file, sample_rows=sample_rows)
    bio = BiochemConfig(tier="tier3")

    # Expected rough CGS baselines from config.
    expected_cgs = {
        "rp": bio.c_RP0 / 1e6,  # plt/ml
        "ap": (0.05 * bio.c_RP0) / 1e6,  # plt/ml
        "apr": bio.APRcrit * 1e3,  # uM
        "aps": bio.APScrit * 1e3,  # uM
        "PT": bio.c_pT0 * 1e3,  # uM
        "th": bio.Tcrit * 1e3,  # uM
        "at": bio.cAT0 * 1e3,  # uM
        "fg": bio.c_Fg0 * 1e3,  # uM
        "fi": bio.c_Fg0 * 1e3,  # uM proxy
        "M": bio.Minf / 1e4,  # plt/cm^2
        "Mas": bio.Minf / 1e4,
        "Mat": bio.Minf / 1e4,
    }

    species_cols = ["rp", "ap", "apr", "aps", "PT", "th", "at", "fg", "fi", "M", "Mas", "Mat"]

    print(f"\n=== Tier3 unit audit: {stem} ===")
    print(f"{'col':<5} {'p95+':>12} {'ref(CGS)':>12} {'ratio':>10}  likely family")
    for col in species_cols:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        pos = vals[vals > 0.0]
        p95 = float(np.nanpercentile(pos, 95)) if pos.size else 0.0
        ref = max(float(expected_cgs[col]), 1e-18)
        ratio = p95 / ref if p95 > 0 else 0.0
        if col in ("rp", "ap"):
            family = "plt/ml-ish" if 0.1 <= ratio <= 10 else "check"
        elif col in ("M", "Mas", "Mat"):
            family = "plt/cm^2-ish" if 0.1 <= ratio <= 10 else "check"
        else:
            family = "uM-ish" if 0.1 <= ratio <= 10 else "check"
        print(f"{col:<5} {p95:12.4g} {ref:12.4g} {ratio:10.3g}  {family}")

    print("Hint: for tier-3 solutes in uM, conversion to SI is uM * 1e-3 -> mol/m^3.")


def summarize_graph(stem: str, graph_dir: Path) -> None:
    graph_file = graph_dir / f"{stem}.pt"
    if not graph_file.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_file}")
    data = torch.load(graph_file, map_location="cpu", weights_only=False)
    print(f"\n=== Graph summary: {stem} ===")
    print(f"x shape              : {tuple(data.x.shape)}")
    print(f"y shape              : {tuple(data.y.shape) if hasattr(data, 'y') else '<missing>'}")
    print(f"num_nodes            : {int(data.num_nodes)}")
    print(f"num_edges            : {int(data.edge_index.shape[1])}")
    print(f"inlet/outlet/wall    : {int(data.mask_inlet.sum())}/{int(data.mask_outlet.sum())}/{int(data.mask_wall.sum())}")
    if hasattr(data, "t"):
        t = data.t.detach().cpu().numpy()
        if t.size > 1:
            print(f"time range           : {float(t.min()):.6g} -> {float(t.max()):.6g} (dt~{float(np.median(np.diff(t))):.6g})")
    if hasattr(data, "re_actual"):
        re_val = float(data.re_actual.mean().item()) if torch.is_tensor(data.re_actual) else float(data.re_actual)
        print(f"re_actual            : {re_val:.4g}")


def print_summary_table(tier: str, export_dir: Path, graph_dir: Path) -> None:
    stems = _domain_txt_stems(export_dir)
    if not stems:
        print(f"No tier3 export stems found in {export_dir}")
        return

    print(f"\n=== Tier3 summary ({tier}) ===")
    print(f"{'stem':<24} {'times':>6} {'graph?':>8}")
    for stem in stems:
        times = _parse_times_from_header(export_dir / f"{stem}.txt")
        g_exists = (graph_dir / f"{stem}.pt").exists()
        print(f"{stem:<24} {len(times):>6} {str(g_exists):>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tier3 exports and processed graphs.")
    parser.add_argument("--tier", type=str, default="tier3_patients", choices=["tier3", "tier3_patients"])
    parser.add_argument("--stem", type=str, default=None, help="Stem name without extension (e.g. vessel_001).")
    parser.add_argument("--summary", action="store_true", help="Print summary table for all stems.")
    parser.add_argument("--boundaries", action="store_true", help="Print boundary node counts for the selected stem.")
    parser.add_argument("--unit-audit", action="store_true", help="Run unit-magnitude audit for the selected stem.")
    parser.add_argument("--graph-summary", action="store_true", help="Print processed .pt summary for the selected stem.")
    parser.add_argument("--sample-rows", type=int, default=50000, help="Rows to sample for unit audit.")
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.tier)
    graph_dir = _resolve_graph_dir(args.tier)
    if not export_dir.exists():
        raise FileNotFoundError(f"Export dir does not exist: {export_dir}")

    if args.summary:
        print_summary_table(args.tier, export_dir, graph_dir)

    # Auto-select first stem when not explicitly provided.
    stem = args.stem
    if stem is None and (args.boundaries or args.unit_audit or args.graph_summary):
        stems = _domain_txt_stems(export_dir)
        if not stems:
            raise FileNotFoundError(f"No domain txt files found in {export_dir}")
        stem = stems[0]

    if args.boundaries:
        inspect_boundaries(stem, export_dir)
    if args.unit_audit:
        audit_units(stem, export_dir, sample_rows=max(1000, args.sample_rows))
    if args.graph_summary:
        summarize_graph(stem, graph_dir)

    if not any([args.summary, args.boundaries, args.unit_audit, args.graph_summary]):
        print_summary_table(args.tier, export_dir, graph_dir)


if __name__ == "__main__":
    main()
