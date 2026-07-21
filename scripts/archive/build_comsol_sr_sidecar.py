"""Match COMSOL ``spf.sr`` wide export to biochem anchor graph nodes.

Writes ``data/processed/cfd_results_biochem_diag/{anchor}_sr.pt``::

    {"gamma_si": Tensor[T, N], "times_s": Tensor[T], "anchor": str}

Usage::

    python scripts/build_comsol_sr_sidecar.py --anchor patient007
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _parse_sr_times(header_line: str) -> list[float]:
    times: list[float] = []
    for match in re.finditer(r"t=([0-9.]+)", header_line):
        t_val = float(match.group(1))
        if t_val not in times:
            times.append(t_val)
    return times


def load_sr_trajectory(filepath: Path) -> tuple[list[float], pd.DataFrame]:
    """Load COMSOL export with columns ``x, y, spf.sr`` per time block."""
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        header_line = ""
        for line in f:
            if "@ t=" in line:
                header_line = line
                break
    if not header_line:
        raise ValueError(f"No time header in {filepath}")

    times = _parse_sr_times(header_line)
    vars_per_step = 3
    df_full = pd.read_csv(filepath, comment="%", sep=r"\s+", header=None)
    if df_full.shape[1] != 2 + vars_per_step * len(times):
        raise ValueError(
            f"Column count {df_full.shape[1]} != 2 + 3*{len(times)} in {filepath.name}"
        )

    blocks: dict[float, pd.DataFrame] = {}
    for i, t_val in enumerate(times):
        start_col = 2 + i * vars_per_step
        block = df_full.iloc[:, start_col : start_col + vars_per_step].copy()
        block.columns = ["x", "y", "spf_sr"]
        blocks[float(t_val)] = block
    return times, blocks


def build_sidecar(
    anchor: str,
    *,
    root: Path,
    sr_txt: Path | None = None,
    graph_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    phys = PhysicsConfig(phase="biochem")
    label_dir = root / "data" / "processed" / "cfd_results_biochem"
    sr_txt = sr_txt or label_dir / f"{anchor}_sr.txt"
    graph_path = graph_path or root / "data" / "processed" / "graphs_biochem_anchors" / f"{anchor}.pt"
    out_path = out_path or root / "data" / "processed" / "cfd_results_biochem_diag" / f"{anchor}_sr.pt"

    if not sr_txt.is_file():
        raise FileNotFoundError(f"Missing COMSOL sr export: {sr_txt}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph: {graph_path}")

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    n_nodes = int(data.num_nodes)
    graph_times = data.t.view(-1).cpu().numpy().astype(np.float64)
    n_steps = int(graph_times.shape[0])

    comsol_times, blocks = load_sr_trajectory(sr_txt)
    comsol_map = {float(t): blocks[float(t)] for t in comsol_times}

    df0 = comsol_map[float(comsol_times[0])]
    coords_m = df0[["x", "y"]].values.astype(np.float64) * float(phys.cm_to_m)
    mesh_nodes = data.x[:, :2].cpu().numpy().astype(np.float64) * float(
        data.d_bar.view(-1)[0].item()
    )
    tree = cKDTree(coords_m)
    dist, match_idx = tree.query(mesh_nodes)
    tol_m = float(phys.comsol_spatial_match_tol_m)
    matched = dist < tol_m
    if int(matched.sum()) < n_nodes * 0.9:
        print(
            f"[WARN] only {int(matched.sum())}/{n_nodes} nodes within tol={tol_m} m",
            flush=True,
        )

    gamma_rows: list[torch.Tensor] = []
    used_times: list[float] = []
    for t_graph in graph_times.tolist():
        key = float(t_graph)
        if key not in comsol_map:
            nearest = min(comsol_map.keys(), key=lambda t: abs(t - key))
            print(f"[WARN] graph t={key} missing in sr export; using t={nearest}", flush=True)
            key = float(nearest)
        df = comsol_map[key]
        sr_raw = df.iloc[match_idx]["spf_sr"].values.astype(np.float64)
        # COMSOL spf.sr is 1/s in SI for Laminar Flow.
        gamma_si = torch.tensor(sr_raw, dtype=torch.float32)
        gamma_rows.append(gamma_si)
        used_times.append(key)

    gamma_stack = torch.stack(gamma_rows, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor": anchor,
        "gamma_si": gamma_stack,
        "times_s": torch.tensor(used_times, dtype=torch.float32),
        "match_frac": float(matched.mean()),
        "sr_txt": str(sr_txt),
        "graph": str(graph_path),
    }
    torch.save(payload, out_path)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "anchor": anchor,
                "shape": list(gamma_stack.shape),
                "times_s_first": used_times[:5],
                "times_s_last": used_times[-3:],
                "gamma_si_median_t0": float(gamma_stack[0].median().item()),
                "match_frac": payload["match_frac"],
                "out": str(out_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] {anchor}: gamma_si {tuple(gamma_stack.shape)} -> {out_path}", flush=True)
    print(f"[save] {meta_path}", flush=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build COMSOL spf.sr sidecar for anchor graph")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--sr-txt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    sr_txt = Path(args.sr_txt) if args.sr_txt.strip() else None
    out_path = Path(args.out) if args.out.strip() else None
    build_sidecar(args.anchor, root=root, sr_txt=sr_txt, out_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
