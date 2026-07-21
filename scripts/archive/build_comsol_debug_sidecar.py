"""Build graph-aligned sidecar from COMSOL debug export (mu1, mu2, Mat, fi, spf.sr).

Writes ``data/processed/cfd_results_biochem_diag/{anchor}_debug.pt``::

    {
      "anchor": str,
      "times_s": Tensor[T],
      "gamma_si": Tensor[T, N],
      "mu1": Tensor[T, N],
      "mu2": Tensor[T, N],
      "mat_si": Tensor[T, N],
      "fi_si": Tensor[T, N],
    }

Usage::

    python scripts/build_comsol_debug_sidecar.py --anchor patient007
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

VARS_PER_STEP = 7
FIELD_NAMES = ("x", "y", "spf_sr", "mu1", "mu2", "mat", "fi")


def _parse_times(header_line: str) -> list[float]:
    return [float(x) for x in re.findall(r"spf\.sr @ t=([0-9.]+)", header_line)]


def load_debug_trajectory(filepath: Path) -> tuple[list[float], dict[float, pd.DataFrame]]:
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        header_line = ""
        for line in f:
            if "spf.sr @ t=" in line:
                header_line = line
                break
    if not header_line:
        raise ValueError(f"No spf.sr time header in {filepath}")

    times = _parse_times(header_line)
    n_times = len(times)
    expected_cols = 2 + VARS_PER_STEP * n_times
    print(f"[i] reading {filepath.name} ({n_times} times, expect {expected_cols} cols)", flush=True)
    df_full = pd.read_csv(filepath, comment="%", sep=r"\s+", header=None)
    if int(df_full.shape[1]) != expected_cols:
        raise ValueError(f"Column count {df_full.shape[1]} != {expected_cols}")

    blocks: dict[float, pd.DataFrame] = {}
    for i, t_val in enumerate(times):
        start = 2 + i * VARS_PER_STEP
        block = df_full.iloc[:, start : start + VARS_PER_STEP].copy()
        block.columns = list(FIELD_NAMES)
        blocks[float(t_val)] = block
    return times, blocks


def build_debug_sidecar(
    anchor: str,
    *,
    root: Path,
    debug_txt: Path | None = None,
    graph_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    phys = PhysicsConfig(phase="biochem")
    label_dir = root / "data" / "processed" / "cfd_results_biochem"
    debug_txt = debug_txt or label_dir / f"{anchor}_debugging.txt"
    graph_path = graph_path or root / "data" / "processed" / "graphs_biochem_anchors" / f"{anchor}.pt"
    out_path = out_path or root / "data" / "processed" / "cfd_results_biochem_diag" / f"{anchor}_debug.pt"

    if not debug_txt.is_file():
        raise FileNotFoundError(f"Missing debug export: {debug_txt}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph: {graph_path}")

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    n_nodes = int(data.num_nodes)
    graph_times = data.t.view(-1).cpu().numpy().astype(np.float64)

    comsol_times, blocks = load_debug_trajectory(debug_txt)
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
    match_frac = float(matched.mean())
    print(f"[i] spatial match {int(matched.sum())}/{n_nodes} ({match_frac:.3f})", flush=True)

    stacks: dict[str, list[torch.Tensor]] = {
        "gamma_si": [],
        "mu1": [],
        "mu2": [],
        "mat_si": [],
        "fi_si": [],
    }
    used_times: list[float] = []
    for t_graph in graph_times.tolist():
        key = float(t_graph)
        if key not in comsol_map:
            nearest = min(comsol_map.keys(), key=lambda t: abs(t - key))
            print(f"[WARN] graph t={key} missing; using t={nearest}", flush=True)
            key = float(nearest)
        df = comsol_map[key]
        idx = match_idx
        stacks["gamma_si"].append(torch.tensor(df.iloc[idx]["spf_sr"].values, dtype=torch.float32))
        stacks["mu1"].append(torch.tensor(df.iloc[idx]["mu1"].values, dtype=torch.float32))
        stacks["mu2"].append(torch.tensor(df.iloc[idx]["mu2"].values, dtype=torch.float32))
        stacks["mat_si"].append(torch.tensor(df.iloc[idx]["mat"].values, dtype=torch.float32))
        stacks["fi_si"].append(torch.tensor(df.iloc[idx]["fi"].values, dtype=torch.float32))
        used_times.append(key)

    payload = {
        "anchor": anchor,
        "times_s": torch.tensor(used_times, dtype=torch.float32),
        "match_frac": match_frac,
        "debug_txt": str(debug_txt),
        "graph": str(graph_path),
    }
    for name, rows in stacks.items():
        payload[name] = torch.stack(rows, dim=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    meta = {
        "anchor": anchor,
        "shape": {k: list(payload[k].shape) for k in stacks},
        "match_frac": match_frac,
        "out": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {anchor} -> {out_path}", flush=True)
    print(f"[save] {meta_path}", flush=True)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Build COMSOL debug sidecar")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--debug-txt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = get_project_root()
    build_debug_sidecar(
        args.anchor,
        root=root,
        debug_txt=Path(args.debug_txt) if args.debug_txt.strip() else None,
        out_path=Path(args.out) if args.out.strip() else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
