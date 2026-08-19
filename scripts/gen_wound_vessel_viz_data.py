"""JSON payload for the wound-vessel data-health visualization.

Extends the two-window temporal template (docs/VIZ_STANDARD.md). There is no model
here: left window is wall identity (mask_wound when present), right is GT clot.
If mask_wound is empty, the left window falls back to wall Mat as a diagnostic
proxy for the wound-flux patch -- labeled as such in the payload, not as the
COMSOL wound selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.mls_gradient import node_positions  # noqa: E402
from src.core_physics.physics_lumen_model import median_edge_length, wall_normal_projection  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils import species_channels as sc  # noqa: E402

DIR = Path("data/processed/graphs_biochem_anchors")
VESSELS = ["wound_patient001", "wound_patient002", "wound_patient003"]
N_FRAMES = 13
MAX_BG_POINTS = 1800
FLUX_WARN = 0.02


def _bool_mask(data, name: str, n: int) -> np.ndarray:
    t = getattr(data, name, None)
    if t is None:
        return np.zeros(n, dtype=bool)
    return np.asarray(t).reshape(-1).astype(bool)


def _meta_flux(stem: str) -> float | None:
    p = DIR / f"{stem}_metadata.json"
    if not p.is_file():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    q = meta.get("quality") or {}
    val = q.get("mass_flux_imbalance")
    return float(val) if val is not None else None


def _pts(pos: np.ndarray, idx: np.ndarray) -> list[list[float]]:
    return [[round(float(pos[i, 0]), 4), round(float(pos[i, 1]), 4)] for i in idx]


def main() -> None:
    phys = PhysicsConfig(phase="biochem")
    mat_i = sc.y_index("Mat")
    out: dict = {}
    for stem in VESSELS:
        path = DIR / f"{stem}.pt"
        if not path.is_file():
            print(f"[WARN] skip {stem}: missing {path}")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        pos = node_positions(d)
        n = int(pos.shape[0])
        wall = _bool_mask(d, "mask_wall", n)
        wound = _bool_mask(d, "mask_wound", n)
        inlet = _bool_mask(d, "mask_inlet", n)
        outlet = _bool_mask(d, "mask_outlet", n)
        # Wound is stored disjoint from wall; identity map wants the union.
        wall_all = wall | wound
        ei = d.edge_index.numpy()
        t = np.asarray(d.t).reshape(-1)
        y = d.y
        interior = np.where(~wall_all & ~inlet & ~outlet)[0]
        stride = max(1, len(interior) // MAX_BG_POINTS)
        bg = interior[::stride]

        gt_hot = np.zeros((len(t), n), dtype=bool)
        mat = np.zeros((len(t), n), dtype=np.float32)
        for i in range(len(t)):
            gt_hot[i] = gt_clot_phi_at_time(d, i, phys, device=torch.device("cpu")).numpy() > 0.5
            mat[i] = y[i, :, mat_i].detach().cpu().numpy().astype(np.float32)

        wall_idx = np.where(wall_all)[0]
        wound_idx = np.where(wound)[0]
        inlet_idx = np.where(inlet)[0]
        outlet_idx = np.where(outlet)[0]
        gt_lumen_any = gt_hot.any(axis=0) & ~wall_all
        lumen_render = np.where(gt_lumen_any)[0]
        frame_idx = np.linspace(0, len(t) - 1, N_FRAMES).round().astype(int)

        dist_raw, _ = wall_normal_projection(pos, wall_all)
        h_edge = median_edge_length(pos, ei)
        dist_norm = np.clip(dist_raw / (1.5 * max(h_edge, 1e-9)), 0.0, 1.0)

        mat_wall_max = float(mat[:, wall_all].max()) if wall_all.any() else 0.0
        mat_scale = mat_wall_max if mat_wall_max > 1e-12 else 1.0
        left_mode = "wound_mask" if bool(wound.any()) else "mat_diagnostic"

        n_wall = int(wall_all.sum())
        n_wound = int(wound.sum())
        n_inlet = int(inlet.sum())
        n_outlet = int(outlet.sum())
        n_clot0 = int(gt_hot[0].sum())
        n_clot_f = int(gt_hot[-1].sum())
        n_clot_wall_f = int((gt_hot[-1] & wall_all).sum())
        n_clot_off_f = int((gt_hot[-1] & ~wall_all).sum())
        n_clot_wound_f = int((gt_hot[-1] & wound).sum()) if n_wound else 0
        flux = _meta_flux(stem)

        flags = {
            "wound_known": n_wound > 0,
            "clot_present": n_clot_f > 0,
            "inlet_ok": n_inlet > 0,
            "outlet_ok": n_outlet > 0,
            "wall_ok": n_wall > 0,
            "flux_ok": flux is not None and flux < FLUX_WARN,
            "clot_starts_empty": n_clot0 == 0,
        }

        clot_wall_frac = []
        clot_off_frac = []
        clot_wound_frac = []
        mat_wall_mean = []
        for i in range(len(t)):
            clot_wall_frac.append(round(float((gt_hot[i] & wall_all).sum()) / max(n_wall, 1), 4))
            clot_off_frac.append(round(float((gt_hot[i] & ~wall_all).sum()) / max(n - n_wall, 1), 4))
            if n_wound:
                clot_wound_frac.append(
                    round(float((gt_hot[i] & wound).sum()) / n_wound, 4)
                )
            else:
                clot_wound_frac.append(0.0)
            mat_wall_mean.append(
                round(float(mat[i, wall_all].mean()) if n_wall else 0.0, 6)
            )

        out[stem] = {
            "left_mode": left_mode,
            "biochem_variant": str(getattr(d, "biochem_variant", "wound")),
            "source_mph": str(getattr(d, "source_mph", "") or ""),
            "t_final": float(t[-1]),
            "n_nodes": n,
            "n_wall": n_wall,
            "n_wound": n_wound,
            "n_inlet": n_inlet,
            "n_outlet": n_outlet,
            "n_clot_t0": n_clot0,
            "n_clot_final": n_clot_f,
            "n_clot_wall_final": n_clot_wall_f,
            "n_clot_off_final": n_clot_off_f,
            "n_clot_wound_final": n_clot_wound_f,
            "mat_wall_max": round(mat_wall_max, 6),
            "flux_imbalance": None if flux is None else round(float(flux), 5),
            "flags": flags,
            "bg": _pts(pos, bg),
            "wall_pos": _pts(pos, wall_idx),
            "wall_is_wound": [bool(wound[i]) for i in wall_idx],
            "inlet_pos": _pts(pos, inlet_idx),
            "outlet_pos": _pts(pos, outlet_idx),
            "lumen_pos": _pts(pos, lumen_render),
            "lumen_dist": [round(float(x), 3) for x in dist_norm[lumen_render]],
            "frame_t": [round(float(t[i]), 1) for i in frame_idx],
            "frame_gt_wall": [[bool(x) for x in gt_hot[i][wall_idx]] for i in frame_idx],
            "frame_gt_lumen": [[bool(x) for x in gt_hot[i][lumen_render]] for i in frame_idx],
            "frame_mat_wall": [
                [round(float(mat[i, j] / mat_scale), 3) for j in wall_idx] for i in frame_idx
            ],
            "frame_n_gt_wall": [int((gt_hot[i] & wall_all).sum()) for i in frame_idx],
            "frame_n_gt_lumen": [int((gt_hot[i] & ~wall_all).sum()) for i in frame_idx],
            "score_t": [round(float(x), 1) for x in t],
            "clot_wall_frac": clot_wall_frac,
            "clot_off_frac": clot_off_frac,
            "clot_wound_frac": clot_wound_frac,
            "mat_wall_mean": mat_wall_mean,
        }
        tag = "wound-ok" if flags["wound_known"] else "NO-WOUND-MASK"
        print(
            f"[i] {stem} [{tag}] left={left_mode} n={n} wall={n_wall} wound={n_wound} "
            f"inlet={n_inlet} outlet={n_outlet} clot_final={n_clot_f} "
            f"(wall {n_clot_wall_f} / lumen {n_clot_off_f}) "
            f"flux={flux if flux is not None else '-'} Mat_wall_max={mat_wall_max:.4g}"
        )

    out_path = Path("outputs/wound_vessel_temporal_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"[save] {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
