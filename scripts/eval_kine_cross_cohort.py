"""Cross-cohort kinematics (GINO-DEQ) evaluation: patient anchors vs synthetic kinematics graphs.

Compares rel-L2 / L1 on [u,v,p] for:
  - ``graphs_biochem_anchors`` (patient stems)
  - ``graphs_kinematics/newtonian`` (synthetic), grouped by mesh JSON ``level`` (0/1/2)

Patient graphs store **biochem** node features (15ch). The kinematics model expects **kine** layout
(18ch with shear_pot, priors, width). Use ``--x-mode native`` (current viz path) vs
``--x-mode kine_layout`` (remap wall normals + Poiseuille priors) to separate schema mismatch
from true geometry generalization.

Example:
    python scripts/eval_kine_cross_cohort.py
    python scripts/eval_kine_cross_cohort.py --max-per-cohort 20 --x-mode both
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal, Optional

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architecture.kinematics_model_config import (
    build_gino_deq_from_ctor,
    kinematics_checkpoint_tensors,
    load_kinematics_reference_record,
    resolve_gino_deq_ctor_kwargs,
)
from src.config import NodeFeat, PhysicsConfig, PredChannels
from src.data_gen.lib.graph_velocity_priors import mass_conserving_umax_nd, width_nd_to_radius_nd
from src.utils.kinematics_geometry import graph_geometry_level, read_geometry_level_from_mesh_json
from src.utils.paths import data_root, resolve_checkpoint

XMode = Literal["native", "kine_layout", "both"]


@dataclass
class GraphMetrics:
    stem: str
    cohort: str
    x_mode: str
    geometry_level: int
    n_nodes: int
    x_channels: int
    x_schema: str
    time_index: int
    rel_l2_uvp: float
    rel_l2_u: float
    rel_l2_v: float
    rel_l2_p: float
    l1_uvp: float
    max_pred_uv: float
    max_tgt_uv: float
    wall_normal_norm_mean: float
    uv_prior_max: float
    mu_prior_mean: float
    width_nd_max: float


def _load_kinematics_model(device: torch.device):
    ckpt_path = resolve_checkpoint("a", "kinematics_best.pth")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta, state = kinematics_checkpoint_tensors(raw)
    ref = load_kinematics_reference_record()
    if ref and not meta.get("model_config"):
        meta = {**meta, "model_config": ref.get("model_config")}
    ctor = resolve_gino_deq_ctor_kwargs(meta, state)
    phys = PhysicsConfig(phase="kinematics", rheology="newtonian")
    model = build_gino_deq_from_ctor(phys, ctor).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt_path, ctor


def _steady_kine_targets(data: torch.Tensor, time_index: int) -> torch.Tensor:
    y = data.y
    if y.dim() == 3:
        t = time_index if time_index >= 0 else y.shape[0] + time_index
        t = max(0, min(int(t), y.shape[0] - 1))
        return y[t, :, :5]
    return y[:, :5]


def _read_x_schema(data) -> str:
    return str(getattr(data, "x_schema", "") or "unknown")


def _build_kine_layout_x(data, phys: PhysicsConfig) -> torch.Tensor:
    """Best-effort kine ``x`` (18ch) from a biochem-anchor graph (correct channel semantics)."""
    xb = data.x
    n = xb.shape[0]
    device, dtype = xb.device, xb.dtype

    pos_nd = xb[:, 0:2]
    sdf = xb[:, 2:3].clamp_min(0.0)
    wall_normal = xb[:, 3:5]
    shear_pot = torch.abs(1.0 - 2.0 * sdf)

    # Crude hydraulic width from SDF (patient graphs lack width bands).
    width_nd = (2.0 * sdf).clamp(min=1e-4, max=6.0)
    R_nd = width_nd_to_radius_nd(width_nd).view(-1, 1)
    u_max_nd = mass_conserving_umax_nd(R_nd).view(-1, 1)
    r_nd = (R_nd - torch.minimum(sdf.view(-1, 1), R_nd)).clamp_min(0.0)
    u_prior_mag = torch.clamp(u_max_nd * (1.0 - (r_nd ** 2 / (R_nd ** 2 + 1e-12))), min=0.0)
    # Flow direction from wall normal (90 deg CCW), unit normalized.
    flow_x = -wall_normal[:, 1:2]
    flow_y = wall_normal[:, 0:1]
    flow_norm = torch.sqrt(flow_x ** 2 + flow_y ** 2).clamp_min(1e-8)
    flow_x = flow_x / flow_norm
    flow_y = flow_y / flow_norm
    u_prior = u_prior_mag * flow_x
    v_prior = u_prior_mag * flow_y

    mu_prior = torch.ones(n, 1, device=device, dtype=dtype)
    gamma_dot = torch.abs(-2.0 * u_max_nd * r_nd / (R_nd ** 2 + 1e-12))
    wss_prior = (mu_prior * gamma_dot) * data.mask_wall.view(-1, 1).to(dtype=dtype)

    width_d1 = torch.zeros(n, 1, device=device, dtype=dtype)
    width_d2 = torch.zeros(n, 1, device=device, dtype=dtype)
    if hasattr(data, "G_x") and hasattr(data, "G_y"):
        try:
            gx = data.G_x
            gy = data.G_y
            grad_wx = torch.sparse.mm(gx, width_nd).squeeze(1)
            grad_wy = torch.sparse.mm(gy, width_nd).squeeze(1)
            width_d1 = (grad_wx * flow_x.squeeze(1) + grad_wy * flow_y.squeeze(1)).unsqueeze(1)
        except Exception:
            pass

    rheo_flag = torch.zeros(n, 1, device=device, dtype=dtype)
    node_type_pad = torch.zeros(n, 4, device=device, dtype=dtype)

    return torch.cat(
        [
            pos_nd,
            sdf,
            shear_pot,
            wall_normal,
            node_type_pad,
            rheo_flag,
            u_prior,
            v_prior,
            mu_prior,
            wss_prior,
            width_nd,
            width_d1,
            width_d2,
        ],
        dim=1,
    )


def _apply_x_mode(data, x_mode: str, phys: PhysicsConfig):
    if x_mode == "native":
        if int(data.x.shape[1]) >= NodeFeat.WIDTH_D2.stop:
            return data
        raise ValueError(
            f"native x has {data.x.shape[1]} channels; re-run PatientDataExtractor for 18ch kine x."
        )
    data = data.clone()
    data.x = _build_kine_layout_x(data, phys)
    return data


def _node_mask(data) -> torch.Tensor:
    if hasattr(data, "is_anchor"):
        ia = data.is_anchor.view(-1).bool()
        if ia.numel() == 1:
            return torch.ones(data.num_nodes, dtype=torch.bool, device=ia.device)
        return ia
    return torch.ones(data.num_nodes, dtype=torch.bool)


def _metrics(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    p = pred[mask, :3]
    t = tgt[mask, :3]
    if p.numel() == 0:
        return {k: float("nan") for k in ("rel_l2_uvp", "rel_l2_u", "rel_l2_v", "rel_l2_p", "l1_uvp")}
    diff = p - t
    rel_l2_uvp = (diff.norm() / (t.norm() + 1e-8)).item()
    out = {"rel_l2_uvp": rel_l2_uvp, "l1_uvp": F.l1_loss(p, t).item()}
    for j, key in enumerate(("rel_l2_u", "rel_l2_v", "rel_l2_p")):
        out[key] = (p[:, j] - t[:, j]).norm().item() / (t[:, j].norm().item() + 1e-8)
    return out


def _feature_diag(x: torch.Tensor) -> dict[str, float]:
    wn = x[:, NodeFeat.WALL_NORMAL]
    return {
        "wall_normal_norm_mean": float(wn.norm(dim=1).mean().item()),
        "uv_prior_max": float(x[:, NodeFeat.UV_PRIOR].abs().max().item()),
        "mu_prior_mean": float(x[:, NodeFeat.MU_PRIOR].mean().item()),
        "width_nd_max": float(x[:, NodeFeat.WIDTH_ND].max().item()) if x.shape[1] >= NodeFeat.WIDTH_D2.stop else 0.0,
    }


def _resolve_geometry_level(stem: str, data, mesh_dir: Path) -> int:
    lvl = graph_geometry_level(data, default=-1)
    if lvl >= 0:
        return lvl
    from_json = read_geometry_level_from_mesh_json(mesh_dir, stem)
    return int(from_json) if from_json is not None else -1


def _eval_one(
    path: Path,
    *,
    cohort: str,
    model,
    device: torch.device,
    phys: PhysicsConfig,
    x_mode: str,
    time_index: int,
    mesh_dir: Path,
) -> GraphMetrics:
    data = torch.load(path, map_location=device, weights_only=False)
    stem = path.stem
    lvl = _resolve_geometry_level(stem, data, mesh_dir)
    x_schema = _read_x_schema(data)
    tgt = _steady_kine_targets(data, time_index)
    data_eval = _apply_x_mode(data, x_mode, phys)
    t_idx = time_index if data.y.dim() == 2 else (
        time_index if time_index >= 0 else data.y.shape[0] + time_index
    )

    with torch.no_grad():
        pred = model(data_eval, solver="anderson", anderson_beta=0.8)
        if isinstance(pred, tuple):
            pred = pred[0]

    mask = _node_mask(data_eval)
    if float(tgt[mask, :3].norm().item()) < 1e-6:
        raise ValueError("degenerate target norm (near-zero u,v,p); skip or check labels")
    m = _metrics(pred, tgt, mask)
    fd = _feature_diag(data_eval.x)

    return GraphMetrics(
        stem=stem,
        cohort=cohort,
        x_mode=x_mode,
        geometry_level=lvl,
        n_nodes=int(data_eval.num_nodes),
        x_channels=int(data_eval.x.shape[1]),
        x_schema=x_schema,
        time_index=int(t_idx),
        rel_l2_uvp=m["rel_l2_uvp"],
        rel_l2_u=m["rel_l2_u"],
        rel_l2_v=m["rel_l2_v"],
        rel_l2_p=m["rel_l2_p"],
        l1_uvp=m["l1_uvp"],
        max_pred_uv=float(pred[:, PredChannels.UV].abs().max().item()),
        max_tgt_uv=float(tgt[:, PredChannels.UV].abs().max().item()),
        **fd,
    )


def _cohort_label(level: int) -> str:
    if level in (0, 1, 2):
        return f"kine_L{level}"
    return "kine_unknown"


def _iter_kine_graphs(kine_dir: Path, levels: Optional[set[int]], mesh_dir: Path) -> Iterable[tuple[str, Path]]:
    for pt in sorted(kine_dir.glob("vessel_*.pt")):
        lvl = read_geometry_level_from_mesh_json(mesh_dir, pt.stem)
        if lvl is None:
            lvl = -1
        if levels is not None and lvl not in levels:
            continue
        yield _cohort_label(lvl), pt


def _summarize(rows: list[GraphMetrics]) -> None:
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[GraphMetrics]] = defaultdict(list)
    for r in rows:
        buckets[(r.cohort, r.x_mode)].append(r)

    print("\n=== Summary (mean rel_L2 uvp) ===")
    print(f"{'cohort':<16} {'x_mode':<12} {'n':>4} {'rel_l2':>8} {'rel_u':>8} {'rel_v':>8} {'rel_p':>8} {'l1':>8}")
    for (cohort, x_mode), items in sorted(buckets.items()):
        n = len(items)
        if n == 0:
            continue
        rel = sum(x.rel_l2_uvp for x in items) / n
        ru = sum(x.rel_l2_u for x in items) / n
        rv = sum(x.rel_l2_v for x in items) / n
        rp = sum(x.rel_l2_p for x in items) / n
        l1 = sum(x.l1_uvp for x in items) / n
        print(f"{cohort:<16} {x_mode:<12} {n:4d} {rel:8.4f} {ru:8.4f} {rv:8.4f} {rp:8.4f} {l1:8.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description="Kinematics model cross-cohort evaluation.")
    p.add_argument("--max-per-cohort", type=int, default=30, help="Cap graphs per cohort (0 = all).")
    p.add_argument("--time-index", type=int, default=0, help="Biochem time index for targets (0 = first export).")
    p.add_argument(
        "--x-mode",
        choices=("native", "kine_layout", "both"),
        default="both",
        help="Patient x: biochem layout vs remapped kine layout. Kine graphs always native.",
    )
    p.add_argument("--levels", type=str, default="0,1", help="Kinematics geometry levels to include (comma sep).")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV path for per-graph rows.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed when subsampling.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dr = data_root()
    kine_dir = dr / "processed/graphs_kinematics/newtonian"
    patient_dir = dr / "processed/graphs_biochem_anchors"
    mesh_dir = dr / "raw/kinematics/meshes"
    if not mesh_dir.is_dir():
        mesh_dir = dr / "raw/kinematics"

    level_set = {int(x.strip()) for x in args.levels.split(",") if x.strip() != ""}

    model, ckpt_path, ctor = _load_kinematics_model(device)
    phys = PhysicsConfig(phase="kinematics", rheology="newtonian")
    print(f"[i] checkpoint: {ckpt_path}")
    print(f"[i] ctor: in_ch={ctor.get('in_channels')} width_priors={ctor.get('use_width_priors')} siren={ctor.get('use_siren_decoder')}")
    print(f"[i] device: {device}")

    x_modes: list[str]
    if args.x_mode == "both":
        x_modes = ["native", "kine_layout"]
    else:
        x_modes = [args.x_mode]

    jobs: list[tuple[str, Path, str]] = []
    for cohort, pt in _iter_kine_graphs(kine_dir, level_set, mesh_dir):
        jobs.append((cohort, pt, "native"))
    for pt in sorted(patient_dir.glob("*.pt")):
        g = torch.load(pt, map_location="cpu", weights_only=False)
        has_kine_x = int(g.x.shape[1]) >= NodeFeat.WIDTH_D2.stop
        if has_kine_x:
            jobs.append(("patient", pt, "native"))
        else:
            for xm in x_modes:
                jobs.append(("patient", pt, xm))

    if args.max_per_cohort > 0:
        import random

        rng = random.Random(args.seed)
        by_cohort: dict[tuple[str, str], list] = {}
        for item in jobs:
            key = (item[0], item[2]) if item[0] == "patient" else (item[0], "native")
            by_cohort.setdefault(key, []).append(item)
        jobs = []
        for key, group in by_cohort.items():
            rng.shuffle(group)
            jobs.extend(group[: args.max_per_cohort])

    rows: list[GraphMetrics] = []
    for cohort, pt, xm in jobs:
        try:
            rows.append(
                _eval_one(
                    pt,
                    cohort=cohort,
                    model=model,
                    device=device,
                    phys=phys,
                    x_mode=xm,
                    time_index=args.time_index,
                    mesh_dir=mesh_dir,
                )
            )
            r = rows[-1]
            print(
                f"  {r.stem:<14} {r.cohort:<12} {r.x_mode:<11} "
                f"rel_l2={r.rel_l2_uvp:.4f} n={r.n_nodes} x={r.x_channels}ch schema={r.x_schema}"
            )
        except Exception as exc:
            print(f"  [ERR] {pt.name} ({cohort}/{xm}): {exc}")

    _summarize(rows)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f"\n[save] {args.out_csv}")

    print(
        "\n[i] Patient graphs need 18ch data.x (kine_x_v1_18ch) from PatientDataExtractor; "
        "15ch-only anchors must be re-extracted. Legacy approx: --x-mode kine_layout."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
