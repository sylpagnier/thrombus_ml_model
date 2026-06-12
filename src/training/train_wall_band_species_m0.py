"""Train M0 wall-band species head (1-step teacher-forced).

Usage::

    python -m src.training.train_wall_band_species_m0 --channel-set fimat
    python -m src.training.train_wall_band_species_m0 --channel-set cascade4 --val-anchor patient007
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi
from src.core_physics.t0_rung4_ladder import rollout_rung4_species_series
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.core_physics.wall_band_species_m0 import (
    CHANNEL_SETS,
    build_m0_bundle,
    load_m0_bundle,
    one_step_loss,
    rollout_m0_species_series,
    save_m0_checkpoint,
)
from src.evaluation.rung4_rollout_health import compute_rung4_rollout_health
from src.training.train_clot_phi_simple import _clot_metrics
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time
from src.utils.paths import get_project_root


def _list_anchors(root: Path) -> list[Path]:
    paths = sorted(root.glob("patient*.pt"))
    if not paths:
        raise FileNotFoundError(f"No anchors in {root}")
    return paths


def _anchor_to_device(data, device: torch.device):
    """Move graph tensors to ``device`` (including ``y`` for species slices)."""
    if getattr(data, "_m0_gpu_ready", False):
        return data
    for key in data.keys():
        val = data[key]
        if isinstance(val, torch.Tensor):
            data[key] = val.to(device=device)
    data._m0_gpu_ready = True
    return data


def _val_clot_f1(data, bundle, *, phys, bio, device) -> float:
    data = _anchor_to_device(data, device)
    pred = rollout_m0_species_series(data, bundle, phys_cfg=phys, bio_cfg=bio, device=device)
    t_last = int(data.y.shape[0]) - 1
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, device,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
            pred_species_series=pred, nucleation=True, nucleation_hops=1,
        )
    phi_traj = {int(t): v["phi"] for t, v in traj.items()}
    phi_gt = gt_clot_phi_at_time(data, t_last, phys, device).reshape(-1)
    phi_p = phi_traj[t_last].reshape(-1)
    m = _clot_metrics(phi_p, phi_gt, torch.ones(phi_gt.numel(), device=device, dtype=torch.bool))
    return float(m["clot_f1"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Train wall-band species M0")
    ap.add_argument("--channel-set", default="fimat", choices=sorted(CHANNEL_SETS))
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--time-stride", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--out",
        default="",
        help="Default outputs/biochem/wall_band_species_m0_<channel_set>/best.pth",
    )
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    cs = args.channel_set.strip().lower()
    out = Path(args.out) if args.out.strip() else (
        root / f"outputs/biochem/wall_band_species_m0_{cs}/best.pth"
    )
    if not out.is_absolute():
        out = root / out

    graph_dir = root / args.graph_dir
    anchors = _list_anchors(graph_dir)
    val_stem = args.val_anchor.strip().lower()
    train_paths = [p for p in anchors if p.stem.lower() != val_stem]
    val_paths = [p for p in anchors if p.stem.lower() == val_stem]
    if not val_paths:
        val_paths = [anchors[-1]]
        train_paths = anchors[:-1]

    bundle = build_m0_bundle(cs, device, hidden=int(args.hidden))
    opt = torch.optim.Adam(bundle.model.parameters(), lr=float(args.lr))
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    stride = max(int(args.time_stride), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.parent / "train_log.jsonl"

    print(
        f"[i] M0 channel_set={cs} channels={bundle.channel_names} "
        f"in_dim={bundle.in_dim} hidden={bundle.hidden}",
        flush=True,
    )
    print(
        f"[i] train={[p.stem for p in train_paths]} val={[p.stem for p in val_paths]} "
        f"epochs={args.epochs} stride={stride}",
        flush=True,
    )

    best_loss = 1e18
    best_f1 = -1.0
    stale = 0
    # Zero-init is the safe floor (rollout with no deltas -> no wall commits).
    save_m0_checkpoint(
        out, bundle,
        meta={
            "val_anchor": val_stem,
            "val_loss": 1e18,
            "val_f1": 0.0,
            "s0_f1": 0.0,
            "epoch": 0,
            "note": "zero_init_seed",
        },
    )

    for ep in range(1, int(args.epochs) + 1):
        t0 = time.perf_counter()
        bundle.model.train()
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        n_steps = 0
        for p in train_paths:
            data = torch.load(p, map_location="cpu", weights_only=False)
            data = _anchor_to_device(data, device)
            n_t = int(data.y.shape[0])
            times = list(range(1, n_t, stride))
            if (n_t - 1) not in times:
                times.append(n_t - 1)
            for t in times:
                loss = one_step_loss(data, t, bundle, phys_cfg=phys, bio_cfg=bio, device=device)
                loss.backward()
                loss_sum += float(loss.detach().item())
                n_steps += 1
            del data
            if device.type == "cuda":
                torch.cuda.empty_cache()
        opt.step()
        train_loss = loss_sum / max(n_steps, 1)

        bundle.model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            n_val_steps = 0
            for p in val_paths:
                data = torch.load(p, map_location="cpu", weights_only=False)
                data = _anchor_to_device(data, device)
                n_t = int(data.y.shape[0])
                times = list(range(1, n_t, stride))
                if (n_t - 1) not in times:
                    times.append(n_t - 1)
                for t in times:
                    val_loss += float(
                        one_step_loss(data, t, bundle, phys_cfg=phys, bio_cfg=bio, device=device).item()
                    )
                    n_val_steps += 1
                val_f1 += _val_clot_f1(data, bundle, phys=phys, bio=bio, device=device)
            val_loss /= max(n_val_steps, 1)
            val_f1 /= max(len(val_paths), 1)

        # s0 baseline on val
        s0_f1 = 0.0
        with torch.no_grad():
            for p in val_paths:
                data = torch.load(p, map_location="cpu", weights_only=False)
                data = _anchor_to_device(data, device)
                s0 = rollout_rung4_species_series(data, phys, bio, device, step="s0")
                t_last = int(data.y.shape[0]) - 1
                with t0_rung2_env():
                    traj = rollout_t0_clot_phi(
                        data, phys, bio, device,
                        gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt",
                        pred_species_series=s0, nucleation=True, nucleation_hops=1,
                    )
                phi_gt = gt_clot_phi_at_time(data, t_last, phys, device).reshape(-1)
                phi_p = traj[t_last]["phi"].reshape(-1)
                m = _clot_metrics(phi_p, phi_gt, torch.ones(phi_gt.numel(), device=device, dtype=torch.bool))
                s0_f1 += float(m["clot_f1"])
            s0_f1 /= max(len(val_paths), 1)

        row = {
            "epoch": ep,
            "channel_set": cs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "s0_f1": s0_f1,
            "delta_vs_s0": val_f1 - s0_f1,
            "sec": round(time.perf_counter() - t0, 2),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"[ep {ep:3d}] loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_f1={val_f1:.3f} s0={s0_f1:.3f} d={row['delta_vs_s0']:+.3f} ({row['sec']}s)",
            flush=True,
        )

        # Promote on val clot F1 only; val_loss can improve while F1 collapses (cascade4).
        f1_improved = val_f1 > best_f1 + 1e-4
        if f1_improved:
            best_f1 = val_f1
            best_loss = val_loss
            stale = 0
            save_m0_checkpoint(
                out, bundle,
                meta={
                    "val_anchor": val_stem,
                    "val_loss": val_loss,
                    "val_f1": val_f1,
                    "s0_f1": s0_f1,
                    "epoch": ep,
                },
            )
        else:
            best_loss = min(best_loss, val_loss)
            stale += 1
        if stale >= int(args.early_stop):
            print(f"[OK] early stop after {stale} epochs without gain", flush=True)
            break

    print(f"[OK] best val_f1={best_f1:.3f} val_loss={best_loss:.5f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
