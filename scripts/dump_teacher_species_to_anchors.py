"""Cache teacher-rolled species onto biochem anchor graphs.

Writes a copy of each anchor ``.pt`` graph with its ``y[:, :, 4:16]`` species channels
replaced by the teacher's predicted species trajectory (log1p ND), keeping kinematics
and GT mu_eff channels intact.
"""

from __future__ import annotations

import os
import sys
import argparse
import threading
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architecture.gnode_biochem import (
    GNODE_Phase3,
    apply_biochem_forward_policy_from_checkpoint_meta,
    resolve_gnode_phase3_ctor_kwargs,
)
from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, build_y_valid_mask, infer_missing_schema
from src.utils.nondim import to_t_nd
from src.utils.paths import get_project_root


def _heartbeat(label: str, interval_s: float, stop: threading.Event) -> None:
    """Periodic progress log while a long-running step is active."""
    tick = 0
    while not stop.wait(interval_s):
        tick += 1
        print(f"[i]  {label} ... {tick * interval_s:.0f}s", flush=True)


def _resolve_rollout_mu_ratio_max(
    bio_cfg: BiochemConfig,
    *,
    cli_value: float | None,
) -> float:
    """COMSOL mu1/mu2 step ceiling for offline rollout (not mu_eff ratio)."""
    if cli_value is not None:
        return max(float(cli_value), 1.0)
    raw = (os.environ.get("BIOCHEM_TEACHER_MU_RATIO_MAX") or "").strip()
    if raw:
        return max(float(raw), 1.0)
    return max(float(getattr(bio_cfg, "mu_ratio_max", 80.0)), 1.0)


def _build_teacher(
    ckpt: dict,
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    mu_ratio_max: float,
) -> GNODE_Phase3:
    state_dict = ckpt.get("model_state_dict") or ckpt
    bio_prior_default = max(0, int(os.environ.get("BIOCHEM_BIO_ENCODER_PRIOR_DIM", "2") or "2"))
    ctor = resolve_gnode_phase3_ctor_kwargs(
        ckpt,
        state_dict,
        bio_encoder_prior_dim_default=bio_prior_default,
        latent_dim_default=256,
        fourier_bands_default=16,
        use_siren_default=True,
        gnode_layers_default=1,
        max_inner_iters_default=10,
    )
    teacher = GNODE_Phase3(
        phys_cfg=phys_cfg,
        in_channels=int(ctor["in_channels"]),
        spatial_channels=int(ctor["spatial_channels"]),
        latent_dim=int(ctor["latent_dim"]),
        max_inner_iters=max(3, int(ctor.get("max_inner_iters", 10))),
        bio_encoder_prior_dim=int(ctor["bio_encoder_prior_dim"]),
        mu_ratio_max=mu_ratio_max,
        mat_crit=float(bio_cfg.viscosity_mat_crit),
        fi_crit=float(bio_cfg.viscosity_fi_crit),
        temp_mat=float(bio_cfg.viscosity_gnode_temp_mat),
        temp_fi=float(bio_cfg.viscosity_gnode_temp_fi),
        num_fourier_freqs=int(ctor["num_fourier_freqs"]),
        use_siren_decoder=bool(ctor["use_siren_decoder"]),
        gnode_layers=int(ctor["gnode_layers"]),
        use_hard_bcs=bool(ctor["use_hard_bcs"]),
    ).to(device)
    teacher.load_state_dict(state_dict, strict=False)
    apply_biochem_forward_policy_from_checkpoint_meta(ckpt, quiet=False)
    teacher.eval()
    print(
        f"[i]  GNODE rollout arch: in={int(ctor['in_channels'])} spatial={int(ctor['spatial_channels'])} "
        f"prior={int(ctor['bio_encoder_prior_dim'])} latent={int(ctor['latent_dim'])} "
        f"siren={int(ctor['use_siren_decoder'])} fourier={int(ctor['num_fourier_freqs'])}",
        flush=True,
    )
    return teacher


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="outputs/biochem/biochem_teacher_last.pth")
    ap.add_argument("--out-dir", default="outputs/biochem/anchors_teacher_species")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--time-stride", type=int, default=6, help="Subsample time axis to speed rollout (default 6).")
    ap.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="If >0, adapt stride per-anchor so subsampled trajectory keeps at least this many timesteps.",
    )
    ap.add_argument("--only", default="", help="Comma-separated anchor stems to process (default: all).")
    ap.add_argument(
        "--src-dir",
        default="",
        help="Read anchors from this directory (default: graphs_biochem_anchors).",
    )
    ap.add_argument(
        "--no-subsample",
        action="store_true",
        help="Keep full time axis on each graph (use with pre-subsampled --src-dir caches).",
    )
    ap.add_argument(
        "--write-kine-macro",
        action="store_true",
        help="Copy teacher-predicted [u,v,p] (y ch 0:3) into output; requires BIOCHEM_GT_KINE_VEL=0.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing cached anchors.")
    ap.add_argument(
        "--mu-ratio-max",
        type=float,
        default=None,
        help="Override teacher mu_ratio_max for rollout (default: env BIOCHEM_TEACHER_MU_RATIO_MAX or bio_cfg).",
    )
    args = ap.parse_args()

    root = get_project_root()
    teacher_path = Path(args.teacher)
    if not teacher_path.is_file():
        teacher_path = root / args.teacher
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    # Default GT [u,v,p] unless caller set BIOCHEM_GT_KINE_VEL=0 (predicted DEQ path).
    if args.write_kine_macro:
        os.environ["BIOCHEM_GT_KINE_VEL"] = "0"
    else:
        os.environ.setdefault("BIOCHEM_GT_KINE_VEL", "1")
    rollout_mu_ratio = _resolve_rollout_mu_ratio_max(bio_cfg, cli_value=args.mu_ratio_max)
    os.environ["BIOCHEM_TEACHER_MU_RATIO_MAX"] = f"{rollout_mu_ratio:g}"
    # Speed knobs for offline rollout.
    os.environ.setdefault("BIOCHEM_ADJOINT_RK4_SUBSTEPS", "1")
    os.environ.setdefault("BIOCHEM_TBPTT_MAX_WINDOW", "6")

    ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
    teacher = _build_teacher(
        ckpt,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        mu_ratio_max=rollout_mu_ratio,
    )
    print(f"[i]  rollout mu_ratio_max={rollout_mu_ratio:g}", flush=True)

    if args.src_dir:
        anchor_dir = Path(args.src_dir)
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    anchors = sorted(p for p in anchor_dir.glob("*.pt") if p.is_file())
    if not anchors:
        raise FileNotFoundError(f"No anchors found in {anchor_dir}")
    only = [s.strip().lower() for s in (args.only or "").split(",") if s.strip()]
    if only:
        anchors = [p for p in anchors if p.stem.lower() in set(only)]

    wrote = 0
    for p in anchors:
        out_path = out_dir / p.name
        if out_path.is_file() and not args.force:
            print(f"[skip] {out_path.name} exists", flush=True)
            continue

        kine_note = " + pred [u,v,p]" if args.write_kine_macro else ""
        print(f"[i]  rolling species{kine_note} for {p.name}", flush=True)
        data = torch.load(p, weights_only=False).to(device)
        data = infer_missing_schema(data, phase_hint="biochem")

        stride = 1 if args.no_subsample else max(int(args.time_stride), 1)
        if hasattr(data, "y") and torch.is_tensor(data.y) and data.y.dim() == 3:
            total_steps = int(data.y.shape[0])
            min_steps = max(int(args.min_steps), 0)
            if min_steps > 1 and total_steps > 1:
                max_stride_for_min = max(1, int((total_steps - 1) // max(min_steps - 1, 1)))
                stride = min(stride, max_stride_for_min)
        if stride > 1 and hasattr(data, "y") and torch.is_tensor(data.y) and data.y.dim() == 3:
            idx = torch.arange(0, int(data.y.shape[0]), stride, device=data.y.device)
            last = int(data.y.shape[0]) - 1
            if int(idx[-1].item()) != last:
                idx = torch.cat([idx, torch.tensor([last], device=idx.device, dtype=idx.dtype)], dim=0)
            data.y = data.y.index_select(0, idx).contiguous()
            if hasattr(data, "y_valid_mask") and torch.is_tensor(data.y_valid_mask):
                try:
                    if data.y_valid_mask.shape[0] >= data.y.shape[0] and data.y_valid_mask.shape[1] == data.y.shape[1]:
                        data.y_valid_mask = data.y_valid_mask.index_select(0, idx).contiguous()
                except Exception:
                    pass
        if stride > 1 and hasattr(data, "t") and getattr(data, "t") is not None and torch.is_tensor(data.t):
            try:
                data.t = data.t.index_select(0, idx).contiguous()
            except Exception:
                pass

        if hasattr(data, "y") and torch.is_tensor(data.y):
            if hasattr(data, "y_valid_mask") and torch.is_tensor(data.y_valid_mask):
                if tuple(data.y_valid_mask.shape) != tuple(data.y.shape):
                    data.y_valid_mask = build_y_valid_mask(
                        data.y, data.y_schema, getattr(data, "mask_wall", None)
                    )
        assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

        # Build evaluation times in ND time units, matching train_biochem_corrector.
        t_si = bio_cfg.resolve_biochem_times(data, device)
        t_ref = float(getattr(bio_cfg, "t_final", 30000.0))
        eval_t = to_t_nd(t_si, t_ref)
        label = f"GNODE rollout {p.name}"
        stop = threading.Event()
        hb = threading.Thread(target=_heartbeat, args=(label, 60.0, stop), daemon=True)
        hb.start()
        try:
            with torch.no_grad():
                pred_series = teacher(
                    data,
                    eval_t,
                    y_true_trajectory=data.y,
                    teacher_forcing_ratio=1.0,
                    start_idx=0,
                    initial_species=None,
                    detach_macro_state=True,
                )
        finally:
            stop.set()
            hb.join(timeout=1.0)
        print(f"[i]  teacher rollout done for {p.name} (T={int(data.y.shape[0])})", flush=True)

        if pred_series.shape[:2] != data.y.shape[:2]:
            raise RuntimeError(f"teacher pred_series shape {tuple(pred_series.shape)} != y {tuple(data.y.shape)} for {p.name}")

        data_out = data.to("cpu")
        pred_cpu = pred_series.to("cpu")
        data_out.y[:, :, 4:16] = pred_cpu[:, :, 4:16]
        if args.write_kine_macro:
            data_out.y[:, :, 0:3] = pred_cpu[:, :, 0:3]
        if hasattr(data_out, "y_valid_mask") and torch.is_tensor(data_out.y_valid_mask):
            if tuple(data_out.y_valid_mask.shape) != tuple(data_out.y.shape):
                data_out.y_valid_mask = build_y_valid_mask(
                    data_out.y, data_out.y_schema, getattr(data_out, "mask_wall", None)
                )

        torch.save(data_out, out_path)
        wrote += 1

    print(f"[OK]  wrote {wrote} teacher-species anchors -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

