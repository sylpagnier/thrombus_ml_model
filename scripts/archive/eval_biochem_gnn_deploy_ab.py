"""A/B eval: legacy GT-leak rollout vs deploy-faithful biochem_gnn (frozen / coupled).

Compares clot F1@t53 on biochem anchors against the locked baseline in the manifest.

Modes:
  legacy_oracle   - pre-fix leaks (GT IC, GT species pin, GT vel, GT clot flow)
  deploy_frozen   - deploy-faithful + pred kine held (default deploy)
  deploy_coupled  - deploy-faithful + mu -> GINO-DEQ refresh each step

Usage::

    python scripts/eval_biochem_gnn_deploy_ab.py
    python scripts/eval_biochem_gnn_deploy_ab.py --modes deploy_frozen,deploy_coupled
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn import BiochemGNN, FlowMode, apply_deploy_env, load_manifest, reference_manifest_path  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.clot_coupled_rollout import reset_coupled_uv_cache  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import (  # noqa: E402
    inc40_baseline_rule_config,
    reset_temporal_kinematics_cache,
    rollout_inc40_phi_trajectory,
)
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6, discover_biochem_anchors  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.evaluation.clot_relaxed_metrics import (
    clot_score_from_deploy_dict,
    compute_clot_relaxed_metrics,
)
from src.evaluation.clot_relaxed_metrics import legacy_clot_f1_metrics as _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ALL_MODES = ("legacy_oracle", "deploy_frozen", "deploy_coupled")


@contextmanager
def rollout_env(
    manifest: dict,
    *,
    anchor: str,
    mode: str,
) -> Iterator[FlowMode]:
    """Set env for one rollout mode; restore prior env on exit."""
    saved = dict(os.environ)
    reset_species_rollout_flow_cache()
    reset_coupled_uv_cache()
    reset_temporal_kinematics_cache()
    try:
        if mode == "legacy_oracle":
            apply_deploy_env(
                manifest,
                anchor=anchor,
                overrides={
                    "SPECIES_ROLLOUT_DEPLOY_FAITHFUL": "0",
                    "SPECIES_ROLLOUT_IC_SOURCE": "gt",
                    "SPECIES_ROLLOUT_PIN_OTHER": "gt",
                    "SPECIES_ROLLOUT_VEL_SOURCE": "gt",
                    "T0_R4_FLOW_SOURCE": "gt",
                },
            )
            yield FlowMode.FROZEN_KINE
        elif mode == "deploy_frozen":
            apply_deploy_env(
                manifest,
                anchor=anchor,
                overrides={"T0_R4_FLOW_SOURCE": "kinematics"},
            )
            yield FlowMode.FROZEN_KINE
        elif mode == "deploy_coupled":
            apply_deploy_env(manifest, anchor=anchor)
            yield FlowMode.COUPLED
        else:
            raise ValueError(f"unknown mode: {mode}")
    finally:
        os.environ.clear()
        os.environ.update(saved)
        reset_species_rollout_flow_cache()
        reset_coupled_uv_cache()
        reset_temporal_kinematics_cache()


def _clot_f1_at(
    phi_by_time: dict[int, torch.Tensor],
    data,
    t: int,
    phys: PhysicsConfig,
    device: torch.device,
) -> dict[str, float]:
    mask = torch.ones(int(data.num_nodes), device=device, dtype=torch.bool)
    phi_gt = gt_clot_phi_at_time(data, int(t), phys, device)
    phi_pred = phi_by_time[int(t)]
    edge_index = data.edge_index.to(device=device)
    relaxed = compute_clot_relaxed_metrics(
        phi_pred.reshape(-1), phi_gt.reshape(-1), edge_index
    )
    legacy = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)
    out = {**legacy, **relaxed}
    out["clot_score"] = clot_score_from_deploy_dict({
        "deploy_clot_f1": legacy["clot_f1"],
        "deploy_clot_guiding": relaxed["clot_guiding"],
        "deploy_clot_relaxed_f05": relaxed["clot_relaxed_f05"],
        "deploy_clot_dil_iou": relaxed["clot_dilation_iou"],
    })
    return out


def _clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _eval_anchor_mode(
    anchor: str,
    mode: str,
    *,
    manifest: dict,
    device: torch.device,
    times: list[int],
) -> dict:
    root = get_project_root()
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n_steps = int(data.y.shape[0])
    eval_times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})

    with rollout_env(manifest, anchor=anchor, mode=mode) as flow_mode:
        model = BiochemGNN.from_manifest(
            manifest,
            anchor=anchor,
            device=device,
            flow_mode=flow_mode,
        )
        try:
            rollout = model.rollout(data)
        except torch.cuda.OutOfMemoryError as exc:
            _clear_cuda()
            return {
                "anchor": anchor,
                "mode": mode,
                "flow_mode": flow_mode.value,
                "error": f"cuda_oom: {exc}",
                "clot_f1_main": float("nan"),
                "time_main": int(times[-1]),
            }
        phi = rollout.phi_by_time
    _clear_cuda()

    clot_rows = {
        int(t): _clot_f1_at(phi, data, int(t), phys, device)
        for t in eval_times
    }
    t_main = eval_times[-1]
    main = clot_rows[t_main]
    return {
        "anchor": anchor,
        "mode": mode,
        "flow_mode": flow_mode.value,
        "clot_f1": {str(t): float(clot_rows[t]["clot_f1"]) for t in eval_times},
        "clot_guiding": {str(t): float(clot_rows[t]["clot_guiding"]) for t in eval_times},
        "clot_relaxed_f05": {str(t): float(clot_rows[t]["clot_relaxed_f05"]) for t in eval_times},
        "clot_dilation_iou": {str(t): float(clot_rows[t]["clot_dilation_iou"]) for t in eval_times},
        "clot_f1_main": float(main["clot_f1"]),
        "clot_guiding_main": float(main["clot_guiding"]),
        "clot_relaxed_f05_main": float(main["clot_relaxed_f05"]),
        "clot_dilation_iou_main": float(main["clot_dilation_iou"]),
        "clot_score_main": float(main.get("clot_score", main["clot_guiding"])),
        "clot_prec_main": float(main["clot_prec"]),
        "clot_rec_main": float(main["clot_rec"]),
        "clot_relaxed_prec_main": float(main["clot_relaxed_prec"]),
        "clot_relaxed_rec_main": float(main["clot_relaxed_rec"]),
        "time_main": int(t_main),
    }


def _eval_inc40(
    anchor: str,
    *,
    device: torch.device,
    times: list[int],
    vel_source: str,
) -> dict:
    root = get_project_root()
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt",
        map_location=device,
        weights_only=False,
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n_steps = int(data.y.shape[0])
    eval_times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})
    os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = vel_source
    reset_temporal_kinematics_cache()
    phi = rollout_inc40_phi_trajectory(data, phys, bio, device, vel_source=vel_source)
    clot_rows = {
        int(t): _clot_f1_at(phi, data, int(t), phys, device)
        for t in eval_times
    }
    t_main = eval_times[-1]
    return {
        "anchor": anchor,
        "mode": "inc40_rules",
        "rule": inc40_baseline_rule_config().name,
        "vel_source": vel_source,
        "clot_f1_main": float(clot_rows[t_main]["clot_f1"]),
        "time_main": int(t_main),
    }


def _load_baseline(manifest: dict) -> dict:
    ev = dict(manifest.get("eval") or {})
    root = get_project_root()
    for name in ("eval_summary.json",):
        p = root / "outputs/biochem/species_gnn_deploy_baseline" / name
        if p.is_file():
            ev.update(json.loads(p.read_text(encoding="utf-8")))
    return ev


def _mean_holdout(rows: list[dict], *, val_anchor: str) -> float | None:
    hold = [r for r in rows if r["anchor"] != val_anchor]
    if not hold:
        return None
    return sum(r["clot_f1_main"] for r in hold) / len(hold)


def _print_table(rows: list[dict], baseline: dict, *, val_anchor: str) -> None:
    modes = sorted({r["mode"] for r in rows})
    anchors = sorted({r["anchor"] for r in rows})
    print("\n[i] clot F1@t53 by anchor (main eval time)", flush=True)
    header = f"{'anchor':<12}" + "".join(f"{m:>16}" for m in modes) + f"{'baseline':>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    per_anchor_base = {
        str(r.get("anchor")): r.get("clot_f1_t53")
        for r in (baseline.get("per_anchor") or [])
        if r.get("flow") == "gt"
    }
    for anc in anchors:
        line = f"{anc:<12}"
        for mode in modes:
            hit = next((r for r in rows if r["anchor"] == anc and r["mode"] == mode), None)
            val = hit["clot_f1_main"] if hit else float("nan")
            line += f"{val:>16.3f}"
        b = per_anchor_base.get(anc)
        line += f"{(b if b is not None else float('nan')):>12.3f}"
        print(line, flush=True)

    print("\n[i] summary", flush=True)
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        hold_mean = _mean_holdout(subset, val_anchor=val_anchor)
        p007 = next((r["clot_f1_main"] for r in subset if r["anchor"] == val_anchor), None)
        print(
            f"  {mode:<16} p007={p007:.3f}  holdout_mean={hold_mean:.3f}"
            if p007 is not None and hold_mean is not None
            else f"  {mode}",
            flush=True,
        )
    p007_base = baseline.get("patient007_clot_f1_t53")
    hold_base = baseline.get("mean_holdout_clot_f1_t53_gt")
    if p007_base is not None:
        print(f"  {'locked_baseline':<16} p007={float(p007_base):.3f}  holdout_mean={float(hold_base):.3f}", flush=True)

    frozen = [r for r in rows if r["mode"] == "deploy_frozen"]
    legacy = [r for r in rows if r["mode"] == "legacy_oracle"]
    if frozen and legacy:
        deltas = []
        for lf in frozen:
            lo = next((r for r in legacy if r["anchor"] == lf["anchor"]), None)
            if lo:
                deltas.append(lf["clot_f1_main"] - lo["clot_f1_main"])
        if deltas:
            print(
                f"\n[i] deploy_frozen - legacy_oracle: "
                f"mean={sum(deltas)/len(deltas):+.3f}  "
                f"min={min(deltas):+.3f}  max={max(deltas):+.3f}",
                flush=True,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="biochem_gnn deploy-faithful A/B vs locked baseline")
    default_anchors = ",".join(discover_biochem_anchors(get_project_root()))
    ap.add_argument("--anchors", default=default_anchors)
    ap.add_argument("--times", default="27,53")
    ap.add_argument("--modes", default=",".join(ALL_MODES))
    ap.add_argument("--manifest", default="")
    ap.add_argument("--inc40", action="store_true", help="Also eval inc40 rules baseline")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/deploy_ab_eval.json")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest_path = args.manifest.strip() or str(reference_manifest_path())
    manifest = load_manifest(manifest_path)
    baseline = _load_baseline(manifest)
    val_anchor = str(manifest.get("train_val_anchor", "patient007"))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    print(f"[i] manifest={manifest_path}", flush=True)
    print(f"[i] modes={modes} times={times}", flush=True)
    t0 = time.perf_counter()
    rows: list[dict] = []
    for mode in modes:
        print(f"\n[i] mode={mode}", flush=True)
        for anc in anchors:
            print(f"  eval {anc} ...", flush=True)
            row = _eval_anchor_mode(anc, mode, manifest=manifest, device=device, times=times)
            rows.append(row)
            if row.get("error"):
                print(f"    [WARN] {row['error']}", flush=True)
            else:
                print(
                    f"    t{row['time_main']} "
                    f"g={row.get('clot_guiding_main', row['clot_f1_main']):.3f} "
                    f"f05={row.get('clot_relaxed_f05_main', 0):.3f} "
                    f"diou={row.get('clot_dilation_iou_main', 0):.3f} "
                    f"f1={row['clot_f1_main']:.3f}",
                    flush=True,
                )
            _clear_cuda()

    inc40_rows: list[dict] = []
    if args.inc40:
        print("\n[i] inc40 rules (pred kine)", flush=True)
        for anc in anchors:
            row = _eval_inc40(anc, device=device, times=times, vel_source="kinematics")
            inc40_rows.append(row)
            print(
                f"  {anc} t{row['time_main']} "
                f"g={row.get('clot_guiding_main', row['clot_f1_main']):.3f} "
                f"f05={row.get('clot_relaxed_f05_main', 0):.3f} "
                f"f1={row['clot_f1_main']:.3f}",
                flush=True,
            )

    _print_table(rows, baseline, val_anchor=val_anchor)

    payload = {
        "manifest": manifest_path,
        "baseline_eval": baseline,
        "modes": modes,
        "times": times,
        "anchors": anchors,
        "rows": rows,
        "inc40": inc40_rows,
        "elapsed_s": time.perf_counter() - t0,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out} ({payload['elapsed_s']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
