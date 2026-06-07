"""8-hour autonomous clot deploy runner: Lane A dump track + forecast ladder track."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@dataclass
class LegResult:
    name: str
    track: str
    leg_dir: Path
    ok: bool
    rank: float = -1.0
    shape: float = 0.0
    f1: float = 0.0
    shape_pred: float = 1.0
    wall_s: float = 0.0
    note: str = ""


def _run(cmd: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[RUN] {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[EXIT] {proc.returncode}\n")
    return int(proc.returncode)


def _best_from_log(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    best: dict | None = None
    best_rank = -1.0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            va = row.get("val") or {}
            shape = float(va.get("clot_shape", 0.0))
            f1 = float(va.get("clot_f1", 0.0))
            rank = shape * 0.65 + f1 * 0.35
            score = float(row.get("val_score", -99.0))
            if score >= 0:
                rank = max(rank, score)
            if best is None or rank > best_rank:
                best = row
                best_rank = rank
    if best is not None:
        best["_rank"] = best_rank
    return best


def _summarize_leg(track: str, name: str, leg_dir: Path, wall_s: float, ok: bool, anchor_dir: str = "") -> LegResult:
    log_path = leg_dir / "clot_phi_train_log.jsonl"
    ckpt = leg_dir / "clot_phi_best.pth"
    if ok and not ckpt.is_file() and log_path.is_file():
        _run(
            [sys.executable, "scripts/recover_clot_phi_best_from_log.py", "--leg-dir", str(leg_dir.relative_to(_REPO))],
            cwd=_REPO,
            log=leg_dir / "autonomy_recover.log",
        )
    if ok and ckpt.is_file() and not (leg_dir / "multi_anchor.jsonl").is_file():
        eval_cmd = [
            sys.executable,
            "scripts/eval_clot_phi_multi_anchor.py",
            "--checkpoint",
            str(ckpt.relative_to(_REPO)),
            "--out",
            str((leg_dir / "multi_anchor.jsonl").relative_to(_REPO)),
        ]
        if anchor_dir:
            eval_cmd.extend(["--anchor-dir", anchor_dir.replace("\\", "/")])
        _run(eval_cmd, cwd=_REPO, log=leg_dir / "autonomy_eval.log")
    best = _best_from_log(log_path)
    res = LegResult(name=name, track=track, leg_dir=leg_dir, ok=ok, wall_s=wall_s)
    if best:
        va = best.get("val") or {}
        res.rank = float(best.get("_rank", -1.0))
        res.shape = float(va.get("clot_shape", 0.0))
        res.f1 = float(va.get("clot_f1", 0.0))
        res.shape_pred = float(va.get("clot_shape_pred_frac", 1.0))
    return res


def _train_forecast_leg(
    *,
    sweep_root: Path,
    leg: str,
    env: dict[str, str],
    epochs: int,
    init_ckpt: str = "",
    anchor_dir: str = "",
) -> LegResult:
    leg_dir = sweep_root / leg
    leg_dir.mkdir(parents=True, exist_ok=True)
    log = leg_dir / "autonomy_train.log"
    env_full = dict(env)
    env_full["CLOT_PHI_SWEEP_DIR"] = str(sweep_root.relative_to(_REPO)).replace("\\", "/")
    env_full["CLOT_PHI_SWEEP_LEG"] = leg
    env_full["CLOT_PHI_EPOCHS"] = str(epochs)
    if anchor_dir:
        env_full["CLOT_PHI_ANCHOR_DIR"] = anchor_dir
    if init_ckpt:
        env_full["CLOT_PHI_INIT_CHECKPOINT"] = init_ckpt
    else:
        env_full.pop("CLOT_PHI_INIT_CHECKPOINT", None)
    # Fresh leg dir ckpt/log unless init finetune
    if not init_ckpt:
        for p in (leg_dir / "clot_phi_best.pth", leg_dir / "clot_phi_train_log.jsonl"):
            if p.is_file():
                p.unlink()

    cmd = [sys.executable, "-m", "src.training.train_clot_phi_simple"]
    t0 = time.time()
    proc_env = os.environ.copy()
    proc_env.update({k: str(v) for k, v in env_full.items()})
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n[ENV] {json.dumps(env_full)}\n")
        proc = subprocess.run(cmd, cwd=str(_REPO), env=proc_env, stdout=f, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0
    ad = env_full.get("CLOT_PHI_ANCHOR_DIR", "")
    return _summarize_leg("forecast" if not ad else "lane_a", leg, leg_dir, time.time() - t0, ok, anchor_dir=ad)


def _lane_a_base_env() -> dict[str, str]:
    return {
        "CLOT_FORECAST_MODE": "one_step",
        "CLOT_FORECAST_PAIR_STRIDE": "1",
        "CLOT_FORECAST_INPUT_MU": "1",
        "CLOT_PHI_FIXED_MU_FROM_PHI": "1",
        "CLOT_PHI_HYBRID": "0",
        "CLOT_PHI_ROLLOUT": "0",
        "CLOT_PHI_MU_SOLID_SI": "0.10",
        "CLOT_PHI_MU_LOG_LAMBDA": "0",
        "CLOT_PHI_SHAPE_USE_T_OUT": "1",
        "BIOCHEM_MLP_NEIGHBOR_SEED": "pred_clot",
        "BIOCHEM_MLP_NEIGHBOR_REQUIRE_PHI": "0",
        "BIOCHEM_MLP_MU_MAP_PHI_THRESH": "0.5",
        "BIOCHEM_MLP_DEPLOY_DGAMMA_SLICE": "0",
        "CLOT_PHI_MODEL": "mlp",
        "CLOT_PHI_HIDDEN": "32",
        "CLOT_PHI_MLP_DEPTH": "2",
        "CLOT_PHI_DROPOUT": "0.15",
        "CLOT_PHI_LR": "1e-3",
        "CLOT_PHI_WEIGHT_DECAY": "1e-4",
        "CLOT_PHI_DICE_LAMBDA": "0.2",
        "CLOT_PHI_MINIMAL_FEATURES": "1",
        "CLOT_PHI_SPECIES_FEATURES": "0",
        "CLOT_PHI_JOINT_BIO": "0",
        "CLOT_PHI_PHYSICS_BLEND": "0",
        "CLOT_PHI_BALANCED": "1",
        "CLOT_PHI_TIME_STRIDE": "1",
        "CLOT_PHI_TIME_STRIDE_AUTO": "0",
        "CLOT_PHI_DGAMMA_SLICE": "0",
        "CLOT_PHI_DGAMMA_FEATURE_TIME": "current",
        "CLOT_PHI_VEL_SOURCE": "gt",
        "CLOT_PHI_SOFT_LABELS": "1",
        "CLOT_PHI_VAL_ANCHOR": "patient007",
        "CLOT_PHI_ANCHOR_DIR": "outputs/biochem/gnode10_sweep/anchors_gnode12_predkine_uvp",
    }


def _forecast_base_env() -> dict[str, str]:
    env = _lane_a_base_env()
    env.pop("CLOT_PHI_ANCHOR_DIR", None)
    return env


def run_lane_a_track(budget_s: float, out_root: Path) -> list[LegResult]:
    results: list[LegResult] = []
    t_start = time.time()
    sweep = out_root / "lane_a"
    sweep.mkdir(parents=True, exist_ok=True)
    base = _lane_a_base_env()

    plan: list[dict] = [
        {"leg": "a01_target_warm", "mask": "target", "ep": 18, "aux": "0.3", "bulk": "0", "init": ""},
        {"leg": "a02_deploy_v1", "mask": "deploy_pred", "ep": 28, "aux": "0.6", "bulk": "0.18", "init": "a01_target_warm"},
        {"leg": "a03_deploy_bulk", "mask": "deploy_pred", "ep": 22, "aux": "0.7", "bulk": "0.28", "init": "a02_deploy_v1"},
        {"leg": "a04_target_long", "mask": "target", "ep": 24, "aux": "0.4", "bulk": "0.05", "init": ""},
        {"leg": "a05_deploy_v2", "mask": "deploy_pred", "ep": 30, "aux": "0.65", "bulk": "0.22", "init": "a04_target_long"},
        {"leg": "a06_wide_deploy", "mask": "deploy_pred", "ep": 24, "aux": "0.55", "bulk": "0.20", "init": "a04_target_long", "hidden": "48", "depth": "3"},
    ]

    init_paths: dict[str, str] = {}
    for step in plan:
        if time.time() - t_start > budget_s:
            print(f"[WARN] Lane A budget exhausted before {step['leg']}", flush=True)
            break
        leg = step["leg"]
        env = dict(base)
        env["CLOT_FORECAST_MASK"] = step["mask"]
        if step.get("hidden"):
            env["CLOT_PHI_HIDDEN"] = step["hidden"]
        if step.get("depth"):
            env["CLOT_PHI_MLP_DEPTH"] = step["depth"]
        if float(step.get("aux", "0")) > 0:
            env["CLOT_PHI_MESH_AUX_LAMBDA"] = step["aux"]
        if float(step.get("bulk", "0")) > 0:
            env["CLOT_PHI_MESH_BULK_LAMBDA"] = step["bulk"]
        init = ""
        if step.get("init"):
            ref = init_paths.get(step["init"], "")
            if ref and Path(ref).is_file():
                init = ref
        res = _train_forecast_leg(
            sweep_root=sweep,
            leg=leg,
            env=env,
            epochs=int(step["ep"]),
            init_ckpt=init,
        )
        results.append(res)
        ckpt = sweep / leg / "clot_phi_best.pth"
        if ckpt.is_file():
            init_paths[leg] = str(ckpt.relative_to(_REPO)).replace("\\", "/")
        print(
            f"[OK]  LaneA {leg} rank={res.rank:.3f} shape={res.shape:.3f} f1={res.f1:.3f} "
            f"shape_pred={res.shape_pred:.3f} wall={res.wall_s/60:.1f}m",
            flush=True,
        )
        # Adaptive: if still over-clotting mesh, queue extra bulk leg once
        if res.shape_pred > 0.60 and "bulk_retry" not in init_paths and time.time() - t_start < budget_s - 900:
            extra = f"{leg}_bulk_retry"
            env2 = dict(env)
            env2["CLOT_FORECAST_MASK"] = "deploy_pred"
            env2["CLOT_PHI_MESH_AUX_LAMBDA"] = "0.85"
            env2["CLOT_PHI_MESH_BULK_LAMBDA"] = "0.35"
            init_paths["bulk_retry"] = "1"
            r2 = _train_forecast_leg(
                sweep_root=sweep,
                leg=extra,
                env=env2,
                epochs=18,
                init_ckpt=init_paths.get(leg, init),
            )
            results.append(r2)
            print(f"[OK]  LaneA adaptive {extra} shape={r2.shape:.3f}", flush=True)

    return results


def run_forecast_track(budget_s: float, out_root: Path) -> list[LegResult]:
    results: list[LegResult] = []
    t_start = time.time()
    sweep = out_root / "forecast"
    sweep.mkdir(parents=True, exist_ok=True)
    base = _forecast_base_env()

    # Recover prior r2a+ if present
    r2a_dir = _REPO / "outputs/biochem/clot_forecast_ladder/r2a_plus_one_step_phi"
    if r2a_dir.is_file() is False and (r2a_dir / "clot_phi_train_log.jsonl").is_file():
        _run(
            [sys.executable, "scripts/recover_clot_phi_best_from_log.py", "--leg-dir", str(r2a_dir.relative_to(_REPO))],
            cwd=_REPO,
            log=sweep / "recover_r2a_plus.log",
        )

    plan: list[dict] = [
        {"leg": "f01_target_25", "mask": "target", "ep": 25, "aux": "0.35", "bulk": "0", "init": ""},
        {"leg": "f02_deploy_30", "mask": "deploy_pred", "ep": 30, "aux": "0.65", "bulk": "0.20", "init": "f01_target_25"},
        {"leg": "f03_target_mesh", "mask": "target", "ep": 20, "aux": "0.45", "bulk": "0.08", "init": ""},
        {"leg": "f04_deploy_bulk", "mask": "deploy_pred", "ep": 28, "aux": "0.75", "bulk": "0.30", "init": "f03_target_mesh"},
        {"leg": "f05_wide_target", "mask": "target", "ep": 22, "aux": "0.4", "bulk": "0.05", "init": "", "hidden": "48", "depth": "3"},
        {"leg": "f06_wide_deploy", "mask": "deploy_pred", "ep": 26, "aux": "0.6", "bulk": "0.22", "init": "f05_wide_target", "hidden": "48", "depth": "3"},
        {"leg": "f07_r2a_refresh", "mask": "deploy_pred", "ep": 35, "aux": "0.6", "bulk": "0.18", "init": ""},
    ]

    init_paths: dict[str, str] = {}
    r2a_ckpt = r2a_dir / "clot_phi_best.pth"
    if r2a_ckpt.is_file():
        init_paths["r2a_plus"] = str(r2a_ckpt.relative_to(_REPO)).replace("\\", "/")

    for step in plan:
        if time.time() - t_start > budget_s:
            print(f"[WARN] Forecast budget exhausted before {step['leg']}", flush=True)
            break
        leg = step["leg"]
        env = dict(base)
        env["CLOT_FORECAST_MASK"] = step["mask"]
        if step.get("hidden"):
            env["CLOT_PHI_HIDDEN"] = step["hidden"]
        if step.get("depth"):
            env["CLOT_PHI_MLP_DEPTH"] = step["depth"]
        if float(step.get("aux", "0")) > 0:
            env["CLOT_PHI_MESH_AUX_LAMBDA"] = step["aux"]
        if float(step.get("bulk", "0")) > 0:
            env["CLOT_PHI_MESH_BULK_LAMBDA"] = step["bulk"]
        init = ""
        if step.get("init"):
            ref = init_paths.get(step["init"], "")
            if ref and Path(_REPO / ref).is_file():
                init = ref
        if leg == "f07_r2a_refresh" and init_paths.get("r2a_plus"):
            init = init_paths["r2a_plus"]
        res = _train_forecast_leg(
            sweep_root=sweep,
            leg=leg,
            env=env,
            epochs=int(step["ep"]),
            init_ckpt=init,
        )
        results.append(res)
        ckpt = sweep / leg / "clot_phi_best.pth"
        if ckpt.is_file():
            init_paths[leg] = str(ckpt.relative_to(_REPO)).replace("\\", "/")
        print(
            f"[OK]  Forecast {leg} rank={res.rank:.3f} shape={res.shape:.3f} f1={res.f1:.3f} "
            f"shape_pred={res.shape_pred:.3f} wall={res.wall_s/60:.1f}m",
            flush=True,
        )

    # Summarize forecast sweep dir copy
    _run(
        [sys.executable, "scripts/summarize_clot_forecast_deploy_sweep.py", "--sweep-dir", str(sweep.relative_to(_REPO))],
        cwd=_REPO,
        log=sweep / "summarize.log",
    )
    return results


def _write_report(out_root: Path, lane: list[LegResult], forecast: list[LegResult]) -> None:
    all_rows = lane + forecast
    all_rows.sort(key=lambda r: r.rank, reverse=True)
    report = out_root / "autonomy_report.jsonl"
    with report.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(
                json.dumps(
                    {
                        "track": r.track,
                        "leg": r.name,
                        "ok": r.ok,
                        "rank": r.rank,
                        "shape": r.shape,
                        "f1": r.f1,
                        "shape_pred_frac": r.shape_pred,
                        "wall_min": round(r.wall_s / 60.0, 2),
                        "leg_dir": str(r.leg_dir.relative_to(_REPO)).replace("\\", "/"),
                    }
                )
                + "\n"
            )
    print(f"[save] autonomy report -> {report}", flush=True)
    if all_rows:
        best = all_rows[0]
        print(
            f"[OK]  BEST {best.track}/{best.name} rank={best.rank:.3f} shape={best.shape:.3f} f1={best.f1:.3f}",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--lane-hours", type=float, default=4.0)
    ap.add_argument("--out", default="outputs/biochem/autonomy_clot_8h")
    ap.add_argument("--lane-only", action="store_true")
    ap.add_argument("--forecast-only", action="store_true")
    args = ap.parse_args()

    out_root = (_REPO / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    lane_budget = args.lane_hours * 3600.0
    forecast_budget = max(0.0, args.hours * 3600.0 - lane_budget) if not args.lane_only else 0.0
    if args.forecast_only:
        lane_budget = 0.0
        forecast_budget = args.hours * 3600.0

    print(f"[NEW] autonomy clot run lane={lane_budget/3600:.1f}h forecast={forecast_budget/3600:.1f}h -> {out_root}", flush=True)

    lane_res: list[LegResult] = []
    forecast_res: list[LegResult] = []
    if lane_budget > 0:
        print("[NEW] === Track 1: Lane A deploy (pred-kine dump) ===", flush=True)
        lane_res = run_lane_a_track(lane_budget, out_root)
    if forecast_budget > 0:
        print("[NEW] === Track 2: Clot forecast ladder ===", flush=True)
        forecast_res = run_forecast_track(forecast_budget, out_root)

    _write_report(out_root, lane_res, forecast_res)


if __name__ == "__main__":
    main()
