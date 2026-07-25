"""Run geometry-sensitivity research sweeps against the locked canonical model.

Usage:
  python scripts/run_research_sweep.py --sweep 01_stenosis_strength
  python scripts/run_research_sweep.py --all
  python scripts/run_research_sweep.py --list

Resolves ``model: locked_canonical`` at run time via CustomerDeployPipeline
defaults (locked/species_gnn_best.pth + DEFAULT_MAT_LEG). Does not bake a
checkpoint hash into the registry JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow ``python scripts/run_research_sweep.py`` from repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation.research_parameters import (  # noqa: E402
    research_parameters_from_trajectory,
    write_scientific_csv,
)
from src.evaluation.research_sweep_geometry import (  # noqa: E402
    default_mesh_cache_dir,
    load_or_build_research_graph,
)
from src.inference.customer_pipeline import (  # noqa: E402
    DEFAULT_MAT_LEG,
    DEFAULT_WALL_CKPT,
    CustomerDeployPipeline,
)
from src.utils.paths import get_project_root  # noqa: E402

SWEEPS_DIR = Path("configs/research_sweeps")


def _abs(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def list_sweep_configs() -> list[Path]:
    d = _abs(SWEEPS_DIR)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.json"))


def resolve_sweep_path(name: str) -> Path:
    raw = str(name).strip()
    p = Path(raw)
    if p.is_file():
        return p.resolve()
    cand = _abs(SWEEPS_DIR) / raw
    if cand.is_file():
        return cand
    if not raw.endswith(".json"):
        cand2 = _abs(SWEEPS_DIR) / f"{raw}.json"
        if cand2.is_file():
            return cand2
    raise FileNotFoundError(f"Sweep config not found: {name!r} (looked under {SWEEPS_DIR})")


def load_sweep_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Sweep config must be a JSON object: {path}")
    if cfg.get("model") not in (None, "locked_canonical"):
        raise ValueError(
            f"Unsupported model={cfg.get('model')!r}; only locked_canonical is supported"
        )
    cfg["model"] = "locked_canonical"
    if "arms" not in cfg or not cfg["arms"]:
        raise ValueError(f"Sweep config has no arms: {path}")
    return cfg


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_one_arm(
    *,
    pipeline: CustomerDeployPipeline,
    arm: dict[str, Any],
    control: dict[str, Any],
    out_dir: Path,
    cache_dir: Path,
    force_rebuild: bool,
) -> dict[str, Any]:
    name = str(arm.get("name", "arm"))
    print(f"[i] Arm {name}: building / loading geometry...", flush=True)
    t0 = time.perf_counter()
    data, geom_spec, mesh_pt = load_or_build_research_graph(
        arm,
        control,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
    )
    print(
        f"[OK] Geometry ready ({time.perf_counter() - t0:.1f}s) "
        f"nodes={int(data.x.shape[0])} cache={mesh_pt.name}",
        flush=True,
    )

    include_velocity = bool(
        arm.get("include_velocity", control.get("include_velocity", False))
    )
    t_final_s = float(geom_spec.get("t_final_s", control.get("t_final_s", 30000.0)))

    env_overrides: dict[str, str] = {}
    for src in (control.get("env_overrides"), arm.get("env_overrides")):
        if isinstance(src, dict):
            env_overrides.update({str(k): str(v) for k, v in src.items()})

    print(f"[i] Arm {name}: rolling out locked canonical deploy...", flush=True)
    if env_overrides:
        print(f"[i] Arm env overrides: {env_overrides}", flush=True)
    t1 = time.perf_counter()
    traj = pipeline.run(
        data,
        t_final_s=t_final_s,
        include_velocity=include_velocity,
        extra_env=env_overrides or None,
        progress=lambda msg: print(msg, flush=True),
    )
    print(f"[OK] Rollout done in {time.perf_counter() - t1:.1f}s", flush=True)

    pack = research_parameters_from_trajectory(traj)
    arm_out = {
        "name": name,
        "axis_value": arm.get("axis_value"),
        "labels": arm.get("labels") or {},
        "geometry_spec": geom_spec,
        "env_overrides": env_overrides,
        "mesh_cache": str(mesh_pt.as_posix()),
        "model": {
            "resolver": "locked_canonical",
            "wall_ckpt": str(pipeline.wall_ckpt.as_posix()),
            "mat_leg": pipeline.mat_leg,
            "offwall_ckpt": (
                str(pipeline.offwall_ckpt.as_posix()) if pipeline.offwall_ckpt else None
            ),
        },
        "rollout": {
            "elapsed_s": float(traj.elapsed_s),
            "n_steps": int(traj.n_steps),
            "include_velocity": include_velocity,
            "t_final_s": t_final_s,
        },
        "research_parameters": pack,
    }

    arm_json = out_dir / f"arm_{name}.json"
    arm_csv = out_dir / f"arm_{name}.csv"
    arm_json.write_text(
        json.dumps(_json_safe(arm_out), indent=2) + "\n", encoding="utf-8"
    )
    write_scientific_csv(arm_csv, pack["timeseries"])
    print(f"[save] {arm_json}", flush=True)
    return arm_out


def run_sweep(
    cfg: dict[str, Any],
    *,
    pipeline: CustomerDeployPipeline,
    force_rebuild: bool = False,
    arm_filter: str | None = None,
) -> dict[str, Any]:
    out_dir = _abs(cfg.get("output_dir") or f"outputs/research_sweeps/{cfg['id']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = default_mesh_cache_dir()
    control = dict(cfg.get("control") or {})

    arms = list(cfg["arms"])
    if arm_filter:
        arms = [a for a in arms if str(a.get("name")) == arm_filter]
        if not arms:
            raise ValueError(f"No arm named {arm_filter!r} in sweep {cfg.get('id')}")

    print(f"[i] Sweep {cfg.get('id')}: {len(arms)} arm(s) -> {out_dir}", flush=True)
    print(f"[i] Model: locked_canonical  ckpt={pipeline.wall_ckpt}", flush=True)
    print(f"[i] mat_leg={pipeline.mat_leg}", flush=True)

    arm_results: list[dict[str, Any]] = []
    for arm in arms:
        arm_results.append(
            run_one_arm(
                pipeline=pipeline,
                arm=arm,
                control=control,
                out_dir=out_dir,
                cache_dir=cache_dir,
                force_rebuild=force_rebuild,
            )
        )

    summary_rows = []
    for ar in arm_results:
        row = {
            "name": ar["name"],
            "axis_value": ar.get("axis_value"),
            **(ar.get("labels") or {}),
            **(ar.get("research_parameters", {}).get("summary") or {}),
        }
        summary_rows.append(row)

    summary = {
        "id": cfg.get("id"),
        "title": cfg.get("title"),
        "axis": cfg.get("axis"),
        "model": "locked_canonical",
        "wall_ckpt": str(pipeline.wall_ckpt.as_posix()),
        "mat_leg": pipeline.mat_leg,
        "output_dir": str(out_dir.as_posix()),
        "n_arms": len(arm_results),
        "arms": summary_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OK] Sweep complete -> {summary_path}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Geometry-sensitivity research sweeps (locked canonical model)"
    )
    ap.add_argument("--sweep", type=str, default="", help="Sweep id or path under configs/research_sweeps/")
    ap.add_argument("--all", action="store_true", help="Run all configs in configs/research_sweeps/")
    ap.add_argument("--list", action="store_true", help="List available sweep configs and exit")
    ap.add_argument("--arm", type=str, default="", help="Optional single arm name filter")
    ap.add_argument("--force-rebuild-mesh", action="store_true", help="Ignore mesh cache")
    ap.add_argument("--cpu", action="store_true", help="Allow CPU (slow; CUDA recommended)")
    ap.add_argument(
        "--wall-ckpt",
        type=str,
        default="",
        help="Override wall ckpt (default: locked canonical)",
    )
    ap.add_argument(
        "--mat-leg",
        type=str,
        default="",
        help=f"Override mat-growth leg (default: {DEFAULT_MAT_LEG})",
    )
    args = ap.parse_args(argv)

    if args.list:
        cfgs = list_sweep_configs()
        if not cfgs:
            print(f"[WARN] No configs under {SWEEPS_DIR}", flush=True)
            return 1
        for p in cfgs:
            try:
                c = load_sweep_config(p)
                print(f"  {c.get('id', p.stem):28s}  {c.get('title', '')}", flush=True)
            except Exception as exc:
                print(f"  {p.name}: [ERR] {exc}", flush=True)
        return 0

    paths: list[Path] = []
    if args.all:
        paths = list_sweep_configs()
        if not paths:
            print(f"[ERR] No configs under {SWEEPS_DIR}", flush=True)
            return 1
    elif args.sweep.strip():
        paths = [resolve_sweep_path(args.sweep)]
    else:
        ap.print_help()
        print("\n[ERR] Pass --sweep <id> or --all (or --list)", flush=True)
        return 2

    wall = Path(args.wall_ckpt) if args.wall_ckpt.strip() else None
    mat_leg = args.mat_leg.strip() or DEFAULT_MAT_LEG
    if wall is None:
        print(f"[i] Using default locked ckpt: {DEFAULT_WALL_CKPT}", flush=True)

    pipeline = CustomerDeployPipeline(
        wall_ckpt=wall,
        mat_leg=mat_leg,
        require_cuda=not bool(args.cpu),
    )

    failures = 0
    for path in paths:
        try:
            cfg = load_sweep_config(path)
            run_sweep(
                cfg,
                pipeline=pipeline,
                force_rebuild=bool(args.force_rebuild_mesh),
                arm_filter=args.arm.strip() or None,
            )
        except Exception as exc:
            failures += 1
            print(f"[ERR] Sweep {path.name} failed: {exc}", flush=True)
            if not args.all:
                raise
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
