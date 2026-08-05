"""Flow-source A/B gate before phase1_sweep_v3.

Trains three typed legs on the small phase1 cohort, then cold-eval each under
identical deploy-faithful protocol (RGP-DEQ @ t=0 + local tiling; no GT velocity):

  FS_ab_gt      - train on COMSOL GT flow (historical crutch)
  FS_ab_kine    - train on clot-blind RGP-DEQ base flow
  FS_ab_coupled - train on RGP-DEQ + local-tiling coupled flow

Decision uses holdout **patient020** only (clot-rich) deploy_clot_score / deploy_clot_f1.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

LEG_ORDER = ("FS_ab_gt", "FS_ab_kine", "FS_ab_coupled")

DEFAULT_TRAIN = "patient005,patient006,patient010,patient023,patient002"
DEFAULT_VAL = "patient020"
DEFAULT_HOLDOUT = "patient020"
DEFAULT_ROOT = "outputs/biochem/eda/flow_source_ab"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(label: str, cmd: list[str], *, cwd: Path) -> int:
    print(f"[NEW] {label}", flush=True)
    print(f"[i] {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd))
    dt = time.perf_counter() - t0
    tag = "OK" if proc.returncode == 0 else "ERR"
    print(f"[{tag}] {label} exit={proc.returncode} ({dt / 60.0:.1f} min)", flush=True)
    return int(proc.returncode)


def _mean_metrics(eval_json: Path) -> dict[str, float]:
    raw = json.loads(eval_json.read_text(encoding="utf-8"))
    mean = ((raw.get("simple") or {}).get("mean")) or {}
    return {
        "deploy_clot_score": float(mean.get("deploy_clot_score") or 0.0),
        "deploy_clot_f1": float(mean.get("deploy_clot_f1") or 0.0),
        "deploy_clot_offwall_relaxed_prec": float(
            mean.get("deploy_clot_offwall_relaxed_prec") or 0.0
        ),
        "deploy_clot_offwall_relaxed_rec": float(
            mean.get("deploy_clot_offwall_relaxed_rec") or 0.0
        ),
    }


def _summarize(out_dir: Path, legs: list[str]) -> Path:
    rows: list[dict[str, object]] = []
    for leg in legs:
        eval_path = out_dir / leg / "eval_holdout_cold.json"
        if not eval_path.is_file():
            rows.append({"leg": leg, "status": "missing"})
            continue
        m = _mean_metrics(eval_path)
        rows.append({"leg": leg, "status": "ok", **m})

    csv_path = out_dir / "flow_source_ab_summary.csv"
    json_path = out_dir / "flow_source_ab_summary.json"
    fields = [
        "leg",
        "status",
        "deploy_clot_score",
        "deploy_clot_f1",
        "deploy_clot_offwall_relaxed_prec",
        "deploy_clot_offwall_relaxed_rec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    json_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    print("", flush=True)
    print("[OK] flow-source A/B summary", flush=True)
    for row in rows:
        if row.get("status") != "ok":
            print(f"  {row['leg']}: MISSING", flush=True)
            continue
        print(
            f"  {row['leg']}: score={row['deploy_clot_score']:.4f} "
            f"f1={row['deploy_clot_f1']:.4f} "
            f"rprec={row['deploy_clot_offwall_relaxed_prec']:.4f} "
            f"rrec={row['deploy_clot_offwall_relaxed_rec']:.4f}",
            flush=True,
        )
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        best = max(ok, key=lambda r: float(r["deploy_clot_score"]))
        print(
            f"[i] winner_by_score={best['leg']} "
            f"(deploy_clot_score={float(best['deploy_clot_score']):.4f})",
            flush=True,
        )
    print(f"[save] {csv_path}", flush=True)
    return csv_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--train-anchors", default=DEFAULT_TRAIN)
    ap.add_argument("--val-anchor", default=DEFAULT_VAL)
    ap.add_argument("--holdout-anchors", default=DEFAULT_HOLDOUT)
    ap.add_argument("--run-root", default=DEFAULT_ROOT)
    ap.add_argument(
        "--arm-filter",
        default="",
        help="Comma list of FS_ab_* codes (default: all three)",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args(argv)

    root = _repo_root()
    out_dir = root / args.run_root
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.arm_filter.strip():
        legs = [x.strip() for x in args.arm_filter.split(",") if x.strip()]
    else:
        legs = list(LEG_ORDER)
    unknown = [x for x in legs if x not in LEG_ORDER]
    if unknown:
        print(f"[ERR] unknown arms {unknown}; choose from {list(LEG_ORDER)}", flush=True)
        return 2

    if args.summary_only:
        _summarize(out_dir, legs)
        return 0

    py = sys.executable
    for leg in legs:
        arm_dir = out_dir / leg
        arm_dir.mkdir(parents=True, exist_ok=True)
        ckpt = arm_dir / "best.pth"
        hold = arm_dir / "eval_holdout_cold.json"

        if args.fresh:
            # Eval-only fresh: drop holdout JSON only. Full fresh also drops train artifacts.
            to_drop = [hold]
            if not args.eval_only:
                to_drop.extend(
                    (ckpt, arm_dir / "best.json", arm_dir / "last.pth", arm_dir / "last.json", arm_dir / "train_log.jsonl")
                )
            for p in to_drop:
                if p.is_file():
                    p.unlink()

        if hold.is_file() and not args.fresh and not args.eval_only:
            print(f"[skip] {leg} already has eval JSON", flush=True)
            continue

        if not args.eval_only:
            if ckpt.is_file() and hold.is_file() and not args.fresh:
                print(f"[skip] {leg} train (eval exists)", flush=True)
            else:
                rc = _run(
                    f"train {leg}",
                    [
                        py,
                        "-m",
                        "src.training.train_species_pushforward_continuous",
                        "--phase",
                        "biochem_gnn",
                        "--recipe",
                        "mat_growth_simple",
                        "--leg",
                        leg,
                        "--out",
                        str(ckpt),
                        "--epochs",
                        str(int(args.epochs)),
                        "--early-stop",
                        str(int(args.early_stop)),
                        "--anchors",
                        str(args.train_anchors),
                        "--val-anchor",
                        str(args.val_anchor),
                        "--exclude-val-from-train",
                        "--no-init",
                        "--drop-xy",
                    ],
                    cwd=root,
                )
                if rc != 0 or not ckpt.is_file():
                    print(f"[WARN] {leg} train failed; skipping eval", flush=True)
                    continue

        if not ckpt.is_file():
            print(f"[WARN] {leg} missing ckpt; skip eval", flush=True)
            continue

        # Do not pass --mat-leg: eval forces deploy-faithful from ckpt meta + recipe.
        rc = _run(
            f"eval {leg}",
            [
                py,
                "scripts/eval_mat_growth_simple.py",
                "--ckpt",
                str(ckpt),
                "--no-baseline",
                "--anchors",
                str(args.holdout_anchors),
                "--out",
                str(hold),
            ],
            cwd=root,
        )
        if rc != 0:
            print(f"[WARN] {leg} eval failed", flush=True)

    _summarize(out_dir, legs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
