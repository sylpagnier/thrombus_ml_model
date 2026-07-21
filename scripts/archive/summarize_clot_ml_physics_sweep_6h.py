"""Summarize clot ML physics sweep legs by mean deploy_score."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _last_train_row(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    last = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last


def summarize_leg(leg_dir: Path) -> dict | None:
    meta_path = leg_dir / "leg_meta.json"
    summary = _read_json(leg_dir / "summary.json")
    train_last = _last_train_row(leg_dir / "train_log.jsonl")
    if not summary and not train_last and not meta_path.is_file():
        return None

    meta = _read_json(meta_path) or {"leg_id": leg_dir.name}
    mean_deploy = float("nan")
    p007_f1 = float("nan")
    p007_deploy = float("nan")
    per_anchor: list[dict] = []
    status = str(meta.get("status", ""))

    if summary:
        mean_deploy = float(summary.get("mean_deploy", float("nan")))
        per_anchor = list(summary.get("per_anchor") or [])
    elif train_last:
        mean_deploy = float(train_last.get("mean_deploy_all", float("nan")))
        per_anchor = list(train_last.get("per_anchor") or [])
        epochs_logged = int(train_last.get("epoch", 0) or 0)
        epochs_target = int(meta.get("epochs", 0) or 0)
        if not summary:
            status = "partial"
        elif epochs_target and epochs_logged < epochs_target:
            status = "partial"

    for row in per_anchor:
        if str(row.get("anchor")) == "patient007":
            p007_f1 = float(row.get("tfinal_band_f1", float("nan")))
            p007_deploy = float(row.get("deploy_score", float("nan")))
            break

    if not status:
        if mean_deploy != mean_deploy:
            status = "not_run"
        else:
            status = "ok"

    return {
        "leg_id": str(meta.get("leg_id", leg_dir.name)),
        "hypothesis": str(meta.get("hypothesis", "")),
        "physics": str(meta.get("physics", "")),
        "mean_deploy": mean_deploy,
        "p007_tfinal_band_f1": p007_f1,
        "p007_deploy": p007_deploy,
        "epochs_ran": int(meta.get("epochs", 0) or 0),
        "ckpt": str(meta.get("ckpt", "")),
        "viz_png": str(meta.get("viz_png", "")),
        "status": status,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize clot ML physics 6h sweep")
    ap.add_argument(
        "--sweep-dir",
        default="outputs/biochem/sweep_clot_ml_physics_6h",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = REPO / sweep_dir

    rows: list[dict] = []
    for leg_dir in sorted(sweep_dir.iterdir()):
        if not leg_dir.is_dir():
            continue
        row = summarize_leg(leg_dir)
        if row is not None:
            rows.append(row)

    rows.sort(
        key=lambda r: (
            -(r["mean_deploy"] if r["mean_deploy"] == r["mean_deploy"] else -1.0),
            -(r["p007_tfinal_band_f1"] if r["p007_tfinal_band_f1"] == r["p007_tfinal_band_f1"] else -1.0),
        ),
    )

    out_path = Path(args.out) if args.out else sweep_dir / "sweep_summary.json"
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sweep_dir": str(sweep_dir), "legs": rows}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[i] legs={len(rows)} -> {out_path}")
    print("leg_id                  mean_deploy  p007_F1  p007_deploy  status")
    print("-" * 72)
    for r in rows:
        md = r["mean_deploy"]
        f1 = r["p007_tfinal_band_f1"]
        pd = r["p007_deploy"]
        md_s = f"{md:.3f}" if md == md else "nan"
        f1_s = f"{f1:.3f}" if f1 == f1 else "nan"
        pd_s = f"{pd:.3f}" if pd == pd else "nan"
        print(f"{r['leg_id']:22} {md_s:>11} {f1_s:>8} {pd_s:>12}  {r['status']}")
    completed = [r for r in rows if r["status"] == "ok" and r["mean_deploy"] == r["mean_deploy"]]
    if completed:
        best = completed[0]
        bmd = best["mean_deploy"]
        bmd_s = f"{bmd:.3f}" if bmd == bmd else "nan"
        print(
            f"[OK] best completed={best['leg_id']} mean_deploy={bmd_s} "
            f"hypothesis={best.get('hypothesis', '')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
