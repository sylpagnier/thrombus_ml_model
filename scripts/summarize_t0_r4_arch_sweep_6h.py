"""Rank T0 Rung4 architecture sweep legs by val F1 vs s0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.t0_r4_sweep import DEFAULT_SWEEP_ORDER, RECIPES, recipe_from_id
from src.utils.paths import get_project_root


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _best_train_row(log_path: Path) -> dict | None:
    if not log_path.is_file():
        return None
    best: dict | None = None
    for line in log_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if best is None or float(row.get("val_f1", -1)) > float(best.get("val_f1", -1)):
            best = row
    return best


def summarize_leg(leg_dir: Path, *, anchor: str = "patient007") -> dict | None:
    leg_id = leg_dir.name
    recipe = recipe_from_id(leg_id) if leg_id in RECIPES else None
    eval_path = leg_dir / f"eval_{anchor}.json"
    eval_row = _load_json(eval_path)
    train_best = _best_train_row(leg_dir / "train_log.jsonl")
    ckpt_meta = _load_json(leg_dir / "best.json")

    if leg_id == "ref_s0":
        eval_path = leg_dir / f"eval_{anchor}.json"
        if not eval_path.is_file():
            eval_path = get_project_root() / f"outputs/biochem/clot_trigger/t0_rung4_s0_{anchor}.json"
        s0_eval = _load_json(eval_path)
        if s0_eval is None:
            return None
        f1 = float(s0_eval["rung4_step"]["clot_nucleation"][-1]["clot_f1"])
        return {
            "leg_id": leg_id,
            "family": "ref",
            "hypothesis": RECIPES["ref_s0"].hypothesis,
            "val_f1": f1,
            "delta_vs_s0": 0.0,
            "health_pass": True,
            "wall_carpet": False,
            "status": "ref",
        }

    if eval_row is None and train_best is None:
        return None

    f1 = float(eval_row["sweep_leg"]["clot_nucleation"][-1]["clot_f1"]) if eval_row else float(train_best["val_f1"])
    health = eval_row.get("rollout_health", {}) if eval_row else {}
    delta = float(eval_row.get("delta_vs_s0", {}).get("f1", 0.0)) if eval_row else 0.0
    row = {
        "leg_id": leg_id,
        "family": recipe.family if recipe else "unknown",
        "hypothesis": recipe.hypothesis if recipe else "",
        "val_f1": f1,
        "delta_vs_s0": delta,
        "health_pass": bool(health.get("health_pass", train_best.get("health_pass") if train_best else False)),
        "wall_carpet": bool(health.get("wall_carpet", train_best.get("wall_carpet") if train_best else False)),
        "health_score": float(health.get("health_score", train_best.get("health_score", 0) if train_best else 0)),
        "best_epoch": int((ckpt_meta or {}).get("meta", {}).get("epoch", 0)),
        "status": "ok",
    }
    diag = _load_json(leg_dir / f"diagnostic_{anchor}.json")
    if diag:
        row["verdict"] = diag.get("verdict")
        row["identical_to_s0"] = diag.get("vs_s0", {}).get("identical_to_s0")
        row["fi_mae_improve"] = diag.get("vs_s0", {}).get("final_fi_mae_improve")
        row["fn_fixed_final"] = (
            diag.get("timeline", [{}])[-1].get("localization", {}).get("fn_fixed")
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize T0 R4 arch sweep")
    ap.add_argument("--sweep-dir", default="outputs/biochem/sweep_t0_r4_arch_6h")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--leg-id", default="")
    ap.add_argument("--train-exit", type=int, default=0)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--manifest-append", default="")
    args = ap.parse_args()

    root = get_project_root()
    sweep_dir = root / args.sweep_dir
    manifest_path = Path(args.manifest_append) if args.manifest_append else (
        Path(args.manifest) if args.manifest else sweep_dir / "manifest.jsonl"
    )
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    if args.leg_id:
        leg_dir = sweep_dir / args.leg_id
        row = summarize_leg(leg_dir, anchor=args.anchor) or {
            "leg_id": args.leg_id,
            "status": "fail" if args.train_exit != 0 else "missing_eval",
            "train_exit": args.train_exit,
            "hypothesis": args.hypothesis,
        }
        row["train_exit"] = args.train_exit
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row, indent=2))
        return 0

    rows: list[dict] = []
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        for leg_id in DEFAULT_SWEEP_ORDER:
            leg_dir = sweep_dir / leg_id
            if leg_dir.is_dir():
                r = summarize_leg(leg_dir, anchor=args.anchor)
                if r:
                    rows.append(r)

    rows.sort(key=lambda r: float(r.get("val_f1", -1)), reverse=True)
    s0_f1 = next((float(r["val_f1"]) for r in rows if r.get("leg_id") == "ref_s0"), 0.408)

    summary = {
        "anchor": args.anchor,
        "s0_baseline_f1": s0_f1,
        "n_legs": len(rows),
        "best_leg": rows[0] if rows else None,
        "beats_s0": [r for r in rows if float(r.get("val_f1", 0)) > s0_f1 + 1e-4 and r.get("leg_id") != "ref_s0"],
        "healthy_beats_s0": [
            r for r in rows
            if float(r.get("val_f1", 0)) > s0_f1 + 1e-4
            and not r.get("wall_carpet")
            and r.get("health_pass")
            and r.get("leg_id") != "ref_s0"
        ],
        "legs": rows,
    }

    out_path = Path(args.out) if args.out else sweep_dir / "summary.json"
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] summary -> {out_path}", flush=True)
    print(f"[i] s0 baseline F1={s0_f1:.3f}", flush=True)
    print("[i] ranked legs (F1 | delta | health | family | id):", flush=True)
    for r in rows[:12]:
        print(
            f"    {float(r.get('val_f1', 0)):.3f}  {float(r.get('delta_vs_s0', 0)):+.3f}  "
            f"{'PASS' if r.get('health_pass') else 'FAIL':4s}  "
            f"{r.get('family', '?'):8s}  {r.get('leg_id')}",
            flush=True,
        )
    if summary["healthy_beats_s0"]:
        best = summary["healthy_beats_s0"][0]
        print(f"[OK] best healthy beat-s0: {best['leg_id']} F1={best['val_f1']:.3f}", flush=True)
    else:
        print("[WARN] no leg beat s0 with health pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
