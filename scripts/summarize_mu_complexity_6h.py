"""Rank comprehensive mu sweep legs by val mu_log_mae (teacher + corrector stages)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = _REPO / "outputs" / "reports" / "training" / "biochem"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _find_run_dir(run_note: str) -> Path | None:
    index = _REPORTS / "runs_index.jsonl"
    if index.is_file():
        for line in reversed(index.read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_note") == run_note:
                rd = Path(row["run_dir"])
                if rd.is_dir():
                    return rd
    for p in sorted(_REPORTS.glob("*/run.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        rows = _load_jsonl(p)
        if rows and rows[0].get("run_note") == run_note:
            return p.parent
    return None


def _stage_vals(run_dir: Path, key: str, stage: str | None = None) -> list[tuple[int, float, str]]:
    out: list[tuple[int, float, str]] = []
    for r in _load_jsonl(run_dir / "run.jsonl"):
        if r.get("event") != "val":
            continue
        st = (r.get("stage") or "teacher").lower()
        if stage is not None and st != stage:
            continue
        v = r.get(key)
        if v is not None:
            out.append((int(r["epoch"]), float(v), st))
    return out


def summarize_run(run_note: str) -> dict | None:
    rd = _find_run_dir(run_note)
    if rd is None:
        return None

    all_mu = _stage_vals(rd, "val_mu_log_mae")
    if not all_mu:
        return None

    best_ep, best_mu, best_stage = min(all_mu, key=lambda x: x[1])
    teacher_mu = _stage_vals(rd, "val_mu_log_mae", "teacher")
    corrector_mu = _stage_vals(rd, "val_mu_log_mae", "corrector")
    wall = _stage_vals(rd, "val_mu_log_mae_wall")
    high = _stage_vals(rd, "val_mu_log_mae_high_mu")
    fi = _stage_vals(rd, "val_species_fi_log_mae", "teacher")

    end = next((e for e in _load_jsonl(rd / "run.jsonl") if e.get("event") == "end"), None)
    pseudo_w = end.get("pseudo_w") if end else None
    pseudo_cov = end.get("pseudo_label_coverage") if end else None

    return {
        "run_note": run_note,
        "run_dir": str(rd),
        "val_mu_log_mae": best_mu,
        "best_epoch": best_ep,
        "best_stage": best_stage,
        "teacher_mu_last": teacher_mu[-1][1] if teacher_mu else None,
        "teacher_mu_min": min((v for _, v, _ in teacher_mu), default=None),
        "corrector_mu_last": corrector_mu[-1][1] if corrector_mu else None,
        "corrector_mu_min": min((v for _, v, _ in corrector_mu), default=None),
        "corrector_val_n": len(corrector_mu),
        "val_mu_log_mae_wall": wall[-1][1] if wall else None,
        "val_mu_log_mae_high_mu": high[-1][1] if high else None,
        "val_species_fi_log_mae": fi[-1][1] if fi else None,
        "pseudo_w": pseudo_w,
        "pseudo_label_coverage": pseudo_cov,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/biochem/sweep_mu_complexity_6h/manifest.jsonl")
    ap.add_argument("--out", default="outputs/biochem/sweep_mu_complexity_6h/summary.json")
    ap.add_argument("--run-note", default="")
    ap.add_argument("--manifest-append", default="")
    ap.add_argument("--leg-id", default="")
    ap.add_argument("--tier", default="")
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--train-exit", type=int, default=0)
    args = ap.parse_args()

    if args.run_note:
        row = summarize_run(args.run_note)
        if row is None:
            print(f"[WARN] no run.jsonl for {args.run_note}", file=sys.stderr)
            row = {"run_note": args.run_note, "val_mu_log_mae": None}
        row["leg_id"] = args.leg_id or args.run_note.replace("mu6h_", "", 1)
        row["tier"] = args.tier
        row["hypothesis"] = args.hypothesis
        row["train_exit"] = args.train_exit
        append = Path(args.manifest_append or args.manifest)
        append.parent.mkdir(parents=True, exist_ok=True)
        with append.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[OK] manifest row: {row['leg_id']} mu={row.get('val_mu_log_mae')} stage={row.get('best_stage')}")
        return 0 if row.get("val_mu_log_mae") is not None else 1

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[ERR] missing manifest: {manifest_path}", file=sys.stderr)
        return 2

    legs = _load_jsonl(manifest_path)
    ok_legs = [l for l in legs if l.get("val_mu_log_mae") is not None]
    ok_legs.sort(key=lambda x: float(x["val_mu_log_mae"]))

    out = {"legs": ok_legs, "n_runs": len(ok_legs), "primary_metric": "val_mu_log_mae"}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[OK] {len(ok_legs)} legs ranked -> {out_path}")
    print(
        f"{'leg_id':16s} {'tier':14s} {'mu_best':>8s} {'stage':>9s} "
        f"{'pseudo_w':>8s} {'corr_n':>6s} {'wall':>8s} {'high':>8s}"
    )
    for leg in ok_legs:
        print(
            f"{leg.get('leg_id',''):16s} "
            f"{leg.get('tier',''):14s} "
            f"{float(leg['val_mu_log_mae']):8.4f} "
            f"{str(leg.get('best_stage') or ''):>9s} "
            f"{float(leg.get('pseudo_w') or 0):8.3f} "
            f"{int(leg.get('corrector_val_n') or 0):6d} "
            f"{float(leg.get('val_mu_log_mae_wall') or 0):8.4f} "
            f"{float(leg.get('val_mu_log_mae_high_mu') or 0):8.4f}"
        )
    if ok_legs:
        top = ok_legs[0]
        print(
            f"[i]  best: {top.get('leg_id')} mu={top.get('val_mu_log_mae')} "
            f"({top.get('best_stage')} ep{top.get('best_epoch')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
