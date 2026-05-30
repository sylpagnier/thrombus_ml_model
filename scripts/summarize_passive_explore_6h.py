"""Rank legs from go_passive_explore_6h.ps1 using run.jsonl + explore_log.jsonl."""

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


def _val_series(run_dir: Path, key: str) -> list[float]:
    eps = [
        r
        for r in _load_jsonl(run_dir / "run.jsonl")
        if r.get("event") == "val" and r.get("stage") == "teacher"
    ]
    eps.sort(key=lambda x: int(x["epoch"]))
    out: list[float] = []
    for r in eps:
        v = r.get(key)
        if v is not None:
            out.append(float(v))
    return out


def summarize_run(run_note: str) -> dict | None:
    rd = _find_run_dir(run_note)
    if rd is None:
        return None
    mu = _val_series(rd, "val_mu_log_mae")
    fi = _val_series(rd, "val_species_fi_log_mae")
    wall = _val_series(rd, "val_mu_log_mae_wall")
    high = _val_series(rd, "val_mu_log_mae_high_mu")
    l_bio = _val_series(rd, "train_L_bio_avg")
    return {
        "run_note": run_note,
        "run_dir": str(rd),
        "n_val": len(mu),
        "mu_first": mu[0] if mu else None,
        "mu_last": mu[-1] if mu else None,
        "mu_drop": (mu[0] - mu[-1]) if len(mu) >= 2 else None,
        "fi_last": fi[-1] if fi else None,
        "wall_last": wall[-1] if wall else None,
        "high_last": high[-1] if high else None,
        "l_bio_first": l_bio[0] if l_bio else None,
        "l_bio_last": l_bio[-1] if l_bio else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="outputs/biochem/explore_6h/explore_log.jsonl")
    ap.add_argument("--out", default="outputs/biochem/explore_6h/summary.json")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.is_file():
        print(f"[ERR] missing log: {log_path}", file=sys.stderr)
        return 2

    steps: list[str] = []
    for row in _load_jsonl(log_path):
        if row.get("status") in ("OK", "WARN", "DONE") and row.get("step", "").startswith("expl6h_"):
            steps.append(str(row["step"]))

    legs: list[dict] = []
    for note in steps:
        s = summarize_run(note)
        if s:
            legs.append(s)

    legs.sort(
        key=lambda x: (
            float(x["fi_last"]) if x.get("fi_last") is not None else 999.0,
            -(float(x["mu_drop"]) if x.get("mu_drop") is not None else -999.0),
        ),
    )

    out = {"legs": legs, "n_runs": len(legs)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[OK] {len(legs)} runs summarized -> {out_path}")
    print(f"{'run_note':28s} {'mu_last':>8s} {'fi_last':>8s} {'wall_last':>10s} {'high_last':>10s}")
    for leg in legs:
        print(
            f"{leg['run_note']:28s} "
            f"{leg.get('mu_last') or float('nan'):8.4f} "
            f"{leg.get('fi_last') or float('nan'):8.4f} "
            f"{leg.get('wall_last') or float('nan'):10.4f} "
            f"{leg.get('high_last') or float('nan'):10.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
