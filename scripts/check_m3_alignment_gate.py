"""M3 alignment gate: data + ADR co-descent on logged teacher metrics.

Pass when (teacher val rows, >=3 epochs):
  - train_L_bio_avg falls (last/first <= max_ratio)
  - train_L_ADR_S_avg falls (masked ADR path used in training when ADR_MASK_MODE != global)
  - optional: train_L_ADR_S_global_avg not exploding vs ep0

Usage:
  python scripts/check_m3_alignment_gate.py --run-note m3_A1_mask_match
"""

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
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _find_run_dir(*, run_note: str) -> Path | None:
    index = _REPORTS / "runs_index.jsonl"
    if index.is_file():
        for line in reversed(index.read_text(encoding="utf-8").splitlines()):
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


def _teacher_epochs(run_dir: Path) -> list[dict]:
    rows = _load_jsonl(run_dir / "run.jsonl")
    out = []
    for r in rows:
        if r.get("event") != "val" or r.get("stage") != "teacher":
            continue
        if r.get("epoch") is None:
            continue
        out.append(r)
    out.sort(key=lambda x: int(x["epoch"]))
    return out


def _series(epochs: list[dict], key: str) -> list[float]:
    vals = []
    for r in epochs:
        v = r.get(key)
        if v is not None:
            vals.append(float(v))
    return vals


def _descent_ok(vals: list[float], *, max_ratio: float, monotone: bool) -> tuple[bool, str]:
    if len(vals) < 2:
        return False, "need>=2_epochs"
    first, last = vals[0], vals[-1]
    if first <= 0:
        return False, "first<=0"
    ratio = last / first
    if ratio > max_ratio:
        return False, f"ratio={ratio:.4f}>{max_ratio}"
    if monotone:
        for a, b in zip(vals, vals[1:]):
            if b > a * 1.05:
                return False, "non_monotone"
    return True, f"ratio={ratio:.4f}"


def eval_m3(run_dir: Path, *, max_ratio: float) -> dict:
    eps = _teacher_epochs(run_dir)
    bio = _series(eps, "train_L_bio_avg") or _series(eps, "train_L_back_avg")
    adr_s = _series(eps, "train_L_ADR_S_avg")
    adr_s_g = _series(eps, "train_L_ADR_S_global_avg")
    if not adr_s_g:
        adr_s_g = adr_s

    bio_ok, bio_note = _descent_ok(bio, max_ratio=max_ratio, monotone=False)
    adr_ok, adr_note = _descent_ok(adr_s, max_ratio=max_ratio, monotone=False)
    adr_g_ok, adr_g_note = _descent_ok(adr_s_g, max_ratio=max_ratio, monotone=False)

    passive_adr = any(bool(r.get("passive_adr_in_backprop")) for r in eps)
    mask_mode = (eps[-1].get("ADR_mask_mode") if eps else None) or "global"

    ok = bool(
        bio_ok
        and len(bio) >= 3
        and (adr_ok or not passive_adr)
        and len(adr_s) >= 3
    )
    if passive_adr and not adr_ok:
        ok = False

    return {
        "ok": ok,
        "n_epochs": len(bio),
        "passive_adr_in_backprop": passive_adr,
        "ADR_mask_mode": mask_mode,
        "L_bio": bio,
        "L_ADR_S": adr_s,
        "L_ADR_S_global": adr_s_g,
        "bio_ok": bio_ok,
        "bio_note": bio_note,
        "adr_s_ok": adr_ok,
        "adr_s_note": adr_note,
        "adr_s_global_ok": adr_g_ok,
        "adr_s_global_note": adr_g_note,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-note", required=True)
    ap.add_argument("--max-ratio", type=float, default=0.92)
    args = ap.parse_args()

    rd = _find_run_dir(run_note=args.run_note)
    if rd is None:
        print(f"[ERR] run not found: {args.run_note}", file=sys.stderr)
        return 2

    out = eval_m3(rd, max_ratio=args.max_ratio)
    out["run_dir"] = str(rd)
    out["run_note"] = args.run_note
    tag = "[OK]" if out.get("ok") else "[FAIL]"
    print(f"{tag} m3_align: {json.dumps(out, sort_keys=True)}")
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
