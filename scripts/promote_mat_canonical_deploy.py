"""Promote Mat W/WC canonical winner to a stable deploy alias path."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import LADDER_ROOT  # noqa: E402

CANONICAL_ROOT = REPO / "outputs/biochem/biochem_gnn/mat_canonical_deploy"
REFERENCE_JSON = REPO / "data/reference/mat_canonical_deploy.json"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        print(f"[WARN] missing {src}", flush=True)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    print(f"[OK] {dst.relative_to(REPO)} <- {src.relative_to(REPO)}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote mat W/WC canonical winner")
    ap.add_argument("--summary", required=True, help="summary JSON from summarize_mat_only_full.py")
    ap.add_argument("--skip-copy", action="store_true", help="write manifest only")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = (REPO / summary_path).resolve()
    if not summary_path.is_file():
        print(f"[ERR] missing summary: {summary_path}", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    winner = summary.get("winner")
    if not isinstance(winner, dict) or not winner.get("leg"):
        print("[ERR] summary has no winner; run with --pick-winner first", file=sys.stderr)
        return 1

    leg = str(winner["leg"])
    leg_dir = Path(LADDER_ROOT) / leg
    if not leg_dir.is_absolute():
        leg_dir = (REPO / leg_dir).resolve()

    src_ckpt = leg_dir / "species" / "best.pth"
    src_meta = leg_dir / "species" / "best.json"
    dest_species = CANONICAL_ROOT / "species"
    dest_ckpt = dest_species / "best.pth"
    dest_meta = dest_species / "best.json"

    if not args.skip_copy:
        if not _copy_if_exists(src_ckpt, dest_ckpt):
            return 1
        _copy_if_exists(src_meta, dest_meta)

    promoted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "stack": "mat_growth_simple",
        "leg": leg,
        "label": winner.get("label"),
        "promoted_at": promoted_at,
        "source_ckpt": str(src_ckpt.relative_to(REPO)).replace("\\", "/"),
        "canonical_ckpt": str(dest_ckpt.relative_to(REPO)).replace("\\", "/"),
        "summary_json": str(summary_path.relative_to(REPO)).replace("\\", "/"),
        "cohort_mean": winner.get("cohort_mean") or {},
        "pick_config": summary.get("pick_config") or {},
    }
    manifest_path = CANONICAL_ROOT / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {manifest_path.relative_to(REPO)}", flush=True)

    REFERENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {REFERENCE_JSON.relative_to(REPO)}", flush=True)
    print(f"[OK] canonical Mat deploy = {leg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
