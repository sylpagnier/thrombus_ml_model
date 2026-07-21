"""Promote biochem teacher ckpt to T5 deploy slot + write manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.training.clot_trigger_stack import (
    default_t5_deploy_paths,
    default_t5_predkine_species_dump_dir,
    snapshot_t5_deploy_train_config,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote teacher to T5 deploy checkpoint")
    ap.add_argument(
        "--src",
        default="outputs/biochem/biochem_teacher_last.pth",
        help="Teacher weights to promote (default: last run)",
    )
    ap.add_argument("--note", default="", help="Optional run note in manifest")
    args = ap.parse_args()

    paths = default_t5_deploy_paths()
    paths.out_root.mkdir(parents=True, exist_ok=True)

    src = Path(args.src)
    if not src.is_absolute():
        src = _REPO / src
    if not src.is_file():
        print(f"[ERR] missing teacher src: {src}", file=sys.stderr)
        return 2

    shutil.copy2(src, paths.teacher_deploy)
    meta = snapshot_t5_deploy_train_config()
    manifest = {
        "role": "clot_trigger_t5_deploy_teacher",
        "promoted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_checkpoint": str(src.relative_to(_REPO) if src.is_relative_to(_REPO) else src),
        "deploy_checkpoint": str(
            paths.teacher_deploy.relative_to(_REPO)
            if paths.teacher_deploy.is_relative_to(_REPO)
            else paths.teacher_deploy
        ),
        "predkine_species_dump_dir": str(
            default_t5_predkine_species_dump_dir().relative_to(_REPO)
            if default_t5_predkine_species_dump_dir().is_relative_to(_REPO)
            else default_t5_predkine_species_dump_dir()
        ),
        "train_config": meta,
        "note": args.note.strip(),
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] promoted -> {paths.teacher_deploy}")
    print(f"[save] {paths.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
