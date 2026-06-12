"""Promote s34+s35 artifacts into deploy manifest for Rung 4 species_gnn step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.inference.species_gnn_deploy_env import (  # noqa: E402
    DEFAULT_MANIFEST,
    default_deploy_manifest,
    write_default_manifest,
)
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Write species GNN deploy manifest")
    ap.add_argument("--species-ckpt", default="outputs/biochem/species_snapshot_s34/best.pth")
    ap.add_argument("--beta", default="outputs/biochem/species_snapshot_s35/beta.pth")
    ap.add_argument("--kine-ckpt", default="outputs/kinematics/production_allfix/kinematics_best.pth")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--loao-dir", default="")
    ap.add_argument("--out", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    root = get_project_root()
    manifest = default_deploy_manifest()
    manifest["species_gnn_ckpt"] = str(args.species_ckpt)
    manifest["viscosity_beta"] = str(args.beta)
    manifest["kinematics_ckpt"] = str(args.kine_ckpt)
    manifest["train_val_anchor"] = str(args.val_anchor)
    if args.loao_dir.strip():
        manifest["loao_dir"] = str(args.loao_dir)
        manifest["phase"] = "species_gnn_deploy_r4_loao"

    for label, rel in (
        ("species_gnn_ckpt", args.species_ckpt),
        ("viscosity_beta", args.beta),
        ("kinematics_ckpt", args.kine_ckpt),
    ):
        p = Path(rel)
        if not p.is_absolute():
            p = root / p
        if not p.is_file():
            print(f"[WARN] missing {label}: {p}", file=sys.stderr)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] manifest -> {out}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
