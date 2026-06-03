"""Re-extract biochem anchor COMSOL graphs with trainer-compatible kinematics layout.

Writes:
  - ``data/processed/graphs_biochem_anchors/<stem>.pt`` (transient biochem + 18ch ``x``)
  - ``data/processed/graphs_kinematics_anchors/carreau/<stem>.pt`` (steady ``KINE_Y_SCHEMA``)

Run after enriching anchor meshes (``prepare_biochem_anchors --enrich-sidecars``) and
with COMSOL exports under ``data/processed/cfd_results_biochem/``.

Example:
    python scripts/reextract_anchor_comsol_graphs.py
    python scripts/reextract_anchor_comsol_graphs.py --stem patient007
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor
from src.tools.prepare_biochem_anchors import enrich_anchor_meshes, stems_in_dir

# Sidecars for .nas-only anchors are written during extract (COMSOL boundary masks).
from src.utils.paths import data_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stem", type=str, default="", help="Only this stem (default: all meshes).")
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip sidecar centerline/d_bar enrichment (use existing JSON).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Anchor mesh dir (default: data/raw/biochem_anchors).",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or (data_root() / "raw" / "biochem_anchors")
    if not raw_dir.is_dir():
        raise SystemExit(f"[ERR] raw anchor dir missing: {raw_dir}")

    stems = [args.stem.strip()] if args.stem.strip() else stems_in_dir(raw_dir)
    if not stems:
        raise SystemExit(f"[ERR] no mesh stems under {raw_dir}")

    if not args.skip_enrich:
        print("[i] Enriching anchor sidecars from mesh Gmsh tags (optional; .nas may skip)...")
        n = enrich_anchor_meshes(raw_dir, overwrite=False, dry_run=False, stems=stems)
        print(f"[OK] Gmsh-tag enrich: {n} sidecar(s); extract also writes sidecar from COMSOL masks")

    extractor = PatientDataExtractor(phase="biochem_anchors")
    for stem in stems:
        print(f"[i] extract {stem}")
        extractor.process_patient(stem)

    print(f"[OK] biochem graphs -> {extractor.proc_dir}")
    print(f"[OK] kinematics anchor graphs -> {extractor.kine_anchor_dir}")


if __name__ == "__main__":
    main()
