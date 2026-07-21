"""Print clot formation survey across all biochem anchor graphs.

Usage:
  python scripts/survey_clot_anchor_patterns.py
  python scripts/survey_clot_anchor_patterns.py --dx-thresh 25
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.clot_anchor_survey import (  # noqa: E402
    aggregate_separability,
    format_survey_table,
    survey_all_anchors,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Survey COMSOL clot nodes on anchor graphs")
    p.add_argument(
        "--dx-thresh",
        type=float,
        default=None,
        help="Override BIOCHEM_PRIOR_DGAMMA_DX_THRESH for scored prior column",
    )
    p.add_argument("--comsol-aligned", action="store_true", default=True)
    args = p.parse_args()

    if args.comsol_aligned:
        os.environ["BIOCHEM_PRIOR_COMSOL_ALIGNED"] = "1"
    os.environ["BIOCHEM_PRIOR_NORM_MASK"] = "adjacent"
    if args.dx_thresh is not None:
        os.environ["BIOCHEM_PRIOR_DGAMMA_DX_THRESH"] = str(args.dx_thresh)
    elif not os.environ.get("BIOCHEM_PRIOR_DGAMMA_DX_THRESH"):
        os.environ["BIOCHEM_PRIOR_DGAMMA_DX_THRESH"] = "800"

    surveys = survey_all_anchors()
    if not surveys:
        print(f"No patient*.pt under {REPO / 'data/processed/graphs_biochem_anchors'}")
        sys.exit(1)

    print(format_survey_table(surveys))
    print()
    agg = aggregate_separability(surveys)
    print("Aggregate (anchors with >=5 strict clots at t_final):")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print()
    print("Notes:")
    for s in surveys:
        if s.notes:
            print(f"  {s.stem}: {', '.join(s.notes)}")


if __name__ == "__main__":
    main()
