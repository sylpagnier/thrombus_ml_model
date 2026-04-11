"""
Master orchestrator: Stage A (predictor / Tier 1–2) → Stage B (corrector / Tier 3).

Runs the existing training modules in-process. Checkpoints go to ``outputs/stage_a`` and
``outputs/stage_b``. Checkpoints load only from ``outputs/stage_a`` / ``outputs/stage_b``. Reports default to ``outputs/reports/`` (``src.utils.paths.reports_dir``).
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _run_module_as_main(rel_path: str) -> None:
    path = Path(__file__).resolve().parent / rel_path
    runpy.run_path(str(path), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run Stage A and/or Stage B training.")
    p.add_argument(
        "stage",
        choices=("a", "b", "all"),
        help="Stage A: Tier 1 + Tier 2 predictor warm-up. Stage B: Tier 3 corrector. "
        "'all' runs A then B sequentially.",
    )
    p.add_argument(
        "--skip-tier1",
        action="store_true",
        help="With stage 'a' or 'all', skip Tier 1 and start at Tier 2 (requires Tier 1 weights or bootstrap).",
    )
    args = p.parse_args(argv)

    if args.stage in ("a", "all") and not args.skip_tier1:
        print("=== Stage A: Tier 1 ===")
        _run_module_as_main("training/train_t1_predictor.py")

    if args.stage in ("a", "all"):
        print("=== Stage A: Tier 2 ===")
        _run_module_as_main("training/train_t2_predictor.py")

    if args.stage in ("b", "all"):
        print("=== Stage B: Tier 3 (corrector) ===")
        _run_module_as_main("training/train_t3_corrector.py")


if __name__ == "__main__":
    main(sys.argv[1:])
