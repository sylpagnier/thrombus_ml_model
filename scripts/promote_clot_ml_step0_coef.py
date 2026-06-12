"""Write _clot_ml_step0_env.ps1 from Step 0 best_coef.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.clot_ml_step0_coef import load_step0_coef_json  # noqa: E402


def _render_ps1(env: dict[str, str], *, source: str) -> str:
    clears = [
        "CLOT_LOCALIZED_SPECIES_GT_Q",
        "CLOT_LOCALIZED_SPECIES_WEIGHT",
        "CLOT_LOCALIZED_SPECIES_TIME",
        "CLOT_TEMPORAL_ACCUM_GAIN",
        "CLOT_TEMPORAL_ACCUM_THRESHOLD",
    ]
    lines = [
        f"# Auto-promoted from {source}",
        "# Step 0 learned rule coefficients (pred GINO-DEQ kine)",
        "# Dot-source AFTER _clot_prior_rule_winner_env.ps1",
        "",
    ]
    for key in sorted(env):
        lines.append(f'$env:{key} = "{env[key]}"')
    for key in clears:
        if key not in env:
            lines.append(f"Remove-Item Env:{key} -ErrorAction SilentlyContinue")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    )
    ap.add_argument("--out", default="scripts/_clot_ml_step0_env.ps1")
    args = ap.parse_args()

    json_path = REPO / args.json
    if not json_path.is_file():
        print(f"[ERR] missing {json_path}", file=sys.stderr)
        return 1

    coef = load_step0_coef_json(json_path)
    env = coef.to_env()
    out_path = REPO / args.out
    out_path.write_text(_render_ps1(env, source=str(json_path)), encoding="utf-8")
    print(f"[OK] promoted ml_step0_coef -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
