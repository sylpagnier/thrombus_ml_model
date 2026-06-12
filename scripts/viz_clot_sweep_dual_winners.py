"""Render patient007 timeline PNGs for shape-best and balanced-best sweep rules."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.promote_clot_architecture_winner import _parse_rule_name  # noqa: E402


def _apply_base_env() -> None:
    os.environ.setdefault("BIOCHEM_PRIOR_COMSOL_ALIGNED", "1")
    os.environ.setdefault("BIOCHEM_PRIOR_NORM_MASK", "adjacent")
    os.environ.setdefault("CLOT_PHI_DGAMMA_SLICE", "1")
    os.environ.setdefault("CLOT_PHI_CEILING_HOPS", "2")
    os.environ.setdefault("CLOT_FORECAST_MODE", "one_step")
    os.environ.setdefault("CLOT_FORECAST_MASK", "ceiling_growth")
    os.environ.setdefault("CLOT_FORECAST_PAIR_SCHEDULE", "from_t0")
    os.environ.setdefault("CLOT_FORECAST_PAIR_STRIDE", "1")
    os.environ.setdefault("CLOT_PHI_VEL_SOURCE", "gt")
    os.environ.setdefault("CLOT_PHI_FIXED_MU_FROM_PHI", "1")
    os.environ.setdefault("CLOT_PHI_HYBRID", "0")
    os.environ.setdefault("CLOT_PHI_HARD_SUPPORT_PROJECTION", "1")
    os.environ.setdefault("CLOT_PHI_SUPPORT_BAND", "ceiling_growth")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_P", "0.80")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_T0_STRIP", "0")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_FLUX_STAG_TOP", "0.20")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_TIE_BREAK", "1")
    os.environ.setdefault("CLOT_PHI_PRIOR_RULE_SKIP_INLET_Q", "0.25")


def _load_prior_winner_env() -> None:
    prior_ps1 = REPO / "scripts" / "_clot_prior_rule_winner_env.ps1"
    if not prior_ps1.is_file():
        return
    for line in prior_ps1.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith('$env:') and ' = "' in line:
            key, _, val = line[5:].partition(' = "')
            val = val.rstrip('"')
            os.environ[key] = val


def _run_viz(*, anchor: str, anchor_dir: str, out_png: Path, keyframes: int) -> None:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "viz_clot_temporal_rule_timeline.py"),
        "--anchor",
        anchor,
        "--anchor-dir",
        anchor_dir,
        "--keyframes",
        str(keyframes),
        "--out",
        str(out_png),
    ]
    subprocess.run(cmd, cwd=str(REPO), check=True)


def _viz_one(
    *,
    label: str,
    rule: str,
    anchor: str,
    anchor_dir: str,
    out_dir: Path,
    keyframes: int,
    source_json: Path,
) -> Path:
    env = _parse_rule_name(rule)
    for key, val in env.items():
        os.environ[key] = val
    out_png = out_dir / f"temporal_rule_{anchor}_{label}_best.png"
    print(f"[i] viz {label}: rule={rule} -> {out_png}", flush=True)
    _run_viz(anchor=anchor, anchor_dir=anchor_dir, out_png=out_png, keyframes=keyframes)
    sidecar = out_dir / f"temporal_rule_{anchor}_{label}_best_rule.json"
    sidecar.write_text(
        json.dumps({"rule": rule, "source": str(source_json), "env": env}, indent=2),
        encoding="utf-8",
    )
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description="Dual-winner clot timeline viz")
    ap.add_argument(
        "--json",
        default="outputs/biochem/diagnostics/clot_rule_ideas_sweep.json",
    )
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--keyframes", type=int, default=8)
    ap.add_argument(
        "--out-dir",
        default="outputs/biochem/viz/clot_deploy",
    )
    ap.add_argument("--shape-rule", default="", help="Override shape winner rule name")
    ap.add_argument("--balanced-rule", default="", help="Override balanced winner rule name")
    args = ap.parse_args()

    json_path = REPO / args.json
    if not json_path.is_file():
        print(f"[ERR] missing {json_path}", file=sys.stderr)
        return 1

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    shape_rule = (args.shape_rule or (payload.get("winner_shape") or {}).get("rule") or "").strip()
    bal_src = (
        payload.get("winner_deploy")
        or payload.get("winner_incubation")
        or payload.get("winner_balanced")
        or {}
    )
    bal_rule = (args.balanced_rule or bal_src.get("rule") or "").strip()
    if not shape_rule or not bal_rule:
        print("[ERR] sweep JSON missing winner_shape / winner_balanced", file=sys.stderr)
        return 1

    _apply_base_env()
    _load_prior_winner_env()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_png = _viz_one(
        label="shape",
        rule=shape_rule,
        anchor=args.anchor,
        anchor_dir=args.anchor_dir,
        out_dir=out_dir,
        keyframes=args.keyframes,
        source_json=json_path,
    )
    bal_png = _viz_one(
        label="balanced",
        rule=bal_rule,
        anchor=args.anchor,
        anchor_dir=args.anchor_dir,
        out_dir=out_dir,
        keyframes=args.keyframes,
        source_json=json_path,
    )

    print(f"[OK] shape PNG: {shape_png}")
    print(f"[OK] balanced PNG: {bal_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
