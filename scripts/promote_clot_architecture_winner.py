"""Write _clot_architecture_winner_env.ps1 from sweep JSON winner rule name."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _parse_rule_name(name: str) -> dict[str, str]:
    """Map sweep rule name -> env vars for timeline viz / deploy."""
    env: dict[str, str] = {}

    # Must precede generic ``_incNN`` — ``sh_*_inc40`` names also end in ``_inc40``.
    shear = re.match(r"^sh_(.+)_inc40$", name)
    if shear:
        tag = shear.group(1)
        _shear_loc_env = {
            "base": {"CLOT_SHEAR_W_NEG_DX": "0.25"},
            "neg55": {"CLOT_SHEAR_W_NEG_DX": "0.55", "CLOT_SHEAR_W_SEP": "0.15", "CLOT_SHEAR_W_STASIS": "0.15", "CLOT_SHEAR_W_LGRAD": "0.15"},
            "neg70": {"CLOT_SHEAR_W_NEG_DX": "0.70", "CLOT_SHEAR_W_SEP": "0.10", "CLOT_SHEAR_W_STASIS": "0.10", "CLOT_SHEAR_W_LGRAD": "0.10"},
            "sep40": {"CLOT_SHEAR_W_NEG_DX": "0.25", "CLOT_SHEAR_W_SEP": "0.40", "CLOT_SHEAR_W_STASIS": "0.20", "CLOT_SHEAR_W_LGRAD": "0.15"},
            "stag40": {"CLOT_SHEAR_W_NEG_DX": "0.20", "CLOT_SHEAR_W_SEP": "0.15", "CLOT_SHEAR_W_STASIS": "0.40", "CLOT_SHEAR_W_LGRAD": "0.25", "CLOT_SHEAR_LSS_SI": "10"},
            "lgrad35": {"CLOT_SHEAR_W_NEG_DX": "0.25", "CLOT_SHEAR_W_SEP": "0.15", "CLOT_SHEAR_W_STASIS": "0.25", "CLOT_SHEAR_W_LGRAD": "0.35"},
            "lss10": {"CLOT_SHEAR_W_NEG_DX": "0.20", "CLOT_SHEAR_W_SEP": "0.15", "CLOT_SHEAR_W_STASIS": "0.45", "CLOT_SHEAR_W_LGRAD": "0.20", "CLOT_SHEAR_LSS_SI": "10"},
            "auto": {"CLOT_SHEAR_W_NEG_DX": "0.30", "CLOT_SHEAR_W_SEP": "0.20", "CLOT_SHEAR_W_STASIS": "0.30", "CLOT_SHEAR_W_LGRAD": "0.20", "CLOT_SHEAR_LSS_SI": "10", "CLOT_SHEAR_SIZE_MODE": "auto"},
        }
        env = _parse_rule_name("loc_prog_both_t20_s0_ndx25_inc40")
        env["CLOT_TEMPORAL_RULE_NAME"] = name
        env.update(_shear_loc_env.get(tag, {"CLOT_SHEAR_W_NEG_DX": "0.25"}))
        return env

    incub = re.match(r"^(.+)_inc(\d+)$", name)
    if incub:
        base_name, inc_tag = incub.groups()
        env = _parse_rule_name(base_name)
        env["CLOT_TEMPORAL_GLOBAL_ONSET"] = f"{float(inc_tag) / 100:.2f}"
        env["CLOT_TEMPORAL_RULE_NAME"] = name
        return env

    offramp = re.match(r"^offramp_off(\d+)_b(\d+)$", name)
    if offramp:
        off_tag, boost_tag = offramp.groups()
        env = _parse_rule_name("loc_prog_both_t20_s0_ndx25")
        env["CLOT_TEMPORAL_RULE_NAME"] = name
        env["CLOT_TEMPORAL_GLOBAL_ONSET"] = f"{float(off_tag) / 100:.2f}"
        env["CLOT_TEMPORAL_PROMOTION_BOOST"] = f"{float(boost_tag) / 10:.2f}"
        return env

    accum = re.match(r"^accum_(?:inc(\d+)_)?thr(\d+)_g(\d+)_sw(\d+)$", name)
    if accum:
        inc_tag, thr_tag, gain_tag, sw_tag = accum.groups()
        env["CLOT_TEMPORAL_RULE_KIND"] = "threshold_accum"
        env["CLOT_TEMPORAL_RULE_NAME"] = name
        env["CLOT_LOCALIZED_MODE"] = "wall_half"
        env["CLOT_LOCALIZED_TOP_FRAC"] = "0.20"
        env["CLOT_LOCALIZED_SKIP_ARC"] = "0.00"
        env["CLOT_TEMPORAL_ACCUM_THRESHOLD"] = f"{float(thr_tag) / 10:.2f}"
        env["CLOT_TEMPORAL_ACCUM_GAIN"] = f"{float(gain_tag) / 100:.2f}"
        env["CLOT_TEMPORAL_ACCUM_SPLIT_WALL"] = f"{float(sw_tag) / 100:.2f}"
        env["CLOT_TEMPORAL_ACCUM_SPLIT_LUMEN"] = "0.03"
        if inc_tag:
            env["CLOT_TEMPORAL_GLOBAL_ONSET"] = f"{float(inc_tag) / 100:.2f}"
        return env

    if name.startswith("hyb_"):
        m = re.match(
            r"hyb_(rank|prog)_(both|lower)_t(\d+)_s(\d+)_ndx(\d+)_sp(\d+)",
            name,
        )
        if not m:
            raise ValueError(f"cannot parse hybrid rule name: {name}")
        ktag, wtag, top, skip, ndx, spw = m.groups()
        env["CLOT_TEMPORAL_RULE_KIND"] = "ranked_onset" if ktag == "rank" else "progressive_topk"
        env["CLOT_TEMPORAL_RULE_NAME"] = name
        env["CLOT_TEMPORAL_START_FRAC"] = "0.05"
        env["CLOT_TEMPORAL_END_FRAC"] = "0.22"
        env["CLOT_TEMPORAL_POWER"] = "1.5"
        env["CLOT_TEMPORAL_ONSET_SPREAD"] = "0.55"
        env["CLOT_TEMPORAL_MIN_ONSET"] = "0.08"
        env["CLOT_LOCALIZED_MODE"] = "wall_half"
        env["CLOT_LOCALIZED_TOP_FRAC"] = f"{float(top) / 100:.2f}"
        env["CLOT_LOCALIZED_SKIP_ARC"] = f"{float(skip) / 100:.2f}"
        env["CLOT_LOCALIZED_SPECIES_WEIGHT"] = f"{float(spw) / 100:.2f}"
        env["CLOT_LOCALIZED_SPECIES_TIME"] = "t_out"
        return env

    if name.startswith("loc_"):
        m = re.match(r"loc_(rank|prog)_(both|lower)_t(\d+)_s(\d+)_ndx(\d+)", name)
        if m:
            ktag, wtag, top, skip, ndx = m.groups()
            env["CLOT_TEMPORAL_RULE_KIND"] = "ranked_onset" if ktag == "rank" else "progressive_topk"
            env["CLOT_TEMPORAL_RULE_NAME"] = name
            env["CLOT_TEMPORAL_START_FRAC"] = "0.05"
            env["CLOT_TEMPORAL_END_FRAC"] = "0.22"
            env["CLOT_TEMPORAL_POWER"] = "1.5"
            env["CLOT_TEMPORAL_ONSET_SPREAD"] = "0.55"
            env["CLOT_TEMPORAL_MIN_ONSET"] = "0.08"
            env["CLOT_LOCALIZED_MODE"] = "wall_half"
            env["CLOT_LOCALIZED_TOP_FRAC"] = f"{float(top) / 100:.2f}"
            env["CLOT_LOCALIZED_SKIP_ARC"] = f"{float(skip) / 100:.2f}"
            return env
        m2 = re.match(r"loc_(rank|prog)_arc4_t(\d+)", name)
        if m2:
            ktag, top = m2.groups()
            env["CLOT_TEMPORAL_RULE_KIND"] = "ranked_onset" if ktag == "rank" else "progressive_topk"
            env["CLOT_TEMPORAL_RULE_NAME"] = name
            env["CLOT_LOCALIZED_MODE"] = "arc_bins"
            env["CLOT_LOCALIZED_ARC_BINS"] = "4"
            env["CLOT_LOCALIZED_TOP_FRAC"] = f"{float(top) / 100:.2f}"
            env["CLOT_LOCALIZED_SKIP_ARC"] = "0.15"
            return env

    global_kinds = {
        "static_global": "static_spatial",
        "ranked_onset_global": "ranked_onset",
        "ranked_onset_global_wide": "ranked_onset",
        "prog_global_std": "progressive_topk",
        "prog_global_late": "progressive_topk",
        "hop_growth_global": "hop_growth",
        "neighbor_ac_global": "neighbor_ac",
    }
    if name in global_kinds:
        env["CLOT_TEMPORAL_RULE_KIND"] = global_kinds[name]
        env["CLOT_TEMPORAL_RULE_NAME"] = name
        env["CLOT_TEMPORAL_START_FRAC"] = "0.05"
        env["CLOT_TEMPORAL_END_FRAC"] = "0.22"
        env["CLOT_TEMPORAL_POWER"] = "1.5"
        return env

    raise ValueError(f"unknown rule name for promote: {name}")


def _render_ps1(env: dict[str, str], *, rule: str, source: str) -> str:
    lines = [
        f"# Auto-promoted from {source}",
        f"# winner rule: {rule}",
        "# Dot-source AFTER _clot_prior_rule_winner_env.ps1",
        "",
    ]
    clears = (
        "CLOT_LOCALIZED_MODE",
        "CLOT_LOCALIZED_TOP_FRAC",
        "CLOT_LOCALIZED_SKIP_ARC",
        "CLOT_LOCALIZED_ARC_BINS",
        "CLOT_LOCALIZED_SPECIES_WEIGHT",
        "CLOT_LOCALIZED_SPECIES_GT_Q",
        "CLOT_LOCALIZED_SPECIES_TIME",
        "CLOT_TEMPORAL_GLOBAL_ONSET",
        "CLOT_TEMPORAL_PROMOTION_BOOST",
        "CLOT_TEMPORAL_ACCUM_GAIN",
        "CLOT_TEMPORAL_ACCUM_THRESHOLD",
        "CLOT_TEMPORAL_ACCUM_SPLIT_WALL",
        "CLOT_TEMPORAL_ACCUM_SPLIT_LUMEN",
    )
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
        default="outputs/biochem/diagnostics/clot_hybrid_species_sweep.json",
        help="Sweep JSON with winner field",
    )
    ap.add_argument(
        "--out",
        default="scripts/_clot_architecture_winner_env.ps1",
    )
    ap.add_argument("--rule", default="", help="Override winner rule name")
    args = ap.parse_args()

    json_path = REPO / args.json
    if not json_path.is_file():
        print(f"[ERR] missing {json_path}", file=sys.stderr)
        return 1

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    winner = payload.get("winner") or {}
    rule = (args.rule or winner.get("rule") or "").strip()
    if not rule:
        print("[ERR] no winner rule in JSON", file=sys.stderr)
        return 1

    env = _parse_rule_name(rule)
    out_path = REPO / args.out
    out_path.write_text(_render_ps1(env, rule=rule, source=str(json_path)), encoding="utf-8")
    print(f"[OK] promoted {rule} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
