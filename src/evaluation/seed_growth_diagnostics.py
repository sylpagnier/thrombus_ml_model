"""Seed / front / FP-FN diagnostic panel for wall-gen / sparse-commitment sweeps.

Maps holdout metrics into a coarse failure mode so sweeps can decide the next knob:
under-seed / stalled front vs overspray vs balanced, without re-deriving thresholds each time.

Also owns the locked wall-gen **gate rule** (patient020): primary ``deploy_clot_f1``,
mass band, no score-alone promote when mass is starved.
"""

from __future__ import annotations

from typing import Any, Mapping

# Locked wall-gen gate (patient020). Primary metric is strict clot F1 -- not guiding score.
# The fh=2 / mass~0.18 run showed score can rise while F1 falls (precision mirage).
WALL_GEN_GATE_PRIMARY = "deploy_clot_f1"
WALL_GEN_GATE_MASS_LO = 0.8
WALL_GEN_GATE_MASS_HI = 1.3
WALL_GEN_GATE_MASS_STARVE = 0.5  # never promote on score alone below this
WALL_GEN_FLOOR_CLOT_F1 = 0.37
WALL_GEN_FLOOR_CLOT_SCORE = 0.35
WALL_GEN_FLOOR_FN = 67.0
WALL_GEN_WIN_CLOT_F1 = 0.45
WALL_GEN_STRETCH_CLOT_F1 = 0.50


def passes_wall_gen_gate(
    row: Mapping[str, Any],
    *,
    floor_f1: float = WALL_GEN_FLOOR_CLOT_F1,
    floor_fn: float = WALL_GEN_FLOOR_FN,
    mass_lo: float = WALL_GEN_GATE_MASS_LO,
    mass_hi: float = WALL_GEN_GATE_MASS_HI,
    mass_starve: float = WALL_GEN_GATE_MASS_STARVE,
    require_lift: bool = False,
) -> tuple[bool, str]:
    """Return (ok, reason) for promoting / claiming a wall-gen win on patient020.

    Rules (locked):
    1. Primary = ``deploy_clot_f1`` (not score).
    2. ``mass_ratio`` in ``[mass_lo, mass_hi]`` (sane volume).
    3. Never promote when ``mass < mass_starve`` even if score looks great.
    4. ``clot_fn`` must not rise vs floor when counts are present.
    """
    f1 = float(row.get("deploy_clot_f1", 0.0) or 0.0)
    score = float(row.get("deploy_clot_score", 0.0) or 0.0)
    mass = float(row.get("deploy_clot_mass_ratio", 0.0) or 0.0)
    fn = row.get("deploy_clot_fn", row.get("clot_fn_median", row.get("clot_fn")))
    fn_v = float(fn) if fn is not None else None

    if mass < float(mass_starve):
        return False, (
            f"mass_starve mass={mass:.3f}<{mass_starve:.2f} "
            f"(score={score:.3f} is a precision mirage; do not promote)"
        )
    if mass < float(mass_lo) or mass > float(mass_hi):
        return False, f"mass_out_of_band mass={mass:.3f} not in [{mass_lo:.2f},{mass_hi:.2f}]"
    if require_lift and f1 < float(floor_f1):
        return False, f"f1_below_floor f1={f1:.3f}<{floor_f1:.3f}"
    if fn_v is not None and fn_v > float(floor_fn) * 1.05 + 1.0:
        return False, f"fn_rose fn={fn_v:.0f}>floor~{floor_fn:.0f}"
    return True, (
        f"ok primary=f1={f1:.3f} score={score:.3f} mass={mass:.3f}"
        + (f" fn={fn_v:.0f}" if fn_v is not None else "")
    )


def classify_seed_growth_mode(row: Mapping[str, Any]) -> str:
    """Coarse mode from Mat seed/front diagnostics + clot FP/FN (when present).

    Heuristics (deploy-faithful cold eval):
    - underseed: FN-heavy and/or slow front / empty-wrong seed
    - overspray: FP-heavy and/or high mass / overpaint with weak seed precision
    - balanced: neither signature dominates
    """
    seed_prec = float(row.get("mat_seed_prec", 0.0) or 0.0)
    seed_count = float(row.get("mat_seed_count", 0.0) or 0.0)
    front_speed = float(row.get("mat_front_speed_ratio", 0.0) or 0.0)
    overpaint = float(row.get("mat_overpaint_frac", 0.0) or 0.0)
    mass = float(row.get("deploy_clot_mass_ratio", 0.0) or 0.0)
    offwall = float(row.get("deploy_clot_offwall_strict_f1", 0.0) or 0.0)

    fp = row.get("deploy_clot_fp", row.get("clot_fp_median", row.get("clot_fp")))
    fn = row.get("deploy_clot_fn", row.get("clot_fn_median", row.get("clot_fn")))
    fp_v = float(fp) if fp is not None else None
    fn_v = float(fn) if fn is not None else None

    under = 0
    over = 0

    if seed_prec < 0.15 and seed_count <= 2.0:
        under += 2
    elif seed_prec < 0.25:
        under += 1

    if front_speed < 0.35:
        under += 2
    elif front_speed < 0.55:
        under += 1

    if offwall <= 1e-6:
        under += 1

    if fn_v is not None and fp_v is not None:
        if fn_v >= 1.5 * max(fp_v, 1.0) and fn_v >= 20.0:
            under += 2
        if fp_v >= 1.5 * max(fn_v, 1.0) and fp_v >= 20.0:
            over += 2
        elif fp_v >= 40.0:
            over += 1

    if mass >= 1.8 or overpaint >= 0.12:
        over += 2
    elif mass >= 1.4 or overpaint >= 0.08:
        over += 1

    if under >= over + 2 and under >= 2:
        return "underseed"
    if over >= under + 2 and over >= 2:
        return "overspray"
    if under >= 2 and over >= 2:
        return "mixed"
    return "balanced"


def seed_growth_diagnostic_panel(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compact panel for JSON reports and ASCII console summaries."""
    mode = classify_seed_growth_mode(row)
    panel = {
        "mode": mode,
        "mat_seed_prec": float(row.get("mat_seed_prec", 0.0) or 0.0),
        "mat_seed_count": float(row.get("mat_seed_count", 0.0) or 0.0),
        "mat_front_prec": float(row.get("mat_front_prec", 0.0) or 0.0),
        "mat_front_speed_ratio": float(row.get("mat_front_speed_ratio", 0.0) or 0.0),
        "mat_overpaint_frac": float(row.get("mat_overpaint_frac", 0.0) or 0.0),
        "deploy_clot_f1": float(row.get("deploy_clot_f1", 0.0) or 0.0),
        "deploy_clot_score": float(row.get("deploy_clot_score", 0.0) or 0.0),
        "deploy_clot_mass_ratio": float(row.get("deploy_clot_mass_ratio", 0.0) or 0.0),
        "deploy_clot_offwall_strict_f1": float(row.get("deploy_clot_offwall_strict_f1", 0.0) or 0.0),
        "deploy_wall_score": float(row.get("deploy_wall_score", 0.0) or 0.0),
    }
    for key in (
        "deploy_clot_fp",
        "deploy_clot_fn",
        "clot_fp_median",
        "clot_fn_median",
        "clot_fp_p90",
        "clot_fn_p90",
        "clot_fp_early_mean",
    ):
        if key in row and row[key] is not None:
            panel[key] = float(row[key])

    if mode == "underseed":
        panel["hint"] = (
            "underseed: do not chase score with hard fh/topk; "
            "prefer closed-loop FT or front-new aux from WG_prec_iter floor"
        )
    elif mode == "overspray":
        panel["hint"] = (
            "overspray / distant FP: prefer physical_fp_gating FT (not hard frontier mask)"
        )
    elif mode == "mixed":
        panel["hint"] = "mixed: run fp_geography viz; physfp if distant FPs, cloop if adjacent"
    else:
        panel["hint"] = (
            "balanced volume: precision ceiling -- viz FP geography then one physfp/cloop FT"
        )
    gate_ok, gate_reason = passes_wall_gen_gate(row)
    panel["gate_ok"] = bool(gate_ok)
    panel["gate_reason"] = str(gate_reason)
    return panel


def format_seed_growth_panel(panel: Mapping[str, Any], *, label: str = "") -> str:
    """One-line ASCII summary for PowerShell-safe console output."""
    tag = f" {label}" if label else ""
    mode = str(panel.get("mode", "?"))
    return (
        f"[diag{tag}] mode={mode} "
        f"seed_prec={float(panel.get('mat_seed_prec', 0.0)):.3f} "
        f"seed_n={float(panel.get('mat_seed_count', 0.0)):.1f} "
        f"front_spd={float(panel.get('mat_front_speed_ratio', 0.0)):.3f} "
        f"overpaint={float(panel.get('mat_overpaint_frac', 0.0)):.3f} "
        f"clot_f1={float(panel.get('deploy_clot_f1', 0.0)):.3f} "
        f"mass={float(panel.get('deploy_clot_mass_ratio', 0.0)):.3f} "
        f"offwall={float(panel.get('deploy_clot_offwall_strict_f1', 0.0)):.3f} "
        f"| {panel.get('hint', '')}"
    )
