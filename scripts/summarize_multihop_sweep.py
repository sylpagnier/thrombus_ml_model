"""Roll the multi-hop sweep outputs into one decision document."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.paths import get_project_root  # noqa: E402

# Windows consoles default to cp1252; a stray non-ASCII char must never crash
# the final report after hours of GPU work.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/biochem/eda/multihop_sweep")
    ap.add_argument("--baseline-f1", type=float, default=0.500)
    ap.add_argument("--baseline-mass", type=float, default=2.418)
    ap.add_argument("--baseline-distant-frac", type=float, default=0.971)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = get_project_root() / root

    L: list[str] = []
    L.append("# Multi-hop flow sweep — results\n")
    L.append(f"Baseline (WG_clotrich_nplus): **F1 {args.baseline_f1:.3f}**, "
             f"mass {args.baseline_mass:.3f}, distant-FP {args.baseline_distant_frac:.1%}\n")

    probe = _load(get_project_root() / "outputs/biochem/eda/probe_multihop.json")
    if probe:
        L.append("\n## Stage A — free probe\n")
        m = probe.get("means", {})
        L.append("| feature set | LOVO held-out AUC |")
        L.append("|---|---|")
        for k, v in m.items():
            L.append(f"| `{k}` | {v:.4f} |")
        L.append(f"\n- best set: `{probe.get('best_set')}`, lift vs hop1 "
                 f"**{probe.get('lift_vs_hop1', 0):+.4f}**")
        L.append(f"- oracle-sign mean {probe.get('oracle_sign_mean', 0):.4f} "
                 f"(stagnation {len(probe.get('stagnation_regime') or [])}, "
                 f"inverted {len(probe.get('inverted_regime') or [])})")
        L.append(f"- verdict: {probe.get('verdict')}\n")

    L.append("\n## Stage B/C — arms on the holdout\n")
    L.append("| arm | F1 | dF1 | mass | FP | FN | distant-FP frac | d_distant |")
    L.append("|---|---|---|---|---|---|---|---|")
    rows = {}
    for arm_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        arm = arm_dir.name
        ev = None
        for f in arm_dir.glob("eval_patient*.json"):
            if "challenge" in f.name:
                continue
            ev = _load(f)
            break
        if not ev:
            continue
        mean = (ev.get("simple") or {}).get("mean") or {}
        f1 = float(mean.get("deploy_clot_f1", 0))
        mass = float(mean.get("deploy_clot_mass_ratio", 0))
        fp = float(mean.get("deploy_clot_fp", 0))
        fn = float(mean.get("deploy_clot_fn", 0))
        geo = None
        for f in arm_dir.glob("fpgeo_*.json"):
            g = _load(f)
            if g:
                pa = g.get("per_anchor") or {}
                if pa:
                    geo = (list(pa.values())[0].get("geography") or {})
            break
        dfrac = float(geo.get("distant_frac", float("nan"))) if geo else float("nan")
        rows[arm] = {"f1": f1, "mass": mass, "fp": fp, "fn": fn, "distant_frac": dfrac}
        dd = dfrac - args.baseline_distant_frac if dfrac == dfrac else float("nan")
        L.append(
            f"| `{arm}` | {f1:.4f} | {f1 - args.baseline_f1:+.4f} | {mass:.3f} | "
            f"{fp:.0f} | {fn:.0f} | "
            + (f"{dfrac:.1%} | {dd:+.1%} |" if dfrac == dfrac else "n/a | n/a |")
        )

    L.append("\n## Verdict\n")
    if not rows:
        L.append("No arm produced an eval. Check `sweep_log.txt`.")
    else:
        best = max(rows, key=lambda k: rows[k]["f1"])
        b = rows[best]
        mh = rows.get("WG_multihop")
        ct = rows.get("WG_multihop_ctrl")
        L.append(f"Best arm: **`{best}`** — F1 {b['f1']:.4f} "
                 f"({b['f1'] - args.baseline_f1:+.4f} vs baseline)\n")
        if mh and ct:
            d = mh["f1"] - ct["f1"]
            L.append(f"**Feature-attributable delta** (multihop - control): **{d:+.4f}**\n")
            if d > 0.02 and mh["distant_frac"] == mh["distant_frac"] and \
                    mh["distant_frac"] < args.baseline_distant_frac - 0.15:
                L.append("=> Feature works AND the distant wrong-pocket shrank. "
                         "This is the mechanism confirmed, not just a threshold shift. "
                         "Scale the cohort and repeat.")
            elif d > 0.02:
                L.append("=> F1 improved but the distant-FP fraction did not fall much. "
                         "Gain may be a threshold/operating-point shift rather than the "
                         "wrong pocket being fixed. Inspect FP geography before scaling.")
            elif abs(d) <= 0.02:
                L.append("=> No feature-attributable gain. Receptive field is not the "
                         "limit; escalate to plan s4 Step 3 (explicit pocket ranking / "
                         "two-stage localise-then-grow).")
            else:
                L.append("=> Feature HURT. Do not pursue. Re-read the probe's inverted-regime "
                         "split — a single global feature may be fighting two clot regimes.")
        else:
            L.append("_Control arm missing — gain is not attributable to the feature alone._")

    txt = "\n".join(L) + "\n"
    if args.out:
        op = Path(args.out)
        if not op.is_absolute():
            op = get_project_root() / op
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
