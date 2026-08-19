"""PHASE 8: read the production COMSOL surface law out of the .mph and diff it against the repo.

``docs/PHASE7_FINDINGS.md`` 0 established that the ``.mph`` node tree is the authority.  This
script prints the pieces of that tree that define wall clot deposition -- the surface
reaction nodes, the analytic functions they call, and the parameters -- so the repo's
``integrate_mat_trajectory`` / ``ap_closure`` can be diffed against them term by term rather
than re-derived from exports.

    python scripts/diag_mph_surface_law.py
    python scripts/diag_mph_surface_law.py --mph comsol_models/phase2_template_wound.mph
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INTERESTING_FN = ("step", "Sat", "flc", "rect")


def load(mph: Path) -> tuple[dict, str]:
    raw = zipfile.ZipFile(mph).read("smodel.json").decode("utf8", errors="replace")
    return json.loads(raw), raw


def walk(o, out: list, path: str = "") -> None:
    if isinstance(o, dict):
        out.append((path, o))
        for k, v in o.items():
            walk(v, out, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, out, path + "[%d]" % i)


#: A COMSOL checkbox that is ON is stored as a settings entry with NO ``value`` key.  Mapping
#: it to Python ``None`` and printing it reads as the string "None" -- which is how you talk
#: yourself into believing ``Convection`` is disabled on ``tds2`` when it is enabled.  Keep
#: the distinction between "absent" and "set to something" visible.
ON = "<checkbox on / no value stored>"


def settings_of(node: dict) -> dict:
    return {s["name"]: s.get("value", ON) for s in node.get("settings", []) if "name" in s}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mph", default="comsol_models/phase2_template_nowound.mph")
    args = ap.parse_args()
    sm, raw = load(REPO / args.mph)
    nodes: list = []
    walk(sm, nodes)

    print("=== 1. SURFACE REACTION NODES (the clot source terms) ===")
    for p, n in nodes:
        if n.get("type") == "Surface_reactions":
            print("\n[%s]  tag=%s  active=%s  path=%s"
                  % (n.get("label"), n.get("tag"), n.get("isActive"),
                     n.get("modelEntityPath")))
            for k, v in settings_of(n).items():
                if k.startswith("J0") and not k.endswith("_src"):
                    print("   %-8s = %s" % (k, v))

    print("\n=== 2. ANALYTIC / PIECEWISE FUNCTIONS THEY CALL ===")
    for p, n in nodes:
        st = settings_of(n)
        ty = str(n.get("type") or "")
        if "function" not in ty.lower() and not any(
                k in st for k in ("expr", "funcname", "smooth")):
            continue
        print("\n[%s]  tag=%s  type=%s" % (n.get("label"), n.get("tag"), ty))
        for k in ("funcname", "expr", "args", "lowerlimit", "upperlimit", "smooth",
                  "smoothzone", "location", "fromlevel", "tolevel", "pieces", "from", "to"):
            if k in st:
                print("   %-12s %s" % (k, st[k]))

    print("\n=== 2b. STUDY TIME RANGE (what the ramp is relative to) ===")
    for p, n in nodes:
        st = settings_of(n)
        for k in ("tlist", "plist", "tunit"):
            if k in st and st[k]:
                print("   %-8s %-10s %s" % (n.get("tag"), k, str(st[k])[:160]))

    print("\n=== 3. PARAMETERS THE LAW USES ===")
    want = ("Da", "L", "gamma_m", "sgt", "lss", "k_rs", "k_as", "k_aa", "M_inf",
            "mu_max", "t_on", "t_start", "tau")
    seen = set()
    for p, n in nodes:
        nm = n.get("name")
        if nm in want and "value" in n and nm not in seen:
            seen.add(nm)
            print("   %-10s = %-28s %s" % (nm, n.get("value"), n.get("description") or ""))
    for nm in want:
        if nm not in seen:
            m = re.search(r'"name":"%s","scalarImag":"[^"]*","scalarReal":"([^"]*)","value":"([^"]*)"'
                          % re.escape(nm), raw)
            if m:
                print("   %-10s = %-28s (scalar %s)" % (nm, m.group(2), m.group(1)))

    print("\n=== 4. SPECIES INTERFACE DISCRETISATION + TRANSPORT ===")
    for p, n in nodes:
        if n.get("tag") in ("tds", "tds1", "tds2"):
            st = settings_of(n)
            print("   [%s] %s" % (n.get("tag"), n.get("label")))
            for k in ("order_concentration", "Convection", "ConvectiveTerm", "CrosswindType",
                      "Residual", "massIsotropicDiffusion", "Migration"):
                if k in st:
                    print("        %-22s %s" % (k, st[k]))

    print("\n=== 4b. THE CLOT VISCOSITY LAW -- what GT clot actually IS ===")
    print("   GT clot is not 'Mat >= crit'.  ``gt_clot_phi_at_time`` thresholds VISCOSITY")
    print("   GROWTH, relu(mu_eff(t) - mu_eff(0)), and mu_eff is COMSOL's own field.  So the")
    print("   clot target is whatever the .mph's viscosity expression says it is:")
    for p, n in nodes:
        st = settings_of(n)
        if st.get("expr", "").startswith("mu_b*"):
            print("      %-22s %s" % (n.get("label"), st["expr"]))
    print("   ...which has TWO routes.  Their step functions, by FUNCTION NAME (not tag):")
    print("      %-8s %-22s %10s %10s %8s %6s"
          % ("name", "label", "location", "smooth", "from", "to"))
    for p, n in nodes:
        if "Step" not in str(n.get("type") or ""):
            continue
        st = settings_of(n)
        nm = str(n.get("name") or "")
        if nm not in ("mu1", "mu2"):
            continue
        print("      %-8s %-22s %10s %10s %8s %6s"
              % (nm, n.get("label"), st.get("location", ""), st.get("smooth", ""),
                 st.get("from", ""), st.get("to", "")))

    print("\n=== 5. TRANSPORT PROPERTIES (does Mat move?) ===")
    for p, n in nodes:
        st = settings_of(n)
        lbl = str(n.get("label") or "")
        if not any(k in st for k in ("D_c", "u", "D")):
            continue
        if "Transport" not in str(n.get("type") or "") and "Properties" not in lbl \
                and "Convection" not in lbl:
            continue
        print("   [%s] type=%s" % (lbl, n.get("type")))
        for k in ("D_c", "D", "u", "minput_velocity_src", "u_src"):
            if k in st:
                print("        %-22s %s" % (k, st[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
