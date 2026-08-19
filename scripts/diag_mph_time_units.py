"""PHASE 8: what time unit does the COMSOL surface-source ramp live in?

``J0_*`` is multiplied by ``step2t(t)`` in every surface reaction node.  The repo models
that as a logistic at ``surface_time_gate_s = 12`` with the pack's ``t`` in SECONDS, where
the pack's first sample is already t = 150 s -- so the repo's ramp is identically 1 over the
whole rollout and contributes nothing.  If COMSOL's ramp is actually in hours it spans the
entire 30000 s simulation and is a first-order timing term the repo is missing.

Prints the study time list, the declared time unit, and every Step/Ramp function with its
location and transition width so the two can be compared directly.

    python scripts/diag_mph_time_units.py
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mph", default="comsol_models/phase2_template_nowound.mph")
    args = ap.parse_args()
    raw = zipfile.ZipFile(REPO / args.mph).read("smodel.json").decode("utf8", errors="replace")

    print("=== 1. ANY 'range(' / time-list-looking settings ===")
    seen = set()
    for m in re.finditer(r'"value":"(range\([^"]*\)|[0-9][^"]{0,80})"', raw):
        v = m.group(1)
        if "range(" not in v:
            continue
        if v in seen:
            continue
        seen.add(v)
        s = max(0, m.start() - 260)
        print("   %-40s   ...%s" % (v, raw[s:m.start()][-170:]))

    print("\n=== 2. TIME-UNIT DECLARATIONS ===")
    for key in ("tunit", "timeunit", "unit", "lengthunit"):
        for m in re.finditer(r'"name":"%s","value":"([^"]*)"' % key, raw):
            print("   %-12s %s" % (key, m.group(1)))

    print("\n=== 3. STEP FUNCTIONS, and what a 'location' means in each unit ===")
    print("   %-8s %-26s %10s %10s %8s %6s" % ("tag", "label", "location", "smooth",
                                               "from", "to"))
    for m in re.finditer(r'"tag":"(step\d+)"', raw):
        seg = raw[max(0, m.start() - 2600):m.start() + 200]
        lbl = re.findall(r'"label":"([^"]*)"', seg)
        get = lambda k: (re.findall(r'"name":"%s","value":"([^"]*)"' % k, seg) or [""])[-1]
        print("   %-8s %-26s %10s %10s %8s %6s"
              % (m.group(1), (lbl[-1] if lbl else "")[:26], get("location"), get("smooth"),
                 get("from"), get("to")))

    print("\n=== 4. WHERE step2t / step4t ARE REFERENCED ===")
    for name in ("step2t", "step4t", "step1t", "step3t", "Act_step"):
        hits = [m.start() for m in re.finditer(re.escape(name), raw)]
        print("   %-10s %d references" % (name, len(hits)))
        for h in hits[:2]:
            seg = raw[max(0, h - 150):h + 90]
            print("        ...%s" % seg[-190:])

    print("\n=== 5. FUNCTION NAME vs TAG (COMSOL allows them to differ) ===")
    for m in re.finditer(r'"name":"funcname","value":"([^"]*)"', raw):
        seg = raw[max(0, m.start() - 2000):m.start()]
        tag = (re.findall(r'"tag":"([^"]*)"', seg) or [""])[-1]
        lbl = (re.findall(r'"label":"([^"]*)"', seg) or [""])[-1]
        print("   funcname=%-12s tag=%-10s label=%s" % (m.group(1), tag, lbl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
