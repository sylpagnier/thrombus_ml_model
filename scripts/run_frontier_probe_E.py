"""After main ladder: frontier-route re-eval of best growth (physics spray pivot)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h"
WALL = REPO / "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
ORIG10 = (
    "patient001,patient002,patient003,patient004,patient005,"
    "patient006,patient007,patient008,patient010,patient011"
)


def main() -> int:
    state = json.loads((ROOT / "state.json").read_text(encoding="utf-8"))
    growth = state.get("best_growth") or str(ROOT / "growth_C" / "best.pth")
    a_mean = float((((state.get("phases") or {}).get("A") or {}).get("wall") or {}).get("mean_f1") or 0.78)
    out = ROOT / "probe_E_frontier.json"
    env = dict(os.environ)
    env["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    env["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = "1"
    cmd = [
        sys.executable,
        "-u",
        str(REPO / "scripts" / "eval_mat_growth_simple.py"),
        "--ckpt",
        str(WALL),
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out",
        str(out),
        "--anchors",
        ORIG10,
        "--offwall-ckpt",
        str(growth),
        "--two-model-route",
        "frontier",
        "--two-model-frontier-hops",
        "2",
    ]
    print("[RUN] frontier probe", growth, flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    if rc != 0:
        return rc
    # reuse gate helper
    sys.path.insert(0, str(REPO / "scripts"))
    from run_wall_lumen_target_9h import _gate_from_eval  # noqa: E402

    gate = _gate_from_eval(out, a_mean_f1=a_mean)
    state.setdefault("phases", {})["E"] = gate
    state.setdefault("notes", []).append(
        f"E frontier: F1={gate['mean_f1']:.3f} 001={gate['focus']['patient001']['ge2_pred']:.0f} "
        f"007={gate['focus']['patient007']['ge2_pred']:.0f} "
        f"spray002={gate['focus']['patient002']['ge2_pred']:.0f} target={gate['target_hit']}"
    )
    if gate["target_hit"]:
        state["target_hit"] = True
        state["best_route"] = "frontier"
    (ROOT / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ROOT / "probe_E_frontier_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"target_hit": gate["target_hit"], "mean_f1": gate["mean_f1"], "focus": gate["focus"]}, indent=2))
    return 0 if gate["target_hit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
