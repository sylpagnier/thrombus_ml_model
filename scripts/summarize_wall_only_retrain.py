"""Summarize step-3 wall-only retrain: A vs S (+ optional refs)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    "deploy_clot_f1",
    "deploy_clot_score",
    "deploy_clot_offwall_relaxed_f1",
    "deploy_clot_offwall_strict_f1",
    "deploy_clot_offwall_n_pred",
    "deploy_clot_offwall_n_pred_hop_ge2",
    "deploy_clot_offwall_n_gt_hop_ge2",
    "deploy_clot_offwall_strict_f1_hop_ge2",
)

FOCUS = ("patient001", "patient002", "patient006", "patient007", "patient010")


def _mean(report: dict) -> dict:
    simple = report.get("simple") or report
    return dict(simple.get("mean") or {})


def _per(report: dict) -> dict:
    simple = report.get("simple") or report
    return dict(simple.get("per_anchor") or {})


def _load_eval(path: str, label: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    report = json.loads(p.read_text(encoding="utf-8"))
    m = _mean(report)
    pa = _per(report)
    out = {
        "label": label,
        "path": str(p),
        "mean": {k: float(m.get(k, 0.0) or 0.0) for k in METRICS},
        "sum_ge2_pred": float(sum(float((pa.get(a) or {}).get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0) for a in pa)),
        "sum_ge2_gt": float(sum(float((pa.get(a) or {}).get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0) for a in pa)),
        "focus": {},
    }
    for a in FOCUS:
        r = pa.get(a) or {}
        out["focus"][a] = {
            "deploy_clot_f1": float(r.get("deploy_clot_f1", 0) or 0),
            "ge2_pred": float(r.get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0),
            "ge2_gt": float(r.get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0),
            "offwall_pred": float(r.get("deploy_clot_offwall_n_pred", 0) or 0),
        }
    return out


def _load_wall_only_compare(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    report = json.loads(p.read_text(encoding="utf-8"))
    mode = (report.get("modes") or {}).get("wall_only") or {}
    if not mode:
        return None
    pa = mode.get("per_anchor") or {}
    m = mode.get("mean") or {}
    return {
        "label": "unretrain_wall_only",
        "path": str(p),
        "mean": {k: float(m.get(k, 0.0) or 0.0) for k in METRICS if k in m or True},
        "sum_ge2_pred": float(mode.get("sum_ge2_pred", 0) or 0),
        "sum_ge2_gt": float(mode.get("sum_ge2_gt", 0) or 0),
        "focus": {
            a: {
                "deploy_clot_f1": float((pa.get(a) or {}).get("deploy_clot_f1", 0) or 0),
                "ge2_pred": float((pa.get(a) or {}).get("deploy_clot_offwall_n_pred_hop_ge2", 0) or 0),
                "ge2_gt": float((pa.get(a) or {}).get("deploy_clot_offwall_n_gt_hop_ge2", 0) or 0),
                "offwall_pred": float((pa.get(a) or {}).get("deploy_clot_offwall_n_pred", 0) or 0),
            }
            for a in FOCUS
            if a in pa
        },
    }


def _print_arm(arm: dict) -> None:
    m = arm["mean"]
    print(
        f"[{arm['label']}] mean_f1={m.get('deploy_clot_f1', 0):.3f} "
        f"score={m.get('deploy_clot_score', 0):.3f} "
        f"sum_ge2={arm['sum_ge2_pred']:.0f}/{arm['sum_ge2_gt']:.0f}",
        flush=True,
    )
    for a, r in arm.get("focus", {}).items():
        print(
            f"  {a}: f1={r['deploy_clot_f1']:.3f} ge2={r['ge2_pred']:.0f}/{r['ge2_gt']:.0f} "
            f"off={r['offwall_pred']:.0f}",
            flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-s", required=True)
    ap.add_argument("--legacy-probe", default="")
    ap.add_argument("--wall-only-unretrain", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = {
        "A": _load_eval(args.arm_a, "A_canonical"),
        "S": _load_eval(args.arm_s, "S_wall_only_retrain"),
    }
    if args.legacy_probe.strip():
        try:
            arms["legacy_D"] = _load_eval(args.legacy_probe.strip(), "legacy_D_Orig10")
        except Exception as exc:
            print(f"[WARN] legacy probe skip: {exc}", flush=True)
    if args.wall_only_unretrain.strip():
        u = _load_wall_only_compare(args.wall_only_unretrain.strip())
        if u is not None:
            arms["unretrain_wall_only"] = u

    print("=" * 72, flush=True)
    print("WALL-ONLY RETRAIN COMPARE", flush=True)
    for key in ("A", "legacy_D", "unretrain_wall_only", "S"):
        if key in arms:
            _print_arm(arms[key])

    s = arms["S"]
    a = arms["A"]
    ge2_001 = float((s.get("focus") or {}).get("patient001", {}).get("ge2_pred", 0) or 0)
    f1_s = float(s["mean"].get("deploy_clot_f1", 0) or 0)
    f1_a = float(a["mean"].get("deploy_clot_f1", 0) or 0)
    spray_002 = float((s.get("focus") or {}).get("patient002", {}).get("ge2_pred", 0) or 0)

    gates = {
        "opened_001": ge2_001 > 0.5,
        "mean_f1_near_A": f1_s >= (f1_a - 0.05),
        "spray_002_lt_50": spray_002 < 50.0,
    }
    if "unretrain_wall_only" in arms:
        u_f1 = float(arms["unretrain_wall_only"]["mean"].get("deploy_clot_f1", 0) or 0)
        gates["f1_up_vs_unretrain"] = f1_s > (u_f1 + 0.02)
        u_spray = float(
            (arms["unretrain_wall_only"].get("focus") or {}).get("patient002", {}).get("ge2_pred", 0) or 0
        )
        gates["spray_down_vs_unretrain"] = spray_002 < (u_spray - 10.0)

    print("\n=== GATES ===", flush=True)
    for k, v in gates.items():
        print(f"  {k}={v}", flush=True)
    verdict = "pass" if gates.get("opened_001") and gates.get("mean_f1_near_A") else "needs_work"
    if gates.get("opened_001") and gates.get("f1_up_vs_unretrain"):
        verdict = "pass_improved"
    print(f"[i] verdict={verdict}", flush=True)

    out = {
        "arms": arms,
        "gates": gates,
        "verdict": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[save] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
