"""Run clot-phi sanity ladder: rule baseline, narrow neighbor shell, oracle μ probe."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from src.utils.paths import get_project_root


def _base_env() -> dict[str, str]:
    return {
        "CLOT_PHI_MASK_MODE": "neighbor",
        "CLOT_PHI_SOFT_LABELS": "0",
        "CLOT_PHI_BALANCED": "1",
        "CLOT_PHI_POS_WEIGHT_CAP": "15",
        "CLOT_PHI_EPOCHS": os.environ.get("CLOT_PHI_ABLATION_EPOCHS", "15"),
        "CLOT_PHI_LR": "3e-3",
        "CLOT_PHI_TIME_STRIDE": "2",
        "CLOT_PHI_VAL_ANCHOR": "patient007",
        "CLOT_PHI_ORACLE_MU": "0",
        "CLOT_PHI_RULE_BASELINE": "0",
        "CLOT_PHI_WALL_HOPS": "1",
        "CLOT_PHI_CLOT_HOPS": "2",
    }


def _run_leg(name: str, extra: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(_base_env())
    env.update(extra)
    print(f"\n=== {name} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "src.training.train_clot_phi_simple"],
        env=env,
        cwd=str(get_project_root()),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out, flush=True)
        raise RuntimeError(f"{name} failed exit={proc.returncode}")
    print(out, flush=True)
    row: dict = {"leg": name, "exit": proc.returncode}
    m = re.search(r"rule val dice=([\d.]+).*bce=([\d.]+).*logMAE=([\d.]+)", out)
    if m:
        row.update({"val_dice": float(m[1]), "val_bce": float(m[2]), "val_log_mae": float(m[3])})
    else:
        m2 = re.search(r"Best val dice=([\d.]+)", out)
        if m2:
            row["val_dice"] = float(m2[1])
        m3 = re.findall(
            r"Ep \d+ \| train bce=[\d.]+ dice=([\d.]+) logMAE=([\d.]+) \| val bce=([\d.]+) dice=([\d.]+) logMAE=([\d.]+)",
            out,
        )
        if m3:
            tr_d, tr_m, vb, vd, vm = m3[-1]
            row.update(
                {
                    "train_dice": float(tr_d),
                    "train_log_mae": float(tr_m),
                    "val_bce": float(vb),
                    "val_dice": float(vd),
                    "val_log_mae": float(vm),
                }
            )
    row["config"] = {k: env[k] for k in sorted(extra.keys()) if k in env}
    return row


def main() -> None:
    legs = [
        (
            "rule_baseline",
            {
                "CLOT_PHI_RULE_BASELINE": "1",
                "CLOT_PHI_WALL_HOPS": "1",
                "CLOT_PHI_CLOT_HOPS": "2",
            },
        ),
        (
            "narrow_shell_mlp",
            {
                "CLOT_PHI_WALL_HOPS": "1",
                "CLOT_PHI_CLOT_HOPS": "1",
                "CLOT_PHI_ORACLE_MU": "0",
            },
        ),
        (
            "oracle_log_mu_mlp",
            {
                "CLOT_PHI_WALL_HOPS": "1",
                "CLOT_PHI_CLOT_HOPS": "2",
                "CLOT_PHI_ORACLE_MU": "1",
            },
        ),
    ]
    rows = [_run_leg(name, cfg) for name, cfg in legs]
    out_path = get_project_root() / "outputs" / "biochem" / "clot_phi_ablations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n=== summary ===", flush=True)
    for r in rows:
        vd = r.get("val_dice", float("nan"))
        vm = r.get("val_log_mae", float("nan"))
        print(f"  {r['leg']:22s}  val Dice={vd:.3f}  val logMAE(mu)={vm:.4f}", flush=True)
    print(f"[OK]  wrote {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
