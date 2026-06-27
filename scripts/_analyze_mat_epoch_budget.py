"""One-off: epoch curves for mat-growth architecture comparison budget."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_log(leg: str) -> list[dict]:
    p = ROOT / f"outputs/biochem/biochem_gnn/mat_growth_ladder/{leg}/species/train_log.jsonl"
    if not p.is_file():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: int(r["epoch"]))


def main() -> None:
    legs = ["P_mat_plain", "G_dual_mat_neighbor_gate"]
    checkpoints = [1, 3, 5, 8, 10, 11, 15, 16, 20, 21, 25, 30, 31, 37, 40]
    print("epoch,leg,cur_unroll,deploy_mat_t,val_growth_f1,val_mat_f1")
    for leg in legs:
        rows = load_log(leg)
        if not rows:
            continue
        by_ep = {int(r["epoch"]): r for r in rows}
        for ep in checkpoints:
            r = by_ep.get(ep)
            if not r:
                continue
            print(
                f"{ep},{leg},{r.get('cur_unroll')},{r.get('deploy_mat_f1', 0):.4f},"
                f"{r.get('val_growth_f1', 0):.4f},{r.get('val_mat_f1', 0):.4f}"
            )
        best = max(rows, key=lambda x: float(x.get("deploy_mat_f1", 0)))
        print(
            f"best,{leg},{best.get('cur_unroll')},{best.get('deploy_mat_f1', 0):.4f},"
            f"{best.get('val_growth_f1', 0):.4f},best_ep={best['epoch']}"
        )

    print("\n--- final eval (all anchors) ---")
    for leg in ["P_mat_plain", "G_dual_mat_neighbor_gate", "O_mat_neighbor_geom_rich", "N_mat_geom_rich"]:
        p = ROOT / f"outputs/biochem/biochem_gnn/mat_growth_ladder/{leg}/compare.json"
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        m = d["simple"]["mean"]
        meta = d["simple"]["meta"]
        print(
            f"{leg}: clot_f1={m['deploy_clot_f1']:.3f} "
            f"overpaint_per_gt={m.get('mat_overpaint_per_gt', 0):.3f} "
            f"best_ep={meta.get('best_epoch')}"
        )


if __name__ == "__main__":
    main()
