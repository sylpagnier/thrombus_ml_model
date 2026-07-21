import json
from pathlib import Path

legs = ["WC_v7_fresh_canonical", "WC_v7_clot_phi_mse", "WC_v7_high_precision"]
root = Path("outputs/biochem/biochem_gnn")

metrics = [
    "clot_fp_median",
    "clot_fp_p90",
    "clot_fn_median",
    "deploy_clot_f1",
    "deploy_clot_score"
]

for m in metrics:
    print("\n" + "=" * 80)
    print(f"METRIC: {m}")
    print(f"{'LEG':<25} | {'BASELINE':<15} | {'LEG VAL':<15} | {'DELTA':<15}")
    print("=" * 80)
    for leg in legs:
        path = root / leg / "compare.json"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        b = data["baseline"]["mean"].get(m, 0.0)
        s = data["simple"]["mean"].get(m, 0.0)
        d = data["delta_simple_minus_baseline"].get(m, 0.0)
        print(f"{leg:<25} | {b:15.3f} | {s:15.3f} | {d:+15.3f}")
