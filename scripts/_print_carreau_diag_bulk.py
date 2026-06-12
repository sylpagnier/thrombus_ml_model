import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/biochem/clot_trigger/t0_carreau_gelation_diag.json"
r = json.load(open(path))
for anchor in r["anchors"]:
    print(f"=== {anchor['anchor']} ratio_max={anchor['ratio_max']} ===")
    for trow in anchor["per_t"]:
        t = trow["t"]
        print(f"  t={t}")
        for k, stats in sorted(trow["models"].items()):
            b = stats.get("bulk", {})
            if not b.get("n"):
                continue
            gt_m = b.get("gt_mean", 0)
            pr_m = b.get("pred_mean", 1e-8)
            print(
                f"    {k:28s} n={b['n']:5d} r={b.get('pearson_r', float('nan')):7.3f} "
                f"gt={gt_m:.5f} pred={pr_m:.5f} gt/pred={gt_m/pr_m:.3f}"
            )
