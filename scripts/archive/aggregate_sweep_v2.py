import os
import json
import csv
import glob

def main():
    sweep_dir = "outputs/biochem/eda/wall_gen_sweep_v2"
    output_csv = os.path.join(sweep_dir, "phase1_sweep_results.csv")
    
    legs = glob.glob(os.path.join(sweep_dir, "WG_sweep_*"))
    legs.sort()
    
    rows = []
    for leg_dir in legs:
        leg_name = os.path.basename(leg_dir)
        json_path = os.path.join(leg_dir, "eval_holdout_cold.json")
        if not os.path.exists(json_path):
            continue
            
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                
            # It should be the simple vs baseline diagnostic format
            if "simple" in data and "mean" in data["simple"]:
                mean_stats = data["simple"]["mean"]
                score = mean_stats.get("deploy_clot_score", 0.0)
                f1 = mean_stats.get("deploy_clot_f1", 0.0)
                rprec = mean_stats.get("deploy_clot_offwall_relaxed_prec", 0.0)
                rrec = mean_stats.get("deploy_clot_offwall_relaxed_rec", 0.0)
                # vstate isn't in mean, let's grab it from meta if present
                vstate = data.get("simple", {}).get("meta", {}).get("val_state_f1", 0.0)
                
                rows.append([leg_name, score, f1, rprec, rrec, vstate])
            # Or if it's the old predict_clot_scores.py format
            else:
                patient_scores = []
                for patient, metrics in data.items():
                    if isinstance(metrics, dict) and "deploy_metrics" in metrics:
                        patient_scores.append(metrics["deploy_metrics"])
                
                if patient_scores:
                    score = sum(p.get("deploy_clot_score", 0.0) for p in patient_scores) / len(patient_scores)
                    f1 = sum(p.get("deploy_clot_f1", 0.0) for p in patient_scores) / len(patient_scores)
                    rprec = sum(p.get("deploy_clot_relaxed_prec", 0.0) for p in patient_scores) / len(patient_scores)
                    rrec = sum(p.get("deploy_clot_relaxed_rec", 0.0) for p in patient_scores) / len(patient_scores)
                    vstate = 0.0
                    rows.append([leg_name, score, f1, rprec, rrec, vstate])
        except Exception as e:
            print(f"[!] Error parsing {json_path}: {e}")
            
    if rows:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Leg", "deploy_clot_score", "strict_f1", "relaxed_prec", "relaxed_rec", "val_state_f1"])
            writer.writerows(rows)
        print(f"[OK] Wrote {len(rows)} leg results to {output_csv}")

if __name__ == "__main__":
    main()
