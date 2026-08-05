import subprocess
import os
import json
import csv

# We will run WG_sweep_01 to WG_sweep_30
NUM_RUNS = 30
OUTPUT_CSV = "outputs/biochem/eda/wall_gen/sweep_results.csv"

def main():
    print(f"[i] Starting 8-hour hyperparameter sweep for {NUM_RUNS} configurations...")
    
    # Initialize CSV header if it doesn't exist
    if not os.path.exists(OUTPUT_CSV):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Leg", "deploy_clot_score", "strict_f1", "relaxed_prec", "relaxed_rec", "val_state_f1"])

    for i in range(1, NUM_RUNS + 1):
        leg_name = f"WG_sweep_{i:02d}"
        print(f"\n========================================")
        print(f"[i] Launching Sweep Leg {i}/{NUM_RUNS}: {leg_name}")
        print(f"========================================")
        
        # Call the probe
        cmd = [
            "pwsh", "-ExecutionPolicy", "Bypass",
            "scripts/go_wall_gen_probe.ps1",
            "-Leg", leg_name,
            "-Epochs", "30",
            "-MaxWindows", "24"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running leg {leg_name}. Continuing to next...")
            continue
            
        # Parse output JSON to append to CSV
        holdout_json_path = f"outputs/biochem/eda/wall_gen/{leg_name}/eval_holdout_cold.json"
        train_log_path = f"outputs/biochem/eda/wall_gen/{leg_name}/train_log.jsonl"
        
        deploy_score = 0.0
        strict_f1 = 0.0
        rprec = 0.0
        rrec = 0.0
        val_state_f1 = 0.0
        
        if os.path.exists(holdout_json_path):
            try:
                with open(holdout_json_path, "r") as f:
                    data = json.load(f)
                    
                    # Compute mean over anchors for deploy metrics
                    patient_scores = []
                    for patient, metrics in data.items():
                        if "deploy_metrics" in metrics:
                            patient_scores.append(metrics["deploy_metrics"])
                    
                    if patient_scores:
                        deploy_score = sum(p.get("deploy_clot_score", 0.0) for p in patient_scores) / len(patient_scores)
                        strict_f1 = sum(p.get("deploy_clot_f1", 0.0) for p in patient_scores) / len(patient_scores)
                        rprec = sum(p.get("deploy_clot_relaxed_prec", 0.0) for p in patient_scores) / len(patient_scores)
                        rrec = sum(p.get("deploy_clot_relaxed_rec", 0.0) for p in patient_scores) / len(patient_scores)
            except Exception as e:
                print(f"[!] Could not parse {holdout_json_path}: {e}")
                
        # Parse train_log.jsonl to get the val_state_f1
        if os.path.exists(train_log_path):
            try:
                with open(train_log_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = json.loads(lines[-1])
                        val_state_f1 = last_line.get("val_state_f1", 0.0)
            except Exception as e:
                print(f"[!] Could not parse {train_log_path}: {e}")
                
        # Append to CSV
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([leg_name, deploy_score, strict_f1, rprec, rrec, val_state_f1])
            
        print(f"[OK] {leg_name} results -> Score: {deploy_score:.4f} | F1: {strict_f1:.4f} | val_state_f1: {val_state_f1:.4f}")

if __name__ == "__main__":
    main()
