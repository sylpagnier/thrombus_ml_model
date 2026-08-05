import subprocess
import os
import json
import csv

# We will run WG_sweep_01 to WG_sweep_10 for the ablation test
NUM_RUNS = 10
OUTPUT_CSV = "outputs/biochem/eda/wall_gen/phase1_sweep_results.csv"

def main():
    print(f"[i] Starting Phase 1 Sweep for {NUM_RUNS} configurations...")
    
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
        
        holdout_json_path = f"outputs/biochem/eda/wall_gen/{leg_name}/eval_holdout_cold.json"
        
        if os.path.exists(holdout_json_path):
            print(f"[i] Leg {leg_name} already completed (eval JSON exists). Skipping!")
            continue

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
                    
                    if "simple" in data and "meta" in data["simple"]:
                        meta = data["simple"]["meta"]
                        deploy_score = meta.get("deploy_clot_score", 0.0)
                        strict_f1 = meta.get("deploy_clot_f1", 0.0)
                        rprec = meta.get("deploy_clot_relaxed_prec", 0.0)
                        rrec = meta.get("deploy_clot_relaxed_rec", 0.0)
                        val_state_f1 = meta.get("val_state_f1", 0.0)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a fast 1-epoch smoke test.")
    args = parser.parse_args()
    
    if args.smoke:
        print("[i] Running 1-epoch smoke test on WG_sweep_01...")
        cmd = [
            "pwsh", "-ExecutionPolicy", "Bypass",
            "scripts/go_wall_gen_probe.ps1",
            "-Leg", "WG_sweep_01",
            "-Epochs", "1",
            "-MaxWindows", "2"
        ]
        subprocess.run(cmd, check=True)
        print("[OK] Smoke test completed successfully.")
    else:
        main()
