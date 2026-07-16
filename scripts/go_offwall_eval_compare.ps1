# Decoupled Off-Wall Clot Growth Model Comparison Script
# Evaluates baseline WC_v6_closed_loop_eval vs combined (WC_v6 + blurring off-wall model)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$Ckpt_v6 = "outputs/biochem/biochem_gnn/mat_growth_ladder/WC_v6_closed_loop_eval/species/best.pth"
$Ckpt_blur = "outputs/biochem/offwall_model/best_blurring.pth"

if (-not (Test-Path $Ckpt_v6)) {
    throw "baseline checkpoint not found at: $Ckpt_v6"
}
if (-not (Test-Path $Ckpt_blur)) {
    throw "off-wall blurring checkpoint not found at: $Ckpt_blur"
}

$Out_Baseline = "outputs/biochem/offwall_model/eval_baseline.json"
$Out_Combined = "outputs/biochem/offwall_model/eval_combined.json"

Write-Host "[i] Step 1: Evaluating baseline WC_v6_closed_loop_eval..." -ForegroundColor Cyan
$env:SPECIES_TWO_MODEL_MODE = "0"
if ($env:SPECIES_OFFWALL_MODEL_CKPT) { Remove-Item env:SPECIES_OFFWALL_MODEL_CKPT }

Invoke-PythonRcCheck -Label "baseline eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $Ckpt_v6,
    "--out", $Out_Baseline
)

Write-Host "[i] Step 2: Evaluating combined model (v6 + blurring off-wall model)..." -ForegroundColor Cyan
$env:SPECIES_TWO_MODEL_MODE = "1"
$env:SPECIES_OFFWALL_MODEL_CKPT = $Ckpt_blur

Invoke-PythonRcCheck -Label "combined eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $Ckpt_v6,
    "--out", $Out_Combined
)

# Summarize results
python -c @"
import json
with open('$Out_Baseline') as f:
    base = json.load(f)
with open('$Out_Combined') as f:
    comb = json.load(f)

print('\n=============================================================')
print('           OFF-WALL CLOT GROWTH MODEL COMPARISON')
print('=============================================================')
print(f'%-30s | %-12s | %-12s' % ('Metric', 'Baseline v6', 'Combined Model'))
print('-' * 61)

metrics = [
    'deploy_clot_score',
    'deploy_clot_f1',
    'deploy_clot_offwall_relaxed_f1',
    'deploy_clot_offwall_strict_f1',
    'deploy_clot_offwall_n_pred',
    'deploy_clot_offwall_n_gt',
]

for m in metrics:
    v_base = base.get('cohort_mean', {}).get(m, base.get(m, 'n/a'))
    v_comb = comb.get('cohort_mean', {}).get(m, comb.get(m, 'n/a'))
    
    if isinstance(v_base, float):
        s_base = f'{v_base:.4f}'
    else:
        s_base = str(v_base)
        
    if isinstance(v_comb, float):
        s_comb = f'{v_comb:.4f}'
    else:
        s_comb = str(v_comb)
        
    print(f'%-30s | %-12s | %-12s' % (m, s_base, s_comb))
print('=============================================================')
"@

# Clean up env
$env:SPECIES_TWO_MODEL_MODE = "0"
Remove-Item env:SPECIES_OFFWALL_MODEL_CKPT -ErrorAction SilentlyContinue
