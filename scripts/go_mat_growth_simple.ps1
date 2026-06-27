# Mat-only single-head pushforward on wall+3hop (triangle6 graph).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_growth_simple.ps1 -Fresh
#   powershell ... -Leg B_backbone -Fresh
#   powershell ... -Leg C_geom -Fresh -Epochs 50
#   powershell ... -Leg D_parity_single -Fresh
#   powershell ... -Fast -Leg D_parity_single -Fresh
#
# Ladder (all three legs + summary):
#   powershell ... -File .\scripts\go_mat_growth_ladder.ps1 -Fresh

param(
    [ValidateSet("A_random", "B_backbone", "C_geom", "D_parity_single", "E_dual_mat", "F_single_fimat", "G_dual_mat_neighbor_gate", "H_dual_mat_crit_focus", "I_dual_fimat_fi_aux", "J_dual_mat_neighbor_crit", "K_fimat_neighbor_gate", "L_fimat_geom_rich", "M_fimat_neighbor_geom_rich", "N_mat_geom_rich", "O_mat_neighbor_geom_rich", "P_mat_plain", "Q_mat_gate_sharp_fp", "R_mat_geom_gate_sharp_fp", "U_mat_frontier_only", "V_mat_frontier_geom", "W_mat_flow_stagnation", "X_mat_flow_seedfront", "Y_mat_tight_seed", "AB_mat_gelation_aux", "S_mat_frontier_nuc", "T_mat_frontier_sharp", "")]
    [string] $Leg = "",
    [int] $Epochs = 50,
    [int] $EarlyStop = 35,
    [int] $MaxWindows = 0,
    [string] $ValAnchor = "patient007",
    [switch] $Fast,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $AllAnchors,
    [string] $InitCkpt = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

# Fixed apples-to-apples fast preset.
$FAST_EPOCHS = 10
$FAST_EARLYSTOP = 6
$FAST_MAX_WINDOWS = 16
if ($Fast) {
    $Epochs = $FAST_EPOCHS
    $EarlyStop = $FAST_EARLYSTOP
    $MaxWindows = $FAST_MAX_WINDOWS
}

$UseLadder = [bool]$Leg.Trim()
if ($UseLadder) {
    $OutDir = "outputs/biochem/biochem_gnn/mat_growth_ladder/$Leg"
    $Ckpt = "$OutDir/species/best.pth"
} else {
    $OutDir = "outputs/biochem/biochem_gnn/mat_growth_simple"
    $Leg = "A_random"
    $Ckpt = "$OutDir/best.pth"
}
$CompareJson = "$OutDir/compare.json"

if ($Fresh) {
    if (Test-Path $Ckpt) { Remove-Item -Force $Ckpt }
    Remove-Item -Force (Join-Path $OutDir "species/best.json") -ErrorAction SilentlyContinue
    Remove-Item -Force (Join-Path $OutDir "species/train_log.jsonl") -ErrorAction SilentlyContinue
    Remove-Item -Force $CompareJson -ErrorAction SilentlyContinue
}

# Leg knobs (applied in trainer via --recipe + --init*).
$pyLeg = python -c @"
from src.biochem_gnn.mat_growth_simple import mat_growth_leg_spec
import json
print(json.dumps(mat_growth_leg_spec('$Leg').__dict__))
"@
$spec = $pyLeg | ConvertFrom-Json
Write-Host "[i] leg=$Leg : $($spec.label)" -ForegroundColor DarkGray
if ($Fast) {
    Write-Host "[i] FAST preset: epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows (fixed)" -ForegroundColor DarkGray
}

if (-not $EvalOnly) {
    $pyArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--recipe", "mat_growth_simple",
        "--leg", $Leg,
        "--init-mode", $spec.init_mode,
        "--out", $Ckpt
    )
    if ($spec.no_init -eq $true) { $pyArgs += "--no-init" }
    $initPath = if ($InitCkpt.Trim()) { $InitCkpt.Trim() } else { $spec.init_ckpt }
    if ($initPath -and -not $spec.no_init) {
        $pyArgs += @("--init", $initPath)
    }
    if ($AllAnchors) { $pyArgs += "--all-anchors" }
    else { $pyArgs += "--all-anchors" }
    Invoke-PythonRcCheck -Label "mat_growth $Leg train" -PyArgs $pyArgs
}

Invoke-PythonRcCheck -Label "mat_growth $Leg eval" -PyArgs @(
    "scripts/eval_mat_growth_simple.py",
    "--ckpt", $Ckpt,
    "--out", $CompareJson
)

Write-Host "[OK] leg=$Leg ckpt=$Ckpt compare=$CompareJson" -ForegroundColor Green
