# ~4h species GNN s34 architecture + tuning sweep.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_species_gnn_s34_arch_sweep_4h.ps1"
#   powershell ... -DryRun
#   powershell ... -Legs fp_wall12,arch_delta_res -SkipCompleted

param(
    [string[]] $Legs = @(),
    [int] $Epochs = 22,
    [int] $EarlyStop = 12,
    [string] $InitCkpt = "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth",
    [switch] $SkipCompleted,
    [switch] $SkipTrain,
    [switch] $DryRun,
    [switch] $PromoteWinner
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$SweepDir = Join-Path $RepoRoot "outputs\biochem\sweep_species_gnn_s34_arch"
New-Item -ItemType Directory -Force -Path $SweepDir | Out-Null

$DefaultLegs = @(
    "ref_baseline", "fp_wall12", "temporal_wide", "closed_hard",
    "arch_delta_res", "arch_temp_offset", "arch_combo_res", "clot_score_w60"
)
$RunLegs = if ($Legs.Count -gt 0) { $Legs } else { $DefaultLegs }

Write-Host "[NEW] species GNN s34 arch sweep (~4h)" -ForegroundColor Cyan
Write-Host "[i] legs=$($RunLegs -join ',') epochs=$Epochs early_stop=$EarlyStop" -ForegroundColor DarkGray

if ($DryRun) {
    Write-Host "[DryRun] would run legs: $($RunLegs -join ', ')" -ForegroundColor Yellow
    exit 0
}

$pyArgs = @(
    "scripts/sweep_species_gnn_s34_arch.py",
    "--epochs", "$Epochs",
    "--early-stop", "$EarlyStop",
    "--init", $InitCkpt,
    "--legs", ($RunLegs -join ",")
)
if ($SkipCompleted) { $pyArgs += "--skip-completed" }
if ($SkipTrain) { $pyArgs += "--skip-train" }

Invoke-PythonRcCheck -Label "species gnn s34 sweep" -PyArgs $pyArgs

Invoke-PythonRcCheck -Label "summarize sweep" -PyArgs @(
    "scripts/summarize_species_gnn_s34_sweep.py"
)

if ($PromoteWinner) {
    Invoke-PythonRcCheck -Label "promote sweep winner" -PyArgs @(
        "scripts/promote_species_gnn_s34_sweep_winner.py"
    )
}

Write-Host "[OK] sweep -> $SweepDir" -ForegroundColor Green
Write-Host "[i] morning: python scripts/summarize_species_gnn_s34_sweep.py" -ForegroundColor DarkGray
