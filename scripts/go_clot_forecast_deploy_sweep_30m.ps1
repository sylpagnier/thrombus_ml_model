# ~30 min deploy-health sweep: one-step R2a+ variants ranked by clot_shape.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_deploy_sweep_30m.ps1"
#   powershell ... -Legs deploy_base,mesh_aux,target_mask -Epochs 8
#
# Default 5 legs x 8 epochs (~6 min/leg GPU; ~30 min total).

param(
    [string] $Legs = "",
    [int] $Epochs = 8,
    [switch] $SkipSummary
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
. (Join-Path $PSScriptRoot "_clot_forecast_r2a_plus_base.ps1")
$env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"

$SweepRoot = "outputs/biochem/sweep_clot_forecast_deploy_30m"
$env:CLOT_PHI_SWEEP_DIR = $SweepRoot
$env:CLOT_PHI_EPOCHS = "$Epochs"

# Leg -> env overrides (all: one-step phi + log(mu@t_in) + fixed mu_solid)
$AllLegs = [ordered]@{
    deploy_base = @{
        ForecastMask = "deploy_pred"
        SoftLabels = "1"
        MeshAux = "0"
        MeshBulk = "0"
        Hidden = "32"
        Depth = "2"
    }
    hard_labels = @{
        ForecastMask = "deploy_pred"
        SoftLabels = "0"
        MeshAux = "0"
        MeshBulk = "0"
        Hidden = "32"
        Depth = "2"
    }
    mesh_aux = @{
        ForecastMask = "deploy_pred"
        SoftLabels = "1"
        MeshAux = "0.6"
        MeshBulk = "0.15"
        Hidden = "32"
        Depth = "2"
    }
    mesh_aux_hard = @{
        ForecastMask = "deploy_pred"
        SoftLabels = "0"
        MeshAux = "0.8"
        MeshBulk = "0.2"
        Hidden = "32"
        Depth = "2"
    }
    target_mask = @{
        ForecastMask = "target"
        SoftLabels = "1"
        MeshAux = "0.3"
        MeshBulk = "0"
        Hidden = "32"
        Depth = "2"
    }
    input_mask = @{
        ForecastMask = "input"
        SoftLabels = "1"
        MeshAux = "0.3"
        MeshBulk = "0"
        Hidden = "32"
        Depth = "2"
    }
    wide_mlp = @{
        ForecastMask = "deploy_pred"
        SoftLabels = "1"
        MeshAux = "0.5"
        MeshBulk = "0.1"
        Hidden = "48"
        Depth = "3"
        Dropout = "0.10"
    }
}

function Set-DeploySweepLegEnv {
    param([string]$LegName, $Cfg)
    $env:CLOT_PHI_SWEEP_LEG = $LegName
    $env:CLOT_FORECAST_MASK = "$($Cfg.ForecastMask)"
    $env:CLOT_PHI_SOFT_LABELS = "$($Cfg.SoftLabels)"
    $env:CLOT_PHI_MESH_AUX_LAMBDA = "$($Cfg.MeshAux)"
    $env:CLOT_PHI_MESH_BULK_LAMBDA = "$($Cfg.MeshBulk)"
    $env:CLOT_PHI_HIDDEN = "$($Cfg.Hidden)"
    $env:CLOT_PHI_MLP_DEPTH = "$($Cfg.Depth)"
    if ($Cfg.Dropout) { $env:CLOT_PHI_DROPOUT = "$($Cfg.Dropout)" } else { $env:CLOT_PHI_DROPOUT = "0.15" }
}

function Invoke-DeploySweepLeg {
    param([string]$LegName, $Cfg)
    $legDir = Join-Path $RepoRoot "$SweepRoot\$LegName"
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "multi_anchor.jsonl")

    Set-DeploySweepLegEnv -LegName $LegName -Cfg $Cfg
    Write-Host ""
    Write-Host "[NEW] leg=$LegName mask=$($Cfg.ForecastMask) soft=$($Cfg.SoftLabels) mesh_aux=$($Cfg.MeshAux) mesh_bulk=$($Cfg.MeshBulk) hidden=$($Cfg.Hidden) depth=$($Cfg.Depth) ep=$Epochs" -ForegroundColor Cyan
    python -m src.training.train_clot_phi_simple
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $ckpt = Join-Path $legDir "clot_phi_best.pth"
    if (Test-Path $ckpt) {
        python scripts/eval_clot_phi_multi_anchor.py --checkpoint $ckpt --out (Join-Path $legDir "multi_anchor.jsonl")
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } else {
        Write-Host "[WARN] no ckpt for leg=$LegName (scorer may have rejected all epochs)" -ForegroundColor Yellow
    }
}

$runList = @()
if ($Legs.Trim()) {
    $runList = $Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
    # Default ~30 min: 5 legs
    $runList = @("deploy_base", "hard_labels", "mesh_aux", "target_mask", "mesh_aux_hard")
}

Write-Host "[NEW] clot forecast deploy sweep (~${Epochs}ep/leg, $($runList.Count) legs) -> $SweepRoot" -ForegroundColor Cyan

foreach ($name in $runList) {
    if (-not $AllLegs.Contains($name)) {
        throw "Unknown leg '$name'. Valid: $($AllLegs.Keys -join ', ')"
    }
    Invoke-DeploySweepLeg -LegName $name -Cfg $AllLegs[$name]
}

if (-not $SkipSummary) {
    python (Join-Path $PSScriptRoot "summarize_clot_forecast_deploy_sweep.py") --sweep-dir $SweepRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[OK]  deploy sweep done -> $SweepRoot/summary.jsonl" -ForegroundColor Green
