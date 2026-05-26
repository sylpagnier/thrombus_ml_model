# MLP trunk architecture sweep for wall-local clot phi (patient007 val).
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_clot_phi_mlp_sweep.ps1
#   powershell ... -Legs baseline,h32_d2 -Epochs 45
#   powershell ... -RetrainBest   # 60-ep full train on winning leg

param(
    [string] $Legs = "",
    [int] $Epochs = 45,
    [switch] $RetrainBest,
    [switch] $SkipSummary
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")

$SweepRoot = "outputs/biochem/sweep_clot_phi_mlp"
$env:CLOT_PHI_SWEEP_DIR = $SweepRoot
$env:CLOT_PHI_DICE_LAMBDA = "0.2"

$AllLegs = [ordered]@{
    baseline   = @{ Hidden = 16; Depth = 1; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    h32        = @{ Hidden = 32; Depth = 1; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    h8         = @{ Hidden = 8;  Depth = 1; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    d2         = @{ Hidden = 16; Depth = 2; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    h32_d2     = @{ Hidden = 32; Depth = 2; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    drop0      = @{ Hidden = 16; Depth = 1; Dropout = 0.0;  Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    drop25     = @{ Hidden = 16; Depth = 1; Dropout = 0.25; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.5" }
    lr5e4      = @{ Hidden = 16; Depth = 1; Dropout = 0.15; Lr = "5e-4";  Wd = "1e-4"; MuLog = "1.5" }
    lr2e3      = @{ Hidden = 16; Depth = 1; Dropout = 0.15; Lr = "2e-3";  Wd = "1e-4"; MuLog = "1.5" }
    mu1        = @{ Hidden = 16; Depth = 1; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "1.0" }
    mu2        = @{ Hidden = 16; Depth = 1; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-4"; MuLog = "2.0" }
    wd1e5      = @{ Hidden = 16; Depth = 1; Dropout = 0.15; Lr = "1e-3";  Wd = "1e-5"; MuLog = "1.5" }
    h24_d2_mu1 = @{ Hidden = 24; Depth = 2; Dropout = 0.10; Lr = "8e-4";  Wd = "1e-4"; MuLog = "1.25" }
}

function Set-ClotPhiLegEnv {
    param($LegName, $Cfg)
    $env:CLOT_PHI_SWEEP_LEG = $LegName
    $env:CLOT_PHI_HIDDEN = "$($Cfg.Hidden)"
    $env:CLOT_PHI_MLP_DEPTH = "$($Cfg.Depth)"
    $env:CLOT_PHI_DROPOUT = "$($Cfg.Dropout)"
    $env:CLOT_PHI_LR = "$($Cfg.Lr)"
    $env:CLOT_PHI_WEIGHT_DECAY = "$($Cfg.Wd)"
    $env:CLOT_PHI_MU_LOG_LAMBDA = "$($Cfg.MuLog)"
    $env:CLOT_PHI_EPOCHS = "$Epochs"
}

function Invoke-ClotPhiLeg {
    param([string]$LegName, $Cfg)
    $legDir = Join-Path $RepoRoot "$SweepRoot\$LegName"
    New-Item -ItemType Directory -Force -Path $legDir | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_best.pth")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $legDir "clot_phi_train_log.jsonl")

    Set-ClotPhiLegEnv -LegName $LegName -Cfg $Cfg
    Write-Host ""
    Write-Host "[NEW] leg=$LegName hidden=$($Cfg.Hidden) depth=$($Cfg.Depth) drop=$($Cfg.Dropout) lr=$($Cfg.Lr) mu_log=$($Cfg.MuLog) epochs=$Epochs" -ForegroundColor Cyan
    python -m src.training.train_clot_phi_simple
}

if ($RetrainBest) {
    $summaryPath = Join-Path $RepoRoot "$SweepRoot\summary.jsonl"
    if (-not (Test-Path $summaryPath)) {
        throw "No sweep summary at $summaryPath; run sweep first."
    }
    $bestLine = Get-Content $summaryPath | Select-Object -First 1
    $best = $bestLine | ConvertFrom-Json
    $legName = $best.leg
    if (-not $AllLegs.Contains($legName)) {
        throw "Unknown best leg: $legName"
    }
    $Epochs = 60
    $env:CLOT_PHI_SWEEP_DIR = ""
    $env:CLOT_PHI_SWEEP_LEG = ""
    Set-ClotPhiLegEnv -LegName $legName -Cfg $AllLegs[$legName]
    $env:CLOT_PHI_EPOCHS = "60"
    Write-Host "[i]  Retrain best leg '$legName' for 60 epochs -> outputs/biochem/clot_phi_best.pth" -ForegroundColor Cyan
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_best.pth"
    Remove-Item -Force -ErrorAction SilentlyContinue "outputs\biochem\clot_phi_train_log.jsonl"
    python -m src.training.train_clot_phi_simple
    Copy-Item -Force "outputs\biochem\clot_phi_best.pth" "outputs\biochem\clot_phi_best_mlp.pth"
    python -m src.evaluation.viz_clot_phi_simple --anchor patient007 --checkpoint outputs/biochem/clot_phi_best.pth
    exit 0
}

$runList = @()
if ($Legs.Trim()) {
    $runList = $Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
} else {
    $runList = @($AllLegs.Keys)
}

foreach ($name in $runList) {
    if (-not $AllLegs.Contains($name)) {
        throw "Unknown leg '$name'. Valid: $($AllLegs.Keys -join ', ')"
    }
    Invoke-ClotPhiLeg -LegName $name -Cfg $AllLegs[$name]
}

if (-not $SkipSummary) {
    python (Join-Path $PSScriptRoot "summarize_clot_phi_mlp_sweep.py") --sweep-dir $SweepRoot
}
