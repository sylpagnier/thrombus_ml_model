# Fast bend_sign A/B: down_only (Apr-2026 style) vs bidirectional (May-2026).
# Writes isolated graph trees under data/processed/graphs_kinematics/ab_bend_<arm>/newtonian
# so the main 500-graph cohort is untouched.
param(
    [ValidateSet("down", "bidir", "both")]
    [string]$Arm = "both",
    [int]$NumVessels = 120,
    [int]$AnchorMax = 0,
    [int]$Seed = 42,
    [switch]$SkipTrain,
    [switch]$DatagenOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Invoke-BendArm {
    param([string]$Name, [string]$Mode)

    $graphRel = "data/processed/graphs_kinematics/ab_bend_$Name/newtonian"
    $graphDir = Join-Path $root $graphRel
    New-Item -ItemType Directory -Force -Path $graphDir | Out-Null

    $env:KINEMATICS_BEND_SIGN_MODE = $Mode
    $env:KINEMATICS_GRAPH_RHEOLOGY_DIR = $graphRel

    Write-Host "`n=== BEND A/B arm: $Name (KINEMATICS_BEND_SIGN_MODE=$Mode) ===" -ForegroundColor Cyan
    Write-Host "Graphs -> $graphDir"

    $dgArgs = @(
        "-m", "src.data_gen.pipeline_kinematics",
        "--batch",
        "--rheology", "newtonian",
        "-n", "$NumVessels",
        "--mixed-levels",
        "--overwrite",
        "--bend-sign-mode", $Mode,
        "--seed", "$Seed"
    )
    if ($AnchorMax -le 0) {
        $dgArgs += "--skip-anchor"
    } else {
        $dgArgs += @("--anchor-max-new", "$AnchorMax")
        if ($AnchorMax -ge $NumVessels) {
            $dgArgs += "--anchor-overwrite"
        }
    }

    & python @dgArgs
    if ($LASTEXITCODE -ne 0) { throw "datagen failed for arm $Name" }

    if ($DatagenOnly) { return }

    Write-Host "Training smoke (limit-data=$NumVessels, L0/L1-only ep 0-3)..." -ForegroundColor Yellow
    & python -m src.training.train_kinematics_predictor --fresh `
        --limit-data $NumVessels `
        --epochs 12 --adam-epochs 10 `
        --stage1-end-epoch 8 --stage2-end-epoch 10 `
        --l0l1-only-epochs 4
    if ($LASTEXITCODE -ne 0) { throw "training failed for arm $Name" }

    Remove-Item Env:KINEMATICS_GRAPH_RHEOLOGY_DIR -ErrorAction SilentlyContinue
}

if ($Arm -eq "both") {
    Invoke-BendArm -Name "down" -Mode "down_only"
    if (-not $SkipTrain) {
        Invoke-BendArm -Name "bidir" -Mode "bidirectional"
    } else {
        $env:KINEMATICS_BEND_SIGN_MODE = "bidirectional"
        $dgArgs = @(
            "-m", "src.data_gen.pipeline_kinematics",
            "--batch", "--rheology", "newtonian",
            "-n", "$NumVessels", "--mixed-levels", "--overwrite",
            "--bend-sign-mode", "bidirectional", "--seed", "$Seed", "--skip-anchor"
        )
        $graphRel = "data/processed/graphs_kinematics/ab_bend_bidir/newtonian"
        $env:KINEMATICS_GRAPH_RHEOLOGY_DIR = $graphRel
        & python @dgArgs
    }
} elseif ($Arm -eq "down") {
    Invoke-BendArm -Name "down" -Mode "down_only"
} else {
    Invoke-BendArm -Name "bidir" -Mode "bidirectional"
}

Write-Host "`nDone. Compare val lines: L0= and especially L1= between arms." -ForegroundColor Green
Write-Host "Optional: python -m src.data_gen.backfill_kinematics_geometry_level (main tree only)." -ForegroundColor DarkGray
