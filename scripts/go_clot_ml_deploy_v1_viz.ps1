# Deploy v1 clot viz: default = pred phi only @ 5x extrap.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_ml_deploy_v1_viz.ps1"
#   powershell ... -Anchor patient007 -SimEndScale 5.0 -MaxFrames 20
#   powershell ... -WithFlow -Coupled -SimEndScale 1.5   # legacy mu/flow panels

param(
    [string] $Anchor = "patient007",
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Recipe = "data/reference/clot_ml_deploy_v1_extrap.json",
    [string] $Step1Ckpt = "outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth",
    [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth",
    [double] $SimEndScale = 5.0,
    [int] $TimeStride = 1,
    [int] $MaxFrames = 20,
    [switch] $WithFlow,
    [switch] $Coupled,
    [switch] $InWindow,
    [switch] $AllHorizons,
    [switch] $Cpu,
    [int] $KineMaxIters = 12
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$env:CLOT_PHI_KINE_CKPT = $KineCkpt
$env:CLOT_PHI_MINIMAL_FEATURES = "1"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_ML_CONTINUOUS_EXTRAP = "1"
$env:CLOT_ML_SIM_END_SCALE = "$SimEndScale"
$env:PYTHONUNBUFFERED = "1"
if ($Cpu) { $env:CLOT_ML_DEVICE = "cpu" }
else { Remove-Item Env:CLOT_ML_DEVICE -ErrorAction SilentlyContinue }

$VizDir = "outputs/biochem/viz/clot_deploy"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

function Invoke-Viz {
    param([string] $Label, [double] $Scale, [switch] $UseCoupled)
    $py = @(
        "scripts/viz_clot_ml_deploy_v1_timeline.py",
        "--anchor", $Anchor,
        "--anchor-dir", $AnchorDir,
        "--recipe", $Recipe,
        "--sim-end-scale", "$Scale",
        "--time-stride", "$TimeStride",
        "--max-frames", "$MaxFrames",
        "--scatter-size", "4.0",
        "--growth-curve",
        "--kine-max-iters", "$KineMaxIters"
    )
    if (-not $WithFlow) {
        $py += "--phi-only"
        $py += "--no-flow"
    }
    elseif ($UseCoupled) {
        $py += "--coupled"
    }
    else {
        $py += "--no-flow"
    }
    Write-Host ""
    Write-Host "[NEW] $Label (H=$Scale phi_only=$(-not $WithFlow) coupled=$UseCoupled)" -ForegroundColor Cyan
    Invoke-PythonRcCheck -Label $Label -PyArgs $py
}

$mode = if ($WithFlow) { "flow panels" } else { "phi-only" }
Write-Host "[NEW] deploy v1 viz anchor=$Anchor mode=$mode H=$SimEndScale stride=$TimeStride frames=$MaxFrames" -ForegroundColor Cyan

if ($InWindow) {
    Invoke-Viz -Label "deploy v1 in-window" -Scale 1.0
}

if ($AllHorizons) {
    foreach ($h in @(1.0, 1.5, 2.0, 5.0)) {
        if ($h -eq 1.0 -and -not $InWindow) { continue }
        Invoke-Viz -Label "deploy v1 H=$h" -Scale $h -UseCoupled:$Coupled
    }
}
else {
    Invoke-Viz -Label "deploy v1 extrap" -Scale $SimEndScale -UseCoupled:$Coupled
}

Write-Host ""
Write-Host "[OK] PNGs -> $VizDir\deploy_v1_${Anchor}*.png" -ForegroundColor Green
if ($WithFlow -and $Coupled -and -not $Cpu) {
    Write-Host "[i] 4GB GPU? use -Cpu or -WithFlow without -Coupled, or lower -MaxFrames" -ForegroundColor DarkGray
}
