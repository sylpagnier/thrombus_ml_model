# V3.2 ~30 min sweep: trajectory metrics + ranker vs Euler + viz (p007, p002).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_v32_sweep_30m.ps1"
#   powershell ... -Fast -SkipTrain -SkipViz
#   powershell ... -Legs "v32_ranker,v32_ranker_onset2x"

param(
    [string] $AnchorDir = "data/processed/graphs_biochem_anchors",
    [string] $Step0Json = "outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json",
    [string] $V3Ckpt = "outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn/clot_ml_v3_growth_gnn_best.pth",
    [string] $V31Ckpt = "outputs/biochem/clot_ml_ladder_v2/v31_growth_gnn/clot_ml_v31_growth_gnn_best.pth",
    [string] $Val = "patient007",
    [string] $VizAnchors = "patient007,patient002",
    [int] $Epochs = 6,
    [int] $MaxFrames = 10,
    [string] $Legs = "",
    [switch] $Fast,
    [switch] $SkipTrain,
    [switch] $SkipViz
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_clot_prior_rule_winner_env.ps1")
. (Join-Path $PSScriptRoot "_python_rc.ps1")

$OutRoot = "outputs/biochem/clot_ml_ladder_v2/v32_sweep_30m"
$VizDir = "outputs/biochem/viz/clot_v2"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $OutRoot) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot $VizDir) | Out-Null

if ($Fast) { $Epochs = 4 }

$env:CLOT_V2_NUCLEATION = "1"
$env:CLOT_ML_USE_MACRO_TAU = "1"
$env:CLOT_V31_RECIPE = "1"
$env:CLOT_V3_RATE_SCALE = "2.5"
$env:CLOT_V31_MAX_STEP_DELTA = "0.10"
$env:CLOT_V31_HARD_COMMIT = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Host "[NEW] V3.2 sweep (~30m) epochs=$Epochs val=$Val viz=$VizAnchors" -ForegroundColor Cyan

$sweepArgs = @(
    "scripts/sweep_clot_v32_growth_30m.py",
    "--anchor-dir", $AnchorDir,
    "--step0-json", $Step0Json,
    "--out-dir", $OutRoot,
    "--val", $Val,
    "--epochs", "$Epochs",
    "--v3-ckpt", $V3Ckpt,
    "--v31-ckpt", $V31Ckpt,
    "--viz-anchors", $VizAnchors,
    "--max-frames", "$MaxFrames"
)
if ($Legs) { $sweepArgs += @("--legs", $Legs) }
if ($SkipTrain) { $sweepArgs += "--skip-train" }
if ($SkipViz) { $sweepArgs += "--skip-viz" }

Invoke-PythonRcCheck -Label "v32 sweep" -PyArgs $sweepArgs

Write-Host ""
Write-Host "[OK] summary -> $OutRoot\sweep_summary.json" -ForegroundColor Green
Write-Host "[OK] viz PNGs -> $VizDir\v32_ranker_patient007.png (and patient002)" -ForegroundColor Green
