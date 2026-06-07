# Viz Lane A deploy autonomy ckpt: timeline + t_final scatter panels.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_lane_a_deploy_viz.ps1"
#   powershell ... -Leg a04_target_long -Anchor patient006 -Keyframes 10

param(
    [string] $RunDir = "outputs\biochem\autonomy_clot_8h\run_20260607_001853",
    [string] $Leg = "a04_target_long",
    [string] $Anchor = "patient007",
    [int] $Keyframes = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Ckpt = Join-Path $RepoRoot "$RunDir\lane_a\$Leg\clot_phi_best.pth"
if (-not (Test-Path $Ckpt)) {
    Write-Host "[ERR] missing $Ckpt" -ForegroundColor Red
    exit 1
}

$OutDir = Join-Path $RepoRoot "outputs\biochem\viz\lane_a_deploy"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$TimelinePng = Join-Path $OutDir "${Leg}_${Anchor}_timeline.png"
$SummaryJson = Join-Path $OutDir "${Leg}_${Anchor}_timeline.jsonl"

Write-Host "[NEW] Lane A deploy viz leg=$Leg anchor=$Anchor" -ForegroundColor Cyan
Write-Host "[i]  ckpt=$Ckpt" -ForegroundColor DarkGray

python -m src.evaluation.viz_clot_forecast_timeline `
    --anchor $Anchor `
    --checkpoint $Ckpt `
    --keyframes $Keyframes `
    --out $TimelinePng `
    --summary-json $SummaryJson
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[OK]  timeline -> $TimelinePng" -ForegroundColor Green
Write-Host "[OK]  t_final band -> $OutDir\${Leg}_${Anchor}_timeline_tfinal_band.png" -ForegroundColor Green
Write-Host "[OK]  metrics  -> $SummaryJson" -ForegroundColor Green
