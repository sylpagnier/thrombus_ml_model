# Optional R1 sanity viz: same-frame phi/mu on supervision band (NOT t->t+1 pair plot).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_forecast_r1_viz.ps1" -Prong D
#   powershell ... -Prong D -Anchor patient007 -TimeIndex -1

param(
    [ValidateSet("A", "B", "C", "D")]
    [string] $Prong = "D",
    [string] $Anchor = "patient007",
    [int] $TimeIndex = -1,
    [ValidateSet("scatter", "tri")]
    [string] $PlotMode = "scatter"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Leg = switch ($Prong) {
    "B" { "r1_prong_b" }
    "C" { "r1_prong_c" }
    "D" { "r1_prong_d" }
    default { "r1_prong_a" }
}

$Ckpt = Join-Path $RepoRoot "outputs\biochem\clot_forecast_ladder\$Leg\clot_phi_best.pth"
if (-not (Test-Path $Ckpt)) {
    Write-Host "[ERR] missing checkpoint $Ckpt (run go_clot_forecast_r1.ps1 -Prong $Prong first)" -ForegroundColor Red
    exit 1
}

$OutDir = Join-Path $RepoRoot "outputs\biochem\clot_forecast_ladder\viz"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutPng = Join-Path $OutDir "${Leg}_${Anchor}_t${TimeIndex}.png"

Write-Host "[NEW] R1 prong $Prong viz sanity (same-frame; forecast pair viz = R2+)" -ForegroundColor Cyan
Write-Host "[i]  ckpt=$Ckpt" -ForegroundColor DarkGray
Write-Host "[WARN] This is NOT a one-step t->t+dt forecast panel; use R2 rollout PNG for temporal growth." -ForegroundColor Yellow

python -m src.evaluation.viz_clot_phi_simple `
    --anchor $Anchor `
    --checkpoint $Ckpt `
    --time-index $TimeIndex `
    --plot-mode $PlotMode `
    --out $OutPng

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "[OK]  wrote $OutPng" -ForegroundColor Green
