# 12h Mat-only precision + SeedFrontMat pivot ladder vs locked fi_mat baseline (0.634).
#
# Dense-field baselines (P/G/N/O) plus deployable sparse-front pivots (U/V/S/T) and gate-sharpening
# ablations (Q/R). Each leg trains at medium-full budget and evals vs the locked fi_mat ckpt.
#
# Pivot legs (deployable; NO GT clot mask / phi in forward):
#   U_mat_frontier_only   - structural pivot only: top-k seed + 1-hop front
#   V_mat_frontier_geom   - pivot + rich 2-hop geometry
#   S_mat_frontier_nuc      - SeedFrontMat_v0: U/V + neighbor gate + geom
#   T_mat_frontier_sharp    - S + sharp gate + spatial FP pressure
#
# Usage (overnight, up to 12h):
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_only_full_overnight.ps1 -Fresh
#   powershell ... -File .\scripts\go_mat_only_full_overnight.ps1 -Fresh -Epochs 50 -EarlyStop 35
#   powershell ... -File .\scripts\go_mat_only_full_overnight.ps1 -EvalOnly

param(
    [switch] $Fresh,
    [switch] $EvalOnly,
    [int] $Epochs = 40,
    [int] $EarlyStop = 25,
    [string] $ValAnchor = "patient007"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$RunRoot = "outputs/biochem/biochem_gnn/mat_only_full"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

# Priority order: dense attribution + baselines first, then pivot ladder U->V->S->T, then Q/R.
$legs = @(
    "P_mat_plain",
    "G_dual_mat_neighbor_gate",
    "N_mat_geom_rich",
    "O_mat_neighbor_geom_rich",
    "U_mat_frontier_only",
    "V_mat_frontier_geom",
    "S_mat_frontier_nuc",
    "T_mat_frontier_sharp",
    "Q_mat_gate_sharp_fp",
    "R_mat_geom_gate_sharp_fp"
)

$started = Get-Date
Write-Host "[NEW] mat-only 12h pivot ladder ($($legs.Count) legs): $($legs -join ', ')" -ForegroundColor Cyan
Write-Host "[i] each leg: $Epochs ep / early-stop $EarlyStop / all windows; eval vs locked fi_mat baseline" -ForegroundColor DarkGray

foreach ($leg in $legs) {
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $leg,
        "-Epochs", "$Epochs",
        "-EarlyStop", "$EarlyStop",
        "-MaxWindows", "0",
        "-ValAnchor", $ValAnchor
    )
    if ($Fresh) { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }

    $legStart = Get-Date
    Write-Host "[leg] $leg starting at $($legStart.ToString('HH:mm:ss'))" -ForegroundColor Cyan
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "$leg failed (exit=$LASTEXITCODE)" }

    $legMin = [int]((Get-Date) - $legStart).TotalMinutes
    $totMin = [int]((Get-Date) - $started).TotalMinutes
    Write-Host "[leg] $leg done in $legMin min (cumulative $totMin min)" -ForegroundColor DarkGray
    if ($totMin -ge 720) {
        Write-Host "[WARN] cumulative runtime >= 12h; remaining legs skipped" -ForegroundColor Yellow
        break
    }
}

Invoke-PythonRcCheck -Label "mat-only ladder summary" -PyArgs @(
    "scripts/summarize_mat_only_full.py",
    "--legs", ($legs -join ","),
    "--out", "$RunRoot/mat_only_full_summary.json"
)

$elapsed = (Get-Date) - $started
Write-Host "[OK] mat-only 12h ladder done in $([int]$elapsed.TotalMinutes) min; summary under $RunRoot" -ForegroundColor Green
