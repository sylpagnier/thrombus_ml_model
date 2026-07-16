# W-fix sweep (~10h cap): test high-leverage mitigations for inlet/wall FP overpaint.
#
# Focused levers:
# - all-node FP pressure (speed + gate/spatial terms)
# - drop flow spatial channels (x_n, y_n) to reduce memorization
# - seed/frontier gating (top-k nucleation + neighbor commit)
# - dynamic/coupled flow refresh
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_mat_w_fix_sweep_10h.ps1 -Fresh
#   powershell ... -File .\scripts\go_mat_w_fix_sweep_10h.ps1 -EvalOnly
#   powershell ... -File .\scripts\go_mat_w_fix_sweep_10h.ps1 -Legs W_mat_flow_stagnation,WK_mat_flow_dropxy

param(
    [switch] $Fresh,
    [switch] $EvalOnly,
    [string] $Legs = "",
    [int] $Epochs = 28,
    [int] $EarlyStop = 16,
    [int] $MaxWindows = 0,
    [int] $TargetHours = 10,
    [string] $ValAnchor = "patient007"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if ($Legs.Trim()) {
    $legList = @($Legs.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
    $legList = @(
        "W_mat_flow_stagnation",       # control
        "WC_mat_flow_dynamic",         # dynamic/coupled flow
        "X_mat_flow_seedfront",        # flow + seed/frontier
        "Y_mat_tight_seed",            # stricter top-k seed
        "WG_mat_flow_neighbor_crit",   # autocat + crit focus
        "WJ_mat_flow_stack",           # dynamic + gate + geom
        "WK_mat_flow_dropxy",          # remove x/y flow channels
        "WL_mat_flow_dropxy_tightfp",  # drop x/y + stronger FP penalties
        "WM_mat_flow_seedfront_tightfp" # seedfront + stronger FP penalties
    )
}

$RunRoot = "outputs/biochem/biochem_gnn/mat_w_fix_sweep_10h"
$SummaryJson = "$RunRoot/mat_w_fix_sweep_10h_summary.json"
$RunRecord = "$RunRoot/run_record_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$budgetMin = $TargetHours * 60
$estMinPerLeg = [math]::Round($Epochs * 2.8, 0)
$estTotalH = [math]::Round(($legList.Count * $estMinPerLeg) / 60, 1)

Write-Host "[NEW] mat W-fix sweep ($($legList.Count) legs)" -ForegroundColor Cyan
Write-Host "[i] $Epochs ep / ES $EarlyStop / max_windows=$MaxWindows ; est ~${estTotalH}h (cap ${TargetHours}h)" -ForegroundColor DarkGray
Write-Host "[i] winner filters: over/gt<=0.02, p90FP<=110, earlyFP<=40 ; rank clot_score, clot_f1" -ForegroundColor DarkGray

if ($Fresh) {
    foreach ($leg in $legList) {
        $legDir = "outputs/biochem/biochem_gnn/mat_growth_ladder/$leg"
        Remove-Item -Force "$legDir/species/best.pth" -ErrorAction SilentlyContinue
        Remove-Item -Force "$legDir/species/best.json" -ErrorAction SilentlyContinue
        Remove-Item -Force "$legDir/species/train_log.jsonl" -ErrorAction SilentlyContinue
        Remove-Item -Force "$legDir/compare.json" -ErrorAction SilentlyContinue
    }
}

if (-not $EvalOnly) {
    Invoke-PythonRcCheck -Label "mat W-fix pytest gate" -PyArgs @(
        "-m", "pytest",
        "src/tests/test_mat_growth_simple_scope.py",
        "src/tests/test_species_flow_feats.py",
        "src/tests/test_clot_timeline_metrics.py",
        "-q"
    )
}

$started = Get-Date
$completed = @()

foreach ($leg in $legList) {
    $totMin = [int]((Get-Date) - $started).TotalMinutes
    if ($totMin -ge $budgetMin) {
        Write-Host "[WARN] cumulative runtime >= ${TargetHours}h; stopping before remaining legs" -ForegroundColor Yellow
        break
    }

    $legStart = Get-Date
    Write-Host "[leg] $leg start=$($legStart.ToString('HH:mm:ss')) cumulative=${totMin}m" -ForegroundColor Cyan
    $legArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "go_mat_growth_simple.ps1"),
        "-Leg", $leg,
        "-Epochs", "$Epochs",
        "-EarlyStop", "$EarlyStop",
        "-MaxWindows", "$MaxWindows",
        "-ValAnchor", $ValAnchor
    )
    if ($Fresh) { $legArgs += "-Fresh" }
    if ($EvalOnly) { $legArgs += "-EvalOnly" }
    & powershell @legArgs
    if ($LASTEXITCODE -ne 0) { throw "$leg failed (exit=$LASTEXITCODE)" }

    $legMin = [int]((Get-Date) - $legStart).TotalMinutes
    $completed += $leg
    Write-Host "[leg] $leg done in ${legMin}m" -ForegroundColor DarkGray
}

$summaryLegs = if ($completed.Count -gt 0) { $completed } else { $legList }
if ($summaryLegs.Count -lt $legList.Count) {
    Write-Host "[WARN] summary uses completed legs only ($($summaryLegs -join ', '))" -ForegroundColor Yellow
}

Invoke-PythonRcCheck -Label "mat W-fix summary + winner" -PyArgs @(
    "scripts/summarize_mat_only_full.py",
    "--legs", ($summaryLegs -join ","),
    "--out", $SummaryJson,
    "--pick-winner",
    "--max-overpaint-per-gt", "0.02",
    "--max-clot-fp-p90", "110",
    "--max-clot-fp-early-mean", "40",
    "--rank-by", "deploy_clot_score,deploy_clot_f1"
)

$elapsedMin = [int]((Get-Date) - $started).TotalMinutes
$record = @{
    started_utc = $started.ToUniversalTime().ToString("o")
    elapsed_min = $elapsedMin
    legs_requested = $legList
    legs_completed = $completed
    epochs = $Epochs
    early_stop = $EarlyStop
    max_windows = $MaxWindows
    val_anchor = $ValAnchor
    target_hours = $TargetHours
    fresh = [bool]$Fresh
    eval_only = [bool]$EvalOnly
    summary_json = $SummaryJson
    promote_rule = "over/gt<=0.02 && clot_fp_p90<=110 && clot_fp_early_mean<=40 ; rank deploy_clot_score,deploy_clot_f1"
}
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $RunRecord

Write-Host "[OK] done in ${elapsedMin}m; summary -> $SummaryJson" -ForegroundColor Green
Write-Host "[save] run record -> $RunRecord" -ForegroundColor DarkGray
