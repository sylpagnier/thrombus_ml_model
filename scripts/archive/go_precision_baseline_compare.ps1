# Evaluate the current best (locked) model under the SAME deploy_frozen + relaxed
# precision metric as the precision_6h sweep, then do an authoritative, baseline-gated
# promotion: only promote a sweep leg if it beats the locked baseline on the objective.
#
# Waits for the in-flight precision_6h sweep to finish (its final summary) to avoid GPU
# contention. Use -NoWait to run immediately.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_precision_baseline_compare.ps1

param(
    [double] $Floor = 0.30,
    [double] $WP007 = 0.5,
    [string] $Anchors = "patient001,patient002,patient003,patient004,patient006,patient007",
    [switch] $NoWait
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/precision_6h"
$DoneMarker = Join-Path $RunRoot "precision_final_summary.json"
$BaselineManifest = Join-Path $RepoRoot "data/reference/biochem_gnn_baseline.json"
if (-not (Test-Path $BaselineManifest)) { throw "missing baseline manifest: $BaselineManifest" }

if (-not $NoWait) {
    Write-Host "[i] waiting for precision_6h sweep to finish ($DoneMarker)" -ForegroundColor DarkGray
    $deadline = (Get-Date).AddHours(8)
    while (-not (Test-Path $DoneMarker)) {
        if ((Get-Date) -gt $deadline) { throw "timed out waiting for precision_6h sweep" }
        Start-Sleep -Seconds 60
    }
    Write-Host "[OK] sweep final summary present; starting baseline eval" -ForegroundColor Green
    Start-Sleep -Seconds 5
}

# --- Evaluate locked baseline under identical deploy_frozen metric ---
$baseDir = Join-Path $RunRoot "baseline_locked"
$baseEvalDir = Join-Path $baseDir "eval"
New-Item -ItemType Directory -Force -Path $baseEvalDir | Out-Null
$baseEval = Join-Path $baseEvalDir "deploy_ab_eval.json"

# Baseline scope is fi_mat; ensure no stray channel override leaks in.
Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS -ErrorAction SilentlyContinue
Remove-Item Env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE -ErrorAction SilentlyContinue

Write-Host "[run] [baseline_locked] eval deploy_frozen (current best)" -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "[baseline_locked] eval" -PyArgs @(
    "scripts/eval_biochem_gnn_deploy_ab.py",
    "--manifest", $BaselineManifest,
    "--modes", "deploy_frozen",
    "--times", "53,200",
    "--anchors", $Anchors,
    "--out", $baseEval
)

# --- Re-summarize (baseline-aware) ---
$sj = Join-Path $RunRoot "precision_final_vs_baseline_summary.json"
$sm = Join-Path $RunRoot "precision_final_vs_baseline_report.md"
$null = Invoke-PythonRcCheck -Label "summary vs baseline" -PyArgs @(
    "scripts/summarize_species_precision_sweep.py",
    "--sweep-root", $RunRoot,
    "--floor", "$Floor",
    "--w-p007", "$WP007",
    "--out-json", $sj,
    "--out-md", $sm
)

# --- Authoritative guarded promotion ---
$summary = Get-Content $sj -Raw | ConvertFrom-Json
$best = $summary.best
$promoteOk = [bool]$summary.promote_ok
if (-not $best) { Write-Host "[WARN] no sweep best found; nothing to promote" -ForegroundColor Yellow; exit 0 }

$bm = $best.metrics
Write-Host ("[i] best sweep leg {0}: score {1:N3} (p007 rprec {2:N3} @ rrec {3:N3})" -f $best.label, $bm.score, $bm.p007_relaxed_prec, $bm.p007_relaxed_rec) -ForegroundColor Cyan
if ($summary.baseline) {
    $cm = $summary.baseline.metrics
    Write-Host ("[i] locked baseline: score {0:N3} (p007 rprec {1:N3} @ rrec {2:N3})" -f $cm.score, $cm.p007_relaxed_prec, $cm.p007_relaxed_rec) -ForegroundColor Cyan
}

if ($promoteOk) {
    $promoteDir = Join-Path $RunRoot "precision_best"
    New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
    $srcCkpt = Join-Path $RunRoot "$($best.label)/species/best.pth"
    $srcManifest = Join-Path $RunRoot "$($best.label)/manifest.json"
    if (Test-Path $srcCkpt) { Copy-Item $srcCkpt (Join-Path $promoteDir "species_gnn_best.pth") -Force }
    if (Test-Path $srcManifest) { Copy-Item $srcManifest (Join-Path $promoteDir "manifest.json") -Force }
    $best | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $promoteDir "best_summary.json")
    Write-Host "[OK] PROMOTE: best beats locked baseline -> $promoteDir" -ForegroundColor Green
    Write-Host "[i] To make canonical, copy species_gnn_best.pth into outputs/biochem/biochem_gnn/locked/ and update data/reference/biochem_gnn_baseline.json (review first)." -ForegroundColor DarkGray
} else {
    Write-Host "[WARN] NO PROMOTE: best sweep leg does not beat locked baseline on relaxed-precision objective. Keeping current best." -ForegroundColor Yellow
    # Remove any stale unconditional promotion left by the orchestrator's Phase C.
    $promoteDir = Join-Path $RunRoot "precision_best"
    $staleCkpt = Join-Path $promoteDir "species_gnn_best.pth"
    if (Test-Path $staleCkpt) { Remove-Item $staleCkpt -Force }
    New-Item -ItemType Directory -Force -Path $promoteDir | Out-Null
    @{ promoted = $false; reason = "best does not beat locked baseline"; best = $best; baseline = $summary.baseline } |
        ConvertTo-Json -Depth 8 | Set-Content (Join-Path $promoteDir "PROMOTION_STATUS.json")
}
Write-Host "[OK] baseline comparison report -> $sm" -ForegroundColor Green
