# WC_v7 canonical vs compound growth A/B/C on the original 10 anchors (true ~9 h budget).
#
# Cohort (same as go_fresh_canonical.ps1):
#   patient001-008, patient010, patient011  (no patient009; excludes expanded 35-graph pack)
#
# Arms:
#   A  - frozen canonical WC_v7_clot_phi_mse (locked/species_gnn_best.pth)
#   B  - frontier compound (revised after 35-anchor Arm B overgrowth):
#          growth specialist: frontier supervise + loss_blurring_prec
#                               + ckpt on offwall_balanced
#          deploy: SPECIES_TWO_MODEL_ROUTE=frontier
#   C  - best-practice compound (recommended):
#          growth specialist: offwall supervise + loss_blurring_prec
#                               + ckpt on offwall_balanced
#          deploy: SPECIES_TWO_MODEL_ROUTE=wall  (WC_v7 keeps all wall)
#
# Why no Arm D (A + skiphop):
#   WC_pivot1_skiphop / WC_v5_skiphop regressed score or sprayed FPs (BIOCHEM_TRAINING_PROGRESS
#   §194 / §201). Skiphop also changes GNN topology, so locked WC_v7 weights cannot be
#   "eval'd with skiphop" cheaply. Prefer two-model compound; next firewall lever after this
#   run would be WC_v5_blind_loss-style hop-1 masking, not a fourth train leg here.
#
# Timing (scaled from 2026-07-20: ~18.7 min/epoch on 35 anchors -> ~5.3 min/epoch on 10):
#   Train one specialist:  ~2.5-3 h  (32 ep / ES 12)
#   Train B then C:        ~5-6 h
#   Eval A + B + C:        ~1.5-2.5 h
#   Full pipeline:         ~7-9 h
#
# Historical all-anchor Arm B (do not overwrite):
#   outputs/biochem/offwall_model/wc_v7_compound_abc_9h/growth_B_frontier_blurring/best.pth
#   Resume that compare: go_wc_v7_compound_growth_abc_9h.ps1 -EvalOnly -SkipC
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_compound_growth_abc_orig10_9h.ps1 -Fresh
#   powershell ... -Fast -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -EvalOnly -SkipC
#   powershell ... -SkipB / -SkipC
#

param(
    [int] $Epochs = 32,
    [int] $EarlyStop = 12,
    [int] $MaxWindows = 0,
    [int] $HopsK = 4,
    [int] $FrontierHops = 2,
    [string] $ValAnchor = "patient007",
    [string] $TrainAnchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [string] $EvalAnchors = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $MatLeg = "WC_v7_clot_phi_mse",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_compound_abc_orig10_9h",
    [switch] $Fast,
    [switch] $Smoke,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $SkipB,
    [switch] $SkipC,
    [switch] $SkipSummary
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
# expandable_segments is unsupported on some Windows CUDA builds; leave unset.

$FAST_EPOCHS = 6
$FAST_EARLYSTOP = 4
$FAST_MAX_WINDOWS = 8
if ($Smoke) {
    $Epochs = 1
    $EarlyStop = 1
    $MaxWindows = 4
    $HopsK = 3
    $TrainAnchors = "patient007"
    $EvalAnchors = "patient007"
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_compound_abc_orig10_smoke"
    $SkipSummary = $true
    Write-Host "[i] SMOKE preset: train B+C (cheap-val) + micro two-model forward" -ForegroundColor Yellow
} elseif ($Fast) {
    $Epochs = $FAST_EPOCHS
    $EarlyStop = $FAST_EARLYSTOP
    $MaxWindows = $FAST_MAX_WINDOWS
}

$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall/canonical ckpt missing: $WallPath (promote WC_v7 first)"
}

$OutDir = Join-Path $RepoRoot $RunRoot
$GrowthB = Join-Path $OutDir "growth_B_frontier_blurring_prec/best.pth"
$GrowthC = Join-Path $OutDir "growth_C_offwall_blurring_prec/best.pth"
$EvalA = Join-Path $OutDir "eval_A_canonical.json"
$EvalB = Join-Path $OutDir "eval_B_compound_frontier.json"
$EvalC = Join-Path $OutDir "eval_C_compound_wall_prec.json"
$Summary = Join-Path $OutDir "compare_summary.json"

New-Item -ItemType Directory -Force -Path (Split-Path $GrowthB) | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $GrowthC) | Out-Null

if ($Fresh) {
    foreach ($p in @($GrowthB, $GrowthC)) {
        if (Test-Path $p) { Remove-Item -Force $p }
        Remove-Item -Force (Join-Path (Split-Path $p) "train_log.jsonl") -ErrorAction SilentlyContinue
    }
    Remove-Item -Force $EvalA, $EvalB, $EvalC, $Summary -ErrorAction SilentlyContinue
}

Write-Host "[NEW] go_wc_v7_compound_growth_abc_orig10_9h" -ForegroundColor Cyan
Write-Host "[i] wall=$WallCkpt mat_leg=$MatLeg epochs=$Epochs early_stop=$EarlyStop max_windows=$MaxWindows" -ForegroundColor DarkGray
Write-Host "[i] train_anchors=$TrainAnchors" -ForegroundColor DarkGray
Write-Host "[i] eval_anchors=$EvalAnchors" -ForegroundColor DarkGray
Write-Host "[i] Arm B: supervise=frontier loss=loss_blurring_prec route=frontier ckpt=offwall_balanced" -ForegroundColor DarkGray
Write-Host "[i] Arm C: supervise=offwall loss=loss_blurring_prec route=wall ckpt=offwall_balanced" -ForegroundColor DarkGray
Write-Host "[i] No Arm D (skiphop): retired; see header. run_root=$RunRoot fresh=$Fresh smoke=$Smoke evalOnly=$EvalOnly skipB=$SkipB skipC=$SkipC" -ForegroundColor DarkGray

function Invoke-GrowthTrain {
    param(
        [string] $Label,
        [string] $OutCkpt,
        [string] $SuperviseMode,
        [string] $LossMode,
        [string] $CkptMetric,
        [int] $FrHops
    )
    $trainArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", $ValAnchor,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "$HopsK",
        "--frontier-hops", "$FrHops",
        "--supervise-mode", $SuperviseMode,
        "--loss-mode", $LossMode,
        "--ckpt-metric", $CkptMetric,
        "--mat-leg", $MatLeg,
        "--init", $WallCkpt,
        "--out", $OutCkpt,
        "--anchors", $TrainAnchors.Trim()
    )
    if ($Smoke) {
        $trainArgs += "--cheap-val"
    }
    Invoke-PythonRcCheck -Label $Label -PyArgs $trainArgs
    if (-not (Test-Path $OutCkpt)) {
        throw "Growth ckpt missing after train: $OutCkpt"
    }
}

function Invoke-CompoundEval {
    param(
        [string] $Label,
        [string] $OutJson,
        [string[]] $ExtraArgs
    )
    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", $MatLeg,
        "--no-baseline",
        "--out", $OutJson,
        "--anchors", $EvalAnchors.Trim()
    )
    if ($ExtraArgs) { $evalArgs += $ExtraArgs }
    Invoke-PythonRcCheck -Label $Label -PyArgs $evalArgs
}

# ---------------------------------------------------------------------------
# Phase 1: train specialists
# ---------------------------------------------------------------------------
if (-not $EvalOnly) {
    if (-not $SkipB) {
        Write-Host "[i] Phase 1a: train Arm B specialist (frontier + blurring_prec + offwall_balanced)..." -ForegroundColor Cyan
        Invoke-GrowthTrain -Label "Arm B growth train" -OutCkpt $GrowthB `
            -SuperviseMode "frontier" -LossMode "loss_blurring_prec" `
            -CkptMetric "offwall_balanced" -FrHops $FrontierHops
    } else {
        Write-Host "[i] Phase 1a: skipped (-SkipB)" -ForegroundColor DarkGray
    }
    if (-not $SkipC) {
        Write-Host "[i] Phase 1b: train Arm C specialist (offwall + blurring_prec + offwall ckpt)..." -ForegroundColor Cyan
        Invoke-GrowthTrain -Label "Arm C growth train" -OutCkpt $GrowthC `
            -SuperviseMode "offwall" -LossMode "loss_blurring_prec" `
            -CkptMetric "offwall_balanced" -FrHops $FrontierHops
    } else {
        Write-Host "[i] Phase 1b: skipped (-SkipC)" -ForegroundColor DarkGray
    }
} else {
    Write-Host "[i] Phase 1: skipped (EvalOnly)" -ForegroundColor DarkGray
}

if (-not $SkipB -and -not (Test-Path $GrowthB)) {
    throw "Arm B growth ckpt missing: $GrowthB"
}
if (-not $SkipC -and -not (Test-Path $GrowthC)) {
    throw "Arm C growth ckpt missing: $GrowthC"
}

# ---------------------------------------------------------------------------
# Phase 2: Arm A — canonical WC_v7 alone
# ---------------------------------------------------------------------------
if (-not $Smoke) {
    Write-Host "[i] Phase 2: eval Arm A (canonical WC_v7)..." -ForegroundColor Cyan
    $env:SPECIES_TWO_MODEL_MODE = "0"
    Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT -ErrorAction SilentlyContinue
    Remove-Item Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue
    Invoke-CompoundEval -Label "eval Arm A canonical" -OutJson $EvalA -ExtraArgs @()
} else {
    Write-Host "[i] Phase 2: skipped canonical eval (Smoke)" -ForegroundColor DarkGray
    @{
        anchors = @($ValAnchor)
        two_model = @{ enabled = $false }
        simple = @{ mean = @{}; label = "smoke_stub_A" }
    } | ConvertTo-Json -Depth 6 | Set-Content -Path $EvalA -Encoding utf8
}

# ---------------------------------------------------------------------------
# Phase 3/4: compound eval (full) or smoke micro-forward (fast)
# ---------------------------------------------------------------------------
if ($Smoke) {
    Write-Host "[i] Phase 3-4: smoke micro two-model forward (skip full deploy eval)..." -ForegroundColor Cyan
    $microOut = Join-Path $OutDir "smoke_micro_forward.json"
    Invoke-PythonRcCheck -Label "smoke two-model micro forward" -PyArgs @(
        "scripts/smoke_two_model_forward.py",
        "--wall-ckpt", $WallCkpt,
        "--growth-b", $GrowthB,
        "--growth-c", $GrowthC,
        "--mat-leg", $MatLeg,
        "--anchor", $ValAnchor,
        "--out", $microOut
    )
} else {
    if (-not $SkipB) {
        Write-Host "[i] Phase 3: eval Arm B (compound frontier route)..." -ForegroundColor Cyan
        Invoke-CompoundEval -Label "eval Arm B compound frontier" -OutJson $EvalB -ExtraArgs @(
            "--offwall-ckpt", $GrowthB,
            "--two-model-route", "frontier",
            "--two-model-frontier-hops", "$FrontierHops"
        )
    }
    if (-not $SkipC) {
        Write-Host "[i] Phase 4: eval Arm C (compound wall route + prec specialist)..." -ForegroundColor Cyan
        Invoke-CompoundEval -Label "eval Arm C compound wall prec" -OutJson $EvalC -ExtraArgs @(
            "--offwall-ckpt", $GrowthC,
            "--two-model-route", "wall"
        )
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if (-not $SkipSummary) {
    $sumArgs = @(
        "scripts/summarize_wc_v7_compound_ab.py",
        "--arm-a", $EvalA,
        "--out", $Summary
    )
    if ((-not $SkipB) -and (Test-Path $EvalB)) { $sumArgs += @("--arm-b", $EvalB) }
    if ((-not $SkipC) -and (Test-Path $EvalC)) { $sumArgs += @("--arm-c", $EvalC) }
    if ($sumArgs -contains "--arm-b" -or $sumArgs -contains "--arm-c") {
        Invoke-PythonRcCheck -Label "summarize WC_v7 compound A/B/C" -PyArgs $sumArgs
    } else {
        Write-Host "[WARN] No Arm B/C evals to summarize" -ForegroundColor Yellow
    }
}

Write-Host "[OK] WC_v7 compound A/B/C orig10 done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Success: Arm C clot F1/score near A; offwall n_pred -> n_gt; offwall relaxed F1 up" -ForegroundColor DarkGray
