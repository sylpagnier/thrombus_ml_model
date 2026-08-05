# ~6h autonomous ladder: open patient001 hop_ge2, then healthy all-anchor compound.
#
# Phase A (~2.0h): Solo001 Freeze band + FN tilt (longer than crack 2ep)
# Phase B (~1.5h): Solo001 Unfreeze if A still closed
# Phase C (~1.0h): Solo001 CC tiles if still closed
# Phase D (~1.5h): Multi-anchor scale (001+007+010 then orig10) once 001 opens,
#                  OR last-resort recall push on 001+007 if still closed
#
# Soft deadline 6h. Stops early when 001 hop_ge2>0 AND all-anchor probe looks healthy.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_open001_6h.ps1
#   powershell ... -Fresh
#

param(
    [switch] $Fresh,
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_open001_6h",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [double] $DeadlineHours = 6.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) { throw "Wall ckpt missing: $WallPath" }

$deadline = (Get-Date).AddHours($DeadlineHours)
$statePath = Join-Path $OutDir "open001_6h_state.json"
Write-Host "[NEW] go_wc_v7_open001_6h deadline=$deadline out=$RunRoot" -ForegroundColor Cyan

function Save-State {
    param([hashtable] $Obj)
    $Obj["updated"] = (Get-Date).ToString("o")
    ($Obj | ConvertTo-Json -Depth 8) | Set-Content -Path $statePath -Encoding utf8
}

function Test-BudgetOk {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[WARN] 6h budget exceeded; stopping" -ForegroundColor Yellow
        return $false
    }
    $mins = [math]::Round(($deadline - (Get-Date)).TotalMinutes, 0)
    Write-Host "[i] budget remaining ~${mins}m" -ForegroundColor DarkGray
    return $true
}

function Clear-TrainEnv {
    Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
    $env:SPECIES_TWO_MODEL_MODE = "0"
    Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT, Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue
}

function Get-HopGe2 {
    param([string] $ProbeJson, [string] $Anchor)
    if (-not (Test-Path $ProbeJson)) { return 0.0 }
    $raw = Get-Content -Raw -Path $ProbeJson | ConvertFrom-Json
    $pa = $raw.simple.per_anchor.$Anchor
    if ($null -eq $pa) { return 0.0 }
    return [double]$pa.deploy_clot_offwall_n_pred_hop_ge2
}

function Get-ClotF1 {
    param([string] $ProbeJson, [string] $Anchor)
    if (-not (Test-Path $ProbeJson)) { return 0.0 }
    $raw = Get-Content -Raw -Path $ProbeJson | ConvertFrom-Json
    $pa = $raw.simple.per_anchor.$Anchor
    if ($null -eq $pa) { return 0.0 }
    return [double]$pa.deploy_clot_f1
}

function Invoke-Probe {
    param(
        [string] $ArmId,
        [string] $GrowthCkpt,
        [string] $Anchors
    )
    if (-not (Test-BudgetOk)) { return $false }
    Clear-TrainEnv
    $probeOut = Join-Path $OutDir "probe_$ArmId.json"
    if ($Fresh) { Remove-Item -Force $probeOut -ErrorAction SilentlyContinue }
    $g = $GrowthCkpt
    if (-not [System.IO.Path]::IsPathRooted($g)) { $g = Join-Path $RepoRoot $g }
    if (-not (Test-Path $g)) {
        Write-Host "[WARN] skip probe $ArmId; missing $g" -ForegroundColor Yellow
        return $false
    }
    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $probeOut,
        "--anchors", $Anchors,
        "--offwall-ckpt", $GrowthCkpt,
        "--two-model-route", "wall",
        "--two-model-frontier-hops", "2"
    )
    Write-Host "[i] Probe $ArmId anchors=$Anchors" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "probe $ArmId" -PyArgs $evalArgs
    return $true
}

function Invoke-GrowthTrain {
    param(
        [string] $ArmId,
        [string] $Anchors,
        [string] $ValAnchor,
        [int] $Epochs,
        [int] $EarlyStop,
        [int] $MaxWindows,
        [switch] $FreezeBackbone,
        [string] $TileMode = "union",
        [string] $SuperviseMode = "frontier_ge2",
        [string] $CkptMetric = "hop_ge2_recall",
        [double] $WallFloorDelta = 0.05,
        [int] $CompoundValEvery = 2,
        [double] $LumenShapeWeight = 10.0
    )
    if (-not (Test-BudgetOk)) { return "" }
    $growthDir = Join-Path $OutDir "growth_$ArmId"
    $growthCkpt = Join-Path $growthDir "best.pth"
    New-Item -ItemType Directory -Force -Path $growthDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force $growthCkpt, (Join-Path $growthDir "best.json"), (Join-Path $growthDir "train_log.jsonl") -ErrorAction SilentlyContinue
        Remove-Item -Force (Join-Path $OutDir "probe_$ArmId.json") -ErrorAction SilentlyContinue
    }
    if ((Test-Path $growthCkpt) -and -not $Fresh) {
        $meta = Join-Path $growthDir "best.json"
        $okBand = $false
        if (Test-Path $meta) {
            try {
                $j = Get-Content -Raw $meta | ConvertFrom-Json
                $okBand = ([string]$j.train_feat_source).Trim().ToLower() -eq "band"
                $up = [string]$j.env_overrides.SPECIES_CONTINUOUS_UNDERPRED_WEIGHT
                if ($okBand -and $up -ne "12.0") { $okBand = $false }
            } catch { $okBand = $false }
        }
        if ($okBand) {
            Write-Host "[skip] train $ArmId; band+FN ckpt exists" -ForegroundColor DarkGray
            return $growthCkpt
        }
        Write-Host "[i] stale/incomplete $ArmId; retraining" -ForegroundColor Yellow
        Remove-Item -Force $growthCkpt, (Join-Path $growthDir "best.json"), (Join-Path $growthDir "train_log.jsonl") -ErrorAction SilentlyContinue
        Remove-Item -Force (Join-Path $OutDir "probe_$ArmId.json") -ErrorAction SilentlyContinue
    }

    $env:SPECIES_LUMEN_SHAPE_FN_W = "25"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "0.35"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "12.0"

    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", $ValAnchor,
        "--anchors", $Anchors,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "5",
        "--tile-mode", $TileMode,
        "--max-tiles-per-window", "8",
        "--supervise-mode", $SuperviseMode,
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "$LumenShapeWeight",
        "--ckpt-metric", $CkptMetric,
        "--train-feat-source", "band",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $WallCkpt,
        "--out", $growthCkpt,
        "--compound-val",
        "--wall-ckpt", $WallCkpt,
        "--wall-clot-floor-delta", "$WallFloorDelta",
        "--compound-val-every", "$CompoundValEvery"
    )
    if ($FreezeBackbone) { $gArgs += @("--freeze-backbone") }

    Write-Host "[i] Train $ArmId anchors=$Anchors epochs=$Epochs windows=$MaxWindows freeze=$([bool]$FreezeBackbone) tile=$TileMode" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "train $ArmId" -PyArgs $gArgs
    Clear-TrainEnv
    if (-not (Test-Path $growthCkpt)) { return "" }
    return $growthCkpt
}

# --- state ---
$state = @{
    phase = "start"
    opened_001 = $false
    best_arm = $null
    notes = @()
}
if ((Test-Path $statePath) -and -not $Fresh) {
    try {
        $prev = Get-Content -Raw $statePath | ConvertFrom-Json
        if ($prev.opened_001) { $state.opened_001 = [bool]$prev.opened_001 }
        if ($prev.best_arm) { $state.best_arm = [string]$prev.best_arm }
    } catch {}
}

$gateAnchors = "patient001"
$sprayAnchors = "patient001,patient007,patient004,patient008"
$orig10 = "patient001,patient002,patient003,patient004,patient005,patient006,patient007,patient008,patient010,patient011"
$teachers = "patient001,patient007,patient010"

# Skip heavy wall-only baseline (saves ~30-40m on 4GB GPU). Train is the priority.

# ===== Phase A: Solo001 Freeze =====
$state.phase = "A_solo001_freeze"
Save-State $state
if (-not $state.opened_001 -and (Test-BudgetOk)) {
    $ck = Invoke-GrowthTrain -ArmId "A_Solo001_Freeze" -Anchors "patient001" -ValAnchor "patient001" `
        -Epochs 8 -EarlyStop 5 -MaxWindows 40 -FreezeBackbone -CompoundValEvery 3
    if ($ck -and (Invoke-Probe -ArmId "A_Solo001_Freeze" -GrowthCkpt $ck -Anchors $gateAnchors)) {
        $n001 = Get-HopGe2 -ProbeJson (Join-Path $OutDir "probe_A_Solo001_Freeze.json") -Anchor "patient001"
        Write-Host "[i] Phase A gate 001 hop_ge2=$n001" -ForegroundColor $(if ($n001 -gt 0.5) { "Green" } else { "Yellow" })
        if ($n001 -gt 0.5) {
            $state.opened_001 = $true
            $state.best_arm = "A_Solo001_Freeze"
            $state.notes += "opened by Solo001 Freeze band 8ep"
            $null = Invoke-Probe -ArmId "A_Solo001_Freeze_spray" -GrowthCkpt $ck -Anchors $sprayAnchors
        }
    }
}
Save-State $state

# ===== Phase B: Solo001 Unfreeze =====
$state.phase = "B_solo001_unfreeze"
Save-State $state
if (-not $state.opened_001 -and (Test-BudgetOk)) {
    $ck = Invoke-GrowthTrain -ArmId "B_Solo001_Unfreeze" -Anchors "patient001" -ValAnchor "patient001" `
        -Epochs 6 -EarlyStop 4 -MaxWindows 40 -CompoundValEvery 3
    if ($ck -and (Invoke-Probe -ArmId "B_Solo001_Unfreeze" -GrowthCkpt $ck -Anchors $gateAnchors)) {
        $n001 = Get-HopGe2 -ProbeJson (Join-Path $OutDir "probe_B_Solo001_Unfreeze.json") -Anchor "patient001"
        Write-Host "[i] Phase B gate 001 hop_ge2=$n001" -ForegroundColor $(if ($n001 -gt 0.5) { "Green" } else { "Yellow" })
        if ($n001 -gt 0.5) {
            $state.opened_001 = $true
            $state.best_arm = "B_Solo001_Unfreeze"
            $state.notes += "opened by Solo001 Unfreeze band"
            $null = Invoke-Probe -ArmId "B_Solo001_Unfreeze_spray" -GrowthCkpt $ck -Anchors $sprayAnchors
        }
    }
}
Save-State $state

# ===== Phase C: Solo001 CC =====
$state.phase = "C_solo001_cc"
Save-State $state
if (-not $state.opened_001 -and (Test-BudgetOk)) {
    $ck = Invoke-GrowthTrain -ArmId "C_Solo001_CC" -Anchors "patient001" -ValAnchor "patient001" `
        -Epochs 5 -EarlyStop 3 -MaxWindows 32 -TileMode "per_component" -CompoundValEvery 3
    if ($ck -and (Invoke-Probe -ArmId "C_Solo001_CC" -GrowthCkpt $ck -Anchors $gateAnchors)) {
        $n001 = Get-HopGe2 -ProbeJson (Join-Path $OutDir "probe_C_Solo001_CC.json") -Anchor "patient001"
        Write-Host "[i] Phase C gate 001 hop_ge2=$n001" -ForegroundColor $(if ($n001 -gt 0.5) { "Green" } else { "Yellow" })
        if ($n001 -gt 0.5) {
            $state.opened_001 = $true
            $state.best_arm = "C_Solo001_CC"
            $state.notes += "opened by Solo001 CC tiles"
            $null = Invoke-Probe -ArmId "C_Solo001_CC_spray" -GrowthCkpt $ck -Anchors $sprayAnchors
        }
    }
}
Save-State $state

# ===== Phase C2: last-resort teachers 001+007+010 unfreeze =====
$state.phase = "C2_teachers_recall"
Save-State $state
if (-not $state.opened_001 -and (Test-BudgetOk)) {
    $ck = Invoke-GrowthTrain -ArmId "C2_Teachers_Recall" -Anchors $teachers -ValAnchor "patient001" `
        -Epochs 6 -EarlyStop 4 -MaxWindows 36 -SuperviseMode "frontier_ge2" -CkptMetric "hop_ge2_recall" `
        -WallFloorDelta 0.08 -CompoundValEvery 3 -LumenShapeWeight 12.0
    if ($ck -and (Invoke-Probe -ArmId "C2_Teachers_Recall" -GrowthCkpt $ck -Anchors $gateAnchors)) {
        $n001 = Get-HopGe2 -ProbeJson (Join-Path $OutDir "probe_C2_Teachers_Recall.json") -Anchor "patient001"
        Write-Host "[i] Phase C2 gate 001 hop_ge2=$n001" -ForegroundColor $(if ($n001 -gt 0.5) { "Green" } else { "Yellow" })
        if ($n001 -gt 0.5) {
            $state.opened_001 = $true
            $state.best_arm = "C2_Teachers_Recall"
            $state.notes += "opened by teachers 001+007+010 recall"
            $null = Invoke-Probe -ArmId "C2_Teachers_Recall_spray" -GrowthCkpt $ck -Anchors $sprayAnchors
        }
    }
}
Save-State $state

# ===== Phase D: multi-anchor healthy compound =====
$state.phase = "D_scale"
Save-State $state
$seedArm = $state.best_arm
if (-not $seedArm) {
    # pick arm with highest 001 ge2 (even if 0) for warm continue
    foreach ($cand in @("C2_Teachers_Recall", "C_Solo001_CC", "B_Solo001_Unfreeze", "A_Solo001_Freeze")) {
        $p = Join-Path $OutDir "probe_$cand.json"
        if (Test-Path $p) { $seedArm = $cand; break }
    }
}

if ((Test-BudgetOk) -and $seedArm) {
    $seedCkpt = Join-Path $OutDir "growth_$seedArm\best.pth"
    # Warm multi-anchor from best specialist
    $env:SPECIES_LUMEN_SHAPE_FN_W = "25"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "0.55"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "8.0"

    $dDir = Join-Path $OutDir "growth_D_Orig10_Band"
    $dCkpt = Join-Path $dDir "best.pth"
    New-Item -ItemType Directory -Force -Path $dDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force $dCkpt, (Join-Path $dDir "best.json"), (Join-Path $dDir "train_log.jsonl") -ErrorAction SilentlyContinue
    }

    $needD = -not (Test-Path $dCkpt)
    if ((Test-Path $dCkpt) -and -not $Fresh) {
        try {
            $j = Get-Content -Raw (Join-Path $dDir "best.json") | ConvertFrom-Json
            if (([string]$j.train_feat_source).Trim().ToLower() -ne "band") { $needD = $true }
        } catch { $needD = $true }
    }

    if ($needD -and (Test-BudgetOk)) {
        $init = if (Test-Path $seedCkpt) { $seedCkpt } else { $WallCkpt }
        Write-Host "[i] Phase D orig10 scale init=$init opened001=$($state.opened_001)" -ForegroundColor Cyan
        $gArgs = @(
            "-m", "src.training.train_offwall_growth",
            "--val-anchor", "patient001",
            "--anchors", $orig10,
            "--epochs", "5",
            "--early-stop", "3",
            "--max-windows", "20",
            "--hops-k", "5",
            "--tile-mode", "union",
            "--max-tiles-per-window", "8",
            "--supervise-mode", "frontier_ge2",
            "--frontier-hops", "2",
            "--loss-mode", "loss_lumen_shape",
            "--lumen-shape-weight", "8",
            "--ckpt-metric", "hop_ge2_balanced",
            "--train-feat-source", "band",
            "--mat-leg", "WC_v7_clot_phi_mse",
            "--init", $init,
            "--out", $dCkpt,
            "--compound-val",
            "--wall-ckpt", $WallCkpt,
            "--wall-clot-floor-delta", "0.03",
            "--compound-val-every", "3",
            "--freeze-backbone"
        )
        $null = Invoke-PythonRcCheck -Label "train D_Orig10_Band" -PyArgs $gArgs
        Clear-TrainEnv
    }

    if ((Test-Path $dCkpt) -and (Test-BudgetOk)) {
        $null = Invoke-Probe -ArmId "D_Orig10_Band" -GrowthCkpt $dCkpt -Anchors $orig10
        $n001 = Get-HopGe2 -ProbeJson (Join-Path $OutDir "probe_D_Orig10_Band.json") -Anchor "patient001"
        $f1 = Get-ClotF1 -ProbeJson (Join-Path $OutDir "probe_D_Orig10_Band.json") -Anchor "patient001"
        Write-Host "[i] Phase D 001 hop_ge2=$n001 clot_f1=$f1" -ForegroundColor Cyan
        if ($n001 -gt 0.5) {
            $state.opened_001 = $true
            $state.best_arm = "D_Orig10_Band"
            $state.notes += "orig10 band compound keeps/opens 001"
        }
    }
}
Save-State $state

# ===== Summarize =====
$state.phase = "done"
Save-State $state
Write-Host "[i] Writing summary..." -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "summarize open001_6h" -PyArgs @(
    "scripts/summarize_open001_6h.py",
    "--run-root", $RunRoot,
    "--out", (Join-Path $OutDir "compare_open001_6h.json")
)

Write-Host "[OK] open001_6h done -> $OutDir best=$($state.best_arm) opened_001=$($state.opened_001)" -ForegroundColor Green
if ($state.opened_001) {
    Write-Host "[OK] GOAL: patient001 lumen hop_ge2 fired" -ForegroundColor Green
} else {
    Write-Host "[WARN] 001 still closed after 6h ladder - see compare_open001_6h.json" -ForegroundColor Yellow
}
