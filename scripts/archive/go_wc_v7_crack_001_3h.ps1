# ~3h hypothesis ladder: crack patient001 lumen lock (hop_ge2 stays 0).
#
# Evidence so far:
#   Layer-1: compound-val full-graph static -> A_floor=0 (fixed: band_static val).
#   Layer-2: A_floor healthy but 001 still 0; solo-001 sprays 007/004/008.
#   Root: train tiles used global feats vs deploy wall-band (kin cos~0.16 on 001).
#   Fix: --train-feat-source band (default) matches eval features.
#
# Ladder (each arm trains ONLY patient001; stop early if 001 opens unless -FullLadder):
#   H1 Solo001_Freeze   - eliminate 007 gradient competition; freeze heads-only
#   H2 Solo001_Unfreeze - same + trainable backbone (feature lock test)
#   H3 Solo001_CC       - unfreeze + per_component tiles (compact-lumen density)
#
# All use compound-val on patient001 + hop_ge2_recall (real lumen metrics).
# Probes: 001 (gate) + 007 (hold) + 004/008 (spray).
#
# Verdicts (summarize_crack_001.py):
#   opened_by_competition_fix / opened_by_unfreeze / opened_by_cc_tiles
#   still_closed_architecture_suspect
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_crack_001_3h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -EvalOnly
#   powershell ... -Resume       # skip existing probes/ckpts; continue remaining stages
#   powershell ... -FullLadder   # run all stages even if 001 opens early
#

param(
    [int] $Epochs = 6,
    [int] $EarlyStop = 4,
    [int] $MaxWindows = 40,
    [int] $HopsK = 5,
    [int] $CompoundValEvery = 2,
    [double] $LumenShapeWeight = 10.0,
    [double] $WallClotFloorDelta = 0.05,
    [string] $ProbeAnchors = "patient001,patient007,patient004,patient008",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $Prec8hCkpt = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_crack_001_3h",
    [switch] $Smoke,
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $Resume,
    [switch] $FullLadder
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

if ($Smoke) {
    $Epochs = 1
    $EarlyStop = 1
    $MaxWindows = 2
    $CompoundValEvery = 1
    $ProbeAnchors = "patient001"
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_crack_001_smoke"
    $EvalOnly = $false
    Write-Host "[i] SMOKE preset (Solo001_Freeze train only)" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall ckpt missing: $WallPath"
}

if ($Fresh -and $Resume) {
    throw "Use either -Fresh or -Resume, not both"
}

$deadlineHours = if ($EvalOnly) { 4.0 } elseif ($Smoke) { 0.5 } else { 3.0 }
$deadline = (Get-Date).AddHours($deadlineHours)
Write-Host "[NEW] go_wc_v7_crack_001_3h epochs=$Epochs max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] probe=$ProbeAnchors out=$RunRoot full_ladder=$FullLadder resume=$Resume" -ForegroundColor DarkGray
Write-Host "[i] soft deadline ~${deadlineHours}h -> $deadline" -ForegroundColor DarkGray

function Test-BudgetOk {
    if ($EvalOnly) { return $true }
    if ((Get-Date) -gt $deadline) {
        Write-Host "[WARN] budget exceeded; skipping remaining work" -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Clear-TrainEnv {
    Remove-Item Env:SPECIES_LUMEN_SHAPE_FN_W, Env:SPECIES_LUMEN_SHAPE_FP_W, Env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT -ErrorAction SilentlyContinue
    $env:SPECIES_TWO_MODEL_MODE = "0"
    Remove-Item Env:SPECIES_OFFWALL_MODEL_CKPT, Env:SPECIES_TWO_MODEL_ROUTE -ErrorAction SilentlyContinue
}

function Get-HopGe2Pred001 {
    param([string] $ProbeJson)
    if (-not (Test-Path $ProbeJson)) { return 0.0 }
    $raw = Get-Content -Raw -Path $ProbeJson | ConvertFrom-Json
    $pa = $raw.simple.per_anchor.patient001
    if ($null -eq $pa) { return 0.0 }
    return [double]$pa.deploy_clot_offwall_n_pred_hop_ge2
}

function Invoke-CompoundProbe {
    param(
        [string] $ArmId,
        [string] $GrowthCkpt = ""
    )
    if (-not (Test-BudgetOk)) { return }
    Clear-TrainEnv
    $probeOut = Join-Path $OutDir "probe_$ArmId.json"
    if ($Fresh) { Remove-Item -Force $probeOut -ErrorAction SilentlyContinue }
    if ($Resume -and (Test-Path $probeOut)) {
        Write-Host "[skip] probe $ArmId already exists" -ForegroundColor DarkGray
        return
    }
    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $probeOut,
        "--anchors", $ProbeAnchors
    )
    if ($GrowthCkpt -and $GrowthCkpt.Trim()) {
        $g = $GrowthCkpt
        if (-not [System.IO.Path]::IsPathRooted($g)) { $g = Join-Path $RepoRoot $g }
        if (-not (Test-Path $g)) {
            Write-Host "[WARN] skip probe $ArmId; missing $g" -ForegroundColor Yellow
            return
        }
        $evalArgs += @(
            "--offwall-ckpt", $GrowthCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2"
        )
    }
    Write-Host "[i] Probe $ArmId -> $probeOut" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "probe $ArmId" -PyArgs $evalArgs
}

function Test-GrowthCkptUsesBandFeats {
    param([string] $GrowthDir)
    $meta = Join-Path $GrowthDir "best.json"
    if (-not (Test-Path $meta)) { return $false }
    try {
        $j = Get-Content -Raw -Path $meta | ConvertFrom-Json
        return ([string]$j.train_feat_source).Trim().ToLower() -eq "band"
    } catch {
        return $false
    }
}

function Invoke-CrackTrain {
    param(
        [string] $ArmId,
        [switch] $FreezeBackbone,
        [string] $TileMode = "union"
    )
    if (-not (Test-BudgetOk)) { return $false }
    $growthDir = Join-Path $OutDir "growth_$ArmId"
    $growthCkpt = Join-Path $growthDir "best.pth"
    $probeOut = Join-Path $OutDir "probe_$ArmId.json"
    New-Item -ItemType Directory -Force -Path $growthDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force $growthCkpt, (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json"), $probeOut -ErrorAction SilentlyContinue
    }
    if ($EvalOnly) {
        return (Test-Path $growthCkpt)
    }
    if ($Resume -and (Test-Path $growthCkpt)) {
        if (Test-GrowthCkptUsesBandFeats -GrowthDir $growthDir) {
            Write-Host "[skip] train $ArmId; band-feat ckpt exists -> $growthCkpt" -ForegroundColor DarkGray
            return $true
        }
        Write-Host "[i] stale ckpt $ArmId missing train_feat_source=band; retraining" -ForegroundColor Yellow
        Remove-Item -Force $growthCkpt, (Join-Path $growthDir "train_log.jsonl"), (Join-Path $growthDir "best.json"), $probeOut -ErrorAction SilentlyContinue
    }

    # Extreme FN tilt: force lumen positives; FP light so 001 can light before spray worry.
    $env:SPECIES_LUMEN_SHAPE_FN_W = "25"
    $env:SPECIES_LUMEN_SHAPE_FP_W = "0.35"
    $env:SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "12.0"

    $gArgs = @(
        "-m", "src.training.train_offwall_growth",
        "--val-anchor", "patient001",
        "--anchors", "patient001",
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--max-windows", "$MaxWindows",
        "--hops-k", "$HopsK",
        "--tile-mode", $TileMode,
        "--max-tiles-per-window", "8",
        "--supervise-mode", "frontier_ge2",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "$LumenShapeWeight",
        "--ckpt-metric", "hop_ge2_recall",
        "--train-feat-source", "band",
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--init", $WallCkpt,
        "--out", $growthCkpt
    )
    if ($FreezeBackbone) {
        $gArgs += @("--freeze-backbone")
    }
    if ($Smoke) {
        $gArgs += @("--cheap-val")
    } else {
        $gArgs += @(
            "--compound-val",
            "--wall-ckpt", $WallCkpt,
            "--wall-clot-floor-delta", "$WallClotFloorDelta",
            "--compound-val-every", "$CompoundValEvery"
        )
    }

    $freezeTag = if ($FreezeBackbone) { "freeze" } else { "unfreeze" }
    Write-Host "[i] Training $ArmId (solo 001, $freezeTag, tile=$TileMode)..." -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "train $ArmId" -PyArgs $gArgs
    Clear-TrainEnv
    return (Test-Path $growthCkpt)
}

function Test-Opened001 {
    param([string] $ArmId)
    $probe = Join-Path $OutDir "probe_$ArmId.json"
    $n = Get-HopGe2Pred001 -ProbeJson $probe
    $opened = $n -gt 0.5
    Write-Host "[i] gate ${ArmId}: patient001 hop_ge2_pred=$n opened=$opened" -ForegroundColor $(if ($opened) { "Green" } else { "Yellow" })
    return $opened
}

# --- Stages ---
$stages = @(
    @{ Id = "Solo001_Freeze"; Freeze = $true; Tile = "union"; Hyp = "H1 competition" },
    @{ Id = "Solo001_Unfreeze"; Freeze = $false; Tile = "union"; Hyp = "H2 backbone lock" },
    @{ Id = "Solo001_CC"; Freeze = $false; Tile = "per_component"; Hyp = "H3 tile density" }
)

if ($Smoke) {
    $ok = Invoke-CrackTrain -ArmId "Solo001_Freeze" -FreezeBackbone -TileMode "union"
    if (-not $ok) { throw "Smoke missing Solo001_Freeze ckpt" }
    Write-Host "[OK] SMOKE crack_001 train path OK" -ForegroundColor Green
    exit 0
}

# Baseline probes (once)
if (Test-BudgetOk) { Invoke-CompoundProbe -ArmId "A" }
if (Test-BudgetOk) {
    $precPath = if ([System.IO.Path]::IsPathRooted($Prec8hCkpt)) { $Prec8hCkpt } else { Join-Path $RepoRoot $Prec8hCkpt }
    if (Test-Path $precPath) {
        Invoke-CompoundProbe -ArmId "Prec8hRef" -GrowthCkpt $Prec8hCkpt
    } else {
        Write-Host "[WARN] Prec8hRef missing; skip" -ForegroundColor Yellow
    }
}

$openedEarly = $false
# If resuming after a prior open, honor existing open probes.
foreach ($st in $stages) {
    $probePath = Join-Path $OutDir "probe_$($st.Id).json"
    if ($Resume -and (Test-Path $probePath) -and (Test-Opened001 -ArmId $st.Id)) {
        $openedEarly = $true
        break
    }
}

foreach ($st in $stages) {
    if ($openedEarly -and -not $FullLadder) {
        Write-Host "[i] skip $($st.Id); 001 already opened (use -FullLadder to continue)" -ForegroundColor DarkGray
        continue
    }
    if (-not (Test-BudgetOk)) { break }

    Write-Host "[i] === $($st.Hyp): $($st.Id) ===" -ForegroundColor Cyan
    $trained = Invoke-CrackTrain -ArmId $st.Id -FreezeBackbone:$st.Freeze -TileMode $st.Tile
    if (-not $trained) {
        Write-Host "[WARN] missing ckpt for $($st.Id); skip probe" -ForegroundColor Yellow
        continue
    }
    $ck = Join-Path $OutDir "growth_$($st.Id)\best.pth"
    Invoke-CompoundProbe -ArmId $st.Id -GrowthCkpt $ck
    if (Test-Opened001 -ArmId $st.Id) {
        $openedEarly = $true
        Write-Host "[OK] 001 opened under $($st.Id) ($($st.Hyp))" -ForegroundColor Green
    }
}

Write-Host "[i] Summarize crack_001..." -ForegroundColor Cyan
$null = Invoke-PythonRcCheck -Label "summarize crack_001" -PyArgs @(
    "scripts/summarize_crack_001.py",
    "--run-root", $RunRoot,
    "--out", (Join-Path $OutDir "compare_crack_001.json")
)

Write-Host "[OK] crack_001 done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Primary gate: patient001 hop_ge2 pred > 0 under any Solo001 arm" -ForegroundColor DarkGray
