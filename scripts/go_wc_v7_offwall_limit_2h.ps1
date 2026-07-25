# WC_v7 off-wall 2h limit-analysis sweep (extreme levers, p007, honest compound probe).
#
# Arms: A (wall baseline) | LumenPush | FrontierPush | SkipHopSpec | BlindSat
# Train: freeze-backbone growth specialist, cheap-val, max-windows 8, epochs 4
# Probe: compound wall-route deploy on patient007 only
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_wc_v7_offwall_limit_2h.ps1 -Fresh
#   powershell ... -Smoke -Fresh
#   powershell ... -SkipBlindSat -Fresh
#

param(
    [int] $Epochs = 4,
    [int] $EarlyStop = 3,
    [int] $MaxWindows = 8,
    [string] $Anchor = "patient007",
    [string] $WallCkpt = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth",
    [string] $RunRoot = "outputs/biochem/offwall_model/wc_v7_offwall_limit_2h",
    [switch] $SkipBlindSat,
    [switch] $Smoke,
    [switch] $Fresh,
    [switch] $EvalOnly
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
    $RunRoot = "outputs/biochem/offwall_model/wc_v7_offwall_limit_smoke"
    Write-Host "[i] SMOKE preset" -ForegroundColor Yellow
}

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$WallPath = Join-Path $RepoRoot $WallCkpt
if (-not (Test-Path $WallPath)) {
    throw "Wall/canonical ckpt missing: $WallPath"
}

$deadline = (Get-Date).AddHours(2)
Write-Host "[NEW] go_wc_v7_offwall_limit_2h anchor=$Anchor epochs=$Epochs max_windows=$MaxWindows" -ForegroundColor Cyan
Write-Host "[i] soft deadline ~2h -> $deadline | out=$RunRoot" -ForegroundColor DarkGray

function Test-BudgetOk {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[WARN] 2h budget exceeded; skipping remaining arms" -ForegroundColor Yellow
        return $false
    }
    return $true
}

function Clear-ArmEnv {
    foreach ($k in @(
        "SPECIES_LUMEN_SHAPE_FN_W",
        "SPECIES_LUMEN_SHAPE_FP_W",
        "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT",
        "SPECIES_SKIP_HOP_GNN",
        "SPECIES_MIDSIDE_BLIND_LOSS",
        "SPECIES_HOP1_SMOOTH",
        "SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL",
        "SPECIES_TWO_MODEL_MODE",
        "SPECIES_OFFWALL_MODEL_CKPT",
        "SPECIES_TWO_MODEL_ROUTE"
    )) {
        Remove-Item "Env:$k" -ErrorAction SilentlyContinue
    }
}

function Invoke-CompoundProbe {
    param(
        [string] $ArmId,
        [string] $GrowthCkpt = ""
    )
    Clear-ArmEnv
    $probeOut = Join-Path $OutDir "probe_$ArmId.json"
    if ($Fresh) { Remove-Item -Force $probeOut -ErrorAction SilentlyContinue }

    # Arm-specific deploy env (skiphop/sat must be on at forward for those arms)
    if ($ArmId -eq "SkipHopSpec") {
        $env:SPECIES_SKIP_HOP_GNN = "1"
    }
    if ($ArmId -eq "BlindSat") {
        $env:SPECIES_MIDSIDE_BLIND_LOSS = "1"
        $env:SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL = "30.0"
    }

    $evalArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $WallCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out", $probeOut,
        "--anchors", $Anchor
    )
    if ($GrowthCkpt -ne "") {
        $evalArgs += @(
            "--offwall-ckpt", $GrowthCkpt,
            "--two-model-route", "wall",
            "--two-model-frontier-hops", "2"
        )
    }
    Invoke-PythonRcCheck -Label "probe $ArmId" -PyArgs $evalArgs
    Clear-ArmEnv
}

function Invoke-TrainArm {
    param(
        [string] $ArmId,
        [hashtable] $TrainEnv,
        [string[]] $ExtraPyArgs
    )
    if (-not (Test-BudgetOk)) { return $false }
    Clear-ArmEnv
    foreach ($k in $TrainEnv.Keys) {
        Set-Item -Path "Env:$k" -Value ([string]$TrainEnv[$k])
    }

    $armDir = Join-Path $OutDir $ArmId
    $ckpt = Join-Path $armDir "best.pth"
    New-Item -ItemType Directory -Force -Path $armDir | Out-Null
    if ($Fresh) {
        Remove-Item -Force $ckpt, (Join-Path $armDir "train_log.jsonl"), (Join-Path $armDir "best.json") -ErrorAction SilentlyContinue
    }

    if (-not $EvalOnly) {
        $gArgs = @(
            "-m", "src.training.train_offwall_growth",
            "--val-anchor", $Anchor,
            "--anchors", $Anchor,
            "--epochs", "$Epochs",
            "--early-stop", "$EarlyStop",
            "--max-windows", "$MaxWindows",
            "--hops-k", "4",
            "--freeze-backbone",
            "--cheap-val",
            "--ckpt-metric", "hop_ge2_balanced",
            "--mat-leg", "WC_v7_clot_phi_mse",
            "--init", $WallCkpt,
            "--out", $ckpt
        ) + $ExtraPyArgs
        Write-Host "[i] Train $ArmId ..." -ForegroundColor Cyan
        Invoke-PythonRcCheck -Label "train $ArmId" -PyArgs $gArgs
    }
    if (-not (Test-Path $ckpt)) {
        throw "Missing growth ckpt after train: $ckpt"
    }
    Invoke-CompoundProbe -ArmId $ArmId -GrowthCkpt $ckpt
    return $true
}

# ---------------------------------------------------------------------------
# A: locked wall baseline
# ---------------------------------------------------------------------------
if (Test-BudgetOk) {
    Write-Host "[i] Probe A (wall alone)..." -ForegroundColor Cyan
    Invoke-CompoundProbe -ArmId "A"
}

# ---------------------------------------------------------------------------
# LumenPush: extreme hop>=2 Dice/FN
# ---------------------------------------------------------------------------
$ok = Invoke-TrainArm -ArmId "LumenPush" -TrainEnv @{
    SPECIES_LUMEN_SHAPE_FN_W = "20"
    SPECIES_LUMEN_SHAPE_FP_W = "0.25"
    SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "8.0"
} -ExtraPyArgs @(
    "--supervise-mode", "hop_ge2",
    "--loss-mode", "loss_lumen_shape",
    "--lumen-shape-weight", "10"
)

# ---------------------------------------------------------------------------
# FrontierPush: extreme underpred on clot neighborhood
# ---------------------------------------------------------------------------
if ($ok) {
    $ok = Invoke-TrainArm -ArmId "FrontierPush" -TrainEnv @{
        SPECIES_LUMEN_SHAPE_FN_W = "20"
        SPECIES_LUMEN_SHAPE_FP_W = "0.25"
        SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "8.0"
    } -ExtraPyArgs @(
        "--supervise-mode", "frontier",
        "--frontier-hops", "2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "10"
    )
}

# ---------------------------------------------------------------------------
# SkipHopSpec: bypass hop1 in specialist
# ---------------------------------------------------------------------------
if ($ok) {
    $ok = Invoke-TrainArm -ArmId "SkipHopSpec" -TrainEnv @{
        SPECIES_SKIP_HOP_GNN = "1"
        SPECIES_LUMEN_SHAPE_FN_W = "5"
        SPECIES_LUMEN_SHAPE_FP_W = "1"
        SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "4.0"
    } -ExtraPyArgs @(
        "--supervise-mode", "hop_ge2",
        "--loss-mode", "loss_lumen_shape",
        "--lumen-shape-weight", "4"
    )
}

# ---------------------------------------------------------------------------
# BlindSat: soft firewall knobs on specialist (drop if overtime)
# ---------------------------------------------------------------------------
if ($ok -and -not $SkipBlindSat) {
    if (Test-BudgetOk) {
        $null = Invoke-TrainArm -ArmId "BlindSat" -TrainEnv @{
            SPECIES_MIDSIDE_BLIND_LOSS = "1"
            SPECIES_CONTINUOUS_SATURATION_SCALE_OFFWALL = "30.0"
            SPECIES_LUMEN_SHAPE_FN_W = "5"
            SPECIES_CONTINUOUS_UNDERPRED_WEIGHT = "4.0"
        } -ExtraPyArgs @(
            "--supervise-mode", "hop_ge2",
            "--loss-mode", "loss_lumen_shape",
            "--lumen-shape-weight", "4"
        )
    } else {
        Write-Host "[skip] BlindSat (budget)" -ForegroundColor Yellow
    }
} elseif ($SkipBlindSat) {
    Write-Host "[skip] BlindSat (-SkipBlindSat)" -ForegroundColor DarkGray
}

Clear-ArmEnv
Write-Host "[i] Summarizing..." -ForegroundColor Cyan
Invoke-PythonRcCheck -Label "summarize limit 2h" -PyArgs @(
    "scripts/summarize_offwall_limit_2h.py",
    "--run-root", $RunRoot,
    "--out", (Join-Path $OutDir "limit_2h_summary.json")
)

Write-Host "[OK] off-wall limit 2h done -> $OutDir" -ForegroundColor Green
Write-Host "[i] Gates: signal=hop_ge2 up + strict; weak=volume only; null=fundamental for local SAGE" -ForegroundColor DarkGray
