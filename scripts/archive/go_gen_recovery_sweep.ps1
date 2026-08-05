# Anti-memorization A/B sweep on the straightish family of 7 (2026-07-29).
#
# Runs cold arms back-to-back with held-out eval on patient020 + patient034,
# then prints a comparison table.
#
# Optimization:
#   In-training val uses uncoupled flow ($env:SPECIES_CLOSED_LOOP_COUPLING = "0") for
#   instant per-epoch validation (~1s/ep). Post-training heldout eval runs full canonical
#   coupled rollout (~5-8 min/arm).
#
# Arms:
#   A  drop-xy only
#   B  drop-xy + latent dropout 0.3
#   C  drop-xy + latent dropout 0.5
#   D  drop-xy + rich geometry
#   E  drop-xy + latent dropout 0.3 + rich geometry (stacked)
#   F  drop-xy + weight decay 1e-3
#   G  drop-xy + weight decay 1e-3 + latent dropout 0.3
#   H  drop-xy + weight decay 1e-3 + rich geometry
#   I  Full Stack (drop-xy + dropout 0.3 + rich geom + wd 1e-3)
#   J  Full Stack 60-epoch deep train
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_gen_recovery_sweep.ps1 -Fresh
#   powershell ... -ArmFilter "A,B,E,I"      # run selected arms
#   powershell ... -EvalOnly                 # skip training, re-eval existing ckpts

param(
    [int]    $Epochs       = 35,
    [int]    $EarlyStop    = 15,
    [int]    $MaxWindows   = 24,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020,patient034",
    [string] $RunRoot      = "outputs/biochem/eda/gen_recovery_sweep",
    [string] $ArmFilter    = "",
    [switch] $Fresh,
    [switch] $EvalOnly,
    [switch] $CheapVal
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"

$OutDir = Join-Path $RepoRoot $RunRoot
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ---- 10-arm anti-memorization matrix ----------------------------------------------
$arms = [ordered]@{
    "A" = @{
        label = "drop-xy only"
        trainFlags = @("--drop-xy")
        epochs = $Epochs
    }
    "B" = @{
        label = "drop-xy + latent dropout 0.3"
        trainFlags = @("--drop-xy", "--latent-dropout", "0.3")
        epochs = $Epochs
    }
    "C" = @{
        label = "drop-xy + latent dropout 0.5"
        trainFlags = @("--drop-xy", "--latent-dropout", "0.5")
        epochs = $Epochs
    }
    "D" = @{
        label = "drop-xy + rich geometry"
        trainFlags = @("--drop-xy", "--geom-rich")
        epochs = $Epochs
    }
    "E" = @{
        label = "drop-xy + latent dropout 0.3 + rich geom"
        trainFlags = @("--drop-xy", "--latent-dropout", "0.3", "--geom-rich")
        epochs = $Epochs
    }
    "F" = @{
        label = "drop-xy + weight decay 1e-3"
        trainFlags = @("--drop-xy", "--weight-decay", "1e-3")
        epochs = $Epochs
    }
    "G" = @{
        label = "drop-xy + weight decay 1e-3 + latent dropout 0.3"
        trainFlags = @("--drop-xy", "--weight-decay", "1e-3", "--latent-dropout", "0.3")
        epochs = $Epochs
    }
    "H" = @{
        label = "drop-xy + weight decay 1e-3 + rich geom"
        trainFlags = @("--drop-xy", "--weight-decay", "1e-3", "--geom-rich")
        epochs = $Epochs
    }
    "I" = @{
        label = "Full Stack (drop-xy + dropout 0.3 + rich geom + wd 1e-3)"
        trainFlags = @("--drop-xy", "--latent-dropout", "0.3", "--geom-rich", "--weight-decay", "1e-3")
        epochs = $Epochs
    }
    "J" = @{
        label = "Full Stack 60-epoch deep train"
        trainFlags = @("--drop-xy", "--latent-dropout", "0.3", "--geom-rich", "--weight-decay", "1e-3")
        epochs = [math]::Max($Epochs, 60)
    }
    "K" = @{
        label = "Full Stack GAT"
        trainFlags = @("--arch", "gat", "--drop-xy", "--latent-dropout", "0.3", "--geom-rich", "--weight-decay", "1e-3")
        epochs = $Epochs
    }
}

# Filter arms if requested
$armKeys = if ($ArmFilter) {
    @($ArmFilter.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ })
} else {
    @($arms.Keys)
}

$started = Get-Date
Write-Host "[NEW] gen_recovery_sweep: $($armKeys.Count) arms, default $Epochs ep / ES $EarlyStop / max_windows $MaxWindows" -ForegroundColor Cyan
Write-Host "[i] train=$TrainAnchors val=$ValAnchor holdout=$HoldoutAnchors" -ForegroundColor DarkGray
Write-Host "[i] arms: $($armKeys -join ', ')" -ForegroundColor DarkGray

$summaryRows = @()

foreach ($armId in $armKeys) {
    if (-not $arms.Contains($armId)) {
        Write-Host "[WARN] unknown arm '$armId', skipping" -ForegroundColor Yellow
        continue
    }

    $arm = $arms[$armId]
    $armDir = Join-Path $OutDir "arm_$armId"
    $armCkpt = Join-Path $armDir "best.pth"
    $armMeta = Join-Path $armDir "best.json"
    $armLog  = Join-Path $armDir "train_log.jsonl"
    $armHold = Join-Path $armDir "eval_holdout.json"
    New-Item -ItemType Directory -Force -Path $armDir | Out-Null

    $armStart = Get-Date
    Write-Host "" -ForegroundColor Cyan
    Write-Host "====== Arm ${armId} : $($arm.label) ======" -ForegroundColor Cyan
    Write-Host "[i] start $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor DarkGray

    if ($Fresh) {
        Remove-Item -Force $armCkpt, $armMeta, $armLog, $armHold -ErrorAction SilentlyContinue
    }

    if (-not $EvalOnly) {
        # Speed optimization: uncoupled in-training validation for instant per-epoch val
        $env:SPECIES_CLOSED_LOOP_COUPLING = "0"

        $epCount = if ($arm.epochs) { "$($arm.epochs)" } else { "$Epochs" }
        $trainArgs = @(
            "-m", "src.training.train_species_pushforward_continuous",
            "--phase", "biochem_gnn",
            "--recipe", "mat_growth_simple",
            "--leg", "WC_v7_clot_phi_mse",
            "--anchors", $TrainAnchors,
            "--val-anchor", $ValAnchor,
            "--exclude-val-from-train",
            "--epochs", $epCount,
            "--early-stop", "$EarlyStop",
            "--max-windows", "$MaxWindows",
            "--no-init",
            "--out", $armCkpt
        )
        $trainArgs += $arm.trainFlags

        $null = Invoke-PythonRcCheck -Label "arm_${armId} train" -PyArgs $trainArgs
    }

    if (-not (Test-Path $armCkpt)) {
        Write-Host "[ERR] arm ${armId} ckpt missing: $armCkpt" -ForegroundColor Red
        continue
    }

    # Verify metadata
    if (Test-Path $armMeta) {
        $metaContent = Get-Content $armMeta -Raw | ConvertFrom-Json
        Write-Host "[i] arm ${armId} meta: flow_drop_xy=$($metaContent.flow_drop_xy) latent_dropout=$($metaContent.latent_dropout) geom_rich=$($metaContent.geom_feats_rich)" -ForegroundColor DarkGray
    }

    # Canonical coupled holdout evaluation
    $evalPyArgs = @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $armCkpt,
        "--mat-leg", "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--anchors", $HoldoutAnchors,
        "--out", $armHold
    )
    if ($CheapVal) { $evalPyArgs += "--cheap-val" }
    $null = Invoke-PythonRcCheck -Label "arm_${armId} holdout eval" -PyArgs $evalPyArgs

    $armMin = [int]((Get-Date) - $armStart).TotalMinutes
    Write-Host "[OK] arm ${armId} done in $armMin min" -ForegroundColor Green

    if (Test-Path $armHold) {
        $holdData = Get-Content $armHold -Raw | ConvertFrom-Json
        $p020 = $holdData.simple.per_anchor.patient020
        $summaryRows += [PSCustomObject]@{
            Arm          = $armId
            Label        = $arm.label
            ClotScore020 = if ($p020) { [math]::Round($p020.deploy_clot_score, 4) } else { "N/A" }
            StrictF1_020 = if ($p020) { [math]::Round($p020.deploy_clot_f1, 4) } else { "N/A" }
            OffwallF1    = if ($p020) { [math]::Round($p020.deploy_clot_offwall_strict_f1, 4) } else { "N/A" }
            MinsTrain    = $armMin
        }
    }
}

$elapsed = [int]((Get-Date) - $started).TotalMinutes

Write-Host "" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "[RESULTS] gen_recovery_sweep ($($armKeys.Count) arms, ${elapsed} min total)" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

Write-Host ""
Write-Host ("Reference baseline (no anti-memorization): clot_score=0.2996, strict_f1=0.2487, offwall=0.000") -ForegroundColor DarkGray
Write-Host ""

if ($summaryRows.Count -gt 0) {
    $summaryRows | Format-Table -AutoSize
}

$summaryJson = Join-Path $OutDir "sweep_summary.json"
$summaryObj = @{
    started_utc = $started.ToUniversalTime().ToString("o")
    elapsed_min = $elapsed
    arms_run    = $armKeys
    reference   = @{
        clot_score_020 = 0.2996
        strict_f1_020  = 0.2487
        offwall_strict_f1_020 = 0.000
    }
    results = @($summaryRows | ForEach-Object {
        @{
            arm          = $_.Arm
            label        = $_.Label
            clot_score   = $_.ClotScore020
            strict_f1    = $_.StrictF1_020
            offwall_f1   = $_.OffwallF1
            minutes      = $_.MinsTrain
        }
    })
}
$summaryObj | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $summaryJson

Write-Host "[i] summary -> $summaryJson" -ForegroundColor DarkGray
Write-Host "[OK] gen_recovery_sweep complete" -ForegroundColor Green
