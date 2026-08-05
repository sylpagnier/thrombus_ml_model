param(
    [int]    $Epochs       = 15,
    [int]    $EarlyStop    = 15,
    [int]    $MaxWindows   = 24,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020",
    [string] $RunRoot      = "outputs/biochem/eda/predicted_flow_probe",
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

$arms = [ordered]@{
    "A" = @{
        label = "Baseline (drop-xy, GT flow)"
        trainFlags = @("--drop-xy")
        envVars = @{ 
            "SPECIES_FLOW_FEATS_SOURCE" = "gt" 
            "SPECIES_CLOSED_LOOP_COUPLING" = "0"
        }
        epochs = $Epochs
    }
    "B" = @{
        label = "Predicted Flow (drop-xy, Kine flow)"
        trainFlags = @("--drop-xy")
        envVars = @{ 
            "SPECIES_FLOW_FEATS_SOURCE" = "kine" 
            "SPECIES_CLOSED_LOOP_COUPLING" = "0"
        }
        epochs = $Epochs
    }
    "C" = @{
        label = "Coupled Flow (drop-xy, Kine + Corrector)"
        trainFlags = @("--drop-xy")
        envVars = @{ 
            "SPECIES_FLOW_FEATS_SOURCE" = "auto" 
            "SPECIES_CLOSED_LOOP_COUPLING" = "1"
            "BIOCHEM_CORRECTOR_COUPLING" = "1"
        }
        epochs = $Epochs
    }
    "D" = @{
        label = "Geom-Rich (drop-xy, Kine + Corrector + Geom)"
        trainFlags = @("--drop-xy", "--geom-rich")
        envVars = @{ 
            "SPECIES_FLOW_FEATS_SOURCE" = "auto" 
            "SPECIES_CLOSED_LOOP_COUPLING" = "1"
            "BIOCHEM_CORRECTOR_COUPLING" = "1"
        }
        epochs = $Epochs
    }
}

$armKeys = if ($ArmFilter) {
    @($ArmFilter.Split(",") | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ })
} else {
    @($arms.Keys)
}

$started = Get-Date
Write-Host "[NEW] wall_predicted_flow_probe: $($armKeys.Count) arms, default $Epochs ep / ES $EarlyStop / max_windows $MaxWindows" -ForegroundColor Cyan
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
        # Remove global environment variable set to allow arms to control it via envVars
        
        # Set environment variables for this arm
        if ($arm.envVars) {
            foreach ($key in $arm.envVars.Keys) {
                Set-Item -Path "Env:\$key" -Value $arm.envVars[$key]
            }
        }

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
        
        # Unset environment variables
        if ($arm.envVars) {
            foreach ($key in $arm.envVars.Keys) {
                Remove-Item -Path "Env:\$key" -ErrorAction SilentlyContinue
            }
        }
    }

    if (-not (Test-Path $armCkpt)) {
        Write-Host "[ERR] arm ${armId} ckpt missing: $armCkpt" -ForegroundColor Red
        continue
    }

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
            WallScore020 = if ($p020) { [math]::Round($p020.deploy_wall_score, 4) } else { "N/A" }
            WallF1_020   = if ($p020) { [math]::Round($p020.deploy_wall_strict_f1, 4) } else { "N/A" }
            ClotScore020 = if ($p020) { [math]::Round($p020.deploy_clot_score, 4) } else { "N/A" }
            StrictF1_020 = if ($p020) { [math]::Round($p020.deploy_clot_f1, 4) } else { "N/A" }
            MinsTrain    = $armMin
        }
    }
}

$elapsed = [int]((Get-Date) - $started).TotalMinutes

Write-Host "" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "[RESULTS] wall_predicted_flow_probe ($($armKeys.Count) arms, ${elapsed} min total)" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan

if ($summaryRows.Count -gt 0) {
    $summaryRows | Format-Table -AutoSize
}

