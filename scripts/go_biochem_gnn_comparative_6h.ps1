# Comparative 6h: gate A/B (GraphSAGE) + species-scope sweep.
#
# Gate legs (fixed fi_mat scope):
#   GA  global_sigmoid adhesion gate (baseline)
#   GB  fourier_tau
#   GC  spatial_mlp (optional; skip with -Legs to save budget)
#
# Species legs (fixed global_sigmoid gate):
#   S0  fi_mat (2 ch: FI+Mat)           [baseline]
#   S1  fi_mat_thrombin (FI+Mat+thrombin)
#   S2  bulk9 (first 9 bulk channels)
#   S3  all (12 channels)
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_biochem_gnn_comparative_6h.ps1
#   powershell ... -Smoke
#   powershell ... -Fresh
#   powershell ... -Legs "GA,GB,S0,S1,S3"
#   powershell ... -TargetHours 6 -SkipTrain
#   powershell ... -SummaryAll          # summary from all completed legs under comparative_6h
#   powershell ... -SkipCompleted         # skip train when best.pth already exists

param(
    [double] $TargetHours = 6.0,
    [int]    $Epochs = 0,
    [int]    $EarlyStop = 0,
    [double] $Lr = 1.5e-4,
    [string] $Legs = "GA,GB,S0,S1,S3",
    [switch] $Fresh,
    [switch] $SkipTrain,
    [switch] $SkipCompleted,
    [switch] $SummaryAll,
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
. (Join-Path $PSScriptRoot "_python_rc.ps1")
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RunRoot = Join-Path $RepoRoot "outputs/biochem/biochem_gnn/comparative_6h"
$SummaryOut = Join-Path $RunRoot "comparative_summary.json"

$LegDefs = @{
    "GA" = @{ Suite = "gate"; Gate = "global_sigmoid"; Scope = "fi_mat"; Label = "gate_a_sigmoid" }
    "GB" = @{ Suite = "gate"; Gate = "fourier_tau";    Scope = "fi_mat"; Label = "gate_b_fourier" }
    "GC" = @{ Suite = "gate"; Gate = "spatial_mlp";    Scope = "fi_mat"; Label = "gate_c_spatial" }
    "S0" = @{ Suite = "species"; Gate = "global_sigmoid"; Scope = "fi_mat";           Label = "scope_fi_mat" }
    "S1" = @{ Suite = "species"; Gate = "global_sigmoid"; Scope = "fi_mat_thrombin"; Label = "scope_fi_mat_th" }
    "S2" = @{ Suite = "species"; Gate = "global_sigmoid"; Scope = "bulk9";            Label = "scope_bulk9" }
    "S3" = @{ Suite = "species"; Gate = "global_sigmoid"; Scope = "all";              Label = "scope_all12" }
}

$RequestedLegs = $Legs -split "," | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ }

if ($Smoke) {
    $Epochs = 8
    $EarlyStop = 4
    Write-Host "[i] smoke mode: $Epochs ep per leg" -ForegroundColor DarkGray
} elseif ($Epochs -le 0) {
    $nLegs = [Math]::Max($RequestedLegs.Count, 1)
    $minsPerLeg = ($TargetHours * 60.0) / $nLegs
    # ~2.9 min/epoch measured on 4GB GPU, 10 anchors, 40 ep (2026-06)
    $epEst = [int][Math]::Floor($minsPerLeg / 2.9)
    $Epochs = [Math]::Max(12, [Math]::Min(40, $epEst))
    $EarlyStop = [Math]::Max(6, [int][Math]::Floor($Epochs * 0.35))
    Write-Host "[i] budget mode: $TargetHours h / $nLegs legs -> $Epochs ep (early_stop=$EarlyStop)" -ForegroundColor DarkGray
} elseif ($EarlyStop -le 0) {
    $EarlyStop = [Math]::Max(6, [int][Math]::Floor($Epochs * 0.35))
}

$InitWarm = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
foreach ($cand in @(
    "outputs/biochem/biochem_gnn/global_guiding_5h/species/best.pth",
    "outputs/biochem/biochem_gnn/arch_ab/sage/species/best.pth",
    "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth"
)) {
    if (-not (Test-Path (Join-Path $RepoRoot $InitWarm))) {
        if (Test-Path (Join-Path $RepoRoot $cand)) { $InitWarm = $cand }
    }
}
Write-Host "[i] warm-start: $InitWarm" -ForegroundColor DarkGray

$SharedBeta = "outputs/biochem/biochem_gnn/locked/viscosity_beta.pth"
foreach ($cand in @(
    "outputs/biochem/biochem_gnn/global_guiding_5h/viscosity/beta.pth",
    "outputs/biochem/biochem_gnn/arch_ab/sage/viscosity/beta.pth"
)) {
    if (-not (Test-Path (Join-Path $RepoRoot $SharedBeta))) {
        if (Test-Path (Join-Path $RepoRoot $cand)) { $SharedBeta = $cand }
    }
}
Write-Host "[i] shared beta: $SharedBeta" -ForegroundColor DarkGray

function Set-ComparativeTrainEnv {
    param([string]$GateMode, [string]$Scope, [int]$Ep, [int]$ES, [double]$LR)
    $env:BIOCHEM_ADHESION_GATE = $GateMode
    $env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE = $Scope
    $env:SPECIES_PUSHFORWARD_ARCH = "sage"
    $env:SPECIES_CONTINUOUS_DEPLOY_EVAL_FULL = "1"
    $env:BIOCHEM_GELATION_PRIOR_GATE = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_TRAINING_LOG = "1"
    $env:TRAIN_EPOCHS = "$Ep"
    $env:TRAIN_EARLY_STOP_PATIENCE = "$ES"
    $env:TRAIN_LR = "$LR"
    # Train: GT flow for vel-decay (no GINO-DEQ on GPU). Eval: pred kine via manifest.
    $env:SPECIES_TRAIN_VEL_SOURCE = "gt"
    $env:SPECIES_ROLLOUT_DEPLOY_FAITHFUL = "1"
    $env:SPECIES_ROLLOUT_VEL_SOURCE = "kinematics"
    $env:SPECIES_ROLLOUT_PIN_OTHER = "rest"
    $env:SPECIES_ROLLOUT_IC_SOURCE = "resting"
    if ($Scope -eq "all" -or $Scope -eq "bulk9") {
        $env:SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS = "0"
    } else {
        Remove-Item Env:\SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS -ErrorAction SilentlyContinue
    }
}

function Test-LegTrained {
    param([string]$LegDir, [int]$MinEpochs = 0)
    $legBest = Join-Path $LegDir "species/best.pth"
    if (-not (Test-Path $legBest)) { return $false }
    if ($MinEpochs -le 0) { return $true }
    $meta = Join-Path $LegDir "species/best.json"
    if (-not (Test-Path $meta)) { return $true }
    try {
        $bj = Get-Content $meta -Raw | ConvertFrom-Json
        $ep = [int]$bj.epoch
        return ($ep -ge $MinEpochs)
    } catch {
        return $true
    }
}

function Test-LegEvaluated {
    param([string]$LegDir)
    $eval = Join-Path $LegDir "eval/deploy_ab_eval.json"
    $ckpt = Join-Path $LegDir "species/best.pth"
    if (-not (Test-Path $eval)) { return $false }
    if (-not (Test-Path $ckpt)) { return $true }
    return (Get-Item $eval).LastWriteTime -ge (Get-Item $ckpt).LastWriteTime
}

function Collect-AllEvalDirs {
    $out = @{}
    foreach ($k in $LegDefs.Keys) {
        $d = Join-Path $RunRoot $LegDefs[$k].Label
        if (Test-LegEvaluated $d) { $out[$k] = Join-Path $d "eval" }
    }
    return $out
}

function Train-ComparativeLeg {
    param([string]$LegKey, [hashtable]$Def, [string]$LegDir)
    $speciesDir = Join-Path $LegDir "species"
    $viscDir = Join-Path $LegDir "viscosity"
    $legBest = Join-Path $speciesDir "best.pth"
    New-Item -ItemType Directory -Force -Path $speciesDir, $viscDir | Out-Null

    if ($SkipCompleted -and (Test-LegTrained -LegDir $legDir -MinEpochs $Epochs) -and -not $Fresh) {
        Write-Host "[skip] [$LegKey] checkpoint exists (epoch>=$Epochs) -> $legBest" -ForegroundColor DarkGray
        return
    }

    $legBeta = Join-Path $viscDir "beta.pth"
    if (-not (Test-Path $legBeta) -and (Test-Path (Join-Path $RepoRoot $SharedBeta))) {
        Copy-Item (Join-Path $RepoRoot $SharedBeta) $legBeta -Force
    }

    $legInit = ""
    if (-not $Fresh -and (Test-Path $legBest)) {
        $legInit = $legBest
    } elseif (Test-Path (Join-Path $RepoRoot $InitWarm)) {
        $legInit = Join-Path $RepoRoot $InitWarm
    }

    Set-ComparativeTrainEnv -GateMode $Def.Gate -Scope $Def.Scope -Ep $Epochs -ES $EarlyStop -LR $Lr
    $env:BIOCHEM_GNN_TRAIN_OUTPUT_DIR = $speciesDir

    $cmd = @(
        "-m", "src.training.train_biochem_gnn",
        "--step", "species",
        "--arch", "sage",
        "--epochs", "$Epochs",
        "--lr", "$Lr",
        "--species-out", (Join-Path $speciesDir "best.pth")
    )
    if ($legInit -ne "") {
        if ($Def.Scope -ne "fi_mat") {
            Write-Host "[i] [$LegKey] partial warm-start (scope=$($Def.Scope) from fi_mat ckpt)" -ForegroundColor DarkGray
        }
        $cmd += @("--init", $legInit)
    }
    if ($Fresh) { $cmd += "--fresh" }

    Write-Host "[run] [$LegKey] gate=$($Def.Gate) scope=$($Def.Scope) ($Epochs ep)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$LegKey] train" -PyArgs $cmd
    Write-Host "[OK] [$LegKey] training done" -ForegroundColor Green
}

$p007DeployTime = "200"
$p007TsFile = Join-Path $RepoRoot "outputs/biochem/patient007_deploy_t_s.txt"
if (Test-Path $p007TsFile) {
    $p007DeployTime = (Get-Content $p007TsFile -Raw).Trim()
}
$DeployTimes = "53,$p007DeployTime"
Write-Host "[i] deploy eval times: $DeployTimes" -ForegroundColor DarkGray

function Write-ComparativeManifest {
    param([string]$LegKey, [hashtable]$Def, [string]$LegDir)
    $speciesDir = Join-Path $LegDir "species"
    $viscDir = Join-Path $LegDir "viscosity"
    $legBest = Join-Path $speciesDir "best.pth"
    $legBeta = Join-Path $viscDir "beta.pth"
    if (-not (Test-Path $legBest)) { return "" }
    $mfPath = Join-Path $RepoRoot "data/reference/biochem_gnn_comparative_$($LegKey.ToLower()).json"
    $betaPath = if (Test-Path $legBeta) { $legBeta } else { Join-Path $RepoRoot $SharedBeta }
    $relCkpt = (Resolve-Path $legBest).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
    $relBeta = (Resolve-Path $betaPath).Path.Substring($RepoRoot.Length).TrimStart('\').Replace('\', '/')
    $payload = @{
        name     = "biochem_gnn_comparative_$($LegKey.ToLower())"
        version  = 1
        baseline = @{
            species_gnn_ckpt  = $relCkpt
            viscosity_beta    = $relBeta
            kinematics_ckpt   = "outputs/kinematics/kinematics_best.pth"
            train_val_anchor  = "patient007"
            flow_modes        = "kinematics"
            gamma_mode        = "max"
            deploy_horizon    = "full"
            clot_score        = "guiding"
            pushforward_arch  = "sage"
            gate_mode         = $Def.Gate
            species_scope     = $Def.Scope
            loao_auto         = "0"
        }
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $mfPath) | Out-Null
    $json = $payload | ConvertTo-Json -Depth 6
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($mfPath, $json, $utf8NoBom)
    Write-Host "[i] [$LegKey] manifest -> $mfPath" -ForegroundColor DarkGray
    return $mfPath
}

function Eval-ComparativeLeg {
    param([string]$LegKey, [hashtable]$Def, [string]$LegDir, [string]$ManifestPath)
    $evalDir = Join-Path $LegDir "eval"
    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    $evalOut = Join-Path $evalDir "deploy_ab_eval.json"
    $env:BIOCHEM_ADHESION_GATE = $Def.Gate
    $env:BIOCHEM_PUSHFORWARD_SPECIES_SCOPE = $Def.Scope
    Write-Host "[run] [$LegKey] eval deploy_frozen (times=$DeployTimes)" -ForegroundColor Cyan
    $null = Invoke-PythonRcCheck -Label "[$LegKey] eval" -PyArgs @(
        "scripts/eval_biochem_gnn_deploy_ab.py",
        "--manifest", $ManifestPath,
        "--modes", "deploy_frozen",
        "--times", $DeployTimes,
        "--out", $evalOut
    )
    Write-Host "[OK] [$LegKey] eval -> $evalOut" -ForegroundColor Green
    return $evalDir
}

Write-Host ""
Write-Host "=== Biochem GNN comparative ($TargetHours h budget) ===" -ForegroundColor White
Write-Host "Legs: $($RequestedLegs -join ', ')" -ForegroundColor DarkGray
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$EvalDirs = @{}
foreach ($LegKey in $RequestedLegs) {
    if (-not $LegDefs.ContainsKey($LegKey)) {
        Write-Host "[WARN] unknown leg '$LegKey'; skip" -ForegroundColor Yellow
        continue
    }
    $def = $LegDefs[$LegKey]
    $legDir = Join-Path $RunRoot $def.Label
    Write-Host ""
    Write-Host "--- $LegKey ($($def.Suite): gate=$($def.Gate) scope=$($def.Scope)) ---" -ForegroundColor Yellow
    if (-not $SkipTrain) { Train-ComparativeLeg -LegKey $LegKey -Def $def -LegDir $legDir }
    $mf = Write-ComparativeManifest -LegKey $LegKey -Def $def -LegDir $legDir
    if ($mf -ne "") {
        if ($SkipCompleted -and (Test-LegEvaluated $legDir) -and -not $Fresh) {
            Write-Host "[skip] [$LegKey] eval exists" -ForegroundColor DarkGray
            $EvalDirs[$LegKey] = Join-Path $legDir "eval"
        } else {
            $EvalDirs[$LegKey] = Eval-ComparativeLeg -LegKey $LegKey -Def $def -LegDir $legDir -ManifestPath $mf
        }
    }
}

if ($SummaryAll) {
    $allEval = Collect-AllEvalDirs
    foreach ($k in $allEval.Keys) {
        if (-not $EvalDirs.ContainsKey($k)) { $EvalDirs[$k] = $allEval[$k] }
    }
}

if ($EvalDirs.Count -gt 0) {
    $meta = @{
        target_hours = $TargetHours
        epochs = $Epochs
        early_stop = $EarlyStop
        legs = $RequestedLegs
    }
    # Write JSON to disk: PowerShell strips backslashes when passing JSON on the CLI.
    $dirsFile = Join-Path $RunRoot "_eval_dirs_summary.json"
    $metaFile = Join-Path $RunRoot "_eval_meta_summary.json"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($dirsFile, ($EvalDirs | ConvertTo-Json -Compress), $utf8NoBom)
    [System.IO.File]::WriteAllText($metaFile, ($meta | ConvertTo-Json -Compress), $utf8NoBom)
    $null = Invoke-PythonRcCheck -Label "comparative summary" -PyArgs @(
        "scripts/summarize_biochem_gnn_comparative.py",
        "--eval-dirs-file", $dirsFile,
        "--meta-file", $metaFile,
        "--output", $SummaryOut
    )
    Write-Host "[OK] summary -> $SummaryOut" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Comparative done ===" -ForegroundColor White
