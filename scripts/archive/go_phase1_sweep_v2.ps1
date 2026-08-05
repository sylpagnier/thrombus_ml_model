param(
    [int]    $Epochs       = 30,
    [int]    $EarlyStop    = 15,
    [int]    $MaxWindows   = 24,
    [string] $TrainAnchors = "patient005,patient006,patient010,patient023,patient002",
    [string] $ValAnchor    = "patient020",
    [string] $HoldoutAnchors = "patient020,patient034",
    [string] $RunRoot      = "outputs/biochem/eda/wall_gen_sweep_v2",
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

$baseEnv = @{
    "SPECIES_FLOW_FEATS_SOURCE" = "auto"
    "SPECIES_CLOSED_LOOP_COUPLING" = "1"
    "BIOCHEM_CORRECTOR_COUPLING" = "1"
    "SPECIES_SCHEDULED_SAMPLING" = "0"
    "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY" = "0"
}

$arms = [ordered]@{
    "01" = @{ label = "Safe Baseline (WC_v7)"; envVars = @{} }
    "02" = @{ label = "Geom Feats"; envVars = @{ "SPECIES_GEOM_FEATS_RICH" = "1" } }
    "03" = @{ label = "Flux Feat"; envVars = @{ "SPECIES_FLUX_STAG_FEAT" = "1" } }
    "04" = @{ label = "Full Species"; envVars = @{ "BIOCHEM_PUSHFORWARD_SPECIES_SCOPE" = "all" } }
    "05" = @{ label = "Wall Loss"; envVars = @{ "CLOT_PHI_PHYSICS_WALL_MAT_ONLY" = "1" } }
    "06" = @{ label = "Single Head"; envVars = @{ "SPECIES_CONTINUOUS_DUAL_HEAD" = "0" } }
    "07" = @{ label = "Teacher Noise (0.1)"; envVars = @{ "SPECIES_CONTINUOUS_TEACHER_NOISE" = "0.1" } }
    "08" = @{ label = "mat_growth_simple combo"; envVars = @{ 
        "SPECIES_CONTINUOUS_DUAL_HEAD" = "0";
        "CLOT_PHI_PHYSICS_WALL_MAT_ONLY" = "1" 
    } }
    "09" = @{ label = "Geom + Flux Combo"; envVars = @{ "SPECIES_GEOM_FEATS_RICH" = "1"; "SPECIES_FLUX_STAG_FEAT" = "1" } }
    "10" = @{ label = "Proper Mirror Y"; envVars = @{ "SPECIES_AUGMENT_MIRROR_Y" = "1" } }
}

$armKeys = if ($ArmFilter) {
    @($ArmFilter.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
} else {
    @($arms.Keys)
}

$started = Get-Date
Write-Host "[NEW] phase1_sweep_v2: $($armKeys.Count) arms, $Epochs ep" -ForegroundColor Cyan

foreach ($armId in $armKeys) {
    $arm = $arms[$armId]
    $armDir = Join-Path $OutDir "WG_sweep_$armId"
    $armCkpt = Join-Path $armDir "best.pth"
    $armLog = Join-Path $armDir "train_log.jsonl"
    $armHold = Join-Path $armDir "eval_holdout_cold.json"
    New-Item -ItemType Directory -Force -Path $armDir | Out-Null
    
    if (Test-Path $armHold) {
        Write-Host "[i] Arm $armId already completed (eval JSON exists). Skipping!" -ForegroundColor DarkGray
        continue
    }

    Write-Host "" -ForegroundColor Cyan
    Write-Host "====== Arm ${armId} : $($arm.label) ======" -ForegroundColor Cyan
    
    # Set Env
    foreach ($k in $baseEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $baseEnv[$k]) }
    foreach ($k in $arm.envVars.Keys) { [Environment]::SetEnvironmentVariable($k, $arm.envVars[$k]) }
    
    $trainArgs = @(
        "-m", "src.training.train_species_pushforward_continuous",
        "--phase", "biochem_gnn",
        "--recipe", "mat_growth_simple",
        "--leg", "WC_v7_clot_phi_mse",
        "--out", $armCkpt,
        "--epochs", "$Epochs",
        "--early-stop", "$EarlyStop",
        "--anchors", $TrainAnchors,
        "--val-anchor", $ValAnchor,
        "--exclude-val-from-train",
        "--no-init",
        "--drop-xy"
    )
    
    $null = Invoke-PythonRcCheck -Label "Arm $armId train" -PyArgs $trainArgs
    
    if (-not (Test-Path $armCkpt)) { 
        Write-Host "[WARN] Arm $armId failed to produce checkpoint!" -ForegroundColor Yellow
        continue 
    }

    $evalCkpt = Join-Path $armDir "last.pth"
    if (-not (Test-Path $evalCkpt)) {
        $evalCkpt = $armCkpt
    }

    # Evaluate
    $null = Invoke-PythonRcCheck -Label "Arm $armId eval" -PyArgs @(
        "scripts/eval_mat_growth_simple.py",
        "--ckpt", $evalCkpt,
        "--no-baseline",
        "--anchors", $HoldoutAnchors,
        "--out", $armHold
    )
    
    # Cleanup Env
    foreach ($k in $baseEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $null) }
    foreach ($k in $arm.envVars.Keys) { [Environment]::SetEnvironmentVariable($k, $null) }
}

Write-Host "" -ForegroundColor Cyan
Write-Host "[i] Aggregating results..." -ForegroundColor Cyan
python scripts/aggregate_sweep_v2.py
