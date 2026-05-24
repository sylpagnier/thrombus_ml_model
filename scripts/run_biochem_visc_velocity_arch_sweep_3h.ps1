# ~3 h architecture sweep: localized high-μ near walls + healthy Carreau bulk + non-degenerate |u|.
# Each leg saves its own teacher checkpoint for visualization (does not rely on the global
# outputs/biochem/biochem_teacher_best_high_mu.pth after the next leg).
#
# Physics map (what each leg tests):
#   μ_eff = μ_kin(u,v) × (1 + μ₁(Mat) + μ₂(FI) + learned_gel + exp(Δlogμ))
#   - μ_kin: frozen Carreau backbone (velocity must stay sane here)
#   - μ₁/μ₂: explicit COMSOL triggers from species (full-domain FI → clot swallow)
#   - Δlogμ: split bulk/tail + optional wall head (bleed / gate saturation failures)
#
# One line (unattended, leaves full console log):
#   powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\pgssy\thrombus_ml_model\scripts\go_visc3h.ps1"
#
# From repo root:
#   .\scripts\go_visc3h.ps1
#   .\scripts\run_biochem_visc_velocity_arch_sweep_3h.ps1
#   .\scripts\run_biochem_visc_velocity_arch_sweep_3h.ps1 -DryRun
#   .\scripts\run_biochem_visc_velocity_arch_sweep_3h.ps1 -Legs L0_mufreeze_ref,L1_softwall_learn
#   .\scripts\run_biochem_visc_velocity_arch_sweep_3h.ps1 -ForcePretrain   # cold AE+ODE on leg 1 only
#
# After sweep, visualize one leg:
#   python -m src.evaluation.visualize_pipeline --biochem-checkpoint outputs\biochem\sweep_visc_velocity_3h\L0_mufreeze_ref\biochem_teacher_best_high_mu.pth
#
# Artifacts:
#   outputs\biochem\sweep_visc_velocity_3h\<leg_id>\biochem_teacher_best_high_mu.pth
#   outputs\biochem\sweep_visc_velocity_3h\manifest.jsonl
#   outputs\reports\training\biochem\visc_velocity_arch_sweep_3h_summary.txt
#
param(
    [int] $TeacherEp = 18,
    [int] $ValEvery = 2,
    [string[]] $Legs = @(),
    [switch] $ForcePretrain,
    [switch] $NoRecoverOrphan,
    [switch] $DryRun,
    [string[]] $ExtraArgs = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$hostName = $env:COMPUTERNAME
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\visc_velocity_arch_sweep_3h_summary.txt"
$SweepDir = Join-Path $RepoRoot "outputs\biochem\sweep_visc_velocity_3h"
$ManifestPath = Join-Path $SweepDir "manifest.jsonl"
$PostPretrain = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$TeacherBest = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth"
$TeacherLast = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
$RunsIndex = Join-Path $RepoRoot "outputs\reports\training\biochem\runs_index.jsonl"

if (-not (Test-Path (Split-Path $SummaryPath))) {
    New-Item -ItemType Directory -Path (Split-Path $SummaryPath) -Force | Out-Null
}
if (-not (Test-Path $SweepDir)) {
    New-Item -ItemType Directory -Path $SweepDir -Force | Out-Null
}

# Leg definitions: Architecture = how μ is composed; Loss = what trains it.
$LegCatalog = [ordered]@{
    L0_mufreeze_ref = @{
        Title = 'Reference: data leash + mu-only teacher (good early velocity)'
        Hypothesis = 'L_Data_Kine leashes flow; mu-path fits wall clot; ODE/bio frozen. Risk: gate_clot saturates late.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef }
    }
    L1_softwall_learn = @{
        Title = 'mu-freeze + wall-only soft gate + learnable temperature'
        Hypothesis = 'Cut wall-delta bleed differentiably; do not sharpen bulk clot gate (avoids gate_all~0.96).'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegSoftWallLearned }
    }
    L2_relu_wall = @{
        Title = 'mu-freeze + ReLU wall residual (Fix D)'
        Hypothesis = 'Wall mu via ReLU(delta_wall) on mask nodes; avoids gate_wall starvation.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegReluWall }
    }
    L3_wall_decay = @{
        Title = 'mu-freeze + exponential wall-delta spatial decay'
        Hypothesis = 'Wall corrections decay into lumen (SDF); bulk stays Carreau-like.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegWallDecay }
    }
    L4_kine_lora = @{
        Title = 'Data leash + mu-freeze + tiny kin LoRA (rank 4)'
        Hypothesis = 'Minimal kinematics adaptation when mu rises near wall without bio/ODE unfreeze.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegKinLora }
    }
    L5_mu_log_suppress = @{
        Title = 'MU_LOG isolate + strong bio gate suppressor (no leash)'
        Hypothesis = 'Direct mu objective; suppressor gates tail on FI/Mat; best historical val mu, check velocity.'
        Preset = ''
        Apply = { Set-LegMuLogSuppress }
    }
    L6_sentinel_leash = @{
        Title = 'sweep_wall_sentinel + supervised data leash'
        Hypothesis = 'Sentinel wall weights plus kine/bio leash: wall recipe plus flow anchor.'
        Preset = 'sweep_wall_sentinel'
        Apply = { Set-LegSentinelLeash }
    }
    L7_early_stop = @{
        Title = 'mu-freeze + soft wall gate + early stop at val all logMAE 0.65'
        Hypothesis = 'Stop before late clot swallow (ref run peaked ~0.57 ep12 then regressed).'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegSoftWallLearned; Set-LegEarlyStop }
    }
}

function Clear-BiochemEnv {
    Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
    }
}

function Set-SweepGpuBase {
    param([int]$Ep, [int]$ValEvery = 2)

    Clear-BiochemEnv
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_LOSS_DATA_ONLY = "1"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Ep"
    $env:BIOCHEM_EPOCHS = "$Ep"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Ep"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "$ValEvery"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "0.35"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
    $env:BIOCHEM_USE_MU_PATH_GROUP = "1"
    $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_USE_SPLIT_MU_HEAD = "1"
    $env:BIOCHEM_USE_WALL_DELTA_HEAD = "1"
    $env:BIOCHEM_WALL_HEAD_ISOLATE_GEOM = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "1"
    $env:BIOCHEM_DEBUG = "0"
    # expandable_segments is unsupported on Windows CUDA builds (stderr warning can kill PS sweeps).
    if ($IsLinux -or $IsMacOS) {
        $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    } else {
        Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
    }
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_LORA_RANK = "0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_TRIGGER_GATE_MIN = "0"
    $env:BIOCHEM_WALL_GATE_MIN = "0"
}

function Set-LegMuFreezeRef {
    $env:BIOCHEM_SUPERVISED_DATA_LEASH = "1"
    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "2.0"
    $env:BIOCHEM_BULK_FLUID_SURGICAL_FIX = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
    $env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
    $env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_ODE_WARMUP_EPOCHS = "999"
}

function Set-LegSoftWallLearned {
    $env:BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH = "0.15"
    $env:BIOCHEM_MU_SOFT_GATE_SCOPE = "wall_only"
    $env:BIOCHEM_MU_GATE_LEARNED_TEMP = "1"
    $env:BIOCHEM_MU_TRIGGER_GATE_HARD_STEEPNESS = "10.0"
}

function Set-LegReluWall {
    $env:BIOCHEM_MU_WALL_MIX_MODE = "relu_add"
    $env:BIOCHEM_MU_WALL_HEAD_ACTIVATION = "relu"
    $env:BIOCHEM_MU_WALL_DELTA_GAIN = "0.85"
}

function Set-LegWallDecay {
    $env:BIOCHEM_WALL_SPATIAL_DECAY = "1"
    $env:BIOCHEM_WALL_SPATIAL_DECAY_FACTOR = "7.0"
    $env:BIOCHEM_WALL_SPATIAL_DECAY_FLOOR = "0.05"
}

function Set-LegKinLora {
    Set-LegMuFreezeRef
    $env:BIOCHEM_TRAIN_KIN_LORA = "1"
    $env:BIOCHEM_LORA_RANK = "4"
}

function Set-LegMuLogSuppress {
    $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    $env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
    $env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"
    $env:BIOCHEM_BIO_SUPPRESSOR_THRESHOLD_SI = "5e-4"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_ODE_WARMUP_EPOCHS = "999"
    Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH -ErrorAction SilentlyContinue
}

function Set-LegSentinelLeash {
    $env:BIOCHEM_SUPERVISED_DATA_LEASH = "1"
    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "2.0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_BULK_FLUID_SURGICAL_FIX = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
    $env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
    $env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_ODE_WARMUP_EPOCHS = "999"
}

function Set-LegEarlyStop {
    $env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE = "0.65"
}

function Invoke-TrainingPython {
    param([string[]]$Cmd)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    # Do not return python stdout on the pipeline (breaks: $exitCode = Invoke-TrainingPython).
    & python @Cmd *>&1 | ForEach-Object { Write-Host $_ }
    $code = [int]$LASTEXITCODE
    $ErrorActionPreference = $prevEap
    return $code
}

function Get-LegCheckpointPath {
    param([string]$LegId)
    return Join-Path $SweepDir (Join-Path $LegId "biochem_teacher_best_high_mu.pth")
}

function Get-RunIndexRow {
    param([string]$RunNote)
    if (-not (Test-Path $RunsIndex)) { return $null }
    $rows = Get-Content $RunsIndex | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.run_note -eq $RunNote }
    if (-not $rows) { return $null }
    return @($rows)[-1]
}

function Try-RecoverOrphanLeg {
    param(
        [string]$LegId,
        [string]$RunNote,
        [string]$Title,
        [string]$Hypothesis
    )
    if ($NoRecoverOrphan) { return $false }
    $legCkpt = Get-LegCheckpointPath -LegId $LegId
    if (Test-Path $legCkpt) { return $false }
    $row = Get-RunIndexRow -RunNote $RunNote
    if (-not $row) { return $false }
    if (-not (Test-Path $TeacherBest)) { return $false }
    $ckptPath = Save-LegArtifacts -LegId $LegId -RunNote $RunNote -Title $Title -Hypothesis $Hypothesis
    Write-Host "RECOVERED $legId from completed run (run_id=$($row.run_id)) -> $ckptPath" -ForegroundColor Cyan
    Add-Content -Path $SummaryPath -Value "RECOVER leg=$legId run_id=$($row.run_id) ts=$(Get-Date -Format o)"
    return $true
}

function Set-WarmStartEnv {
    param([bool]$UseWarm, [bool]$IsFirstLeg)

    if ($UseWarm -and (-not $ForcePretrain)) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    } elseif ($IsFirstLeg -and (-not $ForcePretrain)) {
        $env:BIOCHEM_SKIP_PRETRAIN = "0"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    } else {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    }
}

function Save-LegArtifacts {
    param(
        [string]$LegId,
        [string]$RunNote,
        [string]$Title,
        [string]$Hypothesis
    )

    $legDir = Join-Path $SweepDir $LegId
    New-Item -ItemType Directory -Path $legDir -Force | Out-Null

    $ckptDest = Join-Path $legDir "biochem_teacher_best_high_mu.pth"
    $srcBest = $TeacherBest
    if ($env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR) {
        $archBest = Join-Path $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR "biochem_teacher_best_high_mu.pth"
        if (Test-Path $archBest) { $srcBest = $archBest }
    }
    if (-not (Test-Path $srcBest)) {
        throw "Missing teacher checkpoint after leg $LegId : $srcBest"
    }
    Copy-Item -Path $srcBest -Destination $ckptDest -Force
    if (Test-Path $TeacherLast) {
        Copy-Item -Path $TeacherLast -Destination (Join-Path $legDir "biochem_teacher_last.pth") -Force
    }

    $valAll = ""
    $valWall = ""
    $valHigh = ""
    $bestEp = ""
    $runId = ""
    if (Test-Path $RunsIndex) {
        $match = Get-Content $RunsIndex | ForEach-Object { $_ | ConvertFrom-Json } |
            Where-Object { $_.run_note -eq $RunNote } |
            Select-Object -Last 1
        if ($match) {
            $valAll = [string]$match.val_mu_log_mae
            $valWall = [string]$match.val_mu_log_mae_wall
            $valHigh = [string]$match.val_mu_log_mae_high_mu
            $bestEp = [string]$match.best_epoch
            $runId = [string]$match.run_id
        }
    }

    $manifestRow = @{
        leg_id = $LegId
        title = $Title
        hypothesis = $Hypothesis
        run_note = $RunNote
        run_id = $runId
        best_epoch = $bestEp
        val_mu_log_mae = $valAll
        val_mu_log_mae_wall = $valWall
        val_mu_log_mae_high_mu = $valHigh
        checkpoint = $ckptDest.Replace("\", "/")
        viz_cmd = "python -m src.evaluation.visualize_pipeline --biochem-checkpoint `"$ckptDest`""
    } | ConvertTo-Json -Compress

    Add-Content -Path $ManifestPath -Value $manifestRow
    return $ckptDest
}

if ($Legs.Count -eq 0) {
    $Legs = @($LegCatalog.Keys)
}

$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $PostPretrain)
$legTotal = $Legs.Count
$batchStart = Get-Date
$batchIso = $batchStart.ToString("o")

Write-Host ""
Write-Host "Viscosity+velocity architecture sweep (~3h target)" -ForegroundColor Cyan
Write-Host "  legs=$legTotal  teacher_ep=$TeacherEp  val_every=$ValEvery  host=$hostName" -ForegroundColor DarkGray
Write-Host "  warm_post_pretrain=$UseWarmStart  archive=$SweepDir" -ForegroundColor DarkGray
Write-Host ""

Add-Content -Path $SummaryPath -Value "BATCH_START host=$hostName ts=$batchIso legs=$legTotal ep=$TeacherEp warm=$UseWarmStart"

$legIndex = 0
$firstExecutedLeg = $true
foreach ($legId in $Legs) {
    if (-not $LegCatalog.Contains($legId)) {
        throw "Unknown leg id: $legId (valid: $($LegCatalog.Keys -join ', '))"
    }
    $legIndex++
    $def = $LegCatalog[$legId]
    $title = $def.Title
    $hypothesis = $def.Hypothesis
    $preset = [string]$def.Preset
    $runNote = "visc3h_${legId}_ep${TeacherEp}_${hostName}"
    $legCkpt = Get-LegCheckpointPath -LegId $legId

    Write-Host ""
    Write-Host "========== [$legIndex/$legTotal] $legId ==========" -ForegroundColor Yellow
    Write-Host "  $title" -ForegroundColor DarkGray
    Write-Host "  H: $hypothesis" -ForegroundColor DarkGray

    if (Test-Path $legCkpt) {
        Write-Host "SKIP $legId (archived checkpoint exists)" -ForegroundColor DarkGray
        Add-Content -Path $SummaryPath -Value "SKIP leg=$legId archived=1 ts=$(Get-Date -Format o)"
        continue
    }
    if (Try-RecoverOrphanLeg -LegId $legId -RunNote $runNote -Title $title -Hypothesis $hypothesis) {
        continue
    }

    Set-SweepGpuBase -Ep $TeacherEp -ValEvery $ValEvery
    $env:BIOCHEM_PRESET = $preset
    & $def.Apply
    $UseWarmNow = (-not $ForcePretrain) -and ((Test-Path $PostPretrain) -or (-not $firstExecutedLeg))
    Set-WarmStartEnv -UseWarm $UseWarmNow -IsFirstLeg $firstExecutedLeg

    $env:BIOCHEM_RUN_NOTE = $runNote
    $legDir = Join-Path $SweepDir $legId
    New-Item -ItemType Directory -Path $legDir -Force | Out-Null
    $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $legDir

    $cmd = @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--run-name", $runNote,
        "--epochs", "$TeacherEp", "--save-best"
    )
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    Add-Content -Path $SummaryPath -Value "START leg=$legId preset=$preset note=$runNote ts=$(Get-Date -Format o)"

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        Write-Host "  PRESET=$env:BIOCHEM_PRESET  LEASH=$env:BIOCHEM_SUPERVISED_DATA_LEASH  ISOLATE=$env:BIOCHEM_LOSS_ISOLATE" -ForegroundColor DarkGray
        continue
    }

    $t0 = Get-Date
    $exitCode = [int](Invoke-TrainingPython -Cmd $cmd)
    if ($exitCode -ne 0) {
        $failLine = "FAIL leg=$legId exit=$exitCode ts=$(Get-Date -Format o)"
        Add-Content -Path $SummaryPath -Value $failLine
        throw $failLine
    }
    $firstExecutedLeg = $false

    $ckptPath = Save-LegArtifacts -LegId $legId -RunNote $runNote -Title $title -Hypothesis $hypothesis
    $mins = [int]((Get-Date) - $t0).TotalMinutes
    $okLine = "OK leg=$legId duration=${mins}m ckpt=$ckptPath val_note=$runNote ts=$(Get-Date -Format o)"
    Add-Content -Path $SummaryPath -Value $okLine
    Write-Host $okLine -ForegroundColor Green

    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

$batchMins = [int]((Get-Date) - $batchStart).TotalMinutes
Add-Content -Path $SummaryPath -Value "BATCH_OK host=$hostName duration=${batchMins}m ts=$(Get-Date -Format o)"
Write-Host ""
Write-Host "Sweep complete | ${batchMins}m | manifest: $ManifestPath" -ForegroundColor Green
Write-Host "Compare legs: Get-Content '$ManifestPath' | ForEach-Object { `$_ | ConvertFrom-Json } | Format-Table leg_id,val_mu_log_mae,val_mu_log_mae_wall" -ForegroundColor DarkGray
