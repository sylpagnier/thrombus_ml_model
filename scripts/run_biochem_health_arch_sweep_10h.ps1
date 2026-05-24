# ~10 h overnight sweep: viz-aligned health metrics + Gemini μ fix + simple residual + μ₁/μ₂ probes.
# Each leg writes its own checkpoint under outputs/biochem/sweep_health_arch_10h/<leg_id>/ via
# BIOCHEM_ARCHIVE_CHECKPOINT_DIR (not the global outputs/biochem/biochem_teacher_best_high_mu.pth).
#
# One line (repo root, full ~10h sweep — includes K0 Carreau kinematic probe first):
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_health10h.ps1"
#   .\scripts\go_health10h.ps1
#
# Morning viz (one leg):
#   python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint "outputs\biochem\sweep_health_arch_10h\G0_gemini_leash\biochem_teacher_best_high_mu.pth"
#
# Compare health scores (lower = healthier for rollout):
#   Get-Content outputs\biochem\sweep_health_arch_10h\manifest.jsonl | ForEach-Object { $_ | ConvertFrom-Json } | Sort-Object viz_health_score | Format-Table leg_id, viz_health_score, val_mu_log_mae, viz_t0_speed_mean, viz_final_mu2_mean, viz_final_clot_frac

param(
    [int] $TeacherEp = 22,
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
$SummaryPath = Join-Path $RepoRoot "outputs\reports\training\biochem\health_arch_sweep_10h_summary.txt"
$SweepDir = Join-Path $RepoRoot "outputs\biochem\sweep_health_arch_10h"
$ManifestPath = Join-Path $SweepDir "manifest.jsonl"
$PostPretrain = Join-Path $RepoRoot "outputs\biochem\biochem_post_pretrain.pth"
$RunsIndex = Join-Path $RepoRoot "outputs\reports\training\biochem\runs_index.jsonl"

if (-not (Test-Path (Split-Path $SummaryPath))) {
    New-Item -ItemType Directory -Path (Split-Path $SummaryPath) -Force | Out-Null
}
if (-not (Test-Path $SweepDir)) {
    New-Item -ItemType Directory -Path $SweepDir -Force | Out-Null
}

# Default overnight order: kinematic sanity probe first, then architecture legs (~10h total).
$DefaultLegOrder = @(
    "K0_carreau_kinematic",
    "R0_ref_leash",
    "G0_gemini_leash",
    "G1_gemini_mu_log",
    "S0_simple_residual",
    "S1_simple_residual_leash",
    "M0_mu2_cap_leash",
    "M1_mu1_only_leash",
    "M2_no_explicit_gel"
)

$LegCatalog = [ordered]@{
    K0_carreau_kinematic = @{
        Title = 'Quick: Carreau-only μ, no clot (DATA_KINE, bio/ODE frozen)'
        Hypothesis = 'Baseline kinematics+rollout before biochem μ/clot; expect healthy t=0 |u|, μ₂~0.'
        Preset = ''
        QuickEp = 8
        Apply = { Set-LegCarreauKinematic }
    }
    G0_gemini_leash = @{
        Title = 'Gemini additive μ + data leash + sentinel wall weights'
        Hypothesis = 'Symmetric bulk clip, additive Δlogμ, bio suppressor; fixes t=0 μ collapse and μ₂ flood.'
        # Do not use preset sweep_wall_sentinel here: it forces USE_BIO_GATE_SUPPRESSOR=0 and overwrites Gemini.
        Preset = ''
        Apply = { Set-LegGemini; Set-LegSentinelLeash; Set-LegBulkSurgicalFix }
    }
    G1_gemini_mu_log = @{
        Title = 'Gemini fix + MU_LOG isolate (no leash)'
        Hypothesis = 'Pure μ objective with stable residual path; check if flow leash still needed.'
        Preset = 'sweep_gemini'
        Apply = { Set-LegGemini; Set-LegMuLogOnly }
    }
    S0_simple_residual = @{
        Title = 'Simple log residual: μ_kin*exp(Δlogμ) only (no μ₁/μ₂ multiplier)'
        Hypothesis = 'Can the model memorize μ field without gelation multiplier breaking kine?'
        Preset = ''
        Apply = { Set-LegSimpleResidual; Set-LegMuLogOnly }
    }
    S1_simple_residual_leash = @{
        Title = 'Simple log residual + supervised data leash'
        Hypothesis = 'Simple μ path + flow anchor; best chance of healthy t=0 |u|.'
        Preset = ''
        Apply = { Set-LegSimpleResidual; Set-LegMuFreezeRef }
    }
    M0_mu2_cap_leash = @{
        Title = 'Data leash + cap μ₂ sigmoid at 8 (limit FI flood)'
        Hypothesis = 'High μ₂ drives clot-everywhere; cap explicit FI path while keeping μ₁.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegMu2Cap }
    }
    M1_mu1_only_leash = @{
        Title = 'Data leash + disable μ₂ explicit gelation (Mat-only)'
        Hypothesis = 'Force wall Mat trigger; stop FI domain flood.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegMu1Only }
    }
    M2_no_explicit_gel = @{
        Title = 'Data leash + disable all explicit gelation (delta heads only)'
        Hypothesis = 'Learn μ only via neural Δlogμ; isolate bad μ₁/μ₂ sigmoid coupling.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef; Set-LegNoExplicitGel }
    }
    R0_ref_leash = @{
        Title = 'Reference: current data-leash + mu-freeze stack (no Gemini)'
        Hypothesis = 'Baseline repro of visc3h L0 pathology for A/B.'
        Preset = ''
        Apply = { Set-LegMuFreezeRef }
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
    $env:BIOCHEM_TEACHER_KEEP_GLOBAL_BEST = "0"
    $env:BIOCHEM_DEBUG = "0"
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
    $env:BIOCHEM_VIZ_MU2_FLOOD_THRESH = "10.0"
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

function Set-LegSentinelLeash {
    Set-LegMuFreezeRef
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "3.6"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "1.8"
}

function Set-LegBulkSurgicalFix {
    $env:BIOCHEM_BULK_FLUID_SURGICAL_FIX = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
}

function Set-LegGemini {
    $env:BIOCHEM_MU_GEMINI_FIX = "1"
    $env:BIOCHEM_DELTA_MU_SYMMETRIC_BULK_CLIP = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.05"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_WALL = "5.0"
    $env:BIOCHEM_MU_SOFT_GATE_SCOPE = "wall_only"
    $env:BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH = "0.15"
    $env:BIOCHEM_USE_BIO_GATE_SUPPRESSOR = "1"
    $env:BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR = "0.0"
}

function Set-LegMuLogOnly {
    $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_ODE_WARMUP_EPOCHS = "999"
}

function Set-LegSimpleResidual {
    $env:BIOCHEM_MU_SIMPLE_LOG_RESIDUAL = "1"
    $env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
    $env:BIOCHEM_USE_WALL_DELTA_HEAD = "0"
    $env:BIOCHEM_DELTA_MU_SYMMETRIC_BULK_CLIP = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP_BULK = "0.15"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP = "1.5"
}

function Set-LegMu2Cap {
    $env:BIOCHEM_MU2_SIGMOID_CAP = "8.0"
    $env:BIOCHEM_BIO_SUPPRESSOR_THRESHOLD_SI = "1e-4"
}

function Set-LegMu1Only {
    $env:BIOCHEM_MU_DISABLE_MU2 = "1"
}

function Set-LegNoExplicitGel {
    $env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
}

function Set-LegCarreauKinematic {
    # μ_eff = μ_kin(γ̇) only; species ODE/bio frozen; train flow anchor only.
    $env:BIOCHEM_MU_CARREAU_ONLY = "1"
    $env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
    $env:BIOCHEM_GELATION_PRIOR_GATE = "0"
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "0"
    $env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
    $env:BIOCHEM_USE_WALL_DELTA_HEAD = "0"
    $env:BIOCHEM_LOSS_ISOLATE = "DATA_KINE"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0.0"
    Remove-Item Env:BIOCHEM_SUPERVISED_DATA_LEASH -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TRAIN_MU_ENCODER = "0"
    $env:BIOCHEM_ODE_WARMUP_EPOCHS = "999"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
}

function Invoke-TrainingPython {
    param([string[]]$Cmd)
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
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

function Get-RunJsonlBestVal {
    param([string]$RunId)
    if (-not $RunId) { return $null }
    $runPath = Join-Path $RepoRoot "outputs\reports\training\biochem\$RunId\run.jsonl"
    if (-not (Test-Path $runPath)) { return $null }
    $best = $null
    $bestScore = [double]::PositiveInfinity
    foreach ($line in Get-Content $runPath) {
        $row = $line | ConvertFrom-Json
        if ($row.event -ne "val") { continue }
        $score = $row.val_viz_health_score
        if ($null -ne $score -and [double]$score -lt $bestScore) {
            $bestScore = [double]$score
            $best = $row
        } elseif ($null -eq $best) {
            $best = $row
        }
    }
    return $best
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
    if (-not (Test-Path $ckptDest)) {
        throw "Missing archived checkpoint for leg $LegId at $ckptDest (BIOCHEM_ARCHIVE_CHECKPOINT_DIR must match leg dir)."
    }

    $valAll = ""
    $valWall = ""
    $valHigh = ""
    $bestEp = ""
    $runId = ""
    $vizScore = ""
    $vizT0Speed = ""
    $vizMu2 = ""
    $vizClot = ""
    $match = Get-RunIndexRow -RunNote $RunNote
    if ($match) {
        $valAll = [string]$match.val_mu_log_mae
        $valWall = [string]$match.val_mu_log_mae_wall
        $valHigh = [string]$match.val_mu_log_mae_high_mu
        $bestEp = [string]$match.best_epoch
        $runId = [string]$match.run_id
    }
    $valRow = Get-RunJsonlBestVal -RunId $runId
    if ($valRow) {
        if ($valRow.val_viz_health_score) { $vizScore = [string]$valRow.val_viz_health_score }
        if ($valRow.val_viz_t0_speed_mean) { $vizT0Speed = [string]$valRow.val_viz_t0_speed_mean }
        if ($valRow.val_viz_final_mu2_mean) { $vizMu2 = [string]$valRow.val_viz_final_mu2_mean }
        if ($valRow.val_viz_final_clot_frac) { $vizClot = [string]$valRow.val_viz_final_clot_frac }
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
        viz_health_score = $vizScore
        viz_t0_speed_mean = $vizT0Speed
        viz_final_mu2_mean = $vizMu2
        viz_final_clot_frac = $vizClot
        checkpoint = $ckptDest.Replace("\", "/")
        viz_cmd = "python -m src.evaluation.visualize_pipeline --teacher-only --biochem-checkpoint `"$ckptDest`""
    } | ConvertTo-Json -Compress

    Add-Content -Path $ManifestPath -Value $manifestRow
    return $ckptDest
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

if ($Legs.Count -eq 0) {
    $Legs = @($DefaultLegOrder)
}

$UseWarmStart = (-not $ForcePretrain) -and (Test-Path $PostPretrain)
$legTotal = $Legs.Count
$batchStart = Get-Date
Add-Content -Path $SummaryPath -Value "BATCH_START host=$hostName ts=$($batchStart.ToString('o')) legs=$legTotal default_ep=$TeacherEp order=$($Legs -join ',')"

Write-Host ""
Write-Host "Health architecture sweep (~10h)" -ForegroundColor Cyan
Write-Host "  legs=$legTotal  default_ep=$TeacherEp  (K0 uses QuickEp=8)" -ForegroundColor DarkGray
Write-Host "  order: $($Legs -join ' -> ')" -ForegroundColor DarkGray
Write-Host "  archive: $SweepDir" -ForegroundColor DarkGray
Write-Host ""

$legIndex = 0
$firstExecutedLeg = $true
foreach ($legId in $Legs) {
    if (-not $LegCatalog.Contains($legId)) {
        throw "Unknown leg: $legId"
    }
    $legIndex++
    $def = $LegCatalog[$legId]
    $legDir = Join-Path $SweepDir $legId
    $legCkpt = Get-LegCheckpointPath -LegId $legId
    $legEp = $TeacherEp
    if ($null -ne $def.QuickEp) { $legEp = [int]$def.QuickEp }
    $runNote = "health10h_${legId}_ep${legEp}_${hostName}"

    Write-Host ""
    Write-Host "========== [$legIndex/$legTotal] $legId (ep=$legEp) ==========" -ForegroundColor Yellow
    Write-Host "  $($def.Title)" -ForegroundColor DarkGray

    if (Test-Path $legCkpt) {
        Write-Host "SKIP $legId (checkpoint exists)" -ForegroundColor DarkGray
        continue
    }
    Set-SweepGpuBase -Ep $legEp -ValEvery $ValEvery
    $preset = [string]$def.Preset
    if ($preset) { $env:BIOCHEM_PRESET = $preset }
    & $def.Apply
    $env:BIOCHEM_RUN_NOTE = $runNote
    $env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR = $legDir

    $UseWarmNow = (-not $ForcePretrain) -and ((Test-Path $PostPretrain) -or (-not $firstExecutedLeg))
    Set-WarmStartEnv -UseWarm $UseWarmNow -IsFirstLeg $firstExecutedLeg

    $cmd = @(
        "-m", "src.training.train_biochem_corrector",
        "--new", "--run-name", $runNote,
        "--epochs", "$legEp", "--save-best"
    )
    if ($ExtraArgs.Count -gt 0) { $cmd += $ExtraArgs }

    if ($DryRun) {
        Write-Host "DryRun: python $($cmd -join ' ')" -ForegroundColor DarkGray
        Write-Host "  ARCHIVE=$env:BIOCHEM_ARCHIVE_CHECKPOINT_DIR PRESET=$env:BIOCHEM_PRESET GEMINI=$env:BIOCHEM_MU_GEMINI_FIX" -ForegroundColor DarkGray
        continue
    }

    $t0 = Get-Date
    $exitCode = [int](Invoke-TrainingPython -Cmd $cmd)
    if ($exitCode -ne 0) {
        throw "FAIL leg=$legId exit=$exitCode"
    }
    $firstExecutedLeg = $false
    $ckptPath = Save-LegArtifacts -LegId $legId -RunNote $runNote -Title $def.Title -Hypothesis $def.Hypothesis
    $mins = [int]((Get-Date) - $t0).TotalMinutes
    Write-Host "OK leg=$legId ${mins}m -> $ckptPath" -ForegroundColor Green
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

Write-Host ""
Write-Host "Sweep complete | manifest: $ManifestPath" -ForegroundColor Green
Write-Host "Sort by viz health: Get-Content '$ManifestPath' | % { `$_ | ConvertFrom-Json } | Sort-Object { [double]`$_.viz_health_score } | ft leg_id,viz_health_score,val_mu_log_mae,viz_t0_speed_mean,viz_final_mu2_mean" -ForegroundColor DarkGray
