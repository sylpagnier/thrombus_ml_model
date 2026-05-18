# Shared helpers for teacher complexity marathon scripts (laptop A / B).
# Dot-sourced by run_biochem_teacher_complexity_laptop_*.ps1 — do not run directly.

function Clear-BiochemCudaCache {
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
}

function Clear-BiochemEnv {
    Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
    }
}

function Set-BiochemMuPathDefaults {
    $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
    $env:BIOCHEM_USE_MU_PATH_GROUP = "1"
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_DELTA_MU_LOG_CLIP = "2.0"
    $env:BIOCHEM_MU_SI_MULTI_STEP = "1"
}

function Set-BiochemMarathonBase {
    param(
        [string] $RepoRoot,
        [bool] $UseWarmStart,
        [int] $DefaultTeacherEpochs,
        [string] $TbpttWindow = "4",
        [string] $DetachMacro = "1",
        [string] $Rk4Sub = "10",
        [string] $MachineTag = "marathon"
    )

    $script:BiochemMarathonRepoRoot = $RepoRoot

    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_STOCK_DEFAULTS = "1"
    $env:BIOCHEM_PRESET = ""
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_LOSS_DATA_ONLY = "1"
    $env:BIOCHEM_SKIP_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = if ($UseWarmStart) { "1" } else { "0" }
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "20.0"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
    $env:BIOCHEM_TEACHER_EPOCHS = "$DefaultTeacherEpochs"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "3"
    $env:BIOCHEM_TEACHER_SKIP_VAL = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = $TbpttWindow
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DETACH_MACRO_STATE = $DetachMacro
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = $Rk4Sub
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_DEBUG = "0"
    $env:BIOCHEM_LOW_ANCHOR_MODE = "0"
    $env:BIOCHEM_FI_GATE_START_WEIGHT = "0.0"
    $env:BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
    $env:BIOCHEM_COMSOL_TEMPORAL_WEIGHT = "0.02"
    $env:BIOCHEM_MU_SI_HUBER_DELTA = "0.25"
    $env:BIOCHEM_TBPTT_ANCHOR_END_BIAS = "0"
    $env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    $env:BIOCHEM_TEACHER_TBPTT_RANDOM_ANCHOR = "0"
    Remove-Item Env:BIOCHEM_MAX_LOAD_VESSELS -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_MAX_LOAD_SHUFFLE -ErrorAction SilentlyContinue
    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue

    if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
        $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    }

    Set-BiochemMuPathDefaults
}

function Invoke-BiochemTeacherLeg {
    param(
        [string] $LegKey,
        [hashtable] $LegDef,
        [hashtable] $Base,
        [hashtable] $EarlyWindow,
        [bool] $UseWarmStart,
        [int] $LegIndex,
        [int] $LegTotal,
        [string] $SummaryPath,
        [string[]] $ExtraArgs,
        [switch] $DryRun
    )

    $label = $LegDef.Label
    if ($LegIndex -gt 1) { Clear-BiochemCudaCache }

    Write-Host ""
    Write-Host "========== [$LegKey] $label ($LegIndex / $LegTotal) ==========" -ForegroundColor Cyan

    Clear-BiochemEnv
    foreach ($k in $Base.Keys) { Set-Item -Path "Env:$k" -Value $Base[$k] }
    foreach ($k in $EarlyWindow.Keys) { Set-Item -Path "Env:$k" -Value $EarlyWindow[$k] }

    foreach ($k in $LegDef.Keys) {
        if ($k -in @("Label", "ClearLossIsolate", "TeacherEpochs")) { continue }
        Set-Item -Path "Env:$k" -Value $LegDef[$k]
    }
    if ($LegDef.ClearLossIsolate) {
        Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    }
    if ($null -ne $LegDef.TeacherEpochs) {
        $env:BIOCHEM_TEACHER_EPOCHS = "$($LegDef.TeacherEpochs)"
    }

    $env:BIOCHEM_RUN_NOTE = $label

    if ($LegIndex -gt 1 -or $UseWarmStart) {
        $env:BIOCHEM_SKIP_PRETRAIN = "1"
        $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    }

    Write-Host "  isolate=$($env:BIOCHEM_LOSS_ISOLATE)  W_MuLog=$($env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT)  W_MuSI=$($env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT)" -ForegroundColor DarkGray
    Write-Host "  ep=$($env:BIOCHEM_TEACHER_EPOCHS)  TBPTT=$($env:BIOCHEM_TBPTT_MAX_WINDOW)  DETACH=$($env:BIOCHEM_DETACH_MACRO_STATE)  PhysTemp=$($env:BIOCHEM_DATA_ONLY_PHYS_TEMP)" -ForegroundColor DarkGray

    if ($DryRun) {
        Add-Content -Path $SummaryPath -Value "Leg $LegKey DRYRUN $label"
        return
    }

    $t0 = Get-Date
    python -m src.training.train_biochem_corrector --new @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        $line = "Leg $LegKey FAILED exit=$LASTEXITCODE at $(Get-Date -Format o) note=$label"
        Add-Content -Path $SummaryPath -Value $line
        Write-Host $line -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Clear-BiochemCudaCache
    $dt = (Get-Date) - $t0
    $line = "Leg $LegKey OK $label duration=$([int]$dt.TotalMinutes)m"
    Add-Content -Path $SummaryPath -Value $line
    Write-Host $line -ForegroundColor Green
}
