# Base env for Phase A passive probes (GT flow, teacher-only, no kin training).
# Dot-source: . (Join-Path $PSScriptRoot "_passive_phase_a_env.ps1")

function Set-PhaseAPassiveBaseEnv {
    param(
        [string] $RunNote,
        [string] $LossIsolate = "PASSIVE",
        [string] $TeacherLr = "1e-3",
        [string] $PhysClip = "10",
        [int] $OdeFreezeEpochs = 0,
        [string] $BioMask = "clot_band",
        [string] $FiWeight = "3.0",
        [string] $MatWeight = "2.0"
    )
    $env:PYTHONHASHSEED = "420"
    $env:CUBLAS_WORKSPACE_CONFIG = ":16:8"
    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = $RunNote
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"
    $env:BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS = "$OdeFreezeEpochs"
    $env:BIOCHEM_TRAIN_ODE = "1"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
    $env:BIOCHEM_TEACHER_LR = $TeacherLr
    $env:BIOCHEM_TEACHER_PHYSICS_CLIP_NORM = $PhysClip
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_LOSS_ISOLATE = $LossIsolate
    $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = $BioMask
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = $FiWeight
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = $MatWeight
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    Remove-Item Env:\BIOCHEM_PASSIVE_ADR_WEIGHT -ErrorAction SilentlyContinue
}

function Set-PhaseAYHarnessEnv {
    param([string] $RunNote, [string] $LossIsolate, [string] $TeacherLr = "1e-3")
    Set-PhaseAPassiveBaseEnv -RunNote $RunNote -LossIsolate $LossIsolate -TeacherLr $TeacherLr `
        -PhysClip "10" -OdeFreezeEpochs 0
}
