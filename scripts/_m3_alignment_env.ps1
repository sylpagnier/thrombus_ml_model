# M3 ADR/data alignment sweep env helpers.

function Set-M3AlignmentBaseEnv {
    param(
        [string] $RunNote,
        [string] $TeacherLr = "1e-3",
        [string] $TfMin = "1.0"
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
    $env:BIOCHEM_TEACHER_FORCE_MIN = $TfMin
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"
    $env:BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS = "0"
    $env:BIOCHEM_TRAIN_ODE = "1"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"
    $env:BIOCHEM_TEACHER_LR = $TeacherLr
    $env:BIOCHEM_TEACHER_PHYSICS_CLIP_NORM = "10"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_LOSS_ISOLATE = "PASSIVE"
    $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "1"
    $env:BIOCHEM_PASSIVE_ADR_WEIGHT = "1e-3"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    # Defaults (legs override)
    $env:BIOCHEM_ADR_MASK_MODE = "global"
    Remove-Item Env:\BIOCHEM_ADR_EXCLUDE_WALL -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_ADR_FAST_TRANSIENT -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_WALL_BACKPROP -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_WALL_WEIGHT -ErrorAction SilentlyContinue
}

function Clear-M3AlignmentOverrides {
    Remove-Item Env:\BIOCHEM_ADR_MASK_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_ADR_EXCLUDE_WALL -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_ADR_FAST_TRANSIENT -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_WALL_BACKPROP -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_WALL_WEIGHT -ErrorAction SilentlyContinue
}
