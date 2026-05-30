# Clean passive/GT-flow teacher env for exploration (no passive_transport preset, no post_pretrain clobber).
# Dot-source: . (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")

function Set-PassiveExploreBaseEnv {
    param(
        [string] $RunNote,
        [int] $Epochs = 8,
        [string] $TeacherLr = "1e-3",
        [string] $InitCkpt = ""
    )
    Remove-Item Env:\BIOCHEM_PRESET -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_MU_UNLOCK -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_MU_UNLOCK_FINETUNE -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_STEP2_BRIDGE -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_LOSS_DATA_ONLY -ErrorAction SilentlyContinue

    $env:PYTHONHASHSEED = "420"
    $env:CUBLAS_WORKSPACE_CONFIG = ":16:8"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = $RunNote
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"
    $env:BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TEACHER_LR = $TeacherLr
    $env:BIOCHEM_TEACHER_PHYSICS_CLIP_NORM = "10"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
    $env:BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL = "1"
    $env:BIOCHEM_WARN_SMALL_SUPERVISION_MASK = "1"

    if ($InitCkpt) {
        $env:BIOCHEM_EXPLORE_INIT_CKPT = $InitCkpt
    } else {
        Remove-Item Env:\BIOCHEM_EXPLORE_INIT_CKPT -ErrorAction SilentlyContinue
    }
}

function Set-PassiveExploreLegEnv {
    param(
        [string] $RunNote,
        [ValidateSet("X", "Y", "XY")]
        [string] $Component,
        [int] $Epochs = 8,
        [string] $TeacherLr = "1e-3",
        [string] $LossIsolate = "",
        [switch] $LossDataOnly,
        [switch] $Step2Bridge,
        [switch] $MuUnlock,
        [switch] $MuUnlockFinetune,
        [string] $MuRatioMax = "1",
        [switch] $TrainMu,
        [switch] $FreezeBio,
        [switch] $AdrBackprop,
        [string] $AdrWeight = "1e-4",
        [string] $DataBioMask = "clot_band",
        [string] $MaskTimes = "union",
        [string] $AdrResidualMode = "transport_only",
        [string] $MuLogWeight = "0",
        [string] $MuSiWeight = "0",
        [string] $MuLogWallWeight = "0",
        [string] $MuLogHighWeight = "0",
        [string] $FiWeight = "3.0",
        [string] $MatWeight = "2.0",
        [string] $InitCkpt = ""
    )
    Set-PassiveExploreBaseEnv -RunNote $RunNote -Epochs $Epochs -TeacherLr $TeacherLr -InitCkpt $InitCkpt

    $env:BIOCHEM_EXPLORE_COMPONENT = $Component
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = $MuRatioMax
    $env:BIOCHEM_DATA_BIO_MASK_MODE = $DataBioMask
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = $FiWeight
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = $MatWeight
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = $MaskTimes
    $env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
    $env:BIOCHEM_ADR_EXCLUDE_WALL = "1"
    $env:BIOCHEM_ADR_RESIDUAL_MODE = $AdrResidualMode
    $env:BIOCHEM_ADR_SPECIES_SCOPE = "all"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = $MuLogWeight
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = $MuSiWeight
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = $MuLogWallWeight
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = $MuLogHighWeight
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"

    if ($LossDataOnly) {
        $env:BIOCHEM_LOSS_DATA_ONLY = "1"
    } else {
        $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    }
    if ($LossIsolate) {
        $env:BIOCHEM_LOSS_ISOLATE = $LossIsolate
    }
    if ($Step2Bridge) {
        $env:BIOCHEM_PASSIVE_STEP2_BRIDGE = "1"
        $env:BIOCHEM_LOSS_DATA_ONLY = "1"
        Remove-Item Env:\BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
    }
    if ($MuUnlock -or $MuUnlockFinetune) {
        $env:BIOCHEM_PASSIVE_MU_UNLOCK = "1"
        $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
        $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
        $env:BIOCHEM_LOSS_DATA_ONLY = "0"
        $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
        $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
        $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
        $env:BIOCHEM_TRAIN_ODE = "0"
        if ($MuUnlockFinetune) {
            $env:BIOCHEM_PASSIVE_MU_UNLOCK_FINETUNE = "1"
        }
    } elseif ($TrainMu) {
        $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
        $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
        $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
        $env:BIOCHEM_TRAIN_ODE = "0"
        $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    } else {
        $env:BIOCHEM_TRAIN_MU_ENCODER = "0"
        $env:BIOCHEM_TRAIN_BIO_ENCODER = "1"
        $env:BIOCHEM_TRAIN_BIO_DECODER = "1"
        $env:BIOCHEM_TRAIN_ODE = "1"
        $env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
    }
    if ($FreezeBio) {
        $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
        $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
        $env:BIOCHEM_TRAIN_ODE = "0"
    }
    if ($AdrBackprop) {
        $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "1"
        $env:BIOCHEM_PASSIVE_ADR_WEIGHT = $AdrWeight
    } else {
        $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    }
}
