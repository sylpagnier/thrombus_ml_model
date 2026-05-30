# Passive mu-unlock probe: MU_LOG-only backward, frozen bio/ODE, no passive_transport preset.
# Prereq: species-aligned ckpt (20ep align or locked), NOT a failed mu-unlock last.pth.

function Set-PassiveMuUnlockEnv {
    param(
        [string] $RunNote = "passive_mu_unlock_probe",
        [int] $Epochs = 12,
        [string] $TeacherLr = "1e-3",
        [string] $MuRatioMax = "20",
        [string] $MuLogWeight = "1.0",
        [string] $MuSiWeight = "0.25"
    )
    Remove-Item Env:\BIOCHEM_PRESET -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_PASSIVE_STEP2_BRIDGE -ErrorAction SilentlyContinue
    Remove-Item Env:\BIOCHEM_LOSS_DATA_ONLY -ErrorAction SilentlyContinue

    $env:PYTHONHASHSEED = "420"
    $env:CUBLAS_WORKSPACE_CONFIG = ":16:8"
    $env:BIOCHEM_PASSIVE_MU_UNLOCK = "1"
    $env:BIOCHEM_PASSIVE_MU_UNLOCK_FREEZE_BIO = "1"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = $RunNote
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    # Do not reload biochem_post_pretrain.pth after teacher init (clobbers species-aligned ckpt).
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"
    $env:BIOCHEM_TEACHER_ODE_FREEZE_EPOCHS = "0"
    $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = $MuRatioMax
    $env:BIOCHEM_TEACHER_LR = $TeacherLr
    $env:BIOCHEM_TEACHER_PHYSICS_CLIP_NORM = "10"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_TBPTT_ANCHOR_RANDOM_START = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = $MuLogWeight
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = $MuSiWeight
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    $env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
    $env:BIOCHEM_ADR_EXCLUDE_WALL = "1"
    $env:BIOCHEM_ADR_RESIDUAL_MODE = "transport_only"
    $env:BIOCHEM_ADR_SPECIES_SCOPE = "all"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
    $env:BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL = "1"
    $env:BIOCHEM_WARN_SMALL_SUPERVISION_MASK = "1"
}
