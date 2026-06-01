# K10e/K10f/K10g env for warm-start from passive teacher (skip pretrain, GT_KINE_VEL=1).

function Set-PassiveK10FromInitEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RunNote,
        [Parameter(Mandatory = $true)]
        [int] $Epochs,
        [ValidateSet("wide", "narrow", "bias")]
        [string] $Variant = "wide"
    )

    $env:KINEMATICS_USE_HARD_BCS = "1"
    $env:KINEMATICS_USE_WIDTH_PRIORS = "1"
    # K10E isolate is deprecated in biochem_loss_policy.py; required for M5 K10 explore legs.
    $env:BIOCHEM_LEGACY_LOSSES = "1"
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
    $env:BIOCHEM_MU_K10E_SIMPLE = "1"
    Remove-Item Env:BIOCHEM_MU_K10D_SIMPLE, Env:BIOCHEM_K10G_ORACLE_CLOTS -ErrorAction SilentlyContinue
    $env:BIOCHEM_MU_IC_STEADY_KIN = "1"
    $env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_USE_SPLIT_MU_HEAD = "0"
    Remove-Item Env:BIOCHEM_USE_WALL_DELTA_HEAD, Env:BIOCHEM_MU_ADDITIVE_DELTA -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
    $env:BIOCHEM_USE_MU_PATH_GROUP = "1"
    $env:BIOCHEM_TRAIN_ODE = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "0"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "0"
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_LOSS_ISOLATE = "K10E"
    $env:BIOCHEM_COMPLEXITY_STEP = "2"
    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = "1.0"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
    $env:BIOCHEM_DETACH_MACRO_STATE = "1"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "2"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_RUN_NOTE = $RunNote
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    Remove-Item Env:BIOCHEM_LOSS_DATA_ONLY, Env:BIOCHEM_MU_LOG_WALL_WEIGHT, Env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT -ErrorAction SilentlyContinue

    Remove-Item Env:BIOCHEM_K10E_DELTA_BIAS_ND -ErrorAction SilentlyContinue
    switch ($Variant) {
        "wide" {
            $env:BIOCHEM_K10E_D_PEAK_ND = "0.008"
            $env:BIOCHEM_K10E_SIGMA_ND = "0.008"
            $env:BIOCHEM_K10E_SDF_MAX_ND = "0.04"
            $env:BIOCHEM_K10E_MU_DELTA_ND_MAX = "30"
            $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
            $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "4.0"
            $env:BIOCHEM_MU_LOG_ADJACENT_WEIGHT = "6.0"
            $env:BIOCHEM_K10E_BULK_DELTA_WEIGHT = "0.5"
            $env:BIOCHEM_K10E_DATA_KINE_WEIGHT = "0.25"
        }
        "narrow" {
            $env:BIOCHEM_K10E_D_PEAK_ND = "0.004"
            $env:BIOCHEM_K10E_SIGMA_ND = "0.0035"
            $env:BIOCHEM_K10E_SDF_MAX_ND = "0.02"
            $env:BIOCHEM_K10E_MU_DELTA_ND_MAX = "18"
            $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "1.0"
            $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "4.0"
            $env:BIOCHEM_MU_LOG_ADJACENT_WEIGHT = "3.0"
            $env:BIOCHEM_K10E_BULK_DELTA_WEIGHT = "2.0"
            $env:BIOCHEM_K10E_DATA_KINE_WEIGHT = "0.25"
        }
        "bias" {
            $env:BIOCHEM_K10E_D_PEAK_ND = "0.008"
            $env:BIOCHEM_K10E_SIGMA_ND = "0.008"
            $env:BIOCHEM_K10E_SDF_MAX_ND = "0.04"
            $env:BIOCHEM_K10E_MU_DELTA_ND_MAX = "30"
            $env:BIOCHEM_K10E_DELTA_BIAS_ND = "17"
            $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.5"
            $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "2.0"
            $env:BIOCHEM_MU_LOG_ADJACENT_WEIGHT = "2.0"
            $env:BIOCHEM_K10E_BULK_DELTA_WEIGHT = "0.0"
            $env:BIOCHEM_K10E_DATA_KINE_WEIGHT = "1.0"
        }
    }
    $env:BIOCHEM_K10E_CORONA_GROWTH = "1"
    $env:BIOCHEM_K10E_CORONA_MIX = "0.4"
}
