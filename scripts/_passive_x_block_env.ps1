# I.1 X (species) block: clean GT-flow env on locked align init (no passive_transport preset).
# Dot-source after _passive_explore_base_env.ps1:
#   . (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")
#   . (Join-Path $PSScriptRoot "_passive_x_block_env.ps1")

function Set-PassiveXBlockBaseEnv {
    param(
        [string] $RunNote,
        [int] $Epochs = 6,
        [string] $TeacherLr = "1e-3",
        [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth"
    )
    Set-PassiveExploreBaseEnv -RunNote $RunNote -Epochs $Epochs -TeacherLr $TeacherLr -InitCkpt $InitCkpt
    # Probes: fewer val time points + species-only val (no duplicate mu rollout).
    $env:BIOCHEM_VAL_TIME_STRIDE = "40"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "2"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL_ONLY = "1"

    $env:BIOCHEM_EXPLORE_COMPONENT = "X"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1"
    $env:BIOCHEM_LOSS_ISOLATE = "PASSIVE"
    $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    $env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
    $env:BIOCHEM_ADR_EXCLUDE_WALL = "1"
    $env:BIOCHEM_ADR_RESIDUAL_MODE = "transport_only"
    $env:BIOCHEM_ADR_SPECIES_SCOPE = "all"
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0"
    $env:BIOCHEM_TRAIN_MU_ENCODER = "0"
    $env:BIOCHEM_TRAIN_BIO_ENCODER = "1"
    $env:BIOCHEM_TRAIN_BIO_DECODER = "1"
    $env:BIOCHEM_TRAIN_ODE = "1"
    $env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
    $env:BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL = "1"
    # Short probes: avoid skipped steps when global mask spikes bio grad.
    $env:BIOCHEM_TEACHER_GRAD_SCALE_ON_CAP = "1"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
}

function Set-PassiveXLegEnv {
    param(
        [string] $RunNote,
        [int] $Epochs = 6,
        [string] $TeacherLr = "1e-3",
        [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
        [ValidateSet("PASSIVE", "DATA_BIO")]
        [string] $LossIsolate = "PASSIVE",
        [ValidateSet("clot_band", "global")]
        [string] $BioMask = "clot_band",
        [ValidateSet("union", "last")]
        [string] $MaskTimes = "union",
        [string] $FiWeight = "3.0",
        [string] $MatWeight = "2.0",
        [switch] $AdrBackprop,
        [string] $AdrWeight = "1e-4"
    )
    Set-PassiveXBlockBaseEnv -RunNote $RunNote -Epochs $Epochs -TeacherLr $TeacherLr -InitCkpt $InitCkpt
    $env:BIOCHEM_LOSS_ISOLATE = $LossIsolate
    $env:BIOCHEM_DATA_BIO_MASK_MODE = $BioMask
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = $MaskTimes
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = $FiWeight
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = $MatWeight
    if ($AdrBackprop) {
        $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "1"
        $env:BIOCHEM_PASSIVE_ADR_WEIGHT = $AdrWeight
    } else {
        $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
        Remove-Item Env:\BIOCHEM_PASSIVE_ADR_WEIGHT -ErrorAction SilentlyContinue
    }
}

function Set-PassiveXTurboEnv {
    param(
        [string] $RunNote,
        [int] $Epochs = 2,
        [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth"
    )
    Set-PassiveXBlockBaseEnv -RunNote $RunNote -Epochs $Epochs -InitCkpt $InitCkpt
    $env:BIOCHEM_VAL_TIME_STRIDE = "50"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "2"
    $env:BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL = "0"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"
}
