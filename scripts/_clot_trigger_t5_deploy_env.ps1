# Star 5 (T5): deploy-faithful species teacher retrain (pred GINO-DEQ + FI/Mat neighbor band).
# Dot-source from go_clot_trigger_t5.ps1.

. (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
. (Join-Path $PSScriptRoot "_gnode10_env.ps1")

function Set-ClotTriggerT5DeployTrainEnv {
    param(
        [string] $RunNote = "clot_trigger_t5_deploy",
        [int] $TeacherEpochs = 12,
        [string] $SpeciesScope = "fi_mat",
        [string] $KineCkpt = "outputs/kinematics/kinematics_best.pth"
    )
    Set-Gnode10PredictedKineBaseEnv -RunNote $RunNote -Epochs $TeacherEpochs

    $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_DATA_BIO_SPECIES_SCOPE = $SpeciesScope

    # Neighbor shell (clot-phi deploy band) instead of clot_band-only.
    $env:BIOCHEM_DATA_BIO_MASK_MODE = "neighbor"
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    $env:BIOCHEM_ADR_MASK_MODE = "match_data_bio"
    $env:BIOCHEM_ADR_EXCLUDE_WALL = "1"
    $env:BIOCHEM_MU_DISABLE_EXPLICIT_GELATION = "1"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL_ONLY = "1"

    $env:CLOT_PHI_KINE_CKPT = $KineCkpt
    $env:CLOT_TRIGGER_STAR = "t5"
}
