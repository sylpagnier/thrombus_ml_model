# Shared passive-align recipe (M3 union mask + transport_only ADR). Used by 20ep confirm + step2 bridge base.
. (Join-Path $PSScriptRoot "_m3_align_env.ps1")

function Set-PassiveAlignRecipeEnv {
    param(
        [string] $RunNote = "passive_align_transport_union",
        [int] $Epochs = 12,
        [string] $TeacherLr = "1e-3",
        [string] $AdrWeight = "1e-4",
        [switch] $SpeciesTrainEval
    )
    Set-M3AlignProbeEnv -RunNote $RunNote -Epochs $Epochs -TeacherLr $TeacherLr -AdrWeight $AdrWeight
    $env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
    if ($SpeciesTrainEval) {
        $env:BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL = "1"
    } else {
        Remove-Item Env:\BIOCHEM_PASSIVE_SPECIES_TRAIN_EVAL -ErrorAction SilentlyContinue
    }
    Remove-Item Env:\BIOCHEM_PASSIVE_STEP2_BRIDGE -ErrorAction SilentlyContinue
}
