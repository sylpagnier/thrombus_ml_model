# Step-2 bridge: LOSS_DATA_ONLY + modest mu aux + low-weight transport_only ADR (not COMPLEXITY_STEP=3).
# Uses clean explore base (no passive_transport preset, no post_pretrain clobber).
. (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")

function Set-PassiveStep2BridgeEnv {
    param(
        [string] $RunNote = "passive_step2_bridge",
        [int] $Epochs = 12,
        [string] $TeacherLr = "1e-3",
        [string] $AdrWeight = "1e-4",
        [string] $MuLogWeight = "0.75",
        [string] $MuSiWeight = "0.15",
        [string] $InitCkpt = ""
    )
    Set-PassiveExploreLegEnv -RunNote $RunNote -Component "XY" -Epochs $Epochs -TeacherLr $TeacherLr `
        -Step2Bridge -AdrBackprop -AdrWeight $AdrWeight -MuLogWeight $MuLogWeight -MuSiWeight $MuSiWeight `
        -MuRatioMax "1" -DataBioMask "clot_band" -MaskTimes "union" -FiWeight "3.0" -MatWeight "2.0" `
        -InitCkpt $InitCkpt
}
