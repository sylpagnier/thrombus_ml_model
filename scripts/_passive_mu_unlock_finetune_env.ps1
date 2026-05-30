# Post mu-unlock finetune: MU_LOG + wall/high-mu weights, lower LR, init from unlock-best ckpt.
. (Join-Path $PSScriptRoot "_passive_mu_unlock_env.ps1")

function Set-PassiveMuUnlockFinetuneEnv {
    param(
        [string] $RunNote = "passive_mu_unlock_finetune",
        [int] $Epochs = 8,
        [string] $TeacherLr = "5e-4",
        [string] $MuRatioMax = "20",
        [string] $MuLogWeight = "0.5",
        [string] $MuLogWallWeight = "0.75",
        [string] $MuLogHighWeight = "1.5",
        [string] $MuSiWeight = "0.15"
    )
    Set-PassiveMuUnlockEnv -RunNote $RunNote -Epochs $Epochs -TeacherLr $TeacherLr `
        -MuRatioMax $MuRatioMax -MuLogWeight $MuLogWeight -MuSiWeight $MuSiWeight
    $env:BIOCHEM_PASSIVE_MU_UNLOCK_FINETUNE = "1"
    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = $MuLogWeight
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = $MuLogWallWeight
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = $MuLogHighWeight
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = $MuSiWeight
    $env:BIOCHEM_TEACHER_TARGET_MU_LOG_MAE = "0.65"
}
