# Comprehensive mu sweep: GINO-DEQ (pred kine) + GNODE Phase3 + teacher then corrector/synth.
# Dot-source: . (Join-Path $PSScriptRoot "_mu_complexity_sweep_env.ps1")

. (Join-Path $PSScriptRoot "_gnode11_env.ps1")

function Resolve-MuComplexityInitCkpt {
    param([string] $UserPath = "")
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    if ($UserPath) {
        $p = if ([System.IO.Path]::IsPathRooted($UserPath)) { $UserPath } else { Join-Path $RepoRoot $UserPath }
        if (Test-Path $p) { return $p }
    }
    foreach ($rel in @(
            "outputs\biochem\gnode10_sweep\gnode12_lane_a_promoted\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish_lane12a\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode12_mu_unlock\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\biochem_teacher_best_high_mu.pth"
        )) {
        $c = Join-Path $RepoRoot $rel
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Set-MuComprehensivePredKine {
    # Stage-A GINO-DEQ in the macro loop (no COMSOL [u,v,p] substitute).
    $env:BIOCHEM_GT_KINE_VEL = "0"
    Remove-Item Env:\BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_KIN_LORA = "1"
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.15"
}

function Set-MuComprehensiveDeployMu {
    param([string] $MuRatioMax = "20")
    # Promoted Lane A init already uses delta-mu head; allow clot-viscosity feedback in rollouts.
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = $MuRatioMax
    $env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
    $env:BIOCHEM_TRAIN_MU_ENCODER = "1"
}

function Apply-MuComprehensiveLegEnv {
    param(
        [Parameter(Mandatory = $true)]
        [string] $LegId,
        [Parameter(Mandatory = $true)]
        [string] $RunNote,
        [int] $TeacherEpochs = 10,
        [int] $CorrectorEpochs = 14
    )

    Clear-Gnode10BiochemEnv

    switch ($LegId) {
        "smoke" {
            Set-Gnode11CorrectorSmokeEnv -ComplexityStep "2" -RunNote $RunNote `
                -TeacherEpochs $TeacherEpochs -CorrectorEpochs $CorrectorEpochs `
                -TeacherForceMin "0.5" -KineWeight "0.15" -AdrWeight "1e-4" `
                -MuLogWeight "0.75" -MuSiWeight "0.15"
            $env:BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE = "-2.0"
            $env:BIOCHEM_SYNTH_PSEUDO_WEIGHT = "0.35"
            $env:BIOCHEM_WARMUP_EPOCHS = "1"
            $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
        }
        "FULL_step2" {
            Set-Gnode11FinishEnv -RunNote $RunNote -TeacherEpochs $TeacherEpochs `
                -CorrectorEpochs $CorrectorEpochs -PseudoMinTeacherMuScore "-2.0" -SynthPseudoWeight "0.5"
        }
        "FULL_step2p5" {
            Set-Gnode11FinishEnv -RunNote $RunNote -TeacherEpochs $TeacherEpochs `
                -CorrectorEpochs $CorrectorEpochs -PseudoMinTeacherMuScore "-2.0" -SynthPseudoWeight "0.5"
            $env:BIOCHEM_PRESET = "step2p5"
        }
        "FULL_step3" {
            Set-Gnode11CorrectorSmokeEnv -ComplexityStep "3" -RunNote $RunNote `
                -TeacherEpochs $TeacherEpochs -CorrectorEpochs $CorrectorEpochs `
                -TeacherForceMin "0.5" -KineWeight "0.15" -AdrWeight "1e-4" `
                -MuLogWeight "0.75" -MuSiWeight "0.15"
            $env:BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE = "-2.0"
            $env:BIOCHEM_SYNTH_PSEUDO_WEIGHT = "0.4"
            $env:BIOCHEM_PSEUDO_TEACHER_REF_MU_SCORE = "-0.25"
            $env:BIOCHEM_WARMUP_EPOCHS = "2"
            $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "10"
            $env:BIOCHEM_TEACHER_LR = "5e-4"
            $env:BIOCHEM_TEACHER_PHYSICS_CLIP_NORM = "5"
            $env:BIOCHEM_CKPT_EVERY = "4"
            $env:BIOCHEM_VAL_TIME_STRIDE = "8"
        }
        "FULL_overnight" {
            Set-Gnode11FinishEnv -RunNote $RunNote -TeacherEpochs $TeacherEpochs `
                -CorrectorEpochs $CorrectorEpochs -PseudoMinTeacherMuScore "-2.0" -SynthPseudoWeight "0.5"
            $env:BIOCHEM_PRESET = "overnight_step2"
            $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
            $env:BIOCHEM_EPOCHS = "$CorrectorEpochs"
            $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"
        }
        default {
            throw "Unknown comprehensive leg: $LegId"
        }
    }

    # Full pipeline: teacher on anchors, then corrector on anchors + synthetic graphs + pseudo.
    $env:BIOCHEM_STOP_AFTER_TEACHER = "0"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $env:BIOCHEM_TRAIN_MODE = "new"
    $env:BIOCHEM_RESUME = "0"

    Set-MuComprehensivePredKine
    Set-MuComprehensiveDeployMu -MuRatioMax "20"
}
