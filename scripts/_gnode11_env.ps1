# Rung 11a/11b: corrector smoke on anchors (predicted kine, teacher + Phase 3 corrector).
# Dot-source: . (Join-Path $PSScriptRoot "_gnode11_env.ps1")

. (Join-Path $PSScriptRoot "_gnode10_env.ps1")
. (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")

function Set-Gnode11CorrectorSmokeEnv {
    param(
        [ValidateSet("2", "3")]
        [string] $ComplexityStep = "2",
        [string] $RunNote = "gnode11_corrector_smoke",
        [int] $TeacherEpochs = 2,
        [int] $CorrectorEpochs = 4,
        [string] $TeacherForceMin = "0.5",
        [string] $KineWeight = "0.15",
        [string] $AdrWeight = "1e-4",
        [string] $MuLogWeight = "0.75",
        [string] $MuSiWeight = "0.15"
    )
    Clear-Gnode10BiochemEnv

    $useStep2Bridge = ($ComplexityStep -eq "2")
    if ($useStep2Bridge) {
        Set-PassiveExploreLegEnv -RunNote $RunNote -Component "XY" -Epochs $CorrectorEpochs `
            -TeacherLr "1e-3" -Step2Bridge -AdrBackprop -AdrWeight $AdrWeight `
            -MuLogWeight $MuLogWeight -MuSiWeight $MuSiWeight -MuRatioMax "1" `
            -DataBioMask "clot_band" -MaskTimes "union" -FiWeight "3.0" -MatWeight "2.0"
    } else {
        Set-PassiveExploreLegEnv -RunNote $RunNote -Component "XY" -Epochs $CorrectorEpochs `
            -TeacherLr "1e-3" -AdrBackprop -AdrWeight $AdrWeight `
            -MuLogWeight $MuLogWeight -MuSiWeight $MuSiWeight -MuRatioMax "1" `
            -DataBioMask "clot_band" -MaskTimes "union" -FiWeight "3.0" -MatWeight "2.0"
        Remove-Item Env:BIOCHEM_PASSIVE_STEP2_BRIDGE -ErrorAction SilentlyContinue
        $env:BIOCHEM_COMPLEXITY_STEP = "3"
        $env:BIOCHEM_LOSS_DATA_ONLY = "0"
        $env:BIOCHEM_DATA_ONLY_PHYS_TEMP = "0"
    }

    # Rung 11: run corrector after teacher (Phase 3 loop).
    $env:BIOCHEM_STOP_AFTER_TEACHER = "0"
    $env:BIOCHEM_TEACHER_EPOCHS = "$TeacherEpochs"
    $env:BIOCHEM_EPOCHS = "$CorrectorEpochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$TeacherEpochs"

    # Predicted Stage-A kine (K5 stack).
    $env:BIOCHEM_GT_KINE_VEL = "0"
    Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_KIN_LORA = "1"
    $env:BIOCHEM_TEACHER_FORCE_MIN = $TeacherForceMin
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = $KineWeight
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"

    # Short corrector warmup so Phase 3 runs within CorrectorEpochs.
    $env:BIOCHEM_WARMUP_EPOCHS = "1"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "99"

    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "0"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_TRAIN_MODE = "new"

    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"
    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_VAL_EVERY = "1"
    $env:BIOCHEM_CKPT_EVERY = "2"

    Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue

    if ($useStep2Bridge) {
        $env:BIOCHEM_COMPLEXITY_STEP = "2"
    }
}

function Resolve-Gnode11InitCkpt {
    param([string] $UserPath = "")
    $k5 = Resolve-Gnode10K5Ckpt -UserPath $UserPath
    if ($k5) { return $k5 }
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    foreach ($rel in @(
            "outputs\biochem\gnode10_sweep\gnode11_corrector_smoke\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_teacher_best_high_mu.pth"
        )) {
        $c = Join-Path $RepoRoot $rel
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Set-Gnode11FinishEnv {
    param(
        [string] $RunNote = "gnode11_finish",
        [int] $TeacherEpochs = 8,
        [int] $CorrectorEpochs = 12,
        [string] $PseudoMinTeacherMuScore = "-2.0",
        [string] $SynthPseudoWeight = "0.5"
    )
    Set-Gnode11CorrectorSmokeEnv -ComplexityStep "2" -RunNote $RunNote `
        -TeacherEpochs $TeacherEpochs -CorrectorEpochs $CorrectorEpochs `
        -TeacherForceMin "0.5" -KineWeight "0.15" -AdrWeight "1e-4" `
        -MuLogWeight "0.75" -MuSiWeight "0.15"

    # Phase II.0: enable synthetic pseudo supervision (K5 teacher mu_score ~ -1.45).
    $env:BIOCHEM_PSEUDO_MIN_TEACHER_MU_SCORE = $PseudoMinTeacherMuScore
    $env:BIOCHEM_SYNTH_PSEUDO_WEIGHT = $SynthPseudoWeight
    $env:BIOCHEM_PSEUDO_TEACHER_REF_MU_SCORE = "-0.25"

    $env:BIOCHEM_WARMUP_EPOCHS = "2"
    $env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "12"
    $env:BIOCHEM_CKPT_EVERY = "4"
    $env:BIOCHEM_VAL_TIME_STRIDE = "8"
}
