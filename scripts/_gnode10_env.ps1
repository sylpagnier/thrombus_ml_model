# Rung 10: GNODE teacher with predicted Stage-A kinematics (not GT COMSOL flow).
# Dot-source: . (Join-Path $PSScriptRoot "_gnode10_env.ps1")

function Clear-Gnode10BiochemEnv {
    Get-ChildItem Env:BIOCHEM_* -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item "Env:$($_.Name)" -ErrorAction SilentlyContinue
    }
}

function Resolve-Gnode10K5Ckpt {
    param([string] $UserPath = "")
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    if ($UserPath) {
        $p = if ([System.IO.Path]::IsPathRooted($UserPath)) { $UserPath } else { Join-Path $RepoRoot $UserPath }
        if (Test-Path $p) { return $p }
    }
    foreach ($rel in @(
            "outputs\biochem\gnode10_sweep\K5_kine15_final\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\promoted\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\K5_kine15\biochem_teacher_best_high_mu.pth"
        )) {
        $c = Join-Path $RepoRoot $rel
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Resolve-Gnode10InitCkpt {
    param([string] $UserPath = "")
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    if ($UserPath) {
        $p = if ([System.IO.Path]::IsPathRooted($UserPath)) { $UserPath } else { Join-Path $RepoRoot $UserPath }
        if (Test-Path $p) { return $p }
    }
    foreach ($rel in @(
            "outputs\biochem\gnode_after94_teacher_last.pth",
            "outputs\biochem\gnode_8h_ladder\checkpoints\after_94_biochem_teacher_last.pth"
        )) {
        $c = Join-Path $RepoRoot $rel
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Set-Gnode10PredictedKineBaseEnv {
    param(
        [string] $RunNote,
        [int] $Epochs = 4,
        [switch] $OomSafe
    )
    $env:KINEMATICS_USE_HARD_BCS = "1"
    $env:KINEMATICS_USE_WIDTH_PRIORS = "1"

    $env:BIOCHEM_PRESET = "passive_transport"
    $env:BIOCHEM_STOCK_DEFAULTS = "0"
    $env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
    $env:BIOCHEM_RUN_NOTE = $RunNote

    # Passive backward (not fast-iterate data-only step-2).
    $env:BIOCHEM_LOSS_ISOLATE = "PASSIVE"
    $env:BIOCHEM_LOSS_DATA_ONLY = "0"
    Remove-Item Env:BIOCHEM_COMPLEXITY_STEP -ErrorAction SilentlyContinue

    $env:BIOCHEM_STOP_AFTER_TEACHER = "1"
    $env:BIOCHEM_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_EPOCHS = "$Epochs"
    $env:BIOCHEM_CLI_TEACHER_EPOCHS = "$Epochs"
    $env:BIOCHEM_SKIP_PRETRAIN = "1"
    $env:BIOCHEM_INIT_FROM_BEST = "1"
    $env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
    $env:BIOCHEM_RESUME = "0"
    $env:BIOCHEM_TRAIN_MODE = "new"

    $env:BIOCHEM_VAL_TIME_STRIDE = "10"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_DETACH_MACRO_STATE = "0"

    # Predicted kine (override passive_transport preset defaults).
    $env:BIOCHEM_GT_KINE_VEL = "0"
    Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_KIN_LORA = "1"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = "1.0"

    $env:BIOCHEM_DATA_BIO_MASK_MODE = "clot_band"
    $env:BIOCHEM_PASSIVE_ADR_BACKPROP = "0"
    $env:BIOCHEM_PASSIVE_DATA_BIO_WEIGHT = "1.0"
    $env:BIOCHEM_PASSIVE_SPECIES_VAL = "1"
    $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "3.0"
    $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "2.0"

    $env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "0.0"
    $env:BIOCHEM_MU_LOG_WALL_WEIGHT = "0.0"
    $env:BIOCHEM_MU_LOG_HIGH_WEIGHT = "0.0"
    $env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"

    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_TBPTT_WINDOW_CURRICULUM = "0"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "8"
    $env:BIOCHEM_DATALOADER_WORKERS = "0"
    $env:BIOCHEM_PIN_MEMORY = "0"
    $env:BIOCHEM_KIN_GRADIENT_CHECKPOINT = "1"
    $env:BIOCHEM_ODE_GRADIENT_CHECKPOINT = "1"

    if ($OomSafe) {
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
        $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
    }
}

function Apply-Gnode10LegOverrides {
    param([hashtable] $Leg)
    if ($Leg.ContainsKey("TeacherForceMin")) {
        $env:BIOCHEM_TEACHER_FORCE_MIN = "$($Leg.TeacherForceMin)"
    }
    if ($Leg.ContainsKey("KineWeight")) {
        $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "$($Leg.KineWeight)"
    }
    if ($Leg.ContainsKey("TrainKinLora")) {
        $env:BIOCHEM_TRAIN_KIN_LORA = if ($Leg.TrainKinLora) { "1" } else { "0" }
    }
    if ($Leg.ContainsKey("TbpttWindow")) {
        $env:BIOCHEM_TBPTT_MAX_WINDOW = "$($Leg.TbpttWindow)"
    }
    if ($Leg.ContainsKey("DetachMacro")) {
        $env:BIOCHEM_DETACH_MACRO_STATE = if ($Leg.DetachMacro) { "1" } else { "0" }
    }
    if ($Leg.ContainsKey("AdrBackprop")) {
        $env:BIOCHEM_PASSIVE_ADR_BACKPROP = if ($Leg.AdrBackprop) { "1" } else { "0" }
    }
    if ($Leg.ContainsKey("AdrWeight")) {
        $env:BIOCHEM_PASSIVE_ADR_WEIGHT = "$($Leg.AdrWeight)"
    }
    if ($Leg.ContainsKey("FiWeight")) {
        $env:BIOCHEM_DATA_BIO_FI_WEIGHT = "$($Leg.FiWeight)"
    }
    if ($Leg.ContainsKey("MatWeight")) {
        $env:BIOCHEM_DATA_BIO_MAT_WEIGHT = "$($Leg.MatWeight)"
    }
    if ($Leg.ContainsKey("TeacherLr")) {
        $env:BIOCHEM_TEACHER_LR = "$($Leg.TeacherLr)"
    }
}
