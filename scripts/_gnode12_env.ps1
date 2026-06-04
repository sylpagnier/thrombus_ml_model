# Rung 12 Lane A: mu-unlock (optional) + predicted-kine dump + clot-phi.
# Dot-source: . (Join-Path $PSScriptRoot "_gnode12_env.ps1")

. (Join-Path $PSScriptRoot "_gnode11_env.ps1")
. (Join-Path $PSScriptRoot "_passive_mu_unlock_env.ps1")

function Get-Gnode12RepoRoot {
    if ($PSScriptRoot) { return (Split-Path -Parent $PSScriptRoot) }
    return (Get-Location).Path
}

function Resolve-Gnode12RepoPath {
    param([string] $Path)
    if (-not $Path) { return $null }
    $root = Get-Gnode12RepoRoot
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Get-Gnode12PathRelativeToRepo {
    param(
        [string] $FullPath,
        [string] $RepoRoot = ""
    )
    if (-not $RepoRoot) { $RepoRoot = Get-Gnode12RepoRoot }
    $full = Resolve-Gnode12RepoPath -Path $FullPath
    $root = Resolve-Gnode12RepoPath -Path $RepoRoot
    if ($full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($root.Length).TrimStart("\", "/")
    }
    return $full
}

function Resolve-Gnode12TeacherCkpt {
    param([string] $UserPath = "")
    if ($UserPath) {
        $p = Resolve-Gnode12RepoPath -Path $UserPath
        if ($p -and (Test-Path $p)) { return $p }
    }
    # gnode11_finish archive dir is often empty (ckpts live in outputs/biochem/ during run only).
    foreach ($rel in @(
            "outputs\biochem\gnode10_sweep\gnode12_mu_unlock\biochem_teacher_passive_mu_unlock_best.pth",
            "outputs\biochem\gnode10_sweep\gnode12_mu_unlock\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_teacher_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_teacher_last.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_latest_checkpoint.pth",
            "outputs\biochem\biochem_teacher_best_high_mu.pth"
        )) {
        $c = Resolve-Gnode12RepoPath -Path $rel
        if ($c -and (Test-Path $c)) { return $c }
    }
    $k5 = Resolve-Gnode10K5Ckpt -UserPath ""
    if ($k5) {
        $p = Resolve-Gnode12RepoPath -Path $k5
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Set-Gnode12MuUnlockEnv {
    param(
        [string] $RunNote = "gnode12_mu_unlock",
        [int] $Epochs = 6,
        [string] $MuRatioMax = "20",
        [string] $TeacherForceMin = "0.5",
        [string] $MuLogWeight = "1.0",
        [string] $MuSiWeight = "0.25"
    )
    Set-PassiveMuUnlockEnv -RunNote $RunNote -Epochs $Epochs -MuRatioMax $MuRatioMax `
        -MuLogWeight $MuLogWeight -MuSiWeight $MuSiWeight

    # Predicted Stage-A kine (same stack as gnode11); allow clot viscosity feedback in forward.
    $env:BIOCHEM_GT_KINE_VEL = "0"
    Remove-Item Env:BIOCHEM_GT_KINE_SKIP_DEQ -ErrorAction SilentlyContinue
    $env:BIOCHEM_TRAIN_KIN_LORA = "0"
    $env:BIOCHEM_TEACHER_FORCE_MIN = $TeacherForceMin
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = $MuRatioMax
    $env:BIOCHEM_PASSIVE_DATA_KINE_WEIGHT = "0.15"
    $env:BIOCHEM_VAL_TIME_STRIDE = "8"
    $env:BIOCHEM_TEACHER_VAL_EVERY = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "5"
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "6"
}

function Set-Gnode12DumpRolloutEnv {
    param([string] $MuRatioMax = "20")
    $env:BIOCHEM_GT_KINE_VEL = "0"
    $env:BIOCHEM_TEACHER_MU_RATIO_MAX = $MuRatioMax
    $env:BIOCHEM_ADJOINT_RK4_SUBSTEPS = "1"
    $env:BIOCHEM_TBPTT_MAX_WINDOW = "6"
}
