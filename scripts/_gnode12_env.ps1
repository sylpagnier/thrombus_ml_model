# Rung 12: Lane A (teacher mu-unlock + dump + clot-phi), Lane B (11-finish corrector dump + clot-phi).
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

function Resolve-Gnode12CorrectorCkpt {
    param([string] $UserPath = "")
    if ($UserPath) {
        $p = Resolve-Gnode12RepoPath -Path $UserPath
        if ($p -and (Test-Path $p)) { return $p }
    }
    foreach ($rel in @(
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_best_high_mu.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_best.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_latest_checkpoint.pth",
            "outputs\biochem\gnode10_sweep\gnode11_finish\biochem_teacher_last.pth",
            "outputs\biochem\biochem_best_high_mu.pth",
            "outputs\biochem\biochem_best.pth",
            "outputs\biochem\biochem_latest_checkpoint.pth"
        )) {
        $c = Resolve-Gnode12RepoPath -Path $rel
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Invoke-Gnode12DumpClotLeg {
    param(
        [string] $RolloutCkptPath,
        [string] $JuneAnchorDir = "outputs\biochem\gnode_8h_ladder\anchors_stride_72",
        [string] $OutAnchorDir,
        [string] $ClotLeg,
        [double] $MuRatioMax = 20,
        [int] $ClotEpochs = 35,
        [double] $MinGtPosFrac = 0.55,
        [switch] $SkipDump,
        [switch] $SkipClot,
        [switch] $SkipViz,
        [string] $LaneLabel = "A",
        [string] $BaselineNote = ""
    )

    $RepoRoot = Get-Gnode12RepoRoot

    if (-not (Test-Path (Join-Path $RepoRoot $JuneAnchorDir))) {
        Write-Host "[ERR] June anchors missing: $JuneAnchorDir" -ForegroundColor Red
        exit 1
    }

    $EvalJson = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$ClotLeg\multi_anchor.jsonl"
    Set-Gnode12DumpRolloutEnv -MuRatioMax "$MuRatioMax"
    $ckptRel = Get-Gnode12PathRelativeToRepo -FullPath $RolloutCkptPath -RepoRoot $RepoRoot

    if (-not $SkipDump) {
        Write-Host "[NEW] dump species + pred [u,v,p] (mu_ratio_max=$MuRatioMax)" -ForegroundColor Cyan
        Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
            --teacher $ckptRel `
            --src-dir $JuneAnchorDir `
            --out-dir $OutAnchorDir `
            --device cuda `
            --no-subsample `
            --write-kine-macro `
            --mu-ratio-max $MuRatioMax `
            --force
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } elseif (-not (Test-Path (Join-Path $RepoRoot $OutAnchorDir))) {
        Write-Host "[ERR] -SkipDump but missing $OutAnchorDir" -ForegroundColor Red
        exit 1
    }

    $preflightLeg = "${ClotLeg}_preflight"
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" `
        -AnchorDir $OutAnchorDir -LegName $preflightLeg -Epochs 1 -SkipViz -SkipEval
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $preflightLog = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$preflightLeg\clot_phi_train_log.jsonl"
    $gtPlus = $null
    if (Test-Path $preflightLog) {
        $row = (Get-Content $preflightLog -Tail 1) | ConvertFrom-Json
        $gtPlus = [double]$row.val.gt_pos_frac
    }
    if ($null -eq $gtPlus -or $gtPlus -lt $MinGtPosFrac) {
        Write-Host "[ERR] preflight gt+=$gtPlus (need >= $MinGtPosFrac)" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK]  preflight gt+=$([math]::Round($gtPlus, 3))" -ForegroundColor Green

    if ($SkipClot) {
        return
    }

    $env:CLOT_PHI_ANCHOR_DIR = $OutAnchorDir
    Remove-Item Env:CLOT_PHI_ROLLOUT -ErrorAction SilentlyContinue
    $env:CLOT_PHI_VEL_SOURCE = "gt"

    $clotArgs = @(
        "-AnchorDir", $OutAnchorDir,
        "-LegName", $ClotLeg,
        "-Epochs", "$ClotEpochs"
    )
    if ($SkipViz) { $clotArgs += "-SkipViz" }

    Write-Host "[NEW] clot-phi ${ClotEpochs}ep (vel=file pred u,v,p)" -ForegroundColor Cyan
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_clot_phi_from_anchor_dir.ps1" @clotArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (-not (Test-Path $EvalJson)) {
        Write-Host "[ERR] Missing canonical eval: $EvalJson" -ForegroundColor Red
        exit 1
    }

    $ckptOut = Join-Path $RepoRoot "outputs\biochem\passive_species_focus_compare\$ClotLeg\clot_phi_best.pth"
    Write-Host "[OK]  GNODE 12 Lane $LaneLabel dump+clot complete." -ForegroundColor Green
    Write-Host "[i]  anchors: $OutAnchorDir" -ForegroundColor DarkGray
    Write-Host "[i]  clot-phi: $ckptOut" -ForegroundColor DarkGray
    Write-Host "[i]  eval: $EvalJson $BaselineNote" -ForegroundColor DarkGray
}
