# Headless + optional interactive viz helpers for GNODE ladder (9.x).
# Dot-source: . (Join-Path $PSScriptRoot "_gnode_viz_helpers.ps1")

function Set-GnodeGtFlowVizEnv {
    $env:BIOCHEM_GT_KINE_VEL = "1"
    $env:BIOCHEM_GT_KINE_SKIP_DEQ = "1"
}

function Invoke-BiochemTeacherClotbandViz {
    param(
        [string] $Checkpoint = "outputs/biochem/biochem_teacher_last.pth",
        [string] $Anchor = "patient007",
        [string] $AnchorDir = "",
        [int] $TimeIndex = -1,
        [string] $Out = "",
        [string] $Label = "teacher_clotband"
    )
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    $vizDir = Join-Path $RepoRoot "outputs\biochem\viz"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    if (-not $Out) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($Checkpoint)
        $Out = Join-Path $vizDir "${Label}_${stem}_${Anchor}_t${TimeIndex}.png"
    }
    Write-Host "[viz] Teacher clot-band (phi/mu) -> $Out" -ForegroundColor Cyan
    $pyArgs = @(
        "scripts/snapshot_biochem_teacher_clotband.py",
        "--checkpoint", $Checkpoint,
        "--anchor", $Anchor,
        "--time-index", "$TimeIndex",
        "--out", $Out
    )
    if ($AnchorDir) {
        $pyArgs += @("--anchor-dir", $AnchorDir)
    }
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
}

function Invoke-BiochemTeacherSnapshot {
    param(
        [string] $Checkpoint = "outputs/biochem/biochem_teacher_last.pth",
        [string] $Anchor = "patient007",
        [string] $Out = "",
        [string] $Label = "teacher"
    )
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    $vizDir = Join-Path $RepoRoot "outputs\biochem\viz"
    New-Item -ItemType Directory -Force -Path $vizDir | Out-Null
    if (-not $Out) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($Checkpoint)
        $Out = Join-Path $vizDir "${Label}_${stem}_${Anchor}.png"
    }
    Write-Host "[viz] Teacher snapshot -> $Out" -ForegroundColor Cyan
    $pyArgs = @(
        "scripts/snapshot_biochem_teacher.py",
        "--checkpoint", $Checkpoint,
        "--anchor", $Anchor,
        "--out", $Out
    )
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
}

function Invoke-ClotPhiScatterViz {
    param(
        [string] $Checkpoint,
        [string] $Anchor = "patient007",
        [string] $Out = "",
        [int] $TimeIndex = -1
    )
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    Set-Location $RepoRoot
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    $env:CLOT_PHI_DGAMMA_FEATURE_TIME = "current"
    if (-not $Out) {
        $ck = [IO.Path]::GetFileNameWithoutExtension($Checkpoint)
        $Out = Join-Path $RepoRoot "outputs\biochem\viz\clot_phi_${ck}_${Anchor}_t${TimeIndex}.png"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null
    Write-Host "[viz] Clot-phi scatter -> $Out" -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "src.evaluation.viz_clot_phi_simple",
        "--anchor", $Anchor,
        "--checkpoint", $Checkpoint,
        "--time-index", "$TimeIndex",
        "--plot-mode", "scatter",
        "--out", $Out
    )
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
}

function Invoke-ClotPhiMaskViz {
    param(
        [string] $Anchor = "patient007",
        [int] $TimeIndex = -1,
        [string] $Out = ""
    )
    $RepoRoot = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { Get-Location }
    Set-Location $RepoRoot
    . (Join-Path $PSScriptRoot "_clot_phi_shared_env.ps1")
    if (-not $Out) {
        $Out = Join-Path $RepoRoot "outputs\biochem\viz\mask_${Anchor}_t${TimeIndex}.png"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null
    Write-Host "[viz] Mask/GT sanity -> $Out" -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "src.evaluation.viz_clot_phi_masks",
        "--anchor", $Anchor,
        "--time-index", "$TimeIndex",
        "--out", $Out
    )
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
}

function Invoke-GnodeTeacherInteractiveViz {
    param(
        [string] $Checkpoint = "outputs/biochem/biochem_teacher_last.pth",
        [string] $Anchor = "patient007"
    )
    Write-Host "[viz] Interactive teacher slider (close window to continue)..." -ForegroundColor Cyan
    $pyArgs = @(
        "-m", "src.evaluation.visualize_pipeline",
        "--teacher-only",
        "--biochem-checkpoint", $Checkpoint,
        "--anchor", $Anchor
    )
    . (Join-Path $PSScriptRoot "_python_rc.ps1")
    $rc = Invoke-PythonRc @pyArgs
    if ($rc -ne 0) { exit $rc }
}

function Invoke-GnodeRungVizCheckup {
    param(
        [string] $RungLabel = "gnode",
        [string] $TeacherCheckpoint = "",
        [string] $ClotCheckpoint = "",
        [string] $AnchorDir = "",
        [switch] $InteractiveTeacher,
        [switch] $MaskSanity
    )
    if ($TeacherCheckpoint -and (Test-Path $TeacherCheckpoint)) {
        Invoke-BiochemTeacherSnapshot -Checkpoint $TeacherCheckpoint -Anchor patient007 -Label $RungLabel
        $clotTi = -1
        if ($AnchorDir) { $clotTi = 4 }
        Invoke-BiochemTeacherClotbandViz `
            -Checkpoint $TeacherCheckpoint `
            -Anchor patient007 `
            -AnchorDir $AnchorDir `
            -TimeIndex $clotTi `
            -Label "${RungLabel}_clotband"
        if ($InteractiveTeacher) {
            Invoke-GnodeTeacherInteractiveViz -Checkpoint $TeacherCheckpoint -Anchor patient007
        }
    }
    if ($MaskSanity) {
        Invoke-ClotPhiMaskViz -Anchor patient007 -TimeIndex 0
        Invoke-ClotPhiMaskViz -Anchor patient007 -TimeIndex -1
    }
    if ($ClotCheckpoint -and (Test-Path $ClotCheckpoint)) {
        if ($AnchorDir) { $env:CLOT_PHI_ANCHOR_DIR = $AnchorDir }
        Invoke-ClotPhiScatterViz -Checkpoint $ClotCheckpoint -Anchor patient007 -TimeIndex -1
        if ($AnchorDir) { Remove-Item Env:CLOT_PHI_ANCHOR_DIR -ErrorAction SilentlyContinue }
    }
}
