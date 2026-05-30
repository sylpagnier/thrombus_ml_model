# I.1 X close-out. Default = probe sanity (eval only, no train). Use -Promote for dump/lock.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_x_block_finish.ps1"
#   ... -Promote                  # optional confirm train + species dump + lock
#   ... -Promote -ConfirmEpochs 8 -Sweep
#   ... -TeacherCkpt path.pth -Promote -SkipTrain

param(
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [string] $TeacherCkpt = "",
    [int] $ConfirmEpochs = 8,
    [int] $DumpStride = 36,
    [int] $DumpMinSteps = 6,
    [switch] $Promote,
    [switch] $Sweep,
    [switch] $SkipDump,
    [switch] $SkipTrain,
    [switch] $SkipPytest,
    [switch] $WithPytest
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")
. (Join-Path $PSScriptRoot "_passive_x_block_env.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\x_block"
$AnchorDir = Join-Path $OutRoot ("anchors_stride" + $DumpStride + "_m" + $DumpMinSteps)
$LockedCkpt = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_passive_species_locked.pth"
$ManifestPath = Join-Path $RepoRoot "outputs\biochem\passive_species_locked_manifest.json"
$LogPath = Join-Path $OutRoot "x_block_finish_log.jsonl"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

function Write-FinishLog {
    param([string] $Step, [string] $Status, [hashtable] $Data = @{})
    $row = [ordered]@{
        ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        step = $Step
        status = $Status
    }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    $line = ($row | ConvertTo-Json -Compress) + "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::AppendAllText($LogPath, $line, $utf8NoBom)
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } else { "Cyan" })
}

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing align locked ckpt: $initPath" -ForegroundColor Red
    Write-Host "[i]  Run go_passive_lock_align_ckpt.ps1 after 20ep align." -ForegroundColor Yellow
    exit 1
}

$runPytest = $WithPytest -and -not $SkipPytest

if (-not $Promote) {
    Write-Host "[i]  Probe close: eval locked species only (no train/dump). Use -Promote when probes picked a recipe." -ForegroundColor Cyan
    Write-FinishLog "eval_baseline" "START" @{ ckpt = $InitCkpt; tier = "probe" }
    $evalRc = Invoke-PythonRc scripts/eval_passive_species_anchors.py --checkpoint $InitCkpt --split train `
        --max-fi-mean 0.05 --max-fi-per-anchor 0.04
    Write-FinishLog "eval_baseline" $(if ($evalRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $evalRc }
    @{
        tier = "probe"
        checked_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        init_ckpt = $InitCkpt
        note = "Calibration X1-X2 done; run go_passive_x_probe.ps1 for ablations; -Promote for dump"
    } | ConvertTo-Json -Depth 3 | Set-Content -Path $ManifestPath -Encoding utf8
    Write-Host "[OK] I.1 probe close (no promote). Manifest: $ManifestPath" -ForegroundColor Green
    exit 0
}

if ($runPytest) {
    Write-FinishLog "pytest_passive" "START" @{}
    $rc = Invoke-PythonRc -m pytest src/tests/test_biochem_passive_transport.py src/tests/test_biochem_physics.py -q --tb=line
    if ($rc -ne 0) { Write-FinishLog "pytest_passive" "FAIL" @{ exit = $rc }; exit $rc }
    Write-FinishLog "pytest_passive" "OK" @{}
}

if (-not $SkipTrain) {
    Write-FinishLog "eval_baseline" "START" @{ ckpt = $InitCkpt }
    $evalRc = Invoke-PythonRc scripts/eval_passive_species_anchors.py --checkpoint $InitCkpt --split train `
        --max-fi-mean 0.05 --max-fi-per-anchor 0.04
    Write-FinishLog "eval_baseline" $(if ($evalRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $evalRc }
}

$teacherForDump = $TeacherCkpt
if (-not $teacherForDump -and -not $SkipTrain) {
    if ($Sweep) {
        Write-FinishLog "x_iterate" "START" @{}
        & (Join-Path $PSScriptRoot "go_passive_x_iterate.ps1") -InitCkpt $InitCkpt -SkipPytest
        $summRc = Invoke-PythonRc scripts/summarize_passive_x_block.py --pick-best
        if ($summRc -ne 0) {
            Write-FinishLog "x_iterate" "WARN" @{ summarize = $summRc }
        } else {
            Write-FinishLog "x_iterate" "OK" @{}
        }
        $pickPath = Join-Path $OutRoot "best_teacher_for_dump.txt"
        if (Test-Path $pickPath) {
            $teacherForDump = (Get-Content $pickPath -Raw).Trim()
        }
    }

    if (-not $SkipTrain) {
        $confirmNote = "x_block_X6_confirm"
        Write-FinishLog $confirmNote "START" @{ epochs = $ConfirmEpochs }
        Set-PassiveXLegEnv -RunNote $confirmNote -Epochs $ConfirmEpochs -InitCkpt $InitCkpt
        Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
        $trainRc = Invoke-PythonRc -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
            --epochs $ConfirmEpochs --save-best --run-name $confirmNote
        if ($trainRc -ne 0) {
            Write-FinishLog $confirmNote "FAIL" @{ exit = $trainRc }
            exit $trainRc
        }
        $spRc = Invoke-PythonRc scripts/check_passive_x_species_gate.py --run-note $confirmNote
        Write-FinishLog ($confirmNote + "_gate") $(if ($spRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $spRc }

        $confirmLast = Join-Path $OutRoot ($confirmNote + "_last.pth")
        Copy-Item (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth") $confirmLast -Force
        if (-not $teacherForDump) {
            $teacherForDump = $confirmLast
        }
    }
}

if (-not $teacherForDump) {
    $teacherForDump = $InitCkpt
}

$teacherPath = if ([System.IO.Path]::IsPathRooted($teacherForDump)) {
    $teacherForDump
} else {
    Join-Path $RepoRoot ($teacherForDump -replace "/", "\")
}
if (-not (Test-Path $teacherPath)) {
    Write-Host "[ERR] Teacher ckpt not found: $teacherPath" -ForegroundColor Red
    exit 1
}

Write-FinishLog "lock_teacher" "START" @{ src = $teacherPath }
Copy-Item $teacherPath $LockedCkpt -Force
Copy-Item $LockedCkpt (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
Write-FinishLog "lock_teacher" "OK" @{ dest = $LockedCkpt }

if (-not $SkipDump) {
    Write-FinishLog "species_dump" "START" @{ teacher = $teacherPath; out = $AnchorDir }
    $dumpRc = Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $teacherPath --out-dir $AnchorDir --device cuda `
        --time-stride $DumpStride --min-steps $DumpMinSteps --force
    if ($dumpRc -ne 0) {
        Write-FinishLog "species_dump" "FAIL" @{ exit = $dumpRc }
        exit $dumpRc
    }
    Write-FinishLog "species_dump" "OK" @{ out = $AnchorDir }
}

$manifest = @{
    locked_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    tier          = "promote"
    block         = "I.1_X"
    source_ckpt   = ($teacherPath.Replace($RepoRoot, "").TrimStart("\", "/") -replace "\\", "/")
    dest_ckpt     = "outputs/biochem/biochem_teacher_passive_species_locked.pth"
    anchor_dir    = ($AnchorDir.Replace($RepoRoot, "").TrimStart("\", "/") -replace "\\", "/")
    dump_stride   = $DumpStride
    dump_min_steps = $DumpMinSteps
    init_ckpt     = $InitCkpt
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $ManifestPath -Encoding utf8

Write-FinishLog "final_gate" "START" @{}
$gateRc = Invoke-PythonRc scripts/check_passive_x_block_gate.py --checkpoint $LockedCkpt `
    $(if (-not $SkipDump) { "--anchor-dir"; $AnchorDir; "--skip-eval" })
Write-FinishLog "final_gate" $(if ($gateRc -eq 0) { "OK" } else { "FAIL" }) @{ exit = $gateRc }

Write-Host "[OK] I.1 X promote:" -ForegroundColor Green
Write-Host "     teacher: $LockedCkpt" -ForegroundColor Green
if (-not $SkipDump) {
    Write-Host "     anchors: $AnchorDir" -ForegroundColor Green
    Write-Host "[i]  Clot-phi: set CLOT_PHI_ANCHOR_DIR=$AnchorDir (or go_gt_flow_species_ladder_6h.ps1)" -ForegroundColor Cyan
}
Write-Host "[i]  Log: $LogPath | manifest: $ManifestPath" -ForegroundColor Cyan
exit $gateRc
