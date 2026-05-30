# ~6h passive/GT-flow exploration: isolated X (species), Y (terms), then XY (combinations).
# No Stage-A kin training (GT [u,v,p]). Logs -> outputs/biochem/explore_6h/
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_explore_6h.ps1"
#   powershell ... -SkipPytest -InitCkpt outputs/biochem/biochem_teacher_passive_align_locked.pth
#   powershell ... -DryRun   # print leg schedule only

param(
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [switch] $SkipPytest,
    [switch] $SkipAudit,
    [switch] $SkipClotPhi = $true,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\explore_6h"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "explore_log.jsonl"
$SummaryPath = Join-Path $OutRoot "summary.json"

# Leg schedule (~6h @ ~12-20 min per 6-10ep leg on RTX 500 class GPU)
$Legs = @(
    @{ phase = "preflight"; id = "smoke_x"; component = "X"; epochs = 2; note = "expl6h_smoke_x";
        iso = "PASSIVE"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "X"; id = "X_m3_union"; component = "X"; epochs = 10; note = "expl6h_X_m3_union";
        iso = "PASSIVE"; adr = $true; adrW = "1e-4"; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "X"; id = "X_data_bio"; component = "X"; epochs = 6; note = "expl6h_X_data_bio";
        iso = "DATA_BIO"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "X"; id = "X_mask_global"; component = "X"; epochs = 6; note = "expl6h_X_mask_global";
        iso = "PASSIVE"; adr = $false; muRatio = "1"; mask = "global"; times = "last"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "X"; id = "X_fi2mat2"; component = "X"; epochs = 6; note = "expl6h_X_fi2mat2";
        iso = "PASSIVE"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "2.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "Y"; id = "Y_ADR_S"; component = "Y"; epochs = 4; note = "expl6h_Y_ADR_S";
        iso = "ADR_S"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; lr = "1e-3" },
    @{ phase = "Y"; id = "Y_ADR_F"; component = "Y"; epochs = 4; note = "expl6h_Y_ADR_F";
        iso = "ADR_F"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; lr = "1e-3" },
    @{ phase = "Y"; id = "Y_W_BIO"; component = "Y"; epochs = 4; note = "expl6h_Y_W_BIO";
        iso = "W_BIO"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; lr = "1e-3" },
    @{ phase = "Y"; id = "Y_W_PHY"; component = "Y"; epochs = 4; note = "expl6h_Y_W_PHY";
        iso = "W_PHY"; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; lr = "1e-3" },
    @{ phase = "Y"; id = "Y_MU_LOG"; component = "Y"; epochs = 6; note = "expl6h_Y_MU_LOG";
        muUnlock = $true; muRatio = "20"; mask = "clot_band"; times = "union"; lr = "1e-3";
        muLog = "1.0"; muSi = "0.25" },
    @{ phase = "XY"; id = "XY_adr_low"; component = "XY"; epochs = 8; note = "expl6h_XY_adr_low";
        iso = "PASSIVE"; adr = $true; adrW = "1e-4"; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "XY"; id = "XY_bridge"; component = "XY"; epochs = 10; note = "expl6h_XY_bridge";
        bridge = $true; adr = $true; adrW = "1e-4"; muRatio = "1"; mask = "clot_band"; times = "union";
        muLog = "0.75"; muSi = "0.15"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "XY"; id = "XY_mu_unlock"; component = "XY"; epochs = 8; note = "expl6h_XY_mu_unlock";
        muUnlock = $true; muRatio = "20"; mask = "clot_band"; times = "union"; lr = "1e-3";
        muLog = "1.0"; muSi = "0.25" },
    @{ phase = "XY"; id = "XY_ramp1"; component = "XY"; epochs = 4; note = "expl6h_XY_ramp1";
        dataOnly = $true; adr = $false; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "3.0"; mat = "2.0"; lr = "1e-3" },
    @{ phase = "XY"; id = "XY_ramp2"; component = "XY"; epochs = 8; note = "expl6h_XY_ramp2";
        dataOnly = $true; adr = $true; adrW = "1e-3"; muRatio = "1"; mask = "clot_band"; times = "union"; fi = "3.0"; mat = "2.0"; lr = "1e-3";
        initFromPrev = "expl6h_XY_ramp1" }
)

function Write-ExploreLog {
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
    Write-Host "[$Status] $Step" -ForegroundColor $(if ($Status -eq "OK") { "Green" } elseif ($Status -eq "FAIL") { "Red" } elseif ($Status -eq "WARN") { "Yellow" } else { "Cyan" })
}

function Invoke-ExploreLeg {
    param([hashtable] $Leg, [string] $InitPath)
    $note = $Leg.note
    $ep = [int]$Leg.epochs
    $comp = $Leg.component
    Write-ExploreLog $note "START" @{ phase = $Leg.phase; component = $comp; epochs = $ep }

    $params = @{
        RunNote = $note
        Component = $comp
        Epochs = $ep
        TeacherLr = $(if ($Leg.lr) { $Leg.lr } else { "1e-3" })
        LossIsolate = $(if ($Leg.iso) { $Leg.iso } else { "" })
        MuRatioMax = $(if ($Leg.muRatio) { $Leg.muRatio } else { "1" })
        DataBioMask = $(if ($Leg.mask) { $Leg.mask } else { "clot_band" })
        MaskTimes = $(if ($Leg.times) { $Leg.times } else { "union" })
        FiWeight = $(if ($Leg.fi) { $Leg.fi } else { "3.0" })
        MatWeight = $(if ($Leg.mat) { $Leg.mat } else { "2.0" })
        MuLogWeight = $(if ($Leg.muLog) { $Leg.muLog } else { "0" })
        MuSiWeight = $(if ($Leg.muSi) { $Leg.muSi } else { "0" })
        InitCkpt = $InitPath
    }
    if ($Leg.dataOnly) { $params.LossDataOnly = $true }
    if ($Leg.bridge) { $params.Step2Bridge = $true }
    if ($Leg.muUnlock) { $params.MuUnlock = $true }
    if ($Leg.adr) {
        $params.AdrBackprop = $true
        $params.AdrWeight = $(if ($Leg.adrW) { $Leg.adrW } else { "1e-4" })
    }
    Set-PassiveExploreLegEnv @params

    if ($DryRun) {
        Write-ExploreLog $note "DRY" @{ component = $comp }
        return $true
    }

    Copy-Item $InitPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
    $trainLog = Join-Path $OutRoot ("train_" + $note + ".log")
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    python -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $ep --save-best --run-name $note 2>&1 | Tee-Object -FilePath $trainLog
    $trainRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($trainRc -ne 0) {
        Write-ExploreLog $note "FAIL" @{ exit = $trainRc; log = $trainLog }
        return $false
    }

    $gateRc = 0
    if ($comp -eq "X" -or $comp -eq "XY") {
        if ($Leg.bridge) {
            $gateRc = Invoke-PythonRc scripts/check_passive_step2_bridge_gate.py --run-note $note
        } elseif ($Leg.muUnlock) {
            $gateRc = Invoke-PythonRc scripts/check_passive_mu_unlock_gate.py --run-note $note
        } else {
            $gateRc = Invoke-PythonRc scripts/check_m3_align_gate.py --run-note $note 2>$null
            if ($LASTEXITCODE -ne 0) {
                $gateRc = Invoke-PythonRc scripts/check_passive_teacher_gate.py --run-note $note
            }
        }
    }
    if ($comp -eq "Y" -and $Leg.iso) {
        $yRc = Invoke-PythonRc scripts/check_phase_a_gate.py --mode y --run-note $note --term $Leg.iso -Quiet
        Write-ExploreLog ($note + "_gate_y") $(if ($yRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $yRc }
    }

    Copy-Item (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth") `
        (Join-Path $OutRoot ($note + "_last.pth")) -Force

    Write-ExploreLog $note $(if ($gateRc -eq 0) { "OK" } else { "WARN" }) @{ gate_exit = $gateRc; log = $trainLog }
    return ($trainRc -eq 0)
}

Write-Host "[NEW] Passive explore 6h: X / Y / XY isolation ladder" -ForegroundColor Cyan
Write-Host "[i]  Init default: $InitCkpt | log: $LogPath" -ForegroundColor Cyan

if ($DryRun) {
    foreach ($leg in $Legs) {
        Write-Host ("  [{0}] {1} {2}ep -> {3}" -f $leg.phase, $leg.id, $leg.epochs, $leg.note)
    }
    exit 0
}

if (-not $SkipPytest) {
    Write-ExploreLog "pytest_passive" "START" @{}
    $rc = Invoke-PythonRc -m pytest src/tests/test_biochem_passive_transport.py src/tests/test_biochem_physics.py -q --tb=line
    if ($rc -ne 0) {
        Write-ExploreLog "pytest_passive" "FAIL" @{ exit = $rc }
        exit $rc
    }
    Write-ExploreLog "pytest_passive" "OK" @{}
}

if (-not $SkipAudit) {
    $env:BIOCHEM_SUPERVISION_MASK_TIMES = "union"
    Invoke-PythonRc scripts/audit_passive_adr_alignment.py --anchor patient007 --compare-mask-times | Out-Null
}

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    Write-Host "[i]  Use go_passive_lock_align_ckpt.ps1 or passive_align_20ep last.pth" -ForegroundColor Yellow
    exit 1
}

$chainInit = $initPath
foreach ($leg in $Legs) {
    if ($leg.initFromPrev) {
        $prev = Join-Path $OutRoot ($leg.initFromPrev + "_last.pth")
        if (Test-Path $prev) {
            $chainInit = $prev
            Write-Host "[i]  $($leg.note) init <- $prev" -ForegroundColor Cyan
        }
    } else {
        $chainInit = $initPath
    }
    $ok = Invoke-ExploreLeg -Leg $leg -InitPath $chainInit
    if (-not $ok -and $leg.phase -ne "preflight") {
        Write-Host "[WARN] Leg $($leg.id) failed; continuing ladder" -ForegroundColor Yellow
    }
}

if (-not $SkipClotPhi) {
    $anchorDir = Join-Path $OutRoot "anchors_stride36_m6"
    $teacherForDump = Join-Path $OutRoot "expl6h_X_m3_union_last.pth"
    if (-not (Test-Path $teacherForDump)) {
        $teacherForDump = Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth"
    }
    Write-ExploreLog "species_dump" "START" @{ teacher = $teacherForDump }
    $dumpRc = Invoke-PythonRc scripts/dump_teacher_species_to_anchors.py `
        --teacher $teacherForDump --out-dir $anchorDir --device cuda --time-stride 36 --min-steps 6 --force
    Write-ExploreLog "species_dump" $(if ($dumpRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $dumpRc; out = $anchorDir }
    Write-Host "[i]  Clot-phi train: use go_gt_flow_species_ladder_6h.ps1 with CLOT_PHI_ANCHOR_DIR=$anchorDir" -ForegroundColor Cyan
}

Write-ExploreLog "summarize" "START" @{}
$summRc = Invoke-PythonRc scripts/summarize_passive_explore_6h.py --log $LogPath --out $SummaryPath
Write-ExploreLog "summarize" $(if ($summRc -eq 0) { "OK" } else { "WARN" }) @{ out = $SummaryPath }

Write-Host "[OK] Explore 6h complete. Review:" -ForegroundColor Green
Write-Host "     $LogPath" -ForegroundColor Green
Write-Host "     $SummaryPath" -ForegroundColor Green
Write-Host "[i]  python scripts/summarize_passive_explore_6h.py" -ForegroundColor Cyan
exit 0
