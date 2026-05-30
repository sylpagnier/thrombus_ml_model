# I.1 X probe iteration: 3ep ablation legs from locked align (trends, not final quality).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_passive_x_iterate.ps1"
#   ... -Epochs 3 -LegsOnly X4_data_bio
#   ... -PromoteGates   # strict species gate (for -Promote confirm only)
#   Prefer: go_passive_x_probe.ps1 (iterate + summarize)

param(
    [string] $InitCkpt = "outputs/biochem/biochem_teacher_passive_align_locked.pth",
    [int] $Epochs = 2,
    [string] $LegsOnly = "",
    [switch] $Turbo,
    [switch] $SkipPytest,
    [switch] $WithPytest,
    [switch] $Probe,
    [switch] $PromoteGates
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "_python_rc.ps1")
. (Join-Path $PSScriptRoot "_passive_explore_base_env.ps1")
. (Join-Path $PSScriptRoot "_passive_x_block_env.ps1")

$OutRoot = Join-Path $RepoRoot "outputs\biochem\x_block"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$LogPath = Join-Path $OutRoot "x_block_log.jsonl"

function Write-XBlockLog {
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

$Legs = @(
    @{ id = "X3_mask_global"; note = "x_block_X3_mask_global"; mask = "global"; times = "last"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0"; adr = $false },
    @{ id = "X4_data_bio"; note = "x_block_X4_data_bio"; mask = "clot_band"; times = "union"; iso = "DATA_BIO"; fi = "3.0"; mat = "2.0"; adr = $false },
    @{ id = "X5_fi2mat2"; note = "x_block_X5_fi2mat2"; mask = "clot_band"; times = "union"; iso = "PASSIVE"; fi = "2.0"; mat = "2.0"; adr = $false },
    @{ id = "X_m3_union"; note = "x_block_X_m3_union"; mask = "clot_band"; times = "union"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0"; adr = $true },
    @{ id = "X6_confirm"; note = "x_block_X6_confirm"; mask = "clot_band"; times = "union"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0"; adr = $false; epochs = 3 }
)

if (-not $PromoteGates -and -not $PSBoundParameters.ContainsKey('Probe')) { $Probe = $true }
$useProbeGate = $Probe -and -not $PromoteGates
$runPytest = $WithPytest -and -not $SkipPytest
$speciesGateArgs = if ($useProbeGate) { @("--probe") } else { @() }

$initPath = Join-Path $RepoRoot $InitCkpt
if (-not (Test-Path $initPath)) {
    Write-Host "[ERR] Missing init ckpt: $initPath" -ForegroundColor Red
    exit 1
}

if ($runPytest) {
    Write-XBlockLog "pytest_passive" "START" @{}
    $rc = Invoke-PythonRc -m pytest src/tests/test_biochem_passive_transport.py -q --tb=line
    if ($rc -ne 0) {
        Write-XBlockLog "pytest_passive" "FAIL" @{ exit = $rc }
        exit $rc
    }
    Write-XBlockLog "pytest_passive" "OK" @{}
}

if ($Turbo) {
    $Legs = @(
        @{ id = "X4_data_bio"; note = "x_block_X4_data_bio"; mask = "clot_band"; times = "union"; iso = "DATA_BIO"; fi = "3.0"; mat = "2.0"; adr = $false },
        @{ id = "X5_fi2mat2"; note = "x_block_X5_fi2mat2"; mask = "clot_band"; times = "union"; iso = "PASSIVE"; fi = "2.0"; mat = "2.0"; adr = $false },
        @{ id = "X_m3_union"; note = "x_block_X_m3_union"; mask = "clot_band"; times = "union"; iso = "PASSIVE"; fi = "3.0"; mat = "2.0"; adr = $true }
    )
}

foreach ($leg in $Legs) {
    if ($LegsOnly -and $leg.id -ne $LegsOnly) { continue }
    $note = $leg.note
    $ep = if ($leg.epochs) { [int]$leg.epochs } else { $Epochs }
    Write-XBlockLog $note "START" @{ leg = $leg.id; epochs = $ep; turbo = [bool]$Turbo }

    if ($Turbo) {
        Set-PassiveXTurboEnv -RunNote $note -Epochs $ep -InitCkpt $InitCkpt
        Set-PassiveXLegEnv -RunNote $note -Epochs $ep -InitCkpt $InitCkpt `
            -LossIsolate $leg.iso -BioMask $leg.mask -MaskTimes $leg.times `
            -FiWeight $leg.fi -MatWeight $leg.mat -AdrBackprop:($leg.adr) | Out-Null
    } else {
        Set-PassiveXLegEnv -RunNote $note -Epochs $ep -InitCkpt $InitCkpt `
            -LossIsolate $leg.iso -BioMask $leg.mask -MaskTimes $leg.times `
            -FiWeight $leg.fi -MatWeight $leg.mat -AdrBackprop:($leg.adr)
    }

    Copy-Item $initPath (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_best_high_mu.pth") -Force
    $trainLog = Join-Path $OutRoot ("train_" + $note + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & python -u -m src.training.train_biochem_corrector --new --skip-pretrain --init-from-best `
        --epochs $ep --save-best --run-name $note 2>&1 | ForEach-Object {
        $_ | Out-File -FilePath $trainLog -Append -Encoding utf8
        Write-Host $_
    }
    $trainRc = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($trainRc -ne 0) {
        Write-XBlockLog $note "FAIL" @{ exit = $trainRc; log = $trainLog }
        continue
    }

    Start-Sleep -Seconds 2
    $spRc = Invoke-PythonRc scripts/check_passive_x_species_gate.py --run-note $note -Quiet @speciesGateArgs
    $xRc = 0
    if (-not $useProbeGate) {
        $xRc = Invoke-PythonRc scripts/check_phase_a_gate.py --mode x --run-note $note -Quiet
    }
    Write-XBlockLog ($note + "_gate_species") $(if ($spRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $spRc; probe = $useProbeGate }
    if (-not $useProbeGate) {
        Write-XBlockLog ($note + "_gate_Lbio") $(if ($xRc -eq 0) { "OK" } else { "WARN" }) @{ exit = $xRc }
    }

    $lastOut = Join-Path $OutRoot ($note + "_last.pth")
    Copy-Item (Join-Path $RepoRoot "outputs\biochem\biochem_teacher_last.pth") $lastOut -Force
    Write-XBlockLog $note $(if ($spRc -eq 0) { "OK" } else { "WARN" }) @{ last = $lastOut }
}

Write-Host "[OK] X block iterate log -> $LogPath" -ForegroundColor Green
Write-Host "[i]  python scripts/summarize_passive_x_block.py" -ForegroundColor Cyan
exit 0
