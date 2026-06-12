# Log and terminate stale python processes (training/sweep orphans).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\kill_stale_python.ps1"

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogPath = Join-Path $RepoRoot "outputs\biochem\diagnostics\killed_python_procs.log"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $LogPath -Value "[$stamp] scan python processes"

$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -match '^python' -and
        $_.CommandLine -and
        $_.CommandLine -notmatch 'jedi|language-server|jedilsp|pytest' -and
        (
            $_.CommandLine -match 'sweep_clot|dump_teacher|go_clot|train_biochem|train_clot|go_sweep|go_mlp|go_mu|go_gnode|go_passive|go_kinematics'
        )
    }

if (-not $procs) {
    Add-Content -Path $LogPath -Value "[$stamp] [i] no python processes found"
    Write-Host "[i] no python processes found"
    exit 0
}

foreach ($p in $procs) {
    $line = "pid=$($p.ProcessId) name=$($p.Name) cmd=$($p.CommandLine)"
    Add-Content -Path $LogPath -Value "[$stamp] [kill] $line"
    Write-Host "[kill] $line"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

Add-Content -Path $LogPath -Value "[$stamp] [OK] killed $($procs.Count) process(es)"
Write-Host "[OK] killed $($procs.Count) python process(es); log $LogPath"
