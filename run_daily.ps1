$ErrorActionPreference = "Continue"
Set-Location "$HOME\nepse-ranker"
.\.venv\Scripts\Activate.ps1
New-Item -ItemType Directory -Force -Path logs | Out-Null
$log = "logs\daily.log"

function Write-Log($text) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $text" | Add-Content -Path $log
}

function Invoke-Step($name, $cmdArgs) {
    Write-Log "START $name"
    $output = & python -m nepse_ranker @cmdArgs 2>&1
    $code = $LASTEXITCODE
    $output | Select-Object -Last 12 | ForEach-Object { Add-Content -Path $log -Value "    $_" }
    if ($code -eq 0) { Write-Log "OK $name" } else { Write-Log "FAILED $name (exit code $code)" }
    return $code
}

Write-Log "BEGIN daily run"
$failed = @()
if ((Invoke-Step "floorsheet" @("floorsheet-collect")) -ne 0) { $failed += "floorsheet" }
if ((Invoke-Step "floorsheet-build" @("floorsheet-build")) -ne 0) { $failed += "floorsheet-build" }
if ((Invoke-Step "prices" @("update")) -ne 0) { $failed += "prices" }
if ((Invoke-Step "news-fetch" @("news-fetch")) -ne 0) { $failed += "news-fetch" }
foreach ($step in @("build", "features", "news-build", "predict")) {
    if ((Invoke-Step $step @($step)) -ne 0) { $failed += $step; break }
}
Invoke-Step "status" @("status") | Out-Null
& "$HOME\nepse-ranker\backup.ps1" | ForEach-Object { Write-Log "backup: $_" }
if ($failed.Count -eq 0) { Write-Log "RUN OK" } else { Write-Log ("RUN FINISHED WITH FAILURES: " + ($failed -join ", ")) }
