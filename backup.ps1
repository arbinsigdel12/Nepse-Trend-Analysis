$src = "$HOME\nepse-ranker\data"
$dest = if (Test-Path "$HOME\OneDrive") { "$HOME\OneDrive\nepse-backup" } else { "$HOME\Documents\nepse-backup" }
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$folders = @("floorsheet", "watchlist", "company", "imported", "raw", "news\raw")
foreach ($f in $folders) {
    if (Test-Path "$src\$f") {
        robocopy "$src\$f" "$dest\$f" /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Write-Output "BACKUP FAILED for $f (robocopy code $LASTEXITCODE)" }
    }
}
Write-Output "Backup finished: $dest"
