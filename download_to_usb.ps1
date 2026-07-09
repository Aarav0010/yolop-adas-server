# Clear bad environment token before running any GH CLI commands
Remove-Item Env:\GITHUB_TOKEN -ErrorAction SilentlyContinue

# Automatically grab the latest GitHub Action run ID
$RunId = gh run list --limit 1 --json databaseId -q ".[0].databaseId"
Write-Host "Tracking latest GitHub Action Run ID: $RunId"

while ($true) {
    $status = gh run view $RunId --json status -q ".status"
    $conclusion = gh run view $RunId --json conclusion -q ".conclusion"
    
    if ($status -eq "completed") {
        if ($conclusion -eq "success") {
            Write-Host "Run completed successfully! Downloading artifact to D:\ drive..."
            gh run download $RunId -n yolop-adas-arm64 -D D:\
            Write-Host "Download finished! Copying to USB drive F:\..."
            Copy-Item -Path D:\yolop-adas-arm64.tar -Destination F:\ -Force
            Write-Host "USB Transfer Complete! You can now safely eject the USB drive."
        } else {
            Write-Host "Warning: The GitHub Action failed with conclusion: $conclusion"
        }
        break
    }
    Write-Host "Waiting for GitHub Action to finish... (Status: $status)"
    Start-Sleep -Seconds 30
}
