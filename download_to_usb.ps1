$RunId = "28648790755"
while ($true) {
    Remove-Item Env:\GITHUB_TOKEN -ErrorAction SilentlyContinue
    $status = gh run view $RunId --json status -q ".status"
    if ($status -eq "completed") {
        Write-Host "Run completed! Downloading artifact to USB drive F:\..."
        gh run download $RunId -n yolop-adas-arm64 -D F:\
        Write-Host "Download finished!"
        break
    }
    Write-Host "Waiting for GitHub Action to finish..."
    Start-Sleep -Seconds 60
}
