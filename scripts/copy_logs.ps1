$source = "C:\Users\Ksawery\Desktop\logi"
$destinationRoot = "C:\Users\Ksawery\PycharmProjects\abc-benchmark\artifacts\results"

Get-ChildItem -Path $source -Filter *.log -File | ForEach-Object {

    $file = $_
    $content = Get-Content $file.FullName

    $taskLine = $content | Where-Object { $_ -match "=== .* : info ===" } | Select-Object -First 1
    if (-not $taskLine) { return }

    if ($taskLine -match "=== (.*) : info ===") {
        $task = $matches[1]
    } else {
        return
    }

    # podział: "selective attention - structure sensitive text filtering"
    $parts = $task -split " - "

    if ($parts.Count -lt 2) { return }

    # pierwszy segment
    $firstPart = ($parts[0] -replace " ", "_")

    # drugi segment rozbijamy na słowa
    $words = $parts[1] -split " "

    if ($words.Count -lt 3) { return }

    # "structure sensitive" jako jeden katalog
    $secondPart = ($words[0..1] -join "_")

    # reszta jako osobne katalogi
    $restParts = $words[2..($words.Count - 1)]

    $folderPath = Join-Path $firstPart $secondPart
    foreach ($w in $restParts) {
        $folderPath = Join-Path $folderPath $w
    }

    $runIdLine = $content | Where-Object { $_ -match "run id:" } | Select-Object -First 1
    if (-not $runIdLine) { return }

    if ($runIdLine -match "run id:\s*(.*)") {
        $runId = $matches[1]
    } else {
        return
    }

    $finalDir = Join-Path $destinationRoot $folderPath

    if (!(Test-Path $finalDir)) {
        New-Item -ItemType Directory -Path $finalDir -Force | Out-Null
    }

    $newFileName = "$runId.log"
    $destinationPath = Join-Path $finalDir $newFileName

    Copy-Item -Path $file.FullName -Destination $destinationPath -Force

    Write-Host "Skopiowano: $($file.Name) -> $destinationPath"
}
