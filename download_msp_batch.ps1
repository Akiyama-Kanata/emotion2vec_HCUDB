# ===== 自分の環境に合わせて変更 =====

$ErrorActionPreference = 'Stop'

$MetadataCsv = 'C:\Users\RD004\Box\知能情報処理システム研究室\音声\音声データベース\MSP_PODCAST\Labels\labels_consensus.csv'
$BoxAudioDir = 'C:\Users\RD004\Box\知能情報処理システム研究室\音声\音声データベース\MSP_PODCAST\Audio'
$DestinationRoot = 'C:\Users\RD004\Documents\lab\data\MSP_PODCAST\Audio'



$BatchSize = 500
$BatchNumber = 1

if (-not (Test-Path -LiteralPath $MetadataCsv -PathType Leaf)) {
    throw 'metadata CSVが見つかりません。'
}

if (-not (Test-Path -LiteralPath $BoxAudioDir -PathType Container)) {
    throw 'Box DriveのAudioフォルダが見つかりません。'
}

if ($BatchNumber -lt 1) {
    throw 'BatchNumberは1以上にしてください。'
}

$Destination = Join-Path $DestinationRoot (
    'batch_{0:D3}' -f $BatchNumber
)

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$metadataRows = @(
    Import-Csv -LiteralPath $MetadataCsv
)

if ($metadataRows.Count -eq 0) {
    throw 'metadata CSVを読み込めませんでした。'
}

$requiredColumns = @(
    'FileName'
    'EmoClass'
    'SpkrID'
    'Split_Set'
)

$actualColumns = @(
    $metadataRows[0].PSObject.Properties.Name
)

$missingColumns = @(
    $requiredColumns |
        Where-Object { $_ -notin $actualColumns }
)

if ($missingColumns.Count -gt 0) {
    throw "CSVに必要な列がありません: $($missingColumns -join ', ')"
}

$allTargets = @(
    $metadataRows |
        Where-Object {
            $fileName = [string]$_.FileName
            $emotion = [string]$_.EmoClass
            $speaker = [string]$_.SpkrID
            $split = [string]$_.Split_Set

            $isWav = $fileName.EndsWith(
                '.wav',
                [System.StringComparison]::OrdinalIgnoreCase
            )

            $isWav -and
            $emotion -in @('A', 'H', 'S', 'D') -and
            $speaker.Trim().ToLowerInvariant() -ne 'unknown' -and
            $split -in @('Train', 'Development', 'Test1')
        } |
        Sort-Object -Property FileName
)

$skipCount = ($BatchNumber - 1) * $BatchSize

$batch = @(
    $allTargets |
        Select-Object -Skip $skipCount -First $BatchSize
)

if ($batch.Count -eq 0) {
    throw '指定したバッチに対象WAVがありません。'
}

$completed = 0
$missing = 0

foreach ($item in $batch) {
    $fileName = [string]$item.FileName
    $source = Join-Path $BoxAudioDir $fileName
    $destination = Join-Path $Destination $fileName

    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        $missing++
        continue
    }

    $sourceSize = (Get-Item -LiteralPath $source).Length

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $destinationSize = (
            Get-Item -LiteralPath $destination
        ).Length

        if (
            $sourceSize -gt 0 -and
            $destinationSize -eq $sourceSize
        ) {
            $completed++

            Write-Progress -Activity 'WAV download' `
                -Status "$completed / $($batch.Count)" `
                -PercentComplete (($completed / $batch.Count) * 100)

            continue
        }
    }

    $robocopyArguments = @(
        $BoxAudioDir
        $Destination
        $fileName
        '/J'
        '/R:3'
        '/W:5'
        '/COPY:DAT'
        '/NP'
        '/NFL'
        '/NDL'
        '/NJH'
        '/NJS'
    )

    & robocopy @robocopyArguments
    $robocopyExitCode = $LASTEXITCODE

    if ($robocopyExitCode -ge 8) {
        throw "WAVコピーに失敗しました。robocopy終了コード: $robocopyExitCode"
    }

    $sizeMatched = $false

    for ($wait = 0; $wait -lt 120; $wait++) {
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $sourceSize = (Get-Item -LiteralPath $source).Length
            $destinationSize = (
                Get-Item -LiteralPath $destination
            ).Length

            if (
                $sourceSize -gt 0 -and
                $destinationSize -eq $sourceSize
            ) {
                $sizeMatched = $true
                break
            }
        }

        Start-Sleep -Seconds 1
    }

    if (-not $sizeMatched) {
        throw 'WAVの取得完了を120秒以内に確認できませんでした。'
    }

    $completed++

    Write-Progress -Activity 'WAV download' `
        -Status "$completed / $($batch.Count)" `
        -PercentComplete (($completed / $batch.Count) * 100)
}

Write-Progress -Activity 'WAV download' -Completed

Write-Host "Completed: $completed / $($batch.Count)"
Write-Host "Missing: $missing"
Write-Host "Destination: $Destination"