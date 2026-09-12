[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MetadataCsv,

    [Parameter(Mandatory = $true)]
    [string]$BoxAudioDir,

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot,

    [ValidateRange(1, 1000000)]
    [int]$BatchNumber = 1,

    [ValidateRange(1, 10000)]
    [int]$BatchSize = 500,

    [ValidateRange(1, 3600)]
    [int]$VerificationTimeoutSeconds = 120,

    [ValidateSet('A', 'C', 'D', 'F', 'H', 'N', 'O', 'S', 'U', 'X')]
    [string[]]$EmotionCodes = @('A', 'H', 'S', 'D')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if (-not (Test-Path -LiteralPath $MetadataCsv -PathType Leaf)) {
    throw 'The metadata CSV was not found.'
}

if (-not (Test-Path -LiteralPath $BoxAudioDir -PathType Container)) {
    throw 'The Box Drive audio directory was not found.'
}

$normalizedEmotionCodes = @($EmotionCodes | ForEach-Object { $_.Trim().ToUpperInvariant() } | Select-Object -Unique)
if ($normalizedEmotionCodes.Count -eq 0) {
    throw 'At least one emotion code is required.'
}

New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

$batchDestinationDir = Join-Path $DestinationRoot (
    'batch_{0:D3}' -f $BatchNumber
)
New-Item -ItemType Directory -Force -Path $batchDestinationDir | Out-Null

$metadataRows = @(Import-Csv -LiteralPath $MetadataCsv)
if ($metadataRows.Count -eq 0) {
    throw 'The metadata CSV is empty.'
}

$requiredColumns = @(
    'FileName'
    'EmoClass'
    'SpkrID'
    'Split_Set'
)
$actualColumns = @($metadataRows[0].PSObject.Properties.Name)
$missingColumns = @(
    $requiredColumns | Where-Object { $_ -notin $actualColumns }
)
if ($missingColumns.Count -gt 0) {
    throw "Required CSV columns are missing: $($missingColumns -join ', ')"
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
            $emotion.Trim().ToUpperInvariant() -in $normalizedEmotionCodes -and
            $speaker.Trim().ToLowerInvariant() -ne 'unknown' -and
            $split -in @('Train', 'Development', 'Test1')
        } |
        Sort-Object -Property FileName
)

$skipCount = ($BatchNumber - 1) * $BatchSize
$batch = @(
    $allTargets | Select-Object -Skip $skipCount -First $BatchSize
)
if ($batch.Count -eq 0) {
    throw 'The requested batch has no target WAV files.'
}

# Batches are defined after emotion filtering, so the same utterance can move to
# another batch when EmotionCodes changes. Index the entire destination tree by
# basename to keep resumptions idempotent across those changes.
$existingAudioByName = [System.Collections.Generic.Dictionary[string, string]]::new(
    [System.StringComparer]::OrdinalIgnoreCase
)
Get-ChildItem -LiteralPath $DestinationRoot -Recurse -Filter '*.wav' -File |
    ForEach-Object {
        if ($existingAudioByName.ContainsKey($_.Name)) {
            throw (
                "Duplicate destination WAV basename detected: {0}; {1}" -f
                $existingAudioByName[$_.Name],
                $_.FullName
            )
        }
        $existingAudioByName.Add($_.Name, $_.FullName)
    }

$completed = 0
$missing = 0
$copied = 0
$skipped = 0
$failed = 0
$zeroLengthSource = 0
$processed = 0

foreach ($item in $batch) {
    $fileName = [string]$item.FileName
    $sourceFile = Join-Path $BoxAudioDir $fileName
    $destinationFile = if ($existingAudioByName.ContainsKey($fileName)) {
        $existingAudioByName[$fileName]
    }
    else {
        Join-Path $batchDestinationDir $fileName
    }

    if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
        $missing++
        $processed++
        continue
    }

    $sourceSize = (Get-Item -LiteralPath $sourceFile).Length
    if ($sourceSize -le 0) {
        $zeroLengthSource++
        $failed++
        $processed++
        $progress = @{
            Activity = 'WAV download'
            Status = "$processed / $($batch.Count)"
            PercentComplete = (($processed / $batch.Count) * 100)
        }
        Write-Progress @progress
        continue
    }

    if (Test-Path -LiteralPath $destinationFile -PathType Leaf) {
        $destinationSize = (Get-Item -LiteralPath $destinationFile).Length
        if ($sourceSize -gt 0 -and $destinationSize -eq $sourceSize) {
            $completed++
            $skipped++
            $processed++
            $progress = @{
                Activity = 'WAV download'
                Status = "$processed / $($batch.Count)"
                PercentComplete = (($processed / $batch.Count) * 100)
            }
            Write-Progress @progress
            continue
        }
        if ($destinationSize -gt 0) {
            # Never overwrite a non-empty local file whose size disagrees with
            # the source. Treat it as a data-integrity failure for review.
            $failed++
            $processed++
            continue
        }
    }

    $destinationDirectory = Split-Path -Parent $destinationFile
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

    $robocopyArguments = @(
        $BoxAudioDir
        $destinationDirectory
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

    & robocopy @robocopyArguments *> $null
    $robocopyExitCode = $LASTEXITCODE
    if ($robocopyExitCode -ge 8) {
        $failed++
        $processed++
        continue
    }

    $sizeMatched = $false
    for ($wait = 0; $wait -lt $VerificationTimeoutSeconds; $wait++) {
        if (Test-Path -LiteralPath $destinationFile -PathType Leaf) {
            $sourceSize = (Get-Item -LiteralPath $sourceFile).Length
            $destinationSize = (Get-Item -LiteralPath $destinationFile).Length
            if ($sourceSize -gt 0 -and $destinationSize -eq $sourceSize) {
                $sizeMatched = $true
                break
            }
        }
        Start-Sleep -Seconds 1
    }

    if (-not $sizeMatched) {
        $failed++
        $processed++
        continue
    }

    $completed++
    $copied++
    $processed++
    if (-not $existingAudioByName.ContainsKey($fileName)) {
        $existingAudioByName.Add($fileName, $destinationFile)
    }
    $progress = @{
        Activity = 'WAV download'
        Status = "$processed / $($batch.Count)"
        PercentComplete = (($processed / $batch.Count) * 100)
    }
    Write-Progress @progress
}

Write-Progress -Activity 'WAV download' -Completed

$summary = [pscustomobject]@{
    BatchNumber = $BatchNumber
    BatchSize = $BatchSize
    Selected = $batch.Count
    Completed = $completed
    Copied = $copied
    Skipped = $skipped
    Missing = $missing
    Failed = $failed
    ZeroLengthSource = $zeroLengthSource
    EmotionCodes = ($normalizedEmotionCodes -join ',')
}

$progressLog = Join-Path $PSScriptRoot 'msp_download_progress.csv'
if (Test-Path -LiteralPath $progressLog) {
    $summary | Export-Csv -LiteralPath $progressLog -NoTypeInformation -Append
}
else {
    $summary | Export-Csv -LiteralPath $progressLog -NoTypeInformation
}

$summary
