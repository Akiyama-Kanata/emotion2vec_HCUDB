[CmdletBinding()]
param(
    [ValidateRange(1, 1000000)]
    [int]$StartBatch = 40,

    [ValidateRange(1, 1000000)]
    [int]$EndBatch = 52,

    [ValidateRange(1, 10000)]
    [int]$BatchSize = 500,

    [string]$MetadataCsv = (
        'C:\Users\RD004\Box\知能情報処理システム研究室\音声\' +
        '音声データベース\MSP_PODCAST\Labels\labels_consensus.csv'
    ),

    [string]$BoxAudioDir = (
        'C:\Users\RD004\Box\知能情報処理システム研究室\音声\' +
        '音声データベース\MSP_PODCAST\Audio'
    ),

    [string]$DestinationRoot = (
        'C:\Users\RD004\Documents\lab\data\MSP_PODCAST\Audio'
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if ($StartBatch -gt $EndBatch) {
    throw 'StartBatch must be less than or equal to EndBatch.'
}

$downloadScript = Join-Path $PSScriptRoot 'download_msp_wavs.ps1'
if (-not (Test-Path -LiteralPath $downloadScript -PathType Leaf)) {
    throw 'download_msp_wavs.ps1 was not found.'
}

foreach ($batchNumber in $StartBatch..$EndBatch) {
    Write-Output (
        '[{0:yyyy-MM-dd HH:mm:ss}] Starting batch {1} / {2}' -f
        (Get-Date),
        $batchNumber,
        $EndBatch
    )

    & $downloadScript `
        -MetadataCsv $MetadataCsv `
        -BoxAudioDir $BoxAudioDir `
        -DestinationRoot $DestinationRoot `
        -BatchNumber $batchNumber `
        -BatchSize $BatchSize
}

Write-Output (
    '[{0:yyyy-MM-dd HH:mm:ss}] All requested batches completed.' -f
    (Get-Date)
)
