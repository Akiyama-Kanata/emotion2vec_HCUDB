$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$workspaceDir = 'C:\Users\RD004\Documents\lab\emotion2vec'
$progressLog = Join-Path $workspaceDir 'msp_download_progress.csv'
$retryLog = Join-Path $workspaceDir 'msp_download_retry_progress.csv'
$downloadScript = Join-Path $workspaceDir 'download_msp_wavs.ps1'

$metadataCsv = 'C:\Users\RD004\Box\知能情報処理システム研究室\音声\音声データベース\MSP_PODCAST\Labels\labels_consensus.csv'
$boxAudioDir = 'C:\Users\RD004\Box\知能情報処理システム研究室\音声\音声データベース\MSP_PODCAST\Audio'
$destinationRoot = 'C:\Users\RD004\Documents\lab\data\MSP_PODCAST\Audio'

if (Test-Path -LiteralPath $retryLog) {
    exit 0
}

while ($true) {
    if (Test-Path -LiteralPath $progressLog) {
        $batch52 = @(
            Import-Csv -LiteralPath $progressLog |
                Where-Object { [int]$_.BatchNumber -eq 52 }
        )
        if ($batch52.Count -gt 0) {
            break
        }
    }

    Start-Sleep -Seconds 60
}

$retryResults = foreach ($batchNumber in @(22, 39)) {
    & $downloadScript `
        -MetadataCsv $metadataCsv `
        -BoxAudioDir $boxAudioDir `
        -DestinationRoot $destinationRoot `
        -BatchNumber $batchNumber `
        -BatchSize 500
}

$retryResults |
    Where-Object { $_.PSObject.Properties.Name -contains 'BatchNumber' } |
    Export-Csv -LiteralPath $retryLog -NoTypeInformation
