param([ValidateSet('Prepare', 'Move', 'Verify', 'Restore')][string]$Mode = 'Verify')
# Auditable, no-overwrite retirement and ledger-based restoration of VAD files.
$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\RD004\Documents\lab\emotion2vec'
$retired = 'C:\Users\RD004\Documents\lab\emotion2vec_retired_2026-09-08'
$ledgerPath = Join-Path $PSScriptRoot 'ledger.csv'

function Assert-Path([string]$path, [string]$root) {
    $full = [IO.Path]::GetFullPath($path)
    $base = [IO.Path]::GetFullPath($root).TrimEnd('\')
    if ($full -ne $base -and -not $full.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes allowed root: $full"
    }
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse point: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}

function Get-SafeFiles([string]$path) {
    $null = Assert-Path $path $repo
    $item = Get-Item -LiteralPath $path -Force
    if (-not $item.PSIsContainer) { return $item }
    foreach ($child in Get-ChildItem -LiteralPath $path -Force) {
        Get-SafeFiles $child.FullName
    }
}

function Assert-File($row, [string]$path) {
    $item = Get-Item -LiteralPath $path -Force
    if ($item.PSIsContainer -or $item.Length -ne [long]$row.Size -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $row.SHA256) {
        throw "Size/hash mismatch: $path"
    }
}

if ($Mode -eq 'Prepare') {
    if (Test-Path -LiteralPath $ledgerPath) { throw 'Ledger already exists; preparation is immutable.' }
    if (Test-Path -LiteralPath $retired) { throw 'Retirement destination already exists.' }
    $targets = [ordered]@{
        'vad_downstream' = 'Independent VAD package including Python bytecode and notebook autosaves'
        'notebooks/audio_to_emotion_vad.ipynb' = 'VAD-only notebook'
        'tests/test_parallel_emotion_vad.py' = 'VAD-only test'
        'tests/test_notebook_pipeline.py' = 'VAD-only test'
        'tests/execute_demo_notebook.py' = 'VAD-only notebook execution helper'
        'tests/fixtures/vad_dummy/vad_labels_dummy.csv' = 'VAD-only fixture labels; feature cache retained in place'
        'CODE_STRUCTURE_GUIDE.md' = 'VAD-only architecture guide'
        'REAL_EMOTION2VEC_SMOKE_TEST_JA.md' = 'VAD-only real-encoder smoke guide'
        'docs/reports/2026-06-07_vad_cleanup_report.md' = 'VAD-only cleanup report'
        'archive/vad_iemocap_two_stage' = 'Obsolete VAD implementation'
        'archive/notebook_tools' = 'Obsolete notebook editing helper'
        'archive/logs/2026-06-15-work-log.md' = 'Historical VAD work log'
        'archive/logs/2026-06-17-work-log.md' = 'Historical VAD work log'
        'archive/logs/2026-06-18-work-log.md' = 'Historical VAD work log'
        'archive/logs/2026-07-03-work-log.md' = 'Historical VAD work log'
        'archive/logs/2026-07-16-work-log.md' = 'Historical VAD work log'
        'debug.log' = 'Unrelated application error log'
    }
    foreach ($item in Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_vad_downstream*.py' -File) {
        $targets['tests/' + $item.Name] = 'VAD-only test'
    }
    $rows = @(foreach ($target in $targets.Keys) {
        foreach ($file in Get-SafeFiles (Join-Path $repo $target)) {
            $relative = [IO.Path]::GetRelativePath($repo, $file.FullName).Replace('\', '/')
            # Research assets always stay at their existing paths, even inside a target tree.
            if ($relative -match '/cache/|/checkpoints?/' -or $file.Extension -match '^\.(npy|npz|pt|pth|ckpt|png|jpg|jpeg|svg|gif|webp)$') {
                throw "Protected asset inside selected target; review explicitly: $relative"
            }
            $destination = Assert-Path (Join-Path $retired $relative) $retired
            [pscustomobject]@{ RelativePath=$relative; Source=$file.FullName; Destination=$destination;
                Size=$file.Length; SHA256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash;
                Reason=$targets[$target] }
        }
    })
    $rows | Sort-Object RelativePath | Export-Csv -LiteralPath $ledgerPath -NoTypeInformation -Encoding utf8
    git -C $repo -c core.quotepath=false status --porcelain=v1 -uall | Set-Content (Join-Path $PSScriptRoot 'git-status-before.txt') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'Git status failed' }
    $tracked = @(git -C $repo -c core.quotepath=false ls-files)
    if ($LASTEXITCODE -ne 0) { throw 'Git inventory failed' }
    $baseline = @(foreach ($relative in $tracked) {
        $path = Join-Path $repo $relative
        $exists = Test-Path -LiteralPath $path -PathType Leaf
        [pscustomobject]@{ RelativePath=$relative; Exists=$exists;
            SHA256=$(if ($exists) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash } else { $null }) }
    })
    $baseline | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $PSScriptRoot 'tracked-before.json') -Encoding utf8
    $assets = @(foreach ($root in @('runs', 'artifacts', 'tests/fixtures/vad_dummy/cache', 'src')) {
        foreach ($file in Get-SafeFiles (Join-Path $repo $root)) {
            [pscustomobject]@{ RelativePath=[IO.Path]::GetRelativePath($repo,$file.FullName).Replace('\','/');
                Size=$file.Length; LastWriteUtc=$file.LastWriteTimeUtc.ToString('o');
                SHA256=$(if ($file.Extension -match '^\.(png|jpg|jpeg|svg|gif|webp)$' -or $root -eq 'tests/fixtures/vad_dummy/cache') {
                    (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
                } else { $null }) }
        }
    })
    $assets | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $PSScriptRoot 'assets-before.json') -Encoding utf8
    $index = Join-Path $repo '.git/index'
    (Get-FileHash -LiteralPath $index -Algorithm SHA256).Hash | Set-Content (Join-Path $PSScriptRoot 'index-before.sha256')
    "Prepared $($rows.Count) files, $($tracked.Count) tracked paths, $($assets.Count) retained assets."
    exit
}

$rows = @(Import-Csv -LiteralPath $ledgerPath)
if (-not $rows.Count) { throw 'Empty ledger' }
$null = Assert-Path $repo $repo
$null = Assert-Path $retired $retired
foreach ($row in $rows) {
    if ($row.Source -ne (Join-Path $repo $row.RelativePath) -or $row.Destination -ne (Join-Path $retired $row.RelativePath)) {
        # Normalize directory separators before comparison.
        if ([IO.Path]::GetFullPath($row.Source) -ne [IO.Path]::GetFullPath((Join-Path $repo $row.RelativePath)) -or
            [IO.Path]::GetFullPath($row.Destination) -ne [IO.Path]::GetFullPath((Join-Path $retired $row.RelativePath))) { throw 'Ledger path mismatch' }
    }
    $null = Assert-Path $row.Source $repo
    $null = Assert-Path $row.Destination $retired
}

if ($Mode -eq 'Verify') {
    foreach ($row in $rows) {
        if (Test-Path -LiteralPath $row.Source) { throw "Source remains: $($row.Source)" }
        Assert-File $row $row.Destination
    }
    "Verified $($rows.Count) retired files: sizes and SHA-256 match; sources absent."
    exit
}

# Validate every file before the first move. Restoration also supports partial retirement.
foreach ($row in $rows) {
    $from = if ($Mode -eq 'Move') { $row.Source } else { $row.Destination }
    $to = if ($Mode -eq 'Move') { $row.Destination } else { $row.Source }
    if ($Mode -eq 'Restore' -and -not (Test-Path -LiteralPath $from)) {
        Assert-File $row $to
        continue
    }
    if (Test-Path -LiteralPath $to) { throw "Refusing overwrite: $to" }
    Assert-File $row $from
}
if ($Mode -eq 'Move') {
    git -C $repo -c core.quotepath=false status --porcelain=v1 -uall | Set-Content (Join-Path $PSScriptRoot 'git-status-immediately-before-move.txt') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'Git status failed' }
    $rows | Select-Object RelativePath,Size,Reason | Format-Table -AutoSize
    $null = New-Item -ItemType Directory -Path $retired
    # Store the full immutable ledger and recovery tool outside the repo before moving anything.
    Copy-Item -LiteralPath $ledgerPath -Destination (Join-Path $retired 'ledger.csv')
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $retired 'retirement.ps1')
}
foreach ($row in $rows) {
    $from = if ($Mode -eq 'Move') { $row.Source } else { $row.Destination }
    $to = if ($Mode -eq 'Move') { $row.Destination } else { $row.Source }
    if ($Mode -eq 'Restore' -and -not (Test-Path -LiteralPath $from)) { continue }
    $null = Assert-Path $from $(if ($Mode -eq 'Move') { $repo } else { $retired })
    $null = Assert-Path $to $(if ($Mode -eq 'Move') { $retired } else { $repo })
    Assert-File $row $from
    if (Test-Path -LiteralPath $to) { throw "Refusing overwrite: $to" }
    $parent = Split-Path -Parent $to
    if (-not (Test-Path -LiteralPath $parent)) { $null = New-Item -ItemType Directory -Path $parent -Force }
    # File-only move, no recursion, no Force, with an additional no-overwrite precondition.
    Move-Item -LiteralPath $from -Destination $to
    Assert-File $row $to
    if (Test-Path -LiteralPath $from) { throw "Move did not remove source: $from" }
    [pscustomobject]@{ Mode=$Mode; RelativePath=$row.RelativePath; VerifiedUtc=[DateTime]::UtcNow.ToString('o') } |
        ConvertTo-Json -Compress | Add-Content (Join-Path $PSScriptRoot 'operations.jsonl') -Encoding utf8
}
"$Mode completed: $($rows.Count) ledger files verified. Empty source directories are retained."
