$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")
$sourcePath = Join-Path $root "23RD004_秋山叶太_研究計画.pptx"
$outputPath = Join-Path $root "23RD004_秋山叶太_研究計画_2026-07進捗追加.pptx"
$previewDir = Join-Path $PSScriptRoot "2026-07-progress-added-slides-preview"

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Source pptx not found: $sourcePath"
}

New-Item -ItemType Directory -Force -Path $previewDir | Out-Null
$sourceHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash
Copy-Item -LiteralPath $sourcePath -Destination $outputPath -Force

$ppLayoutBlank = 12
$ppSaveAsOpenXMLPresentation = 24
$msoTextOrientationHorizontal = 1
$msoTrue = -1
$msoFalse = 0

$shapeRectangle = 1
$shapeRoundedRectangle = 5
$shapeRightArrow = 33
$shapeCheck = 50

function Rgb([int]$r, [int]$g, [int]$b) {
    return $r + ($g * 256) + ($b * 65536)
}

$Color = @{
    Ink = Rgb 30 41 59
    Muted = Rgb 71 85 105
    Navy = Rgb 21 47 80
    Blue = Rgb 37 99 235
    PaleBlue = Rgb 239 246 255
    Teal = Rgb 14 116 144
    PaleTeal = Rgb 236 253 245
    Green = Rgb 22 101 52
    PaleGreen = Rgb 240 253 244
    Amber = Rgb 180 83 9
    PaleAmber = Rgb 255 251 235
    Red = Rgb 185 28 28
    PaleRed = Rgb 254 242 242
    Gray = Rgb 226 232 240
    MidGray = Rgb 148 163 184
    White = Rgb 255 255 255
    OffWhite = Rgb 248 250 252
}

function Set-TextStyle($shape, [double]$size, [int]$color, [bool]$bold = $false, [int]$align = 1) {
    $range = $shape.TextFrame.TextRange
    $range.Font.Name = "Yu Gothic"
    try { $range.Font.NameFarEast = "Yu Gothic" } catch {}
    $range.Font.Size = $size
    $range.Font.Color.RGB = $color
    $range.Font.Bold = $(if ($bold) { $msoTrue } else { $msoFalse })
    $range.ParagraphFormat.Alignment = $align
    $shape.TextFrame.WordWrap = $msoTrue
    $shape.TextFrame.MarginLeft = 7
    $shape.TextFrame.MarginRight = 7
    $shape.TextFrame.MarginTop = 4
    $shape.TextFrame.MarginBottom = 4
}

function Add-Text($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [double]$size = 18, [int]$color = $Color.Ink, [bool]$bold = $false, [int]$align = 1) {
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $x, $y, $w, $h)
    $shape.TextFrame.TextRange.Text = $text
    Set-TextStyle $shape $size $color $bold $align
    return $shape
}

function Add-Box($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$line, [double]$size = 16, [int]$textColor = $Color.Ink, [bool]$bold = $false, [int]$shapeType = $shapeRectangle) {
    $shape = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.ForeColor.RGB = $line
    $shape.Line.Weight = 1.1
    $shape.TextFrame.TextRange.Text = $text
    Set-TextStyle $shape $size $textColor $bold 2
    return $shape
}

function Add-Header($slide, [string]$numberText, [string]$headline) {
    Add-Text $slide $numberText 828 22 90 20 12 $Color.Muted $true 3 | Out-Null
    Add-Text $slide $headline 42 25 765 58 22 $Color.Navy $true 1 | Out-Null
    $line = $slide.Shapes.AddShape($shapeRectangle, 42, 88, 876, 2)
    $line.Fill.ForeColor.RGB = $Color.Gray
    $line.Line.Visible = $msoFalse
}

function Add-FooterNote($slide, [string]$text, [int]$lineColor = $Color.Amber, [int]$fill = $Color.PaleAmber) {
    Add-Box $slide $text 82 456 796 42 $fill $lineColor 15 $Color.Ink $true | Out-Null
}

function Add-Arrow($slide, [double]$x, [double]$y, [double]$w = 32, [double]$h = 18, [int]$color = $Color.Muted) {
    $shape = $slide.Shapes.AddShape($shapeRightArrow, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $color
    $shape.Line.Visible = $msoFalse
    return $shape
}

function Add-FlowStep($slide, [string]$label, [string]$detail, [double]$x, [double]$y, [double]$w, [int]$fill, [int]$accent) {
    $text = $label + "`r" + $detail
    $shape = Add-Box $slide $text $x $y $w 86 $fill $accent 13.5 $Color.Ink $false $shapeRoundedRectangle
    $range = $shape.TextFrame.TextRange
    $range.Characters(1, $label.Length).Font.Bold = $msoTrue
    $range.Characters(1, $label.Length).Font.Size = 15.5
    $range.Characters(1, $label.Length).Font.Color.RGB = $accent
    return $shape
}

function Add-ChecklistItem($slide, [string]$text, [double]$x, [double]$y, [double]$w, [int]$accent = $Color.Blue) {
    $mark = $slide.Shapes.AddShape($shapeCheck, $x, $y + 5, 24, 24)
    $mark.Fill.ForeColor.RGB = $Color.PaleGreen
    $mark.Line.ForeColor.RGB = $Color.Green
    $mark.Line.Weight = 1
    Add-Text $slide $text ($x + 38) $y $w 34 18 $Color.Ink $true 1 | Out-Null
}

function Add-TableCell($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$line, [double]$size = 16, [int]$color = $Color.Ink, [bool]$bold = $false, [int]$align = 1) {
    $shape = Add-Box $slide $text $x $y $w $h $fill $line $size $color $bold $shapeRectangle
    $shape.TextFrame.MarginLeft = 12
    $shape.TextFrame.MarginRight = 12
    return $shape
}

function Set-SpeakerNotes($slide, [string]$notes) {
    $text = $notes.Trim()
    $set = $false
    try {
        $placeholder = $slide.NotesPage.Shapes.Placeholders(2)
        $placeholder.TextFrame.TextRange.Text = $text
        $set = $true
    }
    catch {}

    if (-not $set) {
        $shape = $slide.NotesPage.Shapes.AddTextbox($msoTextOrientationHorizontal, 72, 120, 520, 280)
        $shape.TextFrame.TextRange.Text = $text
        Set-TextStyle $shape 12 $Color.Ink $false 1
    }
}

function Update-ShapeSlideNumber($shape, [string]$newText) {
    try {
        if ($shape.Type -eq 6) {
            for ($i = 1; $i -le $shape.GroupItems.Count; $i++) {
                Update-ShapeSlideNumber $shape.GroupItems.Item($i) $newText
            }
        }
    }
    catch {}

    try {
        if ($shape.HasTextFrame -eq $msoTrue -and $shape.TextFrame.HasText -eq $msoTrue) {
            $text = $shape.TextFrame.TextRange.Text.Trim()
            if ($text -match '^\d+\s*/\s*\d+$') {
                $shape.TextFrame.TextRange.Text = $newText
            }
        }
    }
    catch {}
}

function Update-AllSlideNumbers($presentation) {
    for ($i = 2; $i -le $presentation.Slides.Count; $i++) {
        $slide = $presentation.Slides.Item($i)
        $newText = "{0} / 12" -f ($i - 1)
        for ($j = 1; $j -le $slide.Shapes.Count; $j++) {
            Update-ShapeSlideNumber $slide.Shapes.Item($j) $newText
        }
    }
}

function Add-ProgressSlide8($presentation) {
    $slide = $presentation.Slides.Add(8, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide "7 / 12" "今回の進捗は、VADを経由して分類根拠を確認する実装基盤である"

    Add-FlowStep $slide "emotion2vec特徴" "音声表現`r768次元" 50 154 150 $Color.PaleBlue $Color.Blue | Out-Null
    Add-Arrow $slide 210 186 32 18 $Color.Muted | Out-Null
    Add-FlowStep $slide "pooling / FNN" "時系列集約`r非線形変換" 252 154 144 $Color.OffWhite $Color.MidGray | Out-Null
    Add-Arrow $slide 406 186 32 18 $Color.Muted | Out-Null
    Add-FlowStep $slide "predicted VAD" "Valence`rArousal`rDominance" 448 154 142 $Color.PaleTeal $Color.Teal | Out-Null
    Add-Arrow $slide 600 186 32 18 $Color.Muted | Out-Null
    Add-FlowStep $slide "Linear" "VADのみを入力`rlogitへ変換" 642 154 122 $Color.PaleAmber $Color.Amber | Out-Null
    Add-Arrow $slide 774 186 32 18 $Color.Muted | Out-Null
    Add-FlowStep $slide "emotion" "hap / sad`rang / dis" 814 154 104 $Color.PaleBlue $Color.Blue | Out-Null

    Add-Box $slide "fine-tuning結果ではない" 86 316 236 54 $Color.PaleRed $Color.Red 18 $Color.Ink $true | Out-Null
    Add-Box $slide "分類器入力は予測VADのみ" 362 316 236 54 $Color.PaleTeal $Color.Teal 18 $Color.Ink $true | Out-Null
    Add-Box $slide "VAD次元ごとのlogit寄与を出力" 638 316 236 54 $Color.PaleBlue $Color.Blue 17 $Color.Ink $true | Out-Null
    Add-FooterNote $slide "性能改善は未測定。直接分類より優れるとは現時点では主張しない。" $Color.Red $Color.PaleRed

    Set-SpeakerNotes $slide @"
VAD媒介型分類は最終性能の報告ではなく、今後の比較評価で使う経路であると説明する。

分類器の入力はemotion2vec特徴そのものではなく、予測されたVAD値に制限する。これにより、emotion logitに対するValence/Arousal/Dominance各次元の寄与を確認できる。

caveat: 性能改善は未測定。直接分類より優れるとは主張しない。fine-tuning結果ではなく、分類根拠を確認するための実装基盤として扱う。
"@
}

function Add-ProgressSlide9($presentation) {
    $slide = $presentation.Slides.Add(9, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide "8 / 12" "実装済み範囲は、学習から推論JSONまでの評価経路である"

    Add-ChecklistItem $slide "データ読み込み" 96 138 320 $Color.Blue
    Add-ChecklistItem $slide "VAD+感情分類loss" 96 202 320 $Color.Teal
    Add-ChecklistItem $slide "学習CLI" 96 266 320 $Color.Blue
    Add-ChecklistItem $slide "推論JSON" 96 330 320 $Color.Teal
    Add-ChecklistItem $slide "寄与分解・README・関連テスト" 96 394 470 $Color.Blue

    Add-Box $slide "実装単位`r機能で整理" 606 144 212 70 $Color.PaleBlue $Color.Blue 18 $Color.Ink $true | Out-Null
    Add-Box $slide "主な配置`rvad_downstream 配下" 606 246 212 70 $Color.PaleTeal $Color.Teal 18 $Color.Ink $true | Out-Null
    Add-Box $slide "未完了`rPyTorch依存テスト完走" 606 348 212 70 $Color.PaleRed $Color.Red 17 $Color.Ink $true | Out-Null

    Set-SpeakerNotes $slide @"
主な実装箇所は vad_downstream 配下。スライド上ではファイル名ではなく機能単位で示す。

データ読み込み: vad_downstream/data.py
モデル: vad_downstream/model.py
VAD+感情分類の学習経路: vad_downstream/emotion_training.py, vad_downstream/training.py
学習CLI: vad_downstream/train_vad_emotion.py, vad_downstream/train_head.py
推論JSONと寄与分解: vad_downstream/infer_vad_emotion.py, vad_downstream/inference.py
説明と利用手順: vad_downstream/README.md
関連テスト: tests/test_vad_downstream_*.py

caveat: PyTorch依存テスト完走は未完了。依存入り環境での再実行が必要。
"@
}

function Add-ProgressSlide10($presentation) {
    $slide = $presentation.Slides.Add(10, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide "9 / 12" "現時点で確認済みなのは構文・差分であり、性能結果はまだ主張しない"

    Add-TableCell $slide "確認済み" 78 128 382 48 $Color.PaleTeal $Color.Teal 21 $Color.Teal $true 2 | Out-Null
    Add-TableCell $slide "未完了" 500 128 382 48 $Color.PaleRed $Color.Red 21 $Color.Red $true 2 | Out-Null

    Add-TableCell $slide "py_compile exit 0" 78 194 382 52 $Color.White $Color.Gray 18 $Color.Ink $true 1 | Out-Null
    Add-TableCell $slide "git diff --check exit 0" 78 262 382 52 $Color.White $Color.Gray 18 $Color.Ink $true 1 | Out-Null

    Add-TableCell $slide "実checkpoint推論" 500 194 382 42 $Color.White $Color.Gray 16.5 $Color.Ink $true 1 | Out-Null
    Add-TableCell $slide "日本語fine-tuning" 500 248 382 42 $Color.White $Color.Gray 16.5 $Color.Ink $true 1 | Out-Null
    Add-TableCell $slide "日本語・英語SER評価" 500 302 382 42 $Color.White $Color.Gray 16.5 $Color.Ink $true 1 | Out-Null
    Add-TableCell $slide "VAD CCC / WA / UA / F1" 500 356 382 42 $Color.White $Color.Gray 16.5 $Color.Ink $true 1 | Out-Null

    Add-FooterNote $slide "CCC、WA、UA、F1、confusion matrix は未測定。" $Color.Red $Color.PaleRed

    Set-SpeakerNotes $slide @"
現時点で確認済みなのは構文確認と差分チェックである。性能指標はまだ提示しない。

Windows側Python環境で依存が不足しており、性能評価には依存入り環境が必要であると説明する。

未完了: 実checkpoint推論、日本語fine-tuning、日本語・英語SER評価、VAD CCC、WA、UA、F1、confusion matrix。

caveat: CCC、WA、UA、F1、confusion matrix は未測定。数値結果や性能改善は今後の評価後に主張する。
"@
}

function Add-ProgressSlide11($presentation) {
    $slide = $presentation.Slides.Add(11, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide "10 / 12" "次は実データ評価で、日本語SER改善と英語SER維持を数値化する"

    $items = @(
        @("1", "依存入り環境で`rテスト完走", $Color.Blue, $Color.PaleBlue),
        @("2", "hap/sad/ang/dis`r分布確認", $Color.Teal, $Color.PaleTeal),
        @("3", "VAD媒介型分類を`r実データ評価", $Color.Amber, $Color.PaleAmber),
        @("4", "fine-tuning前後で`r日英SER比較", $Color.Green, $Color.PaleGreen)
    )
    $x = 72
    foreach ($item in $items) {
        $num = $slide.Shapes.AddShape($shapeRoundedRectangle, $x, 136, 54, 54)
        $num.Fill.ForeColor.RGB = $item[3]
        $num.Line.ForeColor.RGB = $item[2]
        $num.TextFrame.TextRange.Text = $item[0]
        Set-TextStyle $num 24 $item[2] $true 2

        Add-Box $slide $item[1] ($x - 20) 216 166 94 $Color.White $Color.Gray 17 $Color.Ink $true | Out-Null
        if ($item[0] -ne "4") {
            Add-Arrow $slide ($x + 156) 257 38 18 $Color.Muted | Out-Null
        }
        $x += 210
    }

    Add-Box $slide "先に確認`rdis のfold内分布" 132 366 244 58 $Color.PaleAmber $Color.Amber 17 $Color.Ink $true | Out-Null
    Add-Box $slide "数値化`r日本語SER改善" 428 366 200 58 $Color.PaleBlue $Color.Blue 17 $Color.Ink $true | Out-Null
    Add-Box $slide "同時確認`r英語SER維持" 676 366 200 58 $Color.PaleTeal $Color.Teal 17 $Color.Ink $true | Out-Null

    Set-SpeakerNotes $slide @"
次の段階では実データ評価に移り、日本語SER改善と英語SER維持を数値化する。

dis のfold内分布確認を先に行い、クラス偏りやfold偏りによって評価値の解釈を誤らないようにする。

ロードマップ:
1. 依存入り環境でテスト完走
2. hap/sad/ang/dis分布確認
3. VAD媒介型分類を実データ評価
4. fine-tuning前後で日英SER比較

caveat: 英語性能維持と日本語性能改善は今後の比較実験で確認する。
"@
}

$ppt = $null
$prs = $null
$reopened = $null

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = $msoTrue
    $prs = $ppt.Presentations.Open($outputPath, $msoFalse, $msoFalse, $msoTrue)

    if ($prs.Slides.Count -ne 9) {
        throw "Expected 9 slides before insertion, found $($prs.Slides.Count)."
    }

    Add-ProgressSlide8 $prs
    Add-ProgressSlide9 $prs
    Add-ProgressSlide10 $prs
    Add-ProgressSlide11 $prs
    Update-AllSlideNumbers $prs

    if ($prs.Slides.Count -ne 13) {
        throw "Expected 13 slides after insertion, found $($prs.Slides.Count)."
    }

    $prs.SaveAs($outputPath, $ppSaveAsOpenXMLPresentation)
    $prs.Close()
    $prs = $null

    $reopened = $ppt.Presentations.Open($outputPath, $msoFalse, $msoFalse, $msoTrue)
    if ($reopened.Slides.Count -ne 13) {
        throw "Reopen validation failed: expected 13 slides, found $($reopened.Slides.Count)."
    }

    for ($i = 8; $i -le 11; $i++) {
        $pngPath = Join-Path $previewDir ("slide{0:00}.png" -f $i)
        $reopened.Slides.Item($i).Export($pngPath, "PNG", 1920, 1080)
    }
    $reopened.Save()
}
finally {
    if ($reopened) {
        $reopened.Close()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($reopened) | Out-Null
    }
    if ($prs) {
        $prs.Close()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($prs) | Out-Null
    }
    if ($ppt) {
        $ppt.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$sourceHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash
if ($sourceHashBefore -ne $sourceHashAfter) {
    throw "Source pptx hash changed. Before=$sourceHashBefore After=$sourceHashAfter"
}

Write-Output "Created: $outputPath"
Write-Output "Preview PNGs: $previewDir"
Write-Output "Source SHA256 unchanged: $sourceHashAfter"
