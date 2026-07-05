$ErrorActionPreference = "Stop"

$outputPath = Join-Path (Resolve-Path ".") "23RD004_秋山叶太_研究計画_改訂版.pptx"
if (Test-Path -LiteralPath $outputPath) {
    throw "Output already exists: $outputPath"
}

$ppLayoutBlank = 12
$ppSaveAsOpenXMLPresentation = 24
$msoTextOrientationHorizontal = 1
$msoTrue = -1
$msoFalse = 0

$shapeRectangle = 1
$shapeRoundedRectangle = 5
$shapeOval = 9
$shapeRightArrow = 33
$shapeChevron = 52

function Rgb([int]$r, [int]$g, [int]$b) {
    return $r + ($g * 256) + ($b * 65536)
}

$Color = @{
    Ink = Rgb 30 41 59
    Muted = Rgb 71 85 105
    Navy = Rgb 21 47 80
    Blue = Rgb 37 99 235
    Teal = Rgb 14 116 144
    Green = Rgb 22 101 52
    Amber = Rgb 180 83 9
    Red = Rgb 185 28 28
    Gray = Rgb 226 232 240
    PaleBlue = Rgb 239 246 255
    PaleTeal = Rgb 236 253 245
    PaleAmber = Rgb 255 251 235
    PaleRed = Rgb 254 242 242
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
    $shape.TextFrame.MarginLeft = 8
    $shape.TextFrame.MarginRight = 8
    $shape.TextFrame.MarginTop = 5
    $shape.TextFrame.MarginBottom = 5
}

function Add-Text($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [double]$size = 18, [int]$color = $Color.Ink, [bool]$bold = $false, [int]$align = 1) {
    $shape = $slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $x, $y, $w, $h)
    $shape.TextFrame.TextRange.Text = $text
    Set-TextStyle $shape $size $color $bold $align
    return $shape
}

function Add-Box($slide, [string]$text, [double]$x, [double]$y, [double]$w, [double]$h, [int]$fill, [int]$line, [double]$size = 16, [int]$textColor = $Color.Ink, [bool]$bold = $false, [int]$shapeType = $shapeRoundedRectangle) {
    $shape = $slide.Shapes.AddShape($shapeType, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.ForeColor.RGB = $line
    $shape.Line.Weight = 1.2
    $shape.TextFrame.TextRange.Text = $text
    Set-TextStyle $shape $size $textColor $bold 2
    return $shape
}

function Add-BulletBox($slide, [string]$title, [string[]]$bullets, [double]$x, [double]$y, [double]$w, [double]$h, [int]$accent = $Color.Blue, [int]$fill = $Color.OffWhite) {
    $shape = $slide.Shapes.AddShape($shapeRoundedRectangle, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.ForeColor.RGB = $Color.Gray
    $shape.Line.Weight = 1
    $text = $title + "`r" + (($bullets | ForEach-Object { "・" + $_ }) -join "`r")
    $shape.TextFrame.TextRange.Text = $text
    Set-TextStyle $shape 14.5 $Color.Ink $false 1
    $titleRange = $shape.TextFrame.TextRange.Characters(1, $title.Length)
    $titleRange.Font.Bold = $msoTrue
    $titleRange.Font.Size = 16
    $titleRange.Font.Color.RGB = $accent
    return $shape
}

function Add-Header($slide, [int]$number, [string]$title, [string]$message) {
    Add-Text $slide $title 40 24 720 34 24 $Color.Navy $true 1 | Out-Null
    Add-Text $slide "$number / 9" 842 28 78 24 11 $Color.Muted $true 3 | Out-Null
    $line = $slide.Shapes.AddShape($shapeRectangle, 40, 66, 880, 2)
    $line.Fill.ForeColor.RGB = $Color.Gray
    $line.Line.Visible = $msoFalse
    Add-Text $slide $message 42 78 876 34 15 $Color.Ink $true 1 | Out-Null
}

function Add-Footer($slide) {
    Add-Text $slide "23RD004 秋山叶太  |  emotion2vec研究計画  |  2026年7月" 40 514 540 14 8.5 $Color.Muted $false 1 | Out-Null
}

function Add-Arrow($slide, [double]$x, [double]$y, [double]$w = 36, [double]$h = 18, [int]$color = $Color.Muted) {
    $shape = $slide.Shapes.AddShape($shapeRightArrow, $x, $y, $w, $h)
    $shape.Fill.ForeColor.RGB = $color
    $shape.Line.Visible = $msoFalse
    return $shape
}

function Add-FlowStep($slide, [string]$label, [string]$detail, [double]$x, [double]$y, [double]$w, [int]$fill, [int]$line) {
    $text = $label + "`r" + $detail
    $shape = Add-Box $slide $text $x $y $w 72 $fill $line 12.3 $Color.Ink $false
    $range = $shape.TextFrame.TextRange
    $range.Characters(1, $label.Length).Font.Bold = $msoTrue
    $range.Characters(1, $label.Length).Font.Size = 14.5
    $range.Characters(1, $label.Length).Font.Color.RGB = $line
    return $shape
}

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = $msoTrue
$prs = $ppt.Presentations.Add()
$prs.PageSetup.SlideWidth = 960
$prs.PageSetup.SlideHeight = 540

try {
    # Slide 1
    $slide = $prs.Slides.Add(1, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.OffWhite
    Add-Text $slide "emotion2vecの日本語適応と`rVAD媒介型SER評価計画" 54 70 760 95 34 $Color.Navy $true 1 | Out-Null
    Add-Text $slide "日本語fine-tuningで日本語SERを改善しつつ、英語SERを大きく落とさないかを評価する。" 58 180 840 30 17 $Color.Ink $true 1 | Out-Null
    Add-Box $slide "23RD004`r秋山 叶太" 58 236 178 72 $Color.White $Color.Gray 15 $Color.Ink $true | Out-Null
    Add-Box $slide "主データ`rHCUDB1 4,620発話" 266 236 198 72 $Color.PaleBlue $Color.Blue 14 $Color.Ink $true | Out-Null
    Add-Box $slide "英語評価`rIEMOCAP" 494 236 168 72 $Color.PaleTeal $Color.Teal 14 $Color.Ink $true | Out-Null
    Add-Box $slide "評価経路`rfeatures -> VAD -> emotion" 692 236 206 72 $Color.PaleAmber $Color.Amber 13.5 $Color.Ink $true | Out-Null
    Add-Text $slide "研究計画の焦点: 性能結果の断定ではなく、比較実験と評価設計を明確化する。" 58 356 780 24 15 $Color.Muted $false 1 | Out-Null
    Add-Footer $slide

    # Slide 2
    $slide = $prs.Slides.Add(2, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 2 "背景：emotion2vecの汎用性と日本語適応" "汎用的な音声感情表現を日本語音声に活かすには、適応と汎用性維持を同時に確認する必要がある。"
    Add-BulletBox $slide "背景" @(
        "emotion2vecは自己教師あり事前学習により音声感情表現を抽出する",
        "英語などで有効性が示されているが、日本語音声への適応度は未確認",
        "日本語の韻律・発話スタイル・収録条件に合わせたfine-tuningが候補",
        "日本語へ適応する過程で英語SERが低下する可能性を評価する"
    ) 56 128 380 260 $Color.Blue $Color.PaleBlue | Out-Null
    Add-Box $slide "日本語SER`r改善したい" 514 150 160 78 $Color.PaleBlue $Color.Blue 17 $Color.Ink $true | Out-Null
    Add-Box $slide "英語SER`r維持したい" 728 150 160 78 $Color.PaleTeal $Color.Teal 17 $Color.Ink $true | Out-Null
    Add-Arrow $slide 680 180 42 18 $Color.Muted | Out-Null
    Add-Text $slide "研究計画では、この2軸を分けて測る。" 524 256 350 24 18 $Color.Navy $true 2 | Out-Null
    Add-Box $slide "日本語だけ上がる" 526 314 154 50 $Color.White $Color.Gray 13 $Color.Muted $false | Out-Null
    Add-Box $slide "英語だけ維持" 704 314 154 50 $Color.White $Color.Gray 13 $Color.Muted $false | Out-Null
    Add-Box $slide "両方を満たすかを検証" 584 386 220 58 $Color.PaleAmber $Color.Amber 15 $Color.Ink $true | Out-Null
    Add-Footer $slide

    # Slide 3
    $slide = $prs.Slides.Add(3, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 3 "研究目的：日本語SER改善 + 英語SER維持" "日本語fine-tuningの効果を、日本語の改善量と英語性能の低下量の両面から評価する。"
    Add-BulletBox $slide "目的" @(
        "HCUDBを中心に、日本語SERのfine-tuning前後差を測定する",
        "IEMOCAPを用いて、英語SERが大きく低下していないか確認する",
        "同一指標でpretrainedとJapanese fine-tunedを比較する",
        "改善・維持の主張は実験値が出てから行う"
    ) 54 132 388 270 $Color.Green $Color.PaleTeal | Out-Null
    Add-Box $slide "目的1`r日本語SER改善" 512 142 310 78 $Color.PaleBlue $Color.Blue 20 $Color.Ink $true | Out-Null
    Add-Box $slide "目的2`r英語SER維持" 512 252 310 78 $Color.PaleTeal $Color.Teal 20 $Color.Ink $true | Out-Null
    Add-Box $slide "判定`r日本語指標が改善し、英語指標が大きく悪化しない" 512 366 310 78 $Color.PaleAmber $Color.Amber 15 $Color.Ink $true | Out-Null
    Add-Footer $slide

    # Slide 4
    $slide = $prs.Slides.Add(4, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 4 "研究課題：fine-tuning前後の二言語比較" "pretrainedと日本語fine-tunedを同じ評価指標で測り、適応と汎用性のトレードオフを明らかにする。"
    Add-Box $slide "pretrained`remotion2vec" 76 156 210 70 $Color.PaleBlue $Color.Blue 17 $Color.Ink $true | Out-Null
    Add-Arrow $slide 306 182 48 18 $Color.Muted | Out-Null
    Add-Box $slide "Japanese SER`rbaseline" 376 132 190 60 $Color.White $Color.Gray 14 $Color.Ink $false | Out-Null
    Add-Box $slide "English SER`rbaseline" 376 214 190 60 $Color.White $Color.Gray 14 $Color.Ink $false | Out-Null
    Add-Box $slide "Japanese fine-tuned`remotion2vec" 76 342 210 70 $Color.PaleTeal $Color.Teal 16 $Color.Ink $true | Out-Null
    Add-Arrow $slide 306 368 48 18 $Color.Muted | Out-Null
    Add-Box $slide "Japanese SER`rimproved?" 376 318 190 60 $Color.PaleBlue $Color.Blue 14 $Color.Ink $true | Out-Null
    Add-Box $slide "English SER`rmaintained?" 376 400 190 60 $Color.PaleTeal $Color.Teal 14 $Color.Ink $true | Out-Null
    Add-BulletBox $slide "研究課題" @(
        "RQ1: 日本語fine-tuningで日本語SERは改善するか",
        "RQ2: 日本語適応後も英語SERを大きく落とさないか",
        "RQ3: VAD媒介型経路で分類結果を説明しやすくできるか"
    ) 632 156 260 250 $Color.Amber $Color.PaleAmber | Out-Null
    Add-Footer $slide

    # Slide 5
    $slide = $prs.Slides.Add(5, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 5 "提案手法：VAD媒介型SER評価経路" "emotion2vec特徴を直接分類する経路に加え、予測VADを経由することで分類根拠を説明しやすくする。"
    Add-FlowStep $slide "features" "emotion2vec`r768次元特徴" 44 148 150 $Color.PaleBlue $Color.Blue | Out-Null
    Add-Arrow $slide 202 174 34 18 $Color.Muted | Out-Null
    Add-FlowStep $slide "predicted VAD" "VAを主評価`rDは拡張枠" 244 148 160 $Color.PaleTeal $Color.Teal | Out-Null
    Add-Arrow $slide 412 174 34 18 $Color.Muted | Out-Null
    Add-FlowStep $slide "emotion" "hap / sad / ang / dis`r分類指標で評価" 454 148 178 $Color.PaleAmber $Color.Amber | Out-Null
    Add-Text $slide "emotion2vec features -> predicted VAD -> emotion" 92 246 488 24 18 $Color.Navy $true 2 | Out-Null
    Add-BulletBox $slide "設計上の位置づけ" @(
        "分類器の入力を予測VAD/VAに限定し、768次元特徴を直接見せない",
        "VAD/VA側はCCC、分類側はCrossEntropyで学習・評価する",
        "Linear層の重みからValence/Arousal(/Dominance)のlogit寄与を確認する",
        "現時点の性能成果ではなく、説明しやすくする評価経路として扱う"
    ) 666 132 236 276 $Color.Teal $Color.OffWhite | Out-Null
    Add-Box $slide "注意`r直接分類との優劣は未検証。性能差は実データ評価後に扱う。" 152 340 400 58 $Color.PaleRed $Color.Red 15 $Color.Ink $true | Out-Null
    Add-Footer $slide

    # Slide 6
    $slide = $prs.Slides.Add(6, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 6 "データセット：HCUDB中心、IEMOCAPで英語評価" "日本語はHCUDB1のVA付き4,620発話を主軸に、英語維持はIEMOCAPで評価する。"
    Add-BulletBox $slide "HCUDB1（日本語・主データ）" @(
        "14話者 × 10セリフ × 11感情 × 3テイク = 4,620発話",
        "Valence / Arousal評価を主に使用する",
        "日本語fine-tuningと日本語SER評価の中心に置く"
    ) 54 132 380 216 $Color.Blue $Color.PaleBlue | Out-Null
    Add-BulletBox $slide "IEMOCAP（英語評価）" @(
        "英語SERのbaselineと維持評価に用いる",
        "fine-tuning前後を同じ感情分類指標で比較する",
        "英語性能維持の確認用データとして整理する"
    ) 526 132 380 216 $Color.Teal $Color.PaleTeal | Out-Null
    Add-Box $slide "Dominanceの扱い`rHCUDB中心の日本語実験ではVAを主評価とし、Dominanceは追加ラベルまたは別データでの拡張枠にする。" 96 382 770 64 $Color.PaleAmber $Color.Amber 14.5 $Color.Ink $true | Out-Null
    Add-Footer $slide

    # Slide 7
    $slide = $prs.Slides.Add(7, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 7 "評価設計：SER指標とVAD/VA指標を分けて測る" "分類性能はWA/UA/weighted F1/混同行列、連続値はCCCで検証する。"
    Add-BulletBox $slide "SER（感情分類）" @(
        "WA: 全体正解率",
        "UA: クラスごとのrecall平均",
        "weighted F1: support重み付きF1",
        "confusion matrix: 誤分類パターン確認"
    ) 56 132 250 278 $Color.Blue $Color.PaleBlue | Out-Null
    Add-BulletBox $slide "VAD/VA（連続値）" @(
        "CCCをValence / Arousalごとに算出",
        "Dominanceはラベルがある場合に追加",
        "平均CCCも補助的に確認",
        "中間表現がVA(D)として解釈可能かを見る"
    ) 354 132 250 278 $Color.Teal $Color.PaleTeal | Out-Null
    Add-BulletBox $slide "比較単位" @(
        "pretrained vs Japanese fine-tuned",
        "Japanese SER vs English SER",
        "直接分類 vs VAD媒介型は別途ベースライン化",
        "クラス分布とfold偏りを先に確認"
    ) 652 132 250 278 $Color.Amber $Color.PaleAmber | Out-Null
    Add-Text $slide "評価値が出るまでは、日本語改善・英語維持・VAD媒介型優位を結果として断定しない。" 74 442 812 22 14.5 $Color.Red $true 2 | Out-Null
    Add-Footer $slide

    # Slide 8
    $slide = $prs.Slides.Add(8, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 8 "現在地と次の実験" "現在の成果はVAD媒介型の実装基盤であり、性能主張は実データ評価後に行う。"
    Add-BulletBox $slide "実装済み" @(
        "VAD/VAデータ読み込み、ID整合性検証",
        "VAD回帰head、VAD媒介型分類head",
        "CCC + CrossEntropyの学習経路",
        "学習CLI、推論JSON、logit寄与分解"
    ) 58 132 250 260 $Color.Green $Color.PaleTeal | Out-Null
    Add-BulletBox $slide "未実施" @(
        "emotion2vec本体の日本語fine-tuning",
        "HCUDB実データでのSER評価",
        "IEMOCAPでの英語性能維持評価",
        "実checkpointを使った推論とCCC算出"
    ) 354 132 250 260 $Color.Red $Color.PaleRed | Out-Null
    Add-BulletBox $slide "次の実験" @(
        "依存入り環境でテストを完走する",
        "hap/sad/ang/disとVAラベル分布を確認する",
        "HCUDBでVAD媒介型を学習・評価する",
        "fine-tuning前後の日本語・英語SERを比較する"
    ) 650 132 250 260 $Color.Blue $Color.PaleBlue | Out-Null
    Add-Box $slide "発表での線引き`r「評価経路の実装」と「fine-tuning性能の実証」を混同しない。" 156 424 648 50 $Color.PaleAmber $Color.Amber 15 $Color.Ink $true | Out-Null
    Add-Footer $slide

    # Slide 9
    $slide = $prs.Slides.Add(9, $ppLayoutBlank)
    $slide.Background.Fill.ForeColor.RGB = $Color.White
    Add-Header $slide 9 "スケジュール・まとめ" "2026年7月以降に実データ評価、fine-tuning比較、卒論執筆へ進める。"
    $months = @(
        @("2026年7月", "環境テスト・分布確認`rVAD/VA評価開始", $Color.Blue, $Color.PaleBlue),
        @("2026年8月", "HCUDB fine-tuning条件`r日本語SER比較", $Color.Teal, $Color.PaleTeal),
        @("2026年9月", "IEMOCAP英語維持評価`r直接分類baseline", $Color.Amber, $Color.PaleAmber),
        @("2026年10月", "結果整理`r手法・実験章", $Color.Green, $Color.PaleTeal),
        @("2026年11月", "考察・卒論ドラフト`r指導反映", $Color.Blue, $Color.PaleBlue),
        @("12月-1月", "修正・推敲`r最終提出", $Color.Amber, $Color.PaleAmber)
    )
    $x = 42
    foreach ($m in $months) {
        Add-Box $slide ($m[0] + "`r" + $m[1]) $x 136 136 86 $m[3] $m[2] 11.8 $Color.Ink $true $shapeChevron | Out-Null
        $x += 148
    }
    Add-BulletBox $slide "まとめ" @(
        "中心主張は、日本語fine-tuningで日本語SERを改善しつつ英語SERを大きく落とさないかを評価すること",
        "HCUDBを主データ、IEMOCAPを英語維持評価として整理する",
        "VAD媒介型分類は性能結果ではなく、分類結果を説明しやすくする評価経路として位置づける",
        "未検証の性能主張は入れず、次の実験で数値化する"
    ) 86 270 792 158 $Color.Navy $Color.OffWhite | Out-Null
    Add-Footer $slide

    try {
        $prs.BuiltInDocumentProperties("Title").Value = "emotion2vecの日本語適応とVAD媒介型SER評価計画"
        $prs.BuiltInDocumentProperties("Subject").Value = "23RD004研究計画PPT改訂版"
        $prs.BuiltInDocumentProperties("Author").Value = "23RD004 秋山叶太"
        $prs.BuiltInDocumentProperties("Comments").Value = "2026年7月改訂。未検証の性能主張を避け、HCUDB中心の日本語SER改善とIEMOCAPによる英語SER維持評価を研究計画として整理。"
    }
    catch {
        Write-Warning "Skipped document property metadata: $($_.Exception.Message)"
    }
    $prs.SaveAs($outputPath, $ppSaveAsOpenXMLPresentation)
}
finally {
    if ($prs) { $prs.Close() }
    if ($ppt) { $ppt.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($prs) | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output "Created: $outputPath"
