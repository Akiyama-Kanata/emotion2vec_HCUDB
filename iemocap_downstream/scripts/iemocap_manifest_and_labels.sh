#!/bin/bash
# IEMOCAPの感情ラベルを抽出し、wavパス+フレーム数のTSVマニフェストを生成する（前処理Step1）

IEMOCAP_ROOT=$1  # IEMOCAPデータセットのルートディレクトリ（第1引数）
output_path=$2   # 出力ディレクトリ（第2引数）

mkdir -p $output_path

# セッション1〜5それぞれの感情評価ファイルから対象クラスのラベルを抽出する
for index in {1..5}; do
    cat $IEMOCAP_ROOT/Session$index/dialog/EmoEvaluation/*.txt | \
        grep Ses | cut -f2,3 | \
        # ang/exc/hap/neu/sad の5クラスのみ残し、exc（興奮）をhap（幸福）に統合する
        awk '{if ($2 == "ang" || $2 == "exc" || $2 == "hap" || $2 == "neu" || $2 == "sad") print $0}' | \
        sed 's/\bexc\b/hap/g' > $output_path/Session${index}.emo
done

# 各セッションの一時ラベルファイルを train.emo に結合し、一時ファイルを削除する
for index in {1..5}; do
    cat $output_path/Session${index}.emo >> $output_path/train.emo
    rm -rf $output_path/Session${index}.emo
done

# 統合ラベルをもとに wavパス・フレーム数のマニフェストTSVを生成する
python scripts/iemocap_manifest.py \
    --root $IEMOCAP_ROOT --dest $output_path \
    --label_path $output_path/train.emo
