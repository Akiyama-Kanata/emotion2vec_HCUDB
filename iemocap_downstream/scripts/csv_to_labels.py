"""
IEMOCAP正解ラベル.csv から .emo ファイルと va_labels.txt を生成するスクリプト。

出力:
    <output_dir>/train.emo      : 発話ID + カテゴリラベル（タブ区切り）
    <output_dir>/va_labels.txt  : 発話ID + Valence + Arousal（スペース区切り）

使い方:
    python csv_to_labels.py \
        --csv C:/Users/RD004/Documents/lab/data/iemocap/IEMOCAP正解ラベル.csv \
        --output_dir C:/Users/RD004/Documents/lab/data/iemocap
"""

import argparse
import csv
from pathlib import Path

TARGET_EMOTIONS = {"ang", "hap", "neu", "sad", "exc"}
EMO_MAP = {"exc": "hap"}  # exc（興奮）をhap（幸福）に統合


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emo_path = output_dir / "train.emo"
    va_path  = output_dir / "va_labels.txt"

    skipped = 0
    written = 0

    with open(args.csv, encoding="utf-8-sig") as f, \
         open(emo_path, "w") as emo_f, \
         open(va_path,  "w") as va_f:

        reader = csv.DictReader(f)
        for row in reader:
            emo = row["emo"].strip()
            if emo not in TARGET_EMOTIONS:
                skipped += 1
                continue

            emo = EMO_MAP.get(emo, emo)
            utt_id  = row["ID"].strip()
            valence = row["Valence"].strip()
            arousal = row["Arousal"].strip()

            emo_f.write(f"{utt_id}\t{emo}\n")
            va_f.write(f"{utt_id} {valence} {arousal}\n")
            written += 1

    print(f"書き出し完了: {written} 件  スキップ: {skipped} 件")
    print(f"  {emo_path}")
    print(f"  {va_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="IEMOCAP正解ラベル.csvのパス")
    parser.add_argument("--output_dir", required=True, help="出力先ディレクトリ")
    args = parser.parse_args()
    main(args)
