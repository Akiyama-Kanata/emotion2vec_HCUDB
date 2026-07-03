#!/bin/bash

IEMOCAP_ROOT=$1
output_path=$2
emotion_set=${3:-standard4}

case "$emotion_set" in
    standard4)
        # Original IEMOCAP setting: ang/hap/neu/sad with exc folded into hap.
        label_filter='$2 == "ang" || $2 == "exc" || $2 == "hap" || $2 == "neu" || $2 == "sad"'
        ;;
    vad4)
        # VAD-mediated classifier setting: use real dis annotations, not neu -> dis.
        label_filter='$2 == "ang" || $2 == "exc" || $2 == "hap" || $2 == "sad" || $2 == "dis"'
        ;;
    *)
        echo "unknown emotion_set: $emotion_set" >&2
        echo "usage: $0 IEMOCAP_ROOT output_path [standard4|vad4]" >&2
        exit 1
        ;;
esac

mkdir -p "$output_path"

for index in {1..5}; do
    cat "$IEMOCAP_ROOT"/Session$index/dialog/EmoEvaluation/*.txt | \
        grep Ses | cut -f2,3 | \
        awk "{if ($label_filter) print \$0}" | \
        sed 's/\bexc\b/hap/g' > "$output_path"/Session${index}.emo
done

: > "$output_path"/train.emo
for index in {1..5}; do
    cat "$output_path"/Session${index}.emo >> "$output_path"/train.emo
    rm -f "$output_path"/Session${index}.emo
done

python scripts/iemocap_manifest.py \
    --root "$IEMOCAP_ROOT" --dest "$output_path" \
    --label_path "$output_path"/train.emo
