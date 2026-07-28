#!/bin/bash
# 単一の wav ファイルから emotion2vec 特徴量を抽出して .npy に保存するラッパースクリプト。
# extract_features.py の引数をあらかじめ設定した呼び出し例として機能する。

export CUDA_VISIBLE_DEVICES=0  # 使用するGPUを0番に限定する

python extract_features.py  \
--source_file='/mnt/lustre/sjtu/home/zym22/code/emotion2vec/scripts/test.wav' \
--target_file='/mnt/lustre/sjtu/home/zym22/code/emotion2vec/scripts/test.npy' \
--model_dir='/mnt/lustre/sjtu/home/zym22/code/emotion2vec/upstream' \
--checkpoint_dir='/mnt/lustre/sjtu/home/zym22/models/released/emotion2vec/emotion2vec_base.pt' \
--granularity='utterance' \
