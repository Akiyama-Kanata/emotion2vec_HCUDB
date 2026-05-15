#!/bin/bash
# emotion2vec モデルを使って全データの特徴量を一括抽出する（前処理Step2）

export CUDA_LAUNCH_BLOCKING=1  # CUDA エラーを同期的に検出するためのデバッグ設定
export HYDRA_FULL_ERROR=1      # Hydra の詳細エラーを表示する
export CUDA_VISIBLE_DEVICES=0  # 使用するGPUを0番に限定する

export PYTHONPATH=$1:$PYTHONPATH  # emotion2vec upstream モジュールを参照できるようPYTHONPATHに追加

manifest_path=$2      # TSVマニフェストファイルのディレクトリ
model_path=$3         # emotion2vec upstream モデルのディレクトリ
checkpoint_path=$4    # 事前学習済みチェックポイント (.pt) のパス
save_dir=$5           # 抽出した特徴量 (.npy) の保存先ディレクトリ

# Here we only extract the last layer
# 最終層（layer 11、1-indexed で 12層目）のみを特徴量として抽出する
for layer in {11..11}; do
    true_layer=$[$layer+1]
    echo "Extracting features from layer $true_layer"

    python scripts/emotion2vec_speech_features.py  \
        --data $manifest_path \
        --model $model_path \
        --split=train \
        --checkpoint=$checkpoint_path \
        --save-dir=$save_dir \
        --layer=$layer
done
