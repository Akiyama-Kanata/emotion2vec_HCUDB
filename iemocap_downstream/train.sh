#!/bin/bash
# 感情分類器（BaseModel）の5-fold交差検証学習を起動するエントリポイント。
# 第1引数に特徴量ディレクトリのパスを渡して実行する。

export CUDA_VISIBLE_DEVICES=0  # 使用するGPUを0番に限定する

dataset=IEMOCAP
feat_path=$1  # 特徴量 (.npy/.lengths/.emo) が格納されたディレクトリ

python main.py \
    dataset._name=$dataset \
    dataset.feat_path=$feat_path \
    model._name=BaseModel \
    dataset.batch_size=128 \
    optimization.epoch=100 \
    optimization.lr=5e-4 \
    dataset.eval_is_test=false \
