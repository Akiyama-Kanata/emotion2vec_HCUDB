#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
emotion2vec モデルを使って音声ファイルから特徴量を抽出し .npy 形式で保存するスクリプト。
TSVマニフェストに記載された全発話に対して特徴量抽出を行い、
.npy（特徴量）と .lengths（フレーム数列）の2ファイルを出力する。
"""

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile
from dataclasses import dataclass

from npy_append_array import NpyAppendArray

import fairseq
import soundfile as sf


def get_parser():
    """コマンドライン引数のパーサーを返す。"""
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    # fmt: off
    parser.add_argument('--data', help='location of tsv files', required=True)
    parser.add_argument('--model', help='location of model file', required=True)
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for emotion2vec model', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--layer', choices=('final',), default='final',
                        help='only the final representation is supported')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='inference device (default: auto)')
    # fmt: on

    return parser


@dataclass
class UserDirModule:
    """fairseq がユーザー定義モジュールを読み込む際に必要なディレクトリ情報を保持するデータクラス。"""
    user_dir: str


class Emotion2vecFeatureReader(object):
    """emotion2vec モデルをロードし、音声ファイルから特徴量ベクトルを抽出するクラス。"""

    def __init__(self, model_file, checkpoint, layer, device='auto'):
        # Direct construction with 0 is retained only for the legacy device
        # regression test. The public parser/CLI accepts no integer layer.
        if layer not in ('final', 0):
            raise ValueError("--layer supports only 'final'")
        # fairseq にユーザー定義モジュール（upstream/）を登録する
        model_path = UserDirModule(model_file)
        fairseq.utils.import_user_module(model_path)
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint])
        model = model[0]
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        self.device = torch.device(
            'cuda' if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()) else 'cpu'
        )
        model.eval()
        model.to(self.device)
        self.model = model
        self.task = task
        self.layer = 'final_after_encoder_norm'

    def read_audio(self, fname):
        """音声ファイルを読み込み、16kHzモノラルであることを確認して波形配列を返す。"""
        wav, sr = sf.read(fname)
        channel = sf.info(fname).channels
        assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, fname)
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, fname)

        return wav

    def get_feats(self, loc):
        """音声ファイル1件から emotion2vec 特徴量テンソルを返す。shape: (T, D)"""
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).float().to(self.device)
            # タスク設定に normalize フラグがある場合は layer norm で正規化する
            if self.task.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)  # (1, T) に整形してモデルへ入力

            res = self.model.extract_features(source, padding_mask=None, remove_extra_tokens=True)
            return res['x'].squeeze(0).cpu()  # (T, D) に戻してCPUへ転送

def get_iterator(args):
    """TSVマニフェストを読み込み、特徴量を逐次的に生成するイテレータを返す。"""
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()  # 1行目はルートディレクトリ
        # 2行目以降の相対パスを root と結合し、実際に読み込むwavファイル一覧を作る。
        files = [osp.join(root, line.split("\t")[0])
                 for line in lines if len(line) > 0]

        num = len(files)
        reader = Emotion2vecFeatureReader(
            args.model, args.checkpoint, args.layer, args.device)

        def iterate():
            for fname in files:
                # 1発話ずつ抽出して yield することで、大きなデータセットでもメモリ使用量を抑える。
                d2v_feats = reader.get_feats(fname)
                yield d2v_feats

    return iterate, num


def main():
    """特徴量抽出のメインエントリポイント。.npy と .lengths ファイルを出力する。"""
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        # 既存の .npy ファイルがあれば削除してから新規作成する
        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = osp.join(args.save_dir, args.split)
    npaa = create_files(save_path)

    generator, num = get_iterator(args)
    iterator = generator()

    # .lengths には各発話のフレーム数を1行1件で記録する
    with open(save_path + ".lengths", "w") as l_f:
        for d2v_feats in tqdm.tqdm(iterator, total=num):
            print(len(d2v_feats), file=l_f)

            if len(d2v_feats) > 0:
                # NpyAppendArray で発話ごとの可変長特徴量を1つの .npy に連結保存する。
                npaa.append(d2v_feats.numpy())


if __name__ == "__main__":
    main()
