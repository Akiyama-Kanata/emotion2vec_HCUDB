#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
IEMOCAPのwavファイルパスとフレーム数をTSV形式でリスト化するマニフェスト生成スクリプト。
ラベルファイル（.emo）を参照し、各発話の音声ファイルパスとフレーム数を出力する。
iemocap_manifest_and_labels.sh から呼び出される（前処理Step1の後半）。
"""

import argparse
import os

import soundfile


def get_parser():
    """コマンドライン引数のパーサーを返す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", metavar="DIR",
        default='/path/to/IEMOCAP_full_release',
        help="root directory containing audio files to index"
    )
    parser.add_argument(
        "--dest", default="/path/to/manifest", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--label_path", default="/path/to/train.emo",
    )
    return parser


def main(args):
    """ラベルファイルを読み込み、各発話の相対パスとフレーム数をTSV形式で書き出す。"""
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # Session{N} 形式のパステンプレートを作成する
    root = os.path.join(args.root, 'Session{}')
    with open(args.label_path) as rf, open(args.dest + '/train.tsv', 'w') as wf:
        # 1行目にルートディレクトリを記録する（fairseq マニフェスト形式）
        print(args.root, file=wf)
        for line in rf.readlines():
            fname = line.split('\t')[0]   # 発話ファイル名（拡張子なし）
            session = fname[4]            # ファイル名の5文字目がセッション番号
            folder = fname.rsplit('_', 1)[0]  # 会話フォルダ名（例: Ses01F_impro01）

            fname = os.path.join(root.format(session), 'sentences/wav', folder, fname + '.wav')
            frames = soundfile.info(fname).frames  # 音声の総フレーム数を取得
            suffix = fname.replace(args.root, '', 1).lstrip('/')  # ルートからの相対パス
            print(suffix, frames, sep='\t', file=wf)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
