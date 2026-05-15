"""
単一の wav ファイルから emotion2vec 特徴量を抽出して .npy ファイルに保存するスクリプト。
フレーム粒度（frame）または発話粒度（utterance）の特徴量を選択して出力できる。
"""

import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq

def get_parser():
    """コマンドライン引数のパーサーを返す。"""
    parser = argparse.ArgumentParser(
        description="extract emotion2vec features for downstream tasks"
    )
    parser.add_argument('--source_file', help='location of source wav files', required=True)
    parser.add_argument('--target_file', help='location of target npy files', required=True)
    parser.add_argument('--model_dir', type=str, help='pretrained model', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='checkpoint for pre-trained model', required=True)
    parser.add_argument('--granularity', type=str, help='which granularity to use, frame or utterance', required=True)

    return parser

@dataclass
class UserDirModule:
    """fairseq がユーザー定義モジュールを読み込む際に必要なディレクトリ情報を保持するデータクラス。"""
    user_dir: str

def main():
    """特徴量抽出のメインエントリポイント。wav を読み込み .npy として保存する。"""
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    source_file = args.source_file
    target_file = args.target_file
    model_dir = args.model_dir
    checkpoint_dir = args.checkpoint_dir
    granularity = args.granularity

    # fairseq に upstream/ ディレクトリをユーザー定義モジュールとして登録し、モデルをロードする
    model_path = UserDirModule(model_dir)
    fairseq.utils.import_user_module(model_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_dir])
    model = model[0]
    model.eval()
    model.cuda()

    # wav ファイルを読み込み、16kHz・モノラルであることを確認する
    if source_file.endswith('.wav'):
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)

    with torch.no_grad():
        source = torch.from_numpy(wav).float().cuda()
        if task.cfg.normalize:
            source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)  # (1, T) に整形してモデルへ入力
        try:
            feats = model.extract_features(source, padding_mask=None)
            feats = feats['x'].squeeze(0).cpu().numpy()  # (T, D) に変換
            if granularity == 'frame':
                feats = feats  # フレーム粒度: 各フレームの特徴量をそのまま使用
            elif granularity == 'utterance':
                feats = np.mean(feats, axis=0)  # 発話粒度: フレーム方向に平均して (D,) に圧縮
            else:
                raise ValueError("Unknown granularity: {}".format(args.granularity))
            np.save(target_file, feats)
        except:
            Exception("Error in extracting features from {}".format(source_file))


if __name__ == '__main__':
    main()