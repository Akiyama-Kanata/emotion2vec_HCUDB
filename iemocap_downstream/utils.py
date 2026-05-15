"""
IEMOCAP感情認識タスクの1エポック学習と評価（WA/UA/F1）を実装するユーティリティモジュール。
WA（重み付き正解率）、UA（クラス平均正解率）、重み付きF1スコアの3指標を計算する。
"""

import torch


def train_one_epoch(model, optimizer, criterion, train_loader, device):
    """1エポック分の学習を実行し、累積損失を返す。"""
    model.train()
    train_loss = 0
    for batch in train_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)
        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats, speech_padding_mask)

        loss = criterion(outputs, labels.long())
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    return train_loss


@torch.no_grad()
def validate_and_test(model, data_loader, device, num_classes):
    """
    評価セットでモデルを推論し、WA・UA・重み付きF1スコア（%）を返す。

    Returns:
        weighted_acc: WA（全サンプルに対する正解率）
        unweighted_acc: UA（クラスごとの正解率を単純平均）
        weighted_f1: 重み付きF1（クラスサイズで重み付けしたF1の平均）
    """
    model.eval()
    correct, total = 0, 0

    # unweighted accuracy
    unweightet_correct = [0] * num_classes  # クラスごとの正解数
    unweightet_total = [0] * num_classes    # クラスごとのサンプル数

    # weighted f1
    tp = [0] * num_classes  # True Positive（正しく予測した数）
    fp = [0] * num_classes  # False Positive（誤って予測した数）
    fn = [0] * num_classes  # False Negative（見逃した数）

    for batch in data_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)
        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)

        outputs = model(feats, speech_padding_mask)

        _, predicted = torch.max(outputs.data, 1)  # ロジット最大のクラスを予測クラスとする

        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
        # サンプルごとにTP/FP/FNを集計する
        for i in range(len(labels)):
            unweightet_total[labels[i]] += 1
            if predicted[i] == labels[i]:
                unweightet_correct[labels[i]] += 1
                tp[labels[i]] += 1
            else:
                fp[predicted[i]] += 1
                fn[labels[i]] += 1

    weighted_acc = correct / total * 100
    unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total) * 100
    weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total) * 100

    return weighted_acc, unweighted_acc, weighted_f1


def inference(model, ):
    pass


def compute_unweighted_accuracy(list1, list2):
    """クラスごとの正解率を計算して単純平均（UA）を返す。"""
    result = []
    for i in range(len(list1)):
        result.append(list1[i] / list2[i])
    return sum(result)/len(result)


def compute_weighted_f1(tp, fp, fn, unweightet_total):
    """
    クラスごとのF1スコアをサンプル数で重み付けして平均した、重み付きF1スコアを返す。

    Args:
        tp, fp, fn: クラスごとのTP/FP/FN カウントリスト
        unweightet_total: クラスごとのサンプル数リスト（重みとして使用）
    """
    f1_scores = []
    num_classes = len(tp)

    for i in range(num_classes):
        # precision（精度）: 予測が正の中で実際に正だった割合
        if tp[i] + fp[i] == 0:
            precision = 0
        else:
            precision = tp[i] / (tp[i] + fp[i])
        # recall（再現率）: 実際に正の中で正しく予測できた割合
        if tp[i] + fn[i] == 0:
            recall = 0
        else:
            recall = tp[i] / (tp[i] + fn[i])
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    # クラスサイズで重み付けして平均する
    wf1 = sum([f1_scores[i] * unweightet_total[i] for i in range(num_classes)]) / sum(unweightet_total)
    return wf1
