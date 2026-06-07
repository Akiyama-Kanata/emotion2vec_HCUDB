"""
VAD中間表現を使った2段階学習パイプライン。
Stage 1: VA-CCC損失でVADDecoderを学習
Stage 2: CrossEntropyで線形分類器（＋FNN小LR）を学習
"""

import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim

from data import build_dataloaders, load_iemocap_with_va
from loss import stage1_loss
from model import EmotionClassifier

logger = logging.getLogger("VAD_Downstream")


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def train_stage1(model: EmotionClassifier, loader, optimizer, device):
    """1エポック分の Stage 1 学習（CCC損失のみ）。累積損失を返す。"""
    model.vad_decoder.train()
    model.classifier.eval()  # Stage 1 では分類器は更新しない

    total_loss = 0.0
    for batch in loader:
        net_input = batch["net_input"]
        va_labels = batch.get("va_labels")
        if va_labels is None:
            continue

        feats = net_input["feats"].to(device)
        padding_mask = net_input["padding_mask"].to(device)
        va_labels = va_labels.to(device)

        optimizer.zero_grad()
        vad, _ = model(feats, padding_mask)
        loss = stage1_loss(vad, va_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def eval_stage1(model: EmotionClassifier, loader, device) -> float:
    """Stage 1 評価: 平均 CCC損失を返す。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            net_input = batch["net_input"]
            va_labels = batch.get("va_labels")
            if va_labels is None:
                continue
            feats = net_input["feats"].to(device)
            padding_mask = net_input["padding_mask"].to(device)
            va_labels = va_labels.to(device)
            vad, _ = model(feats, padding_mask)
            total_loss += stage1_loss(vad, va_labels).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def train_stage2(model: EmotionClassifier, loader, optimizer, criterion, device):
    """1エポック分の Stage 2 学習（CrossEntropy損失のみ）。累積損失を返す。"""
    model.train()
    total_loss = 0.0
    for batch in loader:
        net_input = batch["net_input"]
        labels = batch.get("labels")
        if labels is None:
            continue

        feats = net_input["feats"].to(device)
        padding_mask = net_input["padding_mask"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        _, logits = model(feats, padding_mask)
        loss = criterion(logits, labels.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def evaluate(model: EmotionClassifier, loader, device, num_classes: int):
    """WA / UA / weighted-F1 (%) を返す。"""
    model.eval()
    correct = total = 0
    uw_correct = [0] * num_classes  # UA計算用: クラスごとの正解数
    uw_total = [0] * num_classes    # UA/F1計算用: クラスごとのサンプル数
    tp = [0] * num_classes          # F1計算用: True Positive
    fp = [0] * num_classes          # F1計算用: False Positive
    fn = [0] * num_classes          # F1計算用: False Negative

    for batch in loader:
        net_input = batch["net_input"]
        labels = batch.get("labels")
        if labels is None:
            continue
        feats = net_input["feats"].to(device)
        padding_mask = net_input["padding_mask"].to(device)
        labels = labels.to(device)

        _, logits = model(feats, padding_mask)
        predicted = logits.argmax(dim=1)

        # WAは全サンプルの正解率として集計する。
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

        for i in range(len(labels)):
            # UAとF1のために、クラスごとの正解数・TP/FP/FNをサンプル単位で更新する。
            c = labels[i].item()
            p = predicted[i].item()
            uw_total[c] += 1
            if p == c:
                uw_correct[c] += 1
                tp[c] += 1
            else:
                fp[p] += 1
                fn[c] += 1

    wa = correct / total * 100
    ua = sum(uw_correct[i] / max(uw_total[i], 1) for i in range(num_classes)) / num_classes * 100
    wf1 = _weighted_f1(tp, fp, fn, uw_total) * 100
    return wa, ua, wf1


def _weighted_f1(tp, fp, fn, totals):
    """クラスごとのF1を計算し、各クラスのサンプル数で重み付け平均する。"""
    f1s = []
    for i in range(len(tp)):
        prec = tp[i] / max(tp[i] + fp[i], 1)
        rec = tp[i] / max(tp[i] + fn[i], 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return sum(f1s[i] * totals[i] for i in range(len(tp))) / max(sum(totals), 1)


# ---------------------------------------------------------------------------
# メインパイプライン
# ---------------------------------------------------------------------------

def run_fold(cfg: DictConfig, fold: int, data: dict, device: torch.device):
    """1-fold 分の Stage1 → Stage2 → テスト評価を実行し (wa, ua, f1) を返す。"""
    n_samples = cfg.dataset.n_samples
    test_start = sum(n_samples[:fold])
    test_end = test_start + n_samples[fold]

    train_loader, val_loader, test_loader = build_dataloaders(
        data,
        cfg.dataset.batch_size,
        test_start,
        test_end,
        eval_is_test=cfg.dataset.eval_is_test,
    )

    num_classes = cfg.model.num_classes
    model = EmotionClassifier(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        vad_dim=cfg.model.vad_dim,
        num_classes=num_classes,
    ).to(device)

    save_dir = os.path.join(str(Path.cwd()), f"model_fold{fold + 1}.pth")

    # ---------- Stage 1 ----------
    logger.info(f"[Fold {fold + 1}] Stage 1 開始 ({cfg.training.stage1_epochs} epochs)")
    opt1 = optim.Adam(model.vad_decoder.parameters(), lr=cfg.training.stage1_lr)
    best_s1_loss = float("inf")

    for epoch in range(cfg.training.stage1_epochs):
        train_loss = train_stage1(model, train_loader, opt1, device)
        val_loss = eval_stage1(model, val_loader, device)
        logger.info(
            f"  S1 Epoch {epoch + 1:3d} | train_loss={train_loss / max(len(train_loader), 1):.4f}"
            f" | val_loss={val_loss:.4f}"
        )
        if val_loss < best_s1_loss:
            best_s1_loss = val_loss
            torch.save(model.state_dict(), save_dir)

    model.load_state_dict(torch.load(save_dir))

    # ---------- Stage 2 ----------
    logger.info(f"[Fold {fold + 1}] Stage 2 開始 ({cfg.training.stage2_epochs} epochs)")
    opt2 = optim.Adam([
        {"params": model.vad_decoder.parameters(), "lr": cfg.training.stage2_lr_fnn},
        {"params": model.classifier.parameters(), "lr": cfg.training.stage2_lr_cls},
    ])
    criterion = nn.CrossEntropyLoss()
    best_val_wa = 0.0

    for epoch in range(cfg.training.stage2_epochs):
        train_loss = train_stage2(model, train_loader, opt2, criterion, device)
        val_wa, val_ua, val_f1 = evaluate(model, val_loader, device, num_classes)
        logger.info(
            f"  S2 Epoch {epoch + 1:3d} | loss={train_loss / max(len(train_loader), 1):.4f}"
            f" | val WA={val_wa:.2f}% UA={val_ua:.2f}% F1={val_f1:.2f}%"
        )
        if val_wa > best_val_wa:
            best_val_wa = val_wa
            torch.save(model.state_dict(), save_dir)

    model.load_state_dict(torch.load(save_dir))
    test_wa, test_ua, test_f1 = evaluate(model, test_loader, device, num_classes)
    logger.info(
        f"[Fold {fold + 1}] Test WA={test_wa:.2f}% UA={test_ua:.2f}% F1={test_f1:.2f}%"
    )
    return test_wa, test_ua, test_f1


@hydra.main(config_path="config", config_name="default.yaml", version_base=None)
def main(cfg: DictConfig):
    """設定ファイルを読み込み、全foldの2段階学習と平均スコア集計を実行する。"""
    torch.manual_seed(cfg.common.seed)

    # configのlabel_names順を、そのまま分類器の整数ID順として使う。
    label_dict = {k: i for i, k in enumerate(cfg.dataset.label_names)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # emotion2vec特徴量、カテゴリラベル、VA連続値ラベルをまとめて読み込む。
    data = load_iemocap_with_va(
        cfg.dataset.feat_path,
        label_dict,
        cfg.dataset.va_path,
    )

    n_folds = len(cfg.dataset.n_samples)
    wa_avg = ua_avg = f1_avg = 0.0

    for fold in range(n_folds):
        logger.info(f"===== Fold {fold + 1}/{n_folds} =====")
        # foldごとにGPUキャッシュを空け、前foldの一時メモリを持ち越さない。
        torch.cuda.empty_cache()
        wa, ua, f1 = run_fold(cfg, fold, data, device)
        wa_avg += wa
        ua_avg += ua
        f1_avg += f1

    logger.info(
        f"平均 WA={wa_avg / n_folds:.2f}% UA={ua_avg / n_folds:.2f}% F1={f1_avg / n_folds:.2f}%"
    )


if __name__ == "__main__":
    main()
