"""Build tables and figures from saved SER result JSONs without loading data or models."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44)
SCORES = (("uar", "UAR"), ("macro_f1", "Macro F1"), ("wa", "Accuracy"))
MODES = (("none", "Unweighted", "#2563eb"), ("balanced", "Weighted", "#ea580c"))
INPUTS = {
    "comparison42": "runs/msp_class_weight_comparison/seeds-42/comparison_summary.json",
    "comparison4344": "runs/msp_class_weight_comparison/seeds-43-44/comparison_summary.json",
    "study42": "runs/ser_decoder_study/formal/initial-seed-42/study_summary.json",
    "study4344": "runs/ser_decoder_study/formal/followup-seeds-43-44/study_summary.json",
    "timing42": "runs/ser_decoder_timing_check_20260903/formal/initial-seed-42/study_summary.json",
    "timing_details42": "runs/ser_decoder_timing_check_20260903/formal/initial-seed-42/study_timings.json",
}


def check_metrics(metrics):
    """Check saved aggregate metrics against their saved confusion matrix."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
    tp = matrix.diagonal()
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(4), where=matrix.sum(1) != 0)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(4), where=matrix.sum(0) != 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros(4), where=recall + precision != 0)
    for key, value in (("uar", recall.mean()), ("macro_f1", f1.mean()), ("wa", tp.sum() / matrix.sum())):
        if not math.isclose(metrics[key], float(value), abs_tol=1e-10):
            raise ValueError(f"Saved {key} disagrees with confusion matrix")
    for i, row in enumerate(metrics["class_metrics"]):
        assert row["support"] == matrix[i].sum()
        assert math.isclose(row["recall"], recall[i], abs_tol=1e-10)
        assert math.isclose(row["precision"], precision[i], abs_tol=1e-10)
        assert math.isclose(row["f1"], f1[i], abs_tol=1e-10)


def check_training(training):
    history = training["history"]
    assert [row["epoch"] for row in history] == list(range(1, 11))
    assert training["config"]["epochs"] == 10
    assert training["config"]["batch_size"] == 8
    for row in history:
        check_metrics(row["validation"])
        assert math.isfinite(row["train_loss"])
    best = max(history, key=lambda row: (
        row["validation"]["uar"], row["validation"]["macro_f1"], -row["validation"]["loss"]
    ))
    assert best["epoch"] == training["best_epoch"]
    assert best["validation"] == training["best_validation_metrics"]


def load_results(root):
    documents, provenance = {}, []
    for name, relative in INPUTS.items():
        raw = (root / relative).read_bytes()
        documents[name] = json.loads(raw)
        provenance.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    studies = {run["seed"]: run for name in ("study42", "study4344") for run in documents[name]["runs"]}
    comparisons = {run["seed"]: run for name in ("comparison42", "comparison4344") for run in documents[name]["runs"]}
    assert set(studies) == set(comparisons) == set(SEEDS)
    first, second = documents["comparison42"], documents["comparison4344"]
    assert first["validation_signature"] == second["validation_signature"]
    assert first["cache_id"] == second["cache_id"]
    assert first["loss_config"] == second["loss_config"]
    assert not first["test_evaluated"] and not second["test_evaluated"]
    msp = {}
    for seed in SEEDS:
        msp[seed] = {"none": comparisons[seed]["baseline"]["training"], "balanced": comparisons[seed]["weighted"]}
        assert msp[seed]["none"]["history"] == studies[seed]["parent"]["history"]
        for mode in ("none", "balanced"):
            check_training(msp[seed][mode])
        plain = dict(msp[seed]["none"]["config"])
        weighted = dict(msp[seed]["balanced"]["config"])
        assert plain.pop("class_weighting", "none") == "none"
        assert weighted.pop("class_weighting") == "balanced"
        assert plain == weighted
        check_training(studies[seed]["child"])
        for dataset in ("msp_podcast", "hcudb1"):
            before, after = (studies[seed][stage][dataset]["result"] for stage in ("before", "after"))
            assert before["split"] == after["split"] == "test"
            assert before["set_signature"] == after["set_signature"]
            assert before["set_signature"] == studies[42]["before"][dataset]["result"]["set_signature"]
            check_metrics(before["metrics_4class"])
            check_metrics(after["metrics_4class"])
    repeated = documents["timing42"]["runs"][0]
    for stage in ("parent", "child"):
        assert repeated[stage]["history"] == studies[42][stage]["history"]
    for stage in ("before", "after"):
        for dataset in ("msp_podcast", "hcudb1"):
            assert repeated[stage][dataset]["result"]["metrics_4class"] == studies[42][stage][dataset]["result"]["metrics_4class"]
    return documents, msp, studies, provenance


def mean(values):
    return statistics.mean(values)


def percent(value):
    return f"{100 * value:.2f}"


def difference(value):
    return f"{100 * value:+.2f}"


def mean_sd(values):
    return f"{100 * mean(values):.2f} ± {100 * statistics.stdev(values):.2f}"


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=170, bbox_inches="tight")
    figure.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)
    return f"{name}.png"


def paired_panel(axis, before, after, title, labels=("Unweighted", "Weighted"), show_y=True):
    a, b = [100 * value for value in before], [100 * value for value in after]
    a.append(mean(a))
    b.append(mean(b))
    for y, x1, x2 in zip(range(4), a, b):
        axis.plot([x1, x2], [y, y], color="#94a3b8", lw=2, zorder=1)
    axis.scatter(a, range(4), color="#2563eb", label=labels[0], s=[42, 42, 42, 90], zorder=2)
    axis.scatter(b, range(4), color="#ea580c", label=labels[1], s=[42, 42, 42, 90], zorder=3)
    axis.set_yticks(range(4), ["Seed 42", "Seed 43", "Seed 44", "Mean"] if show_y else [])
    axis.invert_yaxis()
    axis.set_title(f"{title}\nMean change {mean(b[:3]) - mean(a[:3]):+.2f} pp")
    axis.set_xlabel("Score (%) — axis zoomed")
    low, high = min(a + b), max(a + b)
    pad = max(2, (high - low) * .22)
    axis.set_xlim(max(0, low - pad), min(100, high + pad))
    axis.grid(axis="x", alpha=.25)


def make_figures(msp, studies, output):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for axis, (key, label) in zip(axes, SCORES):
        a = [msp[seed]["none"]["best_validation_metrics"][key] for seed in SEEDS]
        b = [msp[seed]["balanced"]["best_validation_metrics"][key] for seed in SEEDS]
        paired_panel(axis, a, b, label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, -.015), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("MSP validation: each seed's selected best checkpoint")
    figures["comparison"] = save_figure(fig, output, "01_msp_validation_comparison")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, layout="constrained")
    for col, seed in enumerate(SEEDS):
        for row, (key, label) in enumerate(SCORES):
            axis = axes[row, col]
            for mode, name, color in MODES:
                training = msp[seed][mode]
                history = training["history"]
                axis.plot([h["epoch"] for h in history], [100 * h["validation"][key] for h in history], color=color, marker="o", ms=3, label=name)
                axis.scatter(training["best_epoch"], 100 * training["best_validation_metrics"][key], color=color, marker="*", s=160, edgecolors="black", linewidths=.5, zorder=3)
            axis.set_title(f"Seed {seed} · {label}")
            axis.set_ylim(20, 80)
            axis.set_xticks(range(1, 11))
            axis.grid(alpha=.22)
            if row == 2:
                axis.set_xlabel("Epoch")
            if col == 0:
                axis.set_ylabel("Validation score (%)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("MSP validation over 10 epochs · stars = UAR-selected best epoch · train scores not recorded")
    figures["history"] = save_figure(fig, output, "02_msp_validation_history")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True, layout="constrained")
    for row, (mode, name, _) in enumerate(MODES):
        for col, seed in enumerate(SEEDS):
            history = msp[seed][mode]["history"]
            axis = axes[row, col]
            axis.plot(range(1, 11), [h["train_loss"] for h in history], color="#2563eb", marker="o", ms=3, label="Training objective (batch mean)")
            axis.plot(range(1, 11), [h["validation"]["loss"] for h in history], color="#ea580c", marker="o", ms=3, label="Validation CE (unweighted)")
            axis.axvline(msp[seed][mode]["best_epoch"], ls="--", color="#64748b", alpha=.6)
            axis.set_title(f"{name} · seed {seed}")
            axis.set_ylim(bottom=0)
            axis.set_xticks(range(1, 11))
            axis.grid(alpha=.22)
            if col == 0:
                axis.set_ylabel("Loss")
            if row == 1:
                axis.set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Loss trends · training and validation use different evaluation procedures; read trends, not gaps")
    figures["loss"] = save_figure(fig, output, "03_msp_loss_history")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    classes = [row["class_label"] for row in msp[42]["none"]["best_validation_metrics"]["class_metrics"]]
    for axis, key in zip(axes, ("precision", "recall", "f1")):
        for i, (mode, name, color) in enumerate(MODES):
            values = [mean([msp[s][mode]["best_validation_metrics"]["class_metrics"][c][key] for s in SEEDS]) * 100 for c in range(4)]
            bars = axis.barh(np.arange(4) + (i - .5) * .34, values, height=.32, color=color, label=name)
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axis.set_yticks(range(4), classes)
        axis.invert_yaxis()
        axis.set_xlim(0, 100)
        axis.set_xlabel("Score (%)")
        axis.set_title(key.capitalize())
        axis.grid(axis="x", alpha=.2)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("MSP validation class scores · mean of per-seed metrics at each selected checkpoint")
    figures["classes"] = save_figure(fig, output, "04_msp_class_scores")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, dataset in enumerate(("msp_podcast", "hcudb1")):
        for col, (key, label) in enumerate(SCORES):
            values = [[studies[s][stage][dataset]["result"]["metrics_4class"][key] for s in SEEDS] for stage in ("before", "after")]
            dataset_label = "MSP test" if dataset == "msp_podcast" else "HCUDB test"
            paired_panel(axes[row, col], *values, f"{dataset_label} · {label}", labels=("Before HCUDB", "After HCUDB"))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, -.015), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Saved TEST results · original unweighted MSP → HCUDB experiment · not the class-weight comparison")
    figures["transfer"] = save_figure(fig, output, "05_saved_transfer_test")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout="constrained")
    for axis, (key, label) in zip(axes, SCORES):
        for seed, color in zip(SEEDS, ("#2563eb", "#ea580c", "#059669")):
            training = studies[seed]["child"]
            axis.plot(range(1, 11), [h["validation"][key] * 100 for h in training["history"]], color=color, marker="o", ms=3, label=f"Seed {seed}")
            axis.scatter(training["best_epoch"], training["best_validation_metrics"][key] * 100, color=color, marker="*", s=160, edgecolor="black", linewidth=.5, zorder=4)
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylim(20, 80)
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Validation score (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("HCUDB continuation: validation history · stars = UAR-selected best epoch")
    figures["hcudb"] = save_figure(fig, output, "06_hcudb_validation_history")
    return figures


class Report:
    def __init__(self, output):
        self.output = output
        self.md, self.html = [], []

    def heading(self, title, level=2):
        self.md.append(f"{'#' * level} {title}\n")
        self.html.append(f"<h{level}>{html.escape(title)}</h{level}>")

    def paragraph(self, text):
        self.md.append(text + "\n")
        self.html.append(f"<p>{html.escape(text)}</p>")

    def reference(self, title, url):
        self.md.append(f"根拠：[{title}]({url})\n")
        self.html.append(f'<p>根拠：<a href="{html.escape(url, quote=True)}">{html.escape(title)}</a></p>')

    def table(self, headers, rows):
        self.md.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        self.md.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        self.md.append("")
        self.html.append('<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>')

    def figure(self, filename, caption):
        self.md.append(f"![{caption}]({filename})\n")
        encoded = base64.b64encode((self.output / filename).read_bytes()).decode()
        self.html.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')

    def write(self):
        (self.output / "report.md").write_text("\n".join(self.md), encoding="utf-8")
        style = 'body{font-family:system-ui,"Yu Gothic",sans-serif;color:#172033;background:#f4f6fa;margin:0;line-height:1.8}main{max-width:1200px;margin:24px auto;background:white;padding:38px 44px;border-radius:12px}h1{font-size:29px;line-height:1.5}h2{font-size:23px;margin-top:44px;border-bottom:2px solid #dbeafe;padding-bottom:8px}p{max-width:1000px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #dbe1e9;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#eaf1fc}tbody tr:nth-child(even){background:#f8fafc}.table-wrap{overflow:auto}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:13px;color:#526078}a{color:#174ea6}@media(max-width:700px){main{margin:0;padding:20px;border-radius:0}h1{font-size:23px}}@media print{body{background:white}main{margin:0;padding:0}figure,table{break-inside:avoid}}'
        document = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SER 保存済み結果の整理</title><style>' + style + '</style><main>' + '\n'.join(self.html) + '</main></html>'
        (self.output / "report.html").write_text(document, encoding="utf-8")


def build_report(root, output):
    documents, msp, studies, provenance = load_results(root)
    output.mkdir(parents=True, exist_ok=True)
    figures = make_figures(msp, studies, output)
    report = Report(output)
    report.heading("保存済み結果の整理：MSPのクラス重み比較とHCUDB追加学習", 1)
    report.paragraph("2026-09-03｜seed 42・43・44｜各10 epoch｜保存済みJSONのみを集計。音声・特徴キャッシュ・checkpoint本体は読み込まず、新しい学習・推論・test評価は実行していません。")
    report.paragraph("主な結果：MSPの重み付けでvalidation UARは3 seedすべて改善しました。macro F1は平均ではほぼ不変、正解率は低下しました。学習lossは低下していますが、trainの分類scoreが未記録なので、trainとvalidationのscore差は算出できません。")

    report.heading("1. 保存済みの数値で確認できる範囲")
    report.table(["実験", "train", "validation", "test"], [
        ["MSP 重みなし・3 seed", "各epochの学習lossのみ", "各epochのscore・loss・クラス別成績", "HCUDB追加学習前の結果あり"],
        ["MSP 重みあり・3 seed", "各epochの重み付き学習lossのみ", "各epochのscore・loss・クラス別成績", "未評価"],
        ["HCUDB追加学習・3 seed", "各epochの学習lossのみ", "各epochのscore・loss・クラス別成績", "追加学習後の両データセットの結果あり"],
    ])
    report.paragraph("trainのUAR・macro F1・正解率は、今回の保存結果にはありません。空欄を推定値やvalidationの値で埋めていません。時間計測用に再実行したseed 42は元の履歴・評価指標と一致することを確認し、4個目の独立seedとして数えていません。")

    report.heading("2. MSP：重みなし・ありのvalidation比較")
    report.paragraph("各seedのvalidation UARで選んだbest checkpoint同士を比較しています。同点時はmacro F1、次にlossを使用します。epoch数・バッチサイズ・モデル・学習率などの設定と、保存されたvalidation集合signatureの一致を確認しました。scoreは百分率、差は百分率ポイントです。±は3 seedの標本標準偏差で、信頼区間ではありません。")
    aggregate = {}
    rows = []
    for key, label in SCORES:
        a = [msp[s]["none"]["best_validation_metrics"][key] for s in SEEDS]
        b = [msp[s]["balanced"]["best_validation_metrics"][key] for s in SEEDS]
        aggregate[key] = {"unweighted_mean": mean(a), "weighted_mean": mean(b), "delta_pp": 100 * (mean(b) - mean(a))}
        delta = f"{100 * (mean(b) - mean(a)):+.5f}" if key == "macro_f1" else difference(mean(b) - mean(a))
        rows.append([label, mean_sd(a), mean_sd(b), delta])
    report.table(["指標", "重みなし：平均 ± SD", "重みあり：平均 ± SD", "平均差（ポイント）"], rows)
    report.figure(figures["comparison"], "同一seedを線で接続。各指標の横軸は差を見やすくするため拡大しています。")
    rows = []
    for seed in SEEDS:
        for mode, label, _ in MODES:
            t = msp[seed][mode]
            rows.append([seed, label, t["best_epoch"], *[percent(t["best_validation_metrics"][k]) for k, _ in SCORES], f'{t["best_validation_metrics"]["loss"]:.4f}'])
    report.table(["seed", "損失設定", "best epoch", "UAR (%)", "macro F1 (%)", "正解率 (%)", "validation loss"], rows)
    report.paragraph("UARの改善は『各感情の再現率を等しく平均した成績』の改善です。macro F1も一緒に改善したとは言えず、重み付けで全体的な分類性能が一律に上がった、またはtestでも改善したとは結論しません。")

    report.heading("3. MSP：学習は進んでいたか")
    report.figure(figures["history"], "validationのscore推移。星は、その指標の最大値ではなく、UARを優先して選んだ共通のbest epochを示します。")
    rows = []
    for seed in SEEDS:
        for mode, name, _ in MODES:
            t = msp[seed][mode]
            first, last = t["history"][0], t["history"][-1]
            rows.append([seed, name, f'{first["train_loss"]:.4f} → {last["train_loss"]:.4f}', f'{first["validation"]["loss"]:.4f} → {last["validation"]["loss"]:.4f}', percent(t["best_validation_metrics"]["uar"]), percent(last["validation"]["uar"]), difference(last["validation"]["uar"] - t["best_validation_metrics"]["uar"])])
    report.table(["seed", "損失設定", "train loss：epoch 1 → 10", "validation loss：epoch 1 → 10", "best UAR (%)", "epoch 10 UAR (%)", "最終−best（ポイント）"], rows)
    report.figure(figures["loss"], "lossの経時変化。各曲線の変化を見てください。曲線間の縦の差は、そのまま過学習の程度を表しません。")
    report.paragraph("train lossは学習中にモデルを更新しながら得たバッチlossの平均です。validation lossはepoch終了時のモデルで計算した、発話平均の重みなしcross entropyです。特に重みあり学習では損失関数の重みが異なるので、train lossとvalidation lossの絶対値を直接比較できません。重みなしとありのtrain lossも同じ尺度の成績として比較していません。")
    report.paragraph("観察：MSPの6実行すべてで、epoch 10のtrain lossはepoch 1より低くなっています。一方、重みありのbest epochは42: 2、43: 3、44: 6で、10 epoch目のUARはいずれもbestを下回っています。最適化は進んでおり、後半にvalidationの改善へつながらなくなる傾向が見られます。過学習を疑う材料ですが、trainの分類scoreがないため、その差を使った確認はできません。『学習不足だからepochを増やせば改善する』とも、この結果だけでは言えません。")

    report.heading("4. MSP：どの感情の成績が変わったか")
    report.paragraph("以下は各seedのbest validationでのクラス別指標を計算後、3 seedで平均した値です。平均混同行列からF1を再計算したものではありません。validationは3,600件、happyは1,808件（50.22%）です。")
    report.figure(figures["classes"], "重み付けによりhappy以外の再現率は向上しましたが、precisionの低下も生じています。")
    class_rows = []
    for index, row in enumerate(msp[42]["none"]["best_validation_metrics"]["class_metrics"]):
        values = {mode: {key: mean([msp[s][mode]["best_validation_metrics"]["class_metrics"][index][key] for s in SEEDS]) for key in ("precision", "recall", "f1")} for mode, _, _ in MODES}
        a, b = values["none"], values["balanced"]
        class_rows.append([row["class_label"], row["support"], f'{percent(a["precision"])} → {percent(b["precision"])}', f'{percent(a["recall"])} → {percent(b["recall"])}', f'{percent(a["f1"])} → {percent(b["f1"])}'])
    report.table(["感情", "validation件数", "precision (%)：なし → あり", "recall (%)：なし → あり", "F1 (%)：なし → あり"], class_rows)
    report.paragraph("happyの再現率低下が正解率を押し下げ、他3クラスの再現率改善がUARを押し上げています。sadnessとdisgustは再現率が向上した一方でprecisionが低下し、F1の改善は小さくなっています。これは『見せかけの成績の修正』ではなく、予測が変化して生じたクラス間の成績のトレードオフです。")

    report.heading("5. 保存済みtest：HCUDBへの追加学習前後")
    report.paragraph("この節は元の重みなしMSP→HCUDB実験の保存済みtest結果です。今回のMSP重み付け実験とは別です。beforeはMSP学習後、afterはHCUDB追加学習後を意味します。before/afterと3 seed間で同じ評価集合signatureであることを確認しました。")
    transfer = {}
    rows = []
    for dataset in ("msp_podcast", "hcudb1"):
        transfer[dataset] = {}
        for key, label in SCORES:
            a, b = [[studies[s][stage][dataset]["result"]["metrics_4class"][key] for s in SEEDS] for stage in ("before", "after")]
            transfer[dataset][key] = {"before_mean": mean(a), "after_mean": mean(b), "delta_pp": 100 * (mean(b) - mean(a))}
            rows.append([dataset, label, mean_sd(a), mean_sd(b), difference(mean(b) - mean(a))])
    report.table(["test集合", "指標", "追加学習前：平均 ± SD", "追加学習後：平均 ± SD", "差（ポイント）"], rows)
    report.figure(figures["transfer"], "元の転移実験の保存済みtest成績。重みありMSPモデルのtest成績は含めていません。")
    report.paragraph("HCUDBのtest成績は3 seedすべてで向上しました。一方、MSPのtest成績はUAR・macro F1・正解率のすべてで3 seedとも低下しました。HCUDBへの適応とMSPでの性能維持には、この保存済み実験で明確な差が見られます。")
    report.table(["seed", "HCUDB best epoch", "validation UAR (%)", "validation macro F1 (%)", "validation正解率 (%)"], [[s, studies[s]["child"]["best_epoch"], *[percent(studies[s]["child"]["best_validation_metrics"][k]) for k, _ in SCORES]] for s in SEEDS])
    report.figure(figures["hcudb"], "HCUDBのvalidation推移。seed 43は10 epoch目がbestで、すべての実行が早いepochで頭打ちになったわけではありません。")

    report.heading("6. 結果の参照とデータ漏洩について")
    report.paragraph("保存済み結果を表やグラフに整理すること自体は、testデータを学習入力に混ぜることではありません。ただし、test由来の指標を見てモデル・重み・epochなどを選び直すと、元の音声や特徴に触れていなくても、testの情報が設定選択に入ります。結果ファイルだから無条件に影響がない、という扱いにはしません。今回の整理では設定を選び直していません。validationは設定選択に使い、testの既存結果は記述的な振り返りとして区別しています。")
    report.reference("scikit-learn公式資料：Cross-validation: evaluating estimator performance", "https://scikit-learn.org/stable/modules/cross_validation.html")

    report.heading("7. 時間記録と出典")
    timings = documents["timing_details42"]
    report.paragraph("全実験で同じ範囲の時間が保存されていないため、以下は記録範囲を明示した参考値です。Notebookを起動してからの実時間と同一とは限りません。")
    report.table(["実行", "記録された範囲", "分"], [
        ["高速化後 seed 42", "study処理（入口のキャッシュ完全検証を除く）", f'{timings["study_seconds"] / 60:.2f}'],
        ["高速化後 seed 42", "入口のキャッシュ完全検証（2データセット合計）", f'{sum(v["seconds"] for v in timings["cache_validation"].values()) / 60:.2f}'],
        ["重みあり seed 42", "比較処理（準備セルの完全検証を除く）", f'{documents["comparison42"]["seconds"] / 60:.2f}'],
        ["重みあり seed 43・44", "比較処理合計（準備セルの完全検証を除く）", f'{documents["comparison4344"]["seconds"] / 60:.2f}'],
    ])
    report.paragraph("集計元6ファイルの相対パスとSHA-256をsource_manifest.jsonに保存しました。元のresult JSONを上書きしていません。集計用JSONには発話ID・個別予測を含めず、表とグラフの数値のみを収録しています。")
    for item in provenance:
        report.paragraph(item["path"])
    report.write()
    (output / "source_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    def compact_metrics(metrics):
        return {key: metrics[key] for key in ("uar", "macro_f1", "wa", "loss", "class_metrics") if key in metrics}

    def compact_training(training):
        return {
            "config": training["config"],
            "best_epoch": training["best_epoch"],
            "best_validation_metrics": compact_metrics(training["best_validation_metrics"]),
            "history": [{"epoch": row["epoch"], "train_loss": row["train_loss"], "validation": compact_metrics(row["validation"])} for row in training["history"]],
        }

    aggregate_payload = {
        "seeds": list(SEEDS),
        "score_units": "fractions in [0, 1]; delta_pp is percentage points",
        "standard_deviation": "sample standard deviation across the three seeds; not a confidence interval",
        "msp_validation": aggregate,
        "saved_transfer_test": transfer,
        "train_classification_scores_available": False,
        "weighted_test_evaluated": False,
        "best_validation_rows": [{"seed": s, "weighting": mode, "best_epoch": msp[s][mode]["best_epoch"], "scores": {k: msp[s][mode]["best_validation_metrics"][k] for k, _ in SCORES}} for s in SEEDS for mode, _, _ in MODES],
        "msp_training": {s: {mode: compact_training(msp[s][mode]) for mode, _, _ in MODES} for s in SEEDS},
        "hcudb_training": {s: compact_training(studies[s]["child"]) for s in SEEDS},
        "transfer_test_per_seed": {s: {stage: {dataset: compact_metrics(studies[s][stage][dataset]["result"]["metrics_4class"]) for dataset in ("msp_podcast", "hcudb1")} for stage in ("before", "after")} for s in SEEDS},
    }
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregate_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Check that report generation did not mutate the result files.
    for item in provenance:
        assert hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    print(json.dumps({"report": str(output / "report.html"), "msp_validation": aggregate, "transfer_test": transfer, "figures": len(figures), "source_files_unchanged": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    destination = arguments.output or arguments.root / "docs/reports/2026-09-03-saved-results-review"
    build_report(arguments.root, destination)
