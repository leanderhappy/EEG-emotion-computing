import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METHOD_LABELS = {
    "no_transfer": "No Transfer",
    "with_transfer": "DANN Transfer",
}
METHOD_COLORS = {
    "no_transfer": "#4C72B0",
    "with_transfer": "#DD8452",
}
SOURCE_LABELS = {
    "task1": "Task1",
    "official": "Official",
}
TASK_LABELS = {
    "binary": "Binary",
    "threeclass": "Three-Class",
}

EN_FONT = font_manager.FontProperties(family="Times New Roman")
CN_FONT = font_manager.FontProperties(family="SimSun")


def configure_fonts():
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
    plt.rcParams["font.sans-serif"] = ["SimSun", "Times New Roman"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def subject_number(subject):
    return int(subject.lower().lstrip("s"))


def read_accuracy_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "pair_id": int(row["pair_id"]),
                "source_subject": row["source_subject"],
                "target_subject": row["target_subject"],
                "selection_distance": float(row.get("selection_distance") or "nan"),
                "accuracy": float(row["accuracy"]),
            })
    return rows


def load_result_groups(output_root):
    groups = {}
    pattern = "*/*/mlp/*/pairwise_accuracy.csv"
    for csv_path in output_root.glob(pattern):
        parts = csv_path.relative_to(output_root).parts
        if len(parts) < 5:
            continue
        data_source, task, model_name, method = parts[:4]
        if model_name != "mlp" or method not in METHOD_LABELS:
            continue
        groups[(data_source, task, method)] = read_accuracy_csv(csv_path)
    return groups


def available_experiments(groups):
    candidates = sorted({(key[0], key[1]) for key in groups})
    return [
        (data_source, task)
        for data_source, task in candidates
        if (data_source, task, "no_transfer") in groups
        and (data_source, task, "with_transfer") in groups
    ]


def merge_pair_results(groups, data_source, task):
    no_rows = groups[(data_source, task, "no_transfer")]
    tr_rows = groups[(data_source, task, "with_transfer")]
    no_by_target = {row["target_subject"]: row for row in no_rows}
    tr_by_target = {row["target_subject"]: row for row in tr_rows}
    targets = sorted(set(no_by_target) & set(tr_by_target), key=subject_number)

    merged = []
    for target in targets:
        no_row = no_by_target[target]
        tr_row = tr_by_target[target]
        merged.append({
            "pair_id": no_row["pair_id"],
            "source_subject": no_row["source_subject"],
            "target_subject": target,
            "selection_distance": no_row["selection_distance"],
            "no_transfer": no_row["accuracy"],
            "with_transfer": tr_row["accuracy"],
            "delta": tr_row["accuracy"] - no_row["accuracy"],
        })
    return merged


def label_for(data_source, task):
    source_label = SOURCE_LABELS.get(data_source, data_source)
    task_label = TASK_LABELS.get(task, task)
    return source_label, task_label


def apply_english_tick_font(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(EN_FONT)


def set_bilingual_title(fig, ax, chinese_title, english_title):
    fig.suptitle(chinese_title, fontproperties=CN_FONT, fontsize=15, y=0.995)
    ax.set_title(english_title, fontproperties=EN_FONT, fontsize=13, pad=12)


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pair_line(merged, data_source, task, figure_dir):
    source_label, task_label = label_for(data_source, task)
    targets = [row["target_subject"] for row in merged]
    x = np.arange(len(merged))
    no_acc = np.array([row["no_transfer"] for row in merged])
    tr_acc = np.array([row["with_transfer"] for row in merged])

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        x, no_acc, marker="o", linewidth=2.0, markersize=4,
        color=METHOD_COLORS["no_transfer"], label=METHOD_LABELS["no_transfer"],
    )
    ax.plot(
        x, tr_acc, marker="s", linewidth=2.0, markersize=4,
        color=METHOD_COLORS["with_transfer"], label=METHOD_LABELS["with_transfer"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontproperties=EN_FONT)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Target Subject", fontproperties=EN_FONT, fontsize=12)
    ax.set_ylabel("Accuracy", fontproperties=EN_FONT, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(prop=EN_FONT, frameon=False)
    apply_english_tick_font(ax)
    set_bilingual_title(
        fig, ax,
        f"{source_label} {task_label} 每个目标被试准确率对比",
        f"{source_label} {task_label}: Pair-wise Accuracy Comparison",
    )
    save_figure(fig, figure_dir / f"{data_source}_{task}_pair_line.png")


def plot_delta_bar(merged, data_source, task, figure_dir):
    source_label, task_label = label_for(data_source, task)
    targets = [row["target_subject"] for row in merged]
    x = np.arange(len(merged))
    delta = np.array([row["delta"] for row in merged])
    colors = np.where(delta >= 0, "#55A868", "#C44E52")
    max_abs = max(float(np.max(np.abs(delta))), 0.1)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x, delta, color=colors, width=0.72)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontproperties=EN_FONT)
    ax.set_ylim(-max_abs - 0.05, max_abs + 0.05)
    ax.set_xlabel("Target Subject", fontproperties=EN_FONT, fontsize=12)
    ax.set_ylabel("Accuracy Difference (DANN - No Transfer)", fontproperties=EN_FONT, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(
        0.01, 0.96, f"Mean difference = {delta.mean():.4f}",
        transform=ax.transAxes, fontproperties=EN_FONT, fontsize=11,
        va="top", ha="left",
    )
    apply_english_tick_font(ax)
    set_bilingual_title(
        fig, ax,
        f"{source_label} {task_label} 迁移提升量",
        f"{source_label} {task_label}: Transfer Gain per Target",
    )
    save_figure(fig, figure_dir / f"{data_source}_{task}_delta_bar.png")


def plot_distance_delta_scatter(merged, data_source, task, figure_dir):
    source_label, task_label = label_for(data_source, task)
    distances = np.array([row["selection_distance"] for row in merged])
    delta = np.array([row["delta"] for row in merged])
    colors = np.where(delta >= 0, "#55A868", "#C44E52")

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.scatter(distances, delta, s=52, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("CORAL Selection Distance", fontproperties=EN_FONT, fontsize=12)
    ax.set_ylabel("Accuracy Difference (DANN - No Transfer)", fontproperties=EN_FONT, fontsize=12)
    ax.grid(linestyle="--", alpha=0.35)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    if np.allclose(distances, distances[0]):
        ax.text(
            0.5, 0.96, "All selection distances are identical in CSV.",
            transform=ax.transAxes, fontproperties=EN_FONT, fontsize=10,
            va="top", ha="center",
        )
    apply_english_tick_font(ax)
    set_bilingual_title(
        fig, ax,
        f"{source_label} {task_label} 距离与迁移提升关系",
        f"{source_label} {task_label}: Selection Distance vs Transfer Gain",
    )
    save_figure(fig, figure_dir / f"{data_source}_{task}_distance_delta_scatter.png")


def plot_summary(summary_rows, figure_dir):
    labels = [
        f"{SOURCE_LABELS.get(row['data_source'], row['data_source'])}\n"
        f"{TASK_LABELS.get(row['task'], row['task'])}"
        for row in summary_rows
    ]
    x = np.arange(len(summary_rows))
    width = 0.36
    no_mean = np.array([row["no_mean"] for row in summary_rows])
    no_std = np.array([row["no_std"] for row in summary_rows])
    tr_mean = np.array([row["transfer_mean"] for row in summary_rows])
    tr_std = np.array([row["transfer_std"] for row in summary_rows])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(
        x - width / 2, no_mean, width, yerr=no_std, capsize=4,
        color=METHOD_COLORS["no_transfer"], label=METHOD_LABELS["no_transfer"],
    )
    ax.bar(
        x + width / 2, tr_mean, width, yerr=tr_std, capsize=4,
        color=METHOD_COLORS["with_transfer"], label=METHOD_LABELS["with_transfer"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=EN_FONT)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean Accuracy", fontproperties=EN_FONT, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(prop=EN_FONT, frameon=False)
    apply_english_tick_font(ax)
    set_bilingual_title(
        fig, ax,
        "总体平均准确率对比",
        "Summary: Mean Accuracy Comparison",
    )
    save_figure(fig, figure_dir / "summary_mean_accuracy.png")


def write_summary_csv(summary_rows, figure_dir):
    output_path = figure_dir / "summary_statistics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "data_source", "task", "n_pairs",
        "no_transfer_mean", "no_transfer_std",
        "with_transfer_mean", "with_transfer_std",
        "mean_difference",
        "n_improved", "n_degraded", "n_unchanged",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "data_source": row["data_source"],
                "task": row["task"],
                "n_pairs": row["n_pairs"],
                "no_transfer_mean": f"{row['no_mean']:.6f}",
                "no_transfer_std": f"{row['no_std']:.6f}",
                "with_transfer_mean": f"{row['transfer_mean']:.6f}",
                "with_transfer_std": f"{row['transfer_std']:.6f}",
                "mean_difference": f"{row['delta_mean']:.6f}",
                "n_improved": row["n_improved"],
                "n_degraded": row["n_degraded"],
                "n_unchanged": row["n_unchanged"],
            })
    return output_path


def main():
    current_file = Path(__file__).resolve()
    task_root = current_file.parent.parent
    default_output_root = task_root / "output"
    default_figure_dir = default_output_root / "figures"

    parser = argparse.ArgumentParser(description="Visualize Task3 transfer-learning results")
    parser.add_argument("--output_root", type=Path, default=default_output_root)
    parser.add_argument("--figure_dir", type=Path, default=default_figure_dir)
    args = parser.parse_args()

    configure_fonts()
    groups = load_result_groups(args.output_root)
    experiments = available_experiments(groups)
    if not experiments:
        raise FileNotFoundError(
            f"No paired no_transfer/with_transfer CSV files found under {args.output_root}"
        )

    summary_rows = []
    for data_source, task in experiments:
        merged = merge_pair_results(groups, data_source, task)
        no_acc = np.array([row["no_transfer"] for row in merged])
        tr_acc = np.array([row["with_transfer"] for row in merged])
        delta = tr_acc - no_acc

        plot_pair_line(merged, data_source, task, args.figure_dir)
        plot_delta_bar(merged, data_source, task, args.figure_dir)
        plot_distance_delta_scatter(merged, data_source, task, args.figure_dir)

        summary_rows.append({
            "data_source": data_source,
            "task": task,
            "n_pairs": len(merged),
            "no_mean": float(no_acc.mean()),
            "no_std": float(no_acc.std()),
            "transfer_mean": float(tr_acc.mean()),
            "transfer_std": float(tr_acc.std()),
            "delta_mean": float(delta.mean()),
            "n_improved": int(np.sum(delta > 0)),
            "n_degraded": int(np.sum(delta < 0)),
            "n_unchanged": int(np.sum(delta == 0)),
        })

    summary_rows = sorted(summary_rows, key=lambda row: (row["data_source"], row["task"]))
    plot_summary(summary_rows, args.figure_dir)
    summary_csv = write_summary_csv(summary_rows, args.figure_dir)
    print(f"Figures saved to: {args.figure_dir}")
    print(f"Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
