import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from pathlib import Path
import argparse
import logging
import sys
import warnings
import json

warnings.filterwarnings("ignore")


# ==============================================================================
# 复用 Task2 的特征提取和数据加载
# ==============================================================================

def extract_deap_features(trial_data, sfreq=128):
    channels, time = trial_data.shape
    bands = [(4, 8), (8, 12), (12, 30), (30, 45)]

    freqs, psd = welch(trial_data, fs=sfreq, nperseg=256, axis=1)

    features = []
    for low, high in bands:
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.mean(psd[:, idx], axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-8)
        features.append(de)

    return np.concatenate([f.flatten() for f in features])


def load_data(data_root):
    channels = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]

    data_list = sorted(Path(data_root).glob("*.npz"))
    if len(data_list) == 0:
        raise FileNotFoundError(f"No .npz files found in {data_root}")

    X_all, y_valence, y_arousal = [], [], []
    for data_path in data_list:
        data = np.load(data_path)
        X = data["X"]
        X = X[:, channels, :]

        feat = []
        for i in range(len(X)):
            feat.append(extract_deap_features(X[i]))

        X_all.append(feat)
        y_valence.append(data["valence"] / 9)
        y_arousal.append(data["arousal"] / 9)

    X_all = np.array(X_all)
    y_valence = np.array(y_valence)
    y_arousal = np.array(y_arousal)
    return X_all, y_valence, y_arousal


def classify(data, method):
    if method == "binary":
        return np.array(data > 5 / 9).astype(int)
    elif method == "threeclass":
        return np.digitize(data, bins=[4 / 9, 6 / 9])
    else:
        raise ValueError(f"Unknown method: {method}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# ==============================================================================
# MLP 模型工厂
# ==============================================================================

def make_mlp(seed):
    return MLPClassifier(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        solver="adam",
        alpha=0.05,
        batch_size=64,
        max_iter=200,
        random_state=seed,
    )


# ==============================================================================
# LOSO: Leave-One-Subject-Out
# ==============================================================================

def run_loso(X_all, y_all, task, output_dir, num_repeats=10, seed=42):
    n_subjects = X_all.shape[0]
    per_subject_acc = []
    all_preds, all_trues = [], []

    print(f"\n{'='*60}")
    print(f"LOSO: Leave-One-Subject-Out ({task})")
    print(f"{'='*60}")

    for test_subj in range(n_subjects):
        train_subjs = [i for i in range(n_subjects) if i != test_subj]

        X_train = X_all[train_subjs].reshape(-1, X_all.shape[-1])
        y_train_cont = y_all[train_subjs].reshape(-1)
        y_train_label = classify(y_train_cont, task)

        X_test = X_all[test_subj]
        y_test_cont = y_all[test_subj]
        y_test_label = classify(y_test_cont, task)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        k = min(60, X_train.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        X_train = selector.fit_transform(X_train, y_train_label)
        X_test = selector.transform(X_test)

        n_classes = len(np.unique(y_train_label))
        probas = np.zeros((X_test.shape[0], n_classes))
        for rep in range(num_repeats):
            mlp = make_mlp(seed + rep)
            mlp.fit(X_train, y_train_label)
            probas += mlp.predict_proba(X_test)

        probas /= num_repeats
        y_pred = probas.argmax(axis=1)

        acc = accuracy_score(y_test_label, y_pred)
        per_subject_acc.append(acc)
        all_preds.append(y_pred)
        all_trues.append(y_test_label)
        print(f"  Subject {test_subj + 1:2d}  accuracy: {acc:.4f}")

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    overall_acc = np.mean(per_subject_acc)

    category_labels = _get_category_labels(task)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"LOSO Cross-Subject Results ({task})\n")
        f.write(f"{'='*60}\n")
        f.write(f"Per-subject accuracy:\n")
        for i, acc in enumerate(per_subject_acc):
            f.write(f"  s{i+1:02d}: {acc:.4f}\n")
        f.write(f"\nOverall mean accuracy: {overall_acc:.4f}\n")
        f.write(f"Overall std: {np.std(per_subject_acc):.4f}\n\n")
        f.write(classification_report(all_trues, all_preds,
                                      target_names=category_labels))
    print(f"\nOverall mean accuracy: {overall_acc:.4f}")
    print(f"Results saved to: {results_file}")

    csv_file = output_dir / "per_subject_accuracy.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("subject_id,accuracy\n")
        for i, acc in enumerate(per_subject_acc):
            f.write(f"s{i+1:02d},{acc:.4f}\n")
    print(f"Per-subject CSV saved to: {csv_file}")

    return dict(per_subject_acc=per_subject_acc, overall_acc=overall_acc)


# ==============================================================================
# Subject-Pair Heatmap
# ==============================================================================

def _compute_pair_accuracy(X_train_subj, y_train_subj_cont, y_train_subj_label,
                           X_test_subj, y_test_subj_cont, y_test_subj_label,
                           task, num_repeats, seed):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_subj)
    X_test = scaler.transform(X_test_subj)

    n_classes = len(np.unique(y_train_subj_label))
    probas = np.zeros((X_test.shape[0], max(n_classes, 2)))
    for rep in range(num_repeats):
        mlp = make_mlp(seed + rep)
        mlp.fit(X_train, y_train_subj_label)
        p = mlp.predict_proba(X_test)
        probas[:, :p.shape[1]] += p

    probas /= num_repeats
    y_pred = probas.argmax(axis=1)
    return accuracy_score(y_test_subj_label, y_pred)


def run_subject_pair_heatmap(X_all, y_all, task, output_dir,
                              num_repeats=3, seed=42):
    n_subjects = X_all.shape[0]
    matrix = np.full((n_subjects, n_subjects), np.nan)

    print(f"\n{'='*60}")
    print(f"Subject-Pair Heatmap ({task})")
    print(f"{'='*60}")

    total_pairs = n_subjects * n_subjects
    pair_idx = 0

    for train_subj in range(n_subjects):
        for test_subj in range(n_subjects):
            pair_idx += 1
            X_train = X_all[train_subj]
            y_train_cont = y_all[train_subj]
            y_train_label = classify(y_train_cont, task)

            if train_subj == test_subj:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                fold_accs = []
                for tr_idx, te_idx in skf.split(X_train, y_train_label):
                    acc = _compute_pair_accuracy(
                        X_train[tr_idx], y_train_cont[tr_idx], y_train_label[tr_idx],
                        X_train[te_idx], y_train_cont[te_idx], y_train_label[te_idx],
                        task, num_repeats, seed,
                    )
                    fold_accs.append(acc)
                matrix[train_subj, test_subj] = np.mean(fold_accs)
            else:
                X_test = X_all[test_subj]
                y_test_cont = y_all[test_subj]
                y_test_label = classify(y_test_cont, task)

                acc = _compute_pair_accuracy(
                    X_train, y_train_cont, y_train_label,
                    X_test, y_test_cont, y_test_label,
                    task, num_repeats, seed,
                )
                matrix[train_subj, test_subj] = acc

            print(f"  [{pair_idx:4d}/{total_pairs}]  "
                  f"train s{train_subj+1:02d} -> test s{test_subj+1:02d}: "
                  f"{matrix[train_subj, test_subj]:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "subject_pair_matrix.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        header = "train\\test," + ",".join(f"s{i+1:02d}" for i in range(n_subjects))
        f.write(header + "\n")
        for i in range(n_subjects):
            row = f"s{i+1:02d}," + ",".join(
                f"{matrix[i, j]:.4f}" if not np.isnan(matrix[i, j]) else "nan"
                for j in range(n_subjects)
            )
            f.write(row + "\n")
    print(f"Pair matrix CSV saved to: {csv_file}")

    _plot_heatmap(matrix, task, output_dir)
    _plot_diagonal_vs_offdiagonal(matrix, task, output_dir)

    diag_vals = np.diag(matrix)
    offdiag_vals = matrix[~np.eye(n_subjects, dtype=bool)]
    print(f"\nHeatmap summary:")
    print(f"  Diagonal mean (within-subject):  {np.mean(diag_vals):.4f}")
    print(f"  Off-diagonal mean (cross-subject): {np.mean(offdiag_vals):.4f}")

    return matrix


# ==============================================================================
# 可视化
# ==============================================================================

def _plot_heatmap(matrix, task, output_dir):
    n_subjects = matrix.shape[0]
    labels = [f"s{i+1}" for i in range(n_subjects)]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=labels, yticklabels=labels,
        vmin=0.0, vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Accuracy"},
    )
    plt.xlabel("Test Subject")
    plt.ylabel("Train Subject")
    plt.title(f"Cross-Subject Generalization Heatmap ({task})\n"
              "Row: Train Subject  |  Column: Test Subject  |  Diagonal: within-subject (5-fold CV)")

    fig_file = output_dir / "subject_pair_heatmap.png"
    plt.tight_layout()
    plt.savefig(fig_file, dpi=150)
    plt.close()
    print(f"Heatmap saved to: {fig_file}")


def _plot_diagonal_vs_offdiagonal(matrix, task, output_dir):
    n_subjects = matrix.shape[0]
    diag = np.diag(matrix)
    offdiag = matrix[~np.eye(n_subjects, dtype=bool)]

    plt.figure(figsize=(8, 5))
    plt.hist(diag, bins=15, alpha=0.7, label=f"Within-subject (mean={np.mean(diag):.3f})")
    plt.hist(offdiag, bins=30, alpha=0.7, label=f"Cross-subject (mean={np.mean(offdiag):.3f})")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.title(f"Within-subject vs Cross-subject Accuracy ({task})")
    plt.legend()
    plt.tight_layout()

    fig_file = output_dir / "diagonal_vs_offdiagonal.png"
    plt.savefig(fig_file, dpi=150)
    plt.close()
    print(f"Distribution plot saved to: {fig_file}")


# ==============================================================================
# 辅助
# ==============================================================================

def _get_category_labels(task):
    if task == "binary":
        return ["N", "P"]
    elif task == "threeclass":
        return ["N", "U", "P"]
    else:
        return [str(i) for i in range(10)]


def _print_header(title):
    print(f"\n{'#'*60}")
    print(f"# {title}")
    print(f"{'#'*60}")


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    default_data = project_root / "Task1-preprocess" / "data" / "task2" / "npz"
    default_output = project_root / "Task3-transfer_learning" / "output"

    parser = argparse.ArgumentParser(
        description="Cross-Subject Generalization Analysis (Task3)"
    )
    parser.add_argument(
        "--data_root", type=Path, default=default_data,
        help="Path to .npz data directory (default: Task1 preprocessed data)",
    )
    parser.add_argument(
        "--output_root", type=Path, default=default_output,
        help="Root directory for output (default: Task3-transfer_learning/output)",
    )
    parser.add_argument(
        "--task", type=str, default="binary",
        choices=["binary", "threeclass"],
        help="Classification task: binary or threeclass",
    )
    parser.add_argument(
        "--loso_repeats", type=int, default=10,
        help="MLP repeat count for LOSO (default: 10)",
    )
    parser.add_argument(
        "--heatmap_repeats", type=int, default=3,
        help="MLP repeat count for subject-pair heatmap (default: 3, lower for speed)",
    )
    parser.add_argument(
        "--skip_heatmap", action="store_true", default=False,
        help="Skip the subject-pair heatmap (faster)",
    )
    parser.add_argument(
        "--skip_loso", action="store_true", default=False,
        help="Skip LOSO experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading data from:", args.data_root)
    X_all, y_valence, y_arousal = load_data(args.data_root)
    print(f"  X_all shape: {X_all.shape}  (subjects={X_all.shape[0]}, "
          f"trials={X_all.shape[1]}, features={X_all.shape[2]})")

    y_all = y_valence

    if not args.skip_loso:
        loso_output = args.output_root / "cross_subject_loso" / args.task
        run_loso(X_all, y_all, args.task, loso_output,
                 num_repeats=args.loso_repeats, seed=args.seed)

    if not args.skip_heatmap:
        heatmap_output = args.output_root / "cross_subject_heatmap" / args.task
        run_subject_pair_heatmap(X_all, y_all, args.task, heatmap_output,
                                  num_repeats=args.heatmap_repeats, seed=args.seed)

    _print_header("DONE")


if __name__ == "__main__":
    main()
