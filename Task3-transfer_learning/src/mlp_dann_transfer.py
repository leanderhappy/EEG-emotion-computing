import argparse
import pickle
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.signal import welch
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dann_model import DANNModel, dann_lambda

warnings.filterwarnings("ignore")


def extract_deap_features(trial_data, sfreq=128):
    bands = [(4, 8), (8, 12), (12, 30), (30, 45)]
    freqs, psd = welch(trial_data, fs=sfreq, nperseg=256, axis=1)
    features = []
    for low, high in bands:
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.mean(psd[:, idx], axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-8)
        features.append(de)
    return np.concatenate([np.array(f).flatten() for f in features])


def load_data(data_root):
    channels = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]
    data_list = sorted(
        list(Path(data_root).glob("*.dat")) +
        list(Path(data_root).glob("*.npz"))
    )
    if len(data_list) == 0:
        raise FileNotFoundError(f"No .dat or .npz files found in {data_root}")

    X_all, y_valence, y_arousal = [], [], []
    for data_path in data_list:
        if data_path.suffix == ".dat":
            with open(data_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            X = data["data"]
            baseline = X[:, :, :384].mean(axis=-1, keepdims=True)
            X = X[:, :, 384:] - baseline
            y_val = data["labels"][:, 0]
            y_aro = data["labels"][:, 1]
        else:
            data = np.load(data_path)
            X = data["X"]
            y_val = data["valence"]
            y_aro = data["arousal"]

        X = X[:, channels, :]
        feat = [extract_deap_features(X[i]) for i in range(len(X))]
        X_all.append(feat)
        y_valence.append(y_val / 9)
        y_arousal.append(y_aro / 9)

    return np.array(X_all), np.array(y_valence), np.array(y_arousal)


def classify(data, method):
    if method == "binary":
        return np.array(data > 5 / 9).astype(int)
    if method == "threeclass":
        return np.digitize(data, bins=[4 / 9, 6 / 9])
    raise ValueError(f"Unknown method: {method}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_subject_id(index):
    return f"s{index + 1:02d}"


def coral_distance(X_source, X_target):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    n_features = X_source.shape[1]
    source_cov = np.cov(X_source, rowvar=False)
    target_cov = np.cov(X_target, rowvar=False)
    diff = source_cov - target_cov
    return float(np.sum(diff * diff) / (4 * n_features * n_features))


def make_domain_similarity_pairs(X_all, y_all, task):
    pairs = []
    n_subjects = X_all.shape[0]
    for target in range(n_subjects):
        best_source = None
        best_distance = np.inf
        for source in range(n_subjects):
            if source == target:
                continue
            y_source = classify(y_all[source], task)
            if len(np.unique(y_source)) < 2:
                continue

            distance = coral_distance(X_all[source], X_all[target])
            if distance < best_distance:
                best_source = source
                best_distance = distance

        if best_source is None:
            raise ValueError(
                f"No valid source subject found for target {format_subject_id(target)}."
            )
        pairs.append((best_source, target, best_distance))
    return pairs


def prepare_pair_data(X_all, y_all, source, target, task):
    X_source = X_all[source]
    X_target = X_all[target]
    y_source = classify(y_all[source], task)
    y_target = classify(y_all[target], task)

    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_source)
    X_target = scaler.transform(X_target)

    k = min(60, X_source.shape[1])
    selector = SelectKBest(mutual_info_classif, k=k)
    X_source = selector.fit_transform(X_source, y_source)
    X_target = selector.transform(X_target)
    return X_source, y_source, X_target, y_target


def make_class_weight_tensor(y_source, n_classes, device):
    class_counts = np.bincount(y_source, minlength=n_classes).astype(np.float32)
    class_weights = np.zeros(n_classes, dtype=np.float32)
    present_classes = class_counts > 0
    class_weights[present_classes] = (
        class_counts[present_classes].sum()
        / (present_classes.sum() * class_counts[present_classes])
    )
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def run_pairwise_no_transfer_mlp(X_all, y_all, pairs, task, output_dir,
                                 target_name, epochs=120,
                                 batch_size=16, lr=5e-4,
                                 seed=42, hidden=32, latent=32,
                                 weight_decay=1e-4, grad_clip=5.0,
                                 log_interval=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds, all_trues = [], []
    records = []
    n_classes = len(_get_category_labels(task))

    print(f"Using device: {device}")
    print(f"\n{'=' * 60}")
    print(f"Pairwise no-transfer Torch MLP ({task}) epochs={epochs} "
          f"bs={batch_size} lr={lr} weight_decay={weight_decay} "
          f"grad_clip={grad_clip}")
    print(f"{'=' * 60}")

    for pair_idx, pair in enumerate(pairs, start=1):
        set_seed(seed + pair_idx)
        source, target, selection_distance = pair
        X_source, y_source, X_target, y_target = prepare_pair_data(
            X_all, y_all, source, target, task,
        )

        source_dataset = TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(y_source, dtype=torch.long),
        )
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)

        model = DANNModel(
            in_dim=X_source.shape[1], n_classes=n_classes, n_domains=2,
            hidden=hidden, latent=latent,
        ).to(device)
        cls_criterion = nn.CrossEntropyLoss(
            weight=make_class_weight_tensor(y_source, n_classes, device)
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        steps_per_epoch = max(len(source_loader), 1)
        model.train()
        for epoch in range(epochs):
            epoch_cls_loss = 0.0
            epoch_cls_correct = 0
            epoch_cls_total = 0

            for x_source_batch, y_source_batch in source_loader:
                x_source_batch = x_source_batch.to(device)
                y_source_batch = y_source_batch.to(device)

                y_source_hat, _ = model(x_source_batch, alpha=0.0)
                loss_cls = cls_criterion(y_source_hat, y_source_batch)

                optimizer.zero_grad()
                loss_cls.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_cls_loss += loss_cls.item()
                epoch_cls_correct += (
                    y_source_hat.argmax(dim=1) == y_source_batch
                ).sum().item()
                epoch_cls_total += len(y_source_batch)

            if log_interval > 0 and (epoch + 1) % log_interval == 0:
                source_acc = epoch_cls_correct / max(epoch_cls_total, 1)
                print(f"  [pair {pair_idx:02d} {format_subject_id(source)}->"
                      f"{format_subject_id(target)}] epoch {epoch + 1:3d} "
                      f"cls_loss={epoch_cls_loss / steps_per_epoch:.4f} "
                      f"src_acc={source_acc:.3f}")

        model.eval()
        with torch.no_grad():
            X_target_t = torch.tensor(X_target, dtype=torch.float32).to(device)
            y_hat, _ = model(X_target_t, alpha=0.0)
            y_pred = y_hat.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_target, y_pred)
        all_preds.append(y_pred)
        all_trues.append(y_target)
        records.append(_make_pair_record(
            pair_idx, source, target, acc, selection_distance,
        ))
        print(_format_pair_log(pair_idx, source, target, acc))

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    category_labels = _get_category_labels(task)
    params = {
        "target": target_name,
        "n_pairs": len(pairs),
        "pair_selection": "domain_similarity",
        "selection_metric": "coral",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "hidden": hidden,
        "latent": latent,
    }
    _save_pairwise_results(
        output_dir, "Pairwise Torch MLP No-transfer Results", task, params,
        records, all_trues, all_preds, category_labels,
    )
    return dict(records=records, overall_acc=np.mean([r["accuracy"] for r in records]))


def run_pairwise_dann_mlp(X_all, y_all, pairs, task, output_dir,
                          target_name, epochs=120,
                          batch_size=16, lr=5e-4, lambda_max=0.3,
                          seed=42, hidden=32, latent=32,
                          pretrain_epochs=20, weight_decay=1e-4,
                          grad_clip=5.0,
                          log_interval=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds, all_trues = [], []
    records = []
    n_classes = len(_get_category_labels(task))
    pretrain_epochs = max(0, min(pretrain_epochs, epochs))

    print(f"Using device: {device}")
    print(f"\n{'=' * 60}")
    print(f"Pairwise DANN-MLP transfer ({task}) epochs={epochs} "
          f"pretrain={pretrain_epochs} bs={batch_size} lr={lr} "
          f"lambda_max={lambda_max} weight_decay={weight_decay} "
          f"grad_clip={grad_clip}")
    print(f"{'=' * 60}")

    for pair_idx, pair in enumerate(pairs, start=1):
        source, target, selection_distance = pair
        set_seed(seed + pair_idx)
        X_source, y_source, X_target, y_target = prepare_pair_data(
            X_all, y_all, source, target, task,
        )

        source_dataset = TensorDataset(
            torch.tensor(X_source, dtype=torch.float32),
            torch.tensor(y_source, dtype=torch.long),
        )
        target_dataset = TensorDataset(torch.tensor(X_target, dtype=torch.float32))
        source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

        model = DANNModel(
            in_dim=X_source.shape[1], n_classes=n_classes, n_domains=2,
            hidden=hidden, latent=latent,
        ).to(device)
        cls_criterion = nn.CrossEntropyLoss(
            weight=make_class_weight_tensor(y_source, n_classes, device)
        )
        dom_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        steps_per_epoch = max(len(source_loader), len(target_loader), 1)
        transfer_steps = max((epochs - pretrain_epochs) * steps_per_epoch, 1)
        transfer_step = 0
        model.train()
        for epoch in range(epochs):
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            epoch_cls_loss = 0.0
            epoch_dom_loss = 0.0
            epoch_cls_correct = 0
            epoch_cls_total = 0
            epoch_dom_correct = 0
            epoch_dom_total = 0

            for _ in range(steps_per_epoch):
                try:
                    x_source_batch, y_source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    x_source_batch, y_source_batch = next(source_iter)

                try:
                    (x_target_batch,) = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    (x_target_batch,) = next(target_iter)

                x_source_batch = x_source_batch.to(device)
                y_source_batch = y_source_batch.to(device)
                x_target_batch = x_target_batch.to(device)

                x_domain_batch = torch.cat([x_source_batch, x_target_batch], dim=0)
                d_batch = torch.cat([
                    torch.zeros(len(x_source_batch), dtype=torch.long),
                    torch.ones(len(x_target_batch), dtype=torch.long),
                ]).to(device)

                if epoch < pretrain_epochs:
                    alpha = 0.0
                else:
                    p = transfer_step / max(transfer_steps - 1, 1)
                    alpha = dann_lambda(p, lambda_max)
                    transfer_step += 1

                y_hat, d_hat = model(x_domain_batch, 1.0)
                y_source_hat = y_hat[:len(x_source_batch)]
                loss_cls = cls_criterion(y_source_hat, y_source_batch)
                loss_dom = dom_criterion(d_hat, d_batch)
                loss = loss_cls if alpha == 0.0 else loss_cls + alpha * loss_dom

                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                epoch_cls_loss += loss_cls.item()
                epoch_dom_loss += loss_dom.item()
                epoch_cls_correct += (
                    y_source_hat.argmax(dim=1) == y_source_batch
                ).sum().item()
                epoch_cls_total += len(y_source_batch)
                epoch_dom_correct += (d_hat.argmax(dim=1) == d_batch).sum().item()
                epoch_dom_total += len(d_batch)

            if log_interval > 0 and (epoch + 1) % log_interval == 0:
                source_acc = epoch_cls_correct / max(epoch_cls_total, 1)
                domain_acc = epoch_dom_correct / max(epoch_dom_total, 1)
                print(f"  [pair {pair_idx:02d} {format_subject_id(source)}->"
                      f"{format_subject_id(target)}] epoch {epoch + 1:3d} "
                      f"cls_loss={epoch_cls_loss / steps_per_epoch:.4f} "
                      f"dom_loss={epoch_dom_loss / steps_per_epoch:.4f} "
                      f"src_acc={source_acc:.3f} dom_acc={domain_acc:.3f} "
                      f"alpha={alpha:.4f}")

        model.eval()
        with torch.no_grad():
            X_target_t = torch.tensor(X_target, dtype=torch.float32).to(device)
            y_hat, _ = model(X_target_t, alpha=0.0)
            y_pred = y_hat.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_target, y_pred)
        all_preds.append(y_pred)
        all_trues.append(y_target)
        records.append(_make_pair_record(
            pair_idx, source, target, acc, selection_distance,
        ))
        print(_format_pair_log(pair_idx, source, target, acc))

    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    category_labels = _get_category_labels(task)
    params = {
        "target": target_name,
        "n_pairs": len(pairs),
        "pair_selection": "domain_similarity",
        "selection_metric": "coral",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lambda_max": lambda_max,
        "pretrain_epochs": pretrain_epochs,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "hidden": hidden,
        "latent": latent,
    }
    _save_pairwise_results(
        output_dir, "Pairwise DANN-MLP Transfer Results", task, params,
        records, all_trues, all_preds, category_labels,
    )
    return dict(records=records, overall_acc=np.mean([r["accuracy"] for r in records]))


def _make_pair_record(pair_idx, source, target, accuracy, selection_distance):
    return {
        "pair_id": pair_idx,
        "source_subject": format_subject_id(source),
        "target_subject": format_subject_id(target),
        "selection_distance": selection_distance,
        "accuracy": accuracy,
    }


def _format_pair_log(pair_idx, source, target, accuracy):
    return (f"  Pair {pair_idx:02d} {format_subject_id(source)} -> "
            f"{format_subject_id(target)} accuracy: {accuracy:.4f}")


def _save_pairwise_results(output_dir, title, task, params, records,
                           all_trues, all_preds, category_labels):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(category_labels)))
    accuracies = [r["accuracy"] for r in records]
    overall_acc = np.mean(accuracies)
    overall_std = np.std(accuracies)

    results_file = output_dir / "pairwise_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"{title} ({task})\n")
        f.write(f"{'=' * 60}\n")
        for key, value in params.items():
            f.write(f"{key}={value}\n")
        f.write("\nPair accuracy:\n")
        f.write("pair_id,source_subject,target_subject,selection_distance,accuracy\n")
        for record in records:
            f.write(f"{record['pair_id']},{record['source_subject']},"
                    f"{record['target_subject']},"
                    f"{record['selection_distance']:.8f},"
                    f"{record['accuracy']:.4f}\n")
        f.write(f"\nOverall mean accuracy: {overall_acc:.4f}\n")
        f.write(f"Overall std: {overall_std:.4f}\n\n")
        f.write(classification_report(
            all_trues, all_preds, labels=labels,
            target_names=category_labels, zero_division=0,
        ))
    print(f"\nOverall mean accuracy: {overall_acc:.4f}")
    print(f"Results saved to: {results_file}")

    csv_file = output_dir / "pairwise_accuracy.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("pair_id,source_subject,target_subject,selection_distance,accuracy\n")
        for record in records:
            f.write(f"{record['pair_id']},{record['source_subject']},"
                    f"{record['target_subject']},"
                    f"{record['selection_distance']:.8f},"
                    f"{record['accuracy']:.4f}\n")
    print(f"Pairwise CSV saved to: {csv_file}")

    fig_file = output_dir / "pairwise_confusion_matrix.png"
    cm = confusion_matrix(all_trues, all_preds, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=category_labels, yticklabels=category_labels)
    plt.xlabel("pred")
    plt.ylabel("real")
    plt.title(f"Pairwise Confusion Matrix - {title} ({task})")
    plt.tight_layout()
    plt.savefig(fig_file)
    plt.close()
    print(f"Confusion matrix saved to: {fig_file}")


def _get_category_labels(task):
    if task == "binary":
        return ["N", "P"]
    if task == "threeclass":
        return ["N", "U", "P"]
    return [str(i) for i in range(10)]


def _resolve_data_root(project_root, data_root_arg):
    if data_root_arg == "task1":
        return project_root / "Task1-preprocess" / "data" / "task2" / "npz"
    if data_root_arg == "official":
        return project_root / "data_preprocessed_python"
    return Path(data_root_arg)


def _source_name(data_root_arg):
    if data_root_arg in {"task1", "official"}:
        return data_root_arg
    return Path(data_root_arg).name or "custom"


def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    default_output = project_root / "Task3-transfer_learning" / "output"

    parser = argparse.ArgumentParser(description="Pairwise MLP and DANN-MLP transfer")
    parser.add_argument("--data_root", type=str, default="task1",
                        help="task1 | official | custom path")
    parser.add_argument("--output_root", type=Path, default=default_output)
    parser.add_argument("--task", type=str, default="binary",
                        choices=["binary", "threeclass"])
    parser.add_argument("--target", type=str, default="valence",
                        choices=["valence", "arousal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", type=str, default="both",
                        choices=["both", "no_transfer", "dann"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_max", type=float, default=0.3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--pretrain_epochs", type=int, default=20,
                        help="source-only warmup epochs before domain adaptation")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0,
                        help="max gradient norm; use 0 to disable")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="DANN epoch log interval; use 0 to disable")
    args = parser.parse_args()

    set_seed(args.seed)
    data_root = _resolve_data_root(project_root, args.data_root)
    source_name = _source_name(args.data_root)

    print("Data root:", data_root)
    X_all, y_valence, y_arousal = load_data(data_root)
    y_all = y_valence if args.target == "valence" else y_arousal
    print(f"X_all shape: {X_all.shape}")

    pairs = make_domain_similarity_pairs(X_all, y_all, args.task)
    print(f"Generated {len(pairs)} directed subject pairs by CORAL domain similarity")

    output_base = args.output_root / source_name / args.task / "mlp"

    if args.only in {"both", "no_transfer"}:
        run_pairwise_no_transfer_mlp(
            X_all, y_all, pairs, args.task, output_base / "no_transfer",
            args.target, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            seed=args.seed, hidden=args.hidden, latent=args.latent,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )

    if args.only in {"both", "dann"}:
        run_pairwise_dann_mlp(
            X_all, y_all, pairs, args.task, output_base / "with_transfer",
            args.target, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            lambda_max=args.lambda_max, seed=args.seed,
            hidden=args.hidden, latent=args.latent,
            pretrain_epochs=args.pretrain_epochs,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )


if __name__ == "__main__":
    main()
