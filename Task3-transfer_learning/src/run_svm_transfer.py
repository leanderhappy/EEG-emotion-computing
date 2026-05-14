import argparse
from functools import partial
import logging
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "eeg_emotion_codex_mplconfig"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch
from scipy.stats import ks_2samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


SELECTED_CHANNELS = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]
FREQ_BANDS = [(4, 8), (8, 12), (12, 30), (30, 45)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def classify_labels(labels: np.ndarray, task: str) -> np.ndarray:
    if task == "binary":
        return (labels > 5 / 9).astype(int)
    if task == "threeclass":
        return np.digitize(labels, bins=[4 / 9, 6 / 9]).astype(int)
    raise ValueError(f"Unsupported task: {task}")


def extract_de_features(trial_data: np.ndarray, sfreq: float = 128.0) -> np.ndarray:
    freqs, psd = welch(trial_data, fs=sfreq, nperseg=256, axis=1)
    features = []
    for low, high in FREQ_BANDS:
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_power = np.mean(psd[:, idx], axis=1)
        de = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-8)
        features.append(de)
    return np.concatenate(features, axis=0)


def load_single_subject_raw(
    data_path: Path,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    if data_path.suffix == ".dat":
        with data_path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")
        X = data["data"][:, :, 384:]
        baseline = data["data"][:, :, :384].mean(axis=-1, keepdims=True)
        X = X - baseline
        y = data["labels"][:, 0] / 9.0
        subject_id = data_path.stem
        sfreq = 128.0
    elif data_path.suffix == ".npz":
        data = np.load(data_path)
        X = data["X"]
        y = data["valence"] / 9.0
        subject_id = str(data["subject_id"][0]) if "subject_id" in data else data_path.stem
        sfreq = float(data["sfreq"][0]) if "sfreq" in data else 128.0
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    X = X[:, SELECTED_CHANNELS, :]
    return X, y, subject_id, sfreq


def load_single_subject(data_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    X, y, subject_id, sfreq = load_single_subject_raw(data_path)
    features = np.stack([extract_de_features(trial, sfreq=sfreq) for trial in X], axis=0)
    return features, y, subject_id


def extract_windowed_de_features(
    X_trials: np.ndarray,
    y_trials: np.ndarray,
    sfreq: float,
    window_seconds: float,
    window_step_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_size = max(1, int(round(window_seconds * sfreq)))
    window_step = max(1, int(round(window_step_seconds * sfreq)))
    trial_length = X_trials.shape[-1]
    if window_size > trial_length:
        raise ValueError(
            f"window_size={window_size} exceeds trial_length={trial_length}"
        )

    features = []
    window_labels = []
    trial_indices = []

    for trial_idx, (trial_x, trial_y) in enumerate(zip(X_trials, y_trials)):
        for start in range(0, trial_length - window_size + 1, window_step):
            segment = trial_x[:, start : start + window_size]
            features.append(extract_de_features(segment, sfreq=sfreq))
            window_labels.append(trial_y)
            trial_indices.append(trial_idx)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(window_labels),
        np.asarray(trial_indices, dtype=np.int16),
    )


def load_dataset(data_root: Path) -> dict[str, dict[str, np.ndarray]]:
    subject_data = {}
    data_files = sorted(list(data_root.glob("*.dat")) + list(data_root.glob("*.npz")))
    if not data_files:
        raise FileNotFoundError(f"No .dat or .npz files found in {data_root}")

    for data_path in data_files:
        X, y, subject_id = load_single_subject(data_path)
        subject_data[subject_id] = {"X": X, "y": y}

    return subject_data


def load_windowed_dataset(
    data_root: Path,
    window_seconds: float,
    window_step_seconds: float,
) -> dict[str, dict[str, np.ndarray]]:
    subject_data = {}
    data_files = sorted(list(data_root.glob("*.dat")) + list(data_root.glob("*.npz")))
    if not data_files:
        raise FileNotFoundError(f"No .dat or .npz files found in {data_root}")

    for data_path in data_files:
        X_trials, y_trials, subject_id, sfreq = load_single_subject_raw(data_path)
        X_window, y_window, trial_index_window = extract_windowed_de_features(
            X_trials,
            y_trials,
            sfreq=sfreq,
            window_seconds=window_seconds,
            window_step_seconds=window_step_seconds,
        )
        subject_data[subject_id] = {
            "X_window": X_window,
            "y_window": y_window,
            "window_trial_index": trial_index_window,
            "y_trial": y_trials,
            "num_trials": len(y_trials),
        }

    return subject_data


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval_list: list[np.ndarray],
    feature_k: int,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray], SelectKBest]:
    k = min(feature_k, X_train.shape[1])
    selector = SelectKBest(
        partial(mutual_info_classif, random_state=seed),
        k=k,
    )
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_eval_sel = [selector.transform(X_eval) for X_eval in X_eval_list]
    return X_train_sel, X_eval_sel, selector


def select_domain_invariant_features(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    k: int,
    seed: int = 42,
    alpha: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    n_features = X_source.shape[1]
    k = min(k, n_features)

    mi_scores = mutual_info_classif(X_source, y_source, random_state=seed)
    ks_stats = np.array([
        ks_2samp(X_source[:, idx], X_target[:, idx]).statistic
        for idx in range(n_features)
    ])
    ks_stability = 1.0 - ks_stats

    mi_norm = mi_scores / (mi_scores.max() + 1e-8) if mi_scores.max() > 0 else mi_scores
    ks_norm = (
        ks_stability / (ks_stability.max() + 1e-8)
        if ks_stability.max() > 0
        else ks_stability
    )
    combined_scores = alpha * mi_norm + (1.0 - alpha) * ks_norm
    selected_idx = np.argsort(combined_scores)[-k:]
    return np.sort(selected_idx), combined_scores


def build_subspace(
    X: np.ndarray, d: int, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    n, p = X.shape
    d_actual = min(d, n - 1, p)
    mu = X.mean(axis=0)
    X_c = X - mu
    C = (X_c.T @ X_c) / max(n - 1, 1) + eps * np.eye(p)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, idx[:d_actual]]
    return mu, basis


def subspace_align(
    source_basis: np.ndarray, target_basis: np.ndarray
) -> np.ndarray:
    M = source_basis.T @ target_basis
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def asfm_align(
    X_source: np.ndarray, X_target: np.ndarray, d: int, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    mu_s, basis_s = build_subspace(X_source, d, eps)
    mu_t, basis_t = build_subspace(X_target, d, eps)
    R = subspace_align(basis_s, basis_t)
    scores_s = (X_source - mu_s) @ basis_s
    scores_t = (X_target - mu_t) @ basis_t
    # Use the aligned latent coordinates directly; this is more faithful to
    # subspace-alignment style transfer than reconstructing back to the
    # original feature space before classification.
    Xs_aligned = scores_s @ R
    Xt_proj = scores_t
    return Xs_aligned, Xt_proj


def asfm_align_discriminative(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    d: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """ASFM with LDA-guided source subspace to preserve discriminative directions.

    Instead of using PCA (unsupervised) for the source subspace, uses LDA
    directions as the top basis vectors. This ensures the rotation aligns
    discriminative directions rather than just variance directions.
    """
    classes = np.unique(y_source)
    n_classes = len(classes)
    lda_dim = min(n_classes - 1, d)

    # Step 1: LDA on source for discriminative directions
    lda = LinearDiscriminantAnalysis(n_components=lda_dim)
    lda.fit(X_source, y_source)
    lda_dirs = lda.scalings_[:, :lda_dim]  # (p, lda_dim)
    # Orthonormalize LDA directions
    Q_lda, _ = np.linalg.qr(lda_dirs)

    # Step 2: PCA on source for variance directions
    mu_s, basis_s_pca = build_subspace(X_source, d, eps)

    # Step 3: Orthogonalize PCA against LDA to get complementary directions
    proj = basis_s_pca - Q_lda @ (Q_lda.T @ basis_s_pca)
    Q_extra, _ = np.linalg.qr(proj)
    needed = d - lda_dim
    basis_s_guided = np.hstack([Q_lda, Q_extra[:, :needed]])  # (p, d)

    # Step 4: Target subspace (PCA, unsupervised)
    mu_t, basis_t = build_subspace(X_target, d, eps)

    # Step 5: Align guided source subspace to target subspace
    R = subspace_align(basis_s_guided, basis_t)

    scores_s = (X_source - mu_s) @ basis_s_guided
    scores_t = (X_target - mu_t) @ basis_t

    Xs_aligned = scores_s @ R
    Xt_proj = scores_t
    return Xs_aligned, Xt_proj


def compute_fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    """Compute Fisher's discriminant ratio: mean between-class / mean within-class variance."""
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.0
    overall_mean = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    for c in classes:
        X_c = X[y == c]
        n_c = len(X_c)
        mean_c = X_c.mean(axis=0)
        between += n_c * (mean_c - overall_mean) ** 2
        within += ((X_c - mean_c) ** 2).sum(axis=0)
    within = np.maximum(within, 1e-10)
    return float(np.mean(between / within))


def build_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def estimate_source_importance_weights(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int = 42,
    clip_range: tuple[float, float] = (0.25, 4.0),
) -> np.ndarray:
    X_domain = np.vstack([X_source, X_target])
    y_domain = np.concatenate(
        [
            np.zeros(len(X_source), dtype=int),
            np.ones(len(X_target), dtype=int),
        ]
    )
    domain_clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    domain_clf.fit(X_domain, y_domain)
    target_prob = domain_clf.predict_proba(X_source)[:, 1]
    target_prob = np.clip(target_prob, 1e-4, 1 - 1e-4)
    weights = target_prob / (1.0 - target_prob)
    weights = np.clip(weights, clip_range[0], clip_range[1])
    return weights / (weights.mean() + 1e-8)


def select_single_view_balanced_pseudo_labels(
    transfer_proba: np.ndarray,
    selected_mask: np.ndarray,
    tau: float,
    classes: np.ndarray,
    max_per_class: int,
    imbalance_ratio: float = 2.0,
    single_class_cap: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transfer_labels = np.argmax(transfer_proba, axis=1)

    transfer_conf = np.max(transfer_proba, axis=1)

    candidate_mask = (~selected_mask) & (transfer_conf >= tau)
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) == 0:
        return np.array([], dtype=int), transfer_labels, np.array([], dtype=float)

    candidate_labels = transfer_labels[candidate_idx]
    candidate_conf = transfer_conf[candidate_idx]

    class_counts = []
    for c in classes:
        class_counts.append(int(np.sum(candidate_labels == c)))
    positive_counts = [count for count in class_counts if count > 0]
    if not positive_counts:
        return np.array([], dtype=int), transfer_labels, np.array([], dtype=float)

    selected_idx = []
    selected_conf = []
    active_class_count = sum(count > 0 for count in class_counts)
    if active_class_count == 1:
        per_class_limit = min(single_class_cap, max_per_class, max(positive_counts))
        per_class_limits = {
            c: (per_class_limit if count > 0 else 0)
            for c, count in zip(classes, class_counts)
        }
    else:
        minority_cap = min(max_per_class, min(positive_counts))
        majority_cap = max(minority_cap, int(np.floor(imbalance_ratio * minority_cap)))
        per_class_limits = {
            c: (
                min(count, max_per_class, majority_cap)
                if count > 0
                else 0
            )
            for c, count in zip(classes, class_counts)
        }

    for c in classes:
        class_mask = candidate_labels == c
        class_idx = candidate_idx[class_mask]
        class_conf = candidate_conf[class_mask]
        per_class_limit = per_class_limits[c]
        if per_class_limit <= 0:
            continue
        top_k = np.argsort(class_conf)[-per_class_limit:]
        selected_idx.extend(class_idx[top_k].tolist())
        selected_conf.extend(class_conf[top_k].tolist())

    if len(selected_idx) == 0:
        return np.array([], dtype=int), transfer_labels, np.array([], dtype=float)

    return (
        np.array(selected_idx, dtype=int),
        transfer_labels,
        np.array(selected_conf, dtype=float),
    )


def build_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    use_gridsearch: bool = False,
    seed: int = 42,
    probability: bool = False,
    svm_kernel: str = "rbf",
) -> SVC:
    if not use_gridsearch:
        model = SVC(
            kernel=svm_kernel,
            C=1.0,
            gamma="scale" if svm_kernel == "rbf" else "auto",
            class_weight="balanced",
            random_state=seed,
            probability=probability,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    class_counts = np.bincount(y_train)
    valid_counts = class_counts[class_counts > 0]
    if len(valid_counts) < 2 or valid_counts.min() < 2:
        model = SVC(
            kernel=svm_kernel,
            C=1.0,
            gamma="scale" if svm_kernel == "rbf" else "auto",
            class_weight="balanced",
            random_state=seed,
            probability=probability,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)
        return model

    cv_splits = min(3, int(valid_counts.min()))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        estimator=SVC(class_weight="balanced", random_state=seed, probability=probability),
        param_grid=(
            {
                "kernel": ["linear"],
                "C": [0.1, 1.0, 10.0],
            }
            if svm_kernel == "linear"
            else {
                "kernel": ["rbf"],
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", 0.01, 0.1],
            }
        ),
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    grid.fit(X_train, y_train, sample_weight=sample_weight)
    return grid.best_estimator_


def aggregate_window_predictions(
    scores_or_proba: np.ndarray,
    window_trial_index: np.ndarray,
    num_trials: int,
    task: str,
) -> np.ndarray:
    if task == "binary":
        trial_scores = np.zeros(num_trials, dtype=float)
        trial_counts = np.zeros(num_trials, dtype=int)
        np.add.at(trial_scores, window_trial_index, scores_or_proba.ravel())
        np.add.at(trial_counts, window_trial_index, 1)
        trial_scores /= np.maximum(trial_counts, 1)
        return (trial_scores > 0).astype(int)

    num_classes = scores_or_proba.shape[1]
    trial_proba = np.zeros((num_trials, num_classes), dtype=float)
    trial_counts = np.zeros(num_trials, dtype=int)
    for class_idx in range(num_classes):
        np.add.at(trial_proba[:, class_idx], window_trial_index, scores_or_proba[:, class_idx])
    np.add.at(trial_counts, window_trial_index, 1)
    trial_proba /= np.maximum(trial_counts[:, None], 1)
    return np.argmax(trial_proba, axis=1)


def subsample_rows(
    X: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if len(X) <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx]


def compute_proxy_a_distance(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
    max_samples: int = 400,
) -> float:
    Xs = subsample_rows(X_source, max_samples=max_samples, seed=seed)
    Xt = subsample_rows(X_target, max_samples=max_samples, seed=seed + 1)
    X = np.vstack([Xs, Xt])
    y = np.concatenate(
        [
            np.zeros(len(Xs), dtype=int),
            np.ones(len(Xt), dtype=int),
        ]
    )
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=seed,
    )
    clf = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    test_err = 1.0 - test_acc
    return max(0.0, 2.0 * (1.0 - 2.0 * test_err))


def compute_coral_distance(
    X_source: np.ndarray,
    X_target: np.ndarray,
    seed: int,
    max_samples: int = 400,
    reg: float = 1e-6,
) -> float:
    Xs = subsample_rows(X_source, max_samples=max_samples, seed=seed)
    Xt = subsample_rows(X_target, max_samples=max_samples, seed=seed + 1)
    X_joint = np.vstack([Xs, Xt])
    scaler = StandardScaler()
    X_joint = scaler.fit_transform(X_joint)
    Xs_std = X_joint[: len(Xs)]
    Xt_std = X_joint[len(Xs) :]

    mu_dist = np.linalg.norm(Xs_std.mean(axis=0) - Xt_std.mean(axis=0))
    cov_s = np.cov(Xs_std, rowvar=False) + reg * np.eye(Xs_std.shape[1])
    cov_t = np.cov(Xt_std, rowvar=False) + reg * np.eye(Xt_std.shape[1])
    cov_dist = np.linalg.norm(cov_s - cov_t, ord="fro") / Xs_std.shape[1]
    return float(mu_dist + cov_dist)


def select_best_source_for_target_similarity(
    candidate_sources: list[str],
    target_subject: str,
    source_feature_map: dict[str, np.ndarray],
    seed: int,
    max_samples: int = 400,
) -> tuple[str, list[dict[str, float | str]]]:
    selection_rows = []
    for idx, source_subject in enumerate(candidate_sources):
        X_source = source_feature_map[source_subject]
        X_target = source_feature_map[target_subject]
        proxy_a = compute_proxy_a_distance(
            X_source,
            X_target,
            seed=seed + idx * 13,
            max_samples=max_samples,
        )
        coral_dist = compute_coral_distance(
            X_source,
            X_target,
            seed=seed + idx * 17,
            max_samples=max_samples,
        )
        selection_rows.append(
            {
                "source_subject": source_subject,
                "target_subject": target_subject,
                "proxy_a_distance": proxy_a,
                "coral_distance": coral_dist,
            }
        )

    selection_df = pd.DataFrame(selection_rows)
    selection_df["proxy_rank"] = selection_df["proxy_a_distance"].rank(method="dense")
    selection_df["coral_rank"] = selection_df["coral_distance"].rank(method="dense")
    selection_df["selection_score"] = selection_df["proxy_rank"] + selection_df["coral_rank"]
    selection_df = selection_df.sort_values(
        ["selection_score", "proxy_a_distance", "coral_distance", "source_subject"],
        ascending=[True, True, True, True],
    )
    best_source = str(selection_df.iloc[0]["source_subject"])
    return best_source, selection_df.to_dict("records")


def compute_trial_coherence(
    window_labels: np.ndarray,
    window_trial_index: np.ndarray,
    num_trials: int,
) -> float:
    if len(window_labels) == 0:
        return 0.0
    coherence_scores = []
    for trial_idx in range(num_trials):
        trial_mask = window_trial_index == trial_idx
        if not np.any(trial_mask):
            continue
        labels = window_labels[trial_mask]
        counts = np.bincount(labels, minlength=int(labels.max()) + 1)
        coherence_scores.append(counts.max() / max(len(labels), 1))
    if not coherence_scores:
        return 0.0
    return float(np.mean(coherence_scores))


def probe_asfm_transferability(
    X_source_window: np.ndarray,
    y_source_window: np.ndarray,
    X_target_window: np.ndarray,
    target_window_trial_index: np.ndarray,
    num_target_trials: int,
    feature_k: int,
    subspace_d: int,
    tau: float,
    seed: int,
    max_pseudo_per_class: int,
    pseudo_imbalance_ratio: float,
    pseudo_single_class_cap: int,
    source_weight_clip_min: float,
    source_weight_clip_max: float,
) -> dict[str, float]:
    scaler = StandardScaler()
    X_source_std = scaler.fit_transform(X_source_window)
    X_target_std = scaler.transform(X_target_window)

    # Use DI (domain-invariant) feature selection: MI discriminability + KS domain stability
    selected_idx_fs, _ = select_domain_invariant_features(
        X_source_std,
        y_source_window,
        X_target_std,
        k=feature_k,
        seed=seed,
        alpha=0.5,
    )
    X_source_sel = X_source_std[:, selected_idx_fs]
    X_target_sel = X_target_std[:, selected_idx_fs]

    X_source_asfm, X_target_asfm = asfm_align(
        X_source_sel,
        X_target_sel,
        d=subspace_d,
    )

    source_weights = estimate_source_importance_weights(
        X_source_asfm,
        X_target_asfm,
        seed=seed,
        clip_range=(source_weight_clip_min, source_weight_clip_max),
    )
    transfer_view_model = build_lr(
        X_source_asfm,
        y_source_window,
        sample_weight=source_weights,
        seed=seed,
    )

    transfer_proba = transfer_view_model.predict_proba(X_target_asfm)
    selected_idx, pseudo_labels, selected_conf = select_single_view_balanced_pseudo_labels(
        transfer_proba,
        selected_mask=np.zeros(len(X_target_asfm), dtype=bool),
        tau=tau,
        classes=np.unique(y_source_window),
        max_per_class=max_pseudo_per_class,
        imbalance_ratio=pseudo_imbalance_ratio,
        single_class_cap=pseudo_single_class_cap,
    )

    transfer_labels = np.argmax(transfer_proba, axis=1)
    transfer_conf = np.max(transfer_proba, axis=1)
    trial_coherence = compute_trial_coherence(
        transfer_labels,
        target_window_trial_index,
        num_target_trials,
    )

    candidate_mask = (transfer_conf >= tau)
    candidate_coverage = float(np.mean(candidate_mask))
    agreement = 1.0

    num_classes = len(np.unique(y_source_window))
    pseudo_cap = max(1, max_pseudo_per_class * num_classes)
    selected_fraction_of_cap = min(len(selected_idx) / pseudo_cap, 1.0)

    if len(selected_idx) == 0:
        pseudo_balance = 0.0
        mean_selected_conf = 0.0
    else:
        selected_labels = pseudo_labels[selected_idx]
        counts = np.bincount(selected_labels, minlength=num_classes)
        positive_counts = counts[counts > 0]
        pseudo_balance = (
            float(positive_counts.min() / positive_counts.max())
            if len(positive_counts) >= 2
            else 0.0
        )
        mean_selected_conf = float(np.mean(selected_conf))

    transferability_score = (
        0.15 * selected_fraction_of_cap
        + 0.20 * candidate_coverage
        + 0.20 * pseudo_balance
        + 0.20 * agreement
        + 0.20 * trial_coherence
        + 0.05 * mean_selected_conf
    )

    return {
        "probe_selected_count": float(len(selected_idx)),
        "probe_selected_fraction_of_cap": float(selected_fraction_of_cap),
        "probe_candidate_coverage": candidate_coverage,
        "probe_pseudo_balance": pseudo_balance,
        "probe_agreement": agreement,
        "probe_trial_coherence": trial_coherence,
        "probe_mean_selected_conf": mean_selected_conf,
        "probe_transferability_score": float(transferability_score),
    }


def select_transferable_source_for_target(
    candidate_sources: list[str],
    target_subject: str,
    source_feature_map: dict[str, np.ndarray],
    window_subject_data: dict[str, dict[str, np.ndarray]],
    task: str,
    feature_k: int,
    subspace_d: int,
    tau: float,
    seed: int,
    coarse_topk: int,
    max_samples: int,
    max_pseudo_per_class: int,
    pseudo_imbalance_ratio: float,
    pseudo_single_class_cap: int,
    source_weight_clip_min: float,
    source_weight_clip_max: float,
) -> tuple[str, list[dict[str, float | str]]]:
    _, stage1_rows = select_best_source_for_target_similarity(
        candidate_sources,
        target_subject,
        source_feature_map,
        seed=seed,
        max_samples=max_samples,
    )
    stage1_df = pd.DataFrame(stage1_rows)
    stage1_df = stage1_df.sort_values(
        ["selection_score", "proxy_a_distance", "coral_distance", "source_subject"],
        ascending=[True, True, True, True],
    )
    candidate_df = stage1_df.head(min(coarse_topk, len(stage1_df))).copy()

    target_window_view = window_subject_data[target_subject]
    X_target_window = target_window_view["X_window"]
    target_window_trial_index = target_window_view["window_trial_index"]
    num_target_trials = int(target_window_view["num_trials"])

    probe_rows = []
    for idx, row in candidate_df.iterrows():
        source_subject = str(row["source_subject"])
        source_window_view = window_subject_data[source_subject]
        X_source_window = source_window_view["X_window"]
        y_source_window = classify_labels(source_window_view["y_window"], task)
        probe_metrics = probe_asfm_transferability(
            X_source_window,
            y_source_window,
            X_target_window,
            target_window_trial_index,
            num_target_trials,
            feature_k=feature_k,
            subspace_d=subspace_d,
            tau=tau,
            seed=seed + idx * 31,
            max_pseudo_per_class=max_pseudo_per_class,
            pseudo_imbalance_ratio=pseudo_imbalance_ratio,
            pseudo_single_class_cap=pseudo_single_class_cap,
            source_weight_clip_min=source_weight_clip_min,
            source_weight_clip_max=source_weight_clip_max,
        )
        merged = row.to_dict()
        merged.update(probe_metrics)
        probe_rows.append(merged)

    probe_df = pd.DataFrame(probe_rows)
    if probe_df.empty:
        best_source = str(stage1_df.iloc[0]["source_subject"])
        stage1_df["selected_source"] = best_source
        stage1_df["selection_stage"] = "stage1_only"
        return best_source, stage1_df.to_dict("records")

    probe_df["selection_stage"] = "stage2_probe"
    probe_df = probe_df.sort_values(
        [
            "probe_transferability_score",
            "probe_trial_coherence",
            "probe_agreement",
            "probe_pseudo_balance",
            "selection_score",
            "source_subject",
        ],
        ascending=[False, False, False, False, True, True],
    )
    best_source = str(probe_df.iloc[0]["source_subject"])

    remaining_df = stage1_df[~stage1_df["source_subject"].isin(probe_df["source_subject"])].copy()
    if not remaining_df.empty:
        remaining_df["selection_stage"] = "stage1_pruned"
    full_df = pd.concat([probe_df, remaining_df], ignore_index=True, sort=False)
    full_df["selected_source"] = best_source
    return best_source, full_df.to_dict("records")


def run_source_only_windowed(
    X_source_window: np.ndarray,
    y_source_window: np.ndarray,
    X_target_window: np.ndarray,
    target_window_trial_index: np.ndarray,
    num_target_trials: int,
    feature_k: int,
    use_gridsearch: bool,
    seed: int,
    task: str = "binary",
) -> np.ndarray:
    scaler = StandardScaler()
    X_source_std = scaler.fit_transform(X_source_window)
    X_target_std = scaler.transform(X_target_window)

    X_source_sel, [X_target_sel], _ = select_features(
        X_source_std,
        y_source_window,
        [X_target_std],
        feature_k=feature_k,
        seed=seed,
    )
    model = build_svm(
        X_source_sel,
        y_source_window,
        sample_weight=None,
        use_gridsearch=use_gridsearch,
        seed=seed,
        svm_kernel="linear",
        probability=(task != "binary"),
    )

    if task == "binary":
        window_scores = model.decision_function(X_target_sel).reshape(-1, 1)
        return aggregate_window_predictions(
            window_scores,
            target_window_trial_index,
            num_target_trials,
            task=task,
        )

    window_proba = model.predict_proba(X_target_sel)
    return aggregate_window_predictions(
        window_proba,
        target_window_trial_index,
        num_target_trials,
        task=task,
    )


def run_source_only(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_query: np.ndarray,
    feature_k: int,
    use_gridsearch: bool,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    X_source_std = scaler.fit_transform(X_source)
    X_query_std = scaler.transform(X_query)

    X_source_sel, [X_query_sel], _ = select_features(
        X_source_std,
        y_source,
        [X_query_std],
        feature_k=feature_k,
        seed=seed,
    )
    model = build_svm(
        X_source_sel,
        y_source,
        sample_weight=None,
        use_gridsearch=use_gridsearch,
        seed=seed,
    )
    return model.predict(X_query_sel)


def run_transfer_asfm(
    X_source_window: np.ndarray,
    y_source_window: np.ndarray,
    X_target_window: np.ndarray,
    target_window_trial_index: np.ndarray,
    num_target_trials: int,
    feature_k: int,
    use_gridsearch: bool,
    seed: int,
    subspace_d: int = 22,
    task: str = "binary",
    num_pl_iterations: int = 3,
    pseudo_tau: float = 0.75,
    pseudo_tau_decay: float = 0.05,
    pseudo_tau_min: float = 0.55,
    pseudo_weight: float = 0.5,
    source_weight_clip_min: float = 0.5,
    source_weight_clip_max: float = 2.0,
    max_pseudo_per_class: int = 30,
    pseudo_imbalance_ratio: float = 2.0,
    pseudo_single_class_cap: int = 8,
) -> tuple[np.ndarray, int]:
    scaler = StandardScaler()
    X_source_std = scaler.fit_transform(X_source_window)
    X_target_std = scaler.transform(X_target_window)

    # Use DI (domain-invariant) feature selection: MI discriminability + KS domain stability
    selected_idx, _ = select_domain_invariant_features(
        X_source_std,
        y_source_window,
        X_target_std,
        k=feature_k,
        seed=seed,
        alpha=0.5,
    )
    X_source_sel = X_source_std[:, selected_idx]
    X_target_sel = X_target_std[:, selected_idx]

    # Diagnose: proxy_a_distance before alignment
    proxy_a_before = compute_proxy_a_distance(
        X_source_sel, X_target_sel, seed=seed, max_samples=400,
    )

    # Direct SVD rotation alignment (no PCA subspace).
    # Instead of projecting to PCA subspace then rotating, directly compute
    # the optimal rotation matrix from the cross-covariance of the selected
    # features. This avoids information loss from PCA dimensionality reduction.
    X_s_centered = X_source_sel - X_source_sel.mean(axis=0)
    X_t_centered = X_target_sel - X_target_sel.mean(axis=0)
    M = X_s_centered.T @ X_t_centered
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    X_source_aligned = X_source_sel @ R
    X_target_aligned = X_target_sel  # target stays as-is

    # Diagnose: proxy_a_distance after alignment
    proxy_a_after = compute_proxy_a_distance(
        X_source_aligned, X_target_aligned, seed=seed, max_samples=400,
    )
    logging.info(
        "    proxy_a_distance: %.4f -> %.4f (delta=%.4f)",
        proxy_a_before, proxy_a_after, proxy_a_after - proxy_a_before,
    )

    # Score blending: train two separate SVMs (original + aligned), average scores.
    # This avoids expanding the feature space and lets each SVM operate in its
    # own optimal feature space.

    source_weights = estimate_source_importance_weights(
        X_source_aligned,
        X_target_aligned,
        seed=seed,
        clip_range=(source_weight_clip_min, source_weight_clip_max),
    )

    # Model 1: SVM on aligned features (with importance weights)
    model_aligned = build_svm(
        X_source_aligned,
        y_source_window,
        sample_weight=source_weights,
        use_gridsearch=use_gridsearch,
        seed=seed,
        svm_kernel="linear",
        probability=(task != "binary"),
    )
    # Model 2: SVM on original DI features (no weights)
    model_original = build_svm(
        X_source_sel,
        y_source_window,
        sample_weight=None,
        use_gridsearch=use_gridsearch,
        seed=seed + 1,
        svm_kernel="linear",
        probability=(task != "binary"),
    )

    total_pseudo = 0
    logging.info(
        "    Final training set: %d source windows (no pseudo-labeling, score blending)",
        len(y_source_window),
    )

    if task == "binary":
        scores_aligned = model_aligned.decision_function(X_target_aligned)
        scores_original = model_original.decision_function(X_target_sel)
        # Normalize scores to comparable ranges, then average
        s_a = (scores_aligned - scores_aligned.mean()) / (scores_aligned.std() + 1e-8)
        s_o = (scores_original - scores_original.mean()) / (scores_original.std() + 1e-8)
        blended_scores = 0.5 * s_a + 0.5 * s_o
        trial_pred = aggregate_window_predictions(
            blended_scores.reshape(-1, 1),
            target_window_trial_index,
            num_target_trials,
            task=task,
        )
    else:
        proba_aligned = model_aligned.predict_proba(X_target_aligned)
        proba_original = model_original.predict_proba(X_target_sel)
        blended_proba = 0.5 * proba_aligned + 0.5 * proba_original
        trial_pred = aggregate_window_predictions(
            blended_proba,
            target_window_trial_index,
            num_target_trials,
            task=task,
        )
    return trial_pred, total_pseudo


def summarize_pair_results(records: list[dict]) -> pd.DataFrame:
    pair_df = pd.DataFrame(records)
    if pair_df.empty:
        return pair_df

    agg_dict = {
        "source_only_acc": ("source_only_acc", "mean"),
        "transfer_acc": ("transfer_acc", "mean"),
        "gain": ("gain", "mean"),
        "target_size": ("target_size", "mean"),
    }
    if "num_pseudo_labeled" in pair_df.columns:
        agg_dict["num_pseudo_labeled"] = ("num_pseudo_labeled", "mean")
    if "method" in pair_df.columns:
        agg_dict["method"] = ("method", "first")
    summary = (
        pair_df.groupby(["source_subject", "target_subject"], as_index=False)
        .agg(**agg_dict)
        .sort_values(["gain", "transfer_acc"], ascending=[False, False])
    )
    return summary


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("pred")
    plt.ylabel("real")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def configure_logging(output_dir: Path) -> None:
    log_file = output_dir / "results.txt"
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def resolve_data_root(project_root: Path, data_root_arg: str) -> tuple[Path, str]:
    if data_root_arg == "task1":
        return project_root / "Task1-preprocess" / "data" / "task2" / "npz", "task1-preprocess"
    if data_root_arg == "official":
        return project_root / "data_preprocessed_python", "official-preprocess"
    return Path(data_root_arg), "others"


def make_output_dir(
    output_root: Path,
    data_dir: str,
    task: str,
    pair_tag: str,
    method: str = "asfm",
    split_variant: str = "full-target-unlabeled",
) -> Path:
    split_dir = split_variant
    pair_dir = pair_tag
    output_dir = output_root / data_dir / task / f"svm-{method}" / split_dir / pair_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser("Cross-subject transfer learning with ASFM + SVM")
    parser.add_argument("--data_root", type=str, default="task1",choices=["task1", "official"])
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "Task3-transfer_learning" / "output",
    )
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "threeclass"])
    parser.add_argument("--feature_k", type=int, default=18)
    parser.add_argument("--asfm_d", type=int, default=22)
    parser.add_argument("--asfm_num_pl_iterations", type=int, default=3)
    parser.add_argument("--asfm_pseudo_tau", type=float, default=0.75)
    parser.add_argument("--asfm_pseudo_tau_decay", type=float, default=0.05)
    parser.add_argument("--asfm_pseudo_tau_min", type=float, default=0.55)
    parser.add_argument("--asfm_pseudo_weight", type=float, default=0.5)
    parser.add_argument("--asfm_max_pseudo_per_class", type=int, default=30)
    parser.add_argument("--asfm_pseudo_imbalance_ratio", type=float, default=2.0)
    parser.add_argument("--asfm_pseudo_single_class_cap", type=int, default=8)
    parser.add_argument("--asfm_source_weight_clip_min", type=float, default=0.5)
    parser.add_argument("--asfm_source_weight_clip_max", type=float, default=2.0)
    parser.add_argument("--asfm_window_seconds", type=float, default=4.0)
    parser.add_argument("--asfm_window_step_seconds", type=float, default=2.0)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--num_random_targets", type=int, default=32)
    parser.add_argument("--source_select_max_samples", type=int, default=400)
    parser.add_argument("--source_select_coarse_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_subject", type=str, default=None)
    parser.add_argument("--target_subject", type=str, default=None)
    parser.add_argument("--use_gridsearch", action="store_true", default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    data_root, data_dir = resolve_data_root(PROJECT_ROOT, args.data_root)
    subject_data = load_dataset(data_root)
    window_subject_data = load_windowed_dataset(
        data_root,
        window_seconds=args.asfm_window_seconds,
        window_step_seconds=args.asfm_window_step_seconds,
    )
    subjects = sorted(subject_data.keys())

    source_feature_map = {}
    for subject_id in subjects:
        source_feature_map[subject_id] = window_subject_data[subject_id]["X_window"]

    selected_targets: list[str] = []
    pair_list: list[tuple[str, str]] = []
    source_selection_rows: list[dict[str, float | str]] = []

    if args.source_subject and args.target_subject:
        pair_list = [(args.source_subject, args.target_subject)]
        selected_targets = [args.target_subject]
        sampling_mode = "manual_pair"
        pair_tag = f"{args.source_subject}-to-{args.target_subject}"
    else:
        if args.target_subject:
            selected_targets = [args.target_subject]
            sampling_mode = "single_target_best_source"
            pair_tag = f"best-source-to-{args.target_subject}"
        else:
            selected_targets = list(subjects)
            sampling_mode = "loto_best_source"
            pair_tag = "loto-best-source"

        for target_idx, target_subject in enumerate(selected_targets, start=1):
            if args.source_subject:
                best_source = args.source_subject
                selection_records = [
                    {
                        "source_subject": best_source,
                        "target_subject": target_subject,
                        "proxy_a_distance": np.nan,
                        "coral_distance": np.nan,
                        "proxy_rank": np.nan,
                        "coral_rank": np.nan,
                        "selection_score": np.nan,
                    }
                ]
            else:
                candidate_sources = [s for s in subjects if s != target_subject]
                best_source, selection_records = select_transferable_source_for_target(
                    candidate_sources,
                    target_subject,
                    source_feature_map,
                    window_subject_data,
                    task=args.task,
                    feature_k=args.feature_k,
                    subspace_d=args.asfm_d,
                    tau=args.asfm_pseudo_tau,
                    seed=args.seed + target_idx * 100,
                    coarse_topk=args.source_select_coarse_topk,
                    max_samples=args.source_select_max_samples,
                    max_pseudo_per_class=args.asfm_max_pseudo_per_class,
                    pseudo_imbalance_ratio=args.asfm_pseudo_imbalance_ratio,
                    pseudo_single_class_cap=args.asfm_pseudo_single_class_cap,
                    source_weight_clip_min=args.asfm_source_weight_clip_min,
                    source_weight_clip_max=args.asfm_source_weight_clip_max,
                )
            pair_list.append((best_source, target_subject))
            for row in selection_records:
                row["selected_source"] = best_source
                source_selection_rows.append(row)

    if not pair_list:
        raise ValueError("No subject pairs matched the provided filters.")

    output_dir = make_output_dir(
        args.output_root,
        data_dir,
        args.task,
        pair_tag,
        method="asfm",
        split_variant=f"window-{args.asfm_window_seconds:g}s-step-{args.asfm_window_step_seconds:g}s-trial-agg",
    )
    configure_logging(output_dir)

    logging.info("========== experiment setup ==========")
    logging.info("data_root: %s", data_root)
    logging.info("task: %s", args.task)
    logging.info("model_type: svm")
    logging.info("method: asfm")
    logging.info("transfer_method: window-level ASFM + trial-level aggregation")
    logging.info("asfm_d: %s", args.asfm_d)
    logging.info("asfm_num_pl_iterations: %s", args.asfm_num_pl_iterations)
    logging.info("asfm_pseudo_tau: %s", args.asfm_pseudo_tau)
    logging.info("asfm_pseudo_tau_decay: %s", args.asfm_pseudo_tau_decay)
    logging.info("asfm_pseudo_tau_min: %s", args.asfm_pseudo_tau_min)
    logging.info("asfm_pseudo_weight: %s", args.asfm_pseudo_weight)
    logging.info("asfm_max_pseudo_per_class: %s", args.asfm_max_pseudo_per_class)
    logging.info("asfm_pseudo_imbalance_ratio: %s", args.asfm_pseudo_imbalance_ratio)
    logging.info("asfm_pseudo_single_class_cap: %s", args.asfm_pseudo_single_class_cap)
    logging.info(
        "asfm_source_weight_clip: [%s, %s]",
        args.asfm_source_weight_clip_min,
        args.asfm_source_weight_clip_max,
    )
    logging.info("asfm_window_seconds: %s", args.asfm_window_seconds)
    logging.info("asfm_window_step_seconds: %s", args.asfm_window_step_seconds)
    logging.info("feature_k: %s", args.feature_k)
    logging.info("num_repeats: %s", args.num_repeats)
    logging.info("evaluation_mode: LOTO (all %d subjects as targets)", len(selected_targets))
    logging.info("source_select_max_samples: %s", args.source_select_max_samples)
    logging.info("source_select_coarse_topk: %s", args.source_select_coarse_topk)
    logging.info("use_gridsearch: %s", args.use_gridsearch)
    logging.info("subject_count: %s", len(subjects))
    logging.info("target_label_usage: labels are used only for final evaluation")

    logging.info("pair_sampling_mode: %s", sampling_mode)
    if selected_targets:
        logging.info("selected_targets: %s", selected_targets)
    logging.info("selected_pairs: %s", pair_list)

    all_source_true = []
    all_source_pred = []
    all_transfer_true = []
    all_transfer_pred = []
    repeat_records = []

    for pair_idx, (source_subject, target_subject) in enumerate(pair_list, start=1):
        logging.info("========== pair %s/%s: %s -> %s ==========",
                     pair_idx, len(pair_list), source_subject, target_subject)
        y_target = classify_labels(subject_data[target_subject]["y"], args.task)
        source_window_view = window_subject_data[source_subject]
        target_window_view = window_subject_data[target_subject]
        X_source_window = source_window_view["X_window"]
        y_source_window = classify_labels(source_window_view["y_window"], args.task)
        X_target_window = target_window_view["X_window"]
        target_window_trial_index = target_window_view["window_trial_index"]
        num_target_trials = int(target_window_view["num_trials"])

        for repeat in range(args.num_repeats):
            run_seed = args.seed + pair_idx * 1000 + repeat

            source_pred = run_source_only_windowed(
                X_source_window,
                y_source_window,
                X_target_window,
                target_window_trial_index,
                num_target_trials,
                feature_k=args.feature_k,
                use_gridsearch=args.use_gridsearch,
                seed=run_seed,
                task=args.task,
            )
            transfer_pred, num_pseudo = run_transfer_asfm(
                X_source_window,
                y_source_window,
                X_target_window,
                target_window_trial_index,
                num_target_trials,
                feature_k=args.feature_k,
                use_gridsearch=args.use_gridsearch,
                seed=run_seed,
                subspace_d=args.asfm_d,
                task=args.task,
                num_pl_iterations=args.asfm_num_pl_iterations,
                pseudo_tau=args.asfm_pseudo_tau,
                pseudo_tau_decay=args.asfm_pseudo_tau_decay,
                pseudo_tau_min=args.asfm_pseudo_tau_min,
                pseudo_weight=args.asfm_pseudo_weight,
                max_pseudo_per_class=args.asfm_max_pseudo_per_class,
                pseudo_imbalance_ratio=args.asfm_pseudo_imbalance_ratio,
                pseudo_single_class_cap=args.asfm_pseudo_single_class_cap,
                source_weight_clip_min=args.asfm_source_weight_clip_min,
                source_weight_clip_max=args.asfm_source_weight_clip_max,
            )

            source_acc = accuracy_score(y_target, source_pred)
            transfer_acc = accuracy_score(y_target, transfer_pred)
            gain = transfer_acc - source_acc

            repeat_records.append(
                {
                    "source_subject": source_subject,
                    "target_subject": target_subject,
                    "repeat": repeat,
                    "target_size": len(y_target),
                    "source_only_acc": source_acc,
                    "transfer_acc": transfer_acc,
                    "gain": gain,
                    "num_pseudo_labeled": num_pseudo,
                }
            )

            all_source_true.extend(y_target.tolist())
            all_source_pred.extend(source_pred.tolist())
            all_transfer_true.extend(y_target.tolist())
            all_transfer_pred.extend(transfer_pred.tolist())

    repeat_df = pd.DataFrame(repeat_records)
    pair_summary_df = summarize_pair_results(repeat_records)

    repeat_csv = output_dir / "repeat_results.csv"
    pair_csv = output_dir / "pair_summary.csv"
    repeat_df.to_csv(repeat_csv, index=False)
    pair_summary_df.to_csv(pair_csv, index=False)
    source_selection_csv = None
    if source_selection_rows:
        source_selection_csv = output_dir / "source_selection_summary.csv"
        pd.DataFrame(source_selection_rows).to_csv(source_selection_csv, index=False)

    if args.task == "binary":
        labels = ["N", "P"]
    else:
        labels = ["N", "U", "P"]

    source_cm_path = output_dir / "source_only_confusion_matrix.png"
    transfer_cm_path = output_dir / "transfer_confusion_matrix.png"
    save_confusion_matrix(
        np.array(all_source_true),
        np.array(all_source_pred),
        labels,
        source_cm_path,
        "Source-only Confusion Matrix",
    )
    save_confusion_matrix(
        np.array(all_transfer_true),
        np.array(all_transfer_pred),
        labels,
        transfer_cm_path,
        "Transfer Confusion Matrix",
    )

    logging.info("========== summary ==========")
    logging.info("pair_count: %s", len(pair_list))
    logging.info("source_only_mean_acc: %.4f", pair_summary_df["source_only_acc"].mean())
    logging.info("transfer_mean_acc: %.4f", pair_summary_df["transfer_acc"].mean())
    logging.info("transfer_gain_vs_source_only: %.4f", pair_summary_df["gain"].mean())
    logging.info("transfer_win_rate_vs_source_only: %.4f", (pair_summary_df["gain"] > 0).mean())
    if "num_pseudo_labeled" in pair_summary_df.columns:
        logging.info("mean_pseudo_labeled_per_pair: %.1f", pair_summary_df["num_pseudo_labeled"].mean())
    logging.info("========== source-only report ==========")
    logging.info(
        "\n%s",
        classification_report(
            all_source_true,
            all_source_pred,
            target_names=labels,
            zero_division=0,
        ),
    )
    logging.info("========== transfer report ==========")
    logging.info(
        "\n%s",
        classification_report(
            all_transfer_true,
            all_transfer_pred,
            target_names=labels,
            zero_division=0,
        ),
    )

    if not pair_summary_df.empty:
        logging.info("========== best pairs ==========")
        logging.info("%s", pair_summary_df.head(5).to_string(index=False))
        logging.info("========== worst pairs ==========")
        logging.info("%s", pair_summary_df.tail(5).to_string(index=False))

    logging.info("repeat_results_csv: %s", repeat_csv)
    logging.info("pair_summary_csv: %s", pair_csv)
    if source_selection_csv is not None:
        logging.info("source_selection_summary_csv: %s", source_selection_csv)
    logging.info("source_only_confusion_matrix: %s", source_cm_path)
    logging.info("transfer_confusion_matrix: %s", transfer_cm_path)


if __name__ == "__main__":
    main()
