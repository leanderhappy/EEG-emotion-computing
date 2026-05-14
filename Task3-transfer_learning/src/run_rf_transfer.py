import argparse
import logging
import os
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from run_svm_transfer import (
    PROJECT_ROOT as _,
    classify_labels,
    compute_proxy_a_distance,
    configure_logging,
    estimate_source_importance_weights,
    load_dataset,
    load_windowed_dataset,
    resolve_data_root,
    select_domain_invariant_features,
    select_features,
    select_transferable_source_for_target,
    set_seed,
    aggregate_window_predictions,
    save_confusion_matrix,
    summarize_pair_results,
)


# ---------------------------------------------------------------------------
# RF-specific functions
# ---------------------------------------------------------------------------

def build_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
    use_gridsearch: bool = False,
    max_depth: int = 10,
    min_samples_split: int = 5,
) -> RandomForestClassifier:
    """Build RF with regularization to prevent overfitting on small source data.

    Key differences from previous version:
    - max_depth=10 (was None) to limit tree complexity
    - class_weight only when sample_weight is None to avoid double-weighting
    """
    use_class_weight = sample_weight is None

    if use_gridsearch:
        class_counts = np.bincount(y_train)
        valid_counts = class_counts[class_counts > 0]
        if len(valid_counts) < 2 or valid_counts.min() < 2:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=max_depth, max_features="sqrt",
                min_samples_split=min_samples_split, random_state=seed,
                class_weight="balanced" if use_class_weight else None,
                n_jobs=-1,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
            return model

        cv_splits = min(3, int(valid_counts.min()))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, 20],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
        }
        base = RandomForestClassifier(
            random_state=seed,
            class_weight="balanced" if use_class_weight else None,
            n_jobs=-1,
        )
        gs = GridSearchCV(
            base, param_grid, cv=cv, scoring="balanced_accuracy",
            n_jobs=-1, refit=True, verbose=0,
        )
        gs.fit(X_train, y_train, sample_weight=sample_weight)
        return gs.best_estimator_

    model = RandomForestClassifier(
        n_estimators=200, max_depth=max_depth, max_features="sqrt",
        min_samples_split=min_samples_split, random_state=seed,
        class_weight="balanced" if use_class_weight else None,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def run_source_only_rf(
    X_source_window: np.ndarray,
    y_source_window: np.ndarray,
    X_target_window: np.ndarray,
    target_window_trial_index: np.ndarray,
    num_target_trials: int,
    feature_k: int,
    use_gridsearch: bool,
    seed: int,
    task: str = "binary",
    max_depth: int = 10,
    min_samples_split: int = 5,
    di_alpha: float = 0.5,
) -> np.ndarray:
    scaler = StandardScaler()
    X_source_std = scaler.fit_transform(X_source_window)
    X_target_std = scaler.transform(X_target_window)

    selected_idx, _ = select_domain_invariant_features(
        X_source_std, y_source_window, X_target_std, k=feature_k, seed=seed, alpha=di_alpha,
    )
    X_source_sel = X_source_std[:, selected_idx]
    X_target_sel = X_target_std[:, selected_idx]

    model = build_rf(X_source_sel, y_source_window, seed=seed, use_gridsearch=use_gridsearch,
                     max_depth=max_depth, min_samples_split=min_samples_split)

    if task == "binary":
        # predict_proba returns [0,1]; shift by -0.5 so aggregate_window_predictions
        # threshold at >0 works correctly (0.5 probability -> 0)
        proba = model.predict_proba(X_target_sel)[:, 1]
        scores = (proba - 0.5).reshape(-1, 1)
        return aggregate_window_predictions(
            scores, target_window_trial_index, num_target_trials, task=task,
        )

    proba = model.predict_proba(X_target_sel)
    return aggregate_window_predictions(
        proba, target_window_trial_index, num_target_trials, task=task,
    )


def run_transfer_asfm_rf(
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
    source_weight_clip_min: float = 0.5,
    source_weight_clip_max: float = 2.0,
    max_depth: int = 10,
    min_samples_split: int = 5,
    di_alpha: float = 0.5,
) -> tuple[np.ndarray, dict]:
    # Step 1: Standardize
    scaler = StandardScaler()
    X_source_std = scaler.fit_transform(X_source_window)
    X_target_std = scaler.transform(X_target_window)

    # Step 2: DI feature selection (MI + KS)
    selected_idx, _ = select_domain_invariant_features(
        X_source_std, y_source_window, X_target_std, k=feature_k, seed=seed, alpha=di_alpha,
    )
    X_source_sel = X_source_std[:, selected_idx]
    X_target_sel = X_target_std[:, selected_idx]

    # Step 3: SVD rotation alignment (same as SVM)
    X_s_centered = X_source_sel - X_source_sel.mean(axis=0)
    X_t_centered = X_target_sel - X_target_sel.mean(axis=0)
    M = X_s_centered.T @ X_t_centered
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    X_source_aligned = X_source_sel @ R
    X_target_aligned = X_target_sel

    # Diagnose: proxy_a_distance
    proxy_a_before = compute_proxy_a_distance(X_source_sel, X_target_sel, seed=seed, max_samples=400)
    proxy_a_after = compute_proxy_a_distance(X_source_aligned, X_target_aligned, seed=seed, max_samples=400)
    logging.info(
        "    proxy_a_distance: %.4f -> %.4f (delta=%.4f)",
        proxy_a_before, proxy_a_after, proxy_a_after - proxy_a_before,
    )

    diag = {"proxy_a_before": proxy_a_before, "proxy_a_after": proxy_a_after}

    # Step 4: Importance weights
    source_weights = estimate_source_importance_weights(
        X_source_aligned, X_target_aligned,
        seed=seed, clip_range=(source_weight_clip_min, source_weight_clip_max),
    )

    # Step 5: Train aligned RF with importance weights (no blending)
    # RF probabilities are bounded [0,1] — blending two similar distributions
    # cannot flip predictions across the 0.5 threshold. Use aligned-only.
    model = build_rf(
        X_source_aligned, y_source_window,
        sample_weight=source_weights, seed=seed, use_gridsearch=use_gridsearch,
        max_depth=max_depth, min_samples_split=min_samples_split,
    )

    # Shift by -0.5 so aggregate_window_predictions threshold at >0 works
    # (0.5 probability -> 0, which is the natural decision boundary)
    proba = model.predict_proba(X_target_aligned)[:, 1]
    window_scores = (proba - 0.5).reshape(-1, 1)

    # Step 6: Aggregate window predictions
    trial_pred = aggregate_window_predictions(
        window_scores, target_window_trial_index, num_target_trials, task=task,
    )
    return trial_pred, diag


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

def make_output_dir_rf(
    output_root: Path,
    data_dir: str,
    task: str,
    pair_tag: str,
    method: str = "asfm",
    split_variant: str = "full-target-unlabeled",
) -> Path:
    output_dir = output_root / data_dir / task / f"rf-{method}" / split_variant / pair_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Cross-subject transfer learning with ASFM + RF")
    parser.add_argument("--data_root", type=str, default="task1", choices=["task1", "official"])
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "Task3-transfer_learning" / "output",
    )
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "threeclass"])
    parser.add_argument("--feature_k", type=int, default=22)
    parser.add_argument("--asfm_d", type=int, default=22)
    parser.add_argument("--asfm_source_weight_clip_min", type=float, default=0.5)
    parser.add_argument("--asfm_source_weight_clip_max", type=float, default=2.0)
    parser.add_argument("--asfm_window_seconds", type=float, default=4.0)
    parser.add_argument("--asfm_window_step_seconds", type=float, default=2.0)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--num_random_targets", type=int, default=0,
                        help="Number of random targets. 0 = use all subjects as targets (LOTO)")
    parser.add_argument("--source_select_max_samples", type=int, default=400)
    parser.add_argument("--source_select_coarse_topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_subject", type=str, default=None)
    parser.add_argument("--target_subject", type=str, default=None)
    parser.add_argument("--use_gridsearch", action="store_true", default=False)
    parser.add_argument("--rf_max_depth", type=int, default=10,
                        help="RF max_depth (default: 10)")
    parser.add_argument("--rf_min_samples_split", type=int, default=5,
                        help="RF min_samples_split (default: 5)")
    parser.add_argument("--di_alpha", type=float, default=0.5,
                        help="DI feature selection alpha: MI weight (default: 0.5)")
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
            if args.num_random_targets == 0:
                selected_targets = list(subjects)
                sampling_mode = "loto_all_targets"
                pair_tag = f"loto-{len(subjects)}-targets-best-source"
            else:
                if args.num_random_targets > len(subjects):
                    raise ValueError(
                        f"num_random_targets={args.num_random_targets} exceeds subject_count={len(subjects)}"
                    )
                target_rng = random.Random(args.seed)
                selected_targets = target_rng.sample(subjects, k=args.num_random_targets)
                sampling_mode = "random_targets_best_source"
                pair_tag = f"random-{args.num_random_targets}-targets-best-source"

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
                    tau=0.75,
                    seed=args.seed + target_idx * 100,
                    coarse_topk=args.source_select_coarse_topk,
                    max_samples=args.source_select_max_samples,
                    max_pseudo_per_class=30,
                    pseudo_imbalance_ratio=2.0,
                    pseudo_single_class_cap=8,
                    source_weight_clip_min=args.asfm_source_weight_clip_min,
                    source_weight_clip_max=args.asfm_source_weight_clip_max,
                )
            pair_list.append((best_source, target_subject))
            for row in selection_records:
                row["selected_source"] = best_source
                source_selection_rows.append(row)

    if not pair_list:
        raise ValueError("No subject pairs matched the provided filters.")

    output_dir = make_output_dir_rf(
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
    logging.info("model_type: rf")
    logging.info("method: asfm")
    logging.info("transfer_method: window-level ASFM + trial-level aggregation (RF)")
    logging.info("feature_selection: DI (MI + KS, alpha=0.5)")
    logging.info("score_blending: none (aligned-only RF)")
    logging.info("rf_regularization: max_depth=10, min_samples_split=5")
    logging.info("negative_transfer_detection: none (removed)")
    logging.info("asfm_d: %s", args.asfm_d)
    logging.info(
        "asfm_source_weight_clip: [%s, %s]",
        args.asfm_source_weight_clip_min, args.asfm_source_weight_clip_max,
    )
    logging.info("asfm_window_seconds: %s", args.asfm_window_seconds)
    logging.info("asfm_window_step_seconds: %s", args.asfm_window_step_seconds)
    logging.info("feature_k: %s", args.feature_k)
    logging.info("num_repeats: %s", args.num_repeats)
    logging.info("num_random_targets: %s", args.num_random_targets)
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
        logging.info(
            "========== pair %s/%s: %s -> %s ==========",
            pair_idx, len(pair_list), source_subject, target_subject,
        )
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

            source_pred = run_source_only_rf(
                X_source_window,
                y_source_window,
                X_target_window,
                target_window_trial_index,
                num_target_trials,
                feature_k=args.feature_k,
                use_gridsearch=args.use_gridsearch,
                seed=run_seed,
                task=args.task,
                max_depth=args.rf_max_depth,
                min_samples_split=args.rf_min_samples_split,
                di_alpha=args.di_alpha,
            )
            transfer_pred, diag = run_transfer_asfm_rf(
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
                source_weight_clip_min=args.asfm_source_weight_clip_min,
                source_weight_clip_max=args.asfm_source_weight_clip_max,
                max_depth=args.rf_max_depth,
                min_samples_split=args.rf_min_samples_split,
                di_alpha=args.di_alpha,
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
                    "proxy_a_before": diag.get("proxy_a_before", np.nan),
                    "proxy_a_after": diag.get("proxy_a_after", np.nan),
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
        np.array(all_source_true), np.array(all_source_pred),
        labels, source_cm_path, "Source-only Confusion Matrix (RF)",
    )
    save_confusion_matrix(
        np.array(all_transfer_true), np.array(all_transfer_pred),
        labels, transfer_cm_path, "Transfer Confusion Matrix (RF)",
    )

    logging.info("========== summary ==========")
    logging.info("pair_count: %s", len(pair_list))
    logging.info("source_only_mean_acc: %.4f", pair_summary_df["source_only_acc"].mean())
    logging.info("transfer_mean_acc: %.4f", pair_summary_df["transfer_acc"].mean())
    logging.info("transfer_gain_vs_source_only: %.4f", pair_summary_df["gain"].mean())
    logging.info("transfer_win_rate_vs_source_only: %.4f", (pair_summary_df["gain"] > 0).mean())
    logging.info("========== source-only report ==========")
    logging.info(
        "\n%s",
        classification_report(
            all_source_true, all_source_pred, target_names=labels, zero_division=0,
        ),
    )
    logging.info("========== transfer report ==========")
    logging.info(
        "\n%s",
        classification_report(
            all_transfer_true, all_transfer_pred, target_names=labels, zero_division=0,
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
