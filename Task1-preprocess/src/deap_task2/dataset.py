from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from deap_task2.constants import BINARY_THRESHOLD, EXPECTED_TRIALS_PER_SUBJECT
from deap_task2.io import build_task2_layout, resolve_default_metadata_source, to_project_relative
from deap_task2.labels import export_task2_labels


@dataclass(frozen=True)
class SubjectBundleResult:
    subject_id: str
    trial_count: int
    channel_count: int
    timepoint_count: int
    sfreq: float
    npz_path: str


def collect_subject_ids(processed_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in processed_root.iterdir()
        if path.is_dir() and path.name.startswith("s")
    )


def load_labels_table(
    project_root: Path,
    metadata_source: Path | None = None,
    labels_output_path: Path | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> tuple[pd.DataFrame, Path]:
    metadata_source = metadata_source or resolve_default_metadata_source(project_root)
    if labels_output_path is None:
        layout = build_task2_layout(project_root=project_root)
        labels_output_path = layout["labels"] / "deap_trial_labels.csv"
    labels = export_task2_labels(
        metadata_source=metadata_source,
        output_path=labels_output_path,
        threshold=threshold,
    )
    return labels, labels_output_path


def validate_subject_tables(
    subject_id: str,
    epochs: mne.Epochs,
    label_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    qc_table: pd.DataFrame,
) -> None:
    expected_count = len(epochs)
    counts = {
        "epochs": expected_count,
        "labels": len(label_table),
        "trial_table": len(trial_table),
        "qc_table": len(qc_table),
    }
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(f"{subject_id} has inconsistent trial counts: {counts}")

    if expected_count != EXPECTED_TRIALS_PER_SUBJECT:
        raise ValueError(
            f"{subject_id} expected {EXPECTED_TRIALS_PER_SUBJECT} trials, got {expected_count}"
        )

    expected_trial_index = np.arange(1, expected_count + 1)
    for name, frame in {
        "labels": label_table,
        "trial_table": trial_table,
        "qc_table": qc_table,
    }.items():
        if "trial_index" not in frame.columns:
            raise KeyError(f"{subject_id} {name} is missing trial_index")
        actual = frame["trial_index"].to_numpy(dtype=int)
        if not np.array_equal(actual, expected_trial_index):
            raise ValueError(f"{subject_id} {name} trial_index is not 1..{expected_count}")


def build_subject_manifest(
    subject_id: str,
    project_root: Path,
    epochs_path: Path,
    npz_path: Path,
    label_table: pd.DataFrame,
    trial_table: pd.DataFrame,
    qc_table: pd.DataFrame,
    sfreq: float,
    channel_count: int,
    timepoint_count: int,
) -> pd.DataFrame:
    manifest = (
        label_table.merge(trial_table, on="trial_index", how="inner", validate="1:1")
        .merge(qc_table, on="trial_index", how="inner", validate="1:1")
        .copy()
    )
    manifest["subject_id"] = subject_id
    manifest["epoch_index"] = np.arange(len(manifest), dtype=int)
    manifest["epochs_path"] = to_project_relative(epochs_path, project_root)
    manifest["npz_path"] = to_project_relative(npz_path, project_root)
    manifest["sfreq"] = float(sfreq)
    manifest["channel_count"] = int(channel_count)
    manifest["timepoint_count"] = int(timepoint_count)

    ordered = [
        "subject_id",
        "epoch_index",
        "trial_index",
        "experiment_id",
        "valence",
        "arousal",
        "dominance",
        "liking",
        "familiarity",
        "valence_binary",
        "arousal_binary",
        "dominance_binary",
        "quadrant_id",
        "quadrant_label",
        "muscle_overlap_seconds",
        "video_start_sample",
        "video_end_sample",
        "baseline_start_sample",
        "baseline_seconds",
        "video_duration_seconds",
        "start_time",
        "epochs_path",
        "npz_path",
        "sfreq",
        "channel_count",
        "timepoint_count",
    ]
    extras = [column for column in manifest.columns if column not in ordered]
    return manifest[ordered + extras]


def save_subject_bundle(
    epochs: mne.Epochs,
    manifest: pd.DataFrame,
    output_path: Path,
) -> None:
    data = epochs.get_data(copy=False).astype(np.float32, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=data,
        subject_id=np.asarray([manifest.loc[0, "subject_id"]], dtype="<U8"),
        epoch_index=manifest["epoch_index"].to_numpy(dtype=np.int16),
        trial_index=manifest["trial_index"].to_numpy(dtype=np.int16),
        experiment_id=manifest["experiment_id"].to_numpy(dtype=np.int16),
        valence=manifest["valence"].to_numpy(dtype=np.float32),
        arousal=manifest["arousal"].to_numpy(dtype=np.float32),
        dominance=manifest["dominance"].to_numpy(dtype=np.float32),
        liking=manifest["liking"].to_numpy(dtype=np.float32),
        familiarity=manifest["familiarity"].to_numpy(dtype=np.float32),
        valence_binary=manifest["valence_binary"].to_numpy(dtype=np.int8),
        arousal_binary=manifest["arousal_binary"].to_numpy(dtype=np.int8),
        dominance_binary=manifest["dominance_binary"].to_numpy(dtype=np.int8),
        quadrant_id=manifest["quadrant_id"].to_numpy(dtype=np.int8),
        quadrant_label=manifest["quadrant_label"].to_numpy(dtype="<U4"),
        muscle_overlap_seconds=manifest["muscle_overlap_seconds"].to_numpy(dtype=np.float32),
        channel_names=np.asarray(epochs.ch_names, dtype="<U16"),
        times=epochs.times.astype(np.float32),
        sfreq=np.asarray([epochs.info["sfreq"]], dtype=np.float32),
    )


def export_training_dataset(
    project_root: Path,
    metadata_source: Path | None = None,
    processed_root: Path | None = None,
    interim_root: Path | None = None,
    output_root: Path | None = None,
    subject_ids: list[str] | None = None,
    threshold: float = BINARY_THRESHOLD,
) -> tuple[list[SubjectBundleResult], pd.DataFrame, pd.DataFrame]:
    layout = build_task2_layout(project_root=project_root, output_root=output_root)
    processed_root = processed_root or (project_root / "data" / "processed")
    interim_root = interim_root or (project_root / "data" / "interim")

    labels, labels_path = load_labels_table(
        project_root=project_root,
        metadata_source=metadata_source,
        labels_output_path=layout["labels"] / "deap_trial_labels.csv",
        threshold=threshold,
    )

    subject_ids = subject_ids or collect_subject_ids(processed_root)
    manifest_frames: list[pd.DataFrame] = []
    results: list[SubjectBundleResult] = []

    for subject_id in subject_ids:
        epochs_path = processed_root / subject_id / "epochs_clean.fif"
        trial_table_path = interim_root / subject_id / "trial_events.csv"
        qc_table_path = processed_root / subject_id / "trial_qc.csv"
        if not epochs_path.exists():
            raise FileNotFoundError(f"Missing epochs file: {epochs_path}")
        if not trial_table_path.exists():
            raise FileNotFoundError(f"Missing trial table: {trial_table_path}")
        if not qc_table_path.exists():
            raise FileNotFoundError(f"Missing qc table: {qc_table_path}")

        epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
        subject_labels = (
            labels[labels["subject_id"] == subject_id]
            .sort_values("trial_index")
            .reset_index(drop=True)
        )
        trial_table = pd.read_csv(trial_table_path).sort_values("trial_index").reset_index(drop=True)
        qc_table = pd.read_csv(qc_table_path).sort_values("trial_index").reset_index(drop=True)
        validate_subject_tables(subject_id, epochs, subject_labels, trial_table, qc_table)

        npz_path = layout["npz"] / f"{subject_id}_task2_trials.npz"
        manifest = build_subject_manifest(
            subject_id=subject_id,
            project_root=project_root,
            epochs_path=epochs_path,
            npz_path=npz_path,
            label_table=subject_labels,
            trial_table=trial_table,
            qc_table=qc_table,
            sfreq=float(epochs.info["sfreq"]),
            channel_count=len(epochs.ch_names),
            timepoint_count=len(epochs.times),
        )
        save_subject_bundle(epochs=epochs, manifest=manifest, output_path=npz_path)
        manifest_frames.append(manifest)
        results.append(
            SubjectBundleResult(
                subject_id=subject_id,
                trial_count=len(epochs),
                channel_count=len(epochs.ch_names),
                timepoint_count=len(epochs.times),
                sfreq=float(epochs.info["sfreq"]),
                npz_path=to_project_relative(npz_path, project_root),
            )
        )

    manifest_table = pd.concat(manifest_frames, ignore_index=True)
    manifest_path = layout["manifests"] / "task2_training_index.csv"
    manifest_table.to_csv(manifest_path, index=False, encoding="utf-8")

    summary_table = pd.DataFrame([asdict(result) for result in results])
    summary_path = layout["manifests"] / "task2_subject_summary.csv"
    summary_table.to_csv(summary_path, index=False, encoding="utf-8")

    stats = {
        "labels_path": to_project_relative(labels_path, project_root),
        "manifest_path": to_project_relative(manifest_path, project_root),
        "summary_path": to_project_relative(summary_path, project_root),
        "subject_count": len(results),
        "trial_count": int(summary_table["trial_count"].sum()) if not summary_table.empty else 0,
        "binary_threshold": threshold,
    }
    with (layout["manifests"] / "task2_dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    return results, manifest_table, summary_table
